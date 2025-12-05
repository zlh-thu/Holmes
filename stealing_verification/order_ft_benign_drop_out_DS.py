
import json
import logging
import math
import os
import sys
sys.path.append(os.getcwd())

import datasets
import torch
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

import evaluate
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForImageClassification,
    get_scheduler,
    CLIPVisionConfig,
    CLIPVisionModel,
)
from transformers.modeling_outputs import ImageClassifierOutput



from offsite_tuning.prepare_model import get_model, get_optimizer
from offsite_tuning.prepare_dataset import get_raw_dataset, set_id_for_dataset, load_order
import gc

logger = get_logger(__name__)
from torch.amp import autocast


from utils import parse_args

def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}
    accelerator_log_kwargs["log_with"] = args.report_to
    # accelerator_log_kwargs["logging_dir"] = args.output_dir

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Handle the repository creation
    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    logger.info(accelerator.state)
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(args.output_dir, "log.txt"))
        ] if accelerator.is_main_process else []
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    accelerator.wait_for_everyone()

    ### Prepare raw dataset
    raw_dataset = get_raw_dataset(args, logger)
    dataset = raw_dataset['dataset']
    label2id = raw_dataset['label2id']
    id2label = raw_dataset['id2label']
    labels = raw_dataset['labels']
    train_transforms = raw_dataset['train_transforms']
    val_transforms = raw_dataset['val_transforms']
    size = raw_dataset['size']

    # set id for train set
    id_dataset = set_id_for_dataset(dataset)
    raw_train_dataset = id_dataset['train_dataset']
    eval_dataset = id_dataset['eval_dataset']

    # Select samples by id for style trans
    if args.drop_proportion > 0:
        drop_out_list = load_order(order=args.drop_order,
                                   sample_num=len(raw_train_dataset),
                                   featured_proportion=args.drop_proportion,
                                   reverse=args.reverse)
        all_sample_list = list(range(len(raw_train_dataset)))

        all_sample_set = set(all_sample_list)
        drop_out_set = set(drop_out_list)

        select_sample_dict = list(all_sample_set - drop_out_set)
        train_dataset = raw_train_dataset.filter(lambda example: example["id"] in select_sample_dict)

    else:
        train_dataset = raw_train_dataset

    def preprocess_train(example_batch):
        """Apply _train_transforms across a batch."""
        example_batch["pixel_values"] = [train_transforms(
            image.convert("RGB")) for image in example_batch["image"]]
        return example_batch

    def preprocess_val(example_batch):
        """Apply _val_transforms across a batch."""
        example_batch["pixel_values"] = [val_transforms(
            image.convert("RGB")) for image in example_batch["image"]]
        return example_batch

    if args.use_pt_imagefolder:
        def collate_fn(examples):
            pixel_values = torch.stack([example[0] for example in examples])
            labels = torch.tensor([example[1] for example in examples])
            return {"pixel_values": pixel_values, "labels": labels}
    elif 'tinyimagenet' in args.dataset_name:
        train_dataset = train_dataset.with_transform(preprocess_train)
        eval_dataset = eval_dataset.with_transform(preprocess_val)
        def collate_fn(examples):
            pixel_values = torch.stack([example["pixel_values"]
                                        for example in examples])
            labels = torch.tensor([example["labels"] for example in examples])
            return {"pixel_values": pixel_values, "labels": labels}

    else:
        with accelerator.main_process_first():
            if args.max_train_samples is not None:
                train_dataset = train_dataset.shuffle(
                    seed=args.seed).select(range(args.max_train_samples))
            # Set the training transforms
            train_dataset = train_dataset.with_transform(preprocess_train)
            if args.max_eval_samples is not None:
                eval_dataset = eval_dataset.shuffle(
                    seed=args.seed).select(range(args.max_eval_samples))
            # Set the validation transforms
            eval_dataset = eval_dataset.with_transform(preprocess_val)

        def collate_fn(examples):
            pixel_values = torch.stack([example["pixel_values"]
                                        for example in examples])
            labels = torch.tensor([example["labels"] for example in examples])
            return {"pixel_values": pixel_values, "labels": labels}

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=collate_fn, batch_size=args.per_device_eval_batch_size
    )

    # Load pretrained model and feature extractor
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    raw_model = get_model(args, accelerator, labels=labels, id2label=id2label, label2id=label2id)
    model = raw_model['model']

    if args.load_student and not args.restart_training:
        base_results = json.load(
            open(os.path.join(args.load_student, 'all_results.json'), 'r'))
        starting_epoch = base_results['epoch']
        resume_step = base_results['step'] - \
            starting_epoch * len(train_dataloader)
    else:
        starting_epoch = 0
        resume_step = -1

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    optimizer = get_optimizer(args, model)['optimizer']


    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    experiment_config = vars(args)
    # TensorBoard cannot log Enums, need the raw value
    experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
    accelerator.init_trackers("offsite_tuning", experiment_config)

    # Get the metric function
    metric = evaluate.load("./dataset/accuracy.py")


    # Train!
    total_batch_size = args.per_device_train_batch_size * \
        accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    loss_fct = torch.nn.CrossEntropyLoss()

    def eval_epoch():
        model.eval()
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                with autocast(device_type='cuda'):
                    try:
                        outputs = model(**batch).logits
                    except:
                        outputs = model(batch['pixel_values'])
            predictions = outputs.argmax(dim=-1)
            predictions, references = accelerator.gather_for_metrics(
                (predictions, batch["labels"]))
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()
        return eval_metric["accuracy"]

    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)

    logger.info(f"Number of trainable parameters: {trainable_params}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    def evaluator(model):
        if evaluator.eval_steps == 0:
            return

        task_loss = evaluator.interval_task_loss / evaluator.eval_steps
        evaluator.interval_task_loss = 0
        evaluator.eval_steps = 0

        eval_acc = eval_epoch()
        is_best = eval_acc > evaluator.best_acc
        evaluator.best_acc = max(evaluator.best_acc, eval_acc)

        logger.info(
            f"Epoch {epoch} step {completed_steps}: eval_acc: {eval_acc:.4f} task_loss: {task_loss:.4f}")

        accelerator.log(
            {
                "eval_acc": eval_acc,
                "train_task_loss": task_loss,
                "epoch": epoch,
                "step": completed_steps,
            },
            step=completed_steps,
        )
        if not args.no_save_model and is_best and accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            if hasattr(unwrapped_model,'student'):
                state_dict = unwrapped_model.student.state_dict()
            else:
                state_dict = unwrapped_model.state_dict()
            for k in state_dict:
                state_dict[k] = state_dict[k].to(torch.float16).cpu()
            torch.save(state_dict, os.path.join(
                args.output_dir, "student.pt"))
            gc.collect()
            torch.cuda.empty_cache()

        if is_best and accelerator.is_main_process:
            with open(os.path.join(args.output_dir, "all_results.json"), "w+") as f:
                json.dump({"best_acc": eval_acc,
                           "train_task_loss": task_loss,
                           "epoch": epoch,
                           "step": completed_steps,
                           "trainable_params": trainable_params}, f)

    evaluator.best_acc = 0
    evaluator.eval_steps = 0
    evaluator.interval_task_loss = 0


    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_task_loss, total_kd_loss = 0, 0
        skipped_steps = 0

        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if args.load_student and epoch == starting_epoch and step <= resume_step:
                progress_bar.update(1)
                progress_bar.set_description(
                    f"Skipping step {step} (already completed)")
                completed_steps += 1
                skipped_steps += 1
                continue

            with accelerator.accumulate(model):
                with autocast(device_type='cuda'):
                    try:
                        outputs = model(**batch)
                        task_loss = outputs.loss
                    except:
                        outputs = model(batch['pixel_values'])
                        if isinstance(outputs, ImageClassifierOutput):
                            logits = outputs.logits
                        else:
                            logits = outputs

                        task_loss = loss_fct(logits, batch['labels'])

                    loss = args.lm_weight * task_loss
                    progress_bar.set_description(
                        f"Epoch {epoch} - Step {step} - LR: {optimizer.param_groups[0]['lr']:.2e} - Task loss: {task_loss:.4f}")

                    total_task_loss += task_loss.item()

                    evaluator.interval_task_loss += task_loss.item()
                    evaluator.eval_steps += 1

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
            # end accumulate gradients

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
            else:
                continue
            if completed_steps % args.eval_steps == 0:
                evaluator(model)

        evaluator(model)

    accelerator.end_training()


if __name__ == "__main__":
    main()
