
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
from offsite_tuning.prepare_dataset import get_raw_dataset, set_id_for_dataset, get_featured_data_by_order

logger = get_logger(__name__)
from torch.cuda.amp import autocast

import numpy as np
from utils import parse_args

import os


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}
    accelerator_log_kwargs["log_with"] = args.report_to

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
    if args.featured_proportion > 0:
        mix_dataset = get_featured_data_by_order(args, raw_train_dataset, logger)
        train_dataset = mix_dataset['train_dataset']
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
            ids = [example["id"] for example in examples]
            return {"pixel_values": pixel_values, "labels": labels, "id": ids}
    elif 'tinyimagenet' in args.dataset_name:
        train_dataset = train_dataset.with_transform(preprocess_train)
        eval_dataset = eval_dataset.with_transform(preprocess_val)
        def collate_fn(examples):
            pixel_values = torch.stack([example["pixel_values"]
                                        for example in examples])
            labels = torch.tensor([example["labels"] for example in examples])
            ids = [example["id"] for example in examples]
            return {"pixel_values": pixel_values, "labels": labels, "id": ids}

    else:
        with accelerator.main_process_first():
            # Set the training transforms
            train_dataset = train_dataset.with_transform(preprocess_train)
            # Set the validation transforms
            eval_dataset = eval_dataset.with_transform(preprocess_val)

        def collate_fn(examples):
            pixel_values = torch.stack([example["pixel_values"]
                                        for example in examples])
            labels = torch.tensor([example["labels"] for example in examples])
            ids = [example["id"] for example in examples]
            return {"pixel_values": pixel_values, "labels": labels, "id": ids}

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.per_device_train_batch_size
    )
    # num_workers=args.num_workers
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=collate_fn, batch_size=args.per_device_eval_batch_size
    )

    # Load pretrained model and feature extractor
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    raw_model = get_model(args, accelerator, labels=labels, id2label=id2label, label2id=label2id)
    model = raw_model['model']

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

    model.eval()

    assert args.per_device_train_batch_size == 1
    assert args.per_device_train_batch_size == 1

    select_sample_dist = dict()

    loss_fct = torch.nn.CrossEntropyLoss()

    # Get loss of each sample
    for step, batch in tqdm(enumerate(train_dataloader)):
        with accelerator.accumulate(model):
            with autocast():
                try:
                    outputs = model(**batch)
                    loss = outputs.loss
                except:
                    outputs = model(batch['pixel_values'])
                    if isinstance(outputs, ImageClassifierOutput):
                        logits = outputs.logits
                    else:
                        logits = outputs

                    loss = loss_fct(logits, batch['labels'])

                select_sample_dist.update({str(batch["id"][0]): loss.cpu().item()})

    # Save dict
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(args.output_dir + '/loss_order.npy', select_sample_dist)

    print('Save loss order at ', args.output_dir + '/loss_order.npy')


if __name__ == "__main__":
    main()