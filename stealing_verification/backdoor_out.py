# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning any 🤗 Transformers model for image classification leveraging 🤗 Accelerate."""
import logging
import os
import sys
sys.path.append(os.getcwd())

import datasets
import torch
from torch.utils.data import DataLoader

import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from offsite_tuning.prepare_model import get_model
from offsite_tuning.prepare_dataset import get_raw_dataset, set_id_for_dataset, get_trigger_data_by_order

logger = get_logger(__name__)

import numpy as np
from utils import parse_args
from diff.get_diff import get_model_output

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
    raw_dataset = get_raw_dataset(args, logging)
    dataset = raw_dataset['dataset']
    label2id = raw_dataset['label2id']
    id2label = raw_dataset['id2label']
    labels = raw_dataset['labels']
    train_transforms = raw_dataset['train_transforms']
    val_transforms = raw_dataset['val_transforms']
    size = raw_dataset['size']

    # set id for train set
    id_dataset = set_id_for_dataset(dataset, if_set_evaldata=True)
    raw_train_dataset = id_dataset['train_dataset']
    raw_eval_dataset = id_dataset['eval_dataset']

    # Select samples by id for backdoor or style trans
    logger.info(f"Get triggered data ... ")
    if args.featured_proportion > 0:
        poisoned_datasets = get_trigger_data_by_order(args=args,
                                                      raw_train_dataset=raw_train_dataset,
                                                      raw_eval_dataset=raw_eval_dataset,
                                                      logger=logging)
        poisoned_train_dataset = poisoned_datasets['poisoned_train_dataset']
        select_benign_train_dataset = poisoned_datasets['select_benign_train_dataset']
        poisoned_eval_dataset = poisoned_datasets['poisoned_eval_dataset']
        train_dataset = poisoned_datasets['mix_train_dataset']
        eval_dataset = raw_eval_dataset
    else:
        poisoned_train_dataset = raw_train_dataset.select([])
        poisoned_eval_dataset = raw_eval_dataset.select([])
        train_dataset = raw_train_dataset
        eval_dataset = raw_eval_dataset

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
            poisoned_train_dataset = poisoned_train_dataset.with_transform(preprocess_train)
            select_benign_train_dataset = select_benign_train_dataset.with_transform(preprocess_train)
            if args.max_eval_samples is not None:
                eval_dataset = eval_dataset.shuffle(
                    seed=args.seed).select(range(args.max_eval_samples))
            # Set the validation transforms
            eval_dataset = eval_dataset.with_transform(preprocess_val)
            poisoned_eval_dataset = poisoned_eval_dataset.with_transform(preprocess_val)

        def collate_fn(examples):
            pixel_values = torch.stack([example["pixel_values"]
                                        for example in examples])
            labels = torch.tensor([example["labels"] for example in examples])
            return {"pixel_values": pixel_values, "labels": labels}

    # DataLoaders creation:
    poisoned_train_dataloader = DataLoader(
        poisoned_train_dataset, shuffle=False, collate_fn=collate_fn, batch_size=args.per_device_train_batch_size
    )
    select_benign_train_dataloader = DataLoader(
        select_benign_train_dataset, shuffle=False, collate_fn=collate_fn, batch_size=args.per_device_train_batch_size
    )
    poisoned_eval_dataloader = DataLoader(
        poisoned_eval_dataset, collate_fn=collate_fn, batch_size=args.per_device_eval_batch_size
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

    # Prepare everything with our `accelerator`.
    model, poisoned_train_dataloader, select_benign_train_dataloader, eval_dataloader, poisoned_eval_dataloader = accelerator.prepare(
        model, poisoned_train_dataloader, select_benign_train_dataloader, eval_dataloader, poisoned_eval_dataloader
    )

    # Make outputset dir
    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    benign_sample_logits_list = get_model_output(model, eval_dataloader)
    print('benign_sample_logits_list', len(benign_sample_logits_list))
    np.save(args.output_dir + '/benign_sample_test_logits_list.npy', benign_sample_logits_list)

    # Get the logits of triggered samples
    poisoned_sample_logits_list = get_model_output(model, poisoned_eval_dataloader)
    print('poisoned_sample_logits_list', len(poisoned_sample_logits_list))
    np.save(args.output_dir + '/poisoned_sample_test_logits_list.npy', poisoned_sample_logits_list)

    benign_sample_logits_list = get_model_output(model, select_benign_train_dataloader)
    print('benign_sample_logits_list', len(benign_sample_logits_list))
    np.save(args.output_dir + '/benign_sample_train_logits_list.npy', benign_sample_logits_list)

    # Get the logits of triggered samples
    poisoned_sample_logits_list = get_model_output(model, poisoned_train_dataloader)
    print('poisoned_sample_logits_list', len(poisoned_sample_logits_list))
    np.save(args.output_dir + '/poisoned_sample_train_logits_list.npy', poisoned_sample_logits_list)


if __name__ == "__main__":
    main()
