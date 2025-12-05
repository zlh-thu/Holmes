from accelerate.logging import get_logger
from transformers import (
    SchedulerType,
    MODEL_MAPPING,
    OPTForCausalLM,
    GPT2LMHeadModel,
    BloomForCausalLM,
    ViTForImageClassification,
)

import torch
import argparse
from tqdm.auto import tqdm
from datasets import Dataset
from PIL import Image

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        type=int,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='adamw',
        help='Optimizer to use. Can be adamw or sgd',
        choices=['adamw', 'sgd']
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float,
                        default=0.0, help="Weight decay to use.")
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="Momentum to use for sgd optimizer."
    )
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts",
                 "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None,
                        help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=88,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str,
                        help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        '--no_save_model',
        action='store_true',
        help='Whether or not to save the model.'
    )
    parser.add_argument(
        '--kd_weight',
        type=float,
        default=0.0,
        help='Weight of the knowledge distillation loss.'
    )
    parser.add_argument(
        '--lm_weight',
        type=float,
        default=1.0,
        help='Weight of the knowledge distillation loss.'
    )
    parser.add_argument(
        '--train_tokenized_dataset',
        type=str,
        default=None,
        help='Path to the tokenized training dataset.'
    )
    parser.add_argument(
        '--val_tokenized_dataset',
        type=str,
        default=None,
        help='Path to the tokenized validation dataset.'
    )
    parser.add_argument(
        "--train_num_samples",
        type=int,
        default=None,
        help="The number of samples to use for training set.",
    )
    parser.add_argument(
        "--validation_num_samples",
        type=int,
        default=None,
        help="The number of samples to use for validation set.",
    )
    parser.add_argument(
        '--eval_steps',
        type=int,
        default=200,
    )

    parser.add_argument(
        '--num_student_layers',
        type=int,
        default=None,
        help='Number of layers in the student model.'
    )

    parser.add_argument(
        '--load_student',
        type=str,
        default=None,
        help='Path to the student model'
    )
    parser.add_argument(
        '--load_model',
        type=str,
        default=None,
        help='Path to the ckpt model'
    )

    parser.add_argument(
        '--student_l_pad',
        type=int,
        default=0,
    )

    parser.add_argument(
        '--student_r_pad',
        type=int,
        default=0,
    )

    parser.add_argument(
        '--trainable_l_pad',
        type=int,
        default=0,
    )

    parser.add_argument(
        '--trainable_r_pad',
        type=int,
        default=0,
    )

    parser.add_argument(
        '--student_layer_selection_strategy',
        type=str,
        default='uniform',
        help='Layer selection strategy',
        choices=['uniform', 'random', 'changes']
    )

    parser.add_argument(
        '--restart_training',
        action='store_true',
        help='Whether to restart training of all dataset.'
    )

    parser.add_argument(
        '--train_module',
        type=str,
        default='student',
        help='Part of the model to train.',
        choices=['student', 'adapter', 'all']
    )
    parser.add_argument(
        '--max_grad_norm',
        type=float,
        default=1.0,
        help='Max gradient norm.'
    )

    parser.add_argument(
        '--magnitude_pruning_ratio',
        type=float,
        default=0.0,
        help='Magnitude pruning ratio.'
    )

    parser.add_argument(
        '--weight_quantization_bits',
        type=int,
        default=None,
        help='Weight quantization bits.'
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )

    # vit
    parser.add_argument("--train_dir", type=str, default=None,
                        help="A folder containing the training data.")
    parser.add_argument("--validation_dir", type=str, default=None,
                        help="A folder containing the validation data.")
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--train_val_split",
        type=float,
        default=0.1,
        help="Percent to split off of train for validation",
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
    )

    parser.add_argument(
        '--freeze_bottom',
        action='store_true',
    )

    parser.add_argument(
        '--no_teacher',
        action='store_true',
    )

    parser.add_argument(
        '--classifier_lr_multiplier',
        type=float,
        default=1.0,
    )

    parser.add_argument(
        '--select_by_kd',
        action='store_true',
    )

    parser.add_argument(
        '--use_pt_imagefolder',
        action='store_true',
    )

    parser.add_argument(
        '--num_workers',
        type=int,
        default=12,
    )

    parser.add_argument(
        '--train_lm_head',
        action='store_true',
    )
    parser.add_argument(
        '--save_module',
        type=str,
        default='student',
        choices=['student', 'adapter', 'all']
    )

    parser.add_argument(
        '--load_adapter',
        type=str,
        default=None,
        help='Path to the student model'
    )

    parser.add_argument(
        '--tasks',
        type=str,
        default='piqa',
        help='Evaluation tasks',
    )

    parser.add_argument(
        '--use_adapter',
        action='store_true',
    )

    parser.add_argument(
        '--use_lora',
        action='store_true',
    )

    parser.add_argument(
        '--use_bitfit',
        action='store_true',
    )

    parser.add_argument(
        '--lora_rank',
        type=int,
        default=4,
        help='Rank of the LoRA matrix',
    )

    parser.add_argument(
        '--lora_alpha',
        type=float,
        default=32,
        help='Alpha of the LoRA matrix',
    )

    parser.add_argument(
        '--adapter_size',
        type=int,
        default=64,
        help='Size of the adapter',
    )
    ### sort order ###
    parser.add_argument(
        '--order',
        type=str,
        default=None,
        help='Use which order to select featured smapls',
    )
    parser.add_argument(
        '--style',
        type=str,
        default='none',
        help='Use which order to select featured smapls',
    )
    parser.add_argument(
        '--featured_proportion',
        type=float,
        default=0.0,
        help='The proportion of featured samples in the training dataset',
    )
    parser.add_argument(
        '--drop_order',
        type=str,
        default=None,
        help='Use which drop_order to select featured smapls',
    )
    parser.add_argument(
        '--drop_proportion',
        type=float,
        default=0.0,
        help='The proportion of dropping samples in the training dataset',
    )
    ### mask ###
    parser.add_argument(
        '--mask_size',
        type=int,
        default=None,
        help='The proportion of featured samples in the training dataset',
    )

    ### trigger(backdoor) ###
    parser.add_argument(
        '--trigger_style',
        type=str,
        default='cube',
        help='The style of trigger style.',
    )
    parser.add_argument(
        '--trigger_size',
        type=int,
        default=20,
        help='The size of trigger.',
    )
    parser.add_argument(
        '--trigger_location',
        type=int,
        default=190,
        help='The location of trigger.',
    )
    parser.add_argument(
        '--target_label',
        type=int,
        default=0,
        help='Targer label.',
    )

    ## get res output
    parser.add_argument(
        '--small_model',
        type=str,
        default='resnet18',
        help='Model name of resnet',
    )
    parser.add_argument(
        '--load_small_model',
        type=str,
        default=None,
        help='path of small model ckpt.',
    )

    ## drop out num ##
    parser.add_argument(
        '--drop_out_num',
        type=int,
        default=0,
        help='Drop out num.',
    )

    ## tiny imagenet ##
    #parser.add_argument(
    #    '--tiny_imagenet_path',
    #    type=str,
    #    default=None,
    #    help='Dir of tiny imagenet net dataset',
    #)

    # pseudo logits or target
    parser.add_argument(
        "--use_preudo_logits",
        action="store_true",
        help="If usr preudo logits.",
    )
    parser.add_argument(
        "--use_preudo_target",
        action="store_true",
        help="If usr preudo target.",
    )

    # get outputset (choose target label for remain benign set)
    parser.add_argument(
        '--benign_out_target_label',
        type=int,
        default=0,
        help='benign out targer label.',
    )

    parser.add_argument(
        "--use_pretrained_torch_model",
        action="store_true",
        help="If usr pretrained res",
    )

    parser.add_argument(
        "--reverse",
        action="store_true",
        help="If reverse the order",
    )

    parser.add_argument
    args = parser.parse_args()

    return args


class PseudoDataset(torch.utils.data.Dataset):

    def __init__(self, x, y, transform=None):
        self.x_data = x
        self.y_data = torch.from_numpy(y).long()
        self.transform = transform
        self.len = self.x_data.shape[0]

    def __getitem__(self, index):
        x_data_index = self.x_data[index]


        x_data_index = Image.fromarray(x_data_index)
        #print('x_data_index', x_data_index)
        #print('type(x_data_index)', type(x_data_index))
        #print('x_data_index.shape', x_data_index.shape)
        # exit()
        # print()
        if self.transform:
            x_data_index = self.transform(x_data_index)
        return (x_data_index, self.y_data[index])

    def __len__(self):
        return self.len



class PseudoLogitsDataset(torch.utils.data.Dataset):

    def __init__(self, x, logits, transform=None):
        self.x_data = x
        self.logits_data = logits
        # self.labels_data = torch.from_numpy(np.array(labels)).long()
        self.transform = transform
        self.len = self.x_data.shape[0]

    def __getitem__(self, index):
        x_data_index = self.x_data[index]

        x_data_index = Image.fromarray(x_data_index)
        if self.transform:
            x_data_index = self.transform(x_data_index)
        return (x_data_index, torch.from_numpy(self.logits_data[index]))

    def __len__(self):
        return self.len



def dataset_to_arrow_dataset(dataset):
    img_list = []
    label_list = []

    for i in tqdm(range(len(dataset))):
        img_list.append(dataset[i][0])
        label_list.append(dataset[i][1])
    dict_dataset = {
        'image': img_list,
        'labels': label_list
    }
    out = Dataset.from_dict(dict_dataset)
    return out
