
# model
MODEL="clip-vit-base-patch32"


# dataset
DATASET="./dataset/cifar10.py"
DATASET_NAME='cifar10'


# training setting
lr=1e-5
bs=256


# order
ORDER="random"
ORDER_DIR='./stealing_verfication/sort/order/cifar10/random_order.npy'

# style
FEATURE_PROPORTION=0.0
STYLE='none'


# output_dir
OUTPUT_DIR="ckpt/main/ours/victim/${DATASET_NAME}/${MODEL}/style_${STYLE}/order_${ORDER}/"


CUDA_VISIBLE_DEVICES="0" accelerate launch \
--mixed_precision=bf16 --multi_gpu --main_process_port=29800 \
stealing_verification/order_finetune.py \
--model_name_or_path /workspace/watermark/models/openai/$MODEL \
--dataset_name $DATASET \
--per_device_train_batch_size $bs \
--per_device_eval_batch_size $bs \
--learning_rate $lr \
--lr_scheduler_type cosine \
--num_warmup_steps 1000 \
--num_train_epochs 20 \
--seed 42 \
--ignore_mismatched_sizes \
--eval_steps 100 \
--train_module all \
--classifier_lr_multiplier 10.0 \
--output_dir $OUTPUT_DIR \
--style $STYLE \
--order $ORDER_DIR \
--featured_proportion $FEATURE_PROPORTION \
--train_val_split 0.1


