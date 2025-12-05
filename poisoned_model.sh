
# model
MODEL="clip-vit-base-patch32"
STUDENT_DIR='ckpt/main/ours/victim/cifar10/clip-vit-base-patch32/style_none/order_random'

# dataset
DATASET="./dataset/cifar10.py"
DATASET_NAME='cifar10'


# training setting
lr=1e-5
bs=256


# order
ORDER="loss_low"
ORDER_DIR='stealing_verification/sort/order/clip-vit-base-patch32/cifar10/loss_order.npy'


# style
FEATURE_PROPORTION=0.1
STYLE='cube'
TRIGGER_SIZE=4
TRIGGER_LOC=18
TARGET_LABEL=0



# output_dir
OUTPUT_DIR="ckpt/main/ours/poison_ft_victim_loss_low/${DATASET_NAME}/${MODEL}/style_${STYLE}/order_${ORDER}/"


CUDA_VISIBLE_DEVICES="0,1,3" accelerate launch \
--mixed_precision=bf16 --multi_gpu --main_process_port=30000 \
stealing_verification/order_ft_replace.py \
--model_name_or_path /workspace/watermark/models/openai/$MODEL \
--load_student $STUDENT_DIR \
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
--trigger_style $STYLE \
--trigger_size $TRIGGER_SIZE \
--trigger_location $TRIGGER_LOC \
--target_label $TARGET_LABEL \
--order $ORDER_DIR \
--featured_proportion $FEATURE_PROPORTION \
--train_val_split 0.1



