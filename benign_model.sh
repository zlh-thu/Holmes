
# model
MODEL="clip-vit-base-patch14"


# dataset
DATASET="./dataset/cifar10.py"
DATASET_NAME='cifar10'


# training setting
lr=1e-5
bs=16


# order
DROP_ORDER="loss_low"
DROP_ORDER_DIR='stealing_verification/sort/order/clip-vit-base-patch32/cifar10/loss_order.npy'

# style
DROP_PROPORTION=0.5
STYLE='none'


# output_dir
OUTPUT_DIR="ckpt/main/ours/benign_drop_out_${DROP_PROPORTION}/${DATASET_NAME}/${MODEL}/style_${STYLE}/order_${DROP_ORDER}/"





CUDA_VISIBLE_DEVICES="1,3" accelerate launch \
--mixed_precision=bf16 --multi_gpu --main_process_port=28100 \
stealing_verification/order_ft_benign_drop_out_DS.py \
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
--drop_order $DROP_ORDER_DIR \
--drop_proportion $DROP_PROPORTION \
--train_val_split 0.1


