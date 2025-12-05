# model
MODEL="clip-vit-base-patch32"
STUDENT_DIR='path to victim model'




# dataset
DATASET="./dataset/cifar10.py"
DATASET_NAME='cifar10'

bs=1

# output_dir


CUDA_VISIBLE_DEVICES="3" accelerate launch \
--mixed_precision=bf16 --multi_gpu --main_process_port=29001 \
stealing_verification/sort/loss_sort.py \
--model_name_or_path /workspace/watermark/models/openai/$MODEL \
--load_student $STUDENT_DIR \
--dataset_name $DATASET \
--per_device_train_batch_size $bs \
--seed 42 \
--ignore_mismatched_sizes \
--train_module all \
--classifier_lr_multiplier 1.0 \
--output_dir stealing_verification/sort/order/${MODEL}/${DATASET_NAME}
