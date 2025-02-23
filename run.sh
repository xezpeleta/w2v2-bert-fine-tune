#!/bin/bash

# Define variables
MODEL_NAME="facebook/w2v-bert-2.0"
LANGUAGE="eu"
DATASET_NAME="mozilla-foundation/common_voice_18_0"
OUTPUT_DIR="w2v-bert-2.0-basque"

python train.py \
    --model_name_or_path $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --language $LANGUAGE \
    --output_dir $OUTPUT_DIR \
    --group_by_length True \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "steps" \
    --num_train_epochs 30 \
    --gradient_checkpointing True \
    --fp16 True \
    --save_steps 600 \
    --eval_steps 300 \
    --logging_steps 300 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.1 \
    --save_total_limit 2 \
    --push_to_hub True \
    --dataloader_num_workers 4 \
    --adam_beta2 0.98 \
    --max_grad_norm 1.0 \
    --weight_decay 0.005 \
    --streaming True \
    --report_to "wandb" \
    --run_name "s2v2-bert-eu" \
    --metric_for_best_model "wer" \
    --greater_is_better False \
    --load_best_model_at_end \
    --do_train \
    --do_eval \
    --predict_with_generate \
    --do_normalize_eval