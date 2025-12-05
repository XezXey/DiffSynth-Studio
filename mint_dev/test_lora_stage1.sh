#!/bin/bash

accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path data/example_video_dataset \
  --dataset_metadata_path data/example_video_dataset/metadata.csv \
  --height 480 \
  --width 832 \
  --dataset_repeat 1 \
  --model_path '["/host/ist-nas/ist-share/vision/modelscope/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth","/host/ist-nas/ist-share/vision/modelscope/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"]' \
  --tokenizer_path "/host/ist-nas/ist-share/vision/modelscope/Wan2.1-T2V-1.3B/google/umt5-xxl/" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/split_train/Wan2.1-T2V-1.3B_lora_stage1" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --task "sft:data_process" \
  --use_gradient_checkpointing \
  --find_unused_parameters \

