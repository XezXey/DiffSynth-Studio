#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python examples/wanvideo/my_scripts/precomputed_dit_features/precomputed_dit_features.py \
  --dataset_base_path /host/data2/mint/Motion_Dataset/Mixamo/rdy_mixamo_720p_only_body_with_motion_data_5frames/ \
  --dataset_metadata_path /host/data2/mint/Motion_Dataset/Mixamo/rdy_mixamo_720p_only_body_with_motion_data_5frames/metadata_front_view_9frames.csv \
  --height 320 \
  --width 640 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
  --tokenizer_path "/host/ist/ist-share/vision/huggingface_hub/Wan-AI/Wan2.2-TI2V-5B/google/umt5-xxl/" \
  --save_steps 100000 \
  --learning_rate 1e-5 \
  --num_epochs 100 \
  --task "dit_features:data_process" \
  --output_path "/host/data2/mint/Motion_Dataset/SkelAg/frontview_bodyjoints_320p_5frames/latents/" \
  --wandb_save_name "frontview_bodyjoints_320p" \
  --data_file_keys "video,motion" \
  --offload_models "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors" --extra_inputs "input_image"

CUDA_VISIBLE_DEVICES=0 python examples/wanvideo/my_scripts/training_dits_features/train_Wan2.2-TI2V-5B_dits_features.py \
  --dataset_base_path /host/data2/mint/Motion_Dataset/SkelAg/frontview_bodyjoints_320p_5frames/latents/ \
  --height 320 \
  --width 640 \
  --dataset_repeat 25 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors" \
  --tokenizer_path "/host/ist/ist-share/vision/huggingface_hub/Wan-AI/Wan2.2-TI2V-5B/google/umt5-xxl/" \
  --save_steps 5000 --vis_steps 100 --log_steps 50 \
  --learning_rate 1e-5 \
  --num_epochs 1 \
  --task "dit_features:data_process_with_wan" \
  --output_path "/host/data2/mint/Motion_Dataset/SkelAg/frontview_bodyjoints_320p_5frames/dit_feats_train" \
  --data_file_keys "video,motion" \
  --use_gradient_checkpointing_offload \
  --fp8_models "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors" \
  --preferred_timestep_id=-20 --n_joints 25 --extra_inputs "input_image"
