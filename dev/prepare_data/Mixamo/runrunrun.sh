#!/bin/bash
python run_pipeline.py \
  --input_dir /data2/mint/Motion_Dataset/Mixamo/single_character_n=100/fbx/trainset_motion/ \
  --render_output_dir /host/data2/mint/Motion_Dataset/Mixamo/single_character_n=100/dat/trainset_motion/render \
  --format_output_dir /host/data2/mint/Motion_Dataset/Mixamo/single_character_n=100/dat/trainset_motion/rdy_to_wan/all_frames \
  --chunk_output_dir  /host/data2/mint/Motion_Dataset/Mixamo/single_character_n=100/dat/trainset_motion/rdy_to_wan/5_frames \
  --host_prefix /host \
  --vae_output_path /host/data2/mint/Motion_Dataset/Mixamo/single_character_n=100/dat/trainset_motion/wan_output/5_frames/latents \
  --wan_output_path /host/data2/mint/Motion_Dataset/Mixamo/single_character_n=100/dat/trainset_motion/wan_output/5_frames/train_dit_features \
  --use_gpu --run_projection --run_blender --only_body_joints \
  --wan_height 320 --wan_width 640 --n_frames 5 --gpu_id 0 --dataset_repeat_wan 8 &&
  python run_pipeline.py \
  --input_dir /host/data2/mint/Motion_Dataset/Mixamo/single_character_n=100/fbx/testset_motion/ \
  --render_output_dir /host/data2/mint/Motion_Dataset/Mixamo/single_character_n=100/dat/testset_motion/render \
  --format_output_dir /host/data2/mint/Motion_Dataset/Mixamo/single_character_n=100/dat/testset_motion/rdy_to_wan/all_frames \
  --chunk_output_dir  /host/data2/mint/Motion_Dataset/Mixamo/single_character_n=100/dat/testset_motion/rdy_to_wan/5_frames \
  --host_prefix /host \
  --vae_output_path /host/data2/mint/Motion_Dataset/Mixamo/single_character_n=100/dat/testset_motion/wan_output/5_frames/latents \
  --wan_output_path /host/data2/mint/Motion_Dataset/Mixamo/single_character_n=100/dat/testset_motion/wan_output/5_frames/test_dit_features \
  --use_gpu --run_projection --run_blender --only_body_joints \
  --wan_height 320 --wan_width 640 --n_frames 5 --gpu_id 0 --dataset_repeat_wan 8