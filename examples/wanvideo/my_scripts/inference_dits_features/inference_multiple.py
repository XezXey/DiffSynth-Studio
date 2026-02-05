import os
import glob

motion_list = glob.glob("/host/data/mint/Motion_Dataset/Mixamo/rdy_mixamo_720p_with_motion_data/*_cam_0_render.mp4")
motion_name = []
for motion_path in motion_list:
    motion_name.append(os.path.basename(motion_path).replace("_cam_0_render.mp4", ""))

for m in motion_name:
    print(f"Processing motion: {m}")
    cmd = f"CUDA_VISIBLE_DEVICES=2 python ./examples/wanvideo/my_scripts/inference_dits_features/inference_video_dits_features.py --input_video /host/data/mint/Motion_Dataset/Mixamo/rdy_mixamo_720p_only_body_with_motion_data/{m}_cam_0_render.mp4 --prompt \"A person wearing a grey crop top, yellow pants with blue stripes, black sneakers, orange visor glasses, and orange headphones performs {m}, captured from the front view. Static camera perspective, no zoom or pan.\" --extra_modules_ckpt /host/data/mint/SkelAg/training/ckpts/frontview_bodyjoints_320p/frontview_bodyjoints_320p-step-17100.safetensors  --save_path ./results/from_mixamo_render/{m}_cam_0/  --num_inference_steps 1000 --height 320 --width 640 --preferred_timestep_id=-20 --n_joints 25"
    os.system(cmd)