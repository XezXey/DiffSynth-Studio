# Generate data format for DiffSynth-Studio
# 1. Directory structure:
#       data/example_video_dataset/
#       ├── metadata.csv
#       ├── video_1.mp4
#       └── video_2.mp4
# 2. metadata.csv columns:
#       video,prompt
#       video_1.mp4,"A person walking in the park"

import os
import json
import tqdm
import pandas as pd
import glob
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, required=True, help='Root directory of the video dataset')
parser.add_argument('--output_path', type=str, required=True, help='Output directory for formatted data')
args = parser.parse_args()

def vid_from_frames(input_path, output_video_path):
    # cmd = f'ffmpeg -y -framerate 30 -i {input_path} -c:v libx264 -pix_fmt yuv420p {output_video_path}'
    cmd = f'ffmpeg -y -framerate 30 -i {input_path} -c:v libopenh264 -pix_fmt yuv420p {output_video_path}'
    # print(input_path, output_video_path)
    os.system(cmd + " > /dev/null 2>&1")

def search_prompt(output_path, motion_name, cam_desc):
    # characters = ['mannequin', 'michelle', 'prisoner-b-styperek', 'vampire-a-lusth']
    # with open('./character_prompt_mapping.json', 'r') as f:
    #     char_prompt_mapping = json.load(f)
    # for char in characters:
    #     if char in output_path:
    #         print(f"Found character '{char}' in path. Using corresponding prompt template.")
    #         return char_prompt_mapping[char].replace('<motion_name>', motion_name).replace('<cam_desc>', cam_desc)
        
    # Default prompt if no character name is found in the path
    return f"A person performs {motion_name}, captured from the {cam_desc} view. Static camera perspective, no zoom or panning."

if __name__ == "__main__":
    data_path = args.data_path
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    df_render = pd.DataFrame(columns=['video', 'motion', 'prompt'])
    df_single_motion = pd.DataFrame(columns=['video', 'motion', 'prompt'])
    df_single_sample = pd.DataFrame(columns=['video', 'motion', 'prompt'])
    # Example path: /data/mint/Motion_Dataset/Mixamo/output_mixamo/<motion_name>/<camera_name>/
    # cam-0 = front view, cam-1 = right side view, cam-2 = back view, cam-3 = left side view
    motion_dirs = glob.glob(os.path.join(data_path, '*'))
    for motion in tqdm.tqdm(motion_dirs, desc="Processing motions:"):
        tqdm.tqdm.write(f"Processing motion: {motion}")
        motion_name = os.path.basename(motion)
        camera_dirs = glob.glob(os.path.join(motion, '*'))
        for cam in camera_dirs:
            cam_name = os.path.basename(cam)
            vid_name = f"{motion_name.replace(' ', '_')}_{cam_name}"
            # Generate video file if not exists
            if not os.path.exists(f'{output_path}/{vid_name}_render.mp4') and len(glob.glob(os.path.join(cam, 'frame*.png'))) > 0:
                os.makedirs(f'{output_path}/', exist_ok=True)
                # Create .mp4 from frames
                # Replace spaces in path with '\ ' for ffmpeg command
                input_path = os.path.join(cam, 'frame%04d.png').replace(' ', '\ ')
                output_video_path = os.path.join(output_path, f'{vid_name}_render.mp4').replace(' ', '\ ')
                vid_from_frames(input_path, output_video_path)
            if not os.path.exists(f'{output_path}/{vid_name}_proj.mp4') and len(glob.glob(os.path.join(cam, 'proj*.png'))) > 0:
                os.makedirs(f'{output_path}/', exist_ok=True)
                # Create .mp4 from projection frames
                input_path = os.path.join(cam, 'proj%04d.png').replace(' ', '\ ')
                output_video_path = os.path.join(output_path, f'{vid_name}_proj.mp4').replace(' ', '\ ')
                vid_from_frames(input_path, output_video_path)

            
            # Copy motion_data.npz
            src_npz = os.path.join(cam, 'motion.npz').replace(' ', '\ ')
            dst_npz = os.path.join(output_path, f'{vid_name}_motion.npz').replace(' ', '\ ')
            os.system(f'cp {src_npz} {dst_npz}')
            
            # Write metadata
            with open(os.path.join(output_path, 'metadata.csv'), 'a') as f:
                cam_desc = {'cam-0': 'front', 'cam-1': 'right side', 'cam-2': 'back', 'cam-3': 'left side'}.get(cam_name, cam_name)
                prompt = search_prompt(output_path, motion_name, cam_desc)
                
                motion_file = f"{vid_name}_motion.npz"
                vid_file = f"{vid_name}_render.mp4"
                df_render = pd.concat([df_render, pd.DataFrame([[vid_file, motion_file, prompt]], columns=['video', 'motion', 'prompt'])], ignore_index=True)
            
    df_render.to_csv(os.path.join(output_path, 'metadata.csv'), index=False)
    df_front_view = df_render[df_render['motion'].str.contains('cam-0')]
    df_front_view.to_csv(os.path.join(output_path, 'metadata_front-view.csv'), index=False)
    

            
