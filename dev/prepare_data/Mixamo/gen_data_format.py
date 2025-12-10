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
import tqdm
import pandas as pd
import glob
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, required=True, help='Root directory of the video dataset')
parser.add_argument('--output_path', type=str, required=True, help='Output directory for formatted data')
args = parser.parse_args()

def vid_from_frames(input_path, output_video_path):
    cmd = f'ffmpeg -y -framerate 30 -i {input_path} -c:v libx264 -pix_fmt yuv420p {output_video_path}'
    os.system(cmd + " > /dev/null 2>&1")

if __name__ == "__main__":
    data_path = args.data_path
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    df = pd.DataFrame(columns=['video', 'prompt'])
    # Example path: /data/mint/Motion_Dataset/Mixamo/output_mixamo/<motion_name>/<camera_name>/
    # cam_0 = front view, cam_1 = right side view, cam_2 = back view, cam_3 = left side view
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
                
            # Write metadata
            
            with open(os.path.join(output_path, 'metadata.csv'), 'a') as f:
                cam_desc = {'cam_0': 'front', 'cam_1': 'right side', 'cam_2': 'back', 'cam_3': 'left side'}.get(cam_name, cam_name)
                prompt = f"A person wearing a grey crop top, yellow pants with blue stripes, black sneakers, orange visor glasses, and orange headphones performs {motion_name}, captured from the {cam_desc} view."
                vid_file = f"{vid_name}_render.mp4"
                df = pd.concat([df, pd.DataFrame([[vid_file, prompt]], columns=['video', 'prompt'])], ignore_index=True)
    df.to_csv(os.path.join(output_path, 'metadata.csv'), index=False)
            