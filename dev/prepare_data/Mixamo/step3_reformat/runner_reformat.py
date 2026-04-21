import numpy as np
import os, glob, argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--output_path', type=str, required=True)
# Whether to combine all characters into a single output folder (instead of one folder per character)
args = parser.parse_args()

if __name__ == "__main__":
    char_dirs = glob.glob(os.path.join(args.data_path, '*'))
    for char in char_dirs:
        if not os.path.isdir(char) or "_logs" in char:
            continue
        char_name = os.path.basename(char)
        # cmd = f"python gen_data_format.py --data_path /data2/mint/Motion_Dataset/Mixamo/testset_motion_720p/mannequin/ --output_path /data2/mint/Motion_Dataset/Mixamo/rdy_testset_mixamo_720p_only_body_with_motion_data/"
        cmd = f"python gen_data_format.py --data_path \"{char}\" --output_path \"{os.path.join(args.output_path, char_name)}\""
        print(f"Running command: {cmd}")
        os.system(cmd)
    