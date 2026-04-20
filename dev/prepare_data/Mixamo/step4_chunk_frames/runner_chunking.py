import numpy as np
import os, glob, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--n_frames', type=int, default=5, help='Number of frames per chunk.')
parser.add_argument('--overlap', type=int, default=0, help='Number of overlapping frames between chunks. If None, no overlap is used.')
parser.add_argument('--metadata_example', type=str, default='metadata.csv', help='Name of the metadata file to search for in each character directory')
args = parser.parse_args()

def search_metadata_file(input_dir, metadata_name):
    """Search for the metadata file in the given input directory."""
    for root, dirs, files in os.walk(input_dir):
        if metadata_name in files:
            return os.path.join(root, metadata_name)
    return None


if __name__ == "__main__":
    char_dirs = glob.glob(os.path.join(args.input_dir, '*'))
    for char in char_dirs:
        char_name = os.path.basename(char)
        """
        cmd = python chunk_data.py 
        --input_dir /host/data2/mint/Motion_Dataset/Mixamo/rdy_testset_mixamo_720p_only_body_with_motion_data 
        --output_dir /host/data2/mint/Motion_Dataset/Mixamo/rdy_testset_mixamo_720p_only_body_with_motion_data_5frames 
        --n_frames 5 
        --metadata_file /host/data2/mint/Motion_Dataset/Mixamo/rdy_testset_mixamo_720p_only_body_with_motion_data/metadata.csv
        """
        metadata_file = search_metadata_file(char, os.path.basename(args.metadata_example))
        if metadata_file is None:
            print(f"[Warning] No metadata file named '{args.metadata_example}' found in {char}. Skipping this character.")
            exit()
        chunk_settings = f"--n_frames {args.n_frames} --overlap {args.overlap}"

        cmd = f"python chunk_data.py --input_dir \"{char}\" --output_dir \"{os.path.join(args.output_dir, char_name)}\" {chunk_settings} --metadata_file \"{metadata_file}\""
        print(f"Running command: {cmd}")
        os.system(cmd)
