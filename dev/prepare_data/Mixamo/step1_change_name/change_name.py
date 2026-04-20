#NOTE: Everything need to be run in singularity (Using /host/... instead of /data/...)
import os, glob, tqdm
import shutil
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, required=True)
parser.add_argument('--output_path', type=str, required=True)
args = parser.parse_args()

if __name__ == '__main__':
    if args.output_path is not None:
        os.makedirs(args.output_path, exist_ok=True)
    # pattern = r"([a-zA-Z\s]+?)(?: \((\d+)\))?\.fbx"
    for f in tqdm.tqdm(glob.glob(f'{args.input_path}/*.fbx')):
        filename = os.path.basename(f)
        new_filename = filename.replace(" - ", '-')
        new_filename = new_filename.replace(" Ver. ", '-')
        new_filename = new_filename.replace(" ", "-")
        new_filename = new_filename.replace("(", "v")
        new_filename = new_filename.replace(")", "")
        new_filename = new_filename.replace("'", "-")
        new_filename = new_filename.replace("_", "-")

        os.symlink(f, os.path.join(args.output_path, new_filename))
        