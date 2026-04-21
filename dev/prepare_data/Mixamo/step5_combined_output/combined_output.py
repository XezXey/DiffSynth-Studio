import os, glob, argparse
import pandas as pd
parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, required=True)
parser.add_argument('--metadata_to_combined', type=str, default=None, help='Filename for search and copy metadata CSV to combined output folder (if --combined_output is set)')
args = parser.parse_args()

"""
#NOTE: This script combined all character-specific subfolders into a single folder (For WanPipeline).
"""


if __name__ == "__main__":
    # Move all videos and motion_data.npz files from character subfolders to the main output folder
    combined_path = f'{args.input_path}/all/'
    os.makedirs(combined_path, exist_ok=True)
    meta_data_files = []
    char_dirs = glob.glob(os.path.join(args.input_path, '*'))
    for char in char_dirs:
        char_name = os.path.basename(char)
        if char_name == 'all':
            continue  # Skip the combined folder itself if it already exists
        char_output_folder = os.path.join(args.input_path, char_name)
        for file in glob.glob(os.path.join(char_output_folder, '*')):
            if file.endswith('.csv'):
                # Read and edit the data to match new filename (prepend character name to video column)
                df = pd.read_csv(file)
                if 'video' in df.columns:
                    df['video'] = char_name + '_' + df['video'].astype(str)
                if 'motion' in df.columns:
                    df['motion'] = char_name + '_' + df['motion'].astype(str)
                # Save the edited CSV to the combined folder with a new name
                new_csv_name = f"{char_name}_{os.path.basename(file)}"
                df.to_csv(os.path.join(combined_path, new_csv_name), index=False)
                if args.metadata_to_combined in file and meta_data_files is not None:
                    meta_data_files.append(os.path.join(combined_path, new_csv_name))
            else:
                new_name = f"{char_name}_{os.path.basename(file)}"
                if os.path.exists(os.path.join(combined_path, new_name)):
                    # Unlink existing file if it exists (in case it's a symlink pointing to an old file)
                    os.unlink(os.path.join(combined_path, new_name))
                # os.system(f'ln -s "{file}" "{os.path.join(combined_path, new_name)}"')  # Create symlink to avoid data duplication
                os.system(f'cp "{file}" "{os.path.join(combined_path, new_name)}"')  # Create symlink to avoid data duplication

    # Concat metadata CSV files into a single CSV (use only 1 header)
    if len(meta_data_files) > 0:
        combined_meta_df = None
        for file in meta_data_files:
            df = pd.read_csv(file, header=0)
            if combined_meta_df is None:
                combined_meta_df = df
            else:
                combined_meta_df = pd.concat([combined_meta_df, df], ignore_index=True)
        combined_meta_df.to_csv(os.path.join(combined_path, f'all_{args.metadata_to_combined}'), index=False)