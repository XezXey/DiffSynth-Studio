#NOTE: This code is for chunking the rdy-to-train data into smaller #N frames to prevent OOM during training and complicating edit the dataloader.
import os
import tqdm
import numpy as np
import pandas as pd
import argparse
import torchvision
from mylogger.logger import init_logger
logger = init_logger('chunk_data')

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True, help='Path to the input directory containing .npz and .mp4 files.')
parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory to save chunked data.')
parser.add_argument('--n_frames', type=int, default=81, help='Number of frames per chunk.')
parser.add_argument('--overlap', type=int, default=0, help='Number of overlapping frames between chunks. If None, no overlap is used.')
parser.add_argument('--metadata_file', type=str, required=True, help='Path to the metadata file (optional).')
args = parser.parse_args()

if __name__ == '__main__':


    metadata = pd.read_csv(args.metadata_file)
    logger.info(f'Loaded metadata from {args.metadata_file} with {len(metadata)} entries.')
    logger.info(f'Chunking videos into segments of {args.n_frames} frames')
    logger.warning(f'Overlap duration: {args.overlap} frames')

    os.makedirs(args.output_dir, exist_ok=True)
    new_metadata = pd.DataFrame({'video': [], 'motion': [], 'prompt': []})
    for vid_idx, row in tqdm.tqdm(metadata.iterrows(), total=len(metadata)):
        video_path = os.path.join(args.input_dir, row['video'])
        motion_path = os.path.join(args.input_dir, row['motion'])
        prompt = row['prompt']

        # Load video/motion frames
        video_frames = torchvision.io.read_video(video_path, pts_unit='sec')[0]  # (T, H, W, C)
        T_vid = video_frames.shape[0]
        motion = np.load(motion_path)
        assert T_vid == motion['joints_3d'].shape[0], f"Mismatch in number of frames between video (T={T_vid}) and motion (T={motion['joints_3d'].shape[0]}): {video_path}"
        assert T_vid == motion['joints_2d'].shape[0], f"Mismatch in number of frames between video (T={T_vid}) and motion (T={motion['joints_2d'].shape[0]}): {video_path}"
        assert T_vid == motion['cams_extr'].shape[0], f"Mismatch in number of frames between video (T={T_vid}) and camera extrinsics (T={motion['cams_extr'].shape[0]}): {video_path}"

        # Chunk video frames into overlapping segments
        video_chunks = []
        motion_2d_chunks = []
        motion_3d_chunks = []
        cams_extr_chunks = []
        chunk_metadata = []
        start_frame = 0
        while start_frame < len(video_frames):
            end_frame = start_frame + args.n_frames
            v_chunk = video_frames[start_frame:end_frame]
            m2d_chunk = motion['joints_2d'][start_frame:end_frame]
            m3d_chunk = motion['joints_3d'][start_frame:end_frame]
            ce_chunk = motion['cams_extr'][start_frame:end_frame]
            if len(v_chunk) < args.n_frames:
                v_chunk = video_frames[-args.n_frames:]  # Pad with last frames if needed
                m2d_chunk = motion['joints_2d'][-args.n_frames:]
                m3d_chunk = motion['joints_3d'][-args.n_frames:]
                ce_chunk = motion['cams_extr'][-args.n_frames:]
                video_chunks.append(v_chunk)
                motion_2d_chunks.append(m2d_chunk)
                motion_3d_chunks.append(m3d_chunk)
                cams_extr_chunks.append(ce_chunk)
                chunk_metadata.append((len(video_frames) - args.n_frames, len(video_frames)))
                break
            video_chunks.append(v_chunk)
            motion_2d_chunks.append(m2d_chunk)
            motion_3d_chunks.append(m3d_chunk)
            cams_extr_chunks.append(ce_chunk)
            chunk_metadata.append((start_frame, end_frame))
            start_frame += (args.n_frames - args.overlap)

        # Save chunked data
        video_base_name = row['video'].split('.')[0]
        # print(len(video_chunks), len(motion_2d_chunks), len(motion_3d_chunks), len(cams_extr_chunks), len(chunk_metadata))
        assert len(video_chunks) == len(motion_2d_chunks) == len(motion_3d_chunks) == len(cams_extr_chunks) == len(chunk_metadata)
        for i, (v_chunk, m2d_chunk, m3d_chunk, ce_chunk, (s_f, e_f)) in enumerate(zip(video_chunks, motion_2d_chunks, motion_3d_chunks, cams_extr_chunks, chunk_metadata)):
            chunk_video_path = os.path.join(args.output_dir, f'{video_base_name}_chunk-{i}_video.mp4')
            chunk_motion_path = os.path.join(args.output_dir, f'{video_base_name}_chunk-{i}_motion.npz')
            # Save video chunk
            torchvision.io.write_video(chunk_video_path, v_chunk, fps=30, video_codec='libx264', options={'crf': '17'})
            # Save motion chunk
            arr = {
                'joints_2d': m2d_chunk,
                'joints_3d': m3d_chunk,
                'cams_extr': ce_chunk,
                'cams_intr': motion['cams_intr'],  # Keep original intrinsics
                'joint_names': motion['joint_names'],
                'bones': motion['bones'],
                'video_name': f'{video_base_name}_chunk-{i}_video.mp4',
                'chunk_id': i,
                'frame_range': (s_f, e_f)
            }
            np.savez_compressed(chunk_motion_path, **arr)
            # Update metadata
            new_metadata = pd.concat([new_metadata, pd.DataFrame({'video': [os.path.basename(chunk_video_path)], 'motion': [os.path.basename(chunk_motion_path)], 'prompt': [prompt]})], ignore_index=True)
        # Save new metadata for chunked data
    old_metadata_name = os.path.basename(args.metadata_file).split('.')[0]
    new_metadata_path = os.path.join(args.output_dir, f'{old_metadata_name}_{args.n_frames}frames.csv')
    new_metadata.to_csv(new_metadata_path, index=False)