#!/usr/bin/env python3
"""
Verify and organize output from parallel rendering.
Checks that all frames were rendered correctly and optionally merges skeleton JSON data.
"""

import os
import json
import glob
import argparse
from pathlib import Path
from collections import defaultdict


def verify_rendered_frames(base_dir, n_cams, expected_frames):
    """
    Verify that all expected frames were rendered for all cameras.
    
    Args:
        base_dir: Base output directory (e.g., output/character_name/)
        n_cams: Number of cameras
        expected_frames: Expected number of frames per camera
    
    Returns:
        Dict with verification results
    """
    print(f"[#] Verifying renders in: {base_dir}")
    print(f"[#] Expected: {n_cams} cameras × {expected_frames} frames")
    
    results = {
        'total_expected': n_cams * expected_frames,
        'total_found': 0,
        'cameras': {},
        'missing': [],
        'complete': True
    }
    
    for cam_idx in range(n_cams):
        cam_dir = os.path.join(base_dir, f"cam_{cam_idx}")
        if not os.path.exists(cam_dir):
            print(f"[!] Camera {cam_idx} directory not found: {cam_dir}")
            results['cameras'][cam_idx] = {
                'found': 0,
                'missing': list(range(expected_frames)),
                'complete': False
            }
            results['complete'] = False
            continue
        
        # Find all rendered frames
        frames = glob.glob(os.path.join(cam_dir, "frame*.png"))
        frame_indices = []
        for f in frames:
            basename = os.path.basename(f)
            # Extract frame number from frame0000.png
            try:
                idx = int(basename.replace('frame', '').replace('.png', ''))
                frame_indices.append(idx)
            except ValueError:
                print(f"[!] Could not parse frame index from: {basename}")
        
        frame_indices = sorted(set(frame_indices))
        expected_indices = set(range(expected_frames))
        missing = sorted(expected_indices - set(frame_indices))
        
        results['cameras'][cam_idx] = {
            'found': len(frame_indices),
            'missing': missing,
            'complete': len(missing) == 0
        }
        results['total_found'] += len(frame_indices)
        
        if missing:
            results['complete'] = False
            results['missing'].extend([(cam_idx, f) for f in missing])
            print(f"[!] Camera {cam_idx}: Missing {len(missing)} frames: {missing[:10]}{'...' if len(missing) > 10 else ''}")
        else:
            print(f"[✓] Camera {cam_idx}: All {len(frame_indices)} frames present")
    
    return results


def merge_skeleton_json(base_dir, n_cams, output_path=None):
    """
    Merge skeleton JSON files from multiple frame workers into complete files.
    This is only needed if you used --n_frame_workers > 1.
    
    The script creates skeleton_cam_X.json when rendering all frames together,
    but if you split frames across workers, you need to merge the skeleton data.
    
    Note: For now, this just checks if the JSON exists.
    If you used frame workers, you'll need to implement proper merging.
    """
    print(f"\n[#] Checking skeleton JSON files...")
    
    for cam_idx in range(n_cams):
        cam_dir = os.path.join(base_dir, f"cam_{cam_idx}")
        json_path = os.path.join(cam_dir, f"skeleton_cam_{cam_idx}.json")
        
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
            n_frames = len(data.get('joints_3d', []))
            print(f"[✓] Camera {cam_idx}: skeleton JSON found ({n_frames} frames)")
        else:
            print(f"[!] Camera {cam_idx}: skeleton JSON not found")
            print(f"    Note: JSON is only created when rendering all frames in one job")
            print(f"    If you used --n_frame_workers > 1, the JSON won't be created")


def main():
    parser = argparse.ArgumentParser(description="Verify parallel rendering output")
    parser.add_argument('base_dir', type=str, help='Base output directory (e.g., output/character_name/)')
    parser.add_argument('--n_cams', type=int, required=True, help='Number of cameras')
    parser.add_argument('--n_frames', type=int, required=True, help='Expected number of frames per camera')
    parser.add_argument('--check_json', action='store_true', help='Also check skeleton JSON files')
    
    args = parser.parse_args()
    
    # Verify frames
    results = verify_rendered_frames(args.base_dir, args.n_cams, args.n_frames)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total frames found: {results['total_found']} / {results['total_expected']}")
    print(f"Complete: {'YES' if results['complete'] else 'NO'}")
    
    if results['missing']:
        print(f"\nMissing frames: {len(results['missing'])}")
        print("First 20 missing (cam_idx, frame_idx):")
        for cam_idx, frame_idx in results['missing'][:20]:
            print(f"  Camera {cam_idx}, Frame {frame_idx}")
    
    # Check JSON if requested
    if args.check_json:
        merge_skeleton_json(args.base_dir, args.n_cams)
    
    return 0 if results['complete'] else 1


if __name__ == "__main__":
    exit(main())
