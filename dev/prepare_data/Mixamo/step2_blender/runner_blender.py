"""
run_multiple_chars.py
─────────────────────
Iterate over character-organised FBX directories and run the Blender /
projection pipeline for each motion file.

All UI (progress bars, live output panels, plain fallback) is delegated to
the shared display utilities in examples/wanvideo/my_scripts/utils/.

Layout
------
  _parse_args()       – argparse definitions only
  _build_command()    – assemble the shell command for ONE FBX file
  _collect_tasks()    – walk character dirs and return (label, cmd) pairs
  main()              – discover dirs → collect tasks → dispatch to UI
"""

import numpy as np
import glob
import os
import sys
import torch as th
import shlex
import subprocess
import time
from multiprocessing import Pool
from datetime import datetime

# ── resolve shared display utils ──────────────────────────────────────────────
_UTILS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../examples/wanvideo/my_scripts")
)
sys.path.insert(0, _UTILS_ROOT)

from utils.display_common import die, run_plain_batch

try:
    from utils.display_rich import run_batch
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    print("rich not found – pip install rich  (falling back to plain output)\n")



# ══════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="Batch-process character FBX files through Blender/projection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # paths
    p.add_argument("--blender_bin", default="/host/ist/users/puntawatp/Dev/SkelAg/Blender/blender-5.0.0-linux-x64/blender", type=str,
                   help="Path to Blender executable.")
    p.add_argument("--input_dir",  required=True,
                   help="Directory containing character subdirectories with FBX files.")
    p.add_argument("--output_dir", required=True,
                   help="Root output directory; one sub-folder per character is created.")
    p.add_argument("--max_log_lines", type=int, default=20,
                   help="Rolling log-window size shown while a subprocess is running.")
    # camera / render
    p.add_argument("--n_cam",       type=int,   default=1)
    p.add_argument("--follow_bone", type=str,   default="mixamorig:Hips")
    p.add_argument("--cam_height",  type=float, default=3.0)
    p.add_argument("--cam_radius",  type=float, default=4.5)
    p.add_argument("--img_width",   type=int,   default=1280)
    p.add_argument("--img_height",  type=int,   default=720)
    # flags
    p.add_argument("--run_blender",       action="store_true", default=False,
                   help="Enable Blender execution.")
    p.add_argument("--run_projection",    action="store_true", default=False,
                   help="Enable 2-D projection after rendering.")
    p.add_argument("--use_gpu",           action="store_true", default=False,
                   help="Use GPU for Blender rendering.")
    p.add_argument("--gpu_id",            type=str, nargs="+", default="0",
                   help="GPU ID(s) to use for rendering.")
    p.add_argument("--only_body_joints",  action="store_true", default=False,
                   help="Render body joints only (no fingers).")
    p.add_argument("--skip_plot_map",     action="store_true", default=False,
                   help="Skip 2-D joint heatmap / skeleton overlay plotting.")
    p.add_argument("--n_gpus",       type=int, default=1,
                   help="Available GPU count for splitting the fbx files.")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Command builder
# ══════════════════════════════════════════════════════════════════════════════

def _build_command(motion_file: str, char_output_dir: str, gpu_id: str, args) -> str:
    """Return the complete shell command to process one FBX file."""
    render = (
        f"--n_cam {args.n_cam}"
        f" --follow_bone {shlex.quote(args.follow_bone)}"
        f" --cam_height {args.cam_height}"
        f" --cam_radius {args.cam_radius}"
        f" --img_width {args.img_width}"
        f" --img_height {args.img_height}"
        f" --blender_bin \"{args.blender_bin}\""
        f" --gpu_id {gpu_id}"
    )
    cmd = (
        f"python process_single_fbx.py --fbx {shlex.quote(motion_file)}"
        f" --out_dir {shlex.quote(char_output_dir)} {render}"
    )
    if args.use_gpu:          cmd += " --use_gpu"
    if args.run_blender:      cmd += " --run_blender"
    if args.run_projection:   cmd += " --run_projection"
    if args.only_body_joints: cmd += " --only_body_joints"
    if args.skip_plot_map:    cmd += " --skip_plot_map"
    print(cmd)
    return cmd


# ══════════════════════════════════════════════════════════════════════════════
# Task collector
# ══════════════════════════════════════════════════════════════════════════════

def _collect_tasks(character_dirs: list, args) -> list:
    """
    Walk *character_dirs*, glob FBX files, and return a list of
    (label, command) pairs ready to hand to a display-batch runner.

    Output sub-directories are created here as a side-effect.
    """
    tasks = []
    for char_dir in sorted(character_dirs):
        char_name = os.path.basename(char_dir.rstrip("/"))
        fbx_files = sorted(glob.glob(os.path.join(char_dir, "*.fbx")))
        if not fbx_files:
            print(f"⚠  No FBX files in {char_dir} – skipping.")
            continue
        if th.cuda.is_available() and args.n_gpus > th.cuda.device_count():
            print(f"⚠  Requested {args.n_gpus} GPUs but only {th.cuda.device_count()} available. Adjusting to available count.")
            args.n_gpus = th.cuda.device_count()
        
        print(f"Character: {char_name} | {len(fbx_files)} motion file(s) found.")
        
        if args.gpu_id is None:
            gpu_ids = [str(i) for i in range(args.n_gpus)]
        else:
            gpu_ids = args.gpu_id
        
        for j, motion_file in enumerate(fbx_files):
            # Round-robin assign GPU
            gpu_id = gpu_ids[j % len(gpu_ids)]
            motion_name = os.path.basename(motion_file)
            label = f"{char_name}/{motion_name}"
            out_dir = os.path.join(args.output_dir, char_name)
            os.makedirs(out_dir, exist_ok=True)
            tasks.append((label, _build_command(motion_file, out_dir, gpu_id, args)))
    return tasks


# ══════════════════════════════════════════════════════════════════════════════
# Multi-GPU batch runner using process pool (output to log files)
# ══════════════════════════════════════════════════════════════════════════════

def _run_task_with_gpu(task_input: dict) -> dict:
    """
    Worker process: run one task, redirect output to log file, return result.
    
    Parameters
    ----------
    task_input : dict with keys:
        - global_idx: task index for result ordering
        - label: human-readable task label
        - command: shell command to run
        - gpu_id: GPU ID (set as CUDA_VISIBLE_DEVICES)
        - log_dir: directory to write log file
    
    Returns
    -------
    dict with keys:
        - global_idx: index for ordering
        - label: task label
        - success: True if exit code 0
        - log_file: path to log file
    """
    global_idx = task_input['global_idx']
    label = task_input['label']
    command = task_input['command']
    gpu_id = task_input['gpu_id']
    log_dir = task_input['log_dir']
    
    # Clean label for filename
    safe_label = label.replace("/", "_").replace(" ", "_")[:50]
    log_file = os.path.join(log_dir, f"gpu{gpu_id}_{safe_label}.log")
    
    # Set GPU for this process
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpu_id
    
    # Run and redirect to log file
    try:
        with open(log_file, 'w') as f:
            proc = subprocess.Popen(
                command,
                shell=True,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
                universal_newlines=True,
            )
            proc.wait()
            success = proc.returncode == 0
    except Exception as e:
        with open(log_file, 'a') as f:
            f.write(f"\n[ERROR] Exception: {e}\n")
        success = False
    
    return {
        'global_idx': global_idx,
        'label': label,
        'success': success,
        'log_file': log_file,
    }


def _run_multi_gpu_batch(tasks: list, gpu_ids: list, log_base_dir: str) -> list:
    """
    Run tasks in parallel using process pool, each GPU gets own worker.
    Output redirected to log files per GPU/task.
    
    Parameters
    ----------
    tasks     : list of (label, command) tuples
    gpu_ids   : list of GPU IDs (strings)
    log_base_dir : directory to store log files
    
    Returns
    -------
    list[bool] — success flag for each task (in original order)
    """
    if not tasks or not gpu_ids:
        return []
    
    os.makedirs(log_base_dir, exist_ok=True)
    
    # Prepare task inputs with GPU assignment (round-robin)
    task_inputs = []
    for idx, (label, cmd) in enumerate(tasks):
        gpu_id = gpu_ids[idx % len(gpu_ids)]
        task_inputs.append({
            'global_idx': idx,
            'label': label,
            'command': cmd,
            'gpu_id': gpu_id,
            'log_dir': log_base_dir,
        })
    
    # Run tasks in parallel using process pool
    num_workers = len(gpu_ids)
    print(f"\nSpawning {num_workers} worker processes ({len(gpu_ids)} GPUs)...")
    print(f"Logs will be written to: {log_base_dir}\n")
    
    with Pool(processes=num_workers) as pool:
        results = []
        task_count = len(task_inputs)
        
        # Use imap_unordered for live progress feedback
        for i, result in enumerate(pool.imap_unordered(_run_task_with_gpu, task_inputs), 1):
            idx = result['global_idx']
            label = result['label']
            success = result['success']
            log_file = result['log_file']
            status = "✓" if success else "✗"
            print(f"[{i:4d}/{task_count}] {status} {label}  →  {log_file}")
            results.append((idx, success))
    
    # Reconstruct results in original task order
    results_dict = {idx: ok for idx, ok in results}
    ordered_results = [results_dict[i] for i in range(len(tasks))]
    
    passed = sum(ordered_results)
    print(f"\n{'='*70}")
    print(f"Completed: {passed}/{len(tasks)} succeeded, {len(tasks)-passed} failed")
    print(f"Logs: {log_base_dir}")
    print(f"{'='*70}\n")
    
    return ordered_results

def main() -> None:
    args = _parse_args()

    character_dirs = glob.glob(os.path.join(args.input_dir, "*/"))
    if not character_dirs:
        die(f"No character directories found in: {args.input_dir}")

    tasks = _collect_tasks(character_dirs, args)
    if not tasks:
        die("No FBX files found in any character directory.")

    print(f"Found {len(character_dirs)} character(s) | {len(tasks)} motion file(s)\n")

    # Choose execution mode
    if args.n_gpus > 1:
        # Multi-GPU parallel mode with process pool (output to logs)
        log_dir = os.path.join(args.output_dir, "_logs")
        gpu_ids = [str(i) for i in range(args.n_gpus)] if args.gpu_id is None else args.gpu_id
        results = _run_multi_gpu_batch(tasks, gpu_ids, log_dir)
    elif HAS_RICH:
        # Sequential mode with rich progress
        results = run_batch(
            tasks,
            overall_description="Processing motion files...",
            max_log_lines=args.max_log_lines,
        )
    else:
        # Fallback: sequential plain output
        results = run_plain_batch(tasks)

    failed = [tasks[i][0] for i, ok in enumerate(results) if not ok]
    if failed:
        print(f"\n⚠  {len(failed)} failed task(s):")
        for label in failed:
            print(f"   • {label}")


if __name__ == "__main__":
    main()
