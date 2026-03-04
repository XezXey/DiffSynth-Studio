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

import glob
import os
import sys

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
    p.add_argument("--only_body_joints",  action="store_true", default=False,
                   help="Render body joints only (no fingers).")
    p.add_argument("--skip_plot_map",     action="store_true", default=False,
                   help="Skip 2-D joint heatmap / skeleton overlay plotting.")
    p.add_argument("--cam_workers",       type=int, default=1,
                   help="Number of worker processes for camera rendering in Blender.")
    p.add_argument("--frame_workers",     type=int, default=8,
                   help="Number of worker processes for frame rendering in Blender.")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Command builder
# ══════════════════════════════════════════════════════════════════════════════

def _build_command(motion_file: str, char_output_dir: str, args) -> str:
    """Return the complete shell command to process one FBX file."""
    render = (
        f"--n_cam {args.n_cam}"
        f" --follow_bone {args.follow_bone}"
        f" --cam_height {args.cam_height}"
        f" --cam_radius {args.cam_radius}"
        f" --img_width {args.img_width}"
        f" --img_height {args.img_height}"
        f" --blender_bin \"{args.blender_bin}\""
        f" --cam_workers {args.cam_workers}"
        f" --frame_workers {args.frame_workers}"
    )
    cmd = f'python run.py --fbx "{motion_file}" --out_dir "{char_output_dir}" {render}'
    if args.use_gpu:          cmd += " --use_gpu"
    if args.run_blender:      cmd += " --run_blender"
    if args.run_projection:   cmd += " --run_projection"
    if args.only_body_joints: cmd += " --only_body_joints"
    if args.skip_plot_map:    cmd += " --skip_plot_map"
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
        for motion_file in fbx_files:
            motion_name = os.path.basename(motion_file)
            label       = f"{char_name}/{motion_name}"
            out_dir     = os.path.join(args.output_dir, char_name)
            os.makedirs(out_dir, exist_ok=True)
            tasks.append((label, _build_command(motion_file, out_dir, args)))
    return tasks


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = _parse_args()

    character_dirs = glob.glob(os.path.join(args.input_dir, "*/"))
    if not character_dirs:
        die(f"No character directories found in: {args.input_dir}")

    tasks = _collect_tasks(character_dirs, args)
    if not tasks:
        die("No FBX files found in any character directory.")

    print(f"Found {len(character_dirs)} character(s) | {len(tasks)} motion file(s)\n")

    if HAS_RICH:
        results = run_batch(
            tasks,
            overall_description="Processing motion files...",
            max_log_lines=args.max_log_lines,
        )
    else:
        results = run_plain_batch(tasks)

    failed = [tasks[i][0] for i, ok in enumerate(results) if not ok]
    if failed:
        print(f"\n⚠  {len(failed)} failed task(s):")
        for label in failed:
            print(f"   • {label}")


if __name__ == "__main__":
    main()
