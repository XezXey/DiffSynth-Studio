"""
Parallel render launcher (pure Python – no Blender needed to run this script).

Workflow
--------
1.  (Recommended) Run export_skeleton.py first to produce the per-camera
    skeleton JSON files.  This is fast and single-threaded.  The launcher
    will read ``frame_range`` from these JSON files automatically.

2.  Run this launcher.  It discovers all FBX files, expands the 3-D job
    matrix (FBX × cameras × frame-chunks), and executes them in parallel
    using a subprocess pool.

Usage
-----
    python launch_parallel_render.py \\
        --fbx_path     ./mixamo_fbx/ \\
        --out_dir      ./output/ \\
        --n_cam        4 \\
        --frame_chunks 2 \\
        --workers      8 \\
        [--blender_bin /usr/bin/blender] \\
        [--render_engine cycles] \\
        [--render_samples 16] \\
        [--use_gpu]

Parallelism
-----------
Total jobs = n_fbx × n_cam × frame_chunks.
Each Blender process renders exactly one (camera, frame-chunk) pair of one
animation.  Multiple processes run concurrently up to ``--workers``.

Frame range
-----------
The launcher discovers the frame range (per FBX) by reading
``<out_dir>/<anim>/cam_0/skeleton_cam_0.json`` (written by export_skeleton.py).
If that file does not exist yet you can supply ``--frame_start`` /
``--frame_end`` explicitly on the command line.

Output naming
-------------
Each Blender job writes ``frame{N:04d}.png`` where N is the absolute Blender
frame number.  Chunks therefore share the same output directory and never
overwrite each other.

Logs for each job are written to ``<out_dir>/<anim>/cam_N/render_log_chunk_K.txt``.
"""

import argparse
import glob
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Rich progress tracker (with graceful fallback)
_SCRIPT_DIR_FOR_IMPORT = str(Path(__file__).parent.resolve())
if _SCRIPT_DIR_FOR_IMPORT not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR_FOR_IMPORT)
from lib.progress import create_tracker

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).parent.resolve()
_RENDER_SCRIPT = _SCRIPT_DIR / "render_single_cam.py"
_DEFAULT_BLENDER = os.environ.get("BLENDER_BIN", "blender")


# ---------------------------------------------------------------------------
# Job builder
# ---------------------------------------------------------------------------

def build_blender_cmd(
    blender_bin: str,
    fbx_path: str,
    out_dir: str,
    cam_idx: int,
    n_cam: int,
    cam_height: float,
    cam_radius: float,
    follow_bone: str,
    char_color,
    start_motion_frame: int,
    sub_sampling: int,
    img_width: int,
    img_height: int,
    render_engine: str,
    render_samples: int,
    use_gpu: bool,
    legacy_mode: bool,
    frame_start: int = None,
    frame_end: int = None,
) -> list:
    """Return the argv list for a single Blender render job."""
    cmd = [
        blender_bin, "--background",
        "--python", str(_RENDER_SCRIPT),
        "--",
        "--fbx_path",           fbx_path,
        "--out_dir",            out_dir,
        "--cam_idx",            str(cam_idx),
        "--n_cam",              str(n_cam),
        "--cam_height",         str(cam_height),
        "--cam_radius",         str(cam_radius),
        "--follow_bone",        follow_bone,
        "--start_motion_frame", str(start_motion_frame),
        "--sub_sampling",       str(sub_sampling),
        "--img_width",          str(img_width),
        "--img_height",         str(img_height),
        "--render_engine",      render_engine,
        "--render_samples",     str(render_samples),
    ]
    if char_color:
        cmd += ["--char_color", char_color]
    if use_gpu:
        cmd.append("--use_gpu")
    if legacy_mode:
        cmd.append("--legacy_mode")
    if frame_start is not None:
        cmd += ["--frame_start", str(frame_start)]
    if frame_end is not None:
        cmd += ["--frame_end", str(frame_end)]
    return cmd


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def run_job(job: dict) -> dict:
    """
    Execute one Blender render job.  Returns a dict with job info + exit code.
    Stdout/stderr are written to a per-job log file.
    """
    cmd      = job["cmd"]
    log_path = job["log_path"]
    label    = job["label"]

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, "w") as log_fh:
        result = subprocess.run(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
        )

    return {
        "label":      label,
        "returncode": result.returncode,
        "log_path":   log_path,
    }


# ---------------------------------------------------------------------------
# Frame-range helpers
# ---------------------------------------------------------------------------

def _read_frame_range_from_json(json_path: str):
    """Return (start, end) read from a skeleton JSON, or None on failure."""
    try:
        with open(json_path) as fh:
            data = json.load(fh)
        fr = data.get("frame_range")
        if fr and len(fr) == 2:
            return int(fr[0]), int(fr[1])
    except Exception:
        pass
    return None


def _discover_frame_range(out_dir: str, anim_name: str, fallback_start, fallback_end):
    """
    Try to read the frame range from cam_0's skeleton JSON.
    Falls back to (fallback_start, fallback_end) if the file doesn't exist yet.
    """
    json_path = os.path.join(out_dir, anim_name, "cam_0", "skeleton_cam_0.json")
    result = _read_frame_range_from_json(json_path)
    if result:
        return result
    if fallback_start is not None and fallback_end is not None:
        return int(fallback_start), int(fallback_end)
    return None


def _split_frame_range(start: int, end: int, sub_sampling: int, n_chunks: int):
    """
    Divide [start, end] (with step sub_sampling) into n_chunks contiguous
    ranges.  Returns a list of (chunk_start, chunk_end) tuples.
    Each chunk_end is the last frame that belongs to this chunk (inclusive).
    """
    all_frames = list(range(start, end + 1, sub_sampling))
    if not all_frames:
        return [(start, end)]

    chunk_size = math.ceil(len(all_frames) / n_chunks)
    chunks = []
    for i in range(0, len(all_frames), chunk_size):
        chunk_frames = all_frames[i : i + chunk_size]
        chunks.append((chunk_frames[0], chunk_frames[-1]))
    return chunks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Parallel Blender render launcher (Step 2)"
    )
    p.add_argument("--fbx_path",           type=str,   default="./mixamo_fbx/")
    p.add_argument("--out_dir",            type=str,   default="./output/")
    p.add_argument("--n_cam",              type=int,   default=4)
    p.add_argument("--cam_height",         type=float, default=3.0)
    p.add_argument("--cam_radius",         type=float, default=4.5)
    p.add_argument("--follow_bone",        type=str,   default="mixamorig:Hips")
    p.add_argument("--char_color",         type=str,   default=None)
    p.add_argument("--start_motion_frame", type=int,   default=0)
    p.add_argument("--sub_sampling",       type=int,   default=1)
    p.add_argument("--img_height",         type=int,   default=512)
    p.add_argument("--img_width",          type=int,   default=512)
    p.add_argument("--render_engine",      type=str,   default="cycles",
                   choices=["eevee", "cycles"])
    p.add_argument("--render_samples",     type=int,   default=16)
    p.add_argument("--use_gpu",            action="store_true")
    p.add_argument("--legacy_mode",        action="store_true")
    p.add_argument("--blender_bin",        type=str,   default=_DEFAULT_BLENDER,
                   help="Path to the Blender executable")
    p.add_argument("--workers",            type=int,   default=4,
                   help="Max number of concurrent Blender processes")
    p.add_argument("--frame_chunks",       type=int,   default=1,
                   help="Split each animation into this many frame chunks per camera "
                        "(1 = no frame splitting, just camera-level parallelism)")
    # Explicit frame range (used if skeleton JSONs are not yet produced)
    p.add_argument("--frame_start",        type=int,   default=None,
                   help="Override animation start frame (auto-read from JSON if omitted)")
    p.add_argument("--frame_end",          type=int,   default=None,
                   help="Override animation end frame (auto-read from JSON if omitted)")
    p.add_argument("--dry_run",            action="store_true",
                   help="Print all commands without executing them")
    args = p.parse_args()

    # ---- Discover FBX files ----
    if ".fbx" in args.fbx_path:
        fbx_files = [args.fbx_path]
    else:
        fbx_files = sorted(glob.glob(os.path.join(args.fbx_path, "*.fbx")))
    if not fbx_files:
        print(f"[launcher] ERROR: no .fbx files found in {args.fbx_path}", flush=True)
        sys.exit(1)
    print(f"[launcher] Found {len(fbx_files)} FBX file(s)", flush=True)

    # ---- Build job list ----
    jobs = []
    for fbx in fbx_files:
        anim_name = os.path.basename(fbx).split(".")[0]

        # --- Determine frame range for this animation ---
        frame_range = _discover_frame_range(
            args.out_dir, anim_name, args.frame_start, args.frame_end
        )
        if frame_range is None:
            print(
                f"[launcher] WARNING: cannot determine frame range for '{anim_name}'. "
                f"Run export_skeleton.py first, or pass --frame_start / --frame_end. "
                f"Skipping.",
                flush=True,
            )
            continue

        anim_start, anim_end = frame_range

        # Apply start_motion_frame floor
        if args.start_motion_frame > anim_start:
            anim_start = args.start_motion_frame

        # --- Compute frame chunks ---
        chunks = _split_frame_range(
            anim_start, anim_end, args.sub_sampling, args.frame_chunks
        )

        print(
            f"[launcher] {anim_name}: frames {anim_start}–{anim_end} "
            f"→ {len(chunks)} chunk(s) × {args.n_cam} cam(s) = "
            f"{len(chunks) * args.n_cam} job(s)",
            flush=True,
        )

        for cam_idx in range(args.n_cam):
            out_subdir    = os.path.join(args.out_dir, anim_name, f"cam_{cam_idx}")

            for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks):
                log_path = os.path.join(
                    out_subdir, f"render_log_chunk_{chunk_idx:03d}.txt"
                )
                label = f"{anim_name}/cam_{cam_idx}/chunk_{chunk_idx:03d}"

                cmd = build_blender_cmd(
                    blender_bin        = args.blender_bin,
                    fbx_path           = fbx,
                    out_dir            = out_subdir,
                    cam_idx            = cam_idx,
                    n_cam              = args.n_cam,
                    cam_height         = args.cam_height,
                    cam_radius         = args.cam_radius,
                    follow_bone        = args.follow_bone,
                    char_color         = args.char_color,
                    start_motion_frame = args.start_motion_frame,
                    sub_sampling       = args.sub_sampling,
                    img_width          = args.img_width,
                    img_height         = args.img_height,
                    render_engine      = args.render_engine,
                    render_samples     = args.render_samples,
                    use_gpu            = args.use_gpu,
                    legacy_mode        = args.legacy_mode,
                    frame_start        = chunk_start,
                    frame_end          = chunk_end,
                )

                jobs.append({
                    "cmd": cmd, "log_path": log_path, "label": label,
                    "out_dir": out_subdir,
                    "expected_frames": len(range(chunk_start, chunk_end + 1,
                                                 args.sub_sampling)),
                })

    total = len(jobs)
    print(f"[launcher] Total jobs: {total}  ·  workers: {args.workers}", flush=True)

    if args.dry_run:
        for job in jobs:
            print(" ".join(job["cmd"]))
        return

    if total == 0:
        print("[launcher] No jobs to run.", flush=True)
        return

    # ---- Build per-directory expected frame counts ----
    dir_expectations = defaultdict(int)
    for job in jobs:
        dir_expectations[job["out_dir"]] += job["expected_frames"]

    # ---- Run in parallel with progress tracking ----
    tracker = create_tracker(dict(dir_expectations), poll_interval=2.0)
    tracker.start()

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_job, job): job for job in jobs}
            for fut in as_completed(futures):
                result = fut.result()
                tracker.record_job_result(
                    result["label"], result["returncode"], result["log_path"],
                )
    finally:
        tracker.stop()

    # ---- Summary ----
    n_failed = tracker.print_summary()
    if n_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
