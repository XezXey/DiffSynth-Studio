"""
Unified Mixamo FBX render pipeline (single entry point).

Run with Blender
-----------------
    blender --background --python render_pipeline.py -- \\
        --fbx_path     ./mixamo_fbx/ \\
        --out_dir      ./output/ \\
        --n_cam        4

What it does
------------
1. For each FBX file: load the scene, export ``skeleton_cam_<id>.json`` for
   every camera view (fast, in-process — no rendering).

2. Render all cameras × frames.  Two modes:

   **Sequential** (default):
       Everything happens in the current Blender process, frame by frame.
       No extra Blender startups.  Simple and works everywhere.

   **Parallel** (``--cam_workers`` / ``--frame_workers``):
       The script builds a job matrix of
       ``n_cam × ceil(frames / frame_chunks)`` and spawns child Blender
       processes (each running ``render_single_cam.py``) via a subprocess
       pool.  Use this when you have many CPU cores or multiple GPUs.

Examples
--------
Sequential (simple):
    blender --background --python render_pipeline.py -- \\
        --fbx_path ./fbx/ --out_dir ./out/ --n_cam 4

Parallel (8 workers, 2 frame chunks per camera):
    blender --background --python render_pipeline.py -- \\
        --fbx_path ./fbx/ --out_dir ./out/ --n_cam 4 \\
        --cam_workers 4 --frame_workers 2
"""

import sys
import os
import glob
import math
import subprocess
import argparse

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Make sibling packages importable
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import bpy
import mathutils

from render_utils.scene_utils import (
    clear_scene, load_fbx, create_default_camera,
    ensure_sun_light, setup_white_background,
)
from render_utils.render_engine import configure_render_engine
from render_utils.bone_utils import resolve_follow_bone
from render_utils.skeleton_export import (
    compute_static_rig_info,
    export_skeleton_for_camera,
    save_skeleton_json,
)
from render_utils.progress import create_tracker


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_RENDER_SCRIPT = os.path.join(_HERE, "render_single_cam.py")
_DEFAULT_BLENDER = os.environ.get("BLENDER_BIN", "blender")


def camera_offset_for_index(cam_idx: int, n_cam: int, radius: float,
                            height: float) -> mathutils.Vector:
    theta = 2.0 * math.pi * cam_idx / n_cam
    return mathutils.Vector((
        radius * math.sin(theta),
        -radius * math.cos(theta),
        height,
    ))


def _find_armature():
    """Return the first armature object in the scene, or raise."""
    arm = bpy.data.objects.get("Armature")
    if arm is None:
        for obj in bpy.data.objects:
            if obj.type == "ARMATURE":
                return obj
    if arm is None:
        raise RuntimeError("No armature found in the scene.")
    return arm


def _look_at(obj, target: mathutils.Vector):
    direction = target - obj.location
    rot = direction.to_track_quat("-Z", "Y")
    obj.rotation_euler = rot.to_euler()


# ---------------------------------------------------------------------------
# Step 1 – Export skeleton JSONs (always in-process, fast)
# ---------------------------------------------------------------------------

def export_skeletons(
    arm,
    scene,
    out_dir: str,
    anim_name: str,
    n_cam: int,
    radius: float,
    height: float,
    follow_bone: str,
    start_frame: int,
    end_frame: int,
    sub_sampling: int,
) -> list[str]:
    """
    Export skeleton_cam_<id>.json for each camera.
    Returns the list of JSON paths written.
    """
    resolved_bone = resolve_follow_bone(arm, follow_bone)
    rig_info = compute_static_rig_info(arm)
    json_paths = []

    for cam_idx in range(n_cam):
        cam_offset = camera_offset_for_index(cam_idx, n_cam, radius, height)
        out_subdir = os.path.join(out_dir, anim_name, f"cam-{cam_idx}")
        os.makedirs(out_subdir, exist_ok=True)
        json_path = os.path.join(out_subdir, f"skeleton_cam-{cam_idx}.json")

        print(f"[#] Exporting skeleton cam-{cam_idx}  offset={cam_offset}",
              flush=True)

        cam_data = bpy.data.cameras.new(name=f"ExportCam-{cam_idx}")
        cam_obj  = bpy.data.objects.new(f"ExportCam-{cam_idx}", cam_data)
        scene.collection.objects.link(cam_obj)
        scene.camera = cam_obj

        data = export_skeleton_for_camera(
            scene=scene,
            arm=arm,
            cam=cam_obj,
            cam_offset=cam_offset,
            follow_bone_name=resolved_bone,
            rig_info=rig_info,
            start_frame=start_frame,
            end_frame=end_frame,
            sub_sampling=sub_sampling,
        )
        save_skeleton_json(data, json_path)
        json_paths.append(json_path)

        bpy.data.objects.remove(cam_obj, do_unlink=True)
        bpy.data.cameras.remove(cam_data, do_unlink=True)

    return json_paths


# ---------------------------------------------------------------------------
# Step 2A – Sequential render (in-process, no child Blender)
# ---------------------------------------------------------------------------

def render_sequential(
    arm,
    scene,
    out_dir: str,
    anim_name: str,
    n_cam: int,
    radius: float,
    height: float,
    follow_bone: str,
    start_frame: int,
    end_frame: int,
    sub_sampling: int,
    render_engine: str,
    render_samples: int,
    use_gpu: bool,
    legacy_mode: bool,
):
    """Render every camera × every frame in the current Blender process."""

    configure_render_engine(
        scene,
        engine=render_engine,
        samples=render_samples,
        use_gpu=use_gpu,
        legacy_mode=legacy_mode,
    )

    resolved_bone = resolve_follow_bone(arm, follow_bone)
    render_frames = list(range(start_frame, end_frame + 1, sub_sampling))
    total_frames = len(render_frames)

    for cam_idx in range(n_cam):
        cam_offset = camera_offset_for_index(cam_idx, n_cam, radius, height)
        out_subdir = os.path.join(out_dir, anim_name, f"cam-{cam_idx}")
        os.makedirs(out_subdir, exist_ok=True)

        cam_data = bpy.data.cameras.new(name=f"Cam-{cam_idx}")
        cam_obj  = bpy.data.objects.new(f"Cam-{cam_idx}", cam_data)
        scene.collection.objects.link(cam_obj)
        scene.camera = cam_obj

        print(f"\n[#] Rendering cam-{cam_idx} ({total_frames} frames) → {out_subdir}",
              flush=True)

        for ti, frame in enumerate(render_frames):
            scene.frame_set(frame)

            pb = arm.pose.bones[resolved_bone]
            bone_pos = arm.matrix_world @ pb.head
            cam_obj.location = bone_pos + cam_offset
            bpy.context.view_layer.update()
            _look_at(cam_obj, bone_pos)
            bpy.context.view_layer.update()

            scene.render.filepath = os.path.join(out_subdir, f"frame{frame:04d}.png")
            bpy.ops.render.render(write_still=True)

            print(
                f"[#] cam-{cam_idx} frame {frame}/{end_frame} "
                f"({ti + 1}/{total_frames})",
                flush=True,
            )

        # Clean up camera before next view
        bpy.data.objects.remove(cam_obj, do_unlink=True)
        bpy.data.cameras.remove(cam_data, do_unlink=True)


# ---------------------------------------------------------------------------
# Step 2B – Parallel render (spawn child Blender processes)
# ---------------------------------------------------------------------------

def _split_frame_range(start: int, end: int, sub_sampling: int, n_chunks: int):
    """
    Divide [start, end] (step=sub_sampling) into n_chunks contiguous ranges.
    Returns list of (chunk_start, chunk_end) inclusive tuples.
    """
    all_frames = list(range(start, end + 1, sub_sampling))
    if not all_frames:
        return [(start, end)]
    chunk_size = math.ceil(len(all_frames) / n_chunks)
    chunks = []
    for i in range(0, len(all_frames), chunk_size):
        seg = all_frames[i : i + chunk_size]
        chunks.append((seg[0], seg[-1]))
    return chunks


def _build_blender_cmd(
    blender_bin, fbx_path, out_dir, cam_idx, n_cam,
    cam_height, cam_radius, follow_bone, char_color,
    start_motion_frame, sub_sampling,
    img_width, img_height,
    render_engine, render_samples,
    use_gpu, legacy_mode,
    frame_start=None, frame_end=None,
):
    cmd = [
        blender_bin, "--background",
        "--python", _RENDER_SCRIPT, "--",
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


def _run_job(job: dict) -> dict:
    cmd, log_path, label = job["cmd"], job["log_path"], job["label"]
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as fh:
        result = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True)
    return {"label": label, "returncode": result.returncode, "log_path": log_path}


def render_parallel(
    fbx_path: str,
    out_dir: str,
    anim_name: str,
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
    start_frame: int,
    end_frame: int,
    cam_workers: int,
    frame_workers: int,
    blender_bin: str,
    dry_run: bool = False,
):
    """
    Build the job matrix (n_cam × frame_chunks) and run them in parallel.
    Total concurrent workers = cam_workers × frame_workers.
    """
    frame_chunks = frame_workers  # one chunk per frame-worker
    chunks = _split_frame_range(start_frame, end_frame, sub_sampling, frame_chunks)
    total_workers = cam_workers * frame_workers

    print(
        f"[parallel] {anim_name}: {n_cam} cam(s) × {len(chunks)} chunk(s) "
        f"= {n_cam * len(chunks)} job(s)  ·  max {total_workers} workers",
        flush=True,
    )

    jobs = []
    for cam_idx in range(n_cam):
        out_subdir = os.path.join(out_dir, anim_name, f"cam-{cam_idx}")
        for ci, (cs, ce) in enumerate(chunks):
            label    = f"{anim_name}/cam-{cam_idx}/chunk-{ci:03d}"
            log_path = os.path.join(out_subdir, f"render-log_chunk-{ci:03d}.txt")
            cmd = _build_blender_cmd(
                blender_bin=blender_bin, fbx_path=fbx_path,
                out_dir=out_subdir, cam_idx=cam_idx, n_cam=n_cam,
                cam_height=cam_height, cam_radius=cam_radius,
                follow_bone=follow_bone, char_color=char_color,
                start_motion_frame=start_motion_frame,
                sub_sampling=sub_sampling,
                img_width=img_width, img_height=img_height,
                render_engine=render_engine, render_samples=render_samples,
                use_gpu=use_gpu, legacy_mode=legacy_mode,
                frame_start=cs, frame_end=ce,
            )
            jobs.append({
                "cmd": cmd, "log_path": log_path, "label": label,
                "out_dir": out_subdir,
                "expected_frames": len(range(cs, ce + 1, sub_sampling)),
            })

    if dry_run:
        for j in jobs:
            print(" ".join(j["cmd"]))
        return

    # Build per-directory expected frame counts
    dir_expectations = defaultdict(int)
    for job in jobs:
        dir_expectations[job["out_dir"]] += job["expected_frames"]

    # Run with progress tracking
    total = len(jobs)
    tracker = create_tracker(dict(dir_expectations), poll_interval=7.0)
    tracker.start()

    try:
        with ThreadPoolExecutor(max_workers=total_workers) as pool:
            futures = {pool.submit(_run_job, j): j for j in jobs}
            for fut in as_completed(futures):
                r = fut.result()
                tracker.record_job_result(
                    r["label"], r["returncode"], r["log_path"],
                )
    finally:
        tracker.stop()

    tracker.print_summary()


# ---------------------------------------------------------------------------
# Per-FBX driver
# ---------------------------------------------------------------------------

def process_fbx(args, fbx_path: str):
    """Load one FBX, export skeletons, then render (sequential or parallel)."""
    anim_name = os.path.basename(fbx_path).split(".")[0]
    print(f"\n{'='*60}", flush=True)
    print(f"[#] Processing: {anim_name}", flush=True)
    print(f"{'='*60}", flush=True)

    # ---- Load scene ----
    clear_scene()
    load_fbx(fbx_path, char_color=args.char_color)
    create_default_camera()
    ensure_sun_light()
    setup_white_background()

    scene = bpy.context.scene
    scene.render.image_settings.file_format = "PNG"
    scene.render.resolution_x = args.img_width
    scene.render.resolution_y = args.img_height
    scene.render.resolution_percentage = 100

    # ---- Armature & frame range ----
    arm = _find_armature()
    if arm.animation_data is None or arm.animation_data.action is None:
        raise RuntimeError(f"No animation action in {fbx_path}")

    action = arm.animation_data.action
    start_frame, end_frame = map(int, action.frame_range)
    if args.start_motion_frame > start_frame:
        start_frame = args.start_motion_frame

    print(f"[#] Action: {action.name}  frames {start_frame}–{end_frame}", flush=True)

    # ---- Step 1: export skeleton JSONs (always, fast) ----
    export_skeletons(
        arm=arm, scene=scene,
        out_dir=args.out_dir, anim_name=anim_name,
        n_cam=args.n_cam, radius=args.cam_radius, height=args.cam_height,
        follow_bone=args.follow_bone,
        start_frame=start_frame, end_frame=end_frame,
        sub_sampling=args.sub_sampling,
    )

    # ---- Step 2: render ----
    is_parallel = (args.cam_workers is not None and args.cam_workers >= 1) or \
                  (args.frame_workers is not None and args.frame_workers >= 1)

    if is_parallel:
        cam_workers   = args.cam_workers   or 1
        frame_workers = args.frame_workers or 1
        render_parallel(
            fbx_path=fbx_path,
            out_dir=args.out_dir, anim_name=anim_name,
            n_cam=args.n_cam,
            cam_height=args.cam_height, cam_radius=args.cam_radius,
            follow_bone=args.follow_bone, char_color=args.char_color,
            start_motion_frame=args.start_motion_frame,
            sub_sampling=args.sub_sampling,
            img_width=args.img_width, img_height=args.img_height,
            render_engine=args.render_engine,
            render_samples=args.render_samples,
            use_gpu=args.use_gpu, legacy_mode=args.legacy_mode,
            start_frame=start_frame, end_frame=end_frame,
            cam_workers=cam_workers, frame_workers=frame_workers,
            blender_bin=args.blender_bin,
            dry_run=args.dry_run,
        )
    else:
        render_sequential(
            arm=arm, scene=scene,
            out_dir=args.out_dir, anim_name=anim_name,
            n_cam=args.n_cam,
            radius=args.cam_radius, height=args.cam_height,
            follow_bone=args.follow_bone,
            start_frame=start_frame, end_frame=end_frame,
            sub_sampling=args.sub_sampling,
            render_engine=args.render_engine,
            render_samples=args.render_samples,
            use_gpu=args.use_gpu, legacy_mode=args.legacy_mode,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []

    p = argparse.ArgumentParser(
        description="Unified Mixamo FBX → skeleton JSON + render pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  Sequential (default):   renders in-process, no extra Blender startups.
  Parallel:               pass --cam_workers and/or --frame_workers to
                          spawn child Blender processes.

Sequential example:
  blender --background --python render_pipeline.py -- \\
      --fbx_path ./fbx/ --out_dir ./out/ --n_cam 4

Parallel example (4 camera workers × 2 frame workers = 8 concurrent):
  blender --background --python render_pipeline.py -- \\
      --fbx_path ./fbx/ --out_dir ./out/ --n_cam 4 \\
      --cam_workers 4 --frame_workers 2
""",
    )

    # -- Input / output --
    p.add_argument("--fbx_path",  type=str, default="./mixamo_fbx/",
                   help="Single .fbx file or directory of .fbx files")
    p.add_argument("--out_dir",   type=str, default="./output/")

    # -- Camera geometry --
    p.add_argument("--n_cam",       type=int,   default=4)
    p.add_argument("--cam_height",  type=float, default=3.0)
    p.add_argument("--cam_radius",  type=float, default=4.5)
    p.add_argument("--follow_bone", type=str,   default="mixamorig:Hips")

    # -- Animation range --
    p.add_argument("--start_motion_frame", type=int, default=0)
    p.add_argument("--sub_sampling",       type=int, default=1)

    # -- Appearance --
    p.add_argument("--char_color",  type=str, default=None)
    p.add_argument("--img_height",  type=int, default=512)
    p.add_argument("--img_width",   type=int, default=512)

    # -- Render engine --
    p.add_argument("--render_engine",  type=str, default="cycles",
                   choices=["eevee", "cycles"])
    p.add_argument("--render_samples", type=int, default=16)
    p.add_argument("--use_gpu",        action="store_true")
    p.add_argument("--legacy_mode",    action="store_true")

    # -- Parallel mode (omit these for sequential) --
    p.add_argument("--cam_workers",   type=int, default=None,
                   help="Number of concurrent camera workers.  "
                        "Omit for sequential mode.")
    p.add_argument("--frame_workers", type=int, default=None,
                   help="Split each camera's frame range into this many "
                        "chunks (one child Blender per chunk).  "
                        "Omit for sequential mode.")
    p.add_argument("--blender_bin",   type=str, default=_DEFAULT_BLENDER,
                   help="Path to Blender binary (for parallel child processes)")
    p.add_argument("--dry_run",       action="store_true",
                   help="Print parallel commands without executing")

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    # Discover FBX files
    if ".fbx" in args.fbx_path:
        fbx_files = [args.fbx_path]
    else:
        fbx_files = sorted(glob.glob(os.path.join(args.fbx_path, "*.fbx")))

    if not fbx_files:
        print(f"[#] ERROR: no .fbx files found at {args.fbx_path}", flush=True)
        sys.exit(1)

    is_parallel = (args.cam_workers is not None) or (args.frame_workers is not None)
    mode = "parallel" if is_parallel else "sequential"
    print(f"[#] Found {len(fbx_files)} FBX file(s)  ·  mode: {mode}", flush=True)

    for fbx in fbx_files:
        process_fbx(args, fbx)

    print(f"\n[#] All done.", flush=True)
