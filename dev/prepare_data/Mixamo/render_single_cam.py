"""
Step 2 – Render a single camera view (optionally a sub-range of frames).

This script is designed to be called once per (camera, frame-chunk) job.
The parallel launcher (launch_parallel_render.py) spawns multiple instances
concurrently across cameras AND frame chunks.

Usage (single call)
-------------------
    blender --background --python render_single_cam.py -- \\
        --fbx_path      ./mixamo_fbx/Walking.fbx \\
        --out_dir       ./output/Walking/cam_0/ \\
        --cam_idx       0 \\
        --n_cam         4 \\
        --cam_height    3.0 \\
        --cam_radius    4.5 \\
        [--frame_start  0]   \\
        [--frame_end    49]  \\
        [--render_engine cycles] \\
        [--render_samples 16]    \\
        [--use_gpu]

Frame chunking
--------------
``--frame_start`` / ``--frame_end`` let the launcher assign each worker a
contiguous sub-range of the animation.  Output files are named by the actual
Blender frame number (``frame0042.png``) so chunks from different workers
never overwrite each other and can be collected into a single directory.

Camera placement
----------------
Camera position is derived each frame directly from the armature bone, which
is free since Blender evaluates the animation anyway during rendering.
"""

import sys
import os
import math
import argparse

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



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _look_at(obj, target: mathutils.Vector):
    direction = target - obj.location
    rot = direction.to_track_quat("-Z", "Y")
    obj.rotation_euler = rot.to_euler()


def camera_offset_for_index(cam_idx: int, n_cam: int, radius: float, height: float) -> mathutils.Vector:
    theta = 2.0 * math.pi * cam_idx / n_cam
    return mathutils.Vector((
        radius * math.sin(theta),
        -radius * math.cos(theta),
        height,
    ))


# ---------------------------------------------------------------------------
# Main render routine
# ---------------------------------------------------------------------------

def render_single_cam(
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
    # Optional chunk overrides – if None the full action range is used
    frame_start_override: int = None,
    frame_end_override: int = None,
):
    os.makedirs(out_dir, exist_ok=True)

    # ---- Load scene ----
    clear_scene()
    load_fbx(fbx_path, char_color=char_color)
    create_default_camera()
    ensure_sun_light()
    setup_white_background()

    scene = bpy.context.scene
    scene.render.image_settings.file_format = "PNG"
    scene.render.resolution_x = img_width
    scene.render.resolution_y = img_height
    scene.render.resolution_percentage = 100

    configure_render_engine(
        scene,
        engine=render_engine,
        samples=render_samples,
        use_gpu=use_gpu,
        legacy_mode=legacy_mode,
    )

    # ---- Locate armature ----
    arm = bpy.data.objects.get("Armature")
    if arm is None:
        for obj in bpy.data.objects:
            if obj.type == "ARMATURE":
                arm = obj
                break
    if arm is None:
        raise RuntimeError("No armature found.")
    if arm.animation_data is None or arm.animation_data.action is None:
        raise RuntimeError("Armature has no animation action.")

    action = arm.animation_data.action
    action_start, action_end = map(int, action.frame_range)

    # Apply start_motion_frame floor
    if start_motion_frame > action_start:
        action_start = start_motion_frame

    # Apply chunk overrides (launcher sets these for frame-parallel jobs)
    start_frame = frame_start_override if frame_start_override is not None else action_start
    end_frame   = frame_end_override   if frame_end_override   is not None else action_end

    # Safety clamp to actual action bounds
    start_frame = max(start_frame, action_start)
    end_frame   = min(end_frame,   action_end)

    print(
        f"[#] render_single_cam · fbx={os.path.basename(fbx_path)}"
        f" · cam_idx={cam_idx} · frames {start_frame}–{end_frame}"
        f" (action {action_start}–{action_end})",
        flush=True,
    )

    # ---- Camera ----
    cam_data = bpy.data.cameras.new(name=f"Cam-{cam_idx}")
    cam_obj  = bpy.data.objects.new(f"Cam-{cam_idx}", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    cam_offset = camera_offset_for_index(cam_idx, n_cam, cam_radius, cam_height)
    resolved_bone = resolve_follow_bone(arm, follow_bone)

    # ---- Frame loop ----
    render_frames = list(range(start_frame, end_frame + 1, sub_sampling))
    total = len(render_frames)

    for ti, frame in enumerate(render_frames):
        scene.frame_set(frame)

        # Derive camera position directly from the bone.
        # Blender evaluates the animation for rendering anyway, so this is free.
        pb = arm.pose.bones[resolved_bone]
        bone_pos = arm.matrix_world @ pb.head
        cam_obj.location = bone_pos + cam_offset
        bpy.context.view_layer.update()
        _look_at(cam_obj, bone_pos)

        bpy.context.view_layer.update()

        # Output named by the *actual* Blender frame number so that parallel
        # chunks all write into the same directory without collisions.
        scene.render.filepath = os.path.join(out_dir, f"frame{frame:04d}.png")
        bpy.ops.render.render(write_still=True)

        print(
            f"[#] Rendered frame {frame}/{end_frame} ({ti+1}/{total})",
            flush=True,
        )

    print(f"[#] Camera {cam_idx} render complete → {out_dir}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser(description="Render a single camera view (Step 2)")
    p.add_argument("--fbx_path",           type=str,   required=True)
    p.add_argument("--out_dir",            type=str,   required=True)
    p.add_argument("--cam_idx",            type=int,   required=True)
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
    # Frame-chunk overrides (set by the parallel launcher)
    p.add_argument("--frame_start",        type=int,   default=None,
                   help="First frame of this chunk (absolute Blender frame number)")
    p.add_argument("--frame_end",          type=int,   default=None,
                   help="Last frame of this chunk (inclusive, absolute Blender frame number)")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    render_single_cam(
        fbx_path             = args.fbx_path,
        out_dir              = args.out_dir,
        cam_idx              = args.cam_idx,
        n_cam                = args.n_cam,
        cam_height           = args.cam_height,
        cam_radius           = args.cam_radius,
        follow_bone          = args.follow_bone,
        char_color           = args.char_color,
        start_motion_frame   = args.start_motion_frame,
        sub_sampling         = args.sub_sampling,
        img_width            = args.img_width,
        img_height           = args.img_height,
        render_engine        = args.render_engine,
        render_samples       = args.render_samples,
        use_gpu              = args.use_gpu,
        legacy_mode          = args.legacy_mode,
        frame_start_override = args.frame_start,
        frame_end_override   = args.frame_end,
    )
