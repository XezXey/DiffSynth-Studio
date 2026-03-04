"""
Step 1 – Export skeleton & camera JSON files (no rendering).

Run this with Blender's Python interpreter:

    blender --background --python export_skeleton.py -- \
        --fbx_path ./mixamo_fbx/ \
        --out_dir  ./output/ \
        --n_cam 4

This script iterates over every FBX and every camera angle, collects
per-frame skeleton + camera-extrinsic data, and writes one JSON per camera
into the output directory.

The output structure will be:

    <out_dir>/
      <animation_name>/
        cam_0/skeleton_cam_0.json
        cam_1/skeleton_cam_1.json
        ...

Once these JSON files exist, run render_single_cam.py (or the parallel
launcher launch_parallel_render.py) to do the actual pixel rendering.
"""

import sys
import os
import glob
import math
import argparse

# ---------------------------------------------------------------------------
# Make sure sibling packages (lib/) are importable when this script is
# executed directly by Blender.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import bpy
import mathutils

from lib.scene_utils import clear_scene, load_fbx, create_default_camera, ensure_sun_light, setup_white_background
from lib.bone_utils import resolve_follow_bone
from lib.skeleton_export import (
    compute_static_rig_info,
    export_skeleton_for_camera,
    save_skeleton_json,
)


# ---------------------------------------------------------------------------
# Camera placement geometry
# ---------------------------------------------------------------------------

def camera_offset_for_index(cam_idx: int, n_cam: int, radius: float, height: float) -> mathutils.Vector:
    """
    Return the world-space offset from the followed bone for camera *cam_idx*.
    Cameras are evenly spaced around a circle; theta=0 is the "front" (−Y).
    """
    theta = 2.0 * math.pi * cam_idx / n_cam
    return mathutils.Vector((
        radius * math.sin(theta),
        -radius * math.cos(theta),
        height,
    ))


# ---------------------------------------------------------------------------
# Per-FBX export routine
# ---------------------------------------------------------------------------

def export_fbx(
    fbx_path: str,
    out_dir: str,
    n_cam: int,
    radius: float,
    height: float,
    follow_bone: str,
    char_color,
    start_motion_frame: int,
    sub_sampling: int,
    img_width: int,
    img_height: int,
):
    anim_name = os.path.basename(fbx_path).split(".")[0]
    print(f"\n[#] === Exporting skeleton: {anim_name} ===", flush=True)

    clear_scene()
    load_fbx(fbx_path, char_color=char_color)
    create_default_camera()
    ensure_sun_light()
    setup_white_background()

    scene = bpy.context.scene
    scene.render.resolution_x = img_width
    scene.render.resolution_y = img_height

    # Locate armature
    arm = bpy.data.objects.get("Armature")
    if arm is None:
        # Fallback: first ARMATURE type object
        for obj in bpy.data.objects:
            if obj.type == "ARMATURE":
                arm = obj
                break
    if arm is None:
        raise RuntimeError(f"No armature found in {fbx_path}")

    if arm.animation_data is None or arm.animation_data.action is None:
        raise RuntimeError(f"Armature has no animation in {fbx_path}")

    action = arm.animation_data.action
    start_frame, end_frame = map(int, action.frame_range)
    if start_motion_frame > start_frame:
        start_frame = start_motion_frame

    print(f"[#] Action: {action.name}  frames {start_frame}–{end_frame}", flush=True)

    # Resolve the follow bone once (same across all cameras)
    resolved_bone = resolve_follow_bone(arm, follow_bone)

    # Static rig info is identical for every camera
    rig_info = compute_static_rig_info(arm)

    for cam_idx in range(n_cam):
        cam_offset = camera_offset_for_index(cam_idx, n_cam, radius, height)
        out_subdir = os.path.join(out_dir, anim_name, f"cam_{cam_idx}")
        os.makedirs(out_subdir, exist_ok=True)
        json_path = os.path.join(out_subdir, f"skeleton_cam_{cam_idx}.json")

        print(f"[#] Camera {cam_idx}/{n_cam}  offset={cam_offset}", flush=True)

        # Create a throw-away camera for this view
        cam_data = bpy.data.cameras.new(name=f"ExportCam_{cam_idx}")
        cam_obj  = bpy.data.objects.new(f"ExportCam_{cam_idx}", cam_data)
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

        # Clean up temporary camera
        bpy.data.objects.remove(cam_obj, do_unlink=True)
        bpy.data.cameras.remove(cam_data, do_unlink=True)

    print(f"[#] Done: {anim_name}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser(description="Export skeleton JSON (Step 1, no rendering)")
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
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()

    if ".fbx" in args.fbx_path:
        fbx_files = [args.fbx_path]
    else:
        fbx_files = glob.glob(os.path.join(args.fbx_path, "*.fbx"))

    print(f"[#] Found {len(fbx_files)} FBX file(s)", flush=True)

    for fbx in fbx_files:
        export_fbx(
            fbx_path           = fbx,
            out_dir            = args.out_dir,
            n_cam              = args.n_cam,
            radius             = args.cam_radius,
            height             = args.cam_height,
            follow_bone        = args.follow_bone,
            char_color         = args.char_color,
            start_motion_frame = args.start_motion_frame,
            sub_sampling       = args.sub_sampling,
            img_width          = args.img_width,
            img_height         = args.img_height,
        )

    print("\n[#] All exports complete.", flush=True)
