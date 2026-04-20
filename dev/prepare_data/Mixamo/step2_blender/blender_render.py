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
import argparse

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

def make_pre_render_handler(arm, resolved_bone, cam_obj, cam_offset, frame_list):
    def handler(scene, depsgraph):
        frame = scene.frame_current
        if frame not in frame_list:
            return
        pb = arm.pose.bones[resolved_bone]
        bone_pos = arm.matrix_world @ pb.head
        cam_obj.location = bone_pos + cam_offset
        _look_at(cam_obj, bone_pos)
    return handler

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
    enable_gi: bool,
    legacy_mode: bool,
):
    """Render every camera × every frame in the current Blender process."""
    configure_render_engine(
        scene,
        engine=render_engine,
        samples=render_samples,
        use_gpu=use_gpu,
        legacy_mode=legacy_mode,
        enable_gi=enable_gi,
    )
    resolved_bone = resolve_follow_bone(arm, follow_bone)
    render_frames = list(range(start_frame, end_frame + 1, sub_sampling))
    total_frames = len(render_frames)
    frame_set = set(render_frames)

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

        # --- track render progress for printing ---
        rendered_count = [0]

        def pre_render_handler(scn, depsgraph):
            if scn.frame_current not in frame_set:
                return
            bpy.context.view_layer.update()  # ensure armature is evaluated for this frame
            pb = arm.pose.bones[resolved_bone]
            bone_pos = arm.matrix_world @ pb.head
            cam_obj.location = bone_pos + cam_offset
            _look_at(cam_obj, bone_pos)

        def post_render_handler(scn, depsgraph):
            """Runs after each frame — print progress."""
            if scn.frame_current not in frame_set:
                return
            rendered_count[0] += 1
            print(
                f"[#] cam-{cam_idx} frame {scn.frame_current}/{end_frame} "
                f"({rendered_count[0]}/{total_frames})",
                flush=True,
            )

        # Register handlers
        bpy.app.handlers.render_pre.append(pre_render_handler)
        bpy.app.handlers.render_post.append(post_render_handler)

        try:
            # Set frame range and output — single render call keeps Cycles warm
            scene.frame_start = render_frames[0]
            scene.frame_end   = render_frames[-1]
            scene.frame_step  = sub_sampling
            scene.render.filepath = os.path.join(out_subdir, "frame####.png")

            bpy.ops.render.render(animation=True)

        finally:
            # Always clean up handlers even if render fails
            bpy.app.handlers.render_pre.remove(pre_render_handler)
            bpy.app.handlers.render_post.remove(post_render_handler)

            # Clean up camera before next view
            bpy.data.objects.remove(cam_obj, do_unlink=True)
            bpy.data.cameras.remove(cam_data, do_unlink=True)

"""
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
    enable_gi: bool,
    legacy_mode: bool,
):
    # Render every camera × every frame in the current Blender process.

    configure_render_engine(
        scene,
        engine=render_engine,
        samples=render_samples,
        use_gpu=use_gpu,
        legacy_mode=legacy_mode,
        enable_gi=enable_gi,
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
            # bpy.ops.render.render(write_still=True)
            bpy.ops.render.render(animation=True)

            print(
                f"[#] cam-{cam_idx} frame {frame}/{end_frame} "
                f"({ti + 1}/{total_frames})",
                flush=True,
            )

        # Clean up camera before next view
        bpy.data.objects.remove(cam_obj, do_unlink=True)
        bpy.data.cameras.remove(cam_data, do_unlink=True)
"""

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
        enable_gi=args.enable_gi,
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
    Sequential:   renders in-process, no extra Blender startups.

    Sequential example:
    blender --background --python render_pipeline.py -- \\
        --fbx_path ./fbx/ --out_dir ./out/ --n_cam 4

    """,
    )

    # -- Input / output --
    p.add_argument("--fbx_path",  nargs="+", type=str, default="./mixamo_fbx/",
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
    p.add_argument("--img_height",  type=int, default=720)
    p.add_argument("--img_width",   type=int, default=1280)

    # -- Render engine --
    p.add_argument("--enable_gi", action="store_true", default=False,
                   help="Enable global illumination (more expensive).")
    p.add_argument("--render_engine",  type=str, default="cycles",
                   choices=["eevee", "cycles"])
    p.add_argument("--render_samples", type=int, default=16)
    p.add_argument("--use_gpu",        action="store_true")
    p.add_argument("--gpu_id",      type=str, default="0",)
    p.add_argument("--legacy_mode",    action="store_true")

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
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # Discover FBX files if given a folder instead a list of files
    print("GG")
    print(args.fbx_path)
    print("GG")
    if os.path.isdir(args.fbx_path[0]):
        fbx_files = sorted(glob.glob(os.path.join(args.fbx_path, "*.fbx")))
    else:
        fbx_files = args.fbx_path

    if not fbx_files:
        print(f"[#] ERROR: no .fbx files found at {args.fbx_path}", flush=True)
        sys.exit(1)

    print(f"[#] Found {len(fbx_files)} FBX file(s)", flush=True)

    for fbx in fbx_files:
        process_fbx(args, fbx)

    print(f"\n[#] All done.", flush=True)
