"""
Skeleton and camera data export helpers.

This module is intentionally independent from the rendering step so that
skeleton JSON files can be produced quickly (no render) and then used by the
parallel render step.

Typical usage
-------------
::

    from lib.skeleton_export import (
        compute_static_rig_info,
        collect_camera_intrinsics,
        export_skeleton_for_camera,
        save_skeleton_json,
    )

    rig_info   = compute_static_rig_info(arm)
    intrinsics = collect_camera_intrinsics(cam, scene)
    data       = export_skeleton_for_camera(scene, arm, cam, cam_offset,
                                            follow_bone_name, rig_info,
                                            start_frame, end_frame, sub_sampling)
    data.update(intrinsics)
    save_skeleton_json(data, "/path/to/output.json")
"""

import math
import json
import os

import bpy
import mathutils

from .bone_utils import resolve_follow_bone


# ---------------------------------------------------------------------------
# Static rig information (computed once per armature)
# ---------------------------------------------------------------------------

def compute_static_rig_info(arm) -> dict:
    """
    Return topology information that does not change across frames.

    Returns
    -------
    dict with keys:
        joint_names       : list[str]
        bones             : list[[parent_name, child_name]]
        rest_offsets      : dict[str, [x, y, z]]
        kinematic_chains  : dict[str, list[str]]
    """
    pose_bones = list(arm.pose.bones)
    joint_names = [pb.name for pb in pose_bones]

    bones = [
        [pb.parent.name, pb.name]
        for pb in pose_bones
        if pb.parent
    ]

    rest_offsets = {}
    for b in arm.data.bones:
        if b.parent:
            offset = b.head_local - b.parent.head_local
        else:
            offset = b.head_local
        rest_offsets[b.name] = [float(offset.x), float(offset.y), float(offset.z)]

    kinematic_chains = {}
    for pb in pose_bones:
        chain, cur = [], pb
        while cur is not None:
            chain.append(cur.name)
            cur = cur.parent
        kinematic_chains[pb.name] = list(reversed(chain))

    return {
        "joint_names": joint_names,
        "bones": bones,
        "rest_offsets": rest_offsets,
        "kinematic_chains": kinematic_chains,
    }


# ---------------------------------------------------------------------------
# Camera intrinsics
# ---------------------------------------------------------------------------

def collect_camera_intrinsics(cam, scene) -> dict:
    """
    Compute pinhole camera intrinsics from Blender camera and scene settings.

    Returns
    -------
    dict with keys:
        cams_intr  : [fx, fy, cx, cy]
        resolution : [width, height]
    """
    res_x = scene.render.resolution_x
    res_y = scene.render.resolution_y
    sensor_w = cam.data.sensor_width
    focal = cam.data.lens

    fx = res_x * focal / sensor_w
    fy = fx
    cx = res_x / 2.0
    cy = res_y / 2.0

    return {
        "cams_intr": [float(fx), float(fy), float(cx), float(cy)],
        "resolution": [int(res_x), int(res_y)],
    }


# ---------------------------------------------------------------------------
# Per-frame helpers
# ---------------------------------------------------------------------------

def _get_bone_world_pos(arm, bone_name):
    pb = arm.pose.bones[bone_name]
    return arm.matrix_world @ pb.head


def _look_at(obj, target):
    direction = target - obj.location
    rot = direction.to_track_quat("-Z", "Y")
    obj.rotation_euler = rot.to_euler()
    return obj


def _collect_frame_skeleton(arm):
    """
    Return (joints_3d, quats, euler_deg) lists for the current frame.
    """
    frame_joints, frame_quats, frame_eulers = [], [], []

    for pb in arm.pose.bones:
        pos = arm.matrix_world @ pb.head
        frame_joints.append([float(pos.x), float(pos.y), float(pos.z)])

        local_mat = pb.matrix.copy()
        if pb.parent is not None:
            local_mat = pb.parent.matrix.inverted() @ pb.matrix

        q = local_mat.to_quaternion().normalized()
        frame_quats.append([float(q.w), float(q.x), float(q.y), float(q.z)])

        e = q.to_euler("XYZ")
        frame_eulers.append([
            float(math.degrees(e.x)),
            float(math.degrees(e.y)),
            float(math.degrees(e.z)),
        ])

    return frame_joints, frame_quats, frame_eulers


# ---------------------------------------------------------------------------
# Main export routine
# ---------------------------------------------------------------------------

def export_skeleton_for_camera(
    scene,
    arm,
    cam,
    cam_offset: mathutils.Vector,
    follow_bone_name: str,
    rig_info: dict,
    start_frame: int,
    end_frame: int,
    sub_sampling: int = 1,
) -> dict:
    """
    Iterate over the frame range, move the camera to follow *follow_bone_name*,
    and collect per-frame skeleton + camera extrinsic data.

    No rendering is performed.

    Parameters
    ----------
    scene, arm, cam:
        Blender objects.
    cam_offset:
        World-space offset added to the followed bone's position each frame.
    follow_bone_name:
        Bone to track (will be resolved via ``resolve_follow_bone``).
    rig_info:
        Output of ``compute_static_rig_info``.
    start_frame, end_frame:
        Inclusive frame range.
    sub_sampling:
        Step size (1 = every frame, 2 = every other, etc.).

    Returns
    -------
    dict suitable for passing to ``save_skeleton_json``.
    """
    follow_bone_name = resolve_follow_bone(arm, follow_bone_name)

    intrinsics = collect_camera_intrinsics(cam, scene)
    fps = scene.render.fps / scene.render.fps_base

    joints_3d: list           = []
    joint_rot_quat: list      = []
    joint_rot_euler_deg: list = []
    cams_extr: list           = []
    cam_positions: list       = []   # world-space camera locations per frame

    render_frames = list(range(start_frame, end_frame + 1, sub_sampling))
    total = len(render_frames)

    for ti, frame in enumerate(render_frames):
        scene.frame_set(frame)

        # Position camera
        bone_pos = _get_bone_world_pos(arm, follow_bone_name)
        cam.location = bone_pos + cam_offset
        bpy.context.view_layer.update()
        _look_at(cam, bone_pos)
        bpy.context.view_layer.update()

        # Skeleton
        fj, fq, fe = _collect_frame_skeleton(arm)
        joints_3d.append(fj)
        joint_rot_quat.append(fq)
        joint_rot_euler_deg.append(fe)

        # Camera extrinsic (world → camera)
        M = cam.matrix_world.inverted()
        cams_extr.append([[float(M[r][c]) for c in range(4)] for r in range(4)])

        # Also store world-space camera position (useful for the render step to
        # skip re-simulating the follow logic)
        cam_positions.append([
            float(cam.location.x),
            float(cam.location.y),
            float(cam.location.z),
        ])

        # Store camera rotation as euler
        cam_rotation = [
            float(cam.rotation_euler.x),
            float(cam.rotation_euler.y),
            float(cam.rotation_euler.z),
        ]

        if ti % max(1, total // 10) == 0:
            print(f"[#]   Exported frame {frame}/{end_frame} ({ti+1}/{total})",
                  flush=True)

    data = {
        # -- static rig --
        **rig_info,
        # -- per-frame motion --
        "joints_3d": joints_3d,
        "joint_rot_quat": joint_rot_quat,
        "joint_rot_euler_deg": joint_rot_euler_deg,
        # -- camera --
        **intrinsics,
        "cams_extr": cams_extr,
        "cam_positions": cam_positions,         # world-space, used by render step
        # -- meta --
        "fps": float(fps),
        "frame_range": [int(start_frame), int(end_frame)],
        "sub_sampling": int(sub_sampling),
        "follow_bone": follow_bone_name,
        "cam_offset": [float(cam_offset.x), float(cam_offset.y), float(cam_offset.z)],
        "camera_name": cam.name,
    }

    return data


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_skeleton_json(data: dict, json_path: str):
    """Write *data* to *json_path*, creating parent directories as needed."""
    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
    with open(json_path, "w") as fh:
        json.dump(data, fh, indent=2)
    print(f"[#] Saved skeleton JSON → {json_path}", flush=True)
