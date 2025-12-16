import bpy
import glob
import warnings
import os
import numpy as np
from pathlib import Path
import sys
import mathutils
import math
import json
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='bpy')

def load_fbx(f, char_color=None):
    bpy.ops.import_scene.fbx(filepath=f)
    print(f'[#] Loaded FBX file: {f}')
    if char_color is not None:
        print(f"[#] Setting character color to {char_color}")
        # Create flat-color material
        mat = bpy.data.materials.new(name="FlatColor")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        color = (0., 0., 0., 1.0)  # RGBA
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = 1.0

        # Assign to all mesh objects
        for obj in bpy.context.selected_objects:
            if obj.type == 'MESH':
                obj.data.materials.clear()
                obj.data.materials.append(mat)


def create_camera():
    bpy.ops.object.camera_add(location=(0, -3, 1.5), rotation=(1.5708, 0, 0))
    cam = bpy.context.object
    bpy.context.scene.camera = cam
    print("[#] Camera created and set as active.")

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    print("[#] Scene cleared.")

def ensure_sun_light():
    # Check if a Sun light exists
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT' and obj.data.type == 'SUN':
            return obj

    # Otherwise create one
    light_data = bpy.data.lights.new(name="Sun", type='SUN')
    light_data.energy = 3.0     # increase brightness if needed

    light_obj = bpy.data.objects.new(name="Sun", object_data=light_data)
    bpy.context.scene.collection.objects.link(light_obj)

    # Position the sun
    light_obj.location = (10, -10, 10)
    light_obj.rotation_euler = (0.7, 0.0, 0.8)

    return light_obj

def setup_background():
    world = bpy.context.scene.world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (1, 1, 1, 1)
    # bg.inputs[0].default_value = (0, 0, 0, 1)
    bg.inputs[1].default_value = 1.0  # brightness

def render_multiview(
        base_outpath,
        armature_name="Armature",
        follow_bone_name="mixamorig:Hips",
        resolution=(512, 512),
        num_cams=4,
        radius=3.0,
        height=1.0,
        start_motion_frame=0,
        sub_sampling=1,
):
    """
    Create num_cams cameras around the character (on a circle) and
    render one sequence per camera into separate subfolders.
    """
    scene = bpy.context.scene
    os.makedirs(base_outpath, exist_ok=True)

    # Angles around the circle (yaw around Z)
    # e.g. 4 cams: 0°, 90°, 180°, 270°
    for cam_idx in range(num_cams):
        theta = 2.0 * math.pi * cam_idx / num_cams

        # In Blender: X right, Y forward, Z up.
        # Character usually faces +Y, so "front" is at negative Y (like your original offset).
        # We'll interpret theta=0 as "front" (negative Y), then rotate around.
        offset_x = radius * math.sin(theta)
        offset_y = -radius * math.cos(theta)   # negative for "front" at theta=0
        offset_z = height

        cam_offset = mathutils.Vector((offset_x, offset_y, offset_z))

        # Create a camera for this view
        cam_data = bpy.data.cameras.new(name=f"Cam_{cam_idx}")
        cam_obj = bpy.data.objects.new(f"Cam_{cam_idx}", cam_data)
        scene.collection.objects.link(cam_obj)

        # Subfolder per camera
        outpath = os.path.join(base_outpath, f"cam_{cam_idx}")
        os.makedirs(outpath, exist_ok=True)

        print(f"[#] Rendering camera {cam_idx} with offset {cam_offset} to {outpath}",
              flush=True)

        render_animation(
            outpath=f"{outpath}",
            json_path=f"{outpath}/skeleton_cam_{cam_idx}.json",
            armature_name=armature_name,
            follow_bone_name=follow_bone_name,
            resolution=resolution,
            camera=cam_obj,
            cam_offset=cam_offset,
            start_motion_frame=start_motion_frame,
            sub_sampling=sub_sampling,
        )

def render_animation(
        outpath,
        json_path=None,                       # where to save skeleton/camera data
        armature_name="Armature",
        follow_bone_name="Hips",
        resolution=(512, 512),
        camera=None,
        cam_offset=mathutils.Vector((0.0, -3.0, 1.0)),
        start_motion_frame=0,
        sub_sampling=1,
        char_color=None,
):
    os.makedirs(outpath, exist_ok=True)
    scene = bpy.context.scene

    # --- Armature & animation ---
    arm = bpy.data.objects[armature_name]
    if arm.animation_data is None or arm.animation_data.action is None:
        raise RuntimeError("Armature has no animation_data.action")

    action = arm.animation_data.action
    start_frame, end_frame = map(int, action.frame_range)
    if start_motion_frame > start_frame:
        start_frame = start_motion_frame
    total = end_frame - start_frame + 1

    print(f"[#] Action: {action.name}, frames {start_frame}-{end_frame}", flush=True)

    # --- Render config ---
    scene.render.image_settings.file_format = 'PNG'
    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]
    scene.render.resolution_percentage = 100

    # --- Camera setup ---
    if camera is None:
        if scene.camera is None:
            cam_data = bpy.data.cameras.new("Camera")
            cam = bpy.data.objects.new("Camera", cam_data)
            scene.collection.objects.link(cam)
            cam.location = (0.0, -3.0, 1.0)
        else:
            cam = scene.camera
    else:
        cam = camera

    # make sure Blender renders from THIS camera
    scene.camera = cam
    # print(f"[#] Using camera: {cam.name}", flush=True)
    # print(f"[#] Camera initial location: {cam.location}", flush=True)
    # print(f"[#] Camera matrix_world:\n{cam.matrix_world}", flush=True)
    # print(f"[#] Camera matrix_world:\n{cam.matrix_world.inverted()}", flush=True)

    # --- Small helpers ---
    def look_at(obj, target):
        direction = target - obj.location
        rot = direction.to_track_quat('-Z', 'Y')  # camera looks along -Z, Y up
        obj.rotation_euler = rot.to_euler()
        return obj

    def get_bone_world_pos(obj, bone_name):
        pb = obj.pose.bones[bone_name]
        return obj.matrix_world @ pb.head

    # --- Export buffers (BVH-like info) ---
    export = json_path is not None
    if export:
        # Pose bones (animated)
        pose_bones = list(arm.pose.bones)
        joint_names = [pb.name for pb in pose_bones]

        # Topology: parent-child pairs (by name)
        bones = []
        for pb in pose_bones:
            if pb.parent:
                bones.append([pb.parent.name, pb.name])

        # Rest-pose offsets (BVH-like OFFSET)
        # Use armature.data.bones = rest pose
        rest_offsets = {}
        data_bones = arm.data.bones

        for b in data_bones:
            if b.parent:
                offset = b.head_local - b.parent.head_local
            else:
                offset = b.head_local  # root offset from origin
            rest_offsets[b.name] = [float(offset.x),
                                    float(offset.y),
                                    float(offset.z)]

        # Kinematic chains: root -> ... -> joint
        kinematic_chains = {}
        for pb in pose_bones:
            chain = []
            cur = pb
            while cur is not None:
                chain.append(cur.name)
                cur = cur.parent
            chain.reverse()
            kinematic_chains[pb.name] = chain

        # Per-frame arrays (pure Python lists)
        joints_3d = []           # [T][J][3], world-space
        joint_rot_quat = []      # [T][J][4], local parent-relative, (w,x,y,z)
        joint_rot_euler_deg = [] # [T][J][3], local XYZ in degrees (BVH-like)

        # Camera intrinsics for this camera
        res_x = scene.render.resolution_x
        res_y = scene.render.resolution_y
        sensor_w = cam.data.sensor_width
        focal = cam.data.lens

        fx = res_x * focal / sensor_w
        fy = fx
        cx = res_x / 2.0
        cy = res_y / 2.0
        cams_intr = [float(fx), float(fy), float(cx), float(cy)]

        # Per-frame camera extrinsics world->cam
        cams_extr = []  # [T][4][4]

    # --- Progress bar (optional) ---
    wm = bpy.context.window_manager if bpy.context.window_manager else None
    if wm:
        wm.progress_begin(0, total)

    # --- Main loop over frames ---
    render_frames = list(range(start_frame, end_frame + 1, sub_sampling))
    for ti, frame in enumerate(render_frames):
        if wm:
            wm.progress_update(ti)

        scene.frame_set(frame)

        # Camera follow
        # print(arm.pose.bones.keys(), flush=True)
        # exit()
        bone_pos = get_bone_world_pos(arm, follow_bone_name)
        cam.location = bone_pos + cam_offset
        bpy.context.view_layer.update()
        cam = look_at(cam, bone_pos)
        bpy.context.view_layer.update()

        # ----- Collect skeleton & camera info -----
        if export:
            frame_joints = []
            frame_quats = []
            frame_eulers_deg = []

            # joint positions & local rotations
            for pb in pose_bones:
                # world-space joint position
                pos = arm.matrix_world @ pb.head
                frame_joints.append([float(pos.x), float(pos.y), float(pos.z)])

                # local rotation relative to parent (BVH-style)
                # local_mat = parent^-1 * world_mat
                local_mat = pb.matrix.copy()
                if pb.parent is not None:
                    local_mat = pb.parent.matrix.inverted() @ pb.matrix

                q = local_mat.to_quaternion().normalized()
                frame_quats.append([float(q.w), float(q.x),
                                    float(q.y), float(q.z)])

                # Euler XYZ in degrees (BVH uses degrees)
                e = q.to_euler('XYZ')
                frame_eulers_deg.append([float(math.degrees(e.x)),
                                         float(math.degrees(e.y)),
                                         float(math.degrees(e.z))])

            joints_3d.append(frame_joints)
            joint_rot_quat.append(frame_quats)
            joint_rot_euler_deg.append(frame_eulers_deg)

            # camera extrinsic: world->camera 4x4
            M = cam.matrix_world.inverted()
            # print("Frame = ", frame, "Matrix =", M)
            cams_extr.append([[float(M[r][c]) for c in range(4)]
                              for r in range(4)])

        # ----- Render image -----
        scene.render.filepath = os.path.join(outpath, f"frame{ti:04d}.png")
        bpy.ops.render.render(write_still=True)
        print(f"[#] Rendered frame {frame} / {end_frame}", flush=True)

    if wm:
        wm.progress_end()

    print("[#] Rendering finished.", flush=True)

    # --- Save JSON with all skeleton/camera info ---
    if export:
        fps = scene.render.fps / scene.render.fps_base

        data = {
            "joint_names": joint_names,          # [J]
            "bones": bones,                      # [E][2] parent, child (names)
            "rest_offsets": rest_offsets,        # name -> [x,y,z]
            "kinematic_chains": kinematic_chains,# name -> [root,...,name]

            "joints_3d": joints_3d,              # [T][J][3], world-space
            "joint_rot_quat": joint_rot_quat,    # [T][J][4], local (w,x,y,z)
            "joint_rot_euler_deg": joint_rot_euler_deg,  # [T][J][3], degrees

            "cams_intr": cams_intr,              # [fx, fy, cx, cy]
            "cams_extr": cams_extr,              # [T][4][4], world->cam

            "fps": float(fps),
            "frame_range": [int(start_frame), int(end_frame)],
            "resolution": [int(res_x), int(res_y)],
            "camera_name": cam.name,
        }

        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)

        print(f"[#] Saved JSON to {json_path}", flush=True)

if __name__ == "__main__":
    # Handle arguments
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]  # get all args after "--"
    else:
        argv = []  # no args given
    parser = argparse.ArgumentParser(description="Render FBX files from multiple views.")
    parser.add_argument('--n_cam', type=int, default=5, help='Number of cameras to create')
    parser.add_argument('--fbx_path', type=str, default='./mixamo_fbx/', help='Path to input FBX files')
    parser.add_argument('--out_dir', type=str, default='./output/', help='Output directory')
    parser.add_argument('--cam_height', type=float, default=1.0, help='Camera height from the ground')
    parser.add_argument('--cam_radius', type=float, default=3.0, help='Radius of camera circle around the character')
    parser.add_argument('--follow_bone', type=str, default='mixamorig:Hips', help='Bone name for the camera to follow')
    parser.add_argument('--char_color', type=str, default=None, help='Character color: black/white/red/green/blue')
    parser.add_argument('--start_motion_frame', type=int, default=0, help='Start frame of the animation to render')
    parser.add_argument('--sub_sampling', type=int, default=1, help='Sub-sampling factor for frames')
    parser.add_argument('--img_height', type=int, default=512, help='Image height')
    parser.add_argument('--img_width', type=int, default=512, help='Image width')
    args = parser.parse_args(argv)

    fbx = glob.glob(f'{args.fbx_path}/*.fbx')
    print(f"[#] Available FBX files: {fbx}", flush=True)
    print(f"[#] Number of cameras: {args.n_cam}", flush=True)
    print(f"[#] Output directory: {args.out_dir}", flush=True)
    print(f"[#] Camera height: {args.cam_height}", flush=True)
    print(f"[#] Camera radius: {args.cam_radius}", flush=True)
    print(f"[#] Follow bone: {args.follow_bone}", flush=True)
    print(f"[#] Image size: {args.img_width}x{args.img_height}", flush=True)
    scene = bpy.context.scene
    scene.render.image_settings.file_format = 'PNG'
    for f in fbx:
        clear_scene()
        load_fbx(f, char_color=args.char_color)
        create_camera()
        ensure_sun_light()
        setup_background()
        render_multiview(
            base_outpath=f"{args.out_dir}/{os.path.basename(f).split('.')[0]}/",
            armature_name="Armature",
            follow_bone_name=args.follow_bone,   # or "mixamorig:Hips"
            resolution=(args.img_width, args.img_height),
            num_cams=args.n_cam,
            radius=args.cam_radius,
            height=args.cam_height,
            start_motion_frame=args.start_motion_frame,
            sub_sampling=args.sub_sampling,
        )