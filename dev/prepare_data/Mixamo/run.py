import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--blender_path', default='/ist/users/puntawatp/Dev/SkelAg/Blender/blender-5.0.0-linux-x64/blender', type=str, help='Path to Blender executable')
parser.add_argument('--fbx', type=str, required=True, help='Path to input FBX file')
parser.add_argument('--out_dir', type=str, required=True, help='Output directory')
parser.add_argument('--n_cam', type=int, default=1, help='Number of cameras to create')
parser.add_argument('--follow_bone', type=str, default='mixamorig:Hips', help='Bone name for the camera to follow')
parser.add_argument('--cam_height', type=float, default=3.0, help='Camera height from the ground')
parser.add_argument('--cam_radius', type=float, default=4.5, help='Radius of camera circle around the character')
parser.add_argument('--img_width', type=int, default=512, help='Image width')
parser.add_argument('--img_height', type=int, default=512, help='Image height')
parser.add_argument('--run_blender', action='store_true', default=False, help='Enable blender execution')
parser.add_argument('--run_projection', action='store_true', default=False, help='Enable 2D projection after rendering')
parser.add_argument('--use_gpu', action='store_true', default=False, help='Use GPU for rendering in Blender')
parser.add_argument('--only_body_joints', action='store_true', default=False, help='Only render body joints without the fingers')
parser.add_argument('--skip_plot_map', action='store_true', default=False, help='Skip plotting the 2D joint heatmap and skeleton overlay on the rendered images')
args = parser.parse_args()

# CMD: /ist/users/puntawatp/Dev/SkelAg/Blender/blender-5.0.0-linux-x64/blender -b -P gen_data.py
# CMD: python project2d.py
if __name__ == "__main__":
    if args.run_blender:
        # Arguments for Blender script
        fbx = f"--fbx \"{args.fbx}\""
        out_dir = f"--out_dir \"{args.out_dir}\""
        n_cam = f"--n_cam {args.n_cam}"
        follow_bone = f"--follow_bone {args.follow_bone}"
        cam_height = f"--cam_height {args.cam_height}"
        cam_radius = f"--cam_radius {args.cam_radius}"
        img_width = f"--img_width {args.img_width}"
        img_height = f"--img_height {args.img_height}"
        render_settings = f"{n_cam} {follow_bone} {cam_height} {cam_radius} {img_width} {img_height}"

        blender_cmd = f"{args.blender_path} -b --factory-startup -noaudio -P ./render_fbx.py -- {fbx} {out_dir} {render_settings}"
        blender_cmd += " --use_gpu" if args.use_gpu else ""
        print(blender_cmd)
        os.system(blender_cmd)
        print("#" * 100)

    if args.run_projection:
        # After rendering, run the 2D projection

        projection_cmd = f"python ./project_2d.py --path \"{args.out_dir}\""
        projection_cmd += " --only_body_joints" if args.only_body_joints else ""
        projection_cmd += " --skip_plot_map" if args.skip_plot_map else ""
        os.system(projection_cmd)
