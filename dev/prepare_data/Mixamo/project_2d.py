import json
import cv2
import glob
import os
import tqdm
import numpy as np
from pathlib import Path
import argparse
parser = argparse.ArgumentParser(description="Project 3D joints to 2D using camera parameters from JSON.")
parser.add_argument('--path', type=str, required=True, help='Path to the directory containing skeleton_{name}.json')
args = parser.parse_args()

def project2d(path):
    path = Path(path)
    # -----------------------------
    # 1) Load JSON
    # -----------------------------
    with open(path / f"skeleton_{path.name}.json", "r") as f:
        data = json.load(f)
    # print(len(data['bones']))
    # print(len(data['joint_names']))
    # print(np.array((data['joints_3d'])).shape)
    # exit()
    # print(data.keys())

    # joints_3d: (F, J, 3)
    j3d = np.array(data["joints_3d"], dtype=np.float32)  # (F, J, 3)
    F, J, _ = j3d.shape

    # -----------------------------
    # 2) Intrinsics K from [fx, fy, cx, cy]
    # -----------------------------
    fx, fy, cx, cy = data["cams_intr"]
    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )  # (3, 3)

    # -----------------------------
    # 3) Extrinsics: Blender → OpenCV camera
    # -----------------------------
    # E_bl: world -> Blender camera (F, 4, 4)
    E_bl = np.array(data["cams_extr"], dtype=np.float32)  # (F, 4, 4)

    # Blender camera coords:
    #   X_bl: right, Y_bl: up, Z_bl: backward
    # OpenCV camera coords:
    #   X_cv: right, Y_cv: down, Z_cv: forward
    # So: x_cv = x_bl, y_cv = -y_bl, z_cv = -z_bl
    T_bl_to_cv = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)  # (4, 4)

    # World -> OpenCV camera
    # (1,4,4) @ (F,4,4) -> (F,4,4) via broadcasting
    E_cv = T_bl_to_cv[None, :, :] @ E_bl  # (F, 4, 4)

    # -----------------------------
    # 4) Project 3D joints → 2D pixels (batched)
    # -----------------------------
    # (F, J, 3) -> (F, J, 4) homogeneous
    ones = np.ones((F, J, 1), dtype=np.float32)
    j3d_h = np.concatenate([j3d, ones], axis=-1)  # (F, J, 4)

    # World -> camera (OpenCV)
    # cam_pts[f, j, a] = sum_b E_cv[f, a, b] * j3d_h[f, j, b]
    cam_pts = np.einsum("fab,fjb->fja", E_cv, j3d_h)  # (F, J, 4)

    # Keep XYZ in camera coords
    pts_cam3 = cam_pts[..., :3]  # (F, J, 3)

    # Apply intrinsics: K (3,3) * (F,J,3)
    proj = np.einsum("ab,fjb->fja", K, pts_cam3)  # (F, J, 3)

    # Perspective divide
    z = proj[..., 2]
    eps = 1e-8
    z_safe = np.where(z == 0, eps, z)

    u = proj[..., 0] / z_safe
    v = proj[..., 1] / z_safe
    j2d = np.stack([u, v, z], axis=-1)  # (u,v,depth) => (F, J, 3)

    T, J, _ = j2d.shape
    for ti in range(T):
        # Plot 2D joints on transparent canvas
        canvas = np.ones((512, 512, 4), dtype=np.float32)
        canvas[..., 3] = 0.0  # make transparent
        depth_canvas = np.ones((512, 512, 4), dtype=np.float32)
        depth_canvas[..., 3] = 0.0  # make transparent
        depth_value = j2d[ti, :, 2] - np.min(j2d[ti, :, 2])
        depth_value = depth_value / (np.max(depth_value) + 1e-8)
        
        # Draw bones
        for start_name, end_name in data["bones"]:
            start_idx = data["joint_names"].index(start_name)
            end_idx = data["joint_names"].index(end_name)
            if 'right' in start_name.lower() or 'right' in end_name.lower():
                bcolor = (0.0, 0.0, 1.0, 1.0)  # Blue for right side
            elif 'left' in start_name.lower() or 'left' in end_name.lower():
                bcolor = (1.0, 0.0, 0.0, 1.0)  # Magenta for left side
            else:
                bcolor = (0.0, 1.0, 0.0, 1.0)  # Green for others
                
            x0, y0 = j2d[ti, start_idx][:2]
            x1, y1 = j2d[ti, end_idx][:2]
            if (0 <= x0 < 512 and 0 <= y0 < 512 and 0 <= x1 < 512 and 0 <= y1 < 512):
                cv2.line(
                    canvas,
                    pt1=(int(x0), int(y0)),
                    pt2=(int(x1), int(y1)),
                    color=bcolor,
                    thickness=2,
                )
                cv2.line(
                    depth_canvas,
                    pt1=(int(x0), int(y0)),
                    pt2=(int(x1), int(y1)),
                    color=bcolor,
                    thickness=2,
                )
        
        for tj in range(J):
            x, y = j2d[ti, tj][:2]
            if 0 <= x < 512 and 0 <= y < 512:
                # Draw joint
                cv2.circle(
                    canvas,
                    center=(int(x), int(y)),
                    radius=1,
                    color=(0.0, 1.0, 0.0, 1.0),
                    thickness=2,
                )
                # brightness = 0.3 + 0.7 * depth_value[tj]
                # depth_color = np.array(bcolor[:3]) * brightness
                # print(depth_color)
                # exit()

                value = int(depth_value[tj] * 255)
                gray = np.array([[value]], dtype=np.uint8)   # shape (1,1)
                depth_color = cv2.applyColorMap(gray, cv2.COLORMAP_JET)[0, 0] / 255.0  # Normalize to [0,1]
                
                cv2.circle(
                    depth_canvas,
                    center=(int(x), int(y)),
                    radius=1,
                    color=(depth_color[0], depth_color[1], depth_color[2]),
                    thickness=2,
                )
                
        cv2.imwrite(str(path / f"proj{ti+1:04d}.png"), (canvas[..., :3] * 255).astype(np.uint8))
        cv2.imwrite(str(path / f"depth{ti+1:04d}.png"), (depth_canvas[..., :3] * 255).astype(np.uint8))

if __name__ == "__main__":
    t = tqdm.tqdm(glob.glob(f'{args.path}/*/cam*/'), desc="Processing: ")
    for f in t:
        t.set_postfix(file=f)
        project2d(f)