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
parser.add_argument('--skip_plot_map', default=False, action='store_true', help='Whether to plot 2D joint projections on images')
args = parser.parse_args()

def project(fx, fy, cx, cy, E_bl, j3d):
    F, J, _ = j3d.shape
    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )  # (3, 3)
    H, W = int(cy * 2), int(cx * 2)

    # -----------------------------
    # 3) Extrinsics: Blender → OpenCV camera
    # -----------------------------
    # E_bl: world -> Blender camera (F, 4, 4)

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
    return u, v, z

def unproject(fx, fy, cx, cy, E_bl, j2d):
    F, J, _ = j2d.shape
    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )  # (3, 3)
    K_inv = np.linalg.inv(K)

    # -----------------------------
    # 3) Extrinsics: Blender → OpenCV camera
    # -----------------------------
    # E_bl: world -> Blender camera (F, 4, 4)

    # Blender camera coords:
    #   X_bl: right, Y_bl: up, Z_bl: backward
    # OpenCV camera coords:
    #   X_cv: right, Y_cv: down, Z_cv: forward
    # So: x_cv = x_bl, y_cv = -y_bl, z_cv = -z_bl
    T_bl_to_cv = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)  # (4, 4)
    T_cv_to_bl = np.linalg.inv(T_bl_to_cv)

    # World -> OpenCV camera
    # (1,4,4) @ (F,4,4) -> (F,4,4) via broadcasting
    E_cv = T_bl_to_cv[None, :, :] @ E_bl  # (F, 4, 4)
    E_cv_inv = np.linalg.inv(E_cv)  # (F, 4, 4)

    # -----------------------------
    # 4) Unproject 2D pixels → 3D joints (batched)
    # -----------------------------
    u = j2d[..., 0]
    v = j2d[..., 1]
    depth = j2d[..., 2]

    # (F,J,3)
    pixel_h = np.stack([u * depth, v * depth, depth], axis=-1)

    # K_inv @ pixel_h
    rays_cam = np.einsum("ab,fjb->fja", K_inv, pixel_h)  # (F,J,3)

    # Convert to homogeneous coords
    ones = np.ones((F, J, 1), dtype=np.float32)
    rays_cam_h = np.concatenate([rays_cam, ones], axis=-1)  # (F,J,4)
    # Camera -> world
    # world_pts[f, j, a] = sum_b E_cv_inv[f, a, b] * rays_cam_h[f, j, b]
    world_pts = np.einsum("fab,fjb->fja", E_cv_inv, rays_cam_h)  # (F,J,4)
    j3d_unproj = world_pts[..., :3]  # (F,J,3)
    return j3d_unproj


def process(path):
    path = Path(path)
    # -----------------------------
    # 1) Load JSON
    # -----------------------------
    with open(path / f"skeleton_{path.name}.json", "r") as f:
        data = json.load(f)

    j3d = np.array(data["joints_3d"], dtype=np.float32)  # (F, J, 3)

    # Camera parameters
    fx, fy, cx, cy = data["cams_intr"]
    H, W = int(cy * 2), int(cx * 2)
    E_bl = np.array(data["cams_extr"], dtype=np.float32)  # (F, 4, 4)

    u, v, z = project(fx, fy, cx, cy, E_bl, j3d)
    j2d = np.stack([u, v, z], axis=-1)  # (u,v,depth) => (F, J, 3)

    j3d_unproj = unproject(fx, fy, cx, cy, E_bl, j2d)
    assert np.allclose(j3d, j3d_unproj, atol=1e-6), "Unprojection error too large!"
    j2d_proj = project(fx, fy, cx, cy, E_bl, j3d_unproj)
    assert np.allclose(j2d, np.stack(j2d_proj, axis=-1), atol=1e-6), "Projection after unprojection error too large!"
    # print(j3d, j3d_unproj)
    # print(np.abs(j3d - j3d_unproj).mean())
    # print(j2d.shape, j3d.shape, j3d_unproj.shape)

    T, J, _ = j2d.shape
    if not args.skip_plot_map:
        for ti in range(T):
            # Plot 2D joints on transparent canvas
            canvas = np.ones((H, W, 4), dtype=np.float32)
            canvas[..., 3] = 0.0  # make transparent
            depth_canvas = np.ones((H, W, 4), dtype=np.float32)
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
                if (0 <= x0 < W and 0 <= y0 < H and 0 <= x1 < W and 0 <= y1 < H):
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
                if 0 <= x < W and 0 <= y < H:
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

    save_data = {
        "joints_2d": j2d.tolist(),
        "joints_3d": j3d.tolist(),
        "joints_3d_unproj": j3d_unproj.tolist(),
        "cams_intr": data["cams_intr"],
        "cams_extr": data["cams_extr"],
        "bones": data["bones"],
        "joint_names": data["joint_names"],
        "H": H,
        "W": W,
    }
    
    np.savez_compressed(path / f"motion_data.npz", **save_data)

if __name__ == "__main__":
    t = tqdm.tqdm(glob.glob(f'{args.path}/*/cam*/'), desc="Processing: ")
    for f in t:
        t.set_postfix(file=f)
        process(f)
