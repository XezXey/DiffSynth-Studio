import numpy as np
import torch as th
from einops import rearrange

def unpatchify(x, grid_size, patch_size):
    return rearrange(
        x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
        f=grid_size[0], h=grid_size[1], w=grid_size[2], 
        x=patch_size[0], y=patch_size[1], z=patch_size[2]
    )
    
def unproject_torch(fx, fy, cx, cy, E_bl, j2d, eps=1e-8)
    """
    Args:
        fx, fy, cx, cy: scalars (python float or torch scalar)
        E_bl: (T, 4, 4) world -> Blender camera extrinsics (torch.Tensor)
        j2d:  (T, J, 3) where last dim is (u, v, depth) (torch.Tensor)
            IMPORTANT: depth must be consistent with your projection convention
        eps: small constant for numeric safety

    Returns:
        j3d_unproj: (T, J, 3) unprojected 3D points in world coordinates
    """

    assert E_bl.ndim == 3 and E_bl.shape[-2:] == (4, 4), f"E_bl must be (T,4,4), got {E_bl.shape}"
    assert j2d.ndim == 3 and j2d.shape[-1] == 3, f"j2d must be (T,J,3), got {j2d.shape}"

    device = j2d.device
    dtype  = j2d.dtype
    T, J, _ = j2d.shape

    # Intrinsics and inverse
    K = th.tensor(
        [[fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]],
        device=device, dtype=dtype
    )
    K_inv = th.linalg.inv(K)  # (3,3)

    # Blender -> OpenCV camera coords
    T_bl_to_cv = th.tensor(
        [[1.0,  0.0,  0.0, 0.0],
        [0.0, -1.0,  0.0, 0.0],
        [0.0,  0.0, -1.0, 0.0],
        [0.0,  0.0,  0.0, 1.0]],
        device=device, dtype=dtype
    )

    # World -> OpenCV camera extrinsics and its inverse
    E_cv = T_bl_to_cv.unsqueeze(0) @ E_bl.to(device=device, dtype=dtype)  # (F,4,4)
    E_cv_inv = th.linalg.inv(E_cv)  # (F,4,4)

    # Unproject pixels -> camera coordinates
    u = j2d[..., 0]            # (F,J)
    v = j2d[..., 1]            # (F,J)
    depth = j2d[..., 2]        # (F,J)

    # Optional safety: avoid exactly zero depth
    depth_safe = th.where(depth.abs() < eps, depth.new_full((), eps), depth)

    # pixel_h = [u*depth, v*depth, depth]  (F,J,3)
    pixel_h = th.stack([u * depth_safe, v * depth_safe, depth_safe], dim=-1)

    # rays_cam = K_inv @ pixel_h  (F,J,3)
    rays_cam = th.einsum("ab,fjb->fja", K_inv, pixel_h)

    # Homogeneous (F,J,4)
    ones = th.ones((T, J, 1), device=device, dtype=dtype)
    rays_cam_h = th.cat([rays_cam, ones], dim=-1)

    # Camera -> world (F,J,4)
    world_pts = th.einsum("fab,fjb->fja", E_cv_inv, rays_cam_h)

    # (F,J,3)
    j3d_unproj = world_pts[..., :3]
    return j3d_unproj

def map_to_joint(J, joint_map):
        """
        Inputs: 
            joint_map: (b, c, t, h, w)
                - c = 2 * J (heatmap and depth channels)
        Returns:
            pixel_coords: (b, J, t, 2) - x,y pixel coordinates
            depth: (b, J, t) - depth values
        """
        b, c, t, h, w = joint_map.shape
        joint_map_list = th.chunk(joint_map, J, dim=1)  # List of (b, 2, t, h, w), length J
        all_pixel_coords = []
        all_depths = []
        for map in joint_map_list:
            heatmap = map[:, 0, :, :, :]  # (b, t, h, w)
            depth_map = map[:, 1, :, :, :]  # (b, t, h, w)

            # Softmax over spatial dimensions to get probabilities
            heatmap_flat = heatmap.view(b, t, -1)  # (b, t, h*w)
            prob_map = th.softmax(heatmap_flat, dim=-1).view(b, t, h, w)  # (b, t, h, w)

            # Create coordinate grids
            y_coords, x_coords = th.meshgrid(th.linspace(0, 1, h, device=joint_map_list[0].device), th.linspace(0, 1, w, device=joint_map_list[0].device), indexing='ij')

            y_coords = y_coords.view(1, 1, h, w).expand(b, t, h, w)
            x_coords = x_coords.view(1, 1, h, w).expand(b, t, h, w)

            x_pixel = th.sum(prob_map * x_coords, dim=(2, 3))
            y_pixel = th.sum(prob_map * y_coords, dim=(2, 3)) 

            pixel_coords = th.stack([x_pixel, y_pixel], dim=-1)  # (b, t, 2)

            # Compute expected depth
            depth = th.sum(prob_map * depth_map, dim=(2, 3))  # (b, t)
            depth = depth.unsqueeze(-1)  # (b, t, 1)

            all_pixel_coords.append(pixel_coords[:, None, :, :])  # (b, 1, t, 2)
            all_depths.append(depth[:, None, :, :])  # (b, 1, t, 1)
        
        pixel_coords = th.cat(all_pixel_coords, dim=1)  # (b, J, t, 2)
        depth = th.cat(all_depths, dim=1)  # (b, J, t, 1)

        return pixel_coords, depth