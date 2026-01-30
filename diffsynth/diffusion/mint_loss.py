from .base_pipeline import BasePipeline
import torch

def TrainingOnDitFeaturesLoss(pipe: BasePipeline, extra_modules=None, **inputs):
    # max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    # min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    preferred_timestep_id = inputs.get("preferred_timestep_id", [-1])
    timestep_id = torch.tensor(preferred_timestep_id, dtype=torch.int)

    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    print(timestep)
    print(pipe.scheduler.timesteps)
    exit()
    
    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    # training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred, return_dict = pipe.model_fn(**models, **inputs, timestep=timestep)

    dit_features = return_dict.get("dit_features", None)
    grid_size = return_dict.get("grid_size", None)
    assert dit_features is not None, "Dit features not returned from model_fn."
    assert grid_size is not None, "Grid size not returned from model_fn."

    #NOTE: Motion prediction
    pixel_coords, depth = extra_modules(pipe, dit_features, grid_size)  # pixel_coords = (B, J, T, 2); depth = (B, J, T, 1)
    # motion_pred = torch.cat([pixel_coords, depth], dim=-1)
    # 3D reconstruction from (u, v, depth) to (x, y, z) using camera intrinsics and extrinsics
    # print("motion_pred shape: ", motion_pred.shape)


    fx, fy, cx, cy = inputs["cams_intr"]
    org_h = cy * 2.0 + 1
    org_w = cx * 2.0 + 1
    E_bl = torch.tensor(inputs["cams_extr"]).to(device=pipe.device)
    
    
    gt_motion_3d = torch.tensor(inputs["joints_3d"]).to(device=pipe.device)
    gt_motion_2d = torch.tensor(inputs["joints_2d"]).to(device=pipe.device)[..., :2]
    gt_motion_2d[..., 0] = gt_motion_2d[..., 0] / (org_w - 1)     # normalize to [0,1]
    gt_motion_2d[..., 1] = gt_motion_2d[..., 1] / (org_h - 1)   # normalize to [0,1]
    mask_2d = torch.logical_and(gt_motion_2d >= 0.0, gt_motion_2d <= 1.0)
    
    h = inputs["height"]
    w = inputs["width"]
    u = pixel_coords[..., 0] * (org_w - 1)    # B, J, T
    v = pixel_coords[..., 1] * (org_h - 1)    # B, J, T
    d = depth[..., 0]
    
    motion_pred_2d = torch.stack([u / (org_w - 1), v / (org_h - 1)], dim=-1).squeeze(0).permute(1, 0, 2)  # B, J, T -> T, J, 2
    motion_pred_3d = unproject_torch(fx, fy, cx, cy, E_bl, torch.stack([u, v, d], dim=-1).squeeze(0).permute(1, 0, 2))
    training_target_3d = gt_motion_3d
    assert motion_pred_3d.shape == training_target_3d.shape, f"motion_pred shape {motion_pred_3d.shape} does not match training_target shape {training_target_3d.shape}"
    assert motion_pred_2d.shape == gt_motion_2d.shape, f"motion_pred_2d shape {motion_pred_2d.shape} does not match gt_motion_2d shape {gt_motion_2d.shape}"

    loss_3d = torch.nn.functional.mse_loss(motion_pred_3d.float(), training_target_3d.float())
    loss_2d = torch.nn.functional.mse_loss(motion_pred_2d.float(), gt_motion_2d.float()) * mask_2d.float()
    loss_2d = loss_2d.sum() / (mask_2d.float().sum() + 1e-8)
    loss = loss_3d + loss_2d
    
    inputs.update({"motion_pred": motion_pred_3d, "training_target": training_target_3d, 
                   "motion_pred_2d": motion_pred_2d, "gt_motion_2d": gt_motion_2d})
    return loss, inputs

def unproject_torch(fx, fy, cx, cy, E_bl, j2d, eps=1e-8):
    """
    Args:
        fx, fy, cx, cy: scalars (python float or torch scalar)
        E_bl: (F, 4, 4) world -> Blender camera extrinsics (torch.Tensor)
        j2d:  (F, J, 3) where last dim is (u, v, depth) (torch.Tensor)
              IMPORTANT: depth must be consistent with your projection convention
        eps: small constant for numeric safety

    Returns:
        j3d_unproj: (F, J, 3) unprojected 3D points in world coordinates
    """
    assert E_bl.ndim == 3 and E_bl.shape[-2:] == (4, 4), f"E_bl must be (F,4,4), got {E_bl.shape}"
    assert j2d.ndim == 3 and j2d.shape[-1] == 3, f"j2d must be (F,J,3), got {j2d.shape}"

    device = j2d.device
    dtype  = j2d.dtype
    T, J, _ = j2d.shape

    # Intrinsics and inverse
    K = torch.tensor(
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]],
        device=device, dtype=dtype
    )
    K_inv = torch.linalg.inv(K)  # (3,3)

    # Blender -> OpenCV camera coords
    T_bl_to_cv = torch.tensor(
        [[1.0,  0.0,  0.0, 0.0],
         [0.0, -1.0,  0.0, 0.0],
         [0.0,  0.0, -1.0, 0.0],
         [0.0,  0.0,  0.0, 1.0]],
        device=device, dtype=dtype
    )

    # World -> OpenCV camera extrinsics and its inverse
    E_cv = T_bl_to_cv.unsqueeze(0) @ E_bl.to(device=device, dtype=dtype)  # (F,4,4)
    E_cv_inv = torch.linalg.inv(E_cv)  # (F,4,4)

    # Unproject pixels -> camera coordinates
    u = j2d[..., 0]            # (F,J)
    v = j2d[..., 1]            # (F,J)
    depth = j2d[..., 2]        # (F,J)

    # Optional safety: avoid exactly zero depth
    depth_safe = torch.where(depth.abs() < eps, depth.new_full((), eps), depth)

    # pixel_h = [u*depth, v*depth, depth]  (F,J,3)
    pixel_h = torch.stack([u * depth_safe, v * depth_safe, depth_safe], dim=-1)

    # rays_cam = K_inv @ pixel_h  (F,J,3)
    rays_cam = torch.einsum("ab,fjb->fja", K_inv, pixel_h)

    # Homogeneous (F,J,4)
    ones = torch.ones((T, J, 1), device=device, dtype=dtype)
    rays_cam_h = torch.cat([rays_cam, ones], dim=-1)

    # Camera -> world (F,J,4)
    world_pts = torch.einsum("fab,fjb->fja", E_cv_inv, rays_cam_h)

    # (F,J,3)
    j3d_unproj = world_pts[..., :3]
    return j3d_unproj

def project_torch(fx, fy, cx, cy, E_bl, j3d, eps=1e-8):
    """
    Args:
        fx, fy, cx, cy: scalars (python float or torch scalar)
        E_bl: (F, 4, 4) world -> Blender camera extrinsics (torch.Tensor)
        j3d:  (F, J, 3) 3D joints in world coords (torch.Tensor)
        eps:  small value to avoid division by zero

    Returns:
        u, v, z: each (F, J)
            u,v are pixel coordinates (float), z is depth in OpenCV camera coords.
    """
    assert E_bl.ndim == 3 and E_bl.shape[-2:] == (4, 4), f"E_bl must be (F,4,4), got {E_bl.shape}"
    assert j3d.ndim == 3 and j3d.shape[-1] == 3, f"j3d must be (F,J,3), got {j3d.shape}"

    device = j3d.device
    dtype  = j3d.dtype
    F, J, _ = j3d.shape

    # Intrinsics K (3,3)
    K = torch.tensor(
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]],
        device=device, dtype=dtype
    )

    # Blender -> OpenCV camera coords (4,4): x_cv=x_bl, y_cv=-y_bl, z_cv=-z_bl
    T_bl_to_cv = torch.tensor(
        [[1.0,  0.0,  0.0, 0.0],
         [0.0, -1.0,  0.0, 0.0],
         [0.0,  0.0, -1.0, 0.0],
         [0.0,  0.0,  0.0, 1.0]],
        device=device, dtype=dtype
    )

    # World -> OpenCV camera extrinsics: (F,4,4)
    E_cv = T_bl_to_cv.unsqueeze(0) @ E_bl.to(device=device, dtype=dtype)

    # Homogeneous joints: (F,J,4)
    ones = torch.ones((F, J, 1), device=device, dtype=dtype)
    j3d_h = torch.cat([j3d, ones], dim=-1)

    # World -> camera: (F,J,4)
    cam_pts = torch.einsum("fab,fjb->fja", E_cv, j3d_h)

    # Camera XYZ: (F,J,3)
    pts_cam3 = cam_pts[..., :3]

    # Apply intrinsics: (F,J,3)
    proj = torch.einsum("ab,fjb->fja", K, pts_cam3)

    # Perspective divide
    z = proj[..., 2]                              # (F,J)
    z_safe = torch.where(z.abs() < eps, z.new_full((), eps), z)
    u = proj[..., 0] / z_safe
    v = proj[..., 1] / z_safe

    return u, v, z