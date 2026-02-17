import torch as th
from lightning.pytorch.utilities import rank_zero_only
import lightning as L
import numpy as np
import glob
from dataset import DitFeaturesDataset
from model import JointVAE38, Head
import argparse
from einops import rearrange

class TrainOnDiTFeatures(L.LightningModule):
    def __init__(self, 
                 dim, 
                 out_dim, 
                 patch_size, 
                 J, 
                 out_J_chn, 
                 preferred_dit_block_id,
                 eps=1e-8, 
                 num_res_blocks=0, 
                 lr=1e-4):
        super().__init__()
        self.head = Head(dim=dim, out_dim=out_dim, patch_size=patch_size, eps=eps).train()
        self.joint_vae = JointVAE38(J=J, out_J_chn=out_J_chn, z_dim=48, num_res_blocks=num_res_blocks).train()
        self.joint_head = th.nn.Conv3d(3, J * out_J_chn, 3, padding=1).train()
        self.lr = lr
        self.preferred_dit_block_id = preferred_dit_block_id
        self.J = J
        self.out_j_chn = out_J_chn

        self.make_parameters_trainable(self.head)
        self.make_parameters_trainable(self.joint_vae)
        self.make_parameters_trainable(self.joint_head)
        
        self.training_step_outputs = []
        self.step = 0

    def make_parameters_trainable(self, module):
        for param in module.parameters():
            param.requires_grad = True

    def configure_optimizers(self):
        optimizer = th.optim.AdamW(list(self.head.parameters()) + list(self.joint_vae.parameters()), lr=self.lr)
        return optimizer

    def unpatchify(self, x, grid_size, patch_size):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=patch_size[0], y=patch_size[1], z=patch_size[2]
        )
    
    def training_step(self, batch, batch_idx):
        """
        batch contains the following keys:
        - 'dit_features': pre-extracted DiT features, shape (1, #dit_blocks, 1, #tokens, C)
        - 'joints_3d': ground truth 3D joint positions, shape (1, T, J, 3)
        - 'joints_2d': ground truth 2D joint positions, shape (1, T, J, 2)

        """
        
        inputs = batch
        # for k in inputs.keys():
        #     if isinstance(inputs[k], th.Tensor):
        #         print(f"{k}: shape {inputs[k].shape}, dtype {inputs[k].dtype}, device {inputs[k].device}")
        # exit()
        dit_features = inputs["dit_features"].squeeze(0)  # B, C, H, W
        inp = dit_features.type(th.float32)  # 1, #tokens, C
        grid_size = inputs["grid_size"]
        patch_size = inputs["patch_size"]
        # print(grid_size, patch_size)

        out_head = self.head(inp)  # 1, out_dim, H', W
        # print(out_head.shape)
        out_unpatched = self.unpatchify(out_head, grid_size, patch_size)  # 1, out_dim, T, H, W
        # print(out_unpatched.shape)
        out_decoded = self.joint_vae.decode(out_unpatched, device='cuda')  # 1, J*3, T, 1, 1
        # print(out_decoded.shape)
        out_joints_map = self.joint_head(out_decoded)  # 1, J*2, T, 1, 1
        # print(out_joints_map.shape)

        pixel_coords, depth = self.map_to_joint(out_joints_map)  # pixel_coords: (1, J, T, 2); depth: (1, J, T, 1)
        # print(pixel_coords.shape, depth.shape)

        # print(inputs["cams_intr"].shape)
        # print(inputs["cams_extr"].shape)
        fx, fy, cx, cy = inputs["cams_intr"].squeeze(0) # (4)
        org_h = cy * 2.0 + 1
        org_w = cx * 2.0 + 1
        E_bl = inputs["cams_extr"].squeeze(0)  # (T, 4, 4)
        # print(th.is_tensor(E_bl), E_bl.shape, E_bl.dtype, E_bl.ndim)

        m3d_gt = inputs["joints_3d"].squeeze(0)  # T, J, 3
        m2d_gt = inputs["joints_2d"].squeeze(0)  # T, J, 3
        m2d_gt = m2d_gt[..., :2]    # only keep (u, v)
        assert m3d_gt.shape[0] == m2d_gt.shape[0], "Number of frames mismatch between 3D and 2D joints."
        assert m3d_gt.shape[1] == m2d_gt.shape[1], "Number of joints mismatch between 3D and 2D joints."
        
        
        m2d_gt[..., 0] = m2d_gt[..., 0] / (org_w - 1)     # normalize to [0,1]
        m2d_gt[..., 1] = m2d_gt[..., 1] / (org_h - 1)   # normalize to [0,1]
        mask_2d = th.logical_and(m2d_gt >= 0.0, m2d_gt <= 1.0)
        
        h = inputs["height"]
        w = inputs["width"]
        u = pixel_coords[..., 0] * (org_w - 1)    # B, J, T
        v = pixel_coords[..., 1] * (org_h - 1)    # B, J, T
        d = depth[..., 0]
        
        motion_pred_2d = th.stack([u / (org_w - 1), v / (org_h - 1)], dim=-1).squeeze(0).permute(1, 0, 2)  # B, J, T -> T, J, 2
        motion_pred_3d = self.unproject_torch(fx, fy, cx, cy, E_bl, th.stack([u, v, d], dim=-1).squeeze(0).permute(1, 0, 2))
        training_target_3d = m3d_gt
        assert motion_pred_3d.shape == training_target_3d.shape, f"motion_pred shape {motion_pred_3d.shape} does not match training_target shape {training_target_3d.shape}"
        assert motion_pred_2d.shape == m2d_gt.shape, f"motion_pred_2d shape {motion_pred_2d.shape} does not match gt_motion_2d shape {m2d_gt.shape}"

        loss_3d = th.nn.functional.mse_loss(motion_pred_3d.float(), training_target_3d.float())
        loss_2d = th.nn.functional.mse_loss(motion_pred_2d.float(), m2d_gt.float()) * mask_2d.float()
        loss_2d = loss_2d.sum() / (mask_2d.float().sum() + 1e-8)
        loss = loss_3d + loss_2d * 10.0
        
        # inputs.update({"motion_pred": motion_pred_3d, "training_target": training_target_3d, 
        #             "motion_pred_2d": motion_pred_2d, "gt_motion_2d": m2d_gt})
        
        # self.training_step_outputs.append({"loss": loss, "inputs": inputs})
        # self.step += 1
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    @rank_zero_only
    def on_train_step_end(self, outputs, batch, batch_idx):
        print(outputs)
        exit()

    @rank_zero_only
    def on_train_epoch_end(self, outputs, batch, batch_idx):
        print(outputs)
        exit()

    def unproject_torch(self, fx, fy, cx, cy, E_bl, j2d, eps=1e-8):
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

    def map_to_joint(self, joint_map):
        """
        Inputs: 
            joint_map: (b, c, t, h, w)
                - c = 2 * J (heatmap and depth channels)
        Returns:
            pixel_coords: (b, J, t, 2) - x,y pixel coordinates
            depth: (b, J, t) - depth values
        """
        b, c, t, h, w = joint_map.shape
        joint_map_list = th.chunk(joint_map, self.J, dim=1)  # List of (b, 2, t, h, w), length J
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
