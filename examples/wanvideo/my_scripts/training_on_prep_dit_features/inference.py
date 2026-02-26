import numpy as np
import torch as th
import argparse, os, glob, plotly
from model import JointVAE38, Head
from einops import rearrange
from dataset import DitFeaturesDataset
from diffsynth.diffusion.vis import MultiSkeleton2D3DAnimator
from mylogger.logger import init_logger
logger = init_logger("inference_on_prep_dit_features.log")

from dev.experiment.vae_decoder.vae_decoder import Head
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, required=True)
parser.add_argument('--dit_features_path', type=str, required=True)
parser.add_argument('--output_path', type=str, required=True)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--motion_name', type=str, default=None)
parser.add_argument("--J", type=int, default=25, help="Number of joints.")
parser.add_argument("--out_J_chn", type=int, default=2, help="Output channels for each joint.")
parser.add_argument("--predict_motion_dt", action='store_true', help="Whether the model predicts motion delta (True) or absolute motion (False). If True, the depth output will be treated as motion delta and converted to absolute motion by cumulative summation over time.")
parser.add_argument("--preferred_dit_block_id", type=int, default=-1, help="Preferred DiT block ID.")
args = parser.parse_args()

class SkelAg(th.nn.Module):
    def __init__(self,
                dim,
                out_dim,
                patch_size,
                J,
                out_J_chn,
                num_res_blocks=0,
                predict_motion_dt=False,
                eps=1e-8):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.J = J
        self.out_J_chn = out_J_chn
        self.predict_motion_dt = predict_motion_dt

        self.head = Head(dim=dim, out_dim=out_dim, patch_size=patch_size, eps=eps).eval()
        self.joint_vae = JointVAE38(J=J, out_J_chn=out_J_chn, z_dim=48, num_res_blocks=num_res_blocks).eval()
        self.joint_head = th.nn.Conv3d(3, J * out_J_chn, 3, padding=1).eval()

    def forward(self, batch):
        inputs = batch
        dit_features = inputs["dit_features"].squeeze(0)  # B, C, H, W
        inp = dit_features.type(th.float32).cuda()  # 1, #tokens, C
        grid_size = inputs["grid_size"]
        patch_size = inputs["patch_size"]

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

        m3d_gt = inputs["joints_3d"].squeeze(0).cuda()  # T, J, 3
        m2d_gt = inputs["joints_2d"].squeeze(0).cuda()  # T, J, 3
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
        if self.predict_motion_dt:
            d = th.cumsum(d, dim=2)  # convert from motion delta to absolute motion, assume first timestep would be the absolute motion
        else: 
            d = d
        
        motion_pred_2d = th.stack([u / (org_w - 1), v / (org_h - 1)], dim=-1).squeeze(0).permute(1, 0, 2)  # B, J, T -> T, J, 2
        motion_pred_3d = self.unproject_torch(fx, fy, cx, cy, E_bl, th.stack([u, v, d], dim=-1).squeeze(0).permute(1, 0, 2))
        training_target_3d = m3d_gt
        assert motion_pred_3d.shape == training_target_3d.shape, f"motion_pred shape {motion_pred_3d.shape} does not match training_target shape {training_target_3d.shape}"
        assert motion_pred_2d.shape == m2d_gt.shape, f"motion_pred_2d shape {motion_pred_2d.shape} does not match gt_motion_2d shape {m2d_gt.shape}"

        loss_3d = th.nn.functional.mse_loss(motion_pred_3d.float(), training_target_3d.float())
        loss_2d = th.nn.functional.mse_loss(motion_pred_2d.float(), m2d_gt.float()) * mask_2d.float()
        loss_2d = loss_2d.sum() / (mask_2d.float().sum() + 1e-8)
        loss = loss_3d + loss_2d * 1000.0

        output_dict = {
            "motion_pred_3d": motion_pred_3d.detach().cpu(),
            "motion_gt_3d": training_target_3d.detach().cpu(),
            "motion_pred_2d": motion_pred_2d.detach().cpu(),
            "motion_gt_2d": m2d_gt.detach().cpu(),
            "loss_3d": loss_3d.item(),
            "loss_2d": loss_2d.item(),
            "joint_names": inputs["joint_names"],
            "bones": inputs["bones"]
        }
        # loss_dict = {"loss": loss, "loss_3d": loss_3d, "loss_2d": loss_2d}

        # return loss_dict, output_dict
        return None, output_dict

    def unpatchify(self, x, grid_size, patch_size):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=patch_size[0], y=patch_size[1], z=patch_size[2]
        )

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

    def get_params_stats(self):
        # Compute mean/std of all parameters for debugging
        params = list(self.head.parameters()) + list(self.joint_vae.parameters()) + list(self.joint_head.parameters())
        all_params = th.cat([p.detach().cpu().flatten() for p in params])
        mean = th.mean(all_params).item()
        std = th.std(all_params).item()
        return mean, std

def sorted_by_chunk_id(file_list):
    # File will have *_chunk{chunk_id}_* in its name. Sort by chunk_id using regex.
    import re
    def get_chunk_id(file_name):
        match = re.search(r"_chunk(\d+)_", file_name)
        if match:
            return int(match.group(1))
        else:
            return -1  # If no chunk_id found, put it at the beginning
    return sorted(file_list, key=get_chunk_id)

if __name__ == "__main__":
    # Get params from the sample data
    # dit_features_path_list = glob.glob(f"{args.dit_features_path}/*.pth")
    motion_to_infer = glob.glob(f"{args.dit_features_path}/*_{args.motion_name}_*.pth")
    if len(motion_to_infer) == 0:
        logger.error(f"No sample found for motion name '{args.motion_name}' in path '{args.dit_features_path}'. Please check the motion name and path.")
        exit(1)
    else:
        motion_to_infer = sorted_by_chunk_id(motion_to_infer)

    test_motion_dataset = DitFeaturesDataset(motion_to_infer, preferred_dit_block_id=args.preferred_dit_block_id)
    test_motion_dataloader = th.utils.data.DataLoader(test_motion_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=test_motion_dataset.collate_fn_, persistent_workers=True)

    sample_dat = next(iter(test_motion_dataloader))
    dim = sample_dat["dim"].item()
    out_dim = sample_dat["out_dim"].item()
    patch_size = [i.item() for i in sample_dat["patch_size"]]
    grid_size = [i.item() for i in sample_dat["grid_size"]]

    logger.warning("Parameters loaded from sample data:")
    logger.info(f"dim: {dim}")
    logger.info(f"out_dim: {out_dim}")
    logger.info(f"patch_size: {patch_size}")
    logger.info(f"grid_size: {grid_size}")
    logger.warning("Motion parameters:")
    logger.info(f"J: {args.J}")
    logger.info(f"out_J_chn: {args.out_J_chn}")
    logger.info(f"predict_motion_dt: {args.predict_motion_dt}")
    logger.warning("Inference parameters:")
    logger.info(f"gpu_id: {args.gpu_id}")
    logger.info(f"ckpt: {args.ckpt}")


    # Initialize model
    model = SkelAg(dim=dim, out_dim=out_dim, patch_size=patch_size, J=args.J, out_J_chn=args.out_J_chn, predict_motion_dt=args.predict_motion_dt).cuda(args.gpu_id)
    mean_init, std_init = model.get_params_stats()
    logger.warning(f"Model initialized. Param mean: {mean_init:.6f}, std: {std_init:.6f}")
    state_dict = th.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state_dict)
    mean_loaded, std_loaded = model.get_params_stats()
    logger.warning(f"Model loaded from checkpoint. Param mean: {mean_loaded:.6f}, std: {std_loaded:.6f}")
    assert abs(mean_loaded - mean_init) > 1e-5 or abs(std_loaded - std_init) > 1e-5, "Model parameters do not seem to be loaded properly (mean/std are almost the same as initialized). Please check the checkpoint path and content."
    model.eval().cuda(args.gpu_id)

    
    all_video_frames = []
    motion2d_frames = []
    motion3d_frames = []
    gt_motion2d_frames = []
    gt_motion3d_frames = []

    for data in test_motion_dataloader:
        _, output = model(data)
        all_video_frames.append(data["input_video"])
        motion2d_frames.append(output["motion_pred_2d"].detach().cpu())
        motion3d_frames.append(output["motion_pred_3d"].detach().cpu())
        gt_motion2d_frames.append(output["motion_gt_2d"].detach().cpu())
        gt_motion3d_frames.append(output["motion_gt_3d"].detach().cpu())
    
    motion2d = th.cat(motion2d_frames, dim=0)  # (T, J, 2)
    motion3d = th.cat(motion3d_frames, dim=0)  # (T, J, 3)
    gt_motion2d = th.cat(gt_motion2d_frames, dim=0)  # (T, J, 2)
    gt_motion3d = th.cat(gt_motion3d_frames, dim=0)  # (T, J, 3)
    joint_names = output["joint_names"]
    bones = output["bones"]
    edges = [[joint_names.index(b[0]), joint_names.index(b[1])] for b in bones]

    # Save the output to a .npz file
    model_name = os.path.basename(args.ckpt).replace(".pth", "")
    os.makedirs(os.path.join(args.output_path, model_name, args.motion_name), exist_ok=True)

    def _nonconflict_path(base: str) -> str:
        """Return base if it doesn't exist, otherwise base_1, base_2, …"""
        if not os.path.exists(base):
            return base
        stem, ext = os.path.splitext(base)
        i = 1
        while os.path.exists(f"{stem}_{i}{ext}"):
            i += 1
        return f"{stem}_{i}{ext}"

    output_file = _nonconflict_path(os.path.join(args.output_path, model_name, args.motion_name, f"res.npz"))
    np.savez(output_file, {
        "input_video": all_video_frames, # list of PIL images
        "motion_pred_2d": motion2d.numpy(),
        "motion_pred_3d": motion3d.numpy(),
        "motion_gt_2d": gt_motion2d.numpy(),
        "motion_gt_3d": gt_motion3d.numpy(),
        "joint_names": np.array(joint_names),
        "bones": np.array(bones),
        "edges": np.array(edges),
        "motion_name": args.motion_name,
        "ckpt": args.ckpt,
        "dit_features_path": args.dit_features_path
    })