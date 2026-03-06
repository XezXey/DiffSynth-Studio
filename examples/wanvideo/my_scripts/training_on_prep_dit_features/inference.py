import numpy as np
import torch as th
from model_utils import unpatchify, map_to_joint, unproject_torch
from dataset import DitFeaturesDataset, DitFeaturesByMotionNameDataset
from dev.experiment.vae_decoder.vae_decoder import Head
import glob, os, plotly, argparse
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.console import Console
import lightning as L
from model import JointVAE38, Head
from mylogger.logger import init_logger
logger = init_logger("inference.log")

parser = argparse.ArgumentParser()
# Inference related arguments
parser.add_argument('--ckpt', type=str, required=True)
parser.add_argument('--dit_features_path', type=str, required=True)
parser.add_argument('--output_path', type=str, required=True)
parser.add_argument('--gpu_id', type=int, default=0)
# Dataset related arguments
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--character_name', type=str, default=None)
parser.add_argument('--motion_name', type=str, default=None)
# Model related arguments
parser.add_argument("--J", type=int, default=25, help="Number of joints.")
parser.add_argument("--out_J_chn", type=int, default=2, help="Output channels for each joint.")
parser.add_argument("--predict_motion_dt", action='store_true', help="Whether the model predicts motion delta (True) or absolute motion (False). If True, the depth output will be treated as motion delta and converted to absolute motion by cumulative summation over time.")
parser.add_argument("--predict_motion_ar_depth", action='store_true', help="Whether the model predicts motion delta (True) or absolute motion (False). If True, the depth output will be treated as motion delta and converted to absolute motion by cumulative summation over time.")
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

    def forward_pass(self, batch, start_depth=None):
        inputs = batch
        dit_features = inputs["dit_features"].squeeze(0)  # B, C, H, W
        inp = dit_features.type(th.float32)  # 1, #tokens, C
        grid_size = inputs["grid_size"]
        patch_size = inputs["patch_size"]

        out_head = self.head(inp)  # 1, out_dim, H, W
        out_unpatched = unpatchify(out_head, grid_size, patch_size)  # 1, out_dim, T, H, W
        out_decoded = self.joint_vae.decode(out_unpatched, device='cuda')  # 1, J*3, T, 1, 1
        out_joints_map = self.joint_head(out_decoded)  # 1, J*2, T, 1, 1

        pixel_coords, depth = map_to_joint(self.J, out_joints_map)  # pixel_coords: (1, J, T, 2); depth: (1, J, T, 1)
        
        fx, fy, cx, cy = inputs["cams_intr"].squeeze(0) # (4)
        E_bl = inputs["cams_extr"].squeeze(0)  # (T, 4, 4)
        org_h = cy * 2.0 + 1
        org_w = cx * 2.0 + 1

        pred_u = pixel_coords[..., 0] * (org_w - 1)    # B, J, T
        pred_v = pixel_coords[..., 1] * (org_h - 1)    # B, J, T
        pred_d = depth[..., 0]  # B, J, T
        
        if self.predict_motion_dt:
            if start_depth is not None:
                pred_d[..., 0:1] = start_depth
            pred_d = th.cumsum(pred_d, dim=2)  # convert from motion delta to absolute motion, assume first timestep would be the absolute motion
        else: 
            pred_d = pred_d
        
        j2d_pred = th.stack([pred_u / (org_w - 1), pred_v / (org_h - 1)], dim=-1).squeeze(0).permute(1, 0, 2)  # B, J, T -> T, J, 2
        j3d_pred = unproject_torch(fx, fy, cx, cy, E_bl, th.stack([pred_u, pred_v, pred_d], dim=-1).squeeze(0).permute(1, 0, 2))

        pred_dict = {
            "motion_pred_3d": j3d_pred,
            "motion_pred_2d": j2d_pred,
            "motion_pred_d": pred_d.permute(2, 1, 0),  # T, J, 1
            "joint_names": inputs["joint_names"],
            "bones": inputs["bones"]
        }

        return pred_dict

    def get_params_stats(self):
        # Compute mean/std of all parameters for debugging
        params = list(self.head.parameters()) + list(self.joint_vae.parameters()) + list(self.joint_head.parameters())
        all_params = th.cat([p.detach().cpu().flatten() for p in params])
        mean = th.mean(all_params).item()
        std = th.std(all_params).item()
        return mean, std
        
if __name__ == "__main__":

    val_dataset = DitFeaturesByMotionNameDataset(
        dit_features_path_list=glob.glob(f"{args.dit_features_path}/*.pth"),
        preferred_dit_block_id=args.preferred_dit_block_id
    )
    val_sequences = list(val_dataset.iter_inference_sequences(character=args.character_name, motion_name=args.motion_name, cam_id=args.cam_id))
    ds = DitFeaturesDataset(
        val_sequences[0]["paths"], preferred_dit_block_id=args.preferred_dit_block_id
    )
    loader = th.utils.data.DataLoader(
        ds, batch_size=1, collate_fn=ds.collate_fn_
    )
    sample_dat = next(iter(loader))
    dim = sample_dat["dim"].item()
    out_dim = sample_dat["out_dim"].item()
    patch_size = [i.item() for i in sample_dat["patch_size"]]
    grid_size = [i.item() for i in sample_dat["grid_size"]]
    device = 'cuda'

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

    n_motion = 0
    _console = Console(stderr=True)
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(), transient=True, console=_console) as progress:
        task = progress.add_task("Inference - ", total=len(val_sequences))
        batch_task = progress.add_task(" Chunk-id", total=None, visible=False)
        for seq in val_sequences: # iterate over character-motion sequences
            progress.update(task, description=f"Validating - [cyan]{seq['character']}:{seq['motion_name']}[/]")
            ds = DitFeaturesDataset(
                seq["paths"], preferred_dit_block_id=args.preferred_dit_block_id
            )
            loader = th.utils.data.DataLoader(
                ds, batch_size=1, collate_fn=ds.collate_fn_
            )
            n_batches = len(loader)
            progress.update(batch_task, completed=0, total=n_batches, visible=True)

            all_video_frames = []
            motion2d_frames = []
            motion3d_frames = []
            gt_motion2d_frames = []
            gt_motion3d_frames = []
            for batch_idx, batch in enumerate(loader):
                with th.no_grad():
                    batch = {k: v.to(device) if isinstance(v, th.Tensor) else v for k, v in batch.items()}
                    input_video = [np.array(frame) for frame in batch["input_video"]]  # (T, H, W, 3)
                    all_video_frames.extend(input_video)
                    if args.predict_motion_ar_depth:
                        if batch_idx == 0: 
                            # First frame = takes from input (required shape = B, J, T)
                            start_depth = batch['joints_2d'][..., 2:3]  # B, T, J, 1
                            start_depth = start_depth.squeeze(-1).permute(0, 2, 1)  # B, J, T
                            start_depth = start_depth[..., 0:1] # 1, J, 1
                            # print(start_depth.shape)
                        else:
                            # Not first frame = takes from previous prediction
                            start_depth = pred_dict["motion_pred_d"]    # T, J, 1
                            start_depth = start_depth[-1:, ...] # 1, J, 1
                            # print(start_depth.shape)
                            # exit()
                    else:
                        start_depth = None
                    pred_dict = model.forward_pass(batch, start_depth)
                    motion2d_frames.append(pred_dict["motion_pred_2d"].detach().cpu())
                    motion3d_frames.append(pred_dict["motion_pred_3d"].detach().cpu())

                    fx, fy, cx, cy = batch["cams_intr"].squeeze(0) # (4)
                    org_h = cy * 2.0 + 1
                    org_w = cx * 2.0 + 1
                    gt_motion2d = batch["joints_2d"].squeeze(0).to(device) # (J, T, 2)
                    gt_motion2d[..., 0] = gt_motion2d[..., 0] / (org_w - 1)
                    gt_motion2d[..., 1] = gt_motion2d[..., 1] / (org_h - 1)
                    gt_motion2d_frames.append(gt_motion2d.cpu())
                    gt_motion3d_frames.append(batch["joints_3d"].squeeze(0).detach().cpu())
                    progress.advance(batch_task)
            n_motion += 1
            progress.advance(task)

            motion2d = th.cat(motion2d_frames, dim=0)  # (T, J, 2)
            motion3d = th.cat(motion3d_frames, dim=0)  # (T, J, 3)
            gt_motion2d = th.cat(gt_motion2d_frames, dim=0)  # (T, J, 2)
            gt_motion3d = th.cat(gt_motion3d_frames, dim=0)  # (T, J, 3)
            joint_names = batch["joint_names"]
            bones = batch["bones"]
            edges = [[joint_names.index(b[0]), joint_names.index(b[1])] for b in bones]

            # Save the output to a .npz file
            ckpt_name = os.path.basename(args.ckpt).replace(".pth", "")
            model_name = os.path.normpath(args.ckpt).split(os.sep)[-3]
            
            os.makedirs(os.path.join(args.output_path, model_name, ckpt_name, seq["character"], seq["motion_name"]), exist_ok=True)

            def _nonconflict_path(base: str) -> str:
                """Return base if it doesn't exist, otherwise base_1, base_2, …"""
                if not os.path.exists(base):
                    return base
                stem, ext = os.path.splitext(base)
                i = 1
                while os.path.exists(f"{stem}_{i}{ext}"):
                    i += 1
                return f"{stem}_{i}{ext}"

            all_video_frames = np.concatenate(all_video_frames, axis=0)  # (T, H, W, 3)
            output_file = _nonconflict_path(os.path.join(args.output_path, model_name, ckpt_name, seq["character"], seq["motion_name"], f"res.npz"))
            save_dict = {
                "input_video": all_video_frames,
                "motion_pred_2d": motion2d.numpy(),
                "motion_pred_3d": motion3d.numpy(),
                "motion_gt_2d": gt_motion2d.numpy(),
                "motion_gt_3d": gt_motion3d.numpy(),
                "joint_names": np.array(joint_names),
                "bones": np.array(bones),
                "edges": np.array(edges),
                "motion_name": seq["motion_name"],
                "ckpt": args.ckpt,
                "dit_features_path": args.dit_features_path
            }
            np.savez(output_file, **save_dict)