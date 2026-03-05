import torch as th
from lightning.pytorch.utilities import rank_zero_only
import wandb
import lightning as L
from model_utils import unproject_torch, map_to_joint, unpatchify
import numpy as np
from dataset import DitFeaturesDataset
from model import JointVAE38, Head
from diffsynth.diffusion.vis import MultiSkeleton2D3DAnimator
import glob, os, plotly, argparse

class TrainOnDiTFeatures(L.LightningModule):
    def __init__(
            self, 
            dim, 
            out_dim, 
            patch_size, 
            J, 
            out_J_chn, 
            preferred_dit_block_id,
            log_dir,
            vis_steps,
            save_steps,
            logger,
            predict_motion_dt=False,
            eps=1e-8, 
            num_res_blocks=0, 
            lr=1e-4,
            ):
        super().__init__()
        self.head = Head(dim=dim, out_dim=out_dim, patch_size=patch_size, eps=eps).train()
        self.joint_vae = JointVAE38(J=J, out_J_chn=out_J_chn, z_dim=48, num_res_blocks=num_res_blocks).train()
        self.joint_head = th.nn.Conv3d(3, J * out_J_chn, 3, padding=1).train()
        self.lr = lr
        self.preferred_dit_block_id = preferred_dit_block_id
        self.J = J
        self.out_j_chn = out_J_chn
        self.vis_steps = vis_steps
        self.log_dir = log_dir
        self.wandb_logger = logger
        self.save_steps = save_steps
        self.predict_motion_dt = predict_motion_dt
        
        self.make_parameters_trainable(self.head)
        self.make_parameters_trainable(self.joint_vae)
        self.make_parameters_trainable(self.joint_head)
        
        self._last_plot_data = None  # stores latest batch outputs for plotting
        self._val_loss_dict = {"loss": [], "loss_3d": [], "loss_2d": [], "loss_depth": []}  # accumulate val losses for averaging at epoch end
        self._val_last_plot_data = []
        self.val_plot_max_batches = 10

    def make_parameters_trainable(self, module):
        for param in module.parameters():
            param.requires_grad = True

    def configure_optimizers(self):
        optimizer = th.optim.AdamW(list(self.head.parameters()) + list(self.joint_vae.parameters()), lr=self.lr)
        return optimizer

    
    def validation_step(self, batch, batch_idx):
        if self.global_rank != 0:
            return
        with th.no_grad():
            pred_dict = self.forward_pass(batch, batch_idx)
            loss_dict, acc_dict, pred_dict = self.compute_loss(pred_dict, batch)
            self._val_loss_dict['loss'].append(loss_dict['loss'].item())
            self._val_loss_dict['loss_3d'].append(loss_dict['loss_3d'].item())
            self._val_loss_dict['loss_2d'].append(loss_dict['loss_2d'].item())
            self._val_loss_dict['loss_depth'].append(loss_dict['loss_depth'].item())
        if len(self._val_last_plot_data) < self.val_plot_max_batches:
            self._val_last_plot_data.append(pred_dict)

    @rank_zero_only
    def on_validation_epoch_end(self):
        self.log_dict(
            {f"val/{k}": sum(v) / len(v) for k, v in self._val_loss_dict.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        if len(self._val_last_plot_data) > 0:
            os.makedirs(os.path.join(self.log_dir, "vis"), exist_ok=True)
            for idx, d in enumerate(self._val_last_plot_data):
                joint_names = d["joint_names"]
                edges = [[joint_names.index(b[0]), joint_names.index(b[1])] for b in d["bones"]]
                anim = MultiSkeleton2D3DAnimator(fps=30, title=f"Val Motions (batch {idx})", y_axis_down=True)
                anim.add_sequence(d["motion_gt_3d"], K2=d["motion_gt_2d"], edges=edges, color="blue", name="Ground Truth")
                anim.add_sequence(d["motion_pred_3d"], K2=d["motion_pred_2d"], edges=edges, color="red", name="Prediction")
                save_path = os.path.join(
                    self.log_dir,
                    "vis",
                    f"val_motion_step_{self.global_step}_epoch_{self.current_epoch}_batch_{idx}.html"
                )
                plotly.offline.plot(anim.fig, filename=save_path, auto_open=False)
                if self.wandb_logger is not None:
                    with open(save_path, "r", encoding="utf-8") as f:
                        wandb.log({
                            f"val/motion_batch_{idx}": wandb.Html(f.read()),
                            "step": self.global_step,
                            "epoch": self.current_epoch,
                        })
        # Reset for next val run
        self._val_last_plot_data = []
        self._val_loss_dict = {"loss": [], "loss_3d": [], "loss_2d": []}
    
    def forward_pass(self, batch, batch_idx):
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
    
    def compute_loss(self, pred_dict, batch):
        
        inputs = batch
        
        # Ground truth
        inputs["joints_3d"] = inputs["joints_3d"].squeeze(0)
        inputs["joints_2d"] = inputs["joints_2d"].squeeze(0)
        j3d_gt = inputs["joints_3d"]  # 1, T, J, 3 -> T, J, 3 (xyz)
        j2d_gt = inputs["joints_2d"][..., :2]  # 1, T, J, 3 (uvd) -> T, J, 2 (uv)
        jd_gt = inputs["joints_2d"][..., 2:3] # depth gt
        fx, fy, cx, cy = inputs["cams_intr"].squeeze(0) # (4)
        E_bl = inputs["cams_extr"].squeeze(0)  # (T, 4, 4)
        org_h = cy * 2.0 + 1
        org_w = cx * 2.0 + 1
        j2d_gt[..., 0] = j2d_gt[..., 0] / (org_w - 1)     # normalize to [0,1]
        j2d_gt[..., 1] = j2d_gt[..., 1] / (org_h - 1)   # normalize to [0,1]
        mask = th.logical_and(j2d_gt[..., :2] >= 0.0, j2d_gt[..., :2] <= 1.0).all(dim=-1)   # .all(dim=-1) to ensure both u and v are valid (squeezed last dimension)
        assert j3d_gt.shape[0] == j2d_gt.shape[0], "Number of frames mismatch between 3D and 2D joints."
        assert j3d_gt.shape[1] == j2d_gt.shape[1], "Number of joints mismatch between 3D and 2D joints."
        pred_dict["motion_gt_3d"] = j3d_gt
        pred_dict["motion_gt_2d"] = j2d_gt
        
        # Prediction
        j2d_pred = pred_dict["motion_pred_2d"]  # T, J, 2
        j3d_pred = pred_dict["motion_pred_3d"]  # T, J, 3
        jd_pred = pred_dict["motion_pred_d"]  # T, J, 1
        
        assert j3d_pred.shape == j3d_gt.shape, f"j3d_pred shape {j3d_pred.shape} does not match training_target shape {training_target_3d.shape}"
        assert j2d_pred.shape == j2d_gt.shape, f"j2d_pred shape {j2d_pred.shape} does not match gt_j2d shape {j2d_gt.shape}"

        # Loss
        loss_3d = th.nn.functional.mse_loss(j3d_pred.float(), j3d_gt.float()) * mask.unsqueeze(-1).float()  # zero out loss for joints that are out of frame
        loss_3d = loss_3d.sum() / (mask.float().sum() + 1e-8)  # average over valid joints only
        loss_2d = th.nn.functional.mse_loss(j2d_pred.float(), j2d_gt.float()) * mask.unsqueeze(-1).float()  # zero out loss for joints that are out of frame
        loss_2d = loss_2d.sum() / (mask.float().sum() + 1e-8)
        loss_depth = th.nn.functional.mse_loss(jd_pred.float(), jd_gt.float()) * mask.unsqueeze(-1).float()  # zero out loss for joints that are out of frame
        loss_depth = loss_depth.sum() / (mask.float().sum() + 1e-8)
        loss = loss_3d + loss_2d * 1000.0 + loss_depth
        # Accuracy metrics (for monitoring only, not used in loss)
        rmse_3d = th.sqrt(th.nn.functional.mse_loss(j3d_pred.float(), j3d_gt.float(), reduction='none').mean(dim=-1))  # T, J
        rmse_2d = th.sqrt(th.nn.functional.mse_loss(j2d_pred.float(), j2d_gt.float(), reduction='none').mean(dim=-1))  # T, J
        rmse_depth = th.sqrt(th.nn.functional.mse_loss(jd_pred.float(), jd_gt.float(), reduction='none').mean(dim=-1))  # T, J

        loss_dict = {"loss": loss, "loss_3d": loss_3d, "loss_2d": loss_2d, "loss_depth": loss_depth}
        acc_dict = {"rmse_3d": rmse_3d.mean().item(), "rmse_2d": rmse_2d.mean().item(), "rmse_depth": rmse_depth.mean().item()}
        
        # Detach for visualization and logging to avoid memory leak
        for k, v in pred_dict.items():
            if isinstance(v, th.Tensor):
                pred_dict[k] = v.detach().cpu().numpy()
        
        return loss_dict, acc_dict, pred_dict
        

    def training_step(self, batch, batch_idx):
        """
        batch contains the following keys:
        - 'dit_features': pre-extracted DiT features, shape (1, #dit_blocks, 1, #tokens, C)
        - 'joints_3d': ground truth 3D joint positions, shape (1, T, J, 3)
        - 'joints_2d': ground truth 2D joint positions, shape (1, T, J, 2)

        """
        pred_dict = self.forward_pass(batch, batch_idx)
        loss_dict, acc_dict, pred_dict = self.compute_loss(pred_dict, batch)
        self._last_plot_data = pred_dict  # store for visualization in callbacks
        self.log_dict({f"train/{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log_dict({f"train/{k}": v for k, v in acc_dict.items()}, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss_dict["loss"]    # for backward() to optimize this loss

    @rank_zero_only
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Called after every training step. Plot results every plot_every_n_steps."""
        if self.global_step % self.vis_steps == 0:
            self._plot_results(self.global_step)
        if (self.global_step) % (self.save_steps) == 0:
            self._save_model(self.global_step)

    @rank_zero_only
    def _plot_results(self, step):
        if self._last_plot_data is None:
            return
        plot_data = self._last_plot_data
        motion_pred_3d = plot_data["motion_pred_3d"]
        motion_gt_3d = plot_data["motion_gt_3d"]
        motion_pred_2d = plot_data["motion_pred_2d"]
        motion_gt_2d = plot_data["motion_gt_2d"]

        joint_names = plot_data["joint_names"]
        bones = plot_data["bones"]
        edges = [[joint_names.index(b[0]), joint_names.index(b[1])] for b in bones]

        anim = MultiSkeleton2D3DAnimator(fps=30, title="Motions", y_axis_down=True)
        anim.add_sequence(motion_gt_3d, K2=motion_gt_2d,edges=edges, color="blue", name="Ground Truth")
        anim.add_sequence(motion_pred_3d, K2=motion_pred_2d, edges=edges, color="red",  name="Prediction")
        # Save to html
        save_path = os.path.join(self.log_dir, "vis", f"train_motion_step_{step}.html")
        plotly.offline.plot(anim.fig, filename=save_path, auto_open=False)
        # Log html to wandb
        wandb.log({"train/motion": wandb.Html(open(save_path)), "step": step, "epoch": self.current_epoch})

    @rank_zero_only
    def _save_model(self, step):
        save_path = os.path.join(self.log_dir, "ckpt", f"model_step_{step}.pth")
        th.save(self.state_dict(), save_path)



