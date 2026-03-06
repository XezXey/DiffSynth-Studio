import torch as th
from lightning.pytorch.utilities import rank_zero_only
import wandb
import lightning as L
from model_utils import unproject_torch, map_to_joint, unpatchify
import numpy as np
from dataset import DitFeaturesDataset, DitFeaturesByMotionNameDataset
from model import JointVAE38, Head
from diffsynth.diffusion.vis import MultiSkeleton2D3DAnimator
import glob, os, plotly, argparse
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.console import Console

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
            val_steps,
            logger,
            predict_motion_dt=False,
            eps=1e-8, 
            num_res_blocks=0, 
            lr=1e-4,
            val_dit_features_path=None,
            ):
        super().__init__()
        self.head = Head(dim=dim, out_dim=out_dim, patch_size=patch_size, eps=eps).train()
        self.joint_vae = JointVAE38(J=J, out_J_chn=out_J_chn, z_dim=48, num_res_blocks=num_res_blocks).train()
        self.joint_head = th.nn.Conv3d(3, J * out_J_chn, 3, padding=1).train()
        self.lr = lr
        self.preferred_dit_block_id = preferred_dit_block_id
        self.J = J
        self.out_j_chn = out_J_chn
        self.save_steps = save_steps
        self.vis_steps = vis_steps
        self.val_steps = val_steps
        self.log_dir = log_dir
        self.wandb_logger = logger
        self.predict_motion_dt = predict_motion_dt
        
        self.make_parameters_trainable(self.head)
        self.make_parameters_trainable(self.joint_vae)
        self.make_parameters_trainable(self.joint_head)
        
        self.val_dit_features_path = val_dit_features_path
        if self.val_dit_features_path is not None:
            self.val_dataset = DitFeaturesByMotionNameDataset(
                dit_features_path_list=glob.glob(f"{self.val_dit_features_path}/*.pth"),
                preferred_dit_block_id=self.preferred_dit_block_id
            )
        else:
            self.val_dataset = None

    def make_parameters_trainable(self, module):
        for param in module.parameters():
            param.requires_grad = True

    def configure_optimizers(self):
        optimizer = th.optim.AdamW(list(self.head.parameters()) + list(self.joint_vae.parameters()), lr=self.lr)
        return optimizer

    @rank_zero_only
    def validate(self):
        # Run the validation on rank 0 only
        self.eval()
        all_loss_dict = {}
        all_acc_dict = {}
        all_pred_dict = {}
        all_gt_dict = {}
        all_motion_names = []
        n_motion = 0
        val_sequences = list(self.val_dataset.iter_inference_sequences())
        _console = Console(stderr=True)
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(), transient=True, console=_console) as progress:
            task = progress.add_task("Validating - ", total=len(val_sequences))
            batch_task = progress.add_task(" Chunk-id", total=None, visible=False)
            for seq in val_sequences: # iterate over character-motion sequences
                progress.update(task, description=f"Validating - [cyan]{seq['character']}:{seq['motion_name']}[/]")
                ds = DitFeaturesDataset(
                    seq["paths"], preferred_dit_block_id=self.preferred_dit_block_id
                )
                loader = th.utils.data.DataLoader(
                    ds, batch_size=1, collate_fn=ds.collate_fn_
                )
                n_batches = len(loader)
                progress.update(batch_task, completed=0, total=n_batches, visible=True)

                motion_pred_dict = {}
                motion_gt_dict = {}
                for batch_idx, batch in enumerate(loader):
                    with th.no_grad():
                        batch = {k: v.to(self.device) if isinstance(v, th.Tensor) else v for k, v in batch.items()}
                        pred_dict = self.forward_pass(batch, batch_idx)
                        progress.advance(batch_task)
                        if batch_idx == 0:
                            motion_pred_dict = pred_dict
                            motion_gt_dict = {
                                "joints_3d": batch["joints_3d"].squeeze(0), "joints_2d": batch["joints_2d"].squeeze(0), 
                                "cams_intr": batch["cams_intr"], "cams_extr": batch["cams_extr"],
                                "height": batch["height"], "width": batch["width"], "joint_names": batch["joint_names"], "bones": batch["bones"]
                            }
                        else:
                            for k in ["motion_pred_3d", "motion_pred_2d", "motion_pred_d"]:
                                motion_pred_dict[k] = th.cat([motion_pred_dict[k], pred_dict[k]], dim=0)  # concatenate along time dimension
                            motion_gt_dict["joints_3d"] = th.cat([motion_gt_dict["joints_3d"], batch["joints_3d"].squeeze(0)], dim=0)
                            motion_gt_dict["joints_2d"] = th.cat([motion_gt_dict["joints_2d"], batch["joints_2d"].squeeze(0)], dim=0)
                            motion_gt_dict["cams_extr"] = th.cat([motion_gt_dict["cams_extr"], batch["cams_extr"]], dim=1)
                all_motion_names.append(f"{seq['character']}:{seq['motion_name']}")
                loss_dict, acc_dict, pred_dict = self.compute_loss(motion_pred_dict, motion_gt_dict)
                for k, v in loss_dict.items():
                    if k not in all_loss_dict:
                        all_loss_dict[k] = []
                    all_loss_dict[k].append(v.item())
                for k, v in acc_dict.items():
                    if k not in all_acc_dict:
                        all_acc_dict[k] = []
                    all_acc_dict[k].append(v)
                for k in ["motion_pred_3d", "motion_pred_2d", "motion_pred_d"]:
                    if k not in all_pred_dict:
                        all_pred_dict[k] = []
                    all_pred_dict[k].append(pred_dict[k])
                for k in ["joints_3d", "joints_2d", "joint_names", "bones"]:
                    if k not in all_gt_dict:
                        all_gt_dict[k] = []
                    if k == "joints_2d":
                        motion_gt_dict[k] = motion_gt_dict[k][..., :2]
                    all_gt_dict[k].append(motion_gt_dict[k].cpu().numpy() if isinstance(motion_gt_dict[k], th.Tensor) else motion_gt_dict[k])
                n_motion += 1
                progress.advance(task)
                
        for k in all_loss_dict:
            all_loss_dict[k] = np.mean(all_loss_dict[k])
        for k in all_acc_dict:
            all_acc_dict[k] = np.mean(all_acc_dict[k])

        self.log_dict({f"val_metrics/{k}": v for k, v in all_loss_dict.items()}, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict({f"val_metrics/{k}": v for k, v in all_acc_dict.items()}, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for i in range(n_motion):
            joint_names = all_gt_dict["joint_names"][i]
            edges = [[joint_names.index(b[0]), joint_names.index(b[1])] for b in all_gt_dict["bones"][i]]
            anim = MultiSkeleton2D3DAnimator(fps=30, title=f"Validation (step={self.global_step}, epoch={self.current_epoch}) - {all_motion_names[i]}", y_axis_down=True)
            anim.add_sequence(all_gt_dict["joints_3d"][i], K2=all_gt_dict["joints_2d"][i], edges=edges, color="blue", name="Ground Truth")
            anim.add_sequence(all_pred_dict["motion_pred_3d"][i], K2=all_pred_dict["motion_pred_2d"][i], edges=edges, color="red", name="Prediction")
            save_path = os.path.join(
                self.log_dir,
                "vis",
                f"val_motion_{i}.html"
            )
            plotly.offline.plot(anim.fig, filename=save_path, auto_open=False)
            if self.wandb_logger is not None:
                with open(save_path, "r", encoding="utf-8") as f:
                    wandb.log({
                        f"val_motion/{all_motion_names[i]}": wandb.Html(f.read()),
                        "step": self.global_step,
                        "epoch": self.current_epoch,
                    })
        self.train()  # set back to train mode after validation

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
        self.log_dict({f"train_metrics/{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log_dict({f"train_metrics/{k}": v for k, v in acc_dict.items()}, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss_dict["loss"]    # for backward() to optimize this loss

    @rank_zero_only
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Called after every training step. Plot results every plot_every_n_steps."""
        if self.global_step % self.vis_steps == 0:
            self._plot_results(self.global_step)
        if (self.global_step) % (self.save_steps) == 0:
            self._save_model(self.global_step)
        if (self.val_dataset is not None) and (self.global_step % self.val_steps == 0):
            self.validate()

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

        anim = MultiSkeleton2D3DAnimator(fps=30, title=f"Train (step={self.global_step}, epoch={self.current_epoch})", y_axis_down=True)
        anim.add_sequence(motion_gt_3d, K2=motion_gt_2d,edges=edges, color="blue", name="Ground Truth")
        anim.add_sequence(motion_pred_3d, K2=motion_pred_2d, edges=edges, color="red",  name="Prediction")
        # Save to html
        save_path = os.path.join(self.log_dir, "vis", f"train_motion.html")
        plotly.offline.plot(anim.fig, filename=save_path, auto_open=False)
        # Log html to wandb
        wandb.log({"train_motion/motion": wandb.Html(open(save_path)), "step": step, "epoch": self.current_epoch})

    @rank_zero_only
    def _save_model(self, step):
        save_path = os.path.join(self.log_dir, "ckpt", f"model_step_{step}.pth")
        th.save(self.state_dict(), save_path)



