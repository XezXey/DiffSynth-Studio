import os, torch
import plotly
import wandb
from accelerate import Accelerator
from ..utils.vis.vis import MultiSkeleton3DAnimator

class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0


    def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None, name=None):
        self.num_steps += 1
        if save_steps is not None and self.num_steps % save_steps == 0:
            if name is not None:
                self.save_model(accelerator, model, f"{name}-step-{self.num_steps}.safetensors")
            else:
                self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")



    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id, name=None):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            if name is not None:
                path = os.path.join(self.output_path, f"{name}-epoch-{epoch_id}.safetensors")
            else:
                path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)


    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None, name=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            if name is not None:
                self.save_model(accelerator, model, f"{name}-step-{self.num_steps}.safetensors")
            else:
                self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)

class TrainingLogger:
    def __init__(self, training_logger, log_dir):
        self.training_logger = training_logger
        self.num_steps = 0
        self.log_dir = log_dir

    def on_step_end(self, accelerator: Accelerator, loss, pred_dict: dict, save_steps=None):
        self.num_steps += 1
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if save_steps is not None and self.num_steps % save_steps == 0:
                # Log loss
                self.log_loss(loss.item())
                # Log predictions
                self.log_predictions(pred_dict)

    def on_epoch_end(self, accelerator: Accelerator, loss, pred_dict: dict, save_steps=None):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            # Log loss
            self.log_loss(loss.item())
    
    def log_loss(self, loss):
        self.training_logger.log({"loss": loss})
    
    def log_predictions(self, pred_dict: dict):
        motion_pred = pred_dict['motion_pred'].detach().cpu().numpy()
        motion_gt = pred_dict['training_target'].detach().cpu().numpy()
        if motion_gt.shape[0] == 1:
            motion_gt = motion_gt[0]
        if motion_pred.shape[0] == 1:
            motion_pred = motion_pred[0]
        print(motion_pred.shape)
        print(motion_gt.shape)
        joint_names = pred_dict['joint_names']
        bones = pred_dict['bones']
        edges = [[joint_names.index(b[0]), joint_names.index(b[1])] for b in bones]

        anim = MultiSkeleton3DAnimator(fps=30, title="Motions")
        anim.add_sequence(motion_gt, edges=edges, color="blue", name="Ground Truth")
        anim.add_sequence(motion_pred, edges=edges, color="red",  name="Prediction")
        # Save to html
        save_path = os.path.join(self.log_dir, f"motion_pred_step_{self.num_steps}.html")
        plotly.offline.plot(anim.fig, filename=save_path, auto_open=False)
        # Log html to wandb
        self.training_logger.log({"motion_prediction": wandb.Html(open(save_path))})


        

