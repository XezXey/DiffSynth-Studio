import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dit_features_path", type=str, required=True, help="Path to the DiT features files.")
    parser.add_argument("--val_dit_features_path", default=None, help="Path to the DiT features files.")
    parser.add_argument("--J", type=int, default=25, help="Number of joints.")
    parser.add_argument("--out_J_chn", type=int, default=2, help="Output channels for each joint.")
    parser.add_argument("--preferred_dit_block_id", type=int, default=-1, help="Preferred DiT block ID.")
    parser.add_argument("--dim_mult", nargs="+", type=float, default=[1, 2, 4, 4], help="Dimension multiplier for each layer in the JointVAE38. Should be a list of float.")
    parser.add_argument("--n_gpus", type=int, default=1, help="Number of GPUs to use for training.")
    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for training.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs for training.")
    # Saving and logging
    parser.add_argument("--vis_steps", type=int, default=100, help="Visualization every n steps.")
    parser.add_argument("--save_steps", type=int, default=100, help="Save model every n steps.")
    parser.add_argument("--val_steps", type=int, default=100, help="Validation every n steps.")
    parser.add_argument("--log_steps", type=int, default=10, help="Log training info every n steps.")
    parser.add_argument("--use_wandb", action="store_true", default=False, help="Whether to use wandb for logging.")
    parser.add_argument("--wandb_save_name", type=str, default="train_on_prep_dit_features", help="Name for the wandb run.")
    parser.add_argument("--output_path", type=str, default="./output", help="Path to save the trained model and logs.")
    parser.add_argument("--limit_val_batches", type=float, default=1.0, help="Fraction of val batches to use per validation epoch (e.g. 0.005 for ~0.5%%). Set to 1.0 for full val set.")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=5, help="Run validation every N epochs.")
    parser.add_argument("--force_check_val_loop", action="store_true", default=False, help="Force validation loop to run every epoch for sanity check.")
    parser.add_argument("--overfit_single_batch", action="store_true", default=False, help="Overfit on a single batch to verify code correctness.")
    parser.add_argument("--predict_motion_dt", action="store_true", default=False, help="Whether to predict motion delta (current frame to next frame) instead of absolute motion.")
    # Losses
    parser.add_argument("--loss_type", type=str, default="all", help="Type of loss to use. Comma-separated if multiple. Options: recon, kl, all.")
    args = parser.parse_args()
    args.loss_type = parse_loss_type(args.loss_type)
    return args

def parse_loss_type(loss_type_str):
    available_loss_types = ["depth", "3d", "2d"]
    loss_type_str = loss_type_str.lower()
    if loss_type_str == "all":
        return ["depth", "3d", "2d"]
    else:
        loss_types = [s.strip() for s in loss_type_str.split(",")]
        for lt in loss_types:
            if lt not in available_loss_types:
                raise ValueError(f"Invalid loss type: {lt}. Available options are: {available_loss_types} or 'all'.")
        return loss_types
    
    