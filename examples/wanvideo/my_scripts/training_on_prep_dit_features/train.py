import torch as th
th.set_float32_matmul_precision("high")
import lightning as L
from trainer import TrainOnDiTFeatures
import os
import glob
import argparse
from dataset import DitFeaturesDataset, DitFeaturesByMotionNameDataset, MotionValidationCallback
from lightning.pytorch.loggers import WandbLogger
from mylogger.logger import init_logger

logger = init_logger("train_on_prep_dit_features.log")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dit_features_path", type=str, required=True, help="Path to the DiT features files.")
    parser.add_argument("--val_dit_features_path", default=None, help="Path to the DiT features files.")
    parser.add_argument("--J", type=int, default=25, help="Number of joints.")
    parser.add_argument("--out_J_chn", type=int, default=2, help="Output channels for each joint.")
    parser.add_argument("--preferred_dit_block_id", type=int, default=-1, help="Preferred DiT block ID.")
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
    args = parser.parse_args()

    dit_features_path_list = glob.glob(f"{args.train_dit_features_path}/*.pth")
    train_dataset = DitFeaturesDataset(dit_features_path_list, preferred_dit_block_id=args.preferred_dit_block_id)
    train_dataloader = th.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=train_dataset.collate_fn_, persistent_workers=True)

    sample_dat = next(iter(train_dataloader))
    dim = sample_dat["dim"].item()
    out_dim = sample_dat["out_dim"].item()
    patch_size = [i.item() for i in sample_dat["patch_size"]]
    grid_size = [i.item() for i in sample_dat["grid_size"]]

    logger.warning("Parameters loaded from sample data:")
    logger.info(f"dim: {dim}")
    logger.info(f"out_dim: {out_dim}")
    logger.info(f"patch_size: {patch_size}")
    logger.info(f"grid_size: {grid_size}")

    logger.warning("Training parameters:")
    logger.info(f"learning_rate: {args.learning_rate}")
    logger.info(f"num_epochs: {args.num_epochs}")
    logger.info(f"preferred_dit_block_id: {args.preferred_dit_block_id}")
    logger.info(f"vis_steps: {args.vis_steps}")
    logger.info(f"save_steps: {args.save_steps}")

    logger.warning("Validation parameters:")
    logger.info(f"limit_val_batches: {args.limit_val_batches}")
    logger.info(f"check_val_every_n_epoch: {args.check_val_every_n_epoch}")

    logger.warning("Motion parameters:")
    logger.info(f"J: {args.J}")
    logger.info(f"out_J_chn: {args.out_J_chn}")
    logger.info(f"predict_motion_dt: {args.predict_motion_dt}")

    logger.warning("Dataset:")
    logger.info(f"train_dit_features_path: {args.train_dit_features_path}")
    logger.info(f"val_dit_features_path: {args.val_dit_features_path}")

    logger.info(f"Loaded {len(dit_features_path_list)} training samples from {args.train_dit_features_path} with preferred DiT block ID {train_dataset.preferred_dit_block_id}")
    if args.val_dit_features_path is not None:
        val_dit_features_path_list = glob.glob(f"{args.val_dit_features_path}/*.pth")
        val_dataset = DitFeaturesDataset(val_dit_features_path_list, preferred_dit_block_id=args.preferred_dit_block_id)
        val_dataloader = th.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=val_dataset.collate_fn_, persistent_workers=True)
        logger.info(f"Loaded {len(val_dit_features_path_list)} validation samples from {args.val_dit_features_path} with preferred DiT block ID {val_dataset.preferred_dit_block_id}")
    else:
        val_dataloader = None
        args.limit_val_batches = 0.0

    # Create output directories
    os.makedirs(args.output_path + "/wandb", exist_ok=True)
    os.makedirs(args.output_path + "/vis", exist_ok=True)
    os.makedirs(args.output_path + "/ckpt", exist_ok=True)

    if args.use_wandb:
        wandb_logger = WandbLogger(
            project="SkelAg",
            name=args.wandb_save_name,
            save_dir=args.output_path + "/wandb",
            entity="xezxey",
            config={
                "learning_rate": args.learning_rate,
                "epochs": args.num_epochs,
                "output_path": args.output_path,
                "wandb_save_name": args.wandb_save_name,
                "train_dit_features": args.train_dit_features_path,
                "val_dit_features": args.val_dit_features_path,
            },
        )
    else:
        logger.warning("Not using wandb logger.")
        wandb_logger = None

    model = TrainOnDiTFeatures(
        dim=dim, 
        out_dim=out_dim, 
        patch_size=patch_size, 
        J=args.J,
        out_J_chn=args.out_J_chn, 
        preferred_dit_block_id=args.preferred_dit_block_id, 
        lr=args.learning_rate,
        log_dir=args.output_path,
        vis_steps=args.vis_steps,
        save_steps=args.save_steps,
        val_steps=args.val_steps,
        val_dit_features_path=args.val_dit_features_path,
        logger=wandb_logger,
        predict_motion_dt=args.predict_motion_dt,
    )

    trainer_kwargs = dict(
        log_every_n_steps=args.log_steps,
    )

    if args.overfit_single_batch:
        logger.warning("Overfitting on a single batch for sanity check...")
        overfit_kwargs = dict(overfit_batches=1, limit_train_batches=1)
    trainer_kwargs.update(overfit_kwargs)

    trainer = L.Trainer(
        max_epochs=args.num_epochs,
        accelerator="cuda",
        devices=args.n_gpus,
        logger=wandb_logger,
        default_root_dir=args.output_path + "/lightning_logs",
        **trainer_kwargs,
    )
    trainer.fit(model, train_dataloader)
