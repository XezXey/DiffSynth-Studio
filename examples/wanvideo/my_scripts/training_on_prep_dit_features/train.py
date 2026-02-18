import torch as th
th.set_float32_matmul_precision("high")
import lightning as L
from trainer import TrainOnDiTFeatures
import os
import numpy as np
import glob
from dataset import DitFeaturesDataset
from model import JointVAE38, Head
import argparse
from einops import rearrange
from lightning.pytorch.loggers import WandbLogger
import wandb
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
    parser.add_argument("--log_steps", type=int, default=10, help="Log training info every n steps.")
    parser.add_argument("--use_wandb", action="store_true", default=False, help="Whether to use wandb for logging.")
    parser.add_argument("--wandb_save_name", type=str, default="train_on_prep_dit_features", help="Name for the wandb run.")
    parser.add_argument("--output_path", type=str, default="./output", help="Path to save the trained model and logs.")
    parser.add_argument("--limit_val_batches", type=float, default=1.0, help="Fraction of val batches to use per validation epoch (e.g. 0.005 for ~0.5%%). Set to 1.0 for full val set.")
    args = parser.parse_args()

    dit_features_path_list = glob.glob(f"{args.train_dit_features_path}/*.pth")
    train_dataset = DitFeaturesDataset(dit_features_path_list)
    train_dataloader = th.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=train_dataset.collate_fn_)

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

    logger.warning("Motion parameters:")
    logger.info(f"J: {args.J}")
    logger.info(f"out_J_chn: {args.out_J_chn}")


    preferred_dit_block_id = args.preferred_dit_block_id
    if preferred_dit_block_id == -1:
        preferred_dit_block_id = sample_dat["dit_features"].shape[1] - 1    # last block if use -1
    if preferred_dit_block_id < 0 or preferred_dit_block_id >= sample_dat["dit_features"].shape[1]:
        raise ValueError(f"Exceeds preferred_dit_block_id range: 0 ~ {sample_dat['dit_features'].shape[1]-1}")
    train_dataset.preferred_dit_block_id = preferred_dit_block_id
    logger.info(f"Loaded {len(dit_features_path_list)} training samples from {args.train_dit_features_path} with preferred DiT block ID {preferred_dit_block_id}")
    
    if args.val_dit_features_path is not None:
        val_dit_features_path_list = glob.glob(f"{args.val_dit_features_path}/*.pth")
        val_dataset = DitFeaturesDataset(val_dit_features_path_list)
        val_dataloader = th.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=47, collate_fn=val_dataset.collate_fn_)
        logger.info(f"Loaded {len(val_dit_features_path_list)} validation samples from {args.val_dit_features_path}")
    else:
        val_dataloader = None
    val_dataset.preferred_dit_block_id = preferred_dit_block_id

    os.makedirs(args.output_path + "/wandb", exist_ok=True)
    os.makedirs(args.output_path + "/vis", exist_ok=True)
    os.makedirs(args.output_path + "/ckpt", exist_ok=True)
    if args.use_wandb:
        logger.warning("Using wandb logger...")
        wandb_run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="xezxey",
            # Set the wandb project where this run will be logged.
            project="SkelAg",
            # Name of this run
            name=args.wandb_save_name,
            # Track hyperparameters and run metadata.
            config={
                # Training info
                "learning_rate": args.learning_rate,
                "epochs": args.num_epochs,
                # Saving info
                "output_path": args.output_path,
                "wandb_save_name": args.wandb_save_name,
                # Dataset info
                "train_dit_features": args.train_dit_features_path,
                "val_dit_features": args.val_dit_features_path,
            },
            dir=args.output_path + "/wandb",
        )
        wandb_logger = WandbLogger(experiment=wandb_run)
    else:
        logger.warning("Not using wandb logger...")
        wandb_logger = None

    model = TrainOnDiTFeatures(
        dim=dim, 
        out_dim=out_dim, 
        patch_size=patch_size, 
        J=args.J,
        out_J_chn=args.out_J_chn, 
        preferred_dit_block_id=preferred_dit_block_id, 
        lr=args.learning_rate,
        log_dir=args.output_path,
        vis_steps=args.vis_steps,
        save_steps=args.save_steps,
        logger=wandb_run,
    )

    trainer = L.Trainer(
        max_epochs=args.num_epochs, 
        accelerator="cuda", 
        devices=args.n_gpus, 
        log_every_n_steps=args.log_steps, 
        logger=wandb_logger,
        check_val_every_n_epoch=1,
        limit_val_batches=0.05, #args.limit_val_batches,
        limit_train_batches=0.02,
        num_sanity_val_steps=0,   # skip val sanity check to save time, set to e.g. 2 to enable and check if val dataloader and validation step work without OOM or other errors before actual training starts
        default_root_dir=args.output_path + "/lightning_logs",
    )
    trainer.fit(model, train_dataloader, val_dataloader)
