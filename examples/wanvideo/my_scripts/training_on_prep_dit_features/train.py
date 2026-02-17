import torch as th
import lightning as L
from trainer import TrainOnDiTFeatures
import numpy as np
import glob
from dataset import DitFeaturesDataset
from model import JointVAE38, Head
import argparse
from einops import rearrange
from lightning.pytorch.loggers import WandbLogger

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dit_features_path", type=str, required=True, help="Path to the DiT features files.")
    parser.add_argument("--J", type=int, default=25, help="Number of joints.")
    parser.add_argument("--preferred_dit_block_id", type=int, default=-1, help="Preferred DiT block ID.")
    parser.add_argument("--n_gpus", type=int, default=1, help="Number of GPUs to use for training.")
    args = parser.parse_args()

    dit_features_path_list = glob.glob(f"{args.dit_features_path}/*.pth")
    dataset = DitFeaturesDataset(dit_features_path_list)
    dataloader = th.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1, collate_fn=dataset.collate_fn_)

    sample_dat = next(iter(dataloader))
    dim = sample_dat["dim"].item()
    out_dim = sample_dat["out_dim"].item()
    patch_size = [i.item() for i in sample_dat["patch_size"]]
    grid_size = [i.item() for i in sample_dat["grid_size"]]

    print("Parameters loaded from sample data:")
    print("dim:", dim)
    print("out_dim:", out_dim)
    print("patch_size:", patch_size)
    print("grid_size:", grid_size)

    preferred_dit_block_id = args.preferred_dit_block_id
    if preferred_dit_block_id == -1:
        preferred_dit_block_id = sample_dat["dit_features"].shape[1] - 1    # last block if use -1
    if preferred_dit_block_id < 0 or preferred_dit_block_id >= sample_dat["dit_features"].shape[1]:
        raise ValueError(f"Exceeds preferred_dit_block_id range: 0 ~ {sample_dat['dit_features'].shape[1]-1}")
    dataset.preferred_dit_block_id = preferred_dit_block_id
    
    logger = WandbLogger(project="TrainOnDiTFeatures", name="train_on_prep_dit_features")
    model = TrainOnDiTFeatures(
        dim=dim, 
        out_dim=out_dim, 
        patch_size=patch_size, 
        J=args.J, 
        out_J_chn=2, 
        preferred_dit_block_id=preferred_dit_block_id, 
        lr=1e-5
    )

    trainer = L.Trainer(
        max_epochs=10, 
        accelerator="cuda", 
        devices=args.n_gpus, 
        log_every_n_steps=10, 
        logger=logger,
    )
    trainer.fit(model, dataloader)
