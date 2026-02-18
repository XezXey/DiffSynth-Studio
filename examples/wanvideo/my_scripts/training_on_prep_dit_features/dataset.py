import torch as th
import glob
import numpy as np
from PIL import Image
from torch.utils.data.dataloader import default_collate

class DitFeaturesDataset(th.utils.data.Dataset):
    def __init__(self, dit_features_path_list, preferred_dit_block_id=0):
        self.dit_features_path_list = dit_features_path_list
        self.preferred_dit_block_id = preferred_dit_block_id
        self.focus_fields = ['dit_features', 'grid_size', 'dim', 'out_dim', 'patch_size', 'z_dim', 'joints_3d', 'joints_2d', 'cams_intr', 'cams_extr', 'height', 'width', 'joint_names', 'bones']
    
    def __len__(self):
        return len(self.dit_features_path_list)
    
    def __getitem__(self, idx):
        dit_feature_path = self.dit_features_path_list[idx]
        dit_features = th.load(dit_feature_path, map_location="cpu", weights_only=False)
        input_shared = dit_features[0]
        input_shared["dit_features"] = input_shared["dit_features"][self.preferred_dit_block_id]
        return input_shared

    def collate_fn_(self, batch):
        # batch: list of samples (dicts or anything)
        # If dict samples, keep PIL fields as list; collate tensors normally.
        out = {}
        keys = batch[0].keys()
        assert len(batch) == 1, "Only support batch size 1 for now."
        for k in keys:
            if k in self.focus_fields:
                values = [b[k] for b in batch]
                try: 
                    out[k] = default_collate(values)
                except TypeError:
                    out[k] = values
        return out