import torch as th
import glob
import numpy as np
from PIL import Image
from torch.utils.data.dataloader import default_collate

class DitFeaturesDataset(th.utils.data.Dataset):
    def __init__(self, dit_features_path_list):
        self.dit_features_path_list = dit_features_path_list
    
    def __len__(self):
        return len(self.dit_features_path_list)
    
    def __getitem__(self, idx):
        dit_feature_path = self.dit_features_path_list[idx]
        dit_features = th.load(dit_feature_path, map_location="cpu", weights_only=False)
        input_shared = dit_features[0]
        return input_shared


def collate_fn_(batch):
    # batch: list of samples (dicts or anything)
    # If dict samples, keep PIL fields as list; collate tensors normally.
    out = {}
    keys = batch[0].keys()
    assert len(batch) == 1, "Only support batch size 1 for now."
    for k in keys:
        values = [b[k] for b in batch]
        try: 
            out[k] = default_collate(values)
        except TypeError:
            out[k] = values
    return out