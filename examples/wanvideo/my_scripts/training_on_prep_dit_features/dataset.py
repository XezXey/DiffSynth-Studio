import os
import torch as th
import time
import glob
import numpy as np
from PIL import Image
from collections import defaultdict
from torch.utils.data.dataloader import default_collate

class DitFeaturesDataset(th.utils.data.Dataset):
    def __init__(self, dit_features_path_list, preferred_dit_block_id=-1):
        self.dit_features_path_list = dit_features_path_list
        self.preferred_dit_block_id = preferred_dit_block_id
        self.focus_fields = ['input_video', 'dit_features', 'grid_size', 'dim', 'out_dim', 'patch_size', 'z_dim', 'joints_3d', 'joints_2d', 'cams_intr', 'cams_extr', 'height', 'width', 'joint_names', 'bones']
    
    def __len__(self):
        return len(self.dit_features_path_list)
    
    def __getitem__(self, idx):
        dit_feature_path = self.dit_features_path_list[idx]
        dit_features = th.load(dit_feature_path, map_location="cpu", weights_only=False)
        input_shared = dit_features[0]
        input_shared["dit_features"] = input_shared["dit_features"][self.preferred_dit_block_id]
        print(input_shared['motion_name'])
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

class DitFeaturesByMotionNameDataset():
    def __init__(self, dit_features_path_list, preferred_dit_block_id=-1):
        self.preferred_dit_block_id = preferred_dit_block_id
        self.focus_fields = ['input_video', 'dit_features', 'grid_size', 'dim', 'out_dim', 'patch_size', 'z_dim', 'joints_3d', 'joints_2d', 'cams_intr', 'cams_extr', 'height', 'width', 'joint_names', 'bones']
        self.motion_dict = self.motion_dict_from_path(dit_features_path_list)
        # Flat index — every record is one path; __len__ and __getitem__ use this.
        # Each record: {character, motion_name, cam_id, chunk_id, data_id, path}
        self.index = [
            {"character": char, "motion_name": mname, "cam_id": cid,
             "chunk_id": chunk_id, "data_id": entry["data_id"], "path": entry["path"]}
            for char, mname, cid, chunk_id, entry in self.iter_entries()
        ]
        self.summary()  # Print available data


    def motion_dict_from_path(self, paths):
        """
        Build a nested dict from a list of .pth file paths.

        Filename format:
            {data_id}_{character}_{motion_name}_cam-{cam_id}_render_chunk-{chunk_id}_video.pth

        Returns
        -------
        dict[str, dict[str, dict]]
            motion_dict[character][motion_name] = {
                "n_cam":   int,   # number of unique cam_ids
                "n_chunk": int,   # number of unique chunk_ids
                "cams": {
                    cam_id (int): {
                        chunk_id (int): [
                            {"data_id": int, "path": str},
                            ...   # sorted by data_id; multiple = duplicate data_ids
                        ],
                        ...       # chunk_id keys are sorted ints
                    },
                    ...           # cam_id keys are sorted ints
                }
            }

        Example access
        --------------
            info = d["michelle"]["Dancing-Twerk"]
            info["n_cam"]               # 4
            info["n_chunk"]             # 10
            info["cams"][0]             # all chunks for cam 0
            info["cams"][0][3]          # chunk 3, cam 0 → list of entries
            info["cams"][0][3][0]       # first (lowest data_id) entry
            info["cams"][0][3][0]["path"]

            # Iterate all cams × chunks:
            for cam_id, chunks in info["cams"].items():
                for chunk_id, entries in chunks.items():
                    chosen = entries[0]   # or random.choice(entries)
        """
        # raw[character][motion_name][cam_id][chunk_id] = [entries]
        raw = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(list)
                )
            )
        )

        for path in paths:
            fn      = os.path.basename(path)
            parts   = fn.split("_")
            # parts: [data_id, character, motion_name, cam-X, render, chunk-X, video.pth]
            data_id     = int(parts[0])
            character   = parts[1]
            motion_name = parts[2]
            cam_id      = int(parts[3].replace("cam-", ""))
            chunk_id    = int(parts[5].replace("chunk-", ""))

            raw[character][motion_name][cam_id][chunk_id].append({
                "data_id": data_id,
                "path":    path,
            })

        # Produce clean dict with sorted keys and metadata
        motion_dict = {}
        for character, motions in raw.items():
            motion_dict[character] = {}
            for motion_name, cams in motions.items():
                sorted_cams = {}
                all_chunk_ids = set()
                for cam_id in sorted(cams):
                    sorted_cams[cam_id] = {
                        chunk_id: sorted(cams[cam_id][chunk_id], key=lambda e: e["data_id"])
                        for chunk_id in sorted(cams[cam_id])
                    }
                    all_chunk_ids.update(cams[cam_id].keys())

                motion_dict[character][motion_name] = {
                    "n_cam":   len(sorted_cams),
                    "n_chunk": len(all_chunk_ids),
                    "cams":    sorted_cams,
                }

        return motion_dict

    def iter_entries(self, character=None, motion_name=None, cam_id=None):
        """
        Flat generator over all entries in motion_dict.
        Yields (character, motion_name, cam_id, chunk_id, entry) tuples.

        All parameters are optional filters:
            iter_entries()                              # everything
            iter_entries(cam_id=0)                      # cam 0 only
            iter_entries("michelle", "Dancing-Twerk")   # one motion, all cams
        """
        chars   = [character]   if character    else self.motion_dict.keys()
        for char in chars:
            motions = [motion_name] if motion_name  else self.motion_dict[char].keys()
            for mname in motions:
                info = self.motion_dict[char][mname]
                cams = [cam_id] if cam_id is not None else info["cams"].keys()
                for cid in cams:
                    for chunk_id, entries in info["cams"][cid].items():
                        for entry in entries:
                            yield char, mname, cid, chunk_id, entry

    @property
    def all_paths(self):
        """Flat list of all paths across every character / motion / cam / chunk."""
        return [r["path"] for r in self.index]

    def get_inference_sequence(self, character, motion_name, cam_id=0, variation_idx=0):
        """
        Return an ordered list of paths for one full motion — one path per
        chunk, in chunk_id order.  Use this to run inference and concat results.

        Parameters
        ----------
        character    : str   e.g. "michelle"
        motion_name  : str   e.g. "Dancing-Twerk"
        cam_id       : int   which camera view (default 0)
        variation_idx: int   which noise variation to pick when a chunk has
                             multiple data_ids (0 = lowest data_id, default)

        Returns
        -------
        list[str]  — paths sorted by chunk_id

        Example
        -------
            paths = dataset.get_inference_sequence("michelle", "Dancing-Twerk", cam_id=0)
            # → ["...chunk-0...", "...chunk-1...", "...chunk-2...", ...]
            results = [model(p) for p in paths]
            final_motion = concat(results)

            # To iterate all variations for chunk 4:
            n_var = dataset.n_variations("michelle", "Dancing-Twerk", cam_id=0, chunk_id=4)
            for v in range(n_var):
                paths = dataset.get_inference_sequence(..., variation_idx=v)
        """
        chunks = self.motion_dict[character][motion_name]["cams"][cam_id]
        return [
            entries[min(variation_idx, len(entries) - 1)]["path"]
            for chunk_id, entries in chunks.items()   # already sorted by chunk_id
        ]

    def n_variations(self, character, motion_name, cam_id=0, chunk_id=None):
        """
        Number of data_id variations available.
        If chunk_id is None, returns the max across all chunks (conservative bound).
        """
        chunks = self.motion_dict[character][motion_name]["cams"][cam_id]
        if chunk_id is not None:
            return len(chunks[chunk_id])
        return max(len(entries) for entries in chunks.values())

    def iter_inference_sequences(self, character=None, motion_name=None, cam_id=None, variation_idx=0):
        """
        Yield inference sequences with any combination of optional filters.
        Omit a parameter (or pass None) to include all values for that dimension.

        Parameters
        ----------
        character     : str | None
        motion_name   : str | None
        cam_id        : int | None   None = all cameras
        variation_idx : int          which noise variation (default 0)

        Yields
        ------
        dict: {character, motion_name, cam_id, n_chunk, paths}

        Examples
        --------
            dataset.iter_inference_sequences()                              # everything
            dataset.iter_inference_sequences(character="michelle")          # one char, all motions, all cams
            dataset.iter_inference_sequences(cam_id=0)                      # all chars, all motions, cam 0
            dataset.iter_inference_sequences(motion_name="Dancing-Twerk")   # all chars, one motion, all cams
            dataset.iter_inference_sequences("michelle", "Dancing-Twerk")   # one char, one motion, all cams
            dataset.iter_inference_sequences("michelle", cam_id=0)          # one char, all motions, cam 0
            dataset.iter_inference_sequences("michelle", "Dancing-Twerk", cam_id=0)  # fully specified
        """
        target_chars = [character] if character is not None else list(self.motion_dict.keys())
        for char in target_chars:
            if char not in self.motion_dict:
                raise KeyError(f"Character '{char}' not found.")
            motions = self.motion_dict[char]
            target_motions = [motion_name] if motion_name is not None else list(motions.keys())
            for mname in target_motions:
                if mname not in motions:
                    raise KeyError(f"Motion '{mname}' not found for character '{char}'")
                info = motions[mname]
                target_cams = [cam_id] if cam_id is not None else list(info["cams"].keys())
                for cid in target_cams:
                    if cid not in info["cams"]:
                        continue
                    yield {
                        "character":   char,
                        "motion_name": mname,
                        "cam_id":      cid,
                        "n_chunk":     info["n_chunk"],
                        "paths":       self.get_inference_sequence(char, mname, cid, variation_idx),
                    }

    def summary(self):
        """
        Print a table of all available characters, motions, and their metadata.
        Uses rich if available, plain text otherwise.
        """
        rows = []
        for char, motions in self.motion_dict.items():
            for mname, info in motions.items():
                n_var = max(
                    len(entries)
                    for cid in info["cams"]
                    for entries in info["cams"][cid].values()
                )
                rows.append((char, mname, info["n_cam"], info["n_chunk"], n_var))

        try:
            from rich.console import Console
            from rich.table import Table
            from rich import box
            t = Table(title="Dataset Summary", box=box.ROUNDED, border_style="cyan")
            t.add_column("Character",   style="bold cyan",   no_wrap=True)
            t.add_column("Motion",      style="yellow")
            t.add_column("# Cams",      style="green",  justify="right")
            t.add_column("# Chunks",    style="green",  justify="right")
            t.add_column("# Variations",style="magenta",justify="right")
            for row in rows:
                t.add_row(row[0], row[1], str(row[2]), str(row[3]), str(row[4]))
            Console().print(t)
        except ImportError:
            header = f"{'Character':<20} {'Motion':<30} {'Cams':>6} {'Chunks':>8} {'Variations':>12}"
            print(header)
            print("-" * len(header))
            for char, mname, n_cam, n_chunk, n_var in rows:
                print(f"{char:<20} {mname:<30} {n_cam:>6} {n_chunk:>8} {n_var:>12}")
            print(f"\nTotal: {len(rows)} motion(s)")

if __name__ == "__main__":
    p = "/host/data/mint/Motion_Dataset/Mixamo/testset_testing_pipeline/wan_output/5_frames/train_dit_features/0/"

    dataset = DitFeaturesByMotionNameDataset(glob.glob(p + "/*.pth"), preferred_dit_block_id=0)
    for seq in dataset.iter_inference_sequences(character="michelle"):
        inference_ds = DitFeaturesDataset(seq["paths"], preferred_dit_block_id=0)
        loader = th.utils.data.DataLoader(
            inference_ds,
            batch_size=1,
            collate_fn=inference_ds.collate_fn_,
        )
        for batch in loader:
            print(batch["dit_features"].shape)