"""
gen_vispage.py
==============
Generate a visualisation HTML from one or more inference .npz result files.

Grouping modes
--------------
1. --row_id  (explicit)
   Assign each file to a row by index.  Same index → same row, overlaid.
   Example:  --row_id 0 0 1   puts files 0+1 together, file 2 alone.

2. Auto (default, no --row_id)
   Files that share the same motion_name (read from the npz or inferred from
   the directory structure) are placed on the same row.

3. --no_group
   Every file gets its own row regardless.

Path layouts
------------
Default (4-level):   .../<model_config>/<ckpt_name>/<motion>/res.npz
With --with_character (5-level):
                     .../<model_config>/<ckpt_name>/<character>/<motion>/res.npz
    Groups by "character/motion" so the same character+motion from different
    checkpoints are overlaid on one row automatically.

Usage examples
--------------
# Explicit grouping: first 3 on row 0, 4th on row 1
python gen_vispage.py --out out.html \\
    --prediction_path a.npz b.npz c.npz d.npz \\
    --row_id 0 0 0 1

# Auto-group by motion_name (default)
python gen_vispage.py --out out.html --prediction_path ckptA_walk.npz ckptB_walk.npz ckptA_run.npz

# Every file on its own row
python gen_vispage.py --out out.html --no_group --prediction_path *.npz

# Discover all results under a root; 5-level layout with character dir
# Auto-overlays same character+motion across checkpoints
python gen_vispage.py --out out.html --with_character \\
    --discover results/model_A results/model_B
"""

from viz3d_utils import panel_3d, panel_image_overlay, generate_html
from PIL import Image
from pathlib import Path
from collections import defaultdict
import numpy as np
import argparse

# Colour palette for overlaid predictions (GT always uses _GT_COLOR)
_PRED_PALETTE = [
    "#FF0000",  # coral
    "#33FF00",  # purple
    "#FF00FF",  # yellow
    "#00FFF2",  # cyan
]
# _GT_COLOR = "#50FA7B"  # green
_GT_COLOR = "#0400FF"  # light blue


# ── Path-label helpers ────────────────────────────────────────────────────────

def _shorten(s: str, max_len: int = 28) -> str:
    return s if len(s) <= max_len else s[:max_len - 1] + "…"


def _parse_path_labels(npz_path: str, with_character: bool = False) -> tuple[str, str]:
    """
    Infer (motion_name, ckpt_label) from the directory structure.

    Default layout (4-level):  .../<model_config>/<ckpt_name>/<motion_name>/res.npz
        → motion_name = "motion_name"
        → ckpt_label  = "ckpt_name / model_config"

    With --with_character (5-level):  .../<model_config>/<ckpt_name>/<character>/<motion_name>/res.npz
        → motion_name = "character/motion_name"  (used as grouping key)
        → ckpt_label  = "ckpt_name / model_config"
    """
    parts = Path(npz_path).parts
    if with_character:
        if len(parts) >= 5:
            motion_name  = f"{parts[-3]}/{parts[-2]}"
            ckpt_name    = parts[-4]
            model_config = parts[-5]
            ckpt_label   = f"{model_config}/{ckpt_name}"
        elif len(parts) >= 4:
            motion_name = f"{parts[-3]}/{parts[-2]}"
            ckpt_label  = parts[-4]
        elif len(parts) >= 3:
            motion_name = f"{parts[-3]}/{parts[-2]}"
            ckpt_label  = parts[-3]
        else:
            motion_name = Path(npz_path).stem
            ckpt_label  = Path(npz_path).stem
    else:
        if len(parts) >= 4:
            motion_name  = parts[-2]
            ckpt_name    = parts[-3]
            model_config = parts[-4]
            ckpt_label   = f"{model_config}/{ckpt_name}"
        elif len(parts) >= 3:
            motion_name = parts[-2]
            ckpt_label  = parts[-3]
        elif len(parts) >= 2:
            motion_name = parts[-2]
            ckpt_label  = parts[-2]
        else:
            motion_name = Path(npz_path).stem
            ckpt_label  = Path(npz_path).stem
    return motion_name, ckpt_label


# ── npz loader ────────────────────────────────────────────────────────────────

def _load_npz(path: str, with_character: bool = False) -> dict:
    dat = np.load(path, allow_pickle=True)
    path_motion, path_ckpt = _parse_path_labels(path, with_character=with_character)

    motion_name = str(dat["motion_name"]).strip() if "motion_name" in dat else path_motion
    
    # Always prefer the parsed directory structure (path_ckpt) over dat["ckpt"] 
    # because dat["ckpt"] might just be the ckpt filename and lose the model_name context.
    ckpt_label = path_ckpt

    if not ckpt_label:
        ckpt_label = Path(str(dat["ckpt"])).stem.strip() if "ckpt" in dat else path_ckpt

    print(f"  motion={motion_name}  ckpt={ckpt_label}")
    return {
        "path":           path,
        "motion_name":    motion_name,
        "ckpt_label":     ckpt_label,
        "input_video":    [Image.fromarray(f) for f in dat["input_video"]],
        "motion_pred_2d": dat["motion_pred_2d"],   # (T, J, 2)
        "motion_pred_3d": dat["motion_pred_3d"],   # (T, J, 3)
        "motion_gt_2d":   dat["motion_gt_2d"],     # (T, J, 2)
        "motion_gt_3d":   dat["motion_gt_3d"],     # (T, J, 3)
        "edges":          dat["edges"].tolist(),
    }


# ── Row builder ───────────────────────────────────────────────────────────────

def _build_row(group: list[dict], row_label: str, image_quality: int) -> dict:
    """Turn a list of entries that belong to the same row into a row dict."""
    T      = group[0]["motion_pred_3d"].shape[0]
    edges  = group[0]["edges"]
    images = group[0]["input_video"]   # frames from first entry (same motion/video)
    gt_3d  = group[0]["motion_gt_3d"]
    gt_2d  = group[0]["motion_gt_2d"]

    skels_3d = []
    skels_2d = []
    for i, e in enumerate(group):
        col = _PRED_PALETTE[i % len(_PRED_PALETTE)]
        skels_3d.append({"joints": e["motion_pred_3d"], "color": col, "label": e["ckpt_label"], "path": e["path"]})
        skels_2d.append({"joints": e["motion_pred_2d"], "color": col, "label": e["ckpt_label"], "path": e["path"]})

    skels_3d.append({"joints": gt_3d, "color": _GT_COLOR, "label": "GT"})
    skels_2d.append({"joints": gt_2d, "color": _GT_COLOR, "label": "GT"})

    return {
        "label": row_label,
        "T":     T,
        "panels": [
            panel_3d(
                skeletons=skels_3d,
                edges=edges,
                coord="z_up",
                label="3D",
            ),
            panel_image_overlay(
                images=images,
                skeletons=skels_2d,
                edges=edges,
                joints_space="image",
                image_quality=image_quality,
                label="Overlay",
            ),
        ],
    }


def build_rows(entries: list[dict], row_ids: list[int] | None,
               image_quality: int = 90) -> list[dict]:
    """
    Group entries into rows and build the rows list for generate_html.

    If row_ids is provided (same length as entries), entries with equal
    row_id are placed on the same row in the order they first appear.
    Otherwise entries are grouped by motion_name.
    """
    if row_ids is not None:
        # Explicit grouping — preserve insertion order of row ids
        groups: dict[int, list[dict]] = defaultdict(list)
        seen_order: list[int] = []
        for e, rid in zip(entries, row_ids):
            if rid not in groups:
                seen_order.append(rid)
            groups[rid].append(e)
        ordered_groups = [(rid, groups[rid]) for rid in seen_order]
    else:
        # Auto-group by motion_name
        name_groups: dict[str, list[dict]] = defaultdict(list)
        for e in entries:
            name_groups[e["motion_name"]].append(e)
        ordered_groups = list(name_groups.items())

    rows = []
    for key, group in ordered_groups:
        n = len(group)
        if row_ids is not None:
            label = f"row {key}  ·  {', '.join(e['ckpt_label'] for e in group)}"
        else:
            motion = str(key)
            label  = motion if n == 1 else f"{motion}  [{n} predictions]"
        rows.append(_build_row(group, label, image_quality))
    return rows


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate skeleton visualisation HTML from npz results.")
    ap.add_argument("--out",           default="./viz3d_out.html",  help="Output HTML path.")
    ap.add_argument("--cell_height",   type=int, default=640,       help="Panel height in pixels.")
    ap.add_argument("--image_quality", type=int, default=90,        help="JPEG quality for frames (1-95).")
    ap.add_argument("--title",         default="Skeleton Viewer",   help="Page title.")
    ap.add_argument("--no_group",      action="store_true",         help="Put every file on its own row.")
    ap.add_argument("--row_id",        nargs="+", type=int, default=None,
                    help="Row index for each --prediction_path file (same length). "
                         "Files with the same index share a row. E.g.: --row_id 0 0 0 1")
    ap.add_argument("--prediction_path", nargs="*", default=[],     help="One or more .npz result files.")
    ap.add_argument("--discover",      nargs="+", default=[],       metavar="DIR",
                    help="Directories to recursively search for *.npz files. "
                         "Found files are appended to --prediction_path (sorted). "
                         "E.g.: --discover results_michelle/model_step_100000")
    ap.add_argument("--with_character", action="store_true",
                    help="Use 5-level path layout: .../model/ckpt/character/motion/res.npz. "
                         "The grouping key becomes 'character/motion' so entries from different "
                         "checkpoints but the same character+motion are overlaid on one row.")
    ap.add_argument("--display_indices", default=None,       help="Filter which rows/motions to display (e.g., '1-4, 5, 8'). 0-indexed.")
    args = ap.parse_args()

    # Auto-discover .npz files from --discover directories
    discovered: list[str] = []
    for d in args.discover:
        found = sorted(str(p) for p in Path(d).rglob("*.npz"))
        if not found:
            ap.error(f"--discover: no .npz files found under '{d}'")
        print(f"Discovered {len(found)} file(s) under '{d}':")
        for f in found:
            print(f"  {f}")
        discovered.extend(found)
    all_paths = list(args.prediction_path) + discovered

    if not all_paths:
        ap.error("Provide at least one .npz file via --prediction_path or --discover.")

    if args.row_id is not None and len(args.row_id) != len(all_paths):
        ap.error(f"--row_id must have the same length as the total number of .npz files "
                 f"({len(args.row_id)} vs {len(all_paths)})")

    print(f"Loading {len(all_paths)} file(s)…")
    entries = []
    for p in all_paths:
        print(f"  {p}")
        e = _load_npz(p, with_character=args.with_character)
        if args.no_group:
            e["motion_name"] = Path(p).stem   # unique key → each file gets its own row
        entries.append(e)

    row_ids = args.row_id if not args.no_group else None
    rows = build_rows(entries, row_ids=row_ids, image_quality=args.image_quality)

    if args.display_indices is not None:
        keep = []
        for part in args.display_indices.split(','):
            part = part.strip()
            if not part: continue
            if '-' in part:
                s, e = map(int, part.split('-'))
                keep.extend(range(s, e + 1))
            else:
                keep.append(int(part))
        keep_set = set(keep)
        rows = [r for i, r in enumerate(rows) if i in keep_set]
        print(f"Filtered to {len(rows)} row(s) based on --display_indices.")

    print(f"Built {len(rows)} row(s).")

    generate_html(rows, output_path=args.out, title=args.title, cell_height=args.cell_height)
    sz = Path(args.out).stat().st_size / 1024
    print(f"Written {sz:.0f} KB → {args.out}")