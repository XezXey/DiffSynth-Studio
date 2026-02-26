"""
gen_vispage.py
==============
Generate a visualisation HTML from one or more inference .npz result files.

Grouping rules
--------------
By default, files that share the same `motion_name` field (stored inside the
npz) are placed on the **same row** so their skeletons are overlaid — ideal
for comparing different checkpoints on the same sequence.

Files with **different** motion names get their own rows.

Usage examples
--------------
# Single file
python gen_vispage.py --out out.html --prediction_path result.npz

# Compare two ckpts on the same motion (auto-grouped by motion_name)
python gen_vispage.py --out out.html --prediction_path ckptA_walk.npz ckptB_walk.npz

# Two different motions → two rows each with one ckpt
python gen_vispage.py --out out.html --prediction_path ckptA_walk.npz ckptA_run.npz

# Mixed: two ckpts on walk (same row) + one ckpt on run (separate row)
python gen_vispage.py --out out.html \
    --prediction_path ckptA_walk.npz ckptB_walk.npz ckptA_run.npz

# Force all files onto separate rows regardless of motion name
python gen_vispage.py --out out.html --no_group --prediction_path *.npz
"""

from viz3d_utils import panel_3d, panel_image_overlay, generate_html
from PIL import Image
from pathlib import Path
from collections import defaultdict
import numpy as np
import argparse

# Colour palette for multiple predictions on the same row
# GT always gets the last colour (green)
_PRED_PALETTE = [
    "#4ECDC4",  # teal
    "#FF6B6B",  # coral
    "#FFB86C",  # orange
    "#BD93F9",  # purple
    "#8BE9FD",  # cyan
    "#F1FA8C",  # yellow
]
_GT_COLOR = "#50FA7B"  # green


def _shorten(s: str, max_len: int = 28) -> str:
    """Trim a long directory name to fit in a badge."""
    return s if len(s) <= max_len else s[:max_len - 1] + "…"


def _parse_path_labels(npz_path: str) -> tuple[str, str]:
    """
    Infer (motion_name, ckpt_label) from the directory structure.

    Expected layout (any depth):
        .../<model_config>/<ckpt_name>/<motion_name>/res.npz

    Example:
        .../results/heatmap_TI2V-5B_320x640_t20/model_step_170000/michelle_Jump/res.npz
        → motion_name = "michelle_Jump"
        → ckpt_label  = "model_step_170000 / heatmap_TI2V-5B_320x640_t20"
                         (ckpt first so badges stay readable when truncated)
    """
    parts = Path(npz_path).parts
    if len(parts) >= 4:
        motion_name  = parts[-2]
        ckpt_name    = parts[-3]
        model_config = parts[-4]
        ckpt_label   = f"{ckpt_name} / {_shorten(model_config)}"
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


def _load_npz(path: str) -> dict:
    dat = np.load(path, allow_pickle=True)

    path_motion, path_ckpt = _parse_path_labels(path)
    # Prefer fields stored in the npz, fall back to path-derived labels
    motion_name = str(dat["motion_name"]).strip() if "motion_name" in dat else path_motion
    ckpt_label  = Path(str(dat["ckpt"])).stem.strip() if "ckpt" in dat else path_ckpt
    # If the stored ckpt field is empty / uninformative, use the path-derived one
    if not ckpt_label:
        ckpt_label = path_ckpt

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


def build_rows(entries: list[dict], image_quality: int = 90) -> list[dict]:
    """
    Build the rows list consumed by generate_html.

    Each entry in `entries` is one loaded npz dict (from _load_npz).
    Entries that share the same motion_name are merged into one row.
    """
    # Group preserving insertion order
    groups: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        groups[e["motion_name"]].append(e)

    rows = []
    for motion_name, group in groups.items():
        T       = group[0]["motion_pred_3d"].shape[0]
        edges   = group[0]["edges"]
        images  = group[0]["input_video"]   # video frames from first entry (same motion)
        gt_3d   = group[0]["motion_gt_3d"]
        gt_2d   = group[0]["motion_gt_2d"]

        # ── 3-D panel: all predictions + one shared GT ───────────────────────
        skels_3d = []
        for i, e in enumerate(group):
            skels_3d.append({
                "joints": e["motion_pred_3d"],
                "color":  _PRED_PALETTE[i % len(_PRED_PALETTE)],
                "label":  e["ckpt_label"],
            })
        skels_3d.append({
            "joints": gt_3d,
            "color":  _GT_COLOR,
            "label":  "GT",
        })

        # ── Image-overlay panel: all predictions + one shared GT ─────────────
        # Build skeleton list for the overlay (2-D, image-space)
        joints_list = []
        for i, e in enumerate(group):
            joints_list.append({
                "joints": e["motion_pred_2d"],
                "color":  _PRED_PALETTE[i % len(_PRED_PALETTE)],
                "label":  e["ckpt_label"],
            })
        joints_list.append({
            "joints": gt_2d,
            "color":  _GT_COLOR,
            "label":  "GT",
        })

        row_label = motion_name
        if len(group) > 1:
            row_label += f"  [{len(group)} predictions]"

        rows.append({
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
                    skeletons=joints_list,
                    edges=edges,
                    joints_space="image",
                    image_quality=image_quality,
                    label="Overlay",
                ),
            ],
        })
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate skeleton visualisation HTML from npz results.")
    ap.add_argument("--out",              default="./viz3d_out.html",   help="Output HTML path.")
    ap.add_argument("--cell_height",      type=int, default=640,        help="Panel height in pixels.")
    ap.add_argument("--image_quality",    type=int, default=90,         help="JPEG quality for frames (1-95).")
    ap.add_argument("--no_group",         action="store_true",          help="Put every file on its own row (disable auto-grouping by motion name).")
    ap.add_argument("--title",            default="Skeleton Viewer",    help="Page title.")
    ap.add_argument("--prediction_path",  nargs="+", required=True,     help="One or more .npz result files.")
    args = ap.parse_args()

    print(f"Loading {len(args.prediction_path)} file(s)…")
    entries = []
    for p in args.prediction_path:
        print(f"  {p}")
        e = _load_npz(p)
        if args.no_group:
            # Give each entry a unique motion key so they never merge
            e["motion_name"] = Path(p).stem
        entries.append(e)

    rows = build_rows(entries, image_quality=args.image_quality)
    print(f"Built {len(rows)} row(s).")

    generate_html(rows, output_path=args.out, title=args.title, cell_height=args.cell_height)
    sz = Path(args.out).stat().st_size / 1024
    print(f"Written {sz:.0f} KB → {args.out}")
