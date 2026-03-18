import argparse
import os
import random
import shutil


def parse_ratio(ratio_str: str):
    parts = [p.strip() for p in ratio_str.split(",")]
    if len(parts) != 3:
        raise ValueError("--ratio must contain exactly 3 comma-separated numbers, e.g. 8,1,1")

    try:
        ratios = [float(p) for p in parts]
    except ValueError as e:
        raise ValueError("--ratio contains non-numeric values") from e

    if any(r <= 0 for r in ratios):
        raise ValueError("All ratio values must be > 0")

    return ratios

def split_counts(total: int, ratios):
    ratio_sum = sum(ratios)
    raw_counts = [(total * r) / ratio_sum for r in ratios]
    counts = [int(x) for x in raw_counts]

    # Distribute any remaining samples by largest fractional part.
    remaining = total - sum(counts)
    fractional_order = sorted(
        range(3),
        key=lambda i: (raw_counts[i] - counts[i]),
        reverse=True,
    )
    for i in range(remaining):
        counts[fractional_order[i % 3]] += 1

    return counts

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    parser = argparse.ArgumentParser(description="Split .fbx files into train/val/test subsets.")
    parser.add_argument("--motion_dir", type=str, required=True, help="Directory containing .fbx files")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Output dataset directory")
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Number of files to use in total. If omitted, use all .fbx files.",
    )
    parser.add_argument(
        "--ratio",
        type=str,
        default="8,1,1",
        help="Train/val/test ratio as 3 comma-separated values, e.g. 8,1,1",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic shuffling",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying (default: copy)",
    )
    args = parser.parse_args()

    motion_dir = args.motion_dir
    dataset_dir = args.dataset_dir

    if not os.path.isdir(motion_dir):
        raise FileNotFoundError(f"motion_dir not found: {motion_dir}")

    ensure_dir(dataset_dir)
    train_dir = os.path.join(dataset_dir, "trainset_motion")
    val_dir = os.path.join(dataset_dir, "valset_motion")
    test_dir = os.path.join(dataset_dir, "testset_motion")

    ensure_dir(train_dir)
    ensure_dir(val_dir)
    ensure_dir(test_dir)

    all_files = [
        os.path.join(motion_dir, name)
        for name in os.listdir(motion_dir)
        if os.path.isfile(os.path.join(motion_dir, name)) and name.lower().endswith(".fbx")
    ]

    if not all_files:
        raise ValueError(f"No .fbx files found in: {motion_dir}")

    all_files.sort()
    rng = random.Random(args.seed)
    rng.shuffle(all_files)

    total_available = len(all_files)
    n = total_available if args.n is None else args.n
    if n <= 0:
        raise ValueError("--n must be > 0")
    n = min(n, total_available)

    selected_files = all_files[:n]
    ratios = parse_ratio(args.ratio)
    train_n, val_n, test_n = split_counts(n, ratios)

    train_files = selected_files[:train_n]
    val_files = selected_files[train_n:train_n + val_n]
    test_files = selected_files[train_n + val_n:]

    op = shutil.move if args.move else shutil.copy2

    for src in train_files:
        op(src, os.path.join(train_dir, os.path.basename(src)))
    for src in val_files:
        op(src, os.path.join(val_dir, os.path.basename(src)))
    for src in test_files:
        op(src, os.path.join(test_dir, os.path.basename(src)))

    print("Split complete")
    print(f"Total available: {total_available}")
    print(f"Used: {n}")
    print(f"trainset_motion: {len(train_files)}")
    print(f"valset_motion: {len(val_files)}")
    print(f"testset_motion: {len(test_files)}")
    print(f"Mode: {'move' if args.move else 'copy'}")
    print(f"Seed: {args.seed}")
    print(f"Ratio: {args.ratio}")


if __name__ == "__main__":
    main()