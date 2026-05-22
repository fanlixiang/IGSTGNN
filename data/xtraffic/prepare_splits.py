#!/usr/bin/env python3
"""Create IGSTGNN train/val/test split files from incident_all.npy.

The released incident_all.npy is already ordered and normalized. This
script only slices it into the files expected by the existing dataloader,
so no dataloader changes are required.
"""

import argparse
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate incident_train/val/test.npy from incident_all.npy"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset directory name under data_root, e.g. Alameda, Contra_Costa, or Orange",
    )
    parser.add_argument(
        "--data_root",
        default=Path(__file__).resolve().parent,
        type=Path,
        help="Directory that contains dataset folders. Defaults to this xtraffic directory.",
    )
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing incident_train/val/test.npy files.",
    )
    return parser.parse_args()


def validate_ratios(train_ratio, val_ratio, test_ratio):
    ratios = [train_ratio, val_ratio, test_ratio]
    if any(ratio <= 0 for ratio in ratios):
        raise ValueError("Split ratios must all be positive.")
    total = sum(ratios)
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}.")


def required_file(dataset_dir, filename):
    path = dataset_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def main():
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    dataset_dir = args.data_root / args.dataset
    all_file = required_file(dataset_dir, "incident_all.npy")

    output_files = {
        "train": dataset_dir / "incident_train.npy",
        "val": dataset_dir / "incident_val.npy",
        "test": dataset_dir / "incident_test.npy",
    }
    existing = [str(path) for path in output_files.values() if path.exists()]
    if existing and not args.overwrite:
        joined = "\n  ".join(existing)
        raise FileExistsError(
            "Split files already exist. Use --overwrite to replace them:\n  " + joined
        )

    samples = np.load(all_file, allow_pickle=True)
    total_samples = len(samples)
    if total_samples == 0:
        raise ValueError(f"No samples found in {all_file}")

    train_end = int(total_samples * args.train_ratio)
    val_end = train_end + int(total_samples * args.val_ratio)

    splits = {
        "train": samples[:train_end],
        "val": samples[train_end:val_end],
        "test": samples[val_end:],
    }

    for split_name, split_samples in splits.items():
        np.save(output_files[split_name], np.asarray(split_samples, dtype=object))
        print(f"Saved {split_name}: {len(split_samples)} samples -> {output_files[split_name]}")

    required_file(dataset_dir, "incident_stats.npz")
    print("Done. Existing incident_stats.npz is reused by the dataloader.")


if __name__ == "__main__":
    main()
