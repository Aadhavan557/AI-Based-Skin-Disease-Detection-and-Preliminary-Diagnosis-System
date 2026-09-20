"""
07_split_dataset.py
--------------------
Splits the (augmented) dataset into train / validation / test sets
with stratified sampling to preserve class ratios.

Split ratios (configurable):
  • Train      : 70%
  • Validation : 15%
  • Test        : 15%

Output structure:
  dataset_split/
    ├── train/
    │   ├── class_1/
    │   └── ...
    ├── val/
    │   ├── class_1/
    │   └── ...
    └── test/
        ├── class_1/
        └── ...

Files are COPIED (not moved) to preserve the augmented dataset.
A CSV manifest is also generated for reproducibility.
"""

import sys
import shutil
import random
from pathlib import Path
import json
from datetime import datetime
import csv

try:
    from tqdm import tqdm
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    print("Install with: pip install tqdm numpy matplotlib")
    sys.exit(1)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# Priority order: augmented → resized → original
for candidate in ["dataset_augmented", "dataset_resized", "dataset"]:
    SOURCE_DIR = BASE_DIR / candidate
    if SOURCE_DIR.exists():
        break

OUTPUT_DIR = BASE_DIR / "dataset_split"
LOG_DIR = BASE_DIR / "preprocessing" / "outputs" / "07_split"
LOG_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15   # = 1 - TRAIN_RATIO - VAL_RATIO

RANDOM_SEED = 42

# Set True to use symlinks instead of copying (saves disk space on Linux/Mac)
USE_SYMLINKS = False

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def print_section(title: str):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


def stratified_split(
    image_paths: list[Path],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Shuffles and splits a list of paths into train/val/test."""
    rng = random.Random(seed)
    paths = list(image_paths)
    rng.shuffle(paths)
    n = len(paths)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    train = paths[:n_train]
    val   = paths[n_train: n_train + n_val]
    test  = paths[n_train + n_val:]
    return train, val, test


def copy_or_link(src: Path, dst: Path, use_symlinks: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if use_symlinks:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print_section("SPLIT DATASET — Skin Disease Detection")
    print(f"Source dataset : {SOURCE_DIR}")
    print(f"Output dir     : {OUTPUT_DIR}")
    print(f"Split ratios   : Train {TRAIN_RATIO:.0%} / Val {VAL_RATIO:.0%} / Test {TEST_RATIO:.0%}")
    print(f"Random seed    : {RANDOM_SEED}")

    splits = {"train": [], "val": [], "test": []}
    class_stats = {}
    manifest_rows = []

    # ── Per-class stratified split ─────────────────────────
    print_section("Splitting classes…")
    for cls_dir in sorted(SOURCE_DIR.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls_name = cls_dir.name
        all_images = [
            p for p in cls_dir.rglob("*")
            if p.suffix.lower() in SUPPORTED_FORMATS and p.is_file()
        ]

        if len(all_images) == 0:
            print(f"  [SKIP] {cls_name} — no images found")
            continue

        train, val, test = stratified_split(all_images, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED)

        splits["train"].extend([(p, cls_name) for p in train])
        splits["val"].extend([(p, cls_name) for p in val])
        splits["test"].extend([(p, cls_name) for p in test])

        class_stats[cls_name] = {
            "total": len(all_images),
            "train": len(train),
            "val": len(val),
            "test": len(test),
        }

        for split_name, paths in [("train", train), ("val", val), ("test", test)]:
            for p in paths:
                manifest_rows.append({
                    "split": split_name,
                    "class": cls_name,
                    "original_path": str(p),
                    "filename": p.name,
                })

        print(
            f"  {cls_name[:50]:<52} total={len(all_images):>5} "
            f"train={len(train):>5} val={len(val):>4} test={len(test):>4}"
        )

    # ── Copy files ─────────────────────────────────────────
    print_section("Copying files to split directories…")
    for split_name, items in splits.items():
        split_dir = OUTPUT_DIR / split_name
        for src_path, cls_name in tqdm(items, desc=f"  {split_name:>5}", unit="img"):
            dst_path = split_dir / cls_name / src_path.name
            copy_or_link(src_path, dst_path, USE_SYMLINKS)

    # ── Summary statistics ─────────────────────────────────
    print_section("Split Summary")
    total_train = sum(v["train"] for v in class_stats.values())
    total_val   = sum(v["val"]   for v in class_stats.values())
    total_test  = sum(v["test"]  for v in class_stats.values())
    total_all   = total_train + total_val + total_test

    print(f"  {'Split':<10} {'Images':>8}  {'Ratio':>7}")
    print(f"  {'─'*10}  {'─'*8}  {'─'*7}")
    print(f"  {'Train':<10} {total_train:>8}  {total_train/total_all:.1%}")
    print(f"  {'Val':<10} {total_val:>8}  {total_val/total_all:.1%}")
    print(f"  {'Test':<10} {total_test:>8}  {total_test/total_all:.1%}")
    print(f"  {'TOTAL':<10} {total_all:>8}")

    # ── Visualization ──────────────────────────────────────
    cls_names_short = [n.split(".")[0][:18] for n in class_stats]
    trains = [class_stats[k]["train"] for k in class_stats]
    vals   = [class_stats[k]["val"]   for k in class_stats]
    tests  = [class_stats[k]["test"]  for k in class_stats]

    x = np.arange(len(cls_names_short))
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x, trains, label="Train", color="#2ecc71", alpha=0.9)
    ax.bar(x, vals,   bottom=trains, label="Val", color="#f39c12", alpha=0.9)
    ax.bar(x, tests,  bottom=[t + v for t, v in zip(trains, vals)],
           label="Test", color="#e74c3c", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(cls_names_short, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Number of Images")
    ax.set_title("Dataset Split Distribution per Class", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plot_path = LOG_DIR / "split_distribution.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved: {plot_path}")

    # ── Save manifest CSV ──────────────────────────────────
    manifest_path = LOG_DIR / "dataset_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "class", "original_path", "filename"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"  Manifest CSV saved: {manifest_path}")

    # ── Save summary JSON ──────────────────────────────────
    summary = {
        "run_timestamp": datetime.now().isoformat(),
        "source_dir": str(SOURCE_DIR),
        "output_dir": str(OUTPUT_DIR),
        "split_ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
        "random_seed": RANDOM_SEED,
        "total": {"train": total_train, "val": total_val, "test": total_test, "all": total_all},
        "class_breakdown": class_stats,
    }
    json_path = LOG_DIR / "split_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary JSON saved: {json_path}")

    print_section("Dataset Split Complete ✓")
    print(f"  Split dataset saved to: {OUTPUT_DIR}")
    print(f"    dataset_split/train/  — {total_train} images")
    print(f"    dataset_split/val/    — {total_val} images")
    print(f"    dataset_split/test/   — {total_test} images\n")


if __name__ == "__main__":
    main()
