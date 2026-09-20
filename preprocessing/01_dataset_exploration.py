"""
01_dataset_exploration.py
--------------------------
Explores the raw dataset structure, counts images per class,
checks image formats, sizes, and generates a summary report.

Dataset: AI-Based Skin Disease Detection
Classes: 10 skin disease categories
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
import json

try:
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    import pandas as pd
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    print("Install with: pip install pillow numpy matplotlib seaborn pandas")
    sys.exit(1)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "preprocessing" / "outputs" / "01_exploration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def get_class_folders(dataset_dir: Path) -> dict[str, Path]:
    """Returns a dict mapping clean class names → folder paths."""
    classes = {}
    for folder in sorted(dataset_dir.iterdir()):
        if folder.is_dir():
            classes[folder.name] = folder
    return classes


def collect_image_stats(class_name: str, folder: Path) -> list[dict]:
    """Collect metadata for every image in a class folder."""
    records = []
    for img_path in folder.rglob("*"):
        if img_path.suffix.lower() not in SUPPORTED_FORMATS:
            continue
        stat = {
            "class": class_name,
            "filename": img_path.name,
            "path": str(img_path),
            "format": img_path.suffix.lower(),
            "size_bytes": img_path.stat().st_size,
            "width": None,
            "height": None,
            "channels": None,
            "readable": False,
        }
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                mode = img.mode
                ch = len(img.getbands())
                stat.update({"width": w, "height": h, "channels": ch, "readable": True})
        except Exception:
            pass  # corrupted — handled in script 02
        records.append(stat)
    return records


def print_section(title: str):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print_section("DATASET EXPLORATION — Skin Disease Detection")
    print(f"Dataset path : {DATASET_DIR}")
    print(f"Output path  : {OUTPUT_DIR}")

    if not DATASET_DIR.exists():
        print(f"[ERROR] Dataset directory not found: {DATASET_DIR}")
        sys.exit(1)

    class_folders = get_class_folders(DATASET_DIR)
    print(f"\nFound {len(class_folders)} class folders:\n")
    for name in class_folders:
        print(f"  • {name}")

    # ── Collect stats ──────────────────────────────────────
    print_section("Collecting image metadata (this may take a moment)…")
    all_records = []
    for class_name, folder in class_folders.items():
        records = collect_image_stats(class_name, folder)
        all_records.extend(records)
        readable = sum(1 for r in records if r["readable"])
        print(f"  {class_name:<55} {readable:>5} images")

    df = pd.DataFrame(all_records)

    # ── Summary statistics ─────────────────────────────────
    print_section("Summary Statistics")
    total = len(df)
    readable_df = df[df["readable"]]
    corrupted = total - len(readable_df)

    print(f"  Total images found   : {total}")
    print(f"  Readable images      : {len(readable_df)}")
    print(f"  Corrupted / unreadable: {corrupted}")

    print("\n  Per-class counts:")
    class_counts = df.groupby("class").size().sort_values(ascending=False)
    for cls, cnt in class_counts.items():
        bar = "█" * (cnt // 200)
        print(f"  {cls:<55} {cnt:>5}  {bar}")

    if len(readable_df) > 0:
        print(f"\n  Image size statistics (width × height):")
        print(f"  Min  : {readable_df['width'].min():.0f} × {readable_df['height'].min():.0f}")
        print(f"  Max  : {readable_df['width'].max():.0f} × {readable_df['height'].max():.0f}")
        print(f"  Mean : {readable_df['width'].mean():.0f} × {readable_df['height'].mean():.0f}")
        print(f"  Std  : {readable_df['width'].std():.0f} × {readable_df['height'].std():.0f}")

        print("\n  Format distribution:")
        fmt_counts = df["format"].value_counts()
        for fmt, cnt in fmt_counts.items():
            print(f"    {fmt:<10} {cnt}")

        print("\n  Channel distribution:")
        ch_counts = readable_df["channels"].value_counts()
        for ch, cnt in ch_counts.items():
            label = {1: "Grayscale", 3: "RGB", 4: "RGBA"}.get(ch, f"{ch}-channel")
            print(f"    {label:<12} {cnt}")

    # ── Visualizations ─────────────────────────────────────
    print_section("Generating Visualizations…")

    # 1) Class distribution bar chart
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Skin Disease Dataset — Class Distribution", fontsize=16, fontweight="bold")

    class_counts_sorted = df.groupby("class").size().sort_values(ascending=False)
    short_names = [n.split(".")[0].strip()[:30] for n in class_counts_sorted.index]

    colors = plt.cm.tab10(np.linspace(0, 1, len(short_names)))
    axes[0].barh(short_names[::-1], class_counts_sorted.values[::-1], color=colors[::-1])
    axes[0].set_xlabel("Number of Images")
    axes[0].set_title("Images per Class")
    axes[0].grid(axis="x", alpha=0.3)
    for i, v in enumerate(class_counts_sorted.values[::-1]):
        axes[0].text(v + 30, i, str(v), va="center", fontsize=9)

    # 2) Pie chart
    axes[1].pie(
        class_counts_sorted.values,
        labels=short_names,
        autopct="%1.1f%%",
        colors=colors,
        startangle=140,
    )
    axes[1].set_title("Class Proportion")

    plt.tight_layout()
    dist_path = OUTPUT_DIR / "class_distribution.png"
    plt.savefig(dist_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {dist_path}")

    # 2) Image size scatter
    if len(readable_df) > 0:
        sample = readable_df.sample(min(2000, len(readable_df)), random_state=42)
        fig, ax = plt.subplots(figsize=(10, 7))
        scatter = ax.scatter(
            sample["width"], sample["height"],
            alpha=0.3, s=10, c=sample["class"].astype("category").cat.codes,
            cmap="tab10"
        )
        ax.set_xlabel("Width (px)")
        ax.set_ylabel("Height (px)")
        ax.set_title("Image Dimensions Distribution (sample)")
        ax.grid(alpha=0.3)
        size_path = OUTPUT_DIR / "image_size_distribution.png"
        plt.savefig(size_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {size_path}")

    # ── Save CSV report ────────────────────────────────────
    csv_path = OUTPUT_DIR / "dataset_inventory.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved inventory CSV: {csv_path}")

    # ── Save JSON summary ──────────────────────────────────
    summary = {
        "total_images": int(total),
        "readable_images": int(len(readable_df)),
        "corrupted_images": int(corrupted),
        "num_classes": len(class_folders),
        "classes": {
            cls: int(cnt) for cls, cnt in class_counts.items()
        },
        "format_distribution": {
            k: int(v) for k, v in fmt_counts.items()
        } if len(readable_df) > 0 else {},
        "image_size": {
            "min_width": int(readable_df["width"].min()) if len(readable_df) > 0 else 0,
            "max_width": int(readable_df["width"].max()) if len(readable_df) > 0 else 0,
            "mean_width": float(readable_df["width"].mean()) if len(readable_df) > 0 else 0,
            "min_height": int(readable_df["height"].min()) if len(readable_df) > 0 else 0,
            "max_height": int(readable_df["height"].max()) if len(readable_df) > 0 else 0,
            "mean_height": float(readable_df["height"].mean()) if len(readable_df) > 0 else 0,
        },
    }
    json_path = OUTPUT_DIR / "exploration_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary JSON : {json_path}")

    print_section("Exploration Complete ✓")
    print(f"  All outputs saved to: {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()
