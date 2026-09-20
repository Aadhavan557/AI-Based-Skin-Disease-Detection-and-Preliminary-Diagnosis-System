"""
08_visualize_dataset.py
------------------------
Generates a comprehensive visual report of the final preprocessed
dataset (dataset_split/) including:

  1. Grid of sample images per class
  2. Class distribution bar chart (train/val/test)
  3. Average image (per class) — what a "typical" image looks like
  4. Pixel intensity histograms (R/G/B channels) per split
  5. Image grid saved as a high-resolution PNG report
"""

import sys
from pathlib import Path
import json
import random
from datetime import datetime

try:
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from tqdm import tqdm
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    print("Install with: pip install pillow numpy matplotlib tqdm")
    sys.exit(1)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
SPLIT_DIR = BASE_DIR / "dataset_split"
OUTPUT_DIR = BASE_DIR / "preprocessing" / "outputs" / "08_visualize"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png"}
SAMPLES_PER_CLASS = 5     # images shown in the sample grid
RANDOM_SEED = 42
THUMBNAIL_SIZE = (150, 150)

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def print_section(title: str):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


def load_rgb(path: Path) -> np.ndarray | None:
    try:
        with Image.open(path).convert("RGB") as img:
            return np.array(img, dtype=np.uint8)
    except Exception:
        return None


def get_class_images(split_dir: Path, split: str, cls_name: str) -> list[Path]:
    cls_path = split_dir / split / cls_name
    if not cls_path.exists():
        return []
    return [p for p in cls_path.iterdir() if p.suffix.lower() in SUPPORTED_FORMATS]


def short_class_name(name: str) -> str:
    """Strip leading number and truncate."""
    parts = name.split(". ", 1)
    return parts[1][:30] if len(parts) > 1 else name[:30]


def get_all_class_names(split_dir: Path) -> list[str]:
    train_dir = split_dir / "train"
    if not train_dir.exists():
        return []
    return sorted([d.name for d in train_dir.iterdir() if d.is_dir()])

# ─────────────────────────────────────────────
# Visualization functions
# ─────────────────────────────────────────────

def plot_class_distribution(split_dir: Path, class_names: list[str], output_dir: Path):
    """Stacked bar chart of train/val/test per class."""
    splits = ["train", "val", "test"]
    colors = {"train": "#2ecc71", "val": "#f39c12", "test": "#e74c3c"}

    data = {s: [] for s in splits}
    for cls in class_names:
        for s in splits:
            cls_dir = split_dir / s / cls
            cnt = len([p for p in cls_dir.iterdir() if p.suffix.lower() in SUPPORTED_FORMATS]) if cls_dir.exists() else 0
            data[s].append(cnt)

    short_names = [short_class_name(c) for c in class_names]
    x = np.arange(len(short_names))

    fig, ax = plt.subplots(figsize=(15, 6))
    bottom = np.zeros(len(class_names))
    for s in splits:
        counts = np.array(data[s])
        ax.bar(x, counts, bottom=bottom, label=s.capitalize(), color=colors[s], alpha=0.88)
        bottom += counts

    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=38, ha="right", fontsize=9)
    ax.set_ylabel("Number of Images")
    ax.set_title("Final Dataset — Train / Val / Test Distribution per Class", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = output_dir / "class_distribution_final.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_sample_grid(split_dir: Path, class_names: list[str], output_dir: Path):
    """Grid of SAMPLES_PER_CLASS sample images per class from train split."""
    rng = random.Random(RANDOM_SEED)
    n_classes = len(class_names)
    cols = SAMPLES_PER_CLASS
    rows = n_classes

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    if rows == 1:
        axes = [axes]
    fig.suptitle("Sample Images per Class (Train Split)", fontsize=14, fontweight="bold", y=1.01)

    for row_idx, cls_name in enumerate(class_names):
        images = get_class_images(split_dir, "train", cls_name)
        selected = rng.sample(images, min(SAMPLES_PER_CLASS, len(images)))
        short_name = short_class_name(cls_name)

        for col_idx in range(cols):
            ax = axes[row_idx][col_idx]
            if col_idx < len(selected):
                arr = load_rgb(selected[col_idx])
                if arr is not None:
                    ax.imshow(arr)
                else:
                    ax.text(0.5, 0.5, "ERR", ha="center", va="center", transform=ax.transAxes)
            else:
                ax.axis("off")
            ax.axis("off")
            if col_idx == 0:
                ax.set_ylabel(short_name, fontsize=7, rotation=0, labelpad=60, va="center")

    plt.tight_layout()
    path = output_dir / "sample_image_grid.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_average_images(split_dir: Path, class_names: list[str], output_dir: Path):
    """Computes and displays the average image per class."""
    max_per_class = 300
    rng = random.Random(RANDOM_SEED)

    n = len(class_names)
    cols = min(5, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).flatten() if n > 1 else [axes]
    fig.suptitle("Average Image per Class (Class Prototype)", fontsize=13, fontweight="bold")

    for idx, cls_name in enumerate(tqdm(class_names, desc="  Computing averages")):
        images = get_class_images(split_dir, "train", cls_name)
        sample = rng.sample(images, min(max_per_class, len(images)))

        arrays = []
        ref_size = None
        for p in sample:
            arr = load_rgb(p)
            if arr is None:
                continue
            if ref_size is None:
                ref_size = (arr.shape[1], arr.shape[0])
            if (arr.shape[1], arr.shape[0]) != ref_size:
                arr = np.array(Image.fromarray(arr).resize(ref_size, Image.LANCZOS))
            arrays.append(arr.astype(np.float32))

        ax = axes[idx]
        if arrays:
            avg = np.mean(arrays, axis=0).astype(np.uint8)
            ax.imshow(avg)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(short_class_name(cls_name), fontsize=8, pad=3)
        ax.axis("off")

    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    path = output_dir / "average_images_per_class.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_pixel_histograms(split_dir: Path, class_names: list[str], output_dir: Path):
    """Overlaid R/G/B intensity histograms for each split."""
    max_per_split = 500
    rng = random.Random(RANDOM_SEED)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    split_titles = {"train": "Train", "val": "Validation", "test": "Test"}
    channel_colors = ["red", "green", "blue"]
    channel_labels = ["Red", "Green", "Blue"]

    for ax, split in zip(axes, ["train", "val", "test"]):
        # Sample images from all classes in this split
        all_imgs = []
        for cls in class_names:
            imgs = get_class_images(split_dir, split, cls)
            all_imgs.extend(imgs)

        sampled = rng.sample(all_imgs, min(max_per_split, len(all_imgs)))
        pixel_data = [[], [], []]

        for p in sampled:
            arr = load_rgb(p)
            if arr is None:
                continue
            flat = arr.reshape(-1, 3)
            for ch in range(3):
                pixel_data[ch].extend(flat[::4, ch].tolist())  # subsample pixels

        for ch in range(3):
            if pixel_data[ch]:
                ax.hist(pixel_data[ch], bins=64, color=channel_colors[ch],
                        alpha=0.5, label=channel_labels[ch], density=True)

        ax.set_title(f"{split_titles[split]} Split — Pixel Intensity", fontsize=11)
        ax.set_xlabel("Pixel Value (0–255)")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle("RGB Pixel Intensity Distribution per Split", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = output_dir / "pixel_intensity_histograms.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print_section("VISUALIZE DATASET — Skin Disease Detection")
    print(f"Split dataset : {SPLIT_DIR}")
    print(f"Output dir    : {OUTPUT_DIR}")

    if not SPLIT_DIR.exists():
        print(f"[ERROR] Split dataset not found at {SPLIT_DIR}")
        print("  Run 07_split_dataset.py first.")
        sys.exit(1)

    class_names = get_all_class_names(SPLIT_DIR)
    print(f"\nClasses found: {len(class_names)}")
    for c in class_names:
        print(f"  • {c}")

    print_section("Generating visualizations…")

    plot_class_distribution(SPLIT_DIR, class_names, OUTPUT_DIR)
    plot_sample_grid(SPLIT_DIR, class_names, OUTPUT_DIR)
    plot_average_images(SPLIT_DIR, class_names, OUTPUT_DIR)
    plot_pixel_histograms(SPLIT_DIR, class_names, OUTPUT_DIR)

    # ── Summary JSON ───────────────────────────────────────
    summary = {
        "run_timestamp": datetime.now().isoformat(),
        "split_dir": str(SPLIT_DIR),
        "num_classes": len(class_names),
        "outputs": [
            "class_distribution_final.png",
            "sample_image_grid.png",
            "average_images_per_class.png",
            "pixel_intensity_histograms.png",
        ],
    }
    with open(OUTPUT_DIR / "visualization_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print_section("Visualization Complete ✓")
    print(f"  All plots saved to: {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()
