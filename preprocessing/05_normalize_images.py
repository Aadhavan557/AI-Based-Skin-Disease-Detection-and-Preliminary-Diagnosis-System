"""
05_normalize_images.py
-----------------------
Computes dataset-wide channel mean and standard deviation, then verifies
normalization parameters for use during model training.

Steps:
  1. Sample images from dataset_resized/ (or original dataset/ if not resized)
  2. Compute per-channel (R, G, B) mean and std across the entire dataset
  3. Save these stats to a JSON file for use by the training pipeline
  4. Optionally, validate by sampling a batch and showing normalized values

Note: Normalization is applied at training time via tf.keras preprocessing
layers or torchvision transforms — we only COMPUTE the stats here.
We do NOT write new image files in this step.
"""

import sys
import io
from pathlib import Path
import json
from datetime import datetime

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

try:
    from PIL import Image
    import numpy as np
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    print("Install with: pip install pillow numpy tqdm matplotlib")
    sys.exit(1)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# Prefer resized dataset if it exists
RESIZED_DIR = BASE_DIR / "dataset_resized"
ORIGINAL_DIR = BASE_DIR / "dataset"
DATASET_DIR = RESIZED_DIR if RESIZED_DIR.exists() else ORIGINAL_DIR

OUTPUT_DIR = BASE_DIR / "preprocessing" / "outputs" / "05_normalize"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# For very large datasets, use a random sample to estimate stats
MAX_SAMPLE_SIZE = 10_000   # Set to None to use entire dataset
RANDOM_SEED = 42

# ImageNet baseline (used as comparison / fallback)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def print_section(title: str):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


def load_image_array(img_path: Path) -> np.ndarray | None:
    """Load image as float32 RGB array in [0, 1]."""
    try:
        with Image.open(img_path).convert("RGB") as img:
            arr = np.array(img, dtype=np.float32) / 255.0
        return arr
    except Exception:
        return None

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print_section("COMPUTE NORMALIZATION STATISTICS — Skin Disease Detection")
    print(f"Dataset dir  : {DATASET_DIR}")
    print(f"Output dir   : {OUTPUT_DIR}")

    # Collect all image paths
    all_images = [
        p for p in DATASET_DIR.rglob("*")
        if p.suffix.lower() in SUPPORTED_FORMATS and p.is_file()
    ]
    print(f"\nTotal images available: {len(all_images)}")

    # Optionally sample
    rng = np.random.default_rng(RANDOM_SEED)
    if MAX_SAMPLE_SIZE and len(all_images) > MAX_SAMPLE_SIZE:
        indices = rng.choice(len(all_images), MAX_SAMPLE_SIZE, replace=False)
        sample_paths = [all_images[i] for i in sorted(indices)]
        print(f"Sampling {MAX_SAMPLE_SIZE} images for efficiency.")
    else:
        sample_paths = all_images
        print(f"Using all {len(sample_paths)} images.")

    # ── Pass 1: Compute mean ───────────────────────────────
    print_section("Pass 1: Computing channel means…")
    channel_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    valid_count = 0

    for img_path in tqdm(sample_paths, desc="Mean pass", unit="img"):
        arr = load_image_array(img_path)
        if arr is None:
            continue
        h, w, c = arr.shape
        channel_sum += arr.reshape(-1, 3).sum(axis=0)
        pixel_count += h * w
        valid_count += 1

    if pixel_count == 0:
        print("[ERROR] No valid images found.")
        sys.exit(1)

    channel_mean = channel_sum / pixel_count
    print(f"  Valid images processed : {valid_count}")
    print(f"  Mean (R, G, B)         : {channel_mean}")

    # ── Pass 2: Compute std ────────────────────────────────
    print_section("Pass 2: Computing channel standard deviations…")
    channel_var_sum = np.zeros(3, dtype=np.float64)

    for img_path in tqdm(sample_paths, desc="Std pass", unit="img"):
        arr = load_image_array(img_path)
        if arr is None:
            continue
        pixels = arr.reshape(-1, 3)  # (N, 3)
        channel_var_sum += ((pixels - channel_mean) ** 2).sum(axis=0)

    channel_std = np.sqrt(channel_var_sum / pixel_count)
    print(f"  Std  (R, G, B)         : {channel_std}")

    # ── Per-class stats ────────────────────────────────────
    print_section("Computing per-class statistics…")
    class_stats = {}
    for cls_dir in sorted(DATASET_DIR.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls_images = [
            p for p in cls_dir.rglob("*")
            if p.suffix.lower() in SUPPORTED_FORMATS and p.is_file()
        ]
        cls_sum = np.zeros(3, dtype=np.float64)
        cls_px = 0
        for img_path in cls_images[:500]:  # cap per class for speed
            arr = load_image_array(img_path)
            if arr is None:
                continue
            cls_sum += arr.reshape(-1, 3).sum(axis=0)
            cls_px += arr.shape[0] * arr.shape[1]
        if cls_px > 0:
            cls_mean = cls_sum / cls_px
            class_stats[cls_dir.name] = {
                "mean_r": float(cls_mean[0]),
                "mean_g": float(cls_mean[1]),
                "mean_b": float(cls_mean[2]),
                "num_images": len(cls_images),
            }
            print(f"  {cls_dir.name[:50]:<52} mean={cls_mean.round(3)}")

    # ── Comparison with ImageNet ───────────────────────────
    print_section("Comparison with ImageNet Statistics")
    print(f"  Dataset Mean : R={channel_mean[0]:.4f}  G={channel_mean[1]:.4f}  B={channel_mean[2]:.4f}")
    print(f"  ImageNet Mean: R={IMAGENET_MEAN[0]:.4f}  G={IMAGENET_MEAN[1]:.4f}  B={IMAGENET_MEAN[2]:.4f}")
    print(f"  Dataset Std  : R={channel_std[0]:.4f}  G={channel_std[1]:.4f}  B={channel_std[2]:.4f}")
    print(f"  ImageNet Std : R={IMAGENET_STD[0]:.4f}  G={IMAGENET_STD[1]:.4f}  B={IMAGENET_STD[2]:.4f}")

    diff_mean = np.abs(channel_mean - IMAGENET_MEAN)
    diff_std  = np.abs(channel_std  - IMAGENET_STD)
    print(f"\n  Mean difference from ImageNet : {diff_mean.mean():.4f}")
    print(f"  Std difference from ImageNet  : {diff_std.mean():.4f}")
    if diff_mean.mean() > 0.05:
        print("  ⚠  Dataset mean differs notably from ImageNet — use dataset-specific stats.")
    else:
        print("  ✓  Dataset mean is close to ImageNet — ImageNet normalization may work well.")

    # ── Visualization ──────────────────────────────────────
    channels = ["Red", "Green", "Blue"]
    colors = ["#e74c3c", "#2ecc71", "#3498db"]
    x = np.arange(3)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Normalization Statistics: Dataset vs ImageNet", fontsize=14, fontweight="bold")

    width = 0.35
    axes[0].bar(x - width/2, channel_mean, width, label="Dataset", color=colors, alpha=0.85)
    axes[0].bar(x + width/2, IMAGENET_MEAN, width, label="ImageNet", color=colors, alpha=0.4, hatch="//")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(channels)
    axes[0].set_title("Channel Mean")
    axes[0].set_ylabel("Mean pixel value (0-1)")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x - width/2, channel_std, width, label="Dataset", color=colors, alpha=0.85)
    axes[1].bar(x + width/2, IMAGENET_STD, width, label="ImageNet", color=colors, alpha=0.4, hatch="//")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(channels)
    axes[1].set_title("Channel Std Dev")
    axes[1].set_ylabel("Std deviation")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "normalization_stats.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved: {plot_path}")

    # ── Save normalization config ──────────────────────────
    norm_config = {
        "computed_at": datetime.now().isoformat(),
        "dataset_dir": str(DATASET_DIR),
        "sample_size": valid_count,
        "dataset_stats": {
            "mean": channel_mean.tolist(),
            "std": channel_std.tolist(),
            "mean_r": float(channel_mean[0]),
            "mean_g": float(channel_mean[1]),
            "mean_b": float(channel_mean[2]),
            "std_r": float(channel_std[0]),
            "std_g": float(channel_std[1]),
            "std_b": float(channel_std[2]),
        },
        "imagenet_stats": {
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
        },
        "recommendation": "dataset" if diff_mean.mean() > 0.05 else "imagenet_ok",
        "per_class_stats": class_stats,
        "usage": {
            "tensorflow_keras": (
                "tf.keras.layers.Normalization(mean=[R_mean, G_mean, B_mean], "
                "variance=[R_std**2, G_std**2, B_std**2])"
            ),
            "pytorch_transforms": (
                "transforms.Normalize(mean=[R_mean, G_mean, B_mean], "
                "std=[R_std, G_std, B_std])"
            ),
        },
    }

    json_path = OUTPUT_DIR / "normalization_config.json"
    with open(json_path, "w") as f:
        json.dump(norm_config, f, indent=2)
    print(f"  Normalization config saved: {json_path}")

    print_section("Normalization Statistics Complete ✓")
    print(f"  Use these values in training:")
    print(f"    mean = {channel_mean.round(4).tolist()}")
    print(f"    std  = {channel_std.round(4).tolist()}")
    print()


if __name__ == "__main__":
    main()
