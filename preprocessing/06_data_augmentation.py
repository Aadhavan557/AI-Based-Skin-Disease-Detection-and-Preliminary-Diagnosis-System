"""
06_data_augmentation.py
------------------------
Generates augmented images for under-represented classes to address
class imbalance in the skin disease dataset.

Augmentation techniques applied:
  • Random horizontal / vertical flip
  • Random rotation (±30°)
  • Color jitter (brightness, contrast, saturation, hue)
  • Random zoom (80–120%)
  • Gaussian blur
  • Random affine transformation

Target: Bring every class up to TARGET_PER_CLASS images.
Augmented images are saved alongside originals in dataset_augmented/.
"""

import sys
import shutil
import random
from pathlib import Path
import json
from datetime import datetime

try:
    from PIL import Image, ImageEnhance, ImageFilter
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

RESIZED_DIR = BASE_DIR / "dataset_resized"
ORIGINAL_DIR = BASE_DIR / "dataset"
SOURCE_DIR = RESIZED_DIR if RESIZED_DIR.exists() else ORIGINAL_DIR

OUTPUT_DIR = BASE_DIR / "dataset_augmented"
LOG_DIR = BASE_DIR / "preprocessing" / "outputs" / "06_augmentation"
LOG_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png"}
TARGET_PER_CLASS = 5000   # minimum images per class after augmentation
RANDOM_SEED = 42
MAX_AUG_MULTIPLIER = 10   # max augmented copies per original image

# Augmentation probabilities
AUG_PROB_HFLIP = 0.5
AUG_PROB_VFLIP = 0.2
AUG_PROB_ROTATE = 0.7
AUG_PROB_COLOR = 0.6
AUG_PROB_ZOOM = 0.4
AUG_PROB_BLUR = 0.2

ROTATE_MAX_DEGREES = 30

# ─────────────────────────────────────────────
# Augmentation functions
# ─────────────────────────────────────────────

def augment_image(img: Image.Image, rng: random.Random) -> Image.Image:
    """Applies a random combination of augmentations."""
    img = img.copy()

    # Horizontal flip
    if rng.random() < AUG_PROB_HFLIP:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # Vertical flip
    if rng.random() < AUG_PROB_VFLIP:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    # Rotation
    if rng.random() < AUG_PROB_ROTATE:
        angle = rng.uniform(-ROTATE_MAX_DEGREES, ROTATE_MAX_DEGREES)
        img = img.rotate(angle, resample=Image.BICUBIC, expand=False)

    # Color jitter
    if rng.random() < AUG_PROB_COLOR:
        # Brightness
        factor = rng.uniform(0.7, 1.3)
        img = ImageEnhance.Brightness(img).enhance(factor)
        # Contrast
        factor = rng.uniform(0.8, 1.2)
        img = ImageEnhance.Contrast(img).enhance(factor)
        # Saturation (Color)
        factor = rng.uniform(0.8, 1.2)
        img = ImageEnhance.Color(img).enhance(factor)
        # Sharpness
        factor = rng.uniform(0.8, 1.2)
        img = ImageEnhance.Sharpness(img).enhance(factor)

    # Random zoom (crop then resize back)
    if rng.random() < AUG_PROB_ZOOM:
        w, h = img.size
        zoom = rng.uniform(0.85, 1.0)  # only zoom in
        new_w = int(w * zoom)
        new_h = int(h * zoom)
        left = rng.randint(0, w - new_w)
        top  = rng.randint(0, h - new_h)
        img = img.crop((left, top, left + new_w, top + new_h))
        img = img.resize((w, h), Image.LANCZOS)

    # Gaussian blur
    if rng.random() < AUG_PROB_BLUR:
        radius = rng.uniform(0.3, 1.2)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    return img

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def print_section(title: str):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


def count_images(folder: Path) -> int:
    return sum(1 for p in folder.rglob("*") if p.suffix.lower() in SUPPORTED_FORMATS)


def get_class_image_counts(source_dir: Path) -> dict[str, int]:
    counts = {}
    for cls_dir in sorted(source_dir.iterdir()):
        if cls_dir.is_dir():
            counts[cls_dir.name] = count_images(cls_dir)
    return counts

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print_section("DATA AUGMENTATION — Skin Disease Detection")
    print(f"Source dataset : {SOURCE_DIR}")
    print(f"Output dataset : {OUTPUT_DIR}")
    print(f"Target/class   : {TARGET_PER_CLASS}")

    rng = random.Random(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # ── Count original images per class ───────────────────
    print_section("Current class distribution:")
    class_counts = get_class_image_counts(SOURCE_DIR)
    for cls, cnt in sorted(class_counts.items(), key=lambda x: x[1]):
        needed = max(0, TARGET_PER_CLASS - cnt)
        bar = "▓" * (cnt // 400) + "░" * (needed // 400)
        print(f"  {cls[:52]:<54} {cnt:>5}  (+{needed} needed)")

    # ── Copy all originals first ───────────────────────────
    print_section("Copying originals to output directory…")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for cls_dir in tqdm(sorted(SOURCE_DIR.iterdir()), desc="Copying classes"):
        if not cls_dir.is_dir():
            continue
        out_cls_dir = OUTPUT_DIR / cls_dir.name
        if out_cls_dir.exists():
            shutil.rmtree(out_cls_dir)
        shutil.copytree(cls_dir, out_cls_dir)

    print(f"  Originals copied to {OUTPUT_DIR}")

    # ── Augment under-represented classes ─────────────────
    print_section("Generating augmented images…")
    aug_summary = {}

    for cls_dir in sorted(SOURCE_DIR.iterdir()):
        if not cls_dir.is_dir():
            continue

        cls_name = cls_dir.name
        current_count = class_counts.get(cls_name, 0)
        needed = TARGET_PER_CLASS - current_count

        if needed <= 0:
            print(f"  ✓ {cls_name[:52]:<54} — no augmentation needed ({current_count} images)")
            aug_summary[cls_name] = {"original": current_count, "augmented": 0, "total": current_count}
            continue

        source_images = [
            p for p in cls_dir.rglob("*")
            if p.suffix.lower() in SUPPORTED_FORMATS
        ]

        out_cls_dir = OUTPUT_DIR / cls_name
        out_cls_dir.mkdir(parents=True, exist_ok=True)

        generated = 0
        aug_idx = 0

        print(f"  Augmenting: {cls_name[:45]:<47} ({current_count} → {TARGET_PER_CLASS}, need +{needed})")

        with tqdm(total=needed, desc=f"  {cls_name[:30]}", leave=False, unit="img") as pbar:
            while generated < needed:
                # Pick a random source image
                src_path = rng.choice(source_images)
                try:
                    with Image.open(src_path).convert("RGB") as img:
                        aug_img = augment_image(img, rng)
                        aug_name = f"aug_{aug_idx:06d}_{src_path.stem}.jpg"
                        aug_img.save(out_cls_dir / aug_name, "JPEG", quality=90)
                        generated += 1
                        aug_idx += 1
                        pbar.update(1)
                except Exception as e:
                    continue  # skip bad files silently

        aug_summary[cls_name] = {
            "original": current_count,
            "augmented": generated,
            "total": current_count + generated,
        }

    # ── Report ─────────────────────────────────────────────
    print_section("Augmentation Results")
    total_orig = sum(v["original"] for v in aug_summary.values())
    total_aug  = sum(v["augmented"] for v in aug_summary.values())
    total_new  = sum(v["total"] for v in aug_summary.values())

    print(f"  {'Class':<55} {'Original':>8}  {'Augmented':>9}  {'Total':>7}")
    print(f"  {'─'*55}  {'─'*8}  {'─'*9}  {'─'*7}")
    for cls, info in sorted(aug_summary.items()):
        print(f"  {cls[:55]:<55} {info['original']:>8}  {info['augmented']:>9}  {info['total']:>7}")
    print(f"  {'─'*55}  {'─'*8}  {'─'*9}  {'─'*7}")
    print(f"  {'TOTAL':<55} {total_orig:>8}  {total_aug:>9}  {total_new:>7}")

    # ── Visualization: before vs after ────────────────────
    cls_names_short = [n.split(".")[0][:20] for n in aug_summary]
    originals = [aug_summary[k]["original"] for k in aug_summary]
    totals    = [aug_summary[k]["total"] for k in aug_summary]
    augmented = [aug_summary[k]["augmented"] for k in aug_summary]

    x = np.arange(len(cls_names_short))
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x, originals, label="Original", color="#3498db", alpha=0.8)
    ax.bar(x, augmented, bottom=originals, label="Augmented", color="#e67e22", alpha=0.8)
    ax.axhline(TARGET_PER_CLASS, color="red", linestyle="--", linewidth=1.5, label=f"Target ({TARGET_PER_CLASS})")
    ax.set_xticks(x)
    ax.set_xticklabels(cls_names_short, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Number of Images")
    ax.set_title("Class Distribution: Before and After Augmentation", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plot_path = LOG_DIR / "augmentation_distribution.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved: {plot_path}")

    # ── Sample augmented images visualization ─────────────
    sample_cls = next(iter(aug_summary))
    sample_cls_dir = OUTPUT_DIR / sample_cls
    aug_samples = [p for p in sample_cls_dir.glob("aug_*") if p.suffix.lower() == ".jpg"][:8]
    if aug_samples:
        fig, axes = plt.subplots(2, 4, figsize=(14, 7))
        fig.suptitle(f"Sample Augmented Images — {sample_cls[:40]}", fontsize=12, fontweight="bold")
        for i, ax in enumerate(axes.flat):
            if i < len(aug_samples):
                with Image.open(aug_samples[i]) as img:
                    ax.imshow(img)
                ax.set_title(f"aug_{i+1}", fontsize=8)
            ax.axis("off")
        plt.tight_layout()
        sample_path = LOG_DIR / "augmented_samples.png"
        plt.savefig(sample_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Sample images saved: {sample_path}")

    # ── Save summary ───────────────────────────────────────
    full_summary = {
        "run_timestamp": datetime.now().isoformat(),
        "source_dir": str(SOURCE_DIR),
        "output_dir": str(OUTPUT_DIR),
        "target_per_class": TARGET_PER_CLASS,
        "total_original": total_orig,
        "total_augmented": total_aug,
        "total_final": total_new,
        "class_breakdown": aug_summary,
    }
    with open(LOG_DIR / "augmentation_summary.json", "w") as f:
        json.dump(full_summary, f, indent=2)
    print(f"  Summary saved: {LOG_DIR / 'augmentation_summary.json'}")

    print_section("Data Augmentation Complete ✓")
    print(f"  Augmented dataset saved to: {OUTPUT_DIR}")
    print(f"  Total images: {total_orig} → {total_new}\n")


if __name__ == "__main__":
    main()
