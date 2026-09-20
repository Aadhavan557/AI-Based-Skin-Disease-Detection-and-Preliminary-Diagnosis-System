"""
04_resize_images.py
--------------------
Resizes all dataset images to a uniform target resolution while
preserving aspect ratio (with center-crop or padding strategy).

Strategy options (set RESIZE_STRATEGY below):
  • "stretch"     — force exact size (may distort)
  • "pad"         — resize to fit, then pad with black to exact size
  • "center_crop" — resize shortest side, then center-crop to exact size (default)

Images are saved to a new directory (dataset_resized/) to preserve originals.
"""

import os
import sys
import shutil
from pathlib import Path
import json
from datetime import datetime

try:
    from PIL import Image, ImageOps
    from tqdm import tqdm
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    print("Install with: pip install pillow tqdm pandas numpy matplotlib")
    sys.exit(1)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_DATASET_DIR = BASE_DIR / "dataset_resized"   # resized images go here
LOG_DIR = BASE_DIR / "preprocessing" / "outputs" / "04_resize"
LOG_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# Target resolution (standard for skin disease models)
TARGET_WIDTH = 224
TARGET_HEIGHT = 224

RESIZE_STRATEGY = "center_crop"  # "stretch" | "pad" | "center_crop"
OUTPUT_FORMAT = "JPEG"            # JPEG saves space; use PNG for lossless
JPEG_QUALITY = 95                 # 1-95; ignored for PNG
CONVERT_TO_RGB = True             # Convert all to RGB (removes alpha)

# ─────────────────────────────────────────────
# Resize functions
# ─────────────────────────────────────────────

def resize_stretch(img: Image.Image, w: int, h: int) -> Image.Image:
    return img.resize((w, h), Image.LANCZOS)


def resize_pad(img: Image.Image, w: int, h: int) -> Image.Image:
    img.thumbnail((w, h), Image.LANCZOS)
    padded = Image.new("RGB", (w, h), (0, 0, 0))
    offset = ((w - img.width) // 2, (h - img.height) // 2)
    padded.paste(img, offset)
    return padded


def resize_center_crop(img: Image.Image, w: int, h: int) -> Image.Image:
    # Scale so the shortest side equals min(w, h)
    ratio = max(w / img.width, h / img.height)
    new_w = int(img.width * ratio)
    new_h = int(img.height * ratio)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    # Center crop
    left = (new_w - w) // 2
    top = (new_h - h) // 2
    img = img.crop((left, top, left + w, top + h))
    return img


RESIZE_FN = {
    "stretch": resize_stretch,
    "pad": resize_pad,
    "center_crop": resize_center_crop,
}

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def print_section(title: str):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


def get_output_ext() -> str:
    return ".jpg" if OUTPUT_FORMAT == "JPEG" else ".png"


def save_image(img: Image.Image, dest_path: Path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"format": OUTPUT_FORMAT}
    if OUTPUT_FORMAT == "JPEG":
        save_kwargs["quality"] = JPEG_QUALITY
        save_kwargs["optimize"] = True
    img.save(dest_path, **save_kwargs)

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print_section("RESIZE IMAGES — Skin Disease Detection")
    print(f"Source dataset   : {DATASET_DIR}")
    print(f"Output dataset   : {OUTPUT_DATASET_DIR}")
    print(f"Target size      : {TARGET_WIDTH} × {TARGET_HEIGHT}")
    print(f"Strategy         : {RESIZE_STRATEGY}")
    print(f"Output format    : {OUTPUT_FORMAT}")

    if RESIZE_STRATEGY not in RESIZE_FN:
        print(f"[ERROR] Unknown strategy: {RESIZE_STRATEGY}")
        sys.exit(1)

    resize_fn = RESIZE_FN[RESIZE_STRATEGY]

    # Collect all source images
    all_images = [
        p for p in DATASET_DIR.rglob("*")
        if p.suffix.lower() in SUPPORTED_FORMATS and p.is_file()
    ]
    print(f"\nImages to resize: {len(all_images)}")

    success_count = 0
    error_count = 0
    error_records = []
    size_before = []
    size_after = []

    for img_path in tqdm(all_images, desc="Resizing", unit="img"):
        # Mirror directory structure
        rel_path = img_path.relative_to(DATASET_DIR)
        dest_path = OUTPUT_DATASET_DIR / rel_path.parent / (rel_path.stem + get_output_ext())

        try:
            with Image.open(img_path) as img:
                orig_w, orig_h = img.size
                size_before.append((orig_w, orig_h))

                if CONVERT_TO_RGB:
                    img = img.convert("RGB")

                resized = resize_fn(img, TARGET_WIDTH, TARGET_HEIGHT)
                save_image(resized, dest_path)
                size_after.append((resized.width, resized.height))
                success_count += 1

        except Exception as e:
            error_count += 1
            error_records.append({
                "path": str(img_path),
                "error": str(e),
            })

    # ── Statistics ─────────────────────────────────────────
    print_section("Results")
    print(f"  Successfully resized : {success_count}")
    print(f"  Errors               : {error_count}")

    if size_before:
        widths_before = [s[0] for s in size_before]
        heights_before = [s[1] for s in size_before]
        print(f"\n  Original size (mean) : {np.mean(widths_before):.0f} × {np.mean(heights_before):.0f}")
        print(f"  Original size (min)  : {min(widths_before)} × {min(heights_before)}")
        print(f"  Original size (max)  : {max(widths_before)} × {max(heights_before)}")
        print(f"  All output images    : {TARGET_WIDTH} × {TARGET_HEIGHT}")

    # ── Visualization ──────────────────────────────────────
    if size_before:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Image Size Distribution: Before vs After Resize", fontsize=14, fontweight="bold")

        axes[0].scatter(widths_before, heights_before, alpha=0.1, s=5, color="steelblue")
        axes[0].set_title("Before Resize")
        axes[0].set_xlabel("Width (px)")
        axes[0].set_ylabel("Height (px)")
        axes[0].grid(alpha=0.3)
        axes[0].axvline(TARGET_WIDTH, color="red", linestyle="--", label=f"Target {TARGET_WIDTH}")
        axes[0].axhline(TARGET_HEIGHT, color="orange", linestyle="--", label=f"Target {TARGET_HEIGHT}")
        axes[0].legend()

        axes[1].scatter([TARGET_WIDTH] * len(size_after), [TARGET_HEIGHT] * len(size_after),
                        alpha=0.05, s=5, color="green")
        axes[1].set_title(f"After Resize ({TARGET_WIDTH}×{TARGET_HEIGHT})")
        axes[1].set_xlabel("Width (px)")
        axes[1].set_ylabel("Height (px)")
        axes[1].set_xlim(TARGET_WIDTH - 10, TARGET_WIDTH + 10)
        axes[1].set_ylim(TARGET_HEIGHT - 10, TARGET_HEIGHT + 10)
        axes[1].grid(alpha=0.3)
        axes[1].text(TARGET_WIDTH, TARGET_HEIGHT, f"All {success_count}\nimages here",
                     ha="center", va="center", fontsize=12, color="green",
                     bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

        plt.tight_layout()
        plot_path = LOG_DIR / "resize_distribution.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Plot saved: {plot_path}")

    # ── Save logs ──────────────────────────────────────────
    if error_records:
        df = pd.DataFrame(error_records)
        df.to_csv(LOG_DIR / "resize_errors.csv", index=False)

    summary = {
        "run_timestamp": datetime.now().isoformat(),
        "source_dataset": str(DATASET_DIR),
        "output_dataset": str(OUTPUT_DATASET_DIR),
        "target_width": TARGET_WIDTH,
        "target_height": TARGET_HEIGHT,
        "resize_strategy": RESIZE_STRATEGY,
        "output_format": OUTPUT_FORMAT,
        "convert_to_rgb": CONVERT_TO_RGB,
        "total_images": len(all_images),
        "success": success_count,
        "errors": error_count,
    }
    with open(LOG_DIR / "resize_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print_section("Resize Complete ✓")
    print(f"  Resized images saved to: {OUTPUT_DATASET_DIR}\n")


if __name__ == "__main__":
    main()
