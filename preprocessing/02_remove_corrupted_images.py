"""
02_remove_corrupted_images.py
------------------------------
Scans the dataset for corrupted, truncated, or unreadable images
and moves them to a quarantine folder for review (not hard-deleted).

Checks performed:
  • PIL cannot open the file
  • File size is 0 bytes or suspiciously small (< 1 KB)
  • Image cannot be fully loaded / is truncated
  • Image mode is not RGB/RGBA/L (incompatible for training)
"""

import os
import sys
import shutil
from pathlib import Path
import json
from datetime import datetime

try:
    from PIL import Image, UnidentifiedImageError
    import pandas as pd
    from tqdm import tqdm
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    print("Install with: pip install pillow pandas tqdm")
    sys.exit(1)

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = False  # We want to detect truncated images

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
QUARANTINE_DIR = BASE_DIR / "preprocessing" / "quarantine" / "corrupted"
OUTPUT_DIR = BASE_DIR / "preprocessing" / "outputs" / "02_corrupted"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
MIN_FILE_SIZE_BYTES = 1024  # < 1 KB is suspicious
ALLOWED_MODES = {"RGB", "RGBA", "L", "P"}  # P (palette) can be converted

DRY_RUN = False  # Set True to simulate without moving files

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def print_section(title: str):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


def is_corrupted(img_path: Path) -> tuple[bool, str]:
    """
    Returns (is_bad, reason).
    Reason is an empty string if the image is healthy.
    """
    # Check 1: file size
    size = img_path.stat().st_size
    if size == 0:
        return True, "zero_byte_file"
    if size < MIN_FILE_SIZE_BYTES:
        return True, f"file_too_small_{size}b"

    # Check 2: PIL open
    try:
        with Image.open(img_path) as img:
            # Check 3: mode
            if img.mode not in ALLOWED_MODES:
                return True, f"unsupported_mode_{img.mode}"
            # Check 4: fully load (detects truncated JPEG etc.)
            img.verify()
    except UnidentifiedImageError:
        return True, "unidentified_image_format"
    except Exception as e:
        return True, f"load_error_{type(e).__name__}"

    # Second open needed after verify() (PIL resets stream)
    try:
        with Image.open(img_path) as img:
            img.load()
    except Exception as e:
        return True, f"truncated_or_load_failed_{type(e).__name__}"

    return False, ""


def quarantine_file(img_path: Path, class_name: str, reason: str):
    """Moves a corrupted image to the quarantine folder."""
    dest_dir = QUARANTINE_DIR / class_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / img_path.name
    # Avoid name collision
    if dest.exists():
        dest = dest_dir / f"{img_path.stem}_{img_path.stat().st_size}{img_path.suffix}"
    if not DRY_RUN:
        shutil.move(str(img_path), str(dest))
    return dest

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print_section("REMOVE CORRUPTED IMAGES — Skin Disease Detection")
    print(f"Dataset    : {DATASET_DIR}")
    print(f"Quarantine : {QUARANTINE_DIR}")
    print(f"Dry run    : {DRY_RUN}")

    # Collect all image paths
    all_images = [
        p for p in DATASET_DIR.rglob("*")
        if p.suffix.lower() in SUPPORTED_FORMATS and p.is_file()
    ]
    print(f"\nTotal images to check: {len(all_images)}")

    corrupted_records = []
    healthy_count = 0

    for img_path in tqdm(all_images, desc="Scanning images", unit="img"):
        class_name = img_path.parent.name
        bad, reason = is_corrupted(img_path)

        if bad:
            dest = quarantine_file(img_path, class_name, reason)
            corrupted_records.append({
                "original_path": str(img_path),
                "class": class_name,
                "filename": img_path.name,
                "reason": reason,
                "quarantined_to": str(dest),
                "file_size_bytes": img_path.stat().st_size if img_path.exists() else 0,
                "timestamp": datetime.now().isoformat(),
            })
        else:
            healthy_count += 1

    # ── Report ─────────────────────────────────────────────
    print_section("Results")
    total_corrupted = len(corrupted_records)
    print(f"  Total images scanned : {len(all_images)}")
    print(f"  Healthy images       : {healthy_count}")
    print(f"  Corrupted / moved    : {total_corrupted}")

    if corrupted_records:
        print("\n  Breakdown by reason:")
        reason_counts: dict = {}
        for r in corrupted_records:
            reason_counts[r["reason"]] = reason_counts.get(r["reason"], 0) + 1
        for reason, cnt in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print(f"    {reason:<40} {cnt}")

        print("\n  Breakdown by class:")
        class_counts: dict = {}
        for r in corrupted_records:
            class_counts[r["class"]] = class_counts.get(r["class"], 0) + 1
        for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
            print(f"    {cls:<55} {cnt}")

    # ── Save report ────────────────────────────────────────
    if corrupted_records:
        df = pd.DataFrame(corrupted_records)
        csv_path = OUTPUT_DIR / "corrupted_images_log.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n  Corrupted image log saved: {csv_path}")

    summary = {
        "run_timestamp": datetime.now().isoformat(),
        "dry_run": DRY_RUN,
        "total_scanned": len(all_images),
        "healthy": healthy_count,
        "corrupted_removed": total_corrupted,
        "quarantine_dir": str(QUARANTINE_DIR),
    }
    json_path = OUTPUT_DIR / "removal_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary JSON saved  : {json_path}")

    print_section("Corrupted Image Removal Complete ✓")
    if DRY_RUN:
        print("  ⚠  DRY RUN — no files were actually moved.")
    print()


if __name__ == "__main__":
    main()
