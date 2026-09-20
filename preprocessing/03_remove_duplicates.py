"""
03_remove_duplicates.py
------------------------
Detects and removes near-duplicate images within the dataset using
perceptual hashing (pHash). Keeps one copy and quarantines the rest.

Strategy:
  • Compute pHash for every image
  • Group images with Hamming distance ≤ HASH_THRESHOLD as duplicates
  • Within each duplicate group, keep the largest file (best quality)
  • Move duplicates to a quarantine folder
"""

import os
import sys
import shutil
from pathlib import Path
from collections import defaultdict
import json
from datetime import datetime

try:
    from PIL import Image
    import imagehash
    import pandas as pd
    from tqdm import tqdm
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    print("Install with: pip install pillow imagehash pandas tqdm")
    sys.exit(1)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
QUARANTINE_DIR = BASE_DIR / "preprocessing" / "quarantine" / "duplicates"
OUTPUT_DIR = BASE_DIR / "preprocessing" / "outputs" / "03_duplicates"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
HASH_THRESHOLD = 8    # Hamming distance ≤ 8 → near duplicate (0 = exact)
HASH_SIZE = 16        # pHash resolution (higher = more precise but slower)

DRY_RUN = False  # Set True to simulate without moving files

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def print_section(title: str):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


def compute_phash(img_path: Path) -> imagehash.ImageHash | None:
    """Returns perceptual hash or None if the image can't be opened."""
    try:
        with Image.open(img_path).convert("RGB") as img:
            return imagehash.phash(img, hash_size=HASH_SIZE)
    except Exception:
        return None


def find_duplicate_groups(hash_map: dict[Path, imagehash.ImageHash]) -> list[list[Path]]:
    """
    Groups images whose pHash Hamming distance is ≤ HASH_THRESHOLD.
    Uses union-find for efficiency.
    """
    paths = list(hash_map.keys())
    n = len(paths)

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    print("  Computing pairwise distances…")
    # For large datasets, use a bucket approach based on hash prefix
    # to avoid O(n²) comparisons
    for i in tqdm(range(n), desc="Grouping duplicates", unit="img"):
        for j in range(i + 1, n):
            dist = hash_map[paths[i]] - hash_map[paths[j]]
            if dist <= HASH_THRESHOLD:
                union(i, j)

    # Collect groups
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    # Only return groups with more than 1 image
    return [[paths[i] for i in grp] for grp in groups.values() if len(grp) > 1]


def select_keeper(group: list[Path]) -> Path:
    """Keeps the file with the largest size (proxy for best quality)."""
    return max(group, key=lambda p: p.stat().st_size)


def quarantine_file(img_path: Path, class_name: str):
    """Moves a duplicate image to the quarantine folder."""
    dest_dir = QUARANTINE_DIR / class_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / img_path.name
    if dest.exists():
        dest = dest_dir / f"{img_path.stem}_{img_path.stat().st_size}{img_path.suffix}"
    if not DRY_RUN:
        shutil.move(str(img_path), str(dest))
    return dest

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print_section("REMOVE DUPLICATES — Skin Disease Detection")
    print(f"Dataset      : {DATASET_DIR}")
    print(f"Quarantine   : {QUARANTINE_DIR}")
    print(f"Hash threshold (Hamming): {HASH_THRESHOLD}")
    print(f"Dry run      : {DRY_RUN}")

    # Collect all image paths
    all_images = [
        p for p in DATASET_DIR.rglob("*")
        if p.suffix.lower() in SUPPORTED_FORMATS and p.is_file()
    ]
    print(f"\nTotal images to process: {len(all_images)}")

    # ── Step 1: Compute hashes ─────────────────────────────
    print_section("Computing perceptual hashes…")
    hash_map: dict[Path, imagehash.ImageHash] = {}
    failed = []

    for img_path in tqdm(all_images, desc="Hashing", unit="img"):
        h = compute_phash(img_path)
        if h is not None:
            hash_map[img_path] = h
        else:
            failed.append(img_path)

    print(f"  Hashed successfully : {len(hash_map)}")
    print(f"  Failed (skip)       : {len(failed)}")

    # ── Step 2: Find duplicate groups ─────────────────────
    print_section("Finding duplicate groups…")

    # Warn about large datasets — O(n²) can be slow
    if len(hash_map) > 5000:
        print(f"  ⚠  Large dataset ({len(hash_map)} images). This may take several minutes…")

    dup_groups = find_duplicate_groups(hash_map)
    total_duplicates = sum(len(g) - 1 for g in dup_groups)  # -1 for keeper
    print(f"\n  Duplicate groups found : {len(dup_groups)}")
    print(f"  Images to remove       : {total_duplicates}")

    # ── Step 3: Quarantine duplicates ─────────────────────
    print_section("Quarantining duplicates…")
    removal_records = []

    for group in tqdm(dup_groups, desc="Processing groups", unit="group"):
        keeper = select_keeper(group)
        for img_path in group:
            if img_path == keeper:
                continue
            class_name = img_path.parent.name
            dest = quarantine_file(img_path, class_name)
            removal_records.append({
                "removed_path": str(img_path),
                "class": class_name,
                "filename": img_path.name,
                "kept_path": str(keeper),
                "quarantined_to": str(dest),
                "timestamp": datetime.now().isoformat(),
            })

    # ── Report ─────────────────────────────────────────────
    print_section("Results")
    print(f"  Total images scanned    : {len(all_images)}")
    print(f"  Duplicate groups        : {len(dup_groups)}")
    print(f"  Images quarantined      : {len(removal_records)}")
    print(f"  Remaining unique images : {len(all_images) - len(removal_records)}")

    # Per-class breakdown
    if removal_records:
        class_counts: dict = {}
        for r in removal_records:
            class_counts[r["class"]] = class_counts.get(r["class"], 0) + 1
        print("\n  Duplicates removed per class:")
        for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
            print(f"    {cls:<55} {cnt}")

    # ── Save report ────────────────────────────────────────
    if removal_records:
        df = pd.DataFrame(removal_records)
        csv_path = OUTPUT_DIR / "duplicates_removed_log.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n  Log CSV saved: {csv_path}")

    summary = {
        "run_timestamp": datetime.now().isoformat(),
        "dry_run": DRY_RUN,
        "hash_threshold": HASH_THRESHOLD,
        "hash_size": HASH_SIZE,
        "total_scanned": len(all_images),
        "hashed_successfully": len(hash_map),
        "duplicate_groups": len(dup_groups),
        "duplicates_removed": len(removal_records),
        "unique_remaining": len(all_images) - len(removal_records),
    }
    json_path = OUTPUT_DIR / "dedup_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary JSON saved: {json_path}")

    print_section("Duplicate Removal Complete ✓")
    if DRY_RUN:
        print("  ⚠  DRY RUN — no files were actually moved.")
    print()


if __name__ == "__main__":
    main()
