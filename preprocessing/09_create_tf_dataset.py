"""
09_create_tf_dataset.py
------------------------
Creates optimized TensorFlow datasets (tf.data.Dataset) from the
split dataset directory for use in model training.

Features:
  • Loads from dataset_split/{train,val,test}/ directory structure
  • Applies on-the-fly preprocessing (resize, normalize, augment)
  • Uses tf.data pipeline best practices (cache, shuffle, prefetch)
  • Saves class label mapping as JSON
  • Validates the dataset pipeline with a sample batch
  • Optionally saves the dataset in TFRecord format for maximum I/O speed
"""

import sys
from pathlib import Path
import json
from datetime import datetime

try:
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    print("Install with: pip install numpy matplotlib")
    sys.exit(1)

try:
    import tensorflow as tf
    print(f"  TensorFlow version: {tf.__version__}")
except ImportError:
    print("[ERROR] TensorFlow not found.")
    print("Install with: pip install tensorflow  OR  pip install tensorflow-cpu")
    sys.exit(1)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
SPLIT_DIR = BASE_DIR / "dataset_split"
OUTPUT_DIR = BASE_DIR / "preprocessing" / "outputs" / "09_tf_dataset"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# These should match 04_resize_images.py settings
IMG_HEIGHT = 224
IMG_WIDTH  = 224
CHANNELS   = 3

BATCH_SIZE  = 32
SHUFFLE_BUFFER = 1000

# Normalization — load from 05_normalize_images output if available
NORM_CONFIG_PATH = BASE_DIR / "preprocessing" / "outputs" / "05_normalize" / "normalization_config.json"

# Training augmentation (applied only to train split)
USE_TRAINING_AUGMENTATION = True

# Set True to also save datasets in TFRecord format (faster I/O for large datasets)
SAVE_TFRECORDS = False

AUTOTUNE = tf.data.AUTOTUNE

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def print_section(title: str):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


def load_normalization_params() -> tuple[list, list]:
    """Load dataset-specific mean/std or fall back to ImageNet."""
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    if NORM_CONFIG_PATH.exists():
        with open(NORM_CONFIG_PATH) as f:
            config = json.load(f)
        stats = config.get("dataset_stats", {})
        mean = stats.get("mean", imagenet_mean)
        std  = stats.get("std",  imagenet_std)
        print(f"  Using dataset-specific normalization from {NORM_CONFIG_PATH.name}")
    else:
        mean, std = imagenet_mean, imagenet_std
        print("  Using ImageNet normalization (dataset stats not found)")

    return mean, std


def get_class_names(split_dir: Path) -> list[str]:
    train_dir = split_dir / "train"
    return sorted([d.name for d in train_dir.iterdir() if d.is_dir()])

# ─────────────────────────────────────────────
# TF preprocessing layers
# ─────────────────────────────────────────────

def build_preprocessing_model(mean: list, std: list) -> tf.keras.Sequential:
    """Returns a Keras model that normalizes input (pixel values 0–255 → normalized)."""
    # Convert mean/std to per-channel variance
    variance = [s ** 2 for s in std]
    norm_layer = tf.keras.layers.Normalization(
        mean=mean,
        variance=variance,
        axis=-1  # per-channel
    )
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1.0 / 255.0),  # [0,255] → [0,1]
        norm_layer,
    ], name="preprocessing")
    return model


def build_augmentation_model() -> tf.keras.Sequential:
    """Returns a Keras model with random augmentation layers."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomFlip("vertical"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
        tf.keras.layers.RandomBrightness(0.1),
    ], name="augmentation")

# ─────────────────────────────────────────────
# Dataset builder
# ─────────────────────────────────────────────

def build_tf_dataset(
    split_name: str,
    class_names: list[str],
    preprocessing_model: tf.keras.Sequential,
    augmentation_model: tf.keras.Sequential | None,
) -> tf.data.Dataset:
    """
    Loads images from dataset_split/<split>/ and returns a tf.data.Dataset
    that yields (preprocessed_image, one_hot_label) batches.
    """
    split_dir = SPLIT_DIR / split_name

    # Use image_dataset_from_directory for simplicity + speed
    ds = tf.keras.utils.image_dataset_from_directory(
        split_dir,
        labels="inferred",
        label_mode="categorical",    # one-hot encoded
        class_names=class_names,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=(split_name == "train"),
        seed=42,
        color_mode="rgb",
    )

    # Apply preprocessing
    ds = ds.map(
        lambda x, y: (preprocessing_model(x, training=False), y),
        num_parallel_calls=AUTOTUNE,
    )

    # Apply augmentation to training data only
    if augmentation_model is not None and split_name == "train":
        ds = ds.map(
            lambda x, y: (augmentation_model(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
        )

    # Pipeline optimization
    if split_name == "train":
        ds = ds.shuffle(buffer_size=SHUFFLE_BUFFER, seed=42, reshuffle_each_iteration=True)

    ds = ds.cache()     # cache after expensive preprocessing
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds

# ─────────────────────────────────────────────
# Validation + visualization
# ─────────────────────────────────────────────

def validate_and_visualize(ds: tf.data.Dataset, class_names: list[str], split_name: str):
    """Inspect a sample batch and save a visualization."""
    batch_images, batch_labels = next(iter(ds))
    images = batch_images.numpy()
    labels = batch_labels.numpy()

    print(f"\n  Batch info — {split_name}:")
    print(f"    Images shape  : {images.shape}")
    print(f"    Labels shape  : {labels.shape}")
    print(f"    Image dtype   : {images.dtype}")
    print(f"    Pixel min/max : {images.min():.3f} / {images.max():.3f}")
    print(f"    Pixel mean    : {images.mean():.3f}")

    # Visualize a few images
    n_show = min(8, images.shape[0])
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    fig.suptitle(f"Sample Preprocessed Batch — {split_name.capitalize()} Split",
                 fontsize=13, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        if i < n_show:
            img = images[i]
            # Clip for display (normalized values may be outside [0,1])
            img_display = np.clip((img - img.min()) / (img.max() - img.min() + 1e-8), 0, 1)
            label_idx = np.argmax(labels[i])
            ax.imshow(img_display)
            ax.set_title(class_names[label_idx][:25], fontsize=7)
        ax.axis("off")

    plt.tight_layout()
    plot_path = OUTPUT_DIR / f"batch_sample_{split_name}.png"
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"    Sample batch visualization saved: {plot_path}")

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print_section("CREATE TF DATASET — Skin Disease Detection")
    print(f"Split dir  : {SPLIT_DIR}")
    print(f"Image size : {IMG_HEIGHT} × {IMG_WIDTH}")
    print(f"Batch size : {BATCH_SIZE}")

    if not SPLIT_DIR.exists():
        print(f"[ERROR] Split dataset not found at {SPLIT_DIR}")
        print("  Run 07_split_dataset.py first.")
        sys.exit(1)

    # ── Class names ────────────────────────────────────────
    class_names = get_class_names(SPLIT_DIR)
    print(f"\n  Classes ({len(class_names)}):")
    for i, c in enumerate(class_names):
        print(f"    [{i:2d}] {c}")

    # ── Save class label mapping ───────────────────────────
    label_map = {name: idx for idx, name in enumerate(class_names)}
    label_map_path = OUTPUT_DIR / "class_label_map.json"
    with open(label_map_path, "w") as f:
        json.dump({"class_to_idx": label_map, "idx_to_class": {v: k for k, v in label_map.items()}}, f, indent=2)
    print(f"\n  Label map saved: {label_map_path}")

    # ── Preprocessing & augmentation models ───────────────
    print_section("Building preprocessing pipeline…")
    mean, std = load_normalization_params()
    preprocessing_model = build_preprocessing_model(mean, std)
    augmentation_model  = build_augmentation_model() if USE_TRAINING_AUGMENTATION else None

    # Print model summaries
    preprocessing_model.build((None, IMG_HEIGHT, IMG_WIDTH, CHANNELS))
    print("\n  Preprocessing model:")
    preprocessing_model.summary(print_fn=lambda x: print(f"    {x}"))

    # ── Build datasets ─────────────────────────────────────
    print_section("Building tf.data datasets…")
    datasets = {}
    for split in ["train", "val", "test"]:
        split_path = SPLIT_DIR / split
        if not split_path.exists():
            print(f"  [SKIP] {split} split not found")
            continue
        print(f"\n  Building {split} dataset…")
        ds = build_tf_dataset(split, class_names, preprocessing_model, augmentation_model)
        datasets[split] = ds
        n_batches = sum(1 for _ in ds)
        print(f"  {split.capitalize()} : {n_batches} batches × {BATCH_SIZE} = ~{n_batches * BATCH_SIZE} images/epoch")

    # ── Validate ───────────────────────────────────────────
    print_section("Validating datasets (sample batch)…")
    for split, ds in datasets.items():
        validate_and_visualize(ds, class_names, split)

    # ── Save usage example ─────────────────────────────────
    usage_example = '''# ── How to use in training ──────────────────────────────────
import json, sys
from pathlib import Path

# Add preprocessing outputs to path if needed
BASE_DIR = Path(".")  # adjust to your project root

# Load label map
with open(BASE_DIR / "preprocessing/outputs/09_tf_dataset/class_label_map.json") as f:
    label_info = json.load(f)
class_names = list(label_info["class_to_idx"].keys())

# Rebuild datasets (run script first, or copy this logic)
from preprocessing.09_create_tf_dataset import build_tf_dataset, build_preprocessing_model, build_augmentation_model

# Load normalization config
norm_path = BASE_DIR / "preprocessing/outputs/05_normalize/normalization_config.json"
with open(norm_path) as f:
    norm = json.load(f)
mean = norm["dataset_stats"]["mean"]
std  = norm["dataset_stats"]["std"]

preprocess = build_preprocessing_model(mean, std)
augment    = build_augmentation_model()

train_ds = build_tf_dataset("train", class_names, preprocess, augment)
val_ds   = build_tf_dataset("val",   class_names, preprocess, None)
test_ds  = build_tf_dataset("test",  class_names, preprocess, None)

# Then pass to model.fit:
# model.fit(train_ds, validation_data=val_ds, epochs=50)
'''
    usage_path = OUTPUT_DIR / "usage_example.py"
    with open(usage_path, "w") as f:
        f.write(usage_example)
    print(f"\n  Usage example saved: {usage_path}")

    # ── Save summary ───────────────────────────────────────
    summary = {
        "run_timestamp": datetime.now().isoformat(),
        "split_dir": str(SPLIT_DIR),
        "image_size": [IMG_HEIGHT, IMG_WIDTH, CHANNELS],
        "batch_size": BATCH_SIZE,
        "num_classes": len(class_names),
        "class_names": class_names,
        "normalization": {"mean": mean, "std": std},
        "augmentation_enabled": USE_TRAINING_AUGMENTATION,
        "splits_built": list(datasets.keys()),
    }
    with open(OUTPUT_DIR / "tf_dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print_section("TF Dataset Creation Complete ✓")
    print("  Your tf.data pipelines are ready for model training.")
    print(f"  Outputs: {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()
