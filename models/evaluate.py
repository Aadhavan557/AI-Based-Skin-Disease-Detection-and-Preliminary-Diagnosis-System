"""
evalute.py
-----------
Comprehensive PyTorch model evaluation on the held-out test set.

Generates:
  - Classification report (precision, recall, F1 per class)
  - Confusion matrix (normalized & raw)
  - ROC curves (one-vs-rest per class)
  - Top-K accuracy
  - Per-class sample predictions grid

Usage:
    python models/evalute.py
    python models/evalute.py --model models/outputs/best_model.pth
"""

import sys
import io
import argparse
import json
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

try:
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from PIL import Image
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        roc_curve,
        auc as sklearn_auc,
    )
    from sklearn.preprocessing import label_binarize
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    print("Install with: pip install torch torchvision scikit-learn seaborn matplotlib pillow numpy")
    sys.exit(1)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent
TEST_DIR      = BASE_DIR / "dataset_split" / "test"
MODEL_DIR     = BASE_DIR / "models" / "outputs"
EVAL_OUT_DIR  = BASE_DIR / "models" / "outputs" / "evaluation"
EVAL_OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL = MODEL_DIR / "best_model.pth"
IMG_SIZE      = 224
BATCH_SIZE    = 32

sys.path.insert(0, str(BASE_DIR))
from models.efficientnet_model import build_model, CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def print_section(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def load_model(model_path: Path) -> torch.nn.Module:
    """Load saved PyTorch model."""
    print(f"  Loading model from: {model_path}")
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        sys.exit(1)
    
    model = build_model(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("  Model loaded successfully.")
    return model


def build_test_loader(test_dir: Path):
    """Build test DataLoader."""
    val_test_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    dataset = datasets.ImageFolder(test_dir, transform=val_test_transforms)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    return loader, dataset


def get_true_and_pred(model, loader):
    """Run full inference on DataLoader, return y_true, y_pred, y_prob."""
    print("  Running inference on test set...")
    
    all_true = []
    all_pred = []
    all_prob = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            
            all_true.extend(labels.numpy())
            all_pred.extend(preds)
            all_prob.extend(probs)
            
    return np.array(all_true), np.array(all_pred), np.array(all_prob)


# ─────────────────────────────────────────────
# Evaluation Functions
# ─────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, class_names, output_dir: Path):
    cm     = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    short = [n[:18] for n in class_names]

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    fig.suptitle("Confusion Matrix — Skin Disease Classifier", fontsize=14, fontweight="bold")

    for ax, data, title, fmt in zip(
        axes,
        [cm_norm, cm],
        ["Normalized (row %)", "Raw Counts"],
        [".2f", "d"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=short, yticklabels=short,
            ax=ax, linewidths=0.5,
        )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True", fontsize=10)
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    path = output_dir / "confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved: {path}")


def plot_roc_curves(y_true, y_prob, class_names, output_dir: Path):
    n_classes = len(class_names)
    y_bin     = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(12, 8))
    colors  = plt.cm.tab10(np.linspace(0, 1, n_classes))

    macro_aucs = []
    for i, (name, color) in enumerate(zip(class_names, colors)):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc     = sklearn_auc(fpr, tpr)
        macro_aucs.append(roc_auc)
        ax.plot(fpr, tpr, color=color, lw=1.5,
                label=f"{name[:22]:<22} (AUC={roc_auc:.3f})")

    macro_auc = np.mean(macro_aucs) if macro_aucs else 0.0
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_title(f"ROC Curves per Class (Macro AUC = {macro_auc:.3f})", fontsize=13)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    path = output_dir / "roc_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ROC curves saved: {path}")
    return macro_auc


def print_classification_report(y_true, y_pred, class_names, output_dir: Path):
    report = classification_report(
        y_true, y_pred,
        target_names=[n[:30] for n in class_names],
        digits=4,
    )
    print("\n" + report)
    report_path = output_dir / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Report saved: {report_path}")


def compute_topk_accuracy(y_true, y_prob, k: int = 3) -> float:
    topk_preds = np.argsort(y_prob, axis=1)[:, -k:]
    correct    = sum(1 for true, preds in zip(y_true, topk_preds) if true in preds)
    acc        = correct / len(y_true)
    return acc


def plot_per_class_accuracy(y_true, y_pred, class_names, output_dir: Path):
    cm        = confusion_matrix(y_true, y_pred)
    per_class = cm.diagonal() / cm.sum(axis=1)
    short     = [n[:22] for n in class_names]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors  = ["#2ecc71" if a >= 0.8 else "#f39c12" if a >= 0.6 else "#e74c3c"
               for a in per_class]
    bars    = ax.barh(short, per_class, color=colors, edgecolor="white", height=0.6)

    for bar, val in zip(bars, per_class):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", fontsize=9)

    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Accuracy")
    ax.set_title("Per-Class Accuracy", fontsize=13, fontweight="bold")
    ax.axvline(0.8, color="green", linestyle="--", alpha=0.5, label="80% threshold")
    ax.grid(axis="x", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    path = output_dir / "per_class_accuracy.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Per-class accuracy saved: {path}")


def plot_sample_predictions(model, test_dir: Path, class_names, output_dir: Path, n_per_class=2):
    n_classes  = len(class_names)
    class_dirs = sorted([d for d in test_dir.iterdir() if d.is_dir()])
    folder_to_idx = {d.name: i for i, d in enumerate(class_dirs)}

    fig, axes = plt.subplots(
        n_classes, n_per_class,
        figsize=(n_per_class * 3, n_classes * 3),
    )
    fig.suptitle("Sample Predictions (Green=Correct, Red=Wrong)", fontsize=13, fontweight="bold")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    for row, cls_dir in enumerate(class_dirs):
        images = list(cls_dir.glob("*.jpg"))[:n_per_class]
        for col in range(n_per_class):
            ax = axes[row][col]
            if col < len(images):
                img_path = images[col]
                img = Image.open(img_path).convert("RGB")
                
                # Predict
                t_img = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    out = model(t_img)
                    prob = torch.softmax(out, dim=1)[0].cpu().numpy()
                
                pred_idx = np.argmax(prob)
                true_idx = folder_to_idx.get(cls_dir.name, row)

                # Original unnormalized for display
                disp_img = img.resize((IMG_SIZE, IMG_SIZE))
                ax.imshow(disp_img)
                ax.axis("off")
                
                is_correct = pred_idx == true_idx
                color      = "#27ae60" if is_correct else "#e74c3c"
                label      = f"T: {class_names[true_idx][:14]}\nP: {class_names[pred_idx][:14]}\n{prob[pred_idx]:.1%}"
                ax.set_title(label, fontsize=7, color=color, pad=2)
            else:
                ax.axis("off")

    plt.tight_layout()
    path = output_dir / "sample_predictions.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Sample predictions saved: {path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main(model_path: Path):
    print_section("SKIN DISEASE — PyTorch Model Evaluation")
    print(f"  Model : {model_path}")
    print(f"  Test  : {TEST_DIR}")
    print(f"  Output: {EVAL_OUT_DIR}")

    # ── Load model ─────────────────────────────────
    print_section("Loading Model")
    model = load_model(model_path)

    # ── Build test generator ───────────────────────
    print_section("Building Test Loader")
    test_loader, dataset = build_test_loader(TEST_DIR)
    print(f"  Test samples     : {len(dataset)}")
    
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    ordered_names = [idx_to_class.get(i, CLASS_NAMES[i]) for i in range(len(CLASS_NAMES))]

    # ── Full predictions ───────────────────────────
    print_section("Running Full Inference")
    y_true, y_pred, y_prob = get_true_and_pred(model, test_loader)

    # ── Top-K accuracy ─────────────────────────────
    top1 = compute_topk_accuracy(y_true, y_prob, k=1)
    top3 = compute_topk_accuracy(y_true, y_prob, k=3)
    top5 = compute_topk_accuracy(y_true, y_prob, k=5)
    print(f"\n  Top-1 Accuracy: {top1:.4f}")
    print(f"  Top-3 Accuracy: {top3:.4f}")
    print(f"  Top-5 Accuracy: {top5:.4f}")

    # ── Classification report ──────────────────────
    print_section("Classification Report")
    print_classification_report(y_true, y_pred, ordered_names, EVAL_OUT_DIR)

    # ── Confusion matrix ───────────────────────────
    print_section("Confusion Matrix")
    plot_confusion_matrix(y_true, y_pred, ordered_names, EVAL_OUT_DIR)

    # ── ROC curves ─────────────────────────────────
    print_section("ROC Curves")
    macro_auc = plot_roc_curves(y_true, y_prob, ordered_names, EVAL_OUT_DIR)

    # ── Per-class accuracy ─────────────────────────
    print_section("Per-Class Accuracy")
    plot_per_class_accuracy(y_true, y_pred, ordered_names, EVAL_OUT_DIR)

    # ── Sample predictions ─────────────────────────
    print_section("Sample Predictions Grid")
    plot_sample_predictions(model, TEST_DIR, ordered_names, EVAL_OUT_DIR)

    # ── Save final summary ─────────────────────────
    summary = {
        "model_path"   : str(model_path),
        "test_samples" : len(dataset),
        "top1_accuracy": float(top1),
        "top3_accuracy": float(top3),
        "top5_accuracy": float(top5),
        "macro_auc"    : float(macro_auc),
    }
    summary_path = EVAL_OUT_DIR / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print_section("Evaluation Complete!")
    print(f"  All outputs saved to: {EVAL_OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate skin disease model on test set.")
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL),
        help=f"Path to saved .pth model (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()
    main(Path(args.model))
