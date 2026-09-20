"""
train.py
---------
Full training pipeline for the PyTorch EfficientNetB3 skin disease classifier.

Two-phase training strategy:
  Phase 1 — Transfer Learning : Train only the classification head (base frozen)
  Phase 2 — Fine-Tuning       : Unfreeze top layers of EfficientNetB3 and train with low LR

Usage:
    python models/train.py

Outputs (saved to models/outputs/):
    - best_model.pth          : Best checkpoint (val_accuracy)
    - final_model.pth         : Final model after all phases
    - training_history.json   : Loss/metric curves
    - training_curves.png     : Plotted training curves
"""

import sys
import io
import os
import json
import time
import copy
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

try:
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from torchvision import datasets, transforms
    from sklearn.metrics import roc_auc_score
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    print("Install with: pip install torch torchvision scikit-learn matplotlib numpy")
    sys.exit(1)

# Add project root to path so we can import from models/
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from models.efficientnet_model import (
    build_model, unfreeze_top_layers, CLASS_NAMES, IMG_SIZE, NUM_CLASSES, IMAGENET_MEAN, IMAGENET_STD
)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
DATASET_SPLIT_DIR = BASE_DIR / "dataset_split"
TRAIN_DIR         = DATASET_SPLIT_DIR / "train"
VAL_DIR           = DATASET_SPLIT_DIR / "val"
TEST_DIR          = DATASET_SPLIT_DIR / "test"
OUTPUT_DIR        = BASE_DIR / "models" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Training hyperparameters
BATCH_SIZE       = 16

# Phase 1 – Head training
PHASE1_EPOCHS    = 20
PHASE1_LR        = 1e-3

# Phase 2 – Fine-tuning
PHASE2_EPOCHS    = 30
PHASE2_LR        = 1e-5
UNFREEZE_LAYERS  = 40       # How many top EfficientNetB3 layer tensors to unfreeze

EARLY_STOP_PATIENCE  = 7
REDUCE_LR_PATIENCE   = 3
REDUCE_LR_FACTOR     = 0.3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
# Data Generators
# ─────────────────────────────────────────────

def build_dataloaders(train_dir: Path, val_dir: Path, test_dir: Path, batch_size: int):
    # PyTorch transformations
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=(0.8, 1.2)),
        transforms.ToTensor(),
        # EfficientNet expects standard normalization
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset   = datasets.ImageFolder(val_dir, transform=val_test_transforms)
    test_dataset  = datasets.ImageFolder(test_dir, transform=val_test_transforms)

    # Compute class weights for WeightedRandomSampler (handles imbalance better than Keras class_weights)
    class_counts = np.bincount(train_dataset.targets)
    class_weights = 1. / class_counts
    sample_weights = class_weights[train_dataset.targets]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    return train_loader, val_loader, test_loader, train_dataset.class_to_idx

# ─────────────────────────────────────────────
# Training Utilities
# ─────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, metric):
        if self.best_score is None:
            self.best_score = metric
        elif metric < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = metric
            self.counter = 0

def plot_training_curves(history: dict, output_dir: Path):
    """Plot and save training/validation accuracy & loss curves."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Training Curves — Skin Disease EfficientNetB3 (PyTorch)", fontsize=14, fontweight="bold")

    epochs = range(1, len(history["accuracy"]) + 1)

    # Accuracy
    axes[0].plot(epochs, history["accuracy"],     label="Train Acc",  color="#3498db")
    axes[0].plot(epochs, history["val_accuracy"], label="Val Acc",    color="#e74c3c", linestyle="--")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Loss
    axes[1].plot(epochs, history["loss"],     label="Train Loss", color="#3498db")
    axes[1].plot(epochs, history["val_loss"], label="Val Loss",   color="#e74c3c", linestyle="--")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # AUC
    if "val_auc" in history and history["val_auc"]:
        axes[2].plot(epochs, history["auc"],     label="Train AUC", color="#2ecc71")
        axes[2].plot(epochs, history["val_auc"], label="Val AUC",   color="#e67e22", linestyle="--")
        axes[2].set_title("AUC")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("AUC")
        axes[2].legend()
        axes[2].grid(alpha=0.3)
    else:
        axes[2].axis("off")

    plt.tight_layout()
    plot_path = output_dir / "training_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training curves saved: {plot_path}")

# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, best_model_path, early_stop_patience):
    early_stopping = EarlyStopping(patience=early_stop_patience)
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    history = {'loss': [], 'accuracy': [], 'auc': [], 'val_loss': [], 'val_accuracy': [], 'val_auc': []}

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # For AUC
                probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
                all_preds.extend(probs)
                all_labels.extend(labels.cpu().numpy())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            # Compute macro AUC if possible
            try:
                # one-hot encode labels for roc_auc_score
                labels_onehot = np.zeros((len(all_labels), NUM_CLASSES))
                labels_onehot[np.arange(len(all_labels)), all_labels] = 1
                epoch_auc = roc_auc_score(labels_onehot, np.array(all_preds), average="macro", multi_class="ovr")
            except ValueError:
                epoch_auc = 0.0

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} AUC: {epoch_auc:.4f}")

            if phase == 'train':
                history['loss'].append(float(epoch_loss))
                history['accuracy'].append(float(epoch_acc))
                history['auc'].append(float(epoch_auc))
            else:
                history['val_loss'].append(float(epoch_loss))
                history['val_accuracy'].append(float(epoch_acc))
                history['val_auc'].append(float(epoch_auc))
                scheduler.step(epoch_loss)
                
                # Check early stopping & best model
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), best_model_path)
                    print(f" -> Best model saved with Acc: {best_acc:.4f}")
                
                early_stopping(epoch_acc.item())

        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

# ─────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────

def print_section(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")

def main():
    print_section("SKIN DISEASE — PyTorch EfficientNetB3 Training Pipeline")
    print(f"  PyTorch version    : {torch.__version__}")
    print(f"  Device available   : {device}")
    print(f"  Train dir          : {TRAIN_DIR}")
    print(f"  Val dir            : {VAL_DIR}")
    print(f"  Output dir         : {OUTPUT_DIR}")

    if not TRAIN_DIR.exists():
        print(f"\n[ERROR] Train directory not found: {TRAIN_DIR}")
        sys.exit(1)

    # ── Build data generators ──────────────────────────
    print_section("Building DataLoaders")
    train_loader, val_loader, test_loader, class_to_idx = build_dataloaders(
        TRAIN_DIR, VAL_DIR, TEST_DIR, BATCH_SIZE
    )

    print(f"\n  Training samples   : {len(train_loader.dataset)}")
    print(f"  Validation samples : {len(val_loader.dataset)}")
    print(f"  Test samples       : {len(test_loader.dataset)}")
    print(f"  Classes found      : {list(class_to_idx.keys())}")

    # Save class index mapping
    class_map_path = OUTPUT_DIR / "class_indices.json"
    with open(class_map_path, "w") as f:
        json.dump(class_to_idx, f, indent=2)

    dataloaders = {'train': train_loader, 'val': val_loader}
    criterion = nn.CrossEntropyLoss()

    # ── Phase 1: Transfer Learning ─────────────────────
    print_section("Phase 1 — Transfer Learning (Frozen Base)")
    print(f"  Epochs      : {PHASE1_EPOCHS}")
    print(f"  LR          : {PHASE1_LR}")

    model = build_model(num_classes=NUM_CLASSES, freeze_base=True).to(device)
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=PHASE1_LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=REDUCE_LR_FACTOR, patience=REDUCE_LR_PATIENCE)

    best_model_path = OUTPUT_DIR / "best_model.pth"
    model, history1 = train_model(
        model, dataloaders, criterion, optimizer, scheduler, 
        PHASE1_EPOCHS, best_model_path, EARLY_STOP_PATIENCE
    )

    # ── Phase 2: Fine-Tuning ───────────────────────────
    print_section("Phase 2 — Fine-Tuning (Top Layers Unfrozen)")
    print(f"  Unfreezing top {UNFREEZE_LAYERS} layer parameters")
    print(f"  Epochs      : {PHASE2_EPOCHS}")
    print(f"  LR          : {PHASE2_LR}")

    model = unfreeze_top_layers(model, num_layers=UNFREEZE_LAYERS)
    
    # Must recreate optimizer to include newly unfrozen layers
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=PHASE2_LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=REDUCE_LR_FACTOR, patience=REDUCE_LR_PATIENCE)

    model, history2 = train_model(
        model, dataloaders, criterion, optimizer, scheduler, 
        PHASE2_EPOCHS, best_model_path, EARLY_STOP_PATIENCE
    )

    # ── Save final model & history ─────────────────────
    final_model_path = OUTPUT_DIR / "final_model.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"\n  Final model saved: {final_model_path}")

    # Merge history
    merged = {}
    for key in history1.keys():
        merged[key] = history1[key] + history2[key]
        
    hist_path = OUTPUT_DIR / "training_history.json"
    with open(hist_path, "w") as f:
        json.dump(merged, f, indent=2)

    # Plot curves
    print_section("Plotting Training Curves")
    plot_training_curves(merged, OUTPUT_DIR)

    print_section("Training Complete!")
    print(f"  All outputs saved to: {OUTPUT_DIR}\n")

if __name__ == "__main__":
    main()
