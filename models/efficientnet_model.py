"""
efficientnet_model.py
----------------------
Defines the EfficientNetB3-based model for 10-class skin disease classification using PyTorch.

Architecture:
  - Base : EfficientNetB3 pretrained on ImageNet (frozen initially)
  - Head : GlobalAveragePooling → BatchNorm → Dropout → Dense(256) → Dropout → Dense(10, softmax)

Usage:
    from models.efficientnet_model import build_model, unfreeze_top_layers
    model = build_model(num_classes=10)
"""

import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

try:
    import torch
    import torch.nn as nn
    from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    print("Install with: pip install torch torchvision")
    sys.exit(1)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
IMG_SIZE      = 224           # Input image size (height & width)
NUM_CLASSES   = 10            # Number of skin disease classes
DROPOUT_RATE  = 0.4           # Dropout rate for regularization

CLASS_NAMES = [
    "Eczema",
    "Melanoma",
    "Atopic Dermatitis",
    "Basal Cell Carcinoma (BCC)",
    "Melanocytic Nevi (NV)",
    "Benign Keratosis-like Lesions (BKL)",
    "Psoriasis / Lichen Planus",
    "Seborrheic Keratoses",
    "Tinea / Ringworm / Candidiasis",
    "Warts / Molluscum / Viral Infections",
]

# ImageNet normalization stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────
# Model Builder
# ─────────────────────────────────────────────

class SkinDiseaseEfficientNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE, freeze_base=True):
        super(SkinDiseaseEfficientNet, self).__init__()
        
        # ── Base model: EfficientNetB3 ─────────────────────
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1
        self.base_model = efficientnet_b3(weights=weights)
        
        # Extract features size
        in_features = self.base_model.classifier[1].in_features
        
        # Remove the original classifier head
        self.base_model.classifier = nn.Identity()
        
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
                
        # ── Classification Head ────────────────────────────
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, num_classes)
            # Softmax is handled by nn.CrossEntropyLoss during training. 
        )
        
    def forward(self, x):
        x = self.base_model(x)
        x = self.head(x)
        return x


def build_model(
    num_classes: int = NUM_CLASSES,
    dropout_rate: float = DROPOUT_RATE,
    freeze_base: bool = True
) -> nn.Module:
    """
    Build EfficientNetB3-based classifier.
    """
    model = SkinDiseaseEfficientNet(num_classes, dropout_rate, freeze_base)
    return model


def unfreeze_top_layers(model: nn.Module, num_layers: int = 30) -> nn.Module:
    """
    Unfreeze the top `num_layers` of EfficientNetB3 for fine-tuning.
    """
    all_params = list(model.base_model.parameters())
    total_params = len(all_params)
    
    # Freeze all base model parameters first
    for param in all_params:
        param.requires_grad = False
        
    # Unfreeze the last `num_layers` parameters
    num_to_unfreeze = min(num_layers, total_params)
    for param in all_params[-num_to_unfreeze:]:
        param.requires_grad = True
        
    print(f"  Unfroze top {num_to_unfreeze} parameter tensors of EfficientNetB3 base")
    return model


def get_model_summary(model: nn.Module) -> None:
    """Print a simple model parameter summary."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total parameters    : {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable       : {total_params - trainable_params:,}")


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n[TEST] Building EfficientNetB3 model (PyTorch)...")
    model = build_model(num_classes=NUM_CLASSES, freeze_base=True)
    get_model_summary(model)

    dummy = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
    preds = model(dummy)
    print(f"\n[TEST] Prediction shape: {preds.shape}")
    print("\n[TEST] Model built successfully!")
