"""
predict.py
-----------
Single-image and batch inference using the trained PyTorch EfficientNetB3 model.

Features:
  - Single image prediction with confidence scores
  - Batch prediction from a folder
  - Top-3 predictions with disease info
  - Grad-CAM visualization (saliency map) using PyTorch hooks

Usage:
    # Single image
    python models/predict.py --image path/to/image.jpg

    # Batch folder
    python models/predict.py --folder path/to/images/

    # With Grad-CAM heatmap
    python models/predict.py --image path/to/image.jpg --gradcam
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
    import matplotlib.cm as cm
    from PIL import Image
    import torch
    from torchvision import transforms
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    print("Install with: pip install torch torchvision matplotlib pillow numpy")
    sys.exit(1)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent
MODEL_DIR     = BASE_DIR / "models" / "outputs"
DEFAULT_MODEL = MODEL_DIR / "best_model.pth"
PRED_OUT_DIR  = BASE_DIR / "models" / "outputs" / "predictions"
PRED_OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE_DIR))
from models.efficientnet_model import build_model, CLASS_NAMES, IMG_SIZE, NUM_CLASSES, IMAGENET_MEAN, IMAGENET_STD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Disease info for preliminary diagnosis ──
DISEASE_INFO = {
    "Eczema": {
        "description": "Inflammatory skin condition causing itchy, red, cracked skin.",
        "severity":    "Mild to Moderate",
        "advice":      "Use moisturizers, avoid triggers. Consult a dermatologist if severe.",
        "icd10":       "L20",
    },
    "Melanoma": {
        "description": "Most dangerous form of skin cancer arising from pigment-producing cells.",
        "severity":    "High — Urgent",
        "advice":      "Seek immediate dermatologist/oncologist evaluation. Do NOT delay.",
        "icd10":       "C43",
    },
    "Atopic Dermatitis": {
        "description": "Chronic skin inflammation causing dry, itchy skin. Often starts in childhood.",
        "severity":    "Mild to Severe",
        "advice":      "Topical corticosteroids, antihistamines. Identify and avoid allergens.",
        "icd10":       "L20.9",
    },
    "Basal Cell Carcinoma": {
        "description": "Most common skin cancer. Slow-growing, rarely spreads but needs treatment.",
        "severity":    "Moderate — Medical Attention Required",
        "advice":      "Surgical excision or radiation therapy. Consult a dermatologist.",
        "icd10":       "C44",
    },
    "Melanocytic Nevi": {
        "description": "Common benign moles (nevi). Usually harmless but monitor for changes.",
        "severity":    "Low (Benign)",
        "advice":      "Monitor for ABCDE changes (Asymmetry, Border, Color, Diameter, Evolution).",
        "icd10":       "D22",
    },
    "Benign Keratosis": {
        "description": "Non-cancerous skin growths including seborrheic keratosis and solar lentigo.",
        "severity":    "Low (Benign)",
        "advice":      "Typically no treatment needed. Removal for cosmetic reasons if desired.",
        "icd10":       "L82",
    },
    "Psoriasis / Lichen Planus": {
        "description": "Chronic autoimmune skin condition causing red, scaly patches.",
        "severity":    "Moderate",
        "advice":      "Topical treatments, phototherapy, or systemic medications. Dermatologist consult.",
        "icd10":       "L40",
    },
    "Seborrheic Keratoses": {
        "description": "Benign skin growths that appear waxy or wart-like. Very common in older adults.",
        "severity":    "Low (Benign)",
        "advice":      "No treatment required. Removal possible via cryotherapy or laser.",
        "icd10":       "L82",
    },
    "Tinea / Ringworm": {
        "description": "Fungal skin infection causing ring-shaped rash. Highly treatable.",
        "severity":    "Low to Moderate",
        "advice":      "Antifungal creams (clotrimazole, miconazole). Keep area clean and dry.",
        "icd10":       "B35",
    },
    "Warts / Molluscum": {
        "description": "Viral skin infections causing small, benign growths.",
        "severity":    "Low",
        "advice":      "Often self-resolving. Cryotherapy or topical treatments available.",
        "icd10":       "B07",
    },
}

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def load_pytorch_model(model_path: Path) -> torch.nn.Module:
    """Load a saved PyTorch model."""
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        print("Train the model first: python models/train.py")
        sys.exit(1)
        
    model = build_model(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def get_transforms():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def load_class_index(model_dir: Path) -> dict:
    """Load class_indices.json saved during training."""
    path = model_dir / "class_indices.json"
    if path.exists():
        with open(path) as f:
            idx = json.load(f)
        # Invert: {class_name: idx} → {idx: class_name}
        return {int(v): k for k, v in idx.items()}
    # Fallback to built-in class names
    return {i: name for i, name in enumerate(CLASS_NAMES)}


# ─────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────

def predict_single(model, image_path: Path, idx_to_class: dict) -> dict:
    img = Image.open(image_path).convert("RGB")
    t = get_transforms()(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out = model(t)
        prob = torch.softmax(out, dim=1).cpu().numpy()[0]

    top3_idx  = np.argsort(prob)[::-1][:3]
    top3      = [
        {"class": idx_to_class.get(i, f"Class {i}"), "confidence": float(prob[i])}
        for i in top3_idx
    ]
    pred_class = top3[0]["class"]

    info_key = next(
        (k for k in DISEASE_INFO if k.lower() in pred_class.lower() or pred_class.lower() in k.lower()),
        "Eczema"
    )

    return {
        "image"       : str(image_path),
        "class_name"  : pred_class,
        "confidence"  : float(prob[top3_idx[0]]),
        "top3"        : top3,
        "disease_info": DISEASE_INFO[info_key],
        "probabilities": {idx_to_class.get(i, f"Class_{i}"): float(p) for i, p in enumerate(prob)},
    }


def print_prediction(result: dict):
    print(f"\n  Image      : {Path(result['image']).name}")
    print(f"  Prediction : {result['class_name']}")
    print(f"  Confidence : {result['confidence']:.2%}")

    print(f"\n  Top-3 Predictions:")
    for i, item in enumerate(result["top3"], 1):
        bar = "=" * int(item["confidence"] * 30)
        print(f"    {i}. {item['class']:<35} {item['confidence']:6.2%}  [{bar:<30}]")

    info = result["disease_info"]
    print(f"\n  --- Preliminary Diagnosis ---")
    print(f"  Condition  : {result['class_name']}")
    print(f"  ICD-10     : {info.get('icd10', 'N/A')}")
    print(f"  Severity   : {info['severity']}")
    print(f"  Description: {info['description']}")
    print(f"  Advice     : {info['advice']}")
    print(f"\n  [!] This is an AI-assisted preliminary screening tool.")
    print(f"      Always consult a qualified dermatologist for diagnosis.")


# ─────────────────────────────────────────────
# Grad-CAM (Custom PyTorch Implementation)
# ─────────────────────────────────────────────

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.feature_maps = None
        self.gradients = None
        
        # EfficientNetB3 target layer is the last conv layer in features
        # Specifically: base_model.features[-1]
        target_layer = self.model.base_model.features[-1]
        
        target_layer.register_forward_hook(self.save_feature_maps)
        target_layer.register_full_backward_hook(self.save_gradients)

    def save_feature_maps(self, module, input, output):
        self.feature_maps = output.detach()
        
    def save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor, target_class_idx):
        self.model.eval()
        self.model.zero_grad()
        
        input_tensor.requires_grad = True
        output = self.model(input_tensor)
        target = output[0, target_class_idx]
        target.backward()
        
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.feature_maps, dim=1).squeeze(0)
        cam = torch.relu(cam)
        
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-8)
        return cam.cpu().numpy()


def plot_gradcam(image_path: Path, heatmap: np.ndarray, result: dict, output_dir: Path):
    img      = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    img_arr  = np.array(img)

    heatmap_resized = np.array(
        Image.fromarray(np.uint8(heatmap * 255)).resize((IMG_SIZE, IMG_SIZE), resample=Image.BICUBIC)
    ) / 255.0
    
    colormap  = cm.get_cmap("jet")
    heat_rgb  = colormap(heatmap_resized)[:, :, :3]
    overlay   = (0.6 * img_arr / 255.0 + 0.4 * heat_rgb)
    overlay   = np.clip(overlay, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Grad-CAM | Prediction: {result['class_name']} ({result['confidence']:.1%})",
        fontsize=13, fontweight="bold"
    )

    axes[0].imshow(img_arr)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(heatmap_resized, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    stem = Path(image_path).stem
    path = output_dir / f"{stem}_gradcam.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Grad-CAM saved: {path}")


# ─────────────────────────────────────────────
# Batch Prediction
# ─────────────────────────────────────────────

def predict_batch(model, folder: Path, idx_to_class: dict, output_dir: Path):
    SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    images    = [p for p in folder.rglob("*") if p.suffix.lower() in SUPPORTED]

    if not images:
        print(f"  [WARN] No images found in {folder}")
        return

    print(f"  Found {len(images)} images. Running batch inference...")
    results = []
    for img_path in images:
        try:
            r = predict_single(model, img_path, idx_to_class)
            results.append(r)
            print(f"    {img_path.name:<40} -> {r['class_name']:<35} ({r['confidence']:.1%})")
        except Exception as ex:
            print(f"    [SKIP] {img_path.name}: {ex}")

    out_path = output_dir / "batch_predictions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Batch results saved: {out_path}")

    from collections import Counter
    counts = Counter(r["class_name"] for r in results)
    print(f"\n  Prediction Distribution:")
    for cls, cnt in counts.most_common():
        print(f"    {cls:<35} {cnt}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Skin Disease — AI-based prediction tool (PyTorch)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--image",   type=str, help="Path to a single image for prediction")
    parser.add_argument("--folder",  type=str, help="Path to a folder of images for batch prediction")
    parser.add_argument("--model",   type=str, default=str(DEFAULT_MODEL), help="Path to trained model")
    parser.add_argument("--gradcam", action="store_true", help="Generate Grad-CAM visualization")
    parser.add_argument("--save",    action="store_true", help="Save prediction results to JSON")
    args = parser.parse_args()

    if not args.image and not args.folder:
        parser.print_help()
        sys.exit(0)

    print_section("SKIN DISEASE — AI Prediction System")
    print(f"  Model : {args.model}")

    model = load_pytorch_model(Path(args.model))
    idx_to_class = load_class_index(MODEL_DIR)

    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"[ERROR] Image not found: {img_path}")
            sys.exit(1)

        print_section(f"Predicting: {img_path.name}")
        result = predict_single(model, img_path, idx_to_class)
        print_prediction(result)

        if args.gradcam:
            print_section("Generating Grad-CAM")
            
            # Find class idx
            pred_idx = next(
                (k for k, v in idx_to_class.items() if v == result["class_name"]), 0
            )
            
            cam = GradCAM(model)
            img = Image.open(img_path).convert("RGB")
            t = get_transforms()(img).unsqueeze(0).to(device)
            heatmap = cam.generate(t, pred_idx)
            
            if heatmap is not None:
                plot_gradcam(img_path, heatmap, result, PRED_OUT_DIR)

        if args.save:
            out_path = PRED_OUT_DIR / f"{img_path.stem}_prediction.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n  Result saved: {out_path}")

    if args.folder:
        folder = Path(args.folder)
        if not folder.exists():
            print(f"[ERROR] Folder not found: {folder}")
            sys.exit(1)

        print_section(f"Batch Prediction: {folder}")
        predict_batch(model, folder, idx_to_class, PRED_OUT_DIR)

    print_section("Done!")


if __name__ == "__main__":
    main()
