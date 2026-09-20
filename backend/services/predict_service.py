import sys
import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from backend.config import settings
from backend.utils.logger import logger
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, str(settings.BASE_DIR))
from models.efficientnet_model import build_model, NUM_CLASSES, IMG_SIZE
from models.predict import get_transforms, DISEASE_INFO, GradCAM

class PredictService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.idx_to_class = {}
        
    def load_model(self):
        logger.info("Loading PyTorch model...")
        self.model = build_model(num_classes=NUM_CLASSES)
        if not settings.MODEL_PATH.exists():
            logger.error(f"Model file not found at {settings.MODEL_PATH}")
            return
            
        self.model.load_state_dict(torch.load(settings.MODEL_PATH, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Load classes
        if settings.CLASS_INDEX_PATH.exists():
            with open(settings.CLASS_INDEX_PATH) as f:
                idx = json.load(f)
            self.idx_to_class = {int(v): k for k, v in idx.items()}
        logger.info("PyTorch model loaded successfully.")

    def generate_gradcam(self, image_path: Path, pred_idx: int) -> str:
        cam = GradCAM(self.model)
        img = Image.open(image_path).convert("RGB")
        t = get_transforms()(img).unsqueeze(0).to(self.device)
        heatmap = cam.generate(t, pred_idx)
        
        # Save overlay
        img_arr = np.array(img.resize((IMG_SIZE, IMG_SIZE)))
        heatmap_resized = np.array(
            Image.fromarray(np.uint8(heatmap * 255)).resize((IMG_SIZE, IMG_SIZE), resample=Image.BICUBIC)
        ) / 255.0
        
        import matplotlib
        colormap = matplotlib.colormaps["jet"]
        heat_rgb = colormap(heatmap_resized)[:, :, :3]
        overlay = (0.6 * img_arr / 255.0 + 0.4 * heat_rgb)
        overlay = np.clip(overlay, 0, 1)
        
        gradcam_path = settings.UPLOAD_DIR / f"{image_path.stem}_gradcam.png"
        
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(overlay)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(gradcam_path, dpi=120, bbox_inches="tight")
        plt.close()
        
        return str(gradcam_path)

    def predict(self, image_path_str: str) -> dict:
        image_path = Path(image_path_str)
        img = Image.open(image_path).convert("RGB")
        t = get_transforms()(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            out = self.model(t)
            prob = torch.softmax(out, dim=1).cpu().numpy()[0]
            
        top3_idx = np.argsort(prob)[::-1][:3]
        top3 = [
            {"class_name": self.idx_to_class.get(i, f"Class {i}"), "confidence": float(prob[i])}
            for i in top3_idx
        ]
        pred_class = top3[0]["class_name"]
        pred_idx = top3_idx[0]
        
        info_key = next(
            (k for k in DISEASE_INFO if k.lower() in pred_class.lower() or pred_class.lower() in k.lower()),
            "Eczema"
        )
        
        gradcam_path = self.generate_gradcam(image_path, pred_idx)
        
        return {
            "predicted_class": pred_class,
            "confidence": float(prob[pred_idx]),
            "top3": top3,
            "disease_info": DISEASE_INFO[info_key],
            "gradcam_path": gradcam_path
        }

predict_service = PredictService()
