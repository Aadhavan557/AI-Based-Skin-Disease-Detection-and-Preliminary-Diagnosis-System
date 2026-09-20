import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
from backend.config import settings
import sys

sys.path.insert(0, str(settings.BASE_DIR))
from models.efficientnet_model import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD

def get_inference_transforms():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def preprocess_image(image_path: Path, device: torch.device) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    transform = get_inference_transforms()
    t = transform(img).unsqueeze(0).to(device)
    return t
