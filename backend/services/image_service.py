import os
import uuid
import shutil
from fastapi import UploadFile, HTTPException
from pathlib import Path
from backend.config import settings

def save_upload_image(upload_file: UploadFile) -> str:
    # Ensure upload dir exists
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    # Validate extension
    ext = os.path.splitext(upload_file.filename)[1].lower()
    allowed_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Invalid file extension. Allowed: {allowed_extensions}")
        
    unique_filename = f"{uuid.uuid4().hex}{ext}"
    file_path = settings.UPLOAD_DIR / unique_filename
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
        
    return str(file_path)
