from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from bson import ObjectId
from backend.models.user import PyObjectId

class TopPrediction(BaseModel):
    class_name: str
    confidence: float

class DiseaseInfo(BaseModel):
    description: str
    severity: str
    advice: str
    icd10: str

class PredictionCreate(BaseModel):
    user_id: str
    image_path: str
    gradcam_path: Optional[str] = None
    predicted_class: str
    confidence: float
    top3: List[TopPrediction]
    disease_info: DiseaseInfo

class PredictionInDB(PredictionCreate):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class PredictionResponse(BaseModel):
    id: str
    user_id: str
    image_path: str
    gradcam_path: Optional[str] = None
    predicted_class: str
    confidence: float
    top3: List[TopPrediction]
    disease_info: DiseaseInfo
    created_at: datetime
