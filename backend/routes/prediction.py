from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from backend.models.prediction import PredictionCreate, PredictionResponse, PredictionInDB
from backend.models.user import UserInDB
from backend.routes.auth import get_current_user
from backend.database import get_db
from backend.services.image_service import save_upload_image
from backend.services.predict_service import predict_service
import datetime

router = APIRouter()

@router.post("/upload", response_model=PredictionResponse)
async def upload_and_predict(
    file: UploadFile = File(...),
    current_user: UserInDB = Depends(get_current_user)
):
    try:
        # Save image
        image_path = save_upload_image(file)
        
        # Predict
        prediction_result = predict_service.predict(image_path)
        
        # Create prediction object
        pred_data = PredictionCreate(
            user_id=str(current_user.id),
            image_path=image_path,
            gradcam_path=prediction_result["gradcam_path"],
            predicted_class=prediction_result["predicted_class"],
            confidence=prediction_result["confidence"],
            top3=prediction_result["top3"],
            disease_info=prediction_result["disease_info"]
        )
        
        pred_dict = pred_data.dict()
        pred_dict["created_at"] = datetime.datetime.utcnow()

        # Save to MongoDB (gracefully fail if DB is unreachable or unavailable)
        try:
            db = await get_db()
            if db is None:
                raise RuntimeError("Database unavailable (offline mode).")
            result = await db["Predictions"].insert_one(pred_dict)
            saved_pred = await db["Predictions"].find_one({"_id": result.inserted_id})
            return PredictionResponse(**saved_pred, id=str(saved_pred["_id"]))
        except Exception as db_err:
            print(f"Warning: Failed to save prediction to database: {db_err}")
            # Return result with a temporary ID so the frontend can still proceed
            pred_dict["_id"] = "offline_" + datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
            return PredictionResponse(**pred_dict, id=pred_dict["_id"])
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
