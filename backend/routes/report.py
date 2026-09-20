from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from backend.models.user import UserInDB
from backend.routes.auth import get_current_user
from backend.database import get_db
from backend.services.report_service import generate_pdf_report
from bson import ObjectId
import datetime

router = APIRouter()

@router.get("/{prediction_id}")
async def get_prediction_report(
    prediction_id: str,
    current_user: UserInDB = Depends(get_current_user)
):
    try:
        db = await get_db()
        prediction = await db["Predictions"].find_one({"_id": ObjectId(prediction_id)})
        
        if not prediction:
            raise HTTPException(status_code=404, detail="Prediction not found")
            
        if prediction["user_id"] != str(current_user.id):
            raise HTTPException(status_code=403, detail="Not authorized to access this prediction")
            
        user_data = {
            "username": current_user.username,
            "email": current_user.email
        }
        
        pdf_path = generate_pdf_report(prediction, user_data)
        
        # Save metadata to Reports collection
        report_data = {
            "prediction_id": prediction_id,
            "user_id": str(current_user.id),
            "pdf_path": pdf_path,
            "created_at": datetime.datetime.utcnow()
        }
        await db["Reports"].insert_one(report_data)
        
        return FileResponse(
            path=pdf_path,
            media_type="application/pdf",
            filename=f"Skin_Disease_Report_{prediction_id}.pdf"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
