from fastapi import APIRouter, Depends
from backend.models.user import UserResponse, UserInDB
from backend.routes.auth import get_current_user

router = APIRouter()

@router.get("/me", response_model=UserResponse)
async def read_users_me(current_user: UserInDB = Depends(get_current_user)):
    user_data = current_user.dict(exclude={"id"})
    return UserResponse(**user_data, id=str(current_user.id))
