from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from backend.config import settings
from backend.database import connect_to_mongo, close_mongo_connection
from backend.services.predict_service import predict_service
from backend.utils.logger import logger

from backend.routes import auth, prediction, report, user

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up FastAPI application...")
    await connect_to_mongo()
    predict_service.load_model()
    yield
    # Shutdown
    logger.info("Shutting down FastAPI application...")
    await close_mongo_connection()

app = FastAPI(
    title=settings.APP_NAME,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For dev only, update in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directories
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.REPORT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(settings.UPLOAD_DIR)), name="uploads")
app.mount("/reports", StaticFiles(directory=str(settings.REPORT_DIR)), name="reports")

# Include Routers
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(user.router, prefix="/api/user", tags=["User Profile"])
app.include_router(prediction.router, prefix="/api/prediction", tags=["Inference"])
app.include_router(report.router, prefix="/api/report", tags=["Reporting"])

@app.get("/")
async def root():
    return {"message": "Welcome to the Skin Disease Detection API. Visit /docs for documentation."}
