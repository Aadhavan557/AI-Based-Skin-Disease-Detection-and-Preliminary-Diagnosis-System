from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    # App config
    APP_NAME: str = "Skin Disease Detection API"
    DEBUG_MODE: bool = True
    
    # MongoDB Config
    MONGODB_URL: str = "mongodb://localhost:27017"
    DATABASE_NAME: str = "skindisease_db"
    
    # Auth Config
    SECRET_KEY: str = "super_secret_jwt_key_change_in_production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7 # 7 days
    
    # Paths
    BASE_DIR: Path = _BASE_DIR
    MODEL_PATH: Path = _BASE_DIR / "models" / "outputs" / "best_model.pth"
    CLASS_INDEX_PATH: Path = _BASE_DIR / "models" / "outputs" / "class_indices.json"
    UPLOAD_DIR: Path = _BASE_DIR / "backend" / "uploads"
    REPORT_DIR: Path = _BASE_DIR / "backend" / "reports"

    model_config = SettingsConfigDict(
        env_file=str(_BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

settings = Settings()
