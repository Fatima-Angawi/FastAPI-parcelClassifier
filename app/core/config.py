from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    MODEL_PATH: str = "models/production_model.pt"
    DEVICE: str = "cpu"
    IMAGE_SIZE: int = 224
    ENV: str = "development"

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
    )
    PROJECT_NAME: str = "Parcel Classifier"

settings = Settings()
