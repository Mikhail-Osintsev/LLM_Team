from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# app/backend/core/config.py -> repo root is 3 levels up from "core"
BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data"

class Settings(BaseSettings):
    # --- backend ---
    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8001

    # --- embeddings / data paths ---
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-base"
    INDEX_DIR: str = str(DATA_DIR / "indexes")      # можно переопределить через .env
    PROCESSED_DIR: str = str(DATA_DIR / "processed")
    RAW_DIR: str = str(DATA_DIR / "raw")

    # --- frontend (чтобы .env не ругался) ---
    FRONTEND_HOST: str = "0.0.0.0"
    FRONTEND_PORT: int = 8501

    # pydantic v2: конфиг модели
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",   # игнорируем любые другие лишние ключи из .env
    )

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()