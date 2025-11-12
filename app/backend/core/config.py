from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # --- backend ---
    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8001

    # --- embeddings / data paths ---
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-base"
    INDEX_DIR: str = "./data/indexes"      # локально: ./..., в Docker: /app/data/...
    PROCESSED_DIR: str = "./data/processed"
    RAW_DIR: str = "./data/raw"

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