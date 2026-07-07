import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    SUPABASE_URL: str
    SUPABASE_KEY: str
    DATABASE_URL: str
    JWT_SECRET: str
    JWT_REFRESH_SECRET: str
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_EXPIRE: int = 900       # 15 minutes
    JWT_REFRESH_EXPIRE: int = 604800   # 7 days

    class Config:
        # Resolve the path to the parent directory (.env is in /Backend)
        env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")
        extra = "ignore"

settings = Settings()
