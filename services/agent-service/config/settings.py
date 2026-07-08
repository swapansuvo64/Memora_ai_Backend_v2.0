import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    SUPABASE_URL: str
    SUPABASE_KEY: str
    DATABASE_URL: str
    JWT_SECRET: str
    JWT_ALGORITHM: str = "HS256"
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379

    # GCP Vertex AI Config
    GCP_PROJECT_ID: str = "project-c804c4a3-7d47-47d1-be4"

    # AWS Bedrock config (Optional/Deprecated)
    BEDROCK_AWS_REGION: str = "us-east-1"
    BEDROCK_AWS_ACCESS_KEY_ID: str = ""
    BEDROCK_AWS_SECRET_ACCESS_KEY: str = ""
    BEDROCK_MODEL_SCOUT: str = "us.amazon.nova-micro-v1:0"
    BEDROCK_MODEL_MAVERICK: str = "us.anthropic.claude-3-haiku-20240307-v1:0"
    BEDROCK_MODEL_SAFETY: str = "us.anthropic.claude-sonnet-4-20250514-v1:0"

    class Config:
        env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")
        extra = "ignore"

settings = Settings()
