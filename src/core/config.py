import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Match password defined in docker-compose.yml (postgrespassword)
    POSTGRES_URI: str = "postgresql://postgres:postgrespassword@localhost:5432/agent_master_db"
    LITELLM_MODEL: str = "gpt-4o"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
