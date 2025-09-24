from __future__ import annotations

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", case_sensitive=False)

    PORT: int = 8080

@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


