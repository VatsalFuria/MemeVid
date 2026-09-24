from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # External services
    pexels_api_key: str

    # App metadata
    environment: str = "development"

    # v1 input caps (PROJECT_SPEC.md §10) — named constants, never literals downstream
    audio_max_duration_seconds: int = 180   # Flow 1 cap: 3 minutes
    script_max_words: int = 200             # Flow 2 cap: ~200 words


@lru_cache
def get_settings() -> Settings:
    return Settings()