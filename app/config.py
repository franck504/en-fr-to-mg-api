from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = Field(default="en-fr-to-mg-service")
    app_env: str = Field(default="development")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    provider: str = Field(default="hf_seq2seq")
    hf_model_name: str = Field(default="facebook/nllb-200-distilled-600M")
    hf_model_family: str = Field(default="auto")
    hf_device: str = Field(default="auto")
    gemini_model_name: str = Field(default="gemini-2.5-flash")
    gemini_api_key: str | None = Field(default=None)
    gemini_temperature: float = Field(default=0.1)
    gemini_thinking_budget: int = Field(default=0)
    gemini_timeout_seconds: float = Field(default=120.0)
    source_language_default: str = Field(default="auto")
    target_language: str = Field(default="mg")
    translation_max_length: int = Field(default=256)
    model_cache_dir: str = Field(default="/models")
    load_model_on_startup: bool = Field(default=False)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
