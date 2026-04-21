from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str | None = None
    runs_dir: Path = Path("./runs")
    max_retries: int = 3
    mesh_gen_backend: Literal["stub", "hunyuan"] = "stub"
    mesh_gen_concurrency: int = 4
    log_level: Literal["DEBUG", "INFO"] = "INFO"


MODEL_REGISTRY: dict[str, str] = {
    "claude-opus-4-7": "anthropic",
    "claude-sonnet-4-6": "anthropic",
    "claude-haiku-4-5": "anthropic",
}


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings_for_tests() -> None:
    global _settings
    _settings = None
