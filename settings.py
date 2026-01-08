from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerSettings(BaseModel):
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)


class KBSettings(BaseModel):
    base_url: str
    bot_id: str
    top_k_retrieval: int = Field(default=5)
    top_n_retrieval: int = Field(default=100)
    top_n_reranking: int = Field(default=2)


class LLMSettings(BaseModel):
    base_url: str
    api_key: Optional[str] = None
    model: str = Field(default="llm-small-v4")
    temperature: float = Field(default=0.1)


class ProcessingSettings(BaseModel):
    chunk_size: int = 1000
    chunk_overlap: int = 200
    ocr_delay: int = 5
    search_delay: int = 5
    enable_ocr: bool = Field(default=False)


class OCRSettings(BaseModel):
    add_file_url: str
    session_id_url: str
    result_url: str
    token_id: str = Field(default="")
    token_key: str = Field(default="")
    authorization: str = Field(default="")
    token: str = Field(default="")
    client_session: str = Field(default="")
    timeout_seconds: int = Field(default=2400)
    poll_interval_seconds: int = Field(default=5)


class LoggingSettings(BaseModel):
    level: str = Field(default="INFO")
    log_external_io: bool = Field(default=False)


class AppSettings(BaseSettings):
    server: ServerSettings
    kb_api: KBSettings
    llm_api: LLMSettings
    processing: ProcessingSettings
    ocr_api: OCRSettings
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    model_config = SettingsConfigDict(env_prefix="SMARTAUDITOR_", env_nested_delimiter="__", extra="ignore")


@lru_cache()
def load_yaml_config(path: Path | None = None) -> dict:
    config_path = path or Path(__file__).resolve().parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache()
def get_settings() -> AppSettings:
    yaml_data = load_yaml_config()
    return AppSettings(**yaml_data)


__all__ = [
    "ServerSettings",
    "KBSettings",
    "LLMSettings",
    "ProcessingSettings",
    "OCRSettings",
    "LoggingSettings",
    "AppSettings",
    "get_settings",
]
