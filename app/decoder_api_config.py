from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


class DecoderAPIConfig(BaseSettings):
    model_config = _ENV

    decoder_api_enabled: bool = Field(default=False, validation_alias="DECODER_API_ENABLED")
    decoder_api_base_url: str = Field(
        default="http://127.0.0.1:8001",
        validation_alias="DECODER_API_BASE_URL",
    )
    decoder_api_timeout_sec: float = Field(
        default=120.0, validation_alias="DECODER_API_TIMEOUT_SEC", gt=0
    )
    decoder_api_check_readiness: bool = Field(
        default=True,
        validation_alias="DECODER_API_CHECK_READINESS",
    )


@lru_cache(maxsize=1)
def get_decoder_api_config() -> DecoderAPIConfig:
    return DecoderAPIConfig()
