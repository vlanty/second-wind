"""Bot settings from the environment, checked once at startup.

A bad value stops the start with a message that names the variable but never shows the value:
several of them are secrets.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator


class ConfigError(RuntimeError):
    pass


class BotConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore", populate_by_name=True)

    telegram_bot_token: SecretStr = Field(alias="TELEGRAM_BOT_TOKEN", min_length=1)
    telegram_webhook_secret: SecretStr = Field(SecretStr(""), alias="TELEGRAM_WEBHOOK_SECRET")
    gcp_project: str = Field(alias="GCP_PROJECT", min_length=1)
    bq_dataset: str = Field(alias="BQ_DATASET", min_length=1)
    tenancy: Literal["solo"] = Field("solo", alias="TENANCY")      # "multi" is planned (tenancy.py)
    telegram_chat_id: str = Field("", alias="TELEGRAM_CHAT_ID", pattern=r"^(-?\d+)?$")
    owner_user_id: str = Field("", alias="OWNER_USER_ID")
    model_calls_per_hour: int = Field(20, alias="MODEL_CALLS_PER_HOUR", ge=1, le=1000)
    daily_budget_usd: float = Field(1.0, alias="DAILY_BUDGET_USD", gt=0, le=1000)
    bot_language: Literal["ru", "en"] | None = Field(None, alias="BOT_LANGUAGE")
    data_sources: list[str] = Field(default_factory=list, alias="DATA_SOURCES")

    @field_validator("*", mode="before")
    @classmethod
    def _strip(cls, v):
        return v.strip() if isinstance(v, str) else v

    @field_validator("tenancy", "bot_language", mode="before")
    @classmethod
    def _empty_is_default(cls, v, info):
        return ("solo" if info.field_name == "tenancy" else None) if v == "" else v

    @field_validator("data_sources", mode="before")
    @classmethod
    def _split(cls, v):
        return [s.strip() for s in v.split(",") if s.strip()] if isinstance(v, str) else v

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "BotConfig":
        names = {f.alias for f in cls.model_fields.values() if f.alias}
        try:
            return cls.model_validate({k: v for k, v in env.items() if k in names})
        except ValidationError as e:
            problems = sorted({f"{err['loc'][0]}: {err['msg']}" for err in e.errors(include_input=False)})
            raise ConfigError("invalid bot settings: " + "; ".join(problems)) from None
