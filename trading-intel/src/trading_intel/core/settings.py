"""Layered configuration: ``configs/base.yaml`` -> ``configs/{env}.yaml`` -> env vars.

Settings are frozen. A run's configuration is part of its reproducibility story,
so it must not drift while the process is alive.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from trading_intel.core.errors import ConfigError

#: Repository root, i.e. the directory holding ``configs/``. ``TI_CONFIG_DIR``
#: overrides it for installed deployments where the source tree is gone.
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def config_dir() -> Path:
    override = os.environ.get("TI_CONFIG_DIR")
    return Path(override) if override else PROJECT_ROOT / "configs"


_VALID_ENVS = ("dev", "staging", "prod")


class DataSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    raw_root: Path
    parquet_root: Path
    max_staleness_minutes: int = Field(gt=0)


class RiskLimits(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_position_weight: float = Field(gt=0, le=1)
    max_sector_weight: float = Field(gt=0, le=1)
    max_gross_exposure: float = Field(gt=0)
    max_net_exposure: float = Field(ge=0)
    drawdown_derisk: float = Field(gt=0, le=1)
    drawdown_flatten: float = Field(gt=0, le=1)
    adv_participation_cap: float = Field(gt=0, le=1)
    top5_concentration_cap: float = Field(gt=0, le=1)


class CostModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tw_transaction_tax: float = 0.003
    tw_daytrade_tax: float = 0.0015
    commission_bps: float = Field(ge=0)
    slippage_bps: float = Field(ge=0)
    impact_coefficient: float = Field(ge=0)


class AgentBudget(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    daily_token_budget: int = Field(gt=0)
    max_calls_per_hour: int = Field(gt=0)
    timeout_seconds: float = Field(gt=0)
    max_retries: int = Field(ge=0)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TI_",
        env_nested_delimiter="__",
        frozen=True,
        extra="forbid",
    )

    env: Literal["dev", "staging", "prod"]
    display_tz: str = "Asia/Taipei"
    data: DataSettings
    risk: RiskLimits
    costs: CostModel
    agents: AgentBudget

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],  # noqa: ARG003  (signature fixed by pydantic-settings)
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Environment variables outrank the YAML we pass in as init kwargs."""
        return (env_settings, dotenv_settings, file_secret_settings, init_settings)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError("configuration file not found", path=str(path))
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError("configuration file must contain a mapping", path=str(path))
    return loaded


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


@lru_cache(maxsize=1)
def load_settings(env: str | None = None) -> Settings:
    """Load settings for ``env`` (default: ``$TI_ENV``, else ``dev``).

    Layering is base -> environment file -> process environment, so an operator
    can always override a single value without editing a file.
    """
    resolved = env or os.environ.get("TI_ENV", "dev")
    if resolved not in _VALID_ENVS:
        raise ConfigError("unknown environment", env=resolved, valid=", ".join(_VALID_ENVS))

    data = _deep_merge(
        _read_yaml(config_dir() / "base.yaml"), _read_yaml(config_dir() / f"{resolved}.yaml")
    )
    data["env"] = resolved
    try:
        return Settings(**data)
    except ValidationError as exc:
        paths = ["/".join(str(part) for part in err["loc"]) for err in exc.errors()]
        raise ConfigError(
            "invalid configuration",
            env=resolved,
            fields=", ".join(paths),
            detail=str(exc),
        ) from exc
