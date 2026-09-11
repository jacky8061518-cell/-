"""分層設定：``configs/base.yaml`` -> ``configs/{env}.yaml`` -> 環境變數。

設定是 frozen 的。一次執行所使用的設定屬於其可重現性的一部分，
因此在行程存活期間不得變動。

對應 CLAUDE.md 第 5 條：所有參數放 configs/，程式碼中出現魔術數字即為 bug。
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from trading_intel.core.errors import ConfigError

#: 專案根目錄，也就是放 ``configs/`` 的那一層。安裝後部署已無原始碼樹時，
#: 可用 ``TI_CONFIG_DIR`` 環境變數覆寫。
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


class QualityLimits(BaseModel):
    """資料品質閘門的門檻（SPEC 3.3）。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_staleness_minutes: int = Field(gt=0)
    tw_daily_return_limit: float = Field(gt=0)
    us_daily_return_limit: float = Field(gt=0)
    max_missing_session_ratio: float = Field(ge=0, le=1)
    psi_warn: float = Field(gt=0)
    psi_disable: float = Field(gt=0)
    cross_source_tolerance: float = Field(gt=0)

    @model_validator(mode="after")
    def _validate_psi_order(self) -> Self:
        if self.psi_disable <= self.psi_warn:
            msg = "psi_disable 必須大於 psi_warn"
            raise ValueError(msg)
        return self


class AgentBudget(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    daily_token_budget: int = Field(gt=0)
    max_calls_per_hour: int = Field(gt=0)
    timeout_seconds: float = Field(gt=0)
    max_retries: int = Field(ge=0)


class AutoLevelRules(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    min_live_months: int = Field(ge=0)
    max_position_weight: float = Field(gt=0, le=1)
    require_all_validations: bool


class ConfirmLevelRules(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    new_signal_days: int = Field(ge=0)
    large_adjustment_weight: float = Field(gt=0, le=1)
    during_regime_shift: bool


class ResearchOnlyRules(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    single_source_news: bool
    agent_conflict: bool
    hypothesis_stage: bool


class Governance(BaseModel):
    """人機分權三級（SPEC 7.4）。政策寫在設定檔，不寫在程式碼。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    auto: AutoLevelRules
    confirm: ConfirmLevelRules
    research_only: ResearchOnlyRules


class FusionSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    shrinkage: float | None = None
    correlation_merge_threshold: float = Field(gt=0, le=1)
    min_decayed_weight: float = Field(ge=0, le=1)


class PortfolioSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    target_annual_volatility: float = Field(gt=0)
    max_leverage_from_vol_target: float = Field(gt=0)
    turnover_penalty: float = Field(ge=0)
    min_position_weight: float = Field(ge=0, le=1)


class MonitoringSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    heartbeat_timeout_seconds: int = Field(gt=0)
    reconciliation_interval_minutes: int = Field(gt=0)
    realized_vol_multiple_for_derisk: float = Field(gt=1)
    max_orders_per_minute: int = Field(gt=0)
    max_price_deviation: float = Field(gt=0, le=1)


class ModelRiskSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    ic_decay_periods: int = Field(gt=0)
    min_rolling_ic: float
    max_live_backtest_tracking_error: float = Field(gt=0)
    min_agent_consistency: float = Field(gt=0, le=1)


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
    quality: QualityLimits
    governance: Governance
    fusion: FusionSettings
    portfolio: PortfolioSettings
    monitoring: MonitoringSettings
    model_risk: ModelRiskSettings

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],  # noqa: ARG003  (signature fixed by pydantic-settings)
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """環境變數的優先序高於以 init kwargs 傳入的 YAML 內容。

        pydantic-settings 預設 init kwargs 優先，與 SPEC 要求的疊加順序相反，
        因此在此把 env_settings 排到 init_settings 之前。
        """
        return (env_settings, dotenv_settings, file_secret_settings, init_settings)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError("找不到設定檔", path=str(path))
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError("設定檔內容必須是一個對應表", path=str(path))
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
    """載入 ``env`` 的設定（預設取 ``$TI_ENV``，再預設為 ``dev``）。

    疊加順序為 base -> 環境設定檔 -> 行程環境變數，
    如此維運人員永遠可以不改檔案就覆寫單一數值。
    """
    resolved = env or os.environ.get("TI_ENV", "dev")
    if resolved not in _VALID_ENVS:
        raise ConfigError("未知的環境名稱", env=resolved, valid=", ".join(_VALID_ENVS))

    data = _deep_merge(
        _read_yaml(config_dir() / "base.yaml"), _read_yaml(config_dir() / f"{resolved}.yaml")
    )
    data["env"] = resolved
    try:
        return Settings(**data)
    except ValidationError as exc:
        paths = ["/".join(str(part) for part in err["loc"]) for err in exc.errors()]
        raise ConfigError(
            "設定內容不合法",
            env=resolved,
            fields=", ".join(paths),
            detail=str(exc),
        ) from exc
