"""分層設定的載入順序與錯誤訊息。"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from trading_intel.core.errors import ConfigError
from trading_intel.core.settings import Settings, load_settings

BASE_YAML = """
display_tz: Asia/Taipei
data:
  raw_root: ./data/raw
  parquet_root: ./data/parquet
  max_staleness_minutes: 60
risk:
  max_position_weight: 0.05
  max_sector_weight: 0.25
  max_gross_exposure: 1.0
  max_net_exposure: 0.6
  drawdown_derisk: 0.08
  drawdown_flatten: 0.15
  adv_participation_cap: 0.05
  top5_concentration_cap: 0.4
costs:
  commission_bps: 1.425
  slippage_bps: 5.0
  impact_coefficient: 0.1
agents:
  daily_token_budget: 1000000
  max_calls_per_hour: 200
  timeout_seconds: 60.0
  max_retries: 2
"""


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Iterator[None]:
    load_settings.cache_clear()
    yield
    load_settings.cache_clear()


@pytest.fixture
def config_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "base.yaml").write_text(BASE_YAML, encoding="utf-8")
    (tmp_path / "dev.yaml").write_text(
        "data:\n  max_staleness_minutes: 1440\nagents:\n  max_calls_per_hour: 20\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TI_CONFIG_DIR", str(tmp_path))
    return tmp_path


def test_repository_configs_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """簽入 repo 的 configs/ 本身必須合法，不能只有測試 fixture 合法。"""
    monkeypatch.delenv("TI_CONFIG_DIR", raising=False)
    settings = load_settings("dev")
    assert isinstance(settings, Settings)
    assert settings.env == "dev"
    assert settings.display_tz == "Asia/Taipei"


@pytest.mark.parametrize("env", ["dev", "staging", "prod"])
def test_every_shipped_environment_loads(env: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TI_CONFIG_DIR", raising=False)
    load_settings.cache_clear()
    assert load_settings(env).env == env


def test_environment_file_overrides_base(config_root: Path) -> None:
    assert config_root.exists()
    settings = load_settings("dev")
    assert settings.data.max_staleness_minutes == 1440  # 來自 dev.yaml
    assert settings.agents.max_calls_per_hour == 20  # 來自 dev.yaml
    assert settings.agents.daily_token_budget == 1000000  # 未被覆寫，來自 base.yaml


def test_env_vars_outrank_yaml(config_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert config_root.exists()
    monkeypatch.setenv("TI_DISPLAY_TZ", "America/New_York")
    monkeypatch.setenv("TI_DATA__MAX_STALENESS_MINUTES", "5")
    settings = load_settings("dev")
    assert settings.display_tz == "America/New_York"
    assert settings.data.max_staleness_minutes == 5


def test_settings_are_frozen(config_root: Path) -> None:
    assert config_root.exists()
    settings = load_settings("dev")
    with pytest.raises(Exception, match=r"frozen"):
        settings.display_tz = "UTC"  # type: ignore[misc]


def test_load_settings_is_cached(config_root: Path) -> None:
    assert config_root.exists()
    assert load_settings("dev") is load_settings("dev")


def test_unknown_environment_is_rejected(config_root: Path) -> None:
    assert config_root.exists()
    with pytest.raises(ConfigError, match="未知的環境名稱"):
        load_settings("qa")


def test_missing_config_file_names_the_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TI_CONFIG_DIR", str(tmp_path))
    with pytest.raises(ConfigError) as excinfo:
        load_settings("dev")
    assert "base.yaml" in excinfo.value.context["path"]


def test_missing_field_error_names_the_field_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    incomplete = BASE_YAML.replace("  drawdown_flatten: 0.15\n", "")
    (tmp_path / "base.yaml").write_text(incomplete, encoding="utf-8")
    (tmp_path / "dev.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("TI_CONFIG_DIR", str(tmp_path))
    with pytest.raises(ConfigError) as excinfo:
        load_settings("dev")
    assert excinfo.value.context["fields"] == "risk/drawdown_flatten"
    assert "drawdown_flatten" in excinfo.value.context["detail"]


def test_out_of_range_risk_limit_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broken = BASE_YAML.replace("max_position_weight: 0.05", "max_position_weight: 1.5")
    (tmp_path / "base.yaml").write_text(broken, encoding="utf-8")
    (tmp_path / "dev.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("TI_CONFIG_DIR", str(tmp_path))
    with pytest.raises(ConfigError) as excinfo:
        load_settings("dev")
    assert "risk/max_position_weight" in excinfo.value.context["fields"]


def test_unknown_key_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "base.yaml").write_text(BASE_YAML, encoding="utf-8")
    (tmp_path / "dev.yaml").write_text("risk:\n  max_leverage: 3.0\n", encoding="utf-8")
    monkeypatch.setenv("TI_CONFIG_DIR", str(tmp_path))
    with pytest.raises(ConfigError) as excinfo:
        load_settings("dev")
    assert "max_leverage" in excinfo.value.context["fields"]


def test_non_mapping_config_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "base.yaml").write_text("- just\n- a\n- list\n", encoding="utf-8")
    monkeypatch.setenv("TI_CONFIG_DIR", str(tmp_path))
    with pytest.raises(ConfigError, match="內容必須是一個對應表"):
        load_settings("dev")


def test_empty_overlay_is_allowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "base.yaml").write_text(BASE_YAML, encoding="utf-8")
    (tmp_path / "dev.yaml").write_text("", encoding="utf-8")
    monkeypatch.setenv("TI_CONFIG_DIR", str(tmp_path))
    assert load_settings("dev").data.max_staleness_minutes == 60


def test_default_env_comes_from_ti_env(config_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert config_root.exists()
    monkeypatch.setenv("TI_ENV", "dev")
    assert load_settings().env == "dev"
