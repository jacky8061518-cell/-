"""特徵註冊表：版本、相依欄位、值域把關、橫斷面中性化。"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from trading_intel.core.clock import UTC
from trading_intel.core.errors import SchemaValidationError
from trading_intel.features.registry import (
    FeatureRegistry,
    FeatureSpec,
    neutralize,
)

ASOF = datetime(2024, 3, 19, tzinfo=UTC)


def simple_feature(data, asof, scale: float = 1.0):  # type: ignore[no-untyped-def]  # noqa: ARG001
    return np.asarray(data, dtype=np.float64) * scale


def spec(version: str = "v1", **kwargs) -> FeatureSpec:  # type: ignore[no-untyped-def]
    defaults = {
        "name": "mom_20d",
        "version": version,
        "function": simple_feature,
        "required_columns": (),
    }
    defaults.update(kwargs)
    return FeatureSpec(**defaults)  # type: ignore[arg-type]


# --- 註冊與版本 ------------------------------------------------------------


def test_register_and_get() -> None:
    registry = FeatureRegistry()
    registry.register(spec())
    assert len(registry) == 1
    assert registry.get("mom_20d", "v1").qualified_name == "mom_20d@v1"


def test_qualified_name_includes_the_version() -> None:
    """訊號記錄的是 name@version，不是裸名稱——否則舊訊號無法重現。"""
    assert spec("v2").qualified_name == "mom_20d@v2"


def test_duplicate_version_is_rejected() -> None:
    """改了實作卻忘了升版本，會讓舊訊號無法重現，因此直接拒絕。"""
    registry = FeatureRegistry()
    registry.register(spec())
    with pytest.raises(SchemaValidationError, match="改了實作請升版本"):
        registry.register(spec())


def test_multiple_versions_coexist() -> None:
    registry = FeatureRegistry()
    registry.register(spec("v1"))
    registry.register(spec("v2"))
    assert registry.versions_of("mom_20d") == ("v1", "v2")
    assert registry.names() == ("mom_20d",)


def test_unknown_feature_is_rejected() -> None:
    with pytest.raises(SchemaValidationError, match="未註冊的特徵"):
        FeatureRegistry().get("nope", "v1")


def test_contains_uses_the_qualified_name() -> None:
    registry = FeatureRegistry()
    registry.register(spec())
    assert "mom_20d@v1" in registry
    assert "mom_20d@v2" not in registry


def test_function_without_asof_is_rejected() -> None:
    """CLAUDE.md 第 2 條：讀取歷史資料的函式必須接受 asof。"""

    def bad(data):  # type: ignore[no-untyped-def]  # noqa: ARG001
        return np.array([1.0])

    with pytest.raises(SchemaValidationError, match="必須接受 data 與 asof"):
        FeatureRegistry().register(spec(function=bad))


# --- 計算與把關 ------------------------------------------------------------


def test_compute_applies_the_function() -> None:
    registry = FeatureRegistry()
    registry.register(spec())
    result = registry.compute("mom_20d", "v1", [1.0, 2.0, 3.0], asof=ASOF)
    assert list(result) == [1.0, 2.0, 3.0]


def test_registered_params_are_defaults() -> None:
    registry = FeatureRegistry()
    registry.register(spec(params={"scale": 2.0}))
    result = registry.compute("mom_20d", "v1", [1.0, 2.0], asof=ASOF)
    assert list(result) == [2.0, 4.0]


def test_call_params_override_registered_ones() -> None:
    registry = FeatureRegistry()
    registry.register(spec(params={"scale": 2.0}))
    result = registry.compute("mom_20d", "v1", [1.0], asof=ASOF, scale=10.0)
    assert list(result) == [10.0]


def test_output_outside_the_declared_range_is_rejected() -> None:
    """宣稱輸出在 [-1, 1] 的特徵吐出 15，代表計算或輸入出錯，該報錯而非放行。"""
    registry = FeatureRegistry()
    registry.register(spec(expected_range=(-1.0, 1.0)))
    with pytest.raises(SchemaValidationError, match="超出宣告的預期值域"):
        registry.compute("mom_20d", "v1", [15.0], asof=ASOF)


def test_output_inside_the_range_passes() -> None:
    registry = FeatureRegistry()
    registry.register(spec(expected_range=(-1.0, 1.0)))
    assert list(registry.compute("mom_20d", "v1", [0.5], asof=ASOF)) == [0.5]


def test_all_nan_output_skips_the_range_check() -> None:
    registry = FeatureRegistry()
    registry.register(spec(expected_range=(-1.0, 1.0)))
    result = registry.compute("mom_20d", "v1", [float("nan")], asof=ASOF)
    assert np.isnan(result).all()


def test_missing_required_column_is_rejected() -> None:
    import pandas as pd

    registry = FeatureRegistry()
    registry.register(spec(required_columns=("close", "volume")))
    frame = pd.DataFrame({"close": [1.0, 2.0]})
    with pytest.raises(SchemaValidationError, match="缺少特徵所需的欄位"):
        registry.compute("mom_20d", "v1", frame, asof=ASOF)


def test_data_without_columns_skips_the_column_check() -> None:
    registry = FeatureRegistry()
    registry.register(spec(required_columns=("close",)))
    assert list(registry.compute("mom_20d", "v1", [1.0], asof=ASOF)) == [1.0]


def test_mean_reverting_flag_is_recorded() -> None:
    assert spec(mean_reverting=True).mean_reverting is True
    assert spec().mean_reverting is False


# --- 橫斷面中性化 ----------------------------------------------------------


def test_industry_neutralization_removes_the_group_effect() -> None:
    """不中性化的話，「動量」在半導體大漲那個月會變成「押注半導體」。"""
    values = [10.0, 12.0, 1.0, 3.0]
    groups = ["半導體", "半導體", "金融", "金融"]
    result = neutralize(values, groups)
    # 每組減去組平均後，組內平均為零。
    assert result[0] + result[1] == pytest.approx(0.0)
    assert result[2] + result[3] == pytest.approx(0.0)
    # 但組內的相對強弱保留下來。
    assert result[1] > result[0]
    assert result[3] > result[2]


def test_neutralization_preserves_within_group_ordering() -> None:
    values = [1.0, 5.0, 3.0]
    groups = ["A", "A", "A"]
    result = neutralize(values, groups)
    assert list(np.argsort(result)) == list(np.argsort(values))


def test_market_cap_neutralization_removes_the_size_tilt() -> None:
    """市值中性化後，特徵與 log(市值) 的相關性應接近零。"""
    rng = np.random.default_rng(20240319)
    caps = np.exp(rng.uniform(20, 25, 200))
    # 刻意構造一個純粹由市值驅動的特徵。
    values = 2.0 * np.log(caps) + rng.normal(0, 0.1, 200)
    groups = ["同一產業"] * 200
    result = neutralize(values, groups, market_cap=caps)
    correlation = float(np.corrcoef(result, np.log(caps))[0, 1])
    assert abs(correlation) < 0.05


def test_neutralize_rejects_length_mismatch() -> None:
    with pytest.raises(SchemaValidationError, match="長度不一致"):
        neutralize([1.0, 2.0], ["A"])


def test_neutralize_rejects_market_cap_length_mismatch() -> None:
    with pytest.raises(SchemaValidationError, match="市值與數值的長度不一致"):
        neutralize([1.0, 2.0], ["A", "A"], market_cap=[1.0])


def test_neutralize_handles_all_nan_group() -> None:
    result = neutralize([float("nan"), float("nan")], ["A", "A"])
    assert np.isnan(result).all()


def test_neutralize_skips_regression_with_too_few_points() -> None:
    values = [1.0, 2.0]
    result = neutralize(values, ["A", "A"], market_cap=[100.0, 200.0])
    assert np.isfinite(result).all()


def test_neutralize_ignores_non_positive_market_caps() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    caps = [100.0, 0.0, 300.0, 400.0]
    result = neutralize(values, ["A"] * 4, market_cap=caps)
    assert np.isfinite(result).all()
