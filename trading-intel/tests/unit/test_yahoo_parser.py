"""Yahoo Finance 解析器：純函式，用真實回應樣本測試。

樣本檔（``tests/fixtures/yahoo_2330.json``）是網路開通後對台積電實際發出
一次請求存下來的，因此測的是真實 API 形狀，不是憑印象猜的 schema。
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from trading_intel.core.errors import SchemaValidationError
from trading_intel.ingestion.yahoo import parse_chart

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "yahoo_2330.json"


def test_fixture_exists() -> None:
    assert FIXTURE.exists(), "缺少真實 API 回應樣本，無法離線驗證解析器"


def test_parses_real_response_shape() -> None:
    bars = parse_chart(FIXTURE.read_bytes(), "2330.TW")
    assert len(bars) > 0
    assert all(bar.ticker == "2330.TW" for bar in bars)


def test_prices_are_decimal() -> None:
    bars = parse_chart(FIXTURE.read_bytes(), "2330.TW")
    for bar in bars:
        assert isinstance(bar.close, Decimal)
        assert isinstance(bar.open, Decimal)
        assert bar.close > 0


def test_ohlc_ordering_holds_on_real_data() -> None:
    """真實資料也必須滿足 Phase 0 的 OHLC 不變條件。"""
    bars = parse_chart(FIXTURE.read_bytes(), "2330.TW")
    for bar in bars:
        assert bar.low <= min(bar.open, bar.close)
        assert max(bar.open, bar.close) <= bar.high


def test_bars_are_chronologically_distinct() -> None:
    bars = parse_chart(FIXTURE.read_bytes(), "2330.TW")
    dates = [bar.trade_date for bar in bars]
    assert len(dates) == len(set(dates))


def test_adjclose_defaults_to_close_when_missing() -> None:
    document = json.loads(FIXTURE.read_bytes())
    document["chart"]["result"][0]["indicators"]["adjclose"] = []
    bars = parse_chart(json.dumps(document).encode(), "2330.TW")
    assert all(bar.adjclose == bar.close for bar in bars)


def test_null_values_are_skipped_not_filled() -> None:
    """CLAUDE.md 第 6 條：資料壞掉時停止，不是用插值或前值填補。"""
    document = json.loads(FIXTURE.read_bytes())
    quote = document["chart"]["result"][0]["indicators"]["quote"][0]
    original_count = len(document["chart"]["result"][0]["timestamp"])
    quote["close"][0] = None
    bars = parse_chart(json.dumps(document).encode(), "2330.TW")
    assert len(bars) == original_count - 1


def test_api_error_is_surfaced() -> None:
    payload = json.dumps(
        {"chart": {"result": None, "error": {"code": "Not Found", "description": "無此代號"}}}
    ).encode()
    with pytest.raises(SchemaValidationError, match="Yahoo Finance 回傳錯誤"):
        parse_chart(payload, "0000.TW")


def test_empty_result_yields_empty_list() -> None:
    payload = json.dumps({"chart": {"result": []}}).encode()
    assert parse_chart(payload, "2330.TW") == []


def test_missing_volume_defaults_to_zero() -> None:
    document = json.loads(FIXTURE.read_bytes())
    quote = document["chart"]["result"][0]["indicators"]["quote"][0]
    quote["volume"][0] = None
    bars = parse_chart(json.dumps(document).encode(), "2330.TW")
    assert bars[0].volume == 0
