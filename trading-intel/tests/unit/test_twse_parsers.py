"""TWSE／TPEx 解析器：純函式，吃 bytes。

樣本依照真實回應的形狀構造（欄位數、民國年格式、千分位、``--`` 佔位）。
解析器不碰網路，因此這一組測試在任何環境都能跑——包括 TWSE 被防火牆擋住時。
"""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal

import pytest

from trading_intel.core.enums import Market
from trading_intel.core.errors import SchemaValidationError
from trading_intel.ingestion.twse import (
    DividendEvent,
    _to_decimal,
    dividend_adjustment_factor,
    parse_roc_date,
    parse_tpex_institutional_flows,
    parse_twse_dividends,
    parse_twse_institutional_flows,
    yahoo_ticker,
)

SESSION = date(2024, 3, 19)


def twse_row(code: str, name: str, foreign: str, trust: str, dealer: str, total: str) -> list[str]:
    """TWSE T86 一列共 19 欄；索引 4/10/11/18 是三大法人。"""
    row = [""] * 19
    row[0], row[1] = code, name
    row[4], row[10], row[11], row[18] = foreign, trust, dealer, total
    return row


def tpex_row(code: str, name: str, foreign: str, trust: str, dealer: str, total: str) -> list[str]:
    """TPEx 一列共 24 欄；索引 4/13/22/23 是三大法人，與上市完全不同。"""
    row = [""] * 24
    row[0], row[1] = code, name
    row[4], row[13], row[22], row[23] = foreign, trust, dealer, total
    return row


# --- TWSE 三大法人 ---------------------------------------------------------


def test_parse_twse_flows() -> None:
    payload = json.dumps(
        {
            "stat": "OK",
            "data": [
                twse_row("2330", "台積電", "1,234,000", "56,000", "-7,800", "1,282,200"),
                twse_row("2317", "鴻海", "-500,000", "0", "1,000", "-499,000"),
            ],
        }
    ).encode("utf-8")
    flows = parse_twse_institutional_flows(payload, SESSION)
    assert len(flows) == 2
    assert flows[0].local_symbol == "2330"
    assert flows[0].market is Market.TW
    assert flows[0].foreign_net_shares == Decimal("1234000")
    assert flows[0].dealer_net_shares == Decimal("-7800")
    assert flows[1].foreign_net_shares == Decimal("-500000")
    assert flows[0].trade_date == SESSION


def test_twse_non_ok_status_returns_empty() -> None:
    payload = json.dumps({"stat": "很抱歉，沒有符合條件的資料!"}).encode("utf-8")
    assert parse_twse_institutional_flows(payload, SESSION) == []


def test_twse_short_rows_are_skipped() -> None:
    payload = json.dumps({"stat": "OK", "data": [["2330", "台積電"]]}).encode("utf-8")
    assert parse_twse_institutional_flows(payload, SESSION) == []


# --- TPEx 三大法人 ---------------------------------------------------------


def test_parse_tpex_flows_uses_different_column_indices() -> None:
    payload = json.dumps(
        {
            "stat": "ok",
            "tables": [
                {
                    "data": [
                        tpex_row("6488", "環球晶", "12,000", "3,000", "-500", "14,500"),
                    ]
                }
            ],
        }
    ).encode("utf-8")
    flows = parse_tpex_institutional_flows(payload, SESSION)
    assert len(flows) == 1
    assert flows[0].local_symbol == "6488"
    assert flows[0].trust_net_shares == Decimal("3000")
    assert flows[0].dealer_net_shares == Decimal("-500")


def test_tpex_empty_tables_return_empty() -> None:
    payload = json.dumps({"stat": "ok", "tables": [{"data": []}]}).encode("utf-8")
    assert parse_tpex_institutional_flows(payload, SESSION) == []
    assert parse_tpex_institutional_flows(json.dumps({"stat": "no"}).encode(), SESSION) == []


# --- 數值與日期轉換 --------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1,234", Decimal("1234")),
        ("-500", Decimal("-500")),
        ("--", Decimal("0")),
        ("---", Decimal("0")),
        ("", Decimal("0")),
        ("N/A", Decimal("0")),
        (None, Decimal("0")),
        ("12.5", Decimal("12.5")),
    ],
)
def test_to_decimal_handles_real_world_placeholders(raw: object, expected: Decimal) -> None:
    assert _to_decimal(raw) == expected


def test_to_decimal_rejects_garbage() -> None:
    with pytest.raises(SchemaValidationError, match="無法轉換為數值"):
        _to_decimal("abc")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("113/03/19", date(2024, 3, 19)), ("1130319", date(2024, 3, 19))],
)
def test_parse_roc_date(raw: str, expected: date) -> None:
    assert parse_roc_date(raw) == expected


@pytest.mark.parametrize("raw", ["2024/03/19", "", "11303", "abc/de/fg"])
def test_parse_roc_date_rejects_bad_input(raw: str) -> None:
    with pytest.raises(SchemaValidationError, match="民國年日期格式"):
        parse_roc_date(raw)


def test_yahoo_ticker() -> None:
    assert yahoo_ticker("2330", ".TW") == "2330.TW"
    assert yahoo_ticker(" 6488 ", ".TWO") == "6488.TWO"


# --- 除權除息（Phase 1 新增的資料源）--------------------------------------


def test_parse_dividends() -> None:
    payload = json.dumps(
        [
            {
                "Date": "113/06/17",
                "Code": "2330",
                "Name": "台積電",
                "PreviousClosingPrice": "1,000.00",
                "ExRightsExDividendReferencePrice": "996.50",
                "CashDividend": "3.50",
                "StockDividend": "0",
            }
        ]
    ).encode("utf-8")
    events = parse_twse_dividends(payload)
    assert len(events) == 1
    assert events[0].ex_date == date(2024, 6, 17)
    assert events[0].local_symbol == "2330"
    assert events[0].cash_dividend == Decimal("3.50")


def test_dividend_factor_matches_the_official_reference_price() -> None:
    """因子直接由交易所公告的參考價推導，不是猜的。"""
    event = DividendEvent(
        ex_date=date(2024, 6, 17),
        local_symbol="2330",
        name="台積電",
        prior_close=Decimal("1000"),
        reference_price=Decimal("996.5"),
        cash_dividend=Decimal("3.5"),
        stock_dividend=Decimal("0"),
    )
    assert dividend_adjustment_factor(event) == Decimal("0.9965")


def test_dividend_factor_rejects_non_positive_prior_close() -> None:
    event = DividendEvent(
        ex_date=date(2024, 6, 17),
        local_symbol="2330",
        name="台積電",
        prior_close=Decimal("0"),
        reference_price=Decimal("996.5"),
        cash_dividend=Decimal("3.5"),
        stock_dividend=Decimal("0"),
    )
    with pytest.raises(SchemaValidationError, match="必須為正數"):
        dividend_adjustment_factor(event)


def test_dividend_rows_missing_prices_are_skipped() -> None:
    payload = json.dumps(
        [
            {"Date": "113/06/17", "Code": "2330", "PreviousClosingPrice": "0"},
            {"Date": "", "Code": "2317"},
            "不是字典",
        ]
    ).encode("utf-8")
    assert parse_twse_dividends(payload) == []
