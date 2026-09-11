"""真實網路呼叫的整合測試。

這一組測試**實際連網**，驗證的是「網路政策開通後，哪些資料源真的能用」，
而不是又一次的離線解析驗證（那由 ``tests/unit/test_yahoo_parser.py`` 與
``tests/unit/test_twse_parsers.py`` 負責）。

網路不可用時整組跳過，因此離線開發與網路受限的 CI 不會被這組測試卡住。
證交所（TWSE）openapi 與 www 端點會被 TWSE 自己的 WAF 攔截（見 ADR 0007），
因此本檔不測 TWSE，只測實測證實可用的兩個來源：Yahoo Finance 與 TPEx。
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from trading_intel.core.clock import utc_now
from trading_intel.ingestion.twse import (
    fetch_tpex_institutional_flows,
    parse_tpex_institutional_flows,
)
from trading_intel.ingestion.yahoo import fetch_chart, parse_chart


def _network_available(host: str) -> bool:
    import socket

    try:
        socket.create_connection((host, 443), timeout=5).close()
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _network_available("query1.finance.yahoo.com"),
    reason="需要網路存取 Yahoo Finance；此環境的網路政策未開通",
)


def _last_trading_day() -> date:
    """粗略取一個近期已收盤的交易日，避開週末。

    這是測試碼但仍走 core.clock：CLAUDE.md 第 1 條沒有「測試除外」這種例外，
    ruff 的 banned-api 與 tests/guard 的 AST 守門也不會放過 tests/。
    """
    candidate = utc_now().date() - timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate -= timedelta(days=1)
    return candidate


def test_yahoo_finance_returns_real_tsmc_prices() -> None:
    payload, request = fetch_chart("2330.TW", range_="5d")
    assert request["url"].endswith("2330.TW")
    bars = parse_chart(payload, "2330.TW")
    assert len(bars) > 0
    # 台積電股價應該落在合理量級，不是 0 或荒謬的數字。
    assert all(bar.close > 100 for bar in bars)


def test_yahoo_finance_works_for_a_us_stock_too() -> None:
    """驗證這條路徑不是只對台股湊巧有效。"""
    payload, _request = fetch_chart("AAPL", range_="5d")
    bars = parse_chart(payload, "AAPL")
    assert len(bars) > 0


def test_yahoo_finance_unknown_ticker_raises() -> None:
    """無效代號時 Yahoo 直接回 HTTP 404，這裡確認例外會往外傳而非被吞掉。

    （若 Yahoo 改成回應一個帶 error 欄位的 200，parse_chart 的
    SchemaValidationError 分支會接手——那一段已由離線測試覆蓋。）
    """
    import urllib.error

    with pytest.raises(urllib.error.HTTPError) as excinfo:
        fetch_chart("ZZZZINVALID9999.TW", range_="5d")
    assert excinfo.value.code == 404


@pytest.mark.skipif(
    not _network_available("www.tpex.org.tw"),
    reason="需要網路存取櫃買中心",
)
def test_tpex_institutional_flows_are_reachable() -> None:
    """櫃買中心三大法人資料在網路開通後可正常抓取（Phase 1 原本卡住的部分）。"""
    session = _last_trading_day()
    payload, request = fetch_tpex_institutional_flows(session)
    assert request["params"]["d"]
    flows = parse_tpex_institutional_flows(payload, session)
    # 可能剛好抓到非交易日，因此只驗證格式而非資料筆數。
    if flows:
        assert flows[0].local_symbol
        assert flows[0].trade_date == session
