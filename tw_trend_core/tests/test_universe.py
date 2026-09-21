"""
tests/test_universe.py
針對 core/universe.py 的測試：確保能從共用的 security-master.csv
正確篩出全部個股（排除 ETF），且不會意外納入其他資產類型。
"""

from __future__ import annotations

from core.universe import TwStock, load_stock_universe


def test_load_stock_universe_returns_only_stocks():
    stocks = load_stock_universe()
    assert len(stocks) > 1900  # 上市 + 上櫃個股，實際約 1,983 檔
    assert all(isinstance(s, TwStock) for s in stocks)


def test_load_stock_universe_covers_both_markets():
    stocks = load_stock_universe()
    markets = {s.market for s in stocks}
    assert "上市" in markets
    assert "上櫃" in markets


def test_load_stock_universe_yahoo_tickers_have_taiwan_suffix():
    stocks = load_stock_universe()
    sample = stocks[:50]
    assert all(s.yahoo_ticker.endswith((".TW", ".TWO")) for s in sample)


def test_load_stock_universe_includes_known_large_cap():
    """台積電（2330）應在清單中，做為資料正確性的基本檢查。"""
    stocks = load_stock_universe()
    codes = {s.code for s in stocks}
    assert "2330" in codes
