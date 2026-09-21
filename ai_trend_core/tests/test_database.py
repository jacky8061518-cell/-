"""
tests/test_database.py
針對 core/database.py 的單元測試：確保訊號、AI 推理過程與市場數據能正確持久化。
"""

from __future__ import annotations

from pathlib import Path

import pytest

import core.database as database


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """每個測試使用獨立的臨時資料庫檔案，避免互相污染或寫到正式資料庫。"""
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test_signals.db")
    database.init_db()
    yield


def test_init_db_creates_database_file():
    assert database.DB_PATH.exists()


def test_insert_and_get_recent_signals():
    row_id = database.insert_signal(
        symbol="BTC-USD", signal_type="OVERBOUGHT", price=65000.0, z_score=2.3,
        upper_band=66000.0, lower_band=60000.0, mean_price=63000.0,
        action="Sell", entry=65000.0, take_profit=62000.0, stop_loss=67000.0,
        confidence=0.82, reasoning="測試理由", sentiment_summary="測試情緒",
        raw_market_data={"foo": "bar"},
    )
    assert row_id == 1

    rows = database.get_recent_signals(limit=10)
    assert len(rows) == 1
    assert rows[0]["symbol"] == "BTC-USD"
    assert rows[0]["action"] == "Sell"
    assert rows[0]["confidence"] == pytest.approx(0.82)
    assert rows[0]["raw_market_data"] == '{"foo": "bar"}'


def test_insert_signal_with_missing_optional_fields():
    """CrewAI 解析失敗、AI 未給出完整建議時，選填欄位應允許為 None 而不拋錯。"""
    row_id = database.insert_signal(
        symbol="ETH-USD", signal_type="OVERSOLD", price=3000.0, z_score=-2.1,
        upper_band=3200.0, lower_band=2800.0, mean_price=3000.0,
    )
    rows = database.get_recent_signals(limit=10)
    row = next(r for r in rows if r["id"] == row_id)
    assert row["action"] is None
    assert row["confidence"] is None


def test_get_signals_for_symbol_filters_correctly():
    database.insert_signal(
        symbol="BTC-USD", signal_type="OVERBOUGHT", price=1.0, z_score=2.1,
        upper_band=1.1, lower_band=0.9, mean_price=1.0,
    )
    database.insert_signal(
        symbol="ETH-USD", signal_type="OVERSOLD", price=1.0, z_score=-2.1,
        upper_band=1.1, lower_band=0.9, mean_price=1.0,
    )

    btc_rows = database.get_signals_for_symbol("BTC-USD")
    assert len(btc_rows) == 1
    assert btc_rows[0]["symbol"] == "BTC-USD"


def test_get_recent_signals_orders_newest_first():
    first_id = database.insert_signal(
        symbol="A", signal_type="OVERBOUGHT", price=1.0, z_score=2.1,
        upper_band=1.1, lower_band=0.9, mean_price=1.0,
    )
    second_id = database.insert_signal(
        symbol="B", signal_type="OVERSOLD", price=1.0, z_score=-2.1,
        upper_band=1.1, lower_band=0.9, mean_price=1.0,
    )
    rows = database.get_recent_signals(limit=10)
    assert rows[0]["id"] == second_id
    assert rows[1]["id"] == first_id


def test_get_connection_creates_parent_directory(tmp_path, monkeypatch):
    nested_path = tmp_path / "nested" / "dir" / "signals.db"
    monkeypatch.setattr(database, "DB_PATH", nested_path)
    database.init_db()
    assert nested_path.exists()
    assert nested_path.parent.is_dir()
