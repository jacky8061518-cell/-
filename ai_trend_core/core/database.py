"""
core/database.py
資料庫模組：使用 SQLite 儲存所有產生的交易訊號、AI 推理過程、信心評分及當時的市場數據。
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "trading_signals.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    symbol TEXT NOT NULL,
    signal_type TEXT NOT NULL,
    price REAL NOT NULL,
    z_score REAL NOT NULL,
    upper_band REAL NOT NULL,
    lower_band REAL NOT NULL,
    mean_price REAL NOT NULL,
    action TEXT,
    entry REAL,
    take_profit REAL,
    stop_loss REAL,
    confidence REAL,
    reasoning TEXT,
    sentiment_summary TEXT,
    raw_market_data TEXT
);
"""


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """建立資料庫連線，並確保資料庫所在目錄存在。"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    """初始化資料庫，若資料表不存在則建立。"""
    with get_connection() as conn:
        conn.execute(SCHEMA)
        conn.commit()


def insert_signal(
    symbol: str,
    signal_type: str,
    price: float,
    z_score: float,
    upper_band: float,
    lower_band: float,
    mean_price: float,
    action: str | None = None,
    entry: float | None = None,
    take_profit: float | None = None,
    stop_loss: float | None = None,
    confidence: float | None = None,
    reasoning: str | None = None,
    sentiment_summary: str | None = None,
    raw_market_data: dict[str, Any] | None = None,
) -> int:
    """寫入一筆新的交易訊號紀錄，回傳該筆記錄的 id。"""
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO signals (
                created_at, symbol, signal_type, price, z_score,
                upper_band, lower_band, mean_price, action, entry,
                take_profit, stop_loss, confidence, reasoning,
                sentiment_summary, raw_market_data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                symbol,
                signal_type,
                price,
                z_score,
                upper_band,
                lower_band,
                mean_price,
                action,
                entry,
                take_profit,
                stop_loss,
                confidence,
                reasoning,
                sentiment_summary,
                json.dumps(raw_market_data, default=str) if raw_market_data else None,
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def get_recent_signals(limit: int = 50) -> list[dict[str, Any]]:
    """取得最近的交易訊號紀錄，依時間新到舊排序。"""
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM signals ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]


def get_signals_for_symbol(symbol: str, limit: int = 50) -> list[dict[str, Any]]:
    """取得指定標的的歷史交易訊號紀錄。"""
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM signals WHERE symbol = ? ORDER BY created_at DESC LIMIT ?",
            (symbol, limit),
        ).fetchall()
        return [dict(row) for row in rows]
