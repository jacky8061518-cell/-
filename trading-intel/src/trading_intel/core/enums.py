"""全系統共用的封閉詞彙表。

值與名稱相同，讓序列化、資料庫存放與日誌輸出三者讀起來一致。
"""

from __future__ import annotations

from enum import IntEnum, StrEnum


class Market(StrEnum):
    TW = "TW"
    US = "US"


class TradingState(StrEnum):
    TRADABLE = "TRADABLE"
    NO_TRADE = "NO_TRADE"
    HALTED = "HALTED"
    RESTRICTED = "RESTRICTED"


class Direction(StrEnum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class Horizon(StrEnum):
    INTRADAY = "INTRADAY"
    DAYS = "DAYS"
    WEEKS = "WEEKS"
    MONTHS = "MONTHS"


class Severity(StrEnum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"


class DecisionLevel(IntEnum):
    """刻意用 IntEnum：降級判斷需要比大小，寫成 max(...) 才自然。

    對應 SPEC 7.4 的人機分權三級。
    """

    AUTO = 1
    CONFIRM = 2
    RESEARCH_ONLY = 3


class QualityCheck(StrEnum):
    """SPEC 3.3 的五類資料品質閘門。"""

    FRESHNESS = "FRESHNESS"
    COMPLETENESS = "COMPLETENESS"
    SANITY = "SANITY"
    CONSISTENCY = "CONSISTENCY"
    DRIFT = "DRIFT"


class DocType(StrEnum):
    """NewsAnalystAgent 以此區分抽取提示，但共用同一份輸出 schema（SPEC 4.2）。"""

    NEWS = "NEWS"
    ANNOUNCEMENT = "ANNOUNCEMENT"
    FINANCIAL_REPORT = "FINANCIAL_REPORT"
    EARNINGS_CALL = "EARNINGS_CALL"
    BROKER_REPORT = "BROKER_REPORT"
