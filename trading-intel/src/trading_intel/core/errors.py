"""trading-intel 的例外階層。

每個例外都帶 ``context`` 對應表，讓結構化日誌能以機器可讀的欄位記錄失敗，
而不是只丟一段格式化過的字串。
"""

from __future__ import annotations

from typing import Any


class TradingIntelError(Exception):
    """本專案所有例外的基底類別。"""

    def __init__(self, message: str = "", /, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = dict(context)

    def __str__(self) -> str:
        if not self.context:
            return self.message
        rendered = ", ".join(f"{k}={v!r}" for k, v in sorted(self.context.items()))
        return f"{self.message} ({rendered})" if self.message else rendered

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.message!r}, context={self.context!r})"


class ConfigError(TradingIntelError):
    """設定缺漏、格式錯誤，或內部不一致。"""


class ClockError(TradingIntelError):
    """時鐘相關失敗的基底類別。"""


class NaiveDatetimeError(ClockError):
    """不帶 tzinfo 的 datetime 進入了要求 aware 值的程式碼。"""


class ClockRewindError(ClockError):
    """試圖把模擬時鐘往回撥。"""


class TemporalIntegrityError(TradingIntelError):
    """時間戳違反必要的先後順序，例如 ingest_time 早於 event_time。"""


class LookaheadError(TradingIntelError):
    """讀到了 asof 界線之後的資料，即前視偏誤。"""


class NetworkAccessDenied(TradingIntelError):  # noqa: N818  (名稱由 SPEC 指定)
    """在禁網沙箱內嘗試存取網路。"""


class DataQualityError(TradingIntelError):
    """資料品質檢查失敗，嚴重到必須停止處理。"""


class SchemaValidationError(TradingIntelError):
    """Payload 不符合其宣告的 schema。"""


class BudgetExceededError(TradingIntelError):
    """token、呼叫速率或時間預算已耗盡。"""
