"""Yahoo Finance 抓取與解析。

**這是網路權限開通後新增的資料源**（見 ADR 0007）。台灣證交所的 openapi 與
www 端點會被 TWSE 自己的 WAF 攔截（回應 HTTP 200，但內容是資安警示頁，
不是資料）——這與我們的網路政策無關，是 TWSE 對雲端機房 IP 的封鎖，
無法從程式碼層繞過，也不應該嘗試繞過。Yahoo Finance 與櫃買中心（TPEx）
不受此限制，兩者可正常存取。

因此本模組補上 Phase 1／Phase 2 原本因為連不上證交所而卡住的兩件事：

1. 可以拿到即時／近期收盤價，用來驗證 Phase 2 的基準策略是否落在合理範圍；
2. 未來若要重建無存活者偏誤的宇宙，Yahoo 的 ``quoteSummary`` 也能查到
   個股是否仍在交易，是 TWSE 上市公告之外的替代線索（但不是官方紀錄，
   信心層級低於 OFFICIAL，見 ``normalize.corporate_actions.ActionSource``）。

抓取與解析一樣刻意分離，理由與 ``ingestion/twse.py`` 相同。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any, Final
from urllib.request import Request, urlopen

from trading_intel.core.errors import SchemaValidationError

CHART_API: Final = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
_USER_AGENT: Final = "Mozilla/5.0 trading-intel/0.1"
_TIMEOUT_SECONDS: Final = 20


@dataclass(frozen=True)
class YahooBar:
    """一根日 K，來自 Yahoo Finance 的調整後與未調整收盤價皆保留。

    ``close`` 為未調整；``adjclose`` 為調整後。兩者都存，下游依 SPEC 3.2
    的原則自行選擇要不要信任 Yahoo 的調整方式，而不是被迫只能用一種。
    """

    ticker: str
    trade_date: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    adjclose: Decimal
    volume: int


def fetch_chart(
    ticker: str, *, range_: str = "1y", interval: str = "1d"
) -> tuple[bytes, dict[str, Any]]:
    """抓一檔標的的日線圖表資料。回傳 (原始 bytes, 請求細節)。

    ``ticker`` 直接沿用既有 sector_rotation 專案的 Yahoo 代號慣例
    （例如 ``2330.TW``、``6488.TWO``），不必重新對照。
    """
    from trading_intel.core.clock import utc_now

    url = CHART_API.format(ticker=ticker)
    params = {"range": range_, "interval": interval}
    query = "&".join(f"{key}={value}" for key, value in params.items())
    request = Request(f"{url}?{query}", headers={"User-Agent": _USER_AGENT})
    with urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
        payload: bytes = response.read()
    return payload, {"url": url, "params": params, "fetched_at": utc_now().isoformat()}


def parse_chart(payload: bytes, ticker: str) -> list[YahooBar]:
    """解析 Yahoo chart API 的回應。純函式，不碰網路。

    Yahoo 的回應把每個欄位存成平行陣列（timestamps、opens、highs...），
    而非逐筆物件，這是這支 API 特有的形狀，也是最容易解析錯的地方：
    索引對不齊會讓每一筆資料的 OHLC 悄悄錯位。
    """
    document = json.loads(payload)
    chart = document.get("chart", {})
    if chart.get("error"):
        raise SchemaValidationError(
            "Yahoo Finance 回傳錯誤", ticker=ticker, detail=str(chart["error"])
        )
    results = chart.get("result")
    if not results:
        return []

    result = results[0]
    timestamps = result.get("timestamp", [])
    quote = result.get("indicators", {}).get("quote", [{}])[0]
    adjclose_series = result.get("indicators", {}).get("adjclose", [{}])
    adjcloses = adjclose_series[0].get("adjclose", []) if adjclose_series else []

    opens = quote.get("open", [])
    highs = quote.get("high", [])
    lows = quote.get("low", [])
    closes = quote.get("close", [])
    volumes = quote.get("volume", [])

    bars: list[YahooBar] = []
    for index, timestamp in enumerate(timestamps):
        close = _safe_decimal(closes, index)
        open_ = _safe_decimal(opens, index)
        high = _safe_decimal(highs, index)
        low = _safe_decimal(lows, index)
        adjclose = _safe_decimal(adjcloses, index) if adjcloses else close
        volume = volumes[index] if index < len(volumes) and volumes[index] is not None else 0
        # 停牌日或資料缺漏時，Yahoo 回傳 null；這種天沒有價格可用，整筆跳過
        # 而不是塞一個假值進去（CLAUDE.md 第 6 條：資料壞掉時停止，不是填補）。
        if close is None or open_ is None or high is None or low is None:
            continue
        trade_date = datetime.fromtimestamp(timestamp, tz=UTC).date()
        bars.append(
            YahooBar(
                ticker=ticker,
                trade_date=trade_date,
                open=open_,
                high=high,
                low=low,
                close=close,
                adjclose=adjclose if adjclose is not None else close,
                volume=int(volume),
            )
        )
    return bars


def _safe_decimal(series: list[float | None], index: int) -> Decimal | None:
    if index >= len(series):
        return None
    value = series[index]
    if value is None:
        return None
    # 先轉字串再進 Decimal，避免把 float 的二進位雜訊帶進來（Phase 0 決定）。
    return Decimal(str(round(value, 6)))


def latest_price(ticker: str) -> tuple[Decimal, date] | None:
    """抓某標的最新一筆收盤價，供快速驗證用。"""
    payload, _request = fetch_chart(ticker, range_="5d", interval="1d")
    bars = parse_chart(payload, ticker)
    if not bars:
        return None
    latest = max(bars, key=lambda bar: bar.trade_date)
    return latest.close, latest.trade_date
