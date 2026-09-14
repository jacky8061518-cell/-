"""TWSE 與 TPEx 的抓取與解析。

**抓取與解析刻意分離**：``fetch_*`` 只負責把 bytes 拿回來並落地，
``parse_*`` 是吃 bytes 的純函式。這樣做有三個好處：

1. 解析器可以用真實格式的樣本完整測試，不需要網路；
2. 解析器改版後，可以拿 raw landing zone 裡的舊 bytes 重跑；
3. 回測期間只會用到 ``parse_*``，而它碰不到網路（CLAUDE.md 第 3 條）。

欄位索引與日期格式的知識沿用自既有 sector_rotation 專案，見
``docs/PHASE1-REUSE.md`` 的 A 級清單。這些是踩過坑才有的東西，逐字保留。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any, Final
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from trading_intel.core.clock import utc_now
from trading_intel.core.enums import Market
from trading_intel.core.errors import SchemaValidationError

TWSE_COMPANY_API: Final = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
TPEX_COMPANY_API: Final = "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O"
TWSE_INSTITUTIONAL_URL: Final = "https://www.twse.com.tw/rwd/zh/fund/T86"
TPEX_INSTITUTIONAL_URL: Final = (
    "https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge_result.php"
)
#: 除權除息計算結果表。Phase 1 新增的資料源，用於企業行動因子。
TWSE_DIVIDEND_API: Final = "https://openapi.twse.com.tw/v1/exchangeReport/TWT49U"

_USER_AGENT: Final = "Mozilla/5.0 trading-intel/0.1"
_TIMEOUT_SECONDS: Final = 30

# 三大法人的欄位索引。上市與上櫃完全不同，這是最容易出錯的地方。
_TWSE_FLOW_COLUMNS: Final = {"foreign": 4, "trust": 10, "dealer": 11, "total": 18}
_TWSE_FLOW_MIN_WIDTH: Final = 19
_TPEX_FLOW_COLUMNS: Final = {"foreign": 4, "trust": 13, "dealer": 22, "total": 23}
_TPEX_FLOW_MIN_WIDTH: Final = 24


@dataclass(frozen=True)
class InstitutionalFlow:
    """一檔標的一個交易日的三大法人買賣超（張數為單位的原始股數）。"""

    trade_date: date
    local_symbol: str
    market: Market
    name: str
    foreign_net_shares: Decimal
    trust_net_shares: Decimal
    dealer_net_shares: Decimal
    total_net_shares: Decimal


@dataclass(frozen=True)
class DividendEvent:
    """一筆除權除息公告。企業行動因子的官方來源。"""

    ex_date: date
    local_symbol: str
    name: str
    #: 除權息前一日收盤價。
    prior_close: Decimal
    #: 除權息參考價。兩者相除即為調整因子。
    reference_price: Decimal
    cash_dividend: Decimal
    stock_dividend: Decimal


def _to_decimal(value: object) -> Decimal:
    """把 TWSE 回傳的字串轉成 Decimal。

    來源會出現 ``"1,234"``、``"--"``、``"---"``、``""``。
    這些都代表「沒有數字」而非零以外的值，一律轉 0。
    """
    if value is None:
        return Decimal("0")
    text = str(value).replace(",", "").strip()
    if text in {"", "--", "---", "N/A"}:
        return Decimal("0")
    try:
        return Decimal(text)
    except InvalidOperation as exc:
        raise SchemaValidationError("無法轉換為數值", value=repr(value)) from exc


def _roc_date(session: date) -> str:
    """民國年日期格式，TPEx 專用。例如 2024-03-19 → ``113/03/19``。"""
    return f"{session.year - 1911:03d}/{session:%m/%d}"


def parse_roc_date(text: str) -> date:
    """把民國年字串轉回西元日期。接受 ``113/03/19`` 與 ``1130319``。"""
    cleaned = text.strip().replace("/", "")
    if len(cleaned) != 7 or not cleaned.isdigit():
        raise SchemaValidationError("非預期的民國年日期格式", value=text)
    year = int(cleaned[:3]) + 1911
    return date(year, int(cleaned[3:5]), int(cleaned[5:7]))


def _yahoo_suffix(market: Market) -> str:
    return ".TW" if market is Market.TW else ".TWO"


# --- 解析：純函式，吃 bytes ------------------------------------------------


def parse_twse_institutional_flows(payload: bytes, session: date) -> list[InstitutionalFlow]:
    """解析 TWSE T86 三大法人買賣超。"""
    document = json.loads(payload)
    if document.get("stat") != "OK":
        return []
    flows: list[InstitutionalFlow] = []
    for row in document.get("data", []):
        if len(row) < _TWSE_FLOW_MIN_WIDTH:
            continue
        flows.append(
            InstitutionalFlow(
                trade_date=session,
                local_symbol=str(row[0]).strip(),
                market=Market.TW,
                name=str(row[1]).strip(),
                foreign_net_shares=_to_decimal(row[_TWSE_FLOW_COLUMNS["foreign"]]),
                trust_net_shares=_to_decimal(row[_TWSE_FLOW_COLUMNS["trust"]]),
                dealer_net_shares=_to_decimal(row[_TWSE_FLOW_COLUMNS["dealer"]]),
                total_net_shares=_to_decimal(row[_TWSE_FLOW_COLUMNS["total"]]),
            )
        )
    return flows


def parse_tpex_institutional_flows(payload: bytes, session: date) -> list[InstitutionalFlow]:
    """解析 TPEx 三大法人買賣超。欄位索引與上市完全不同。"""
    document = json.loads(payload)
    if document.get("stat") != "ok" or not document.get("tables"):
        return []
    tables = [table for table in document["tables"] if table.get("data")]
    if not tables:
        return []
    flows: list[InstitutionalFlow] = []
    for row in tables[0]["data"]:
        if len(row) < _TPEX_FLOW_MIN_WIDTH:
            continue
        flows.append(
            InstitutionalFlow(
                trade_date=session,
                local_symbol=str(row[0]).strip(),
                market=Market.TW,
                name=str(row[1]).strip(),
                foreign_net_shares=_to_decimal(row[_TPEX_FLOW_COLUMNS["foreign"]]),
                trust_net_shares=_to_decimal(row[_TPEX_FLOW_COLUMNS["trust"]]),
                dealer_net_shares=_to_decimal(row[_TPEX_FLOW_COLUMNS["dealer"]]),
                total_net_shares=_to_decimal(row[_TPEX_FLOW_COLUMNS["total"]]),
            )
        )
    return flows


def parse_twse_dividends(payload: bytes) -> list[DividendEvent]:
    """解析 TWSE 除權除息計算結果表。

    這是 Phase 1 新增的外部資料源，用來取代「由價格跳動反推因子」的猜測做法。
    """
    document = json.loads(payload)
    events: list[DividendEvent] = []
    for row in document:
        if not isinstance(row, dict):
            continue
        raw_date = str(row.get("Date", "")).strip()
        symbol = str(row.get("Code", "")).strip()
        if not raw_date or not symbol:
            continue
        prior = _to_decimal(row.get("PreviousClosingPrice"))
        reference = _to_decimal(row.get("ExRightsExDividendReferencePrice"))
        if prior <= 0 or reference <= 0:
            continue
        events.append(
            DividendEvent(
                ex_date=parse_roc_date(raw_date),
                local_symbol=symbol,
                name=str(row.get("Name", "")).strip(),
                prior_close=prior,
                reference_price=reference,
                cash_dividend=_to_decimal(row.get("CashDividend")),
                stock_dividend=_to_decimal(row.get("StockDividend")),
            )
        )
    return events


def dividend_adjustment_factor(event: DividendEvent) -> Decimal:
    """由除權息參考價推導調整因子。

    因子 = 參考價 / 前一日收盤價。除息前的歷史價格乘上它，
    就與除息後的價格在同一個尺度上。這是交易所公告的事實，不是估計。
    """
    if event.prior_close <= 0:
        raise SchemaValidationError(
            "前一日收盤價必須為正數",
            symbol=event.local_symbol,
            prior_close=str(event.prior_close),
        )
    return event.reference_price / event.prior_close


def yahoo_ticker(local_symbol: str, market_suffix: str) -> str:
    """組出 Yahoo Finance 代號。既有價格快取以此為欄位名。"""
    return f"{local_symbol.strip()}{market_suffix}"


# --- 抓取：會碰網路，回測期間絕不呼叫 --------------------------------------


def _http_get(url: str, params: dict[str, str] | None = None) -> bytes:
    """發一次 GET 並回傳原始 bytes。

    刻意回傳 bytes 而非解析後的物件：呼叫端要先把它落地到 raw landing zone，
    再交給解析器。原始 payload 永不覆寫（SPEC 3.2）。
    """
    target = f"{url}?{urlencode(params)}" if params else url
    request = Request(target, headers={"User-Agent": _USER_AGENT})
    with urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
        payload: bytes = response.read()
    return payload


def fetch_twse_institutional_flows(session: date) -> tuple[bytes, dict[str, Any]]:
    """抓一個交易日的 TWSE 三大法人。回傳 (原始 bytes, 請求細節)。"""
    params = {
        "date": session.strftime("%Y%m%d"),
        "selectType": "ALLBUT0999",
        "response": "json",
    }
    return _http_get(TWSE_INSTITUTIONAL_URL, params), {
        "url": TWSE_INSTITUTIONAL_URL,
        "params": params,
        "fetched_at": utc_now().isoformat(),
    }


def fetch_tpex_institutional_flows(session: date) -> tuple[bytes, dict[str, Any]]:
    """抓一個交易日的 TPEx 三大法人。日期須轉為民國年。"""
    params = {"l": "zh-tw", "o": "json", "se": "EW", "t": "D", "d": _roc_date(session)}
    return _http_get(TPEX_INSTITUTIONAL_URL, params), {
        "url": TPEX_INSTITUTIONAL_URL,
        "params": params,
        "fetched_at": utc_now().isoformat(),
    }


def fetch_twse_dividends(session: date) -> tuple[bytes, dict[str, Any]]:
    """抓除權除息計算結果表。"""
    params = {"response": "json", "date": session.strftime("%Y%m%d")}
    return _http_get(TWSE_DIVIDEND_API, params), {
        "url": TWSE_DIVIDEND_API,
        "params": params,
        "fetched_at": utc_now().isoformat(),
    }


def fetch_company_master(market: Market) -> tuple[bytes, dict[str, Any]]:
    """抓公司主檔。"""
    url = TWSE_COMPANY_API if market is Market.TW else TPEX_COMPANY_API
    return _http_get(url), {"url": url, "fetched_at": utc_now().isoformat()}
