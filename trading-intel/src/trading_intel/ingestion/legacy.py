"""既有 sector_rotation 資料的匯入：唯讀歷史種子。

背景：``data/databases/tw/`` 有 2012-01-02 起、3560 個交易日、2333 檔標的的
價格歷史，以及 47317 列三大法人資料。重抓要很久，而且更早的歷史未必拿得回來。

問題：那份價格是**已調整價**，違反 SPEC 3.2「存未調整原始價 + 獨立因子表」。

處置（經核定）：把它當唯讀歷史種子保留，但**明確標記為 LEGACY**，
在資料上帶著「這段歷史無法重建原始未調整價」這個事實，而不是假裝它乾淨。

代價要說清楚：2012–2026 這段歷史的企業行動無法重新推導，回測跨越這段期間時，
調整假設是繼承來的、不是自己算的。因此由本模組產生的每一筆價格，其
``CorporateAction`` 來源都是 ``ActionSource.LEGACY``，信心值低於 1.0，
下游可以據此決定要不要採用。
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Final

from trading_intel.core.enums import Market, TradingState
from trading_intel.core.errors import ConfigError
from trading_intel.core.ids import EntityId, make_entity_id
from trading_intel.normalize.corporate_actions import RawPrice
from trading_intel.normalize.universe import MembershipSpan

#: 標記來源，讓下游一眼看出這批資料的性質。
LEGACY_SOURCE: Final = "legacy-adjusted"

#: 既有資料的信心值。刻意不給 1.0：它是已調整價，成因不可考。
LEGACY_CONFIDENCE: Final = 0.8

_TW_SUFFIX: Final = ".TW"
_TPEX_SUFFIX: Final = ".TWO"


def _entity_from_yahoo_ticker(ticker: str) -> EntityId | None:
    """``2330.TW`` → ``TW:2330``。非台股代號回傳 None。"""
    text = ticker.strip().upper()
    for suffix in (_TPEX_SUFFIX, _TW_SUFFIX):
        if text.endswith(suffix):
            symbol = text[: -len(suffix)]
            return make_entity_id(Market.TW, symbol) if symbol else None
    return None


def legacy_ingest_time(path: Path) -> datetime:
    """以檔案的 mtime 作為這批資料的 ``ingest_time``。

    這是我們對「何時得知」能取得的最佳近似。它不精確，但它保守：
    整批資料被視為在同一時刻送達，不會讓任何一筆看起來比實際更早可用。
    """
    if not path.exists():
        raise ConfigError("找不到既有資料檔", path=str(path))
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)


def iter_legacy_prices(
    parquet_path: Path,
    *,
    entity_ids: frozenset[EntityId] | None = None,
    start: date | None = None,
    end: date | None = None,
) -> Iterator[RawPrice]:
    """逐筆產出既有價格。

    以 ``Decimal`` 承接，因為下游一律用 Decimal（Phase 0 決定）。
    來源是 float，轉換時先經字串以避免帶進二進位浮點的雜訊。

    注意：這裡產出的 ``close`` 欄位裝的是**已調整價**。欄位名叫 close 而非
    adjusted_close，是因為對本系統而言它就是「我們手上僅有的那個價格」；
    其 LEGACY 性質由 ``docs/ADR/0005`` 與本模組的常數負責標示。
    """
    import pandas as pd

    ingest_time = legacy_ingest_time(parquet_path)
    frame = pd.read_parquet(parquet_path)
    frame.index = pd.to_datetime(frame.index)

    for ticker in frame.columns:
        entity_id = _entity_from_yahoo_ticker(str(ticker))
        if entity_id is None:
            continue
        if entity_ids is not None and entity_id not in entity_ids:
            continue
        series = frame[ticker].dropna()
        # pandas-stubs 把 index 標成 Index[Any]，此處已知為 DatetimeIndex。
        timestamps: list[date] = [
            item.date() for item in pd.DatetimeIndex(series.index).to_pydatetime()
        ]
        for trade_date, value in zip(timestamps, series.to_numpy(), strict=True):
            if start is not None and trade_date < start:
                continue
            if end is not None and trade_date > end:
                continue
            close = Decimal(str(round(float(value), 6)))
            if close <= 0:
                continue
            yield RawPrice(
                entity_id=entity_id,
                trade_date=trade_date,
                close=close,
                volume=0,  # 既有價格快取沒有成交量欄位。
                ingest_time=ingest_time,
            )


def load_security_master(
    csv_path: Path,
) -> list[dict[str, Any]]:
    """讀既有的證券主檔，回傳正規化後的紀錄。"""
    import pandas as pd

    if not csv_path.exists():
        raise ConfigError("找不到證券主檔", path=str(csv_path))
    frame = pd.read_csv(csv_path, dtype=str).fillna("")
    records: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        entity_id = _entity_from_yahoo_ticker(str(row.get("Yahoo ticker", "")))
        if entity_id is None:
            continue
        records.append(
            {
                "entity_id": entity_id,
                "local_symbol": str(row.get("Code", "")).strip(),
                "name": str(row.get("Name", "")).strip(),
                "full_name": str(row.get("Full name", "")).strip(),
                "industry": str(row.get("Industry", "")).strip(),
                "market_board": str(row.get("Market", "")).strip(),
                "asset_type": str(row.get("Asset type", "")).strip(),
            }
        )
    return records


@dataclass(frozen=True)
class SurvivorshipReport:
    """既有資料的存活者偏誤診斷。

    這份報告存在的理由：實測發現既有的 2333 檔標的**全部**的最後有值日期
    都等於資料結尾，代表這份資料只收錄了「今天還在市」的標的，
    下市股票從未被抓進來。用它選股回測，等於事先知道哪些公司不會倒。

    這個缺陷無法靠處理現有資料修好——資料裡就是沒有那些公司。
    要修只能接上交易所的上市下市公告，那是 Phase 1 之後的資料源。
    在那之前，本報告負責讓這個缺陷是**已知且被量化的**，而不是隱形的。
    """

    total_entities: int
    #: 序列在資料結尾前就停止的標的數。理想上應該是個可觀的數字。
    ended_before_dataset_end: int
    #: 尾端價格連續不變的標的數，是前值填補（ffill）留下的痕跡。
    constant_tail_entities: int
    dataset_end: date

    @property
    def has_survivorship_bias(self) -> bool:
        """沒有任何標的中途退出，即代表宇宙由倖存者組成。"""
        return self.ended_before_dataset_end == 0

    def summary(self) -> str:
        if self.has_survivorship_bias:
            return (
                f"存在存活者偏誤：{self.total_entities} 檔標的全部延續到 "
                f"{self.dataset_end}，資料中沒有任何下市標的。"
                "以此建構的宇宙不可用於選股回測。"
            )
        return (
            f"{self.ended_before_dataset_end}/{self.total_entities} 檔標的在 "
            f"{self.dataset_end} 之前退出宇宙。"
        )


def diagnose_survivorship(
    parquet_path: Path,
    *,
    constant_tail_days: int = 20,
) -> SurvivorshipReport:
    """量化既有資料的存活者偏誤程度。

    匯入既有資料前應該先跑這個，並把結果記錄下來。
    """
    import pandas as pd

    frame = pd.read_parquet(parquet_path)
    frame.index = pd.to_datetime(frame.index)
    dataset_end = pd.DatetimeIndex(frame.index).max().date()

    total = 0
    ended = 0
    constant_tail = 0
    tail = frame.tail(constant_tail_days)
    for ticker in frame.columns:
        if _entity_from_yahoo_ticker(str(ticker)) is None:
            continue
        total += 1
        series = frame[ticker].dropna()
        if series.empty:
            continue
        if pd.Timestamp(series.index.max()).date() < dataset_end:
            ended += 1
        tail_series = tail[ticker]
        if tail_series.notna().all() and bool((tail_series == tail_series.iloc[0]).all()):
            constant_tail += 1

    return SurvivorshipReport(
        total_entities=total,
        ended_before_dataset_end=ended,
        constant_tail_entities=constant_tail,
        dataset_end=dataset_end,
    )


def build_membership_from_prices(
    parquet_path: Path,
    *,
    entity_ids: frozenset[EntityId] | None = None,
) -> list[MembershipSpan]:
    """由既有價格序列反推成員資格區間。

    做法：某標的有價格的第一天到最後一天，視為它在宇宙內的期間。
    最後一天若明顯早於整份資料的結尾，代表它中途消失——下市、暫停、或改代號。

    **這是推論，不是事實**，而且在既有資料上這個推論幾乎沒有作用：
    實測顯示 2333 檔標的的序列全部延續到資料結尾（見 ``diagnose_survivorship``），
    因為那份資料只收錄了現存標的。因此本函式產生的區間絕大多數 ``valid_to``
    為 None，宇宙看起來像是從來沒有人下市過。

    保留這個函式是因為它對**未來**正確抓取的資料是對的做法；
    對既有資料，呼叫端必須先看 ``diagnose_survivorship`` 的結果，
    知道自己拿到的是什麼。
    """
    import pandas as pd

    ingest_time = legacy_ingest_time(parquet_path)
    frame = pd.read_parquet(parquet_path)
    frame.index = pd.to_datetime(frame.index)
    dataset_end = frame.index.max().date()

    spans: list[MembershipSpan] = []
    for ticker in frame.columns:
        entity_id = _entity_from_yahoo_ticker(str(ticker))
        if entity_id is None:
            continue
        if entity_ids is not None and entity_id not in entity_ids:
            continue
        series = frame[ticker].dropna()
        if series.empty:
            continue
        first = series.index.min().date()
        last = series.index.max().date()
        # 資料在結尾前就斷掉，視為已退出宇宙。
        still_listed = last >= dataset_end
        spans.append(
            MembershipSpan(
                entity_id=entity_id,
                market=Market.TW,
                valid_from=first,
                valid_to=None if still_listed else last,
                state=TradingState.TRADABLE,
                ingest_time=ingest_time,
                reason=f"由既有價格序列推得（{LEGACY_SOURCE}）",
            )
        )
    return spans
