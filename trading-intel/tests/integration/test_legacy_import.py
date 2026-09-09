"""既有十四年資料的匯入：以真實檔案驗證，而非合成資料。

這一組測試讀 repo 根目錄的 ``data/databases/tw/``。檔案不在時整組跳過，
讓 CI 與離線開發不會因為資料檔缺席而失敗。
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from trading_intel.core.enums import Market, TradingState
from trading_intel.core.ids import make_entity_id
from trading_intel.ingestion.legacy import (
    LEGACY_SOURCE,
    build_membership_from_prices,
    diagnose_survivorship,
    iter_legacy_prices,
    legacy_ingest_time,
    load_security_master,
)
from trading_intel.normalize.universe import UniverseStore

DATA_ROOT = Path(__file__).resolve().parents[3] / "data" / "databases" / "tw"
PRICES = DATA_ROOT / "adjusted-prices.parquet"
MASTER = DATA_ROOT / "security-master.csv"

TSMC = make_entity_id(Market.TW, "2330")

pytestmark = pytest.mark.skipif(
    not PRICES.exists() or not MASTER.exists(),
    reason=f"既有資料檔不存在：{DATA_ROOT}",
)


def test_security_master_loads() -> None:
    records = load_security_master(MASTER)
    assert len(records) > 1000
    by_id = {record["entity_id"]: record for record in records}
    assert TSMC in by_id
    assert by_id[TSMC]["name"] == "台積電"


def test_prices_load_as_decimal_with_legacy_ingest_time() -> None:
    prices = list(
        iter_legacy_prices(
            PRICES,
            entity_ids=frozenset({TSMC}),
            start=date(2024, 1, 1),
            end=date(2024, 3, 31),
        )
    )
    assert len(prices) > 40
    assert all(isinstance(price.close, Decimal) for price in prices)
    assert all(price.entity_id == TSMC for price in prices)
    # ingest_time 統一取檔案 mtime：保守但誠實的近似。
    assert len({price.ingest_time for price in prices}) == 1
    assert prices[0].ingest_time == legacy_ingest_time(PRICES)


def test_legacy_prices_are_invisible_before_their_ingest_time() -> None:
    """CLAUDE.md 第 2 條：舊資料也受 asof 約束。"""
    ingest = legacy_ingest_time(PRICES)
    prices = list(
        iter_legacy_prices(
            PRICES, entity_ids=frozenset({TSMC}), start=date(2024, 1, 1), end=date(2024, 1, 31)
        )
    )
    before = [price for price in prices if price.ingest_time <= ingest.replace(year=2000)]
    assert before == []


def test_legacy_data_is_diagnosed_as_survivorship_biased() -> None:
    """把既有資料的缺陷變成有紀錄的事實，而不是隱形的。

    實測結果：2333 檔標的的序列全部延續到資料結尾，代表這份資料只收錄了
    「現在還在市」的標的。這個測試會在資料被換成無偏誤版本時失敗——
    那正是我們希望被通知的時刻。
    """
    report = diagnose_survivorship(PRICES)
    assert report.total_entities > 2000
    assert report.has_survivorship_bias, "資料似乎已修正，請更新此測試與 docs/ADR/0005 的記載"
    assert report.ended_before_dataset_end == 0
    assert "不可用於選股回測" in report.summary()


def test_membership_spans_are_built_but_carry_the_bias() -> None:
    """區間可以建出來，但它繼承了來源資料的偏誤，不會憑空修好。"""
    spans = build_membership_from_prices(PRICES)
    assert len(spans) > 1000
    assert all(span.reason.endswith(f"（{LEGACY_SOURCE}）") for span in spans)
    ended = [span for span in spans if span.valid_to is not None]
    assert ended == [], "來源資料沒有下市標的，因此不應憑空產生結束日"


def test_universe_can_be_rebuilt_for_any_past_date() -> None:
    """SPEC 第 11 節 Phase 1 驗收：給定任一歷史日期，可完整重建當日宇宙。"""
    store = UniverseStore(build_membership_from_prices(PRICES))
    # asof 必須晚於這批資料的 ingest_time，否則會被正確地整批濾掉。
    asof = legacy_ingest_time(PRICES) + timedelta(days=1)

    old = store.snapshot(date(2015, 6, 30), asof=asof)
    recent = store.snapshot(date(2025, 6, 30), asof=asof)

    assert len(old) > 500
    assert len(recent) > len(old), "宇宙應隨時間擴大（新上市的標的陸續進入）"
    assert all(state is TradingState.TRADABLE for state in old.states.values())
    # 機制本身是對的：早於某標的上市日的快照不會含有它。
    earliest = store.snapshot(date(2012, 1, 3), asof=asof)
    assert len(earliest) < len(old)


def test_no_entity_ever_leaves_the_legacy_universe() -> None:
    """記錄現實：這份資料裡沒有任何標的退出過宇宙。

    這是缺陷不是特性。接上交易所的上市下市公告之後，這個測試應該要失敗，
    屆時改成斷言「確實有標的退出」。
    """
    store = UniverseStore(build_membership_from_prices(PRICES))
    # asof 必須晚於這批資料的 ingest_time，否則會被正確地整批濾掉。
    asof = legacy_ingest_time(PRICES) + timedelta(days=1)
    old = store.tradable_on(date(2015, 6, 30), asof=asof)
    recent = store.tradable_on(date(2025, 6, 30), asof=asof)
    assert old - recent == frozenset()
    assert old < recent, "舊宇宙應為新宇宙的真子集"
