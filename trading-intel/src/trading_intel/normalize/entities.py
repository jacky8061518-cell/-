"""實體解析：把「台積」「TSMC」「2330」「台積電」對到同一個 entity_id。

SPEC 3.2 第 3 點的關鍵一句是「模糊比對閾值以外的丟人工佇列，不要猜」。

猜錯的代價不對稱：把一則利多新聞掛到錯的公司，會產生一個有證據支持、
看起來完全合理、但完全錯誤的訊號。而漏掉一則新聞只是少一個訊號。
所以本模組寧可送人工，不做低信心的自動配對。
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from enum import StrEnum
from typing import Final

from trading_intel.core.clock import ensure_utc
from trading_intel.core.enums import Market
from trading_intel.core.ids import EntityId, make_entity_id

#: 自動採用的最低相似度。低於此值一律進人工佇列。
AUTO_ACCEPT_THRESHOLD: Final = 0.92

#: 低於此值連候選都不列出，視為完全無關。
CANDIDATE_THRESHOLD: Final = 0.60

#: 台股常見的公司名稱後綴，比對前先剝除。
_COMPANY_SUFFIXES: Final = (
    "股份有限公司",
    "控股股份有限公司",
    "有限公司",
    "公司",
    "集團",
)

_TICKER_PATTERN: Final = re.compile(r"^(\d{4,6})(?:\.(TW|TWO))?$", re.IGNORECASE)


class MatchMethod(StrEnum):
    """配對是怎麼來的。稽核時要能分辨「查表」與「猜的」。"""

    EXACT_CODE = "EXACT_CODE"
    EXACT_ALIAS = "EXACT_ALIAS"
    FUZZY = "FUZZY"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class ResolutionCandidate:
    entity_id: EntityId
    matched_text: str
    score: float


@dataclass(frozen=True)
class Resolution:
    """一次解析的結果，含它為什麼是這個答案。"""

    query: str
    entity_id: EntityId | None
    method: MatchMethod
    score: float
    candidates: tuple[ResolutionCandidate, ...] = ()

    @property
    def resolved(self) -> bool:
        return self.entity_id is not None

    @property
    def needs_review(self) -> bool:
        """需要人工判斷：有候選但信心不足。"""
        return self.entity_id is None and bool(self.candidates)


@dataclass(frozen=True)
class ReviewItem:
    """人工佇列的一筆待審項目。"""

    query: str
    candidates: tuple[ResolutionCandidate, ...]
    queued_at: datetime
    context: str = ""


def normalize_text(value: str) -> str:
    """正規化：全形轉半形、去空白與標點、剝除公司後綴、轉大寫。

    這一步負責吸收書寫差異，讓相似度只反映「名字本身」的差異。
    """
    text = unicodedata.normalize("NFKC", value).strip().upper()
    for suffix in _COMPANY_SUFFIXES:
        text = text.replace(suffix.upper(), "")
    return re.sub(r"[\s\-_.,()（）「」【】·・]", "", text)


def similarity(left: str, right: str) -> float:
    """兩個已正規化字串的相似度，落在 0 到 1 之間。"""
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    return SequenceMatcher(None, left, right).ratio()


@dataclass
class EntityResolver:
    """別名表加模糊比對的解析器。

    別名可以隨時新增（例如從新聞裡學到的簡稱經人工確認後回填），
    但**自動學習是刻意不做的**：讓系統自己把模糊配對變成別名，
    等於把一次猜測固化成永久的事實。
    """

    market: Market
    _by_code: dict[str, EntityId] = field(default_factory=dict)
    _aliases: dict[str, EntityId] = field(default_factory=dict)
    _alias_display: dict[str, str] = field(default_factory=dict)
    _review_queue: list[ReviewItem] = field(default_factory=list)

    def register(
        self,
        local_symbol: str,
        names: Iterable[str] = (),
    ) -> EntityId:
        """登錄一個標的及其所有已知名稱。"""
        entity_id = make_entity_id(self.market, local_symbol)
        self._by_code[normalize_text(local_symbol)] = entity_id
        for name in names:
            key = normalize_text(name)
            if key:
                self._aliases[key] = entity_id
                self._alias_display[key] = name
        return entity_id

    def add_alias(self, alias: str, entity_id: EntityId) -> None:
        """新增一個別名。用於人工審核通過後的回填。"""
        key = normalize_text(alias)
        if not key:
            msg = "別名正規化後為空字串"
            raise ValueError(msg)
        self._aliases[key] = entity_id
        self._alias_display[key] = alias

    def resolve(self, query: str, *, now: datetime, context: str = "") -> Resolution:
        """解析一段文字。信心不足時回傳未解析並排入人工佇列。"""
        raw = query.strip()
        normalized = normalize_text(raw)
        if not normalized:
            return Resolution(query=raw, entity_id=None, method=MatchMethod.UNRESOLVED, score=0.0)

        # 1. 代號直接命中。"2330"、"2330.TW" 都算。
        ticker_match = _TICKER_PATTERN.match(raw.strip())
        code = ticker_match.group(1) if ticker_match else normalized
        if code in self._by_code:
            return Resolution(
                query=raw,
                entity_id=self._by_code[code],
                method=MatchMethod.EXACT_CODE,
                score=1.0,
            )

        # 2. 別名精確命中。
        if normalized in self._aliases:
            return Resolution(
                query=raw,
                entity_id=self._aliases[normalized],
                method=MatchMethod.EXACT_ALIAS,
                score=1.0,
            )

        # 3. 模糊比對。
        scored = [
            ResolutionCandidate(
                entity_id=entity_id,
                matched_text=self._alias_display.get(alias, alias),
                score=similarity(normalized, alias),
            )
            for alias, entity_id in self._aliases.items()
        ]
        candidates = tuple(
            sorted(
                (item for item in scored if item.score >= CANDIDATE_THRESHOLD),
                key=lambda item: item.score,
                reverse=True,
            )[:5]
        )
        if candidates and candidates[0].score >= AUTO_ACCEPT_THRESHOLD:
            # 若前兩名分數過於接近，代表這是個歧義，不該自動決定。
            if len(candidates) > 1 and candidates[0].score - candidates[1].score < 0.02:
                self._queue(raw, candidates, now, context)
                return Resolution(
                    query=raw,
                    entity_id=None,
                    method=MatchMethod.UNRESOLVED,
                    score=candidates[0].score,
                    candidates=candidates,
                )
            return Resolution(
                query=raw,
                entity_id=candidates[0].entity_id,
                method=MatchMethod.FUZZY,
                score=candidates[0].score,
                candidates=candidates,
            )

        if candidates:
            self._queue(raw, candidates, now, context)
        return Resolution(
            query=raw,
            entity_id=None,
            method=MatchMethod.UNRESOLVED,
            score=candidates[0].score if candidates else 0.0,
            candidates=candidates,
        )

    def _queue(
        self,
        query: str,
        candidates: tuple[ResolutionCandidate, ...],
        now: datetime,
        context: str,
    ) -> None:
        self._review_queue.append(
            ReviewItem(
                query=query,
                candidates=candidates,
                queued_at=ensure_utc(now),
                context=context,
            )
        )

    @property
    def review_queue(self) -> tuple[ReviewItem, ...]:
        return tuple(self._review_queue)

    def clear_review_queue(self) -> None:
        self._review_queue.clear()
