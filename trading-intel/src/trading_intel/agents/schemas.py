"""四個 LLM agent 的輸出 schema。

全部是 frozen 的 Pydantic 模型，欄位一律使用列舉而非自由文字——
超出列舉值的輸出會在驗證階段被拒收，這是提示注入防護的第三層（SPEC 4.3）。

**數值欄位刻意極少。** LLM 不碰數字（SPEC 1），因此「幅度」用三分類而非百分比，
「信心」是唯一的連續值，且它只用於加權與排序，不進入任何價格計算。
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from trading_intel.core.enums import Horizon

FROZEN = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class EventType(StrEnum):
    """事件分類法 v1，共 26 類。完整定義與推理見 docs/EVENT-TAXONOMY.md。

    每一類都必須對應一個可檢驗的價格反應假設——「公司有好消息」不是一類，
    因為它無法被回測。
    """

    # A. 財務績效
    EARNINGS_BEAT = "EARNINGS_BEAT"
    EARNINGS_MISS = "EARNINGS_MISS"
    GUIDANCE_RAISE = "GUIDANCE_RAISE"
    GUIDANCE_CUT = "GUIDANCE_CUT"
    REVENUE_MONTHLY = "REVENUE_MONTHLY"
    # B. 營運與產能
    CAPACITY_EXPANSION = "CAPACITY_EXPANSION"
    CAPACITY_CUT = "CAPACITY_CUT"
    MAJOR_ORDER = "MAJOR_ORDER"
    ORDER_LOSS = "ORDER_LOSS"
    # C. 公司行動
    MA_ACQUIRER = "MA_ACQUIRER"
    MA_TARGET = "MA_TARGET"
    CAPITAL_RAISE = "CAPITAL_RAISE"
    BUYBACK = "BUYBACK"
    DIVIDEND_CHANGE = "DIVIDEND_CHANGE"
    # D. 產品與技術
    PRODUCT_LAUNCH = "PRODUCT_LAUNCH"
    TECH_MILESTONE = "TECH_MILESTONE"
    PRODUCT_ISSUE = "PRODUCT_ISSUE"
    # E. 治理與法律
    MANAGEMENT_CHANGE = "MANAGEMENT_CHANGE"
    GOVERNANCE_ISSUE = "GOVERNANCE_ISSUE"
    LITIGATION = "LITIGATION"
    REGULATORY_ACTION = "REGULATORY_ACTION"
    # F. 外部環境
    MACRO_POLICY = "MACRO_POLICY"
    TRADE_POLICY = "TRADE_POLICY"
    INDUSTRY_CYCLE = "INDUSTRY_CYCLE"
    SUPPLY_CHAIN = "SUPPLY_CHAIN"
    PEER_READACROSS = "PEER_READACROSS"
    # G. 其他
    OTHER = "OTHER"


class SurpriseDirection(StrEnum):
    """相對市場預期的方向，不是消息本身的好壞。

    「獲利衰退但優於預期」是 POSITIVE，這個區別是事件驅動訊號的全部重點。
    """

    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"


class MagnitudeBucket(StrEnum):
    """影響幅度。刻意用三分類而非百分比——LLM 不得產生進入計算路徑的數值。"""

    MINOR = "MINOR"
    MODERATE = "MODERATE"
    MAJOR = "MAJOR"


class Sentiment(StrEnum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class NewsAnalysis(BaseModel):
    """NewsAnalystAgent 的輸出（SPEC 4.2）。

    新聞、公告、財報、法說會、券商報告共用這一份 schema，
    差別只在餵進去的抽取提示不同。
    """

    model_config = FROZEN

    entity_symbol: str = Field(min_length=1, max_length=20)
    event_type: EventType
    surprise_direction: SurpriseDirection
    magnitude_bucket: MagnitudeBucket
    horizon: Horizon
    sentiment: Sentiment
    #: 財報與法說會專用：抓語氣轉變與財測修正，而非摘要內容。
    delta_vs_prior: str = Field(default="", max_length=300)
    confidence: float = Field(ge=0, le=1)
    reasoning_digest: str = Field(max_length=200)


class Hypothesis(BaseModel):
    """HypothesisAgent 的輸出（SPEC 4.2）。

    **禁止輸出無法回測的假設。** 因此 falsification_condition 與
    required_features 都是必填且有最小長度——說不出「什麼情況代表我錯了」
    與「要用哪些特徵驗證」的假設，就不是假設，是感想。
    """

    model_config = FROZEN

    statement: str = Field(min_length=10, max_length=500)
    #: 可證偽條件。這是本 schema 存在的核心理由。
    falsification_condition: str = Field(min_length=10, max_length=300)
    expected_direction: SurpriseDirection
    expected_horizon: Horizon
    #: 驗證所需的特徵名稱。空清單會被拒收——無從驗證的假設不得產出。
    required_features: tuple[str, ...] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    reasoning_digest: str = Field(max_length=200)


class AttackVector(StrEnum):
    """RedTeamAgent 的八項固定檢查清單（SPEC 4.2）。

    固定清單而非自由發揮，理由是**涵蓋率可稽核**：
    八項都檢查過才算檢查完，而不是模型想到什麼講什麼。
    """

    DATA_LEAKAGE = "DATA_LEAKAGE"
    SURVIVORSHIP_BIAS = "SURVIVORSHIP_BIAS"
    MULTIPLE_TESTING = "MULTIPLE_TESTING"
    CAPACITY_LIMIT = "CAPACITY_LIMIT"
    CROWDING = "CROWDING"
    COST_EROSION = "COST_EROSION"
    SINGLE_REGIME = "SINGLE_REGIME"
    EVENT_DRIVEN_SAMPLE = "EVENT_DRIVEN_SAMPLE"


class AttackSeverity(StrEnum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class AttackFinding(BaseModel):
    """單一攻擊點。"""

    model_config = FROZEN

    vector: AttackVector
    severity: AttackSeverity
    detail: str = Field(max_length=300)


class RedTeamReport(BaseModel):
    """RedTeamAgent 的輸出。八項必須全部出現，不得只挑有問題的講。"""

    model_config = FROZEN

    findings: tuple[AttackFinding, ...] = Field(min_length=8, max_length=8)
    confidence: float = Field(ge=0, le=1)
    reasoning_digest: str = Field(max_length=200)

    @property
    def blocking(self) -> tuple[AttackFinding, ...]:
        """嚴重度足以擋下訊號的發現。"""
        return tuple(
            finding
            for finding in self.findings
            if finding.severity in {AttackSeverity.CRITICAL, AttackSeverity.HIGH}
        )


class SimilarityVerdict(StrEnum):
    """LibrarianAgent 的判重結論。"""

    DUPLICATE = "DUPLICATE"
    RELATED = "RELATED"
    NOVEL = "NOVEL"


class LibrarianVerdict(BaseModel):
    """LibrarianAgent 的輸出（SPEC 4.2）。

    LLM 在此**只做語意相似度判斷**：「這個假設是不是已經測過了」。
    測過幾次、deflated Sharpe 怎麼算，都是確定性程式碼的事。
    """

    model_config = FROZEN

    verdict: SimilarityVerdict
    #: 最相似的既有假設 id。NOVEL 時為空字串。
    closest_hypothesis_id: str = Field(default="", max_length=64)
    confidence: float = Field(ge=0, le=1)
    reasoning_digest: str = Field(max_length=200)
