"""四個 LLM agent 的角色定義與組裝（SPEC 4.2）。

每個 agent 都是 ``AgentRunner`` 加一段角色指示。刻意不繼承、不建立類別階層——
agent 之間唯一共用的是契約，而契約已經由 ``AgentRunner`` 表達了。

**這四個是全部。** SPEC 原本規劃的 regime 分類、損益歸因、衝突仲裁三項
改為純程式碼模組，不佔 LLM 預算。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel

from trading_intel.agents.base import AgentCall, AgentRunner, BudgetTracker, LLMClient
from trading_intel.agents.injection import scan_for_injection
from trading_intel.agents.schemas import (
    Hypothesis,
    LibrarianVerdict,
    NewsAnalysis,
    RedTeamReport,
)
from trading_intel.core.enums import DocType
from trading_intel.core.ids import EvidenceId
from trading_intel.core.settings import AgentBudget
from trading_intel.core.types import AgentOutput, Document

_ABSTAIN_RULE: Final = (
    "\n\n證據不足時，回傳 abstain 而非猜測。棄權是合法且被鼓勵的輸出——"
    "一個錯誤的判斷比沒有判斷昂貴得多。"
)

_JSON_RULE: Final = (
    "\n\n只輸出一個 JSON 物件，不要有其他文字、不要用 markdown 包裹。"
    "所有列舉欄位必須使用允許值之一，不得自創。"
)

#: 各文件型別的抽取重點。共用同一份輸出 schema，差別只在提示（SPEC 4.2）。
_DOC_TYPE_FOCUS: Final[dict[DocType, str]] = {
    DocType.NEWS: "判斷這則新聞相對市場既有預期是否帶來新資訊。轉載與重複報導不算新資訊。",
    DocType.ANNOUNCEMENT: "公告多為事實陳述。重點在它是否改變了公司的營運或財務軌跡。",
    DocType.FINANCIAL_REPORT: (
        "重點不是摘要數字，而是**相對前期與相對市場預期的變化**。"
        "特別注意毛利率走向、存貨與應收帳款的異常、以及財測的修正。"
    ),
    DocType.EARNINGS_CALL: (
        "重點是**語氣轉變**，不是內容摘要。比較管理層這次與上次對同一議題的說法："
        "從「審慎樂觀」變成「持續觀察」是重要訊號，即使財測數字沒變。"
    ),
    DocType.BROKER_REPORT: (
        "券商報告有立場。重點在評等與目標價的**變動方向**，以及變動的理由是否為新資訊。"
    ),
}

_NEWS_ROLE: Final = (
    "你是一位台股與美股的事件分析師。你的任務是把一份文件轉換成結構化的事件標記。\n\n"
    "重要原則：\n"
    "- `surprise_direction` 是相對**市場既有預期**的方向，不是消息本身的好壞。"
    "「獲利衰退但優於預期」是 POSITIVE。\n"
    "- `magnitude_bucket` 只有三級，不要試圖給出百分比或任何數值估計。\n"
    "- 你不負責判斷這則消息是否新穎，那由程式計算。"
)

_HYPOTHESIS_ROLE: Final = (
    "你是一位量化研究員。你會看到一份市場異常清單，任務是提出**可檢驗的**假設。\n\n"
    "硬性要求：\n"
    "- 每個假設必須附上可證偽條件：什麼情況出現，代表這個假設是錯的。\n"
    "- 必須列出驗證所需的特徵名稱。想不出要用什麼特徵驗證的假設，就不要提。\n"
    "- 禁止提出無法回測的假設（例如「市場情緒轉好」這種無法操作化的說法）。"
)

_REDTEAM_ROLE: Final = (
    "你是一位專門找碴的量化風控人員。你的任務**不是**判斷這個訊號好不好，"
    "而是假設它是錯的，然後找出它為什麼錯。\n\n"
    "你必須逐一檢視以下八項，每一項都要給出結論，不得只挑有問題的講：\n"
    "DATA_LEAKAGE 資料洩漏、SURVIVORSHIP_BIAS 生存者偏誤、"
    "MULTIPLE_TESTING 多重檢定、CAPACITY_LIMIT 容量限制、"
    "CROWDING 擁擠度、COST_EROSION 交易成本吞噬、"
    "SINGLE_REGIME 僅在單一市場狀態有效、"
    "EVENT_DRIVEN_SAMPLE 績效由樣本期少數特殊事件驅動。\n"
    "某一項確實不適用時，給 NOT_APPLICABLE 並說明理由。"
)

_LIBRARIAN_ROLE: Final = (
    "你是研究知識庫的管理員。你會看到一個新假設與一批既有假設，"
    "任務是判斷這個新假設是否已經被測試過。\n\n"
    "你只做語意相似度判斷。措辭不同但本質相同的假設是 DUPLICATE——"
    "「小型股在月初表現較好」與「月初的規模效應」是同一件事。\n"
    "這件事很重要：重複測試同一個假設會造成隱性的多重檢定，"
    "讓一個其實是運氣的結果看起來像是被反覆驗證。"
)


def _runner[T: BaseModel](
    *,
    name: str,
    payload_type: type[T],
    client: LLMClient,
    budget: AgentBudget,
    role: str,
    max_tokens: int = 2048,
) -> AgentRunner[T]:
    """組裝一個 agent。角色指示之後一律接上棄權規則與 JSON 輸出規則。"""
    return AgentRunner(
        agent_name=name,
        payload_type=payload_type,
        client=client,
        budget=BudgetTracker(budget=budget),
        role_instructions=role + _ABSTAIN_RULE + _JSON_RULE,
        max_tokens=max_tokens,
    )


@dataclass
class NewsAnalystAgent:
    """感知層唯一的 LLM agent，處理所有非結構化文本（SPEC 4.2）。"""

    client: LLMClient
    budget: AgentBudget

    def __post_init__(self) -> None:
        self._runner: AgentRunner[NewsAnalysis] = _runner(
            name="NewsAnalystAgent",
            payload_type=NewsAnalysis,
            client=self.client,
            budget=self.budget,
            role=_NEWS_ROLE,
        )

    @property
    def call_log(self) -> list[AgentCall]:
        return self._runner.call_log

    def analyze(self, document: Document) -> AgentOutput[NewsAnalysis]:
        """分析一份文件。

        文件內容一律經過注入掃描並包在資料標記內。掃描結果若判定可疑，
        **直接棄權而不呼叫模型**——既省錢，也讓可疑文本完全不進入模型。
        """
        scan = scan_for_injection(document.body)
        if scan.suspicious or document.suspicious:
            return self._runner.abstain(
                evidence_ids=(document.doc_id,),
                reason=f"文件含指令性語句，已標記為可疑：{'、'.join(scan.reasons)}",
            )

        focus = _DOC_TYPE_FOCUS.get(document.doc_type, "")
        instruction = (
            f"文件型別：{document.doc_type.value}\n"
            f"抽取重點：{focus}\n"
            f"來源可信度：{document.source_credibility:.2f}\n"
            f"標題：{document.title}"
        )
        return self._runner.run(
            user_content=instruction,
            evidence_ids=(document.doc_id,),
            untrusted_documents={document.source: document.body},
        )


@dataclass
class HypothesisAgent:
    """觀察 L2 的異常清單，提出可檢驗的假設（SPEC 4.2）。"""

    client: LLMClient
    budget: AgentBudget

    def __post_init__(self) -> None:
        self._runner: AgentRunner[Hypothesis] = _runner(
            name="HypothesisAgent",
            payload_type=Hypothesis,
            client=self.client,
            budget=self.budget,
            role=_HYPOTHESIS_ROLE,
        )

    @property
    def call_log(self) -> list[AgentCall]:
        return self._runner.call_log

    def propose(
        self,
        anomalies: Sequence[str],
        *,
        available_features: Sequence[str],
    ) -> AgentOutput[Hypothesis]:
        """由異常清單提出假設。

        ``available_features`` 是目前註冊表裡有的特徵。把它給模型，
        是為了讓「驗證所需特徵」落在實際存在的範圍內，而不是憑空捏造。
        """
        if not anomalies:
            return self._runner.abstain(reason="沒有異常可供分析")
        content = (
            "以下是本期觀察到的市場異常：\n"
            + "\n".join(f"- {item}" for item in anomalies)
            + "\n\n目前可用的特徵："
            + "、".join(available_features)
        )
        return self._runner.run(user_content=content)


@dataclass
class RedTeamAgent:
    """對每一個候選訊號主動找碴（SPEC 4.2）。"""

    client: LLMClient
    budget: AgentBudget

    def __post_init__(self) -> None:
        self._runner: AgentRunner[RedTeamReport] = _runner(
            name="RedTeamAgent",
            payload_type=RedTeamReport,
            client=self.client,
            budget=self.budget,
            role=_REDTEAM_ROLE,
            max_tokens=4096,
        )

    @property
    def call_log(self) -> list[AgentCall]:
        return self._runner.call_log

    def attack(
        self,
        signal_description: str,
        *,
        backtest_summary: Mapping[str, str],
        evidence_ids: Sequence[EvidenceId] = (),
    ) -> AgentOutput[RedTeamReport]:
        """攻擊一個候選訊號。

        ``backtest_summary`` 傳入的是**已經算好的**統計量（Sharpe、PBO、
        換手率等）。模型只解讀，不計算——這是 SPEC 1 第一條鐵律。
        """
        summary = "\n".join(f"- {key}：{value}" for key, value in backtest_summary.items())
        content = f"候選訊號：\n{signal_description}\n\n回測統計：\n{summary}"
        return self._runner.run(user_content=content, evidence_ids=evidence_ids)


@dataclass
class LibrarianAgent:
    """研究知識庫，避免重複測試造成隱性多重檢定（SPEC 4.2）。"""

    client: LLMClient
    budget: AgentBudget

    def __post_init__(self) -> None:
        self._runner: AgentRunner[LibrarianVerdict] = _runner(
            name="LibrarianAgent",
            payload_type=LibrarianVerdict,
            client=self.client,
            budget=self.budget,
            role=_LIBRARIAN_ROLE,
        )

    @property
    def call_log(self) -> list[AgentCall]:
        return self._runner.call_log

    def check_duplicate(
        self,
        new_hypothesis: str,
        *,
        existing: Mapping[str, str],
    ) -> AgentOutput[LibrarianVerdict]:
        """判斷新假設是否已經測過。``existing`` 是 id 對假設敘述。"""
        if not existing:
            return self._runner.abstain(reason="知識庫為空，無從比對")
        catalogue = "\n".join(f"[{key}] {value}" for key, value in existing.items())
        content = f"新假設：\n{new_hypothesis}\n\n既有假設：\n{catalogue}"
        return self._runner.run(user_content=content)
