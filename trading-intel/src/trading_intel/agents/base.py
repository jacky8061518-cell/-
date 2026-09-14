"""Agent 通用契約（SPEC 4.1）。

所有 agent 共用同一套骨架，關鍵設計有三點：

**一、棄權是合法輸出。** 證據不足時回傳 ``abstain=True``，不得硬猜。
逾時、格式驗證失敗、重試三次仍失敗，一律降級為棄權，**不得讓例外向上冒泡到
決策層**——因為決策層收到例外只能整批放棄，收到棄權則可以繼續處理其他標的。

**二、agent 的工具集裡不存在危險能力。** 下單、修改限額、關閉風控這些能力
在程式碼層就不存在於 agent 可觸及的範圍，不是靠提示詞請它不要做（SPEC 4.3）。

**三、LLM 客戶端是注入的。** 這讓離線測試集（固定輸入對固定期望輸出）成為可能，
也讓「回測期間禁止網路」這條規則自動成立——測試用的假客戶端根本不連網。
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Final, Protocol, runtime_checkable

from pydantic import BaseModel, ValidationError

from trading_intel.agents.injection import build_system_prompt, wrap_untrusted
from trading_intel.core.clock import utc_now
from trading_intel.core.errors import BudgetExceededError, SchemaValidationError
from trading_intel.core.ids import CorrelationId, EvidenceId
from trading_intel.core.settings import AgentBudget
from trading_intel.core.types import AgentOutput

#: 預設模型。SPEC 未指定，採用目前最強的通用模型。
DEFAULT_MODEL: Final = "claude-opus-5"

#: 重試上限。SPEC 4.1 明訂「重試三次仍失敗，一律降級為 abstain」。
MAX_ATTEMPTS: Final = 3

#: 棄權時的預設信心值。棄權代表「我不知道」，信心自然為零。
ABSTAIN_CONFIDENCE: Final = 0.0


@dataclass(frozen=True)
class LLMResponse:
    """一次 LLM 呼叫的原始回應。"""

    text: str
    input_tokens: int
    output_tokens: int
    model: str
    #: stop_reason 為 refusal 時，代表模型拒絕回答，應降級為棄權而非重試。
    refused: bool = False


@runtime_checkable
class LLMClient(Protocol):
    """LLM 客戶端的最小介面。

    刻意只有一個方法。agent 不需要串流、不需要工具呼叫、不需要多輪對話——
    它們做的是「給一段結構化輸入，回一個結構化輸出」的單次任務。
    介面越窄，測試替身越可信。
    """

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        timeout_seconds: float,
    ) -> LLMResponse: ...


@dataclass
class AgentCall:
    """一次 agent 呼叫的稽核紀錄（SPEC 4.1 要求寫入 agent_calls 表）。"""

    agent_name: str
    model: str
    prompt_hash: str
    called_at: datetime
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    attempts: int
    abstained: bool
    correlation_id: CorrelationId | None = None
    error: str = ""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class BudgetTracker:
    """每個 agent 獨立的 token 預算與速率上限（SPEC 4.1）。

    超支時拋 ``BudgetExceededError``，由 ``AgentRunner`` 接住並降級為棄權——
    預算用完應該讓系統安靜地少做事，而不是整個停擺。
    """

    budget: AgentBudget
    _tokens_today: dict[date, int] = field(default_factory=dict)
    _calls: list[datetime] = field(default_factory=list)

    def check(self, now: datetime) -> None:
        """呼叫前檢查。超出任一上限即拋例外。"""
        today = now.date()
        if self._tokens_today.get(today, 0) >= self.budget.daily_token_budget:
            raise BudgetExceededError(
                "已達每日 token 預算上限",
                used=self._tokens_today.get(today, 0),
                limit=self.budget.daily_token_budget,
            )
        window_start = now - timedelta(hours=1)
        recent = [item for item in self._calls if item > window_start]
        if len(recent) >= self.budget.max_calls_per_hour:
            raise BudgetExceededError(
                "已達每小時呼叫次數上限",
                used=len(recent),
                limit=self.budget.max_calls_per_hour,
            )

    def record(self, now: datetime, tokens: int) -> None:
        today = now.date()
        self._tokens_today[today] = self._tokens_today.get(today, 0) + tokens
        self._calls.append(now)
        # 只保留最近兩小時，避免列表無限成長。
        cutoff = now - timedelta(hours=2)
        self._calls = [item for item in self._calls if item > cutoff]

    def tokens_used_on(self, day: date) -> int:
        return self._tokens_today.get(day, 0)

    def calls_in_last_hour(self, now: datetime) -> int:
        window_start = now - timedelta(hours=1)
        return len([item for item in self._calls if item > window_start])


def prompt_fingerprint(system: str, user: str) -> str:
    """提示的內容指紋，用於稽核與快取分析。"""
    import hashlib

    return hashlib.sha256(f"{system}\x1f{user}".encode()).hexdigest()[:16]


@dataclass
class AgentRunner[T: BaseModel]:
    """把 LLM 呼叫包成符合 ``AgentOutput`` 契約的結果。

    這個類別存在的唯一理由是**讓失敗變得無害**：所有可能的失敗路徑
    （逾時、格式錯誤、模型拒答、預算用盡）最終都收斂成一個 ``abstain=True``
    的合法輸出，決策層因此永遠不需要處理例外。
    """

    agent_name: str
    payload_type: type[T]
    client: LLMClient
    budget: BudgetTracker
    role_instructions: str
    model: str = DEFAULT_MODEL
    max_tokens: int = 4096
    #: 稽核紀錄的接收端。實際部署時寫入 agent_calls 表。
    call_log: list[AgentCall] = field(default_factory=list)

    def run(
        self,
        *,
        user_content: str,
        evidence_ids: Sequence[EvidenceId] = (),
        untrusted_documents: Mapping[str, str] | None = None,
        correlation_id: CorrelationId | None = None,
    ) -> AgentOutput[T]:
        """執行一次 agent 任務。永遠回傳合法的 ``AgentOutput``，不拋例外。

        ``untrusted_documents`` 的內容會被包進資料標記內（SPEC 4.3）。
        """
        system = build_system_prompt(self.role_instructions)
        user = self._compose_user_prompt(user_content, untrusted_documents)
        fingerprint = prompt_fingerprint(system, user)
        started = utc_now()

        attempts = 0
        last_error = ""
        total_input = 0
        total_output = 0

        while attempts < MAX_ATTEMPTS:
            attempts += 1
            try:
                self.budget.check(utc_now())
            except BudgetExceededError as exc:
                last_error = str(exc)
                break

            try:
                response = self.client.complete(
                    system=system,
                    user=user,
                    max_tokens=self.max_tokens,
                    timeout_seconds=self.budget.budget.timeout_seconds,
                )
            except Exception as exc:  # 逾時、連線錯誤、SDK 例外一律視為可重試
                last_error = f"{type(exc).__name__}: {exc}"
                continue

            total_input += response.input_tokens
            total_output += response.output_tokens
            self.budget.record(utc_now(), response.input_tokens + response.output_tokens)

            if response.refused:
                # 模型拒答不是暫時性錯誤，重試沒有意義。
                last_error = "模型拒絕回答"
                break

            try:
                payload = self._parse(response.text)
            except SchemaValidationError as exc:
                last_error = str(exc)
                continue

            elapsed = (utc_now() - started).total_seconds()
            self._log(
                fingerprint=fingerprint,
                started=started,
                elapsed=elapsed,
                input_tokens=total_input,
                output_tokens=total_output,
                attempts=attempts,
                abstained=False,
                correlation_id=correlation_id,
            )
            return AgentOutput[self.payload_type](  # type: ignore[name-defined]
                payload=payload,
                confidence=self._confidence_of(payload),
                evidence_ids=tuple(evidence_ids),
                reasoning_digest=self._digest_of(payload),
                abstain=False,
                model_name=response.model,
                prompt_hash=fingerprint,
            )

        # 所有路徑失敗 → 棄權。這是合法輸出，不是例外。
        elapsed = (utc_now() - started).total_seconds()
        self._log(
            fingerprint=fingerprint,
            started=started,
            elapsed=elapsed,
            input_tokens=total_input,
            output_tokens=total_output,
            attempts=attempts,
            abstained=True,
            correlation_id=correlation_id,
            error=last_error,
        )
        return self.abstain(evidence_ids=evidence_ids, reason=last_error, prompt_hash=fingerprint)

    def abstain(
        self,
        *,
        evidence_ids: Sequence[EvidenceId] = (),
        reason: str = "",
        prompt_hash: str = "",
    ) -> AgentOutput[T]:
        """產生一個棄權輸出。"""
        digest = f"棄權：{reason}"[:200] if reason else "棄權：證據不足"
        return AgentOutput[self.payload_type](  # type: ignore[name-defined]
            payload=None,
            confidence=ABSTAIN_CONFIDENCE,
            evidence_ids=tuple(evidence_ids),
            reasoning_digest=digest,
            abstain=True,
            model_name=self.model,
            prompt_hash=prompt_hash,
        )

    def _compose_user_prompt(
        self,
        content: str,
        untrusted: Mapping[str, str] | None,
    ) -> str:
        if not untrusted:
            return content
        blocks = [wrap_untrusted(text, source=source) for source, text in untrusted.items()]
        return content + "\n\n" + "\n\n".join(blocks)

    def _parse(self, text: str) -> T:
        """把模型輸出解析成 payload 型別。超出列舉值的欄位會在此被拒收。"""
        stripped = text.strip()
        # 模型偶爾會用 ```json 包住輸出。
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            stripped = "\n".join(lines[1:-1]) if len(lines) > 2 else stripped
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise SchemaValidationError("模型輸出不是合法 JSON", detail=str(exc)) from exc
        try:
            return self.payload_type.model_validate(data)
        except ValidationError as exc:
            raise SchemaValidationError("模型輸出不符合 schema", detail=str(exc)) from exc

    def _confidence_of(self, payload: T) -> float:
        value = getattr(payload, "confidence", None)
        if isinstance(value, int | float):
            return max(0.0, min(1.0, float(value)))
        return 0.5

    def _digest_of(self, payload: T) -> str:
        value = getattr(payload, "reasoning_digest", None)
        if isinstance(value, str):
            return value[:200]
        return ""

    def _log(
        self,
        *,
        fingerprint: str,
        started: datetime,
        elapsed: float,
        input_tokens: int,
        output_tokens: int,
        attempts: int,
        abstained: bool,
        correlation_id: CorrelationId | None,
        error: str = "",
    ) -> None:
        self.call_log.append(
            AgentCall(
                agent_name=self.agent_name,
                model=self.model,
                prompt_hash=fingerprint,
                called_at=started,
                latency_seconds=elapsed,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                attempts=attempts,
                abstained=abstained,
                correlation_id=correlation_id,
                error=error,
            )
        )


@dataclass
class ScriptedClient:
    """離線測試用的假客戶端：依序回傳預先寫好的回應。

    SPEC 第 11 節 Phase 3 要求每個 agent 都有離線測試集
    （固定輸入對固定期望輸出）。這個類別就是那套測試的執行器。
    它不連網，因此測試在任何環境都能跑。
    """

    # 用 Sequence 而非 list：list 在型別上是不變的，會讓呼叫端傳
    # list[LLMResponse] 時被拒絕。改用索引推進，不消耗原序列。
    responses: Sequence[LLMResponse | Exception]
    calls: list[tuple[str, str]] = field(default_factory=list)
    _cursor: int = 0

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,  # noqa: ARG002
        timeout_seconds: float,  # noqa: ARG002
    ) -> LLMResponse:
        self.calls.append((system, user))
        if self._cursor >= len(self.responses):
            msg = "腳本已用盡"
            raise RuntimeError(msg)
        item = self.responses[self._cursor]
        self._cursor += 1
        if isinstance(item, Exception):
            raise item
        return item


def json_response(
    payload: Mapping[str, Any],
    *,
    model: str = DEFAULT_MODEL,
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> LLMResponse:
    """把一個 dict 包成 LLM 回應，供測試使用。"""
    return LLMResponse(
        text=json.dumps(payload, ensure_ascii=False),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=model,
    )


def measure_consistency(
    outputs: Sequence[AgentOutput[Any]],
    *,
    key: Callable[[Any], object] | None = None,
) -> float:
    """一致性指標：同一輸入重複詢問，多數答案所佔的比例（SPEC 7.3）。

    回傳 1.0 代表完全一致。分歧度過高代表該任務不適合 LLM，應退回確定性做法。
    """
    if not outputs:
        return 0.0
    extract = key or (
        lambda payload: json.dumps(
            payload.model_dump(mode="json"), sort_keys=True, ensure_ascii=False
        )
    )
    answers = [
        "ABSTAIN" if output.abstain or output.payload is None else extract(output.payload)
        for output in outputs
    ]
    counts: dict[object, int] = {}
    for answer in answers:
        counts[answer] = counts.get(answer, 0) + 1
    return max(counts.values()) / len(answers)


class AnthropicClient:
    """正式環境使用的 Anthropic 客戶端。

    刻意只實作 ``LLMClient`` 那一個方法。agent 需要的是「單次結構化問答」，
    不需要串流或工具呼叫——介面越窄，agent 能做的危險事情越少。

    **本類別在本執行環境無法實測**：需要 ANTHROPIC_API_KEY，而離線測試一律
    使用 ``ScriptedClient``。網路放行且設定金鑰後即可使用。
    """

    def __init__(self, model: str = DEFAULT_MODEL, *, client: Any = None) -> None:
        self.model = model
        self._client = client

    def _ensure_client(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        timeout_seconds: float,
    ) -> LLMResponse:
        client = self._ensure_client()
        started = time.monotonic()
        response = client.with_options(timeout=timeout_seconds).messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        del started
        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        return LLMResponse(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=response.model,
            refused=response.stop_reason == "refusal",
        )
