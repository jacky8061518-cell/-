"""L3 代理人層：LLM 代理人，只讀 L2 的結構化輸出，只輸出 typed schema。

通用契約定義於 core.types.AgentOutput：confidence、evidence_ids、reasoning_digest、abstain。
棄權是合法輸出，證據不足必須回傳 abstain=True。

工具集中不得存在下單、改限額、關風控的能力——這是程式碼層的限制，不是提示詞的請求。

Phase 3 實作。Phase 2 驗收未過前不得動工（SPEC 1）。
"""
