"""訊號模型與融合。

含 regime.py 市場狀態分類（純程式碼，不使用 LLM）。
訊號融合禁止簡單平均，須依 regime 條件加權並以 Ledoit-Wolf 收縮處理協方差（SPEC 6）。

Phase 2 起逐步實作，融合部分於 Phase 4。
"""
