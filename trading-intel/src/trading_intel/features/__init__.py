"""L2 特徵層：point-in-time feature store 與統計量。

所有特徵為純函式，簽章 f(data, asof, **params) -> Series，並且版本化。
均值回歸類特徵必須先過定態檢定（ADF/KPSS），視窗長度由 OU 過程半衰期決定，
不得拍腦袋設 20 日（SPEC 5.1）。

Phase 2 實作。覆蓋率門檻 90%。
"""
