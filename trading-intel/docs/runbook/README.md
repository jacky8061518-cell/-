# 故障排除手冊

每一類可能在半夜把人叫醒的故障各一份文件，內容為症狀、判斷步驟、處置動作、
以及「什麼情況下應該直接停止交易而不是繼續排查」。

Phase 0 尚無執行中的元件，因此本目錄暫時只有這份說明。
隨 Phase 1 的資料源與事件匯流排上線後開始累積，預期的第一批項目：

- `data-source-outage.md` 資料源中斷（P0）
- `event-bus-lag.md` 事件匯流排堆積與 consumer group 落後
- `position-reconciliation-mismatch.md` 部位對帳不符（P0）
- `clock-skew.md` 時鐘偏移導致 TemporalIntegrityError
