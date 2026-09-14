"""L0 擷取層：行情、新聞、公告、籌碼、總經、社群。

各資料源一個 worker，負責拉取與落地。原始 payload 進 raw landing zone，
以 Parquet 儲存且永不覆寫，路徑含抓取時間戳。多數 worker 為確定性程式碼，不用 LLM。

Phase 1 實作。
"""
