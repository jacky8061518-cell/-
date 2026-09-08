"""structlog 接線。

每一行日誌的 UTC 時間戳都取自當前生效的時鐘，因此回測內產生的日誌會被標上
模擬時間，與它所描述的事件對得起來。

dev 輸出彩色文字，其餘環境輸出 JSON（SPEC 2）。
"""

from __future__ import annotations

import logging
from typing import Any

import structlog

from trading_intel.core.clock import utc_now
from trading_intel.core.ids import CorrelationId

_CORRELATION_KEY = "correlation_id"


def _add_timestamp(
    logger: Any,  # noqa: ARG001  (簽章由 structlog processor 規定)
    method_name: str,  # noqa: ARG001
    event_dict: structlog.types.EventDict,
) -> structlog.types.EventDict:
    """一律以當前生效的時鐘蓋時間戳，絕不直接取牆上時間。"""
    event_dict["timestamp"] = utc_now().isoformat()
    return event_dict


def configure_logging(env: str, level: str = "INFO") -> None:
    """dev 環境輸出人眼可讀的文字，其他環境輸出 JSON。"""
    renderer: structlog.types.Processor = (
        structlog.dev.ConsoleRenderer(colors=True)
        if env == "dev"
        else structlog.processors.JSONRenderer(sort_keys=True)
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            _add_timestamp,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping()[level.upper()]
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def bind_correlation(cid: CorrelationId) -> None:
    """把 ``cid`` 附加到此 context 內產生的每一行日誌。"""
    structlog.contextvars.bind_contextvars(**{_CORRELATION_KEY: str(cid)})


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """明確綁定模組名稱；以 print 為底的 factory 本身沒有名字。"""
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name).bind(logger=name)
    return logger
