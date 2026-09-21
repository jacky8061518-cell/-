"""
agents/trading_crew.py
AI 智能體系統：使用 CrewAI 架構，指定 Claude Sonnet 作為大腦，
由「市場偵察員」「情緒分析師」「首席策略官」三個智能體協作產出交易建議。
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import tool
from duckduckgo_search import DDGS

from agents.parsing import parse_strategist_output

if TYPE_CHECKING:
    from core.quant_engine import MarketSnapshot

# 可透過環境變數覆寫模型名稱，預設使用 Claude Sonnet 5
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")


@tool("新聞搜尋工具")
def search_news(query: str) -> str:
    """使用 DuckDuckGo 搜尋與標的相關的最新新聞，回傳標題、日期與摘要列表。"""
    with DDGS() as ddgs:
        results = list(ddgs.news(query, max_results=5))
    if not results:
        return f"找不到與「{query}」相關的最新新聞。"
    lines = [
        f"- [{item.get('date', '未知日期')}] {item.get('title', '')}：{item.get('body', '')}"
        for item in results
    ]
    return "\n".join(lines)


def _build_llm() -> LLM:
    """建立作為所有智能體大腦的 Claude Sonnet 模型（使用 CrewAI 原生 LLM，經由 Anthropic 官方 API 呼叫）。

    註：刻意不傳入 temperature —— 目前安裝的 anthropic SDK 已從
    Messages.create() 移除該參數，而 crewai 只要偵測到 temperature 有值
    就會原樣轉傳，兩者版本組合會導致呼叫失敗（TypeError: unexpected
    keyword argument 'temperature'）。留白讓 crewai 略過此參數即可避開。
    """
    return LLM(model=ANTHROPIC_MODEL)


def build_agents(llm: LLM) -> tuple[Agent, Agent, Agent]:
    """建立市場偵察員、情緒分析師、首席策略官三個智能體。"""
    scout = Agent(
        role="市場偵察員",
        goal="從量化引擎提供的 Z-Score 與技術指標中，識別出具有統計顯著性的異常交易機會。",
        backstory=(
            "你是一位專精於統計套利與均值回歸策略的量化分析師，"
            "擅長從大量市場數據中快速找出價格偏離常態分布的訊號。"
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    sentiment = Agent(
        role="情緒分析師",
        goal="針對偵察員發現的異常標的，搜尋最新新聞並判斷市場情緒是利多、利空還是中性。",
        backstory="你是一位資深財經新聞編輯，擅長快速消化新聞內容並萃取出對交易決策有幫助的情緒判斷。",
        llm=llm,
        tools=[search_news],
        verbose=True,
        allow_delegation=False,
    )

    strategist = Agent(
        role="首席策略官",
        goal="整合量化數據與情緒報告，產出具體可執行的交易建議。",
        backstory="你是基金的首席策略官，負責在風險可控的前提下，將量化訊號與市場情緒轉化為明確的交易指令。",
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )
    return scout, sentiment, strategist


def build_tasks(
    scout: Agent, sentiment: Agent, strategist: Agent, symbol: str, quant_summary: str
) -> tuple[Task, Task, Task]:
    """建立三個智能體依序執行的任務。"""
    scout_task = Task(
        description=(
            f"以下是標的 {symbol} 的量化引擎輸出：\n{quant_summary}\n\n"
            "請分析這份數據，說明目前的統計異常程度（Z-Score 意義）、"
            "價格相對布林帶的位置，以及這是否構成值得進一步調查的交易機會。"
        ),
        expected_output="一段繁體中文分析，說明異常訊號的統計意義與初步判斷。",
        agent=scout,
    )

    sentiment_task = Task(
        description=(
            f"針對標的 {symbol}，使用新聞搜尋工具查詢最新相關新聞，"
            "並根據偵察員的分析結果，判斷目前市場情緒是利多、利空還是中性，並說明理由。"
        ),
        expected_output="一段繁體中文情緒報告，包含新聞來源摘要與利多/利空/中性的判斷。",
        agent=sentiment,
        context=[scout_task],
    )

    strategist_task = Task(
        description=(
            "整合市場偵察員的量化分析與情緒分析師的新聞報告，給出最終交易建議。"
            "請務必以下列格式回覆（每個欄位獨立一行）：\n"
            "Action: Buy/Sell/Hold\n"
            "Entry: <進場價格>\n"
            "TP: <停利價格>\n"
            "SL: <停損價格>\n"
            "Confidence: <0 到 1 之間的信心評分>\n"
            "Reasoning: <繁體中文理由，需同時引用量化數據與情緒報告>"
        ),
        expected_output="依照指定格式輸出的繁體中文交易建議。",
        agent=strategist,
        context=[scout_task, sentiment_task],
    )
    return scout_task, sentiment_task, strategist_task


def analyze_opportunity(symbol: str, snapshot: "MarketSnapshot") -> dict[str, Any]:
    """針對單一異常標的執行完整的 AI 智能體分析流程，回傳結構化的交易建議。"""
    llm = _build_llm()
    scout, sentiment, strategist = build_agents(llm)

    quant_summary = (
        f"目前價格：{snapshot.price:.2f}\n"
        f"20 週期均值：{snapshot.mean:.2f}\n"
        f"布林上軌：{snapshot.upper_band:.2f}\n"
        f"布林下軌：{snapshot.lower_band:.2f}\n"
        f"Z-Score：{snapshot.z_score:.2f}\n"
        f"觸發訊號：{snapshot.signal}"
    )

    scout_task, sentiment_task, strategist_task = build_tasks(
        scout, sentiment, strategist, symbol, quant_summary
    )

    crew = Crew(
        agents=[scout, sentiment, strategist],
        tasks=[scout_task, sentiment_task, strategist_task],
        process=Process.sequential,
        verbose=True,
    )

    crew_result = crew.kickoff()
    final_text = str(crew_result)

    return {
        "symbol": symbol,
        "final_report": final_text,
        "scout_analysis": str(scout_task.output) if scout_task.output else "",
        "sentiment_report": str(sentiment_task.output) if sentiment_task.output else "",
        **parse_strategist_output(final_text),
    }
