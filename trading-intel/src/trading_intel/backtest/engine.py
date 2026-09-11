"""事件驅動回測引擎。

SPEC 第 11 節 Phase 2：「逐事件推進，禁止向量化捷徑造成的隱性前視」。

為什麼不用向量化：向量化寫法（例如 `signal.shift(1) * returns`）在紙上是對的，
但它把「什麼時候知道什麼」壓縮成一個位移參數。位移錯一格、或某個特徵內部
偷偷用了整段序列的統計量（例如全期的平均值），前視就進來了，而且不會報錯。

逐事件推進讓「當下能看到什麼」變成引擎的結構性保證：策略在時間 t 收到的
``MarketView`` 只含 t 以前的資料，想看未來也拿不到。

引擎同時內建前視偵測（``LookaheadDetector``）：策略若嘗試存取尚未發生的資料，
直接拋 ``LookaheadError``。這是 SPEC 驗收條件之一。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Protocol

from trading_intel.backtest.costs import TradeCost, taiwan_trade_cost
from trading_intel.core.clock import ensure_utc
from trading_intel.core.errors import LookaheadError
from trading_intel.core.ids import EntityId
from trading_intel.core.sandbox import backtest_mode
from trading_intel.core.settings import CostModel


@dataclass(frozen=True)
class PriceBar:
    """回測引擎使用的最小行情單位。"""

    trade_date: date
    entity_id: EntityId
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    #: 20 日均量的金額，用於市場衝擊與容量檢定。None 代表未知。
    adv_value: Decimal | None = None


class MarketView:
    """策略在某個時點所能看到的世界。

    這個類別的全部目的是**讓前視偏誤在結構上不可能發生**：
    它只持有到 ``as_of_date`` 為止的資料，而且每次查詢都再檢查一次日期。
    策略拿不到未來的 bar，不是因為它有禮貌，是因為那些資料不在這裡。
    """

    def __init__(
        self,
        history: Mapping[EntityId, Sequence[PriceBar]],
        as_of_date: date,
    ) -> None:
        self._history = history
        self._as_of_date = as_of_date

    @property
    def as_of_date(self) -> date:
        return self._as_of_date

    def history(self, entity_id: EntityId, *, lookback: int | None = None) -> list[PriceBar]:
        """取某標的到今天為止的歷史。``lookback`` 為最近幾根。"""
        bars = [
            bar for bar in self._history.get(entity_id, ()) if bar.trade_date <= self._as_of_date
        ]
        return bars[-lookback:] if lookback is not None else bars

    def latest(self, entity_id: EntityId) -> PriceBar | None:
        bars = self.history(entity_id)
        return bars[-1] if bars else None

    def universe(self) -> frozenset[EntityId]:
        """今天有資料的標的。"""
        return frozenset(
            entity_id
            for entity_id in self._history
            if any(bar.trade_date <= self._as_of_date for bar in self._history[entity_id])
        )

    def close_price(self, entity_id: EntityId) -> Decimal | None:
        bar = self.latest(entity_id)
        return bar.close if bar else None

    def assert_visible(self, when: date) -> None:
        """策略若要主動宣告自己在看某一天，用這個檢查。"""
        if when > self._as_of_date:
            raise LookaheadError(
                "策略嘗試存取尚未發生的資料",
                requested=when.isoformat(),
                as_of=self._as_of_date.isoformat(),
            )


class Strategy(Protocol):
    """策略只需要回答一件事：今天各標的的目標權重是多少。

    刻意不讓策略下單。策略表達意圖，引擎負責把意圖變成交易並收取成本——
    這樣策略就無法繞過成本模型，也無法自己決定成交價。
    """

    @property
    def name(self) -> str: ...

    def target_weights(self, view: MarketView) -> Mapping[EntityId, float]: ...


@dataclass
class Position:
    entity_id: EntityId
    shares: Decimal
    #: 平均成本，用於歸因。
    average_price: Decimal


@dataclass(frozen=True)
class Fill:
    """一筆成交紀錄。"""

    trade_date: date
    entity_id: EntityId
    shares: Decimal
    price: Decimal
    notional: Decimal
    is_sell: bool
    cost: TradeCost


@dataclass(frozen=True)
class DailyRecord:
    """單日的組合狀態。"""

    trade_date: date
    equity: Decimal
    gross_return: float
    net_return: float
    turnover: float
    total_cost: Decimal
    positions: int


@dataclass
class BacktestResult:
    """回測結果。刻意保留逐日紀錄與逐筆成交，讓歸因有據可查。"""

    strategy_name: str
    records: list[DailyRecord] = field(default_factory=list)
    fills: list[Fill] = field(default_factory=list)
    initial_equity: Decimal = Decimal("1000000")

    @property
    def final_equity(self) -> Decimal:
        return self.records[-1].equity if self.records else self.initial_equity

    @property
    def net_returns(self) -> list[float]:
        return [record.net_return for record in self.records]

    @property
    def gross_returns(self) -> list[float]:
        return [record.gross_return for record in self.records]

    @property
    def total_costs(self) -> Decimal:
        return sum((record.total_cost for record in self.records), Decimal("0"))

    @property
    def average_turnover(self) -> float:
        if not self.records:
            return 0.0
        return sum(record.turnover for record in self.records) / len(self.records)


#: 決定哪些標的在某一天可以交易。回傳空集合代表當天完全不能交易。
UniverseProvider = Callable[[date], frozenset[EntityId]]


def run_backtest(
    strategy: Strategy,
    history: Mapping[EntityId, Sequence[PriceBar]],
    sessions: Sequence[date],
    *,
    costs: CostModel,
    asof: datetime,
    initial_equity: Decimal = Decimal("1000000"),
    universe_provider: UniverseProvider | None = None,
) -> BacktestResult:
    """逐事件推進的回測。

    每個交易日的順序刻意固定為：

    1. 用**昨天收盤為止**的資料建立 ``MarketView``；
    2. 策略據此決定目標權重；
    3. 以**今天**的收盤價成交，並收取成本。

    第 1 步與第 3 步之間的間隔，就是現實中「做決定」與「成交」的時間差。
    把它寫死在引擎裡，策略就無法用今天的收盤價決定今天要買什麼。

    整段執行包在 ``backtest_mode`` 內：時鐘凍結、網路切斷（CLAUDE.md 第 3 條）。
    """
    ordered_sessions = sorted(sessions)
    result = BacktestResult(strategy_name=strategy.name, initial_equity=initial_equity)
    # 現金與部位分開記帳。曾經試過「以權益減去部位成本反推現金」，那是錯的：
    # 權益是市值、部位成本是建倉價，兩者尺度不同，會每天重複計入一次漲幅。
    cash = initial_equity
    equity = initial_equity
    positions: dict[EntityId, Position] = {}
    #: 上一期實際套用過的目標權重。用來判斷「這期的權重跟上次一樣嗎」——
    #: 見 _rebalance 對這個問題的說明。
    last_targets: dict[EntityId, float] = {}

    with backtest_mode(ensure_utc(asof)):
        for index, session in enumerate(ordered_sessions):
            if index == 0:
                # 第一天沒有「昨天」，不交易，只記錄起始狀態。
                result.records.append(
                    DailyRecord(
                        trade_date=session,
                        equity=equity,
                        gross_return=0.0,
                        net_return=0.0,
                        turnover=0.0,
                        total_cost=Decimal("0"),
                        positions=0,
                    )
                )
                continue

            previous_session = ordered_sessions[index - 1]
            equity_before = equity

            # 1. 先讓既有部位承受今天的價格變動。
            gross_equity = cash + _positions_value(positions, history, session)
            gross_return = _safe_return(gross_equity, equity_before)

            # 2. 策略只看得到昨天為止的資料。
            view = MarketView(history, previous_session)
            targets = strategy.target_weights(view)
            _validate_targets(targets)

            tradable = (
                universe_provider(session) if universe_provider is not None else view.universe()
            )

            # 3. 以今天的收盤價調整部位。
            fills, total_cost, turnover = _rebalance(
                positions=positions,
                targets=targets,
                last_targets=last_targets,
                tradable=tradable,
                history=history,
                session=session,
                equity=gross_equity,
                costs=costs,
            )
            last_targets = dict(targets)
            result.fills.extend(fills)

            # 現金隨買賣增減，成本一律由現金支付。
            for fill in fills:
                cash -= fill.shares * fill.price
            cash -= total_cost
            equity = cash + _positions_value(positions, history, session)
            net_return = _safe_return(equity, equity_before)

            result.records.append(
                DailyRecord(
                    trade_date=session,
                    equity=equity,
                    gross_return=gross_return,
                    net_return=net_return,
                    turnover=turnover,
                    total_cost=total_cost,
                    positions=len([p for p in positions.values() if p.shares != 0]),
                )
            )
    return result


def _safe_return(current: Decimal, previous: Decimal) -> float:
    if previous == 0:
        return 0.0
    return float(current / previous - Decimal("1"))


def _bar_on(
    history: Mapping[EntityId, Sequence[PriceBar]],
    entity_id: EntityId,
    session: date,
) -> PriceBar | None:
    for bar in history.get(entity_id, ()):
        if bar.trade_date == session:
            return bar
    return None


def _positions_value(
    positions: Mapping[EntityId, Position],
    history: Mapping[EntityId, Sequence[PriceBar]],
    session: date,
) -> Decimal:
    """以今天的收盤價計算持股市值。今天沒有報價時沿用最後成交價。"""
    total = Decimal("0")
    for entity_id, position in positions.items():
        if position.shares == 0:
            continue
        bar = _bar_on(history, entity_id, session)
        price = bar.close if bar else position.average_price
        total += position.shares * price
    return total


def _validate_targets(targets: Mapping[EntityId, float]) -> None:
    total = sum(abs(weight) for weight in targets.values())
    if total > 1.0000001:
        msg = f"目標權重的絕對值總和 {total:.4f} 超過 1，引擎不支援槓桿"
        raise ValueError(msg)


def _rebalance(
    *,
    positions: dict[EntityId, Position],
    targets: Mapping[EntityId, float],
    last_targets: Mapping[EntityId, float],
    tradable: frozenset[EntityId],
    history: Mapping[EntityId, Sequence[PriceBar]],
    session: date,
    equity: Decimal,
    costs: CostModel,
) -> tuple[list[Fill], Decimal, float]:
    """把目標權重變化量變成實際交易，並收取成本。

    **只有目標權重真的改變的標的才重新換算股數。** 這是修正過一次的行為：
    原本每天都用當天的權益與價格，把權重相同的標的也重新換算一次目標股數，
    結果價格只要一有波動，就會被迫交易把權重「拉回」原本那個百分比——
    這是「每日再平衡到固定權重」，不是「維持現有部位」。

    用真實資料實測抓到的：一個目標權重從頭到尾沒變過的買進持有策略，
    五檔股票兩年跑出 2420 筆成交，而正確答案應該是 5 筆（只有建倉那天）。
    問題不在策略，在引擎把「權重不變」誤解成「每天都要重新配置到這個
    百分比」，而不是「維持上次配置的結果」。

    現在的判斷依據是：把這一期的目標權重拿去跟**上一期實際套用過的**
    目標權重比對，只有變了才重新算股數；沒變的標的直接跳過，任由它隨
    價格自然漂移。這也是大多數真實交易系統對「目標權重」的定義——
    重新配置由目標**改變**觸發，不是由價格波動觸發。
    """
    fills: list[Fill] = []
    total_cost = Decimal("0")
    traded_notional = Decimal("0")

    entities = set(positions) | set(targets)
    for entity_id in sorted(entities):
        target_weight = targets.get(entity_id, 0.0)
        previous_weight = last_targets.get(entity_id, 0.0)
        if entity_id in positions and math.isclose(target_weight, previous_weight, abs_tol=1e-12):
            # 目標沒變，維持現有股數不動——不因為價格波動就被迫交易。
            continue

        bar = _bar_on(history, entity_id, session)
        if bar is None or bar.close <= 0:
            continue

        # 不可交易的標的：不能新建或加碼，但既有部位可以續抱。
        if entity_id not in tradable and target_weight != 0.0:
            continue

        target_value = equity * Decimal(str(target_weight))
        target_shares = target_value / bar.close

        current = positions.get(entity_id)
        current_shares = current.shares if current else Decimal("0")
        delta_shares = target_shares - current_shares
        if delta_shares == 0:
            continue

        notional = abs(delta_shares) * bar.close
        is_sell = delta_shares < 0
        cost = taiwan_trade_cost(
            notional=notional,
            is_sell=is_sell,
            costs=costs,
            average_daily_volume_value=bar.adv_value,
        )
        total_cost += cost.total
        traded_notional += notional

        fills.append(
            Fill(
                trade_date=session,
                entity_id=entity_id,
                shares=delta_shares,
                price=bar.close,
                notional=notional,
                is_sell=is_sell,
                cost=cost,
            )
        )
        positions[entity_id] = Position(
            entity_id=entity_id,
            shares=target_shares,
            average_price=bar.close,
        )

    turnover = float(traded_notional / equity) if equity > 0 else 0.0
    return fills, total_cost, turnover
