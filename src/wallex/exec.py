"""Execution adapter: quantised LIMIT orders in two modes (docs/plan.md §4, R4-Q02).

- ``maker``: post at the best *same-side* price, poll every 30 s, re-peg after 5 min if the
  best moved, and at 20 min after cycle start cancel the rest and cross with a LIMIT capped
  at 0.2 % beyond the opposite best (unless ``allow_cross=False``).
- ``through``: LIMIT at the opposite best ± 0.2 % immediately; if not filled after 60 s,
  cancel and re-place the remainder once (plan §2 "retry once after 60 s").

Orders reach Wallex only through :class:`LiveBroker`, which needs ``live=True`` *and*
``WALLEX_LIVE=1``. :class:`PaperBroker` simulates fills against the real (or injected)
order book: a resting order fills when the opposite best reaches its price; fees are taker.
Time and sleeping are injectable so the whole loop is unit-testable without waiting.
"""

from __future__ import annotations

import os
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import ROUND_DOWN, ROUND_UP, Decimal
from typing import Any, Protocol

from wallex.client import WallexClient, WallexError

MAKER_FEE = 0.0010
TAKER_FEE = 0.0013
CROSS_CAP = Decimal("0.002")  # 0.2 % beyond the opposite best (plan §2)
POLL_S = 30
REPEG_S = 300
CROSS_AT_S = 1200
THROUGH_WAIT_S = 60
RATE_N, RATE_WINDOW_S = 20, 10  # order creation: 20 req / 10 s (docs)

# --- market rules and quantisation -------------------------------------------------


@dataclass(frozen=True)
class MarketRules:
    tick: int  # price decimals
    step: int  # quantity decimals
    min_notional: Decimal
    min_qty: Decimal = Decimal(0)

    @classmethod
    def from_market(cls, m: dict[str, Any]) -> MarketRules:
        """From one ``/v1/markets`` symbol record (tickSize/stepSize are decimal counts)."""
        return cls(
            tick=int(m["tickSize"]),
            step=int(m["stepSize"]),
            min_notional=Decimal(str(m.get("minNotional") or 0)),
            min_qty=Decimal(str(m.get("minQty") or 0)),
        )

    def q_price(self, price: Decimal | float | str, side: str) -> Decimal:
        """Round toward the passive side: buys down, sells up (never pay more than asked)."""
        rounding = ROUND_DOWN if side.upper() == "BUY" else ROUND_UP
        return Decimal(str(price)).quantize(Decimal(1).scaleb(-self.tick), rounding=rounding)

    def q_qty(self, qty: Decimal | float | str) -> Decimal:
        return Decimal(str(qty)).quantize(Decimal(1).scaleb(-self.step), rounding=ROUND_DOWN)

    def valid(self, qty: Decimal, price: Decimal) -> bool:
        return qty >= self.min_qty and qty > 0 and qty * price >= self.min_notional


def load_rules(client: WallexClient, symbols: list[str]) -> dict[str, MarketRules]:
    mk = client.markets()["symbols"]
    return {s: MarketRules.from_market(mk[s]) for s in symbols}


# --- broker abstraction ----------------------------------------------------------------


@dataclass
class OrderState:
    client_id: str
    filled: Decimal = Decimal(0)
    vwap: Decimal = Decimal(0)
    fee: Decimal = Decimal(0)  # in quote (USDT), best effort
    active: bool = True


class Broker(Protocol):
    def best(self, symbol: str) -> tuple[Decimal, Decimal]: ...  # (bid, ask)
    def place(self, symbol: str, side: str, qty: Decimal, price: Decimal, cid: str) -> None: ...
    def status(self, cid: str) -> OrderState: ...
    def cancel(self, cid: str) -> OrderState: ...


def _best_from_depth(depth: dict[str, list[dict[str, str]]]) -> tuple[Decimal, Decimal]:
    bids = [Decimal(str(b["price"])) for b in depth.get("bid", [])]
    asks = [Decimal(str(a["price"])) for a in depth.get("ask", [])]
    if not bids or not asks:
        raise WallexError("empty order book")
    return max(bids), min(asks)


def parse_order(o: dict[str, Any], cid: str) -> OrderState:
    """Map a Wallex OrderResource to :class:`OrderState` (fee converted to quote)."""
    filled = Decimal(str(o.get("executedQty") or 0))
    vwap = Decimal(str(o.get("executedPrice") or 0))
    if not vwap and filled:
        vwap = Decimal(str(o.get("executedSum") or 0)) / filled
    fee = Decimal(0)
    fills = o.get("fills") or []
    if fills:
        for f in fills:
            f_fee = Decimal(str(f.get("fee") or 0))
            asset = str(f.get("feeAsset") or "")
            if asset and not o.get("symbol", "").endswith(asset):  # fee in base coin
                f_fee *= Decimal(str(f.get("price") or vwap))
            fee += f_fee
    else:
        fee = Decimal(str(o.get("fee") or 0))
        if str(o.get("side", "")).upper() == "BUY":
            fee *= vwap  # buys are charged in the base coin
    return OrderState(cid, filled, vwap, fee, bool(o.get("active", False)))


class LiveBroker:
    """Real orders. Requires ``live=True`` and env ``WALLEX_LIVE=1``; otherwise raises."""

    def __init__(self, client: WallexClient, *, live: bool = False):
        if not (live and os.environ.get("WALLEX_LIVE") == "1"):
            raise RuntimeError("LiveBroker needs live=True and WALLEX_LIVE=1")
        self.client = client

    def best(self, symbol: str) -> tuple[Decimal, Decimal]:
        return _best_from_depth(self.client.depth(symbol))

    def place(self, symbol: str, side: str, qty: Decimal, price: Decimal, cid: str) -> None:
        self.client.place_order(
            symbol, side, "LIMIT", str(qty), str(price), client_id=cid, live=True
        )

    def status(self, cid: str) -> OrderState:
        return parse_order(self.client.order(cid), cid)

    def cancel(self, cid: str) -> OrderState:
        try:
            self.client.cancel_order(cid, live=True)
        except WallexError:
            pass  # already filled or gone; the status call below is authoritative
        return self.status(cid)


class PaperBroker:
    """Synthetic fills against an order book (``depth_fn(symbol)`` → Wallex depth dict).

    A resting order fills in full when the opposite best reaches its limit price, at the
    better of the two prices. Fees are charged at the taker rate (plan §4, conservative).
    """

    def __init__(self, depth_fn: Callable[[str], dict[str, Any]], fee: float = TAKER_FEE):
        self.depth_fn = depth_fn
        self.fee = Decimal(str(fee))
        self.orders: dict[str, dict[str, Any]] = {}
        self._state: dict[str, OrderState] = {}
        self.log: list[tuple[str, str]] = []  # (action, cid)

    def best(self, symbol: str) -> tuple[Decimal, Decimal]:
        return _best_from_depth(self.depth_fn(symbol))

    def place(self, symbol: str, side: str, qty: Decimal, price: Decimal, cid: str) -> None:
        self.orders[cid] = {"symbol": symbol, "side": side.upper(), "qty": qty, "price": price}
        self.log.append(("place", cid))
        self._state[cid] = OrderState(cid)
        self._match(cid)

    def _match(self, cid: str) -> None:
        st, o = self._state[cid], self.orders[cid]
        if not st.active:
            return
        bid, ask = self.best(o["symbol"])
        if o["side"] == "BUY" and ask <= o["price"]:
            px = ask
        elif o["side"] == "SELL" and bid >= o["price"]:
            px = bid
        else:
            return
        st.filled, st.vwap, st.fee, st.active = o["qty"], px, o["qty"] * px * self.fee, False

    def status(self, cid: str) -> OrderState:
        self._match(cid)
        return self._state[cid]

    def cancel(self, cid: str) -> OrderState:
        self.log.append(("cancel", cid))
        st = self.status(cid)
        st.active = False
        return st


# --- executor -----------------------------------------------------------------------------


@dataclass
class Intent:
    symbol: str
    side: str  # BUY | SELL
    qty: Decimal | None = None  # base units (sells)
    notional: Decimal | None = None  # quote units (buys); converted at the first price

    def __post_init__(self) -> None:
        self.side = self.side.upper()


@dataclass
class ExecResult:
    symbol: str
    side: str
    requested: Decimal
    filled: Decimal = Decimal(0)
    vwap: Decimal = Decimal(0)
    fee: Decimal = Decimal(0)
    crossed: bool = False
    orders: list[str] = field(default_factory=list)
    note: str = ""

    @property
    def status(self) -> str:
        if self.filled <= 0:
            return "none"
        return "filled" if self.filled >= self.requested else "partial"


@dataclass
class _Work:
    intent: Intent
    result: ExecResult
    rules: MarketRules
    phase: str  # maker | through
    cid: str | None = None
    placed_at: float = 0.0
    price: Decimal = Decimal(0)
    through_tries: int = 0
    n: int = 0
    done: bool = False

    @property
    def remaining(self) -> Decimal:
        return self.result.requested - self.result.filled


class Executor:
    def __init__(
        self,
        broker: Broker,
        rules: dict[str, MarketRules],
        *,
        mode: str = "maker",
        cap: Decimal = CROSS_CAP,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        poll_s: float = POLL_S,
        repeg_s: float = REPEG_S,
        cross_at_s: float = CROSS_AT_S,
        through_wait_s: float = THROUGH_WAIT_S,
    ):
        if mode not in ("maker", "through"):
            raise ValueError(mode)
        self.broker, self.rules, self.mode, self.cap = broker, rules, mode, cap
        self.clock, self.sleep = clock, sleep
        self.poll_s, self.repeg_s = poll_s, repeg_s
        self.cross_at_s, self.through_wait_s = cross_at_s, through_wait_s
        self._sent: deque[float] = deque()

    # -- pricing --
    def _price(self, symbol: str, side: str, phase: str) -> Decimal:
        bid, ask = self.broker.best(symbol)
        r = self.rules[symbol]
        if phase == "maker":
            return r.q_price(bid if side == "BUY" else ask, side)
        if side == "BUY":
            return r.q_price(ask * (1 + self.cap), side)
        return r.q_price(bid * (1 - self.cap), side)

    # -- order plumbing --
    def _throttle(self) -> None:
        now = self.clock()
        while self._sent and now - self._sent[0] > RATE_WINDOW_S:
            self._sent.popleft()
        if len(self._sent) >= RATE_N:
            self.sleep(RATE_WINDOW_S - (now - self._sent[0]) + 0.1)
        self._sent.append(self.clock())

    def _place(self, w: _Work, tag: str, phase: str) -> bool:
        """Place the remainder at the phase's price; False if nothing valid to place."""
        it, r = w.intent, w.rules
        price = self._price(it.symbol, it.side, phase)
        if w.result.requested == 0 and it.notional is not None:  # first placement of a buy
            w.result.requested = r.q_qty(it.notional / price)
        qty = r.q_qty(w.remaining)
        if not r.valid(qty, price):
            w.result.note = w.result.note or "below min size"
            return False
        w.n += 1
        cid = f"{tag}-{it.symbol}-{it.side[0]}{w.n}"
        for attempt in (1, 2):
            self._throttle()
            try:
                self.broker.place(it.symbol, it.side, qty, price, cid)
                break
            except WallexError as e:
                w.result.note = f"place error: {e}"
                if attempt == 2:
                    return False
                self.sleep(THROUGH_WAIT_S)
        w.cid, w.price, w.placed_at, w.phase = cid, price, self.clock(), phase
        w.result.orders.append(cid)
        w.result.crossed = w.result.crossed or phase == "through"
        return True

    def _absorb(self, w: _Work, st: OrderState) -> None:
        """Fold a closed order's fills into the result (VWAP-weighted)."""
        if st.filled > 0:
            tot = w.result.filled + st.filled
            w.result.vwap = (w.result.vwap * w.result.filled + st.vwap * st.filled) / tot
            w.result.filled, w.result.fee = tot, w.result.fee + st.fee
        w.cid = None

    def _close(self, w: _Work) -> None:
        if w.cid is not None:
            self._absorb(w, self.broker.cancel(w.cid))

    # -- main loop --
    def execute(
        self,
        intents: list[Intent],
        *,
        tag: str,
        allow_cross: bool = True,
        mode: str | None = None,
        cycle_start: float | None = None,
    ) -> list[ExecResult]:
        """Work all intents in parallel until filled, crossed, or given up."""
        mode = mode or self.mode
        start = self.clock() if cycle_start is None else cycle_start
        works: list[_Work] = []
        for it in intents:
            res = ExecResult(it.symbol, it.side, it.qty or Decimal(0))
            w = _Work(it, res, self.rules[it.symbol], mode)
            works.append(w)
            if not self._place(w, tag, mode):
                w.done = True
        while not all(w.done for w in works):
            for w in works:
                if w.done:
                    continue
                st = self.broker.status(w.cid) if w.cid else None
                if st is not None and not st.active:
                    self._absorb(w, st)
                    if w.remaining <= 0 or not self._place(w, tag, w.phase):
                        w.done = True
                    continue
                now = self.clock()
                if w.phase == "maker":
                    if now - start >= self.cross_at_s:
                        self._close(w)
                        if w.remaining <= 0 or not allow_cross:
                            w.result.note = w.result.note or "maker deadline, no cross"
                            w.done = True
                        elif not self._place(w, tag, "through"):
                            w.done = True
                    elif now - w.placed_at >= self.repeg_s:
                        if self._price(w.intent.symbol, w.intent.side, "maker") != w.price:
                            self._close(w)
                            if w.remaining <= 0 or not self._place(w, tag, "maker"):
                                w.done = True
                else:  # through: wait, then one retry at the new price
                    if now - w.placed_at >= self.through_wait_s:
                        self._close(w)
                        w.through_tries += 1
                        if (
                            w.remaining <= 0
                            or w.through_tries >= 2
                            or not self._place(w, tag, "through")
                        ):
                            w.done = True
            if not all(w.done for w in works):
                self.sleep(self.poll_s)
        return [w.result for w in works]
