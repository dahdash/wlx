"""Risk rules (docs/plan.md §1–2): hard bounds that the AI officer may only tighten.

Pure state + rules, no network except :func:`flatten`, which uses an :class:`Executor`.
Bars are cycle indices (60-min by default; the G runner scales the windows to 30-min);
equity is in USDT.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

import numpy as np

from wallex.exec import Executor, Intent


@dataclass(frozen=True)
class RiskConfig:
    stop: float | None = 0.02  # per-slot stop from entry VWAP; None = no stop
    cooldown_bars: int = 4
    crash_k: float = 1.5  # 24-bar log return < −k·σ24 → w = 0
    crash_bars: int = 24
    vol_bars: int = 72
    pause_loss: float = 0.04  # rolling 6-h loss ≥ 4 % → exposure 0 for 2 h
    pause_window_bars: int = 6
    pause_bars: int = 2
    kill_dd: float = 0.10  # equity ≤ 0.90 × run peak → flatten and stop
    max_pos_frac: float = 0.20
    max_slots: int = 5


@dataclass
class Slot:
    qty: Decimal = Decimal(0)
    entry_vwap: Decimal = Decimal(0)
    cooldown_until: int = -1  # bar index; trading allowed when bar > cooldown_until
    stop: float | None = None  # AI-tightened stop (pct ≥ stop config), None = default

    def add_fill(self, side: str, qty: Decimal, price: Decimal) -> None:
        if side.upper() == "BUY":
            tot = self.qty + qty
            self.entry_vwap = (self.entry_vwap * self.qty + price * qty) / tot if tot else price
            self.qty = tot
        else:
            self.qty = max(Decimal(0), self.qty - qty)
            if self.qty == 0:
                self.entry_vwap = Decimal(0)


@dataclass
class RiskState:
    cfg: RiskConfig = field(default_factory=RiskConfig)
    slots: dict[str, Slot] = field(default_factory=dict)
    peak_equity: float = 0.0
    equity_hist: list[tuple[int, float]] = field(default_factory=list)  # (bar, equity)
    paused_until: int = -1
    killed: bool = False
    events: list[str] = field(default_factory=list)

    def slot(self, symbol: str) -> Slot:
        return self.slots.setdefault(symbol, Slot())

    # -- rules --
    def crash_gate(self, closes: np.ndarray) -> bool:
        """True while the symbol's 24-bar log return < −k·σ24 (σ from 72 one-bar returns)."""
        c = self.cfg
        if len(closes) < c.vol_bars + 1:
            return False
        lr = np.diff(np.log(np.asarray(closes, dtype=float)))
        sigma = float(np.std(lr[-c.vol_bars :])) * np.sqrt(c.crash_bars)
        return bool(lr[-c.crash_bars :].sum() < -c.crash_k * sigma)

    def check_stops(self, prices: dict[str, Decimal], bar: int) -> list[str]:
        """Symbols whose price fell ≥ stop below entry VWAP; starts their cool-down."""
        hit = []
        for sym, s in self.slots.items():
            if s.qty <= 0 or sym not in prices:
                continue
            stop = s.stop if s.stop is not None else self.cfg.stop
            if stop is None:
                continue
            if prices[sym] <= s.entry_vwap * (1 - Decimal(str(stop))):
                s.cooldown_until = bar + self.cfg.cooldown_bars
                hit.append(sym)
                self.events.append(f"bar {bar}: stop {sym} at {prices[sym]} (entry {s.entry_vwap})")
        return hit

    def update_equity(self, bar: int, equity: float) -> str | None:
        """Record equity; returns 'kill', 'pause' or None."""
        c = self.cfg
        self.peak_equity = max(self.peak_equity, equity)
        self.equity_hist.append((bar, equity))
        if equity <= self.peak_equity * (1 - c.kill_dd):
            self.killed = True
            self.events.append(f"bar {bar}: KILL equity {equity:.2f} peak {self.peak_equity:.2f}")
            return "kill"
        past = [e for b, e in self.equity_hist if b <= bar - c.pause_window_bars]
        if past and equity <= past[-1] * (1 - c.pause_loss) and bar > self.paused_until:
            self.paused_until = bar + c.pause_bars
            self.events.append(f"bar {bar}: PAUSE until {self.paused_until}")
            return "pause"
        return None

    def tighten_stop(self, symbol: str, pct: float) -> None:
        """AI may only tighten: stop = min(current, pct), never below 1 %."""
        s = self.slot(symbol)
        cur = s.stop if s.stop is not None else self.cfg.stop
        s.stop = max(0.01, pct if cur is None else min(cur, pct))

    def apply(
        self,
        targets: dict[str, float],
        equity: float,
        bar: int,
        *,
        crash: dict[str, bool] | None = None,
        paused_symbols: set[str] | None = None,
    ) -> dict[str, float]:
        """Clamp target notionals: zero when killed/paused/cooldown/crash, cap per slot."""
        out: dict[str, float] = {}
        cap = equity * self.cfg.max_pos_frac
        for sym, t in list(targets.items())[: self.cfg.max_slots]:
            if (
                self.killed
                or bar <= self.paused_until
                or bar <= self.slot(sym).cooldown_until
                or (crash or {}).get(sym, False)
                or sym in (paused_symbols or set())
            ):
                out[sym] = 0.0
            else:
                out[sym] = float(min(max(t, 0.0), cap))
        return out


def flatten(
    executor: Executor,
    holdings: dict[str, Decimal],
    *,
    tag: str,
    open_order_ids: tuple[str, ...] = (),
) -> list:
    """Cancel open orders and sell every holding to USDT, crossing the book (plan §4/§5)."""
    for cid in open_order_ids:
        executor.broker.cancel(cid)
    intents = [Intent(sym, "SELL", qty=q) for sym, q in holdings.items() if q > 0]
    if not intents:
        return []
    return executor.execute(intents, tag=tag, mode="through", allow_cross=True)
