"""Strategy G (R6 decision, docs/validation-sma.md): 30-min SMA-cross ensemble with a band
and a 28-day trend gate, long-only spot, one equal slot per coin.

Per coin: six fast×slow SMA pairs on 30-min closes vote; a vote turns on when
SMA_fast > SMA_slow·(1+band), off when SMA_fast < SMA_slow·(1−band), else keeps its state.
Weight = votes / 6 while the coin's 28-day return (daily close 00:00 UTC) is > 0, else 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

COINS = ["BTC", "ETH", "XRP", "SOL", "BNB"]
PAIRS = [(24, 72), (36, 108), (48, 144), (72, 192), (96, 240), (144, 336)]
BAND = 0.003
GATE_DAYS = 28
BAR = pd.Timedelta("30min")


def votes(closes: pd.Series, pairs=PAIRS, band: float = BAND) -> pd.DataFrame:
    """One 0/1 column per pair (hysteresis state), 0 while the slow SMA is not yet defined."""
    out = {}
    for f, s in pairs:
        fast, slow = closes.rolling(f).mean(), closes.rolling(s).mean()
        state = np.where(fast > slow * (1 + band), 1.0, np.nan)
        state = np.where(fast < slow * (1 - band), 0.0, state)
        vote = pd.Series(state, index=closes.index).ffill().fillna(0.0)
        out[f"{f}/{s}"] = vote.where(slow.notna(), 0.0)
    return pd.DataFrame(out)


def gate(hourly: pd.Series, now: pd.Timestamp, days: int = GATE_DAYS) -> tuple[bool, float] | None:
    """28-day return at the latest completed 00:00 UTC close, from hourly bars labelled by
    open time (the bar labelled 23:00 closes at 00:00). None if either close is missing."""
    day = now.floor("1D")
    ref = day - pd.Timedelta(days=days)
    h = pd.Timedelta("1h")
    if day - h not in hourly.index or ref - h not in hourly.index:
        return None
    ret = float(hourly[day - h] / hourly[ref - h] - 1)
    return ret > 0, ret


@dataclass
class Signal:
    coin: str
    votes: dict[str, float]
    w_sma: float
    gate_on: bool
    gate_ret: float | None
    weight: float

    def reason(self) -> str:
        on = int(sum(self.votes.values()))
        g = "n/a" if self.gate_ret is None else f"{self.gate_ret:+.1%}"
        tail = "" if self.gate_on else " → gate off, w=0"
        return f"{self.coin}: {on}/{len(self.votes)} SMA pairs up, 28d {g}{tail}"


def signal(coin: str, closes30: pd.Series, gate_on: bool, gate_ret: float | None) -> Signal:
    v = votes(closes30).iloc[-1].to_dict()
    w = sum(v.values()) / len(v)
    return Signal(coin, v, w, gate_on, gate_ret, w if gate_on else 0.0)
