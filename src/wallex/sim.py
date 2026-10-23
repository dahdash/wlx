"""Backtest core for strategy G: per-slot simulation, costs, metrics and the circular null.

Model (one equal slot per coin, long-only spot, weight 0…1 per bar):
- weight ``w_t`` decided at the close of bar ``t``, traded at that close, earns bar ``t+1``;
- trade only if ``|target − held| ≥ max(minNotional / slot, hysteresis)``; a full exit
  (target 0) always executes;
- cost per side is charged on the traded slot fraction;
- optional hard stop: previous close below entry VWAP × (1 − stop) → sell all, cool down.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from wallex import data

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache"
N5 = ["BTC", "ETH", "XRP", "SOL", "BNB"]
START, END = "2020-09-01", "2026-09-30 23:00"
SPANS = {"discovery": ("2020-09-01", "2023-12-31 23:00"), "holdout": ("2024-01-01", END)}
TAKER_FEE = 0.0013
HALF_SPREAD = {"BTC": 0.0006, "ETH": 0.00075}  # others: 0.3 % spread → 0.15 % half
DEFAULT_HALF_SPREAD = 0.0015
COSTS = {"frictionless": None, "maker 0.10%": 0.0010, "taker": "taker", "stress 0.20%": 0.0020}
HOURS_PER_YEAR = 24 * 365
N_PERM = 150


def cost_of(row, coin: str) -> float:
    if row is None:
        return 0.0
    if row == "taker":
        return TAKER_FEE + HALF_SPREAD.get(coin, DEFAULT_HALF_SPREAD)
    return float(row)


def load(interval: int = 30) -> dict[str, pd.DataFrame]:
    """N5 bars on a common index, 30-day warm-up before ``START``; cached in ``data/cache``."""
    CACHE.mkdir(parents=True, exist_ok=True)
    bars = {}
    for c in N5:
        f = CACHE / f"{c}USDT_{interval}.parquet"
        if not f.exists():
            data.load_bars(c, interval).to_parquet(f)
        bars[c] = pd.read_parquet(f)
    common = None
    for b in bars.values():
        common = b.index if common is None else common.intersection(b.index)
    warm = pd.Timestamp(START) - pd.Timedelta(days=30)
    common = common[(common >= warm) & (common <= pd.Timestamp(END) + pd.Timedelta("45min"))]
    return {c: b.loc[common] for c, b in bars.items()}


def tsmom_book(closes: pd.Series, lookback: int) -> np.ndarray:
    """Hold while the ``lookback``-bar return is > 0, re-checked once a day (bars at hour 23)."""
    up = (closes / closes.shift(lookback) > 1).astype(float)
    daily = up.where(up.index.hour == 23).ffill().fillna(0.0)
    return daily.to_numpy()


# ---------------------------------------------------------------- simulation + metrics


@dataclass(frozen=True)
class Run:
    hysteresis: float = 0.0
    stop: float | None = None
    cooldown: int = 4
    slot_usdt: float = 3.7
    min_notional: float = 1.0

    @property
    def threshold(self) -> float:
        return max(self.min_notional / self.slot_usdt, self.hysteresis)


def simulate_slot(
    closes: np.ndarray, weights: np.ndarray, run: Run, cost: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return (slot value per bar starting at 1.0, trade flag per bar)."""
    n = len(closes)
    val = np.ones(n)
    traded = np.zeros(n, dtype=bool)
    h, entry, cool, v = 0.0, 0.0, 0, 1.0
    for t in range(1, n):
        p_prev = closes[t - 1]
        if cool > 0:
            cool -= 1
            tgt = 0.0
        else:
            tgt = weights[t - 1]
            tgt = 0.0 if np.isnan(tgt) else tgt
        if h > 0 and run.stop is not None and p_prev < entry * (1 - run.stop):
            tgt, cool = 0.0, run.cooldown
        d = tgt - h
        if abs(d) >= run.threshold or (tgt == 0.0 and h > 0):
            v *= 1 - abs(d) * cost
            if d > 0:
                entry = (h * entry + d * p_prev) / (h + d)
            h = tgt
            traded[t] = True
        v *= 1 + h * (closes[t] / p_prev - 1)
        val[t] = v
    return val, traded


def simulate(bars, books, run: Run, cost_row) -> pd.DataFrame:
    """Net return per bar per slot (columns = coins); trade flags in ``.attrs["trades"]``."""
    rets, trades = {}, {}
    for c, b in bars.items():
        v, tr = simulate_slot(b["close"].to_numpy(), books[c], run, cost_of(cost_row, c))
        rets[c] = np.r_[0.0, v[1:] / v[:-1] - 1]
        trades[c] = tr
    index = next(iter(bars.values())).index
    out = pd.DataFrame(rets, index=index)
    out.attrs["trades"] = pd.DataFrame(trades, index=index)
    return out


def stats(slot_rets: pd.DataFrame, span: tuple[str, str], bph: int = 1) -> dict[str, float]:
    """Metrics on ``span``: portfolio = equal slots re-equalised every bar (no cost).

    ``bph`` = bars per hour (2 for 30-min bars); ``mean_h`` is per bar."""
    r = slot_rets.loc[span[0] : span[1]]
    tr = slot_rets.attrs["trades"].loc[span[0] : span[1]]
    p = r.mean(axis=1)
    eq = (1 + p).cumprod()
    years = len(p) / (HOURS_PER_YEAR * bph)
    # rolling 36-h windows, 1-h step, slots restart equal (same as the live run)
    g = (1 + r).cumprod()
    w = (g.shift(-36 * bph) / g).iloc[::bph].mean(axis=1).dropna() - 1
    return {
        "total": eq.iloc[-1] - 1,
        "cagr": eq.iloc[-1] ** (1 / years) - 1,
        "sharpe": p.mean() / p.std() * np.sqrt(HOURS_PER_YEAR * bph) if p.std() > 0 else 0.0,
        "mdd": float((1 - eq / eq.cummax()).max()),
        "mean36": w.mean(),
        "median36": w.median(),
        "win36": (w > 0).mean(),
        "p10_36": w.quantile(0.10),
        "trades36": tr.sum().sum() / len(p) * 36 * bph,
        "mean_h": p.mean(),
        "exposure": float((r != 0).mean().mean()),
    }


def circular_null(bars, books, run, cost_row, span, n=N_PERM, seed=7, bph=1) -> np.ndarray:
    """Mean per-bar net return on ``span`` with each coin's weight book circularly shifted
    by ≥ 30 days (keeps exposure and trade count, breaks the timing)."""
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for i in range(n):
        shifted = {}
        for c, w in books.items():
            k = int(rng.integers(24 * 30 * bph, len(w) - 24 * 30 * bph))
            shifted[c] = np.roll(w, k)
        out[i] = stats(simulate(bars, shifted, run, cost_row), span, bph)["mean_h"]
    return out


def p_value(real: float, null: np.ndarray) -> float:
    return (1 + int((null >= real).sum())) / (1 + len(null))


def pct(x: float) -> str:
    return f"{x:+.2%}"


HEAD = (
    "| span | cost / side | total | CAGR | Sharpe | MDD | mean 36 h | median 36 h | win 36 h |"
    " p10 36 h | trades / 36 h |\n| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    " ---: | ---: |"
)


def row(span: str, cost: str, s: dict[str, float]) -> str:
    return (
        f"| {span} | {cost} | {pct(s['total'])} | {pct(s['cagr'])} | {s['sharpe']:.2f} |"
        f" {s['mdd']:.1%} | {pct(s['mean36'])} | {pct(s['median36'])} | {s['win36']:.0%} |"
        f" {pct(s['p10_36'])} | {s['trades36']:.1f} |"
    )
