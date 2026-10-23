"""Market data: Wallex USOON, Binance WTI perp, live G gates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from wallex.client import WallexClient, WallexError
from wallex.oil import config
from wallex.oil.config import BINANCE_FAPI, G_COINS, REF, SYMBOL

BAR_S = 900  # 15-min candles


def _ret(c: list[float], n: int) -> float | None:
    return round(c[-1] / c[-1 - n] - 1, 5) if len(c) > n and c[-1 - n] else None


def _summary(t: list[int], c: list[float], now_s: int) -> dict[str, Any]:
    """15-min closes → returns over 15 m / 1 h / 4 h / 24 h and hourly σ over 24 h, from
    closed bars only (open + 15 min ≤ now); ``last`` is the live price (forming bar)."""
    n = sum(1 for x in t if x + BAR_S <= now_s)
    if n < 2:
        return {"status": "unavailable"}
    live, t, c = c[-1], t[:n], c[:n]
    r = pd.Series(c).pct_change().dropna().tail(96)
    return {
        "last": live,
        "last_closed": c[-1],
        "last_bar_open_utc": pd.Timestamp(t[-1], unit="s").isoformat(),
        "ret_15m": _ret(c, 1),
        "ret_1h": _ret(c, 4),
        "ret_4h": _ret(c, 16),
        "ret_24h": _ret(c, 96),
        "sigma_1h_24h": round(float(r.std() * 2), 5) if len(r) > 8 else None,
    }


def book_stats(depth: dict[str, list[dict[str, Any]]], band: float) -> dict[str, float]:
    bids = sorted(((float(b["price"]), float(b["quantity"])) for b in depth["bid"]), reverse=True)
    asks = sorted((float(a["price"]), float(a["quantity"])) for a in depth["ask"])
    if not bids or not asks:
        raise WallexError("empty order book")
    bid, ask = bids[0][0], asks[0][0]
    mid = (bid + ask) / 2
    return {
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "spread": round((ask - bid) / mid, 6),
        "bid_depth_usdt": round(sum(p * q for p, q in bids if p >= bid * (1 - band)), 2),
        "ask_depth_usdt": round(sum(p * q for p, q in asks if p <= ask * (1 + band)), 2),
    }


def read_g(
    g_run: str | None, now: pd.Timestamp, max_age_s: float, state_dir: Path | None = None
) -> dict[str, Any]:
    """Last cycle of the live G run → gates; ``all_off`` is None when unknown (fail closed)."""
    if not g_run:
        return {"status": "unavailable", "all_off": None}
    path = (state_dir or config.STATE_DIR) / g_run / "cycles.jsonl"
    try:
        with path.open("rb") as f:
            f.seek(max(0, path.stat().st_size - 200_000))
            rec = json.loads(f.read().splitlines()[-1])
    except (OSError, IndexError, json.JSONDecodeError) as e:
        return {"status": f"unavailable: {type(e).__name__}", "all_off": None}
    close = pd.Timestamp(rec["bar_close"]).tz_localize(None)
    age = (now.tz_localize(None) - close).total_seconds()
    gates = {s: bool(rec["signals"][s]["gate_on"]) for s in G_COINS if s in rec["signals"]}
    fresh = age <= max_age_s and len(gates) == len(G_COINS)
    return {
        "run_id": g_run,
        "bar_close_utc": close.isoformat(),
        "age_s": round(age),
        "gates_on": gates,
        "gate_ret_28d": {s: rec["signals"][s].get("gate_ret") for s in gates},
        "equity_usdt": rec.get("equity_after"),
        "all_off": (not any(gates.values())) if fresh else None,
        "status": "ok" if fresh else "stale",
    }


class Feeds:
    """Market data: Wallex (USOON) + Binance futures (WTI perp). Injectable for tests."""

    def __init__(self, client: WallexClient, http: httpx.Client | None = None):
        self.client = client
        self.http = http or httpx.Client(base_url=BINANCE_FAPI, timeout=10)

    def depth(self) -> dict[str, Any]:
        return self.client.depth(SYMBOL)

    def usoon_15m(self, end: int) -> tuple[list[int], list[float]]:
        r = self.client.candles(SYMBOL, "15", end - 26 * 3600, end)
        return [int(x) for x in r["t"]], [float(x) for x in r["c"]]

    def ref_15m(self) -> tuple[list[int], list[float], dict[str, Any]]:
        k = self.http.get(
            "/fapi/v1/klines", params={"symbol": REF, "interval": "15m", "limit": 100}
        )
        k.raise_for_status()
        p = self.http.get("/fapi/v1/premiumIndex", params={"symbol": REF})
        p.raise_for_status()
        rows = k.json()
        pi = p.json()
        mark = {
            "mark": float(pi["markPrice"]),
            "index": float(pi["indexPrice"]),
            "time_utc": pd.Timestamp(int(pi["time"]), unit="ms").isoformat(),
        }
        return [int(r[0]) // 1000 for r in rows], [float(r[4]) for r in rows], mark
