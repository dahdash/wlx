"""Deterministic entry gate and exits; the analyst never overrides them."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pandas as pd

from wallex.exec import MarketRules
from wallex.oil.analyst import _ts
from wallex.oil.config import SYMBOL, OilConfig, _last_entry, round_trip_cost


def gate(
    dec: dict[str, Any],
    book: dict[str, Any],
    book_age_s: float,
    candle_open: pd.Timestamp | None,
    g_state: dict[str, Any],
    st: dict[str, Any],
    cfg: OilConfig,
    rules: MarketRules,
    now: pd.Timestamp,
    symbol: str = SYMBOL,
    gap_4h: float | None = None,
) -> tuple[bool, list[str], Decimal]:
    """Entry check on a fresh book → (ok, reasons for rejection, quantised qty).

    ``gap_4h`` = WTI perp 4-h return − USOON 4-h return; unknown fails closed. Only a WTI lead
    of at least ``min_gap_4h`` beat the round-trip cost on 58 d of history (T14.7)."""
    now = now.tz_localize(None) if now.tzinfo else now
    why: list[str] = []
    if symbol != SYMBOL:
        why.append(f"symbol {symbol} not allowed")
    if dec["decision"] != "BUY_CANDIDATE":
        why.append(f"decision {dec['decision']}")
    if st.get("killed"):
        why.append("kill switch")
    if st.get("paused_reason"):
        why.append(f"paused: {st['paused_reason']}")
    if st.get("position"):
        why.append("position already open")
    hid = dec.get("hypothesis_id")
    if hid and hid in st.get("entered", []):
        why.append(f"hypothesis {hid} already traded")
    if g_state.get("all_off") is not True:
        why.append(f"G not all gates off ({g_state.get('status')})")
    day = st.get("day", {})
    if day.get("date") == now.date().isoformat():
        if day.get("entries", 0) >= cfg.max_entries_day:
            why.append("daily entry limit")
        if day.get("pnl", 0.0) <= -cfg.day_loss_cap:
            why.append("daily loss cap")
    if st.get("loss_streak", 0) >= cfg.max_losses:
        why.append("loss streak limit")
    if not cfg.entry_hours_utc[0] <= now.hour < cfg.entry_hours_utc[1]:
        why.append(f"outside entry hours {cfg.entry_hours_utc} UTC")
    last = _last_entry(cfg)
    if last and now >= pd.Timestamp(last):
        why.append(f"too close to contest end (no entries after {last})")
    if book_age_s > cfg.max_quote_age_s:
        why.append(f"stale quote {book_age_s:.0f} s")
    if candle_open is None or (now - candle_open).total_seconds() > cfg.max_candle_age_s:
        why.append("stale or missing candles")
    spread = book.get("spread", 1.0)
    if spread > cfg.max_spread:
        why.append(f"spread {spread:.3%} > {cfg.max_spread:.2%}")
    need = cfg.notional * cfg.depth_mult
    if min(book.get("bid_depth_usdt", 0), book.get("ask_depth_usdt", 0)) < need:
        why.append(f"depth < {need:.1f} USDT within {cfg.depth_band:.1%}")
    h = dec.get("horizon_hours")
    if not isinstance(h, (int, float)) or not cfg.min_horizon_h <= h <= cfg.max_hold_h:
        why.append(f"horizon {h!r} outside [{cfg.min_horizon_h}, {cfg.max_hold_h}] h")
    exp = _ts(dec.get("expiry_utc"))
    if exp is None or exp <= now or exp > now + pd.Timedelta(hours=cfg.max_hold_h + 0.5):
        why.append(f"expiry {dec.get('expiry_utc')!r} invalid")
    elif cfg.end_utc and exp > pd.Timestamp(cfg.end_utc) + pd.Timedelta(hours=1):
        why.append(f"expiry {dec.get('expiry_utc')!r} invalid")
    if not dec.get("supporting_evidence_ids"):
        why.append("no supporting evidence")
    bad = set(dec.get("lookahead_or_unsourced", [])) & set(dec.get("supporting_evidence_ids", []))
    if bad:
        why.append(f"look-ahead or unsourced evidence {sorted(bad)}")
    if not dec.get("invalidation_conditions"):
        why.append("no invalidation conditions")
    gross = dec.get("estimated_gross_opportunity_pct")
    cost = round_trip_cost(spread, cfg.entry_kind)
    if gross is None:
        why.append("no gross estimate")
    elif gross / 100 - cost < cfg.min_net_edge:
        why.append(f"net edge {gross / 100 - cost:.3%} < {cfg.min_net_edge:.2%} (cost {cost:.3%})")
    if gap_4h is None:
        why.append("WTI−USOON 4h gap unknown")
    elif gap_4h < cfg.min_gap_4h:
        why.append(f"WTI−USOON 4h gap {gap_4h:.2%} < {cfg.min_gap_4h:.1%}")
    qty = Decimal(0)
    if "bid" in book:
        price = rules.q_price(book["bid"], "BUY")
        qty = rules.q_qty(Decimal(str(cfg.notional)) / Decimal(str(book["ask"])))
        if not rules.valid(qty, price):
            why.append(f"size {qty} @ {price} below market minimum")
    return not why, why, qty


def exit_reason(pos: dict[str, Any], bid: float, now: pd.Timestamp, cfg: OilConfig) -> str | None:
    """Deterministic exits; they never wait for Claude."""
    now = now.tz_localize(None) if now.tzinfo else now
    if bid <= pos["entry"] * (1 - cfg.stop_pct):
        return f"stop: bid {bid} ≤ entry {pos['entry']} − {cfg.stop_pct:.1%}"
    if now >= pd.Timestamp(pos["opened_utc"]) + pd.Timedelta(hours=cfg.max_hold_h):
        return f"max hold {cfg.max_hold_h} h"
    exp = _ts(pos.get("expiry_utc"))
    if exp is not None and now >= exp:
        return f"thesis expiry {pos['expiry_utc']}"
    if cfg.end_utc and now >= pd.Timestamp(cfg.end_utc):
        return f"contest end {cfg.end_utc}"
    return None
