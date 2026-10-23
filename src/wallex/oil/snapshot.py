"""As-of snapshot packet for the analyst."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from wallex.client import WallexError
from wallex.oil.calendar import calendar
from wallex.oil.config import SYMBOL, OilConfig, _last_entry, round_trip_cost
from wallex.oil.feeds import Feeds, _summary, book_stats, read_g
from wallex.oil.memory import track_record, update_evals


def snapshot(
    feeds: Feeds,
    now: pd.Timestamp,
    state: dict[str, Any],
    cfg: OilConfig,
    g_run: str | None,
    state_dir: Path | None = None,
    memory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """As-of packet for the agent; every value carries a source and a timestamp. ``memory``:
    the shared lessons and all-run stats (:meth:`Memory.view`), shown in track_record."""
    end = int(now.timestamp())
    missing: list[str] = ["USO ETF quote (no keyless feed)", "EIA API (no key; use the web)"]
    snap: dict[str, Any] = {"as_of_utc": now.isoformat(), "symbol": SYMBOL}
    try:
        b = book_stats(feeds.depth(), cfg.depth_band)
        snap["usoon_book"] = b | {"source": "wallex /v1/depth", "fetched_utc": now.isoformat()}
        snap["est_round_trip_cost"] = round(round_trip_cost(b["spread"], cfg.entry_kind), 5)
    except (WallexError, httpx.HTTPError, KeyError, ValueError) as e:
        snap["usoon_book"] = {"status": f"unavailable: {e}"}
        missing.append("USOON order book")
    try:
        t, c = feeds.usoon_15m(end)
        snap["usoon_15m"] = _summary(t, c, end) | {"source": "wallex /v1/udf/history res 15"}
    except (WallexError, httpx.HTTPError, KeyError, ValueError) as e:
        snap["usoon_15m"] = {"status": f"unavailable: {e}"}
        missing.append("USOON candles")
    try:
        t, c, mark = feeds.ref_15m()
        snap["wti_perp_15m"] = _summary(t, c, end) | {"mark": mark, "source": "binance fapi CLUSDT"}
    except (httpx.HTTPError, KeyError, ValueError) as e:
        snap["wti_perp_15m"] = {"status": f"unavailable: {e}"}
        missing.append("WTI perp")
    u, w = snap["usoon_15m"], snap["wti_perp_15m"]
    snap["divergence_usoon_minus_wti"] = {
        k: round(u[k] - w[k], 5) if u.get(k) is not None and w.get(k) is not None else None
        for k in ("ret_1h", "ret_4h", "ret_24h")
    } | {"note": "hourly corr USOON vs CL perp ≈ +0.59 over 30 d (docs/commod-slots.md §5)"}
    snap["calendar"] = calendar(now, cfg)
    snap["strategy_g"] = read_g(g_run, now, cfg.max_g_age_s, state_dir)
    snap["position"] = state.get("position")
    snap["active_hypotheses"] = [
        {k: h[k] for k in ("id", "thesis", "decision", "invalidation", "expiry_utc", "status")}
        for h in state.get("hypotheses", {}).values()
        if h["status"] == "active"
    ][-3:]
    snap["limits"] = {
        "notional_usdt": cfg.notional,
        "horizon_h": [cfg.min_horizon_h, cfg.max_hold_h],
        "stop_pct": cfg.stop_pct,
        "min_net_edge": cfg.min_net_edge,
        "min_wti_minus_usoon_gap_4h": cfg.min_gap_4h,
        "entry_fill": cfg.entry_kind,
        "no_entries_after_utc": _last_entry(cfg),
        "flat_by_utc": cfg.end_utc,
    }
    if "bid" in snap["usoon_book"]:  # score past decisions against this quote first
        update_evals(state.setdefault("evals", []), snap["usoon_book"]["bid"], now)
    snap["track_record"] = track_record(state, memory)
    snap["evidence_cache"] = {
        "note": f"items you found in the last {cfg.evidence_ttl_h:g} h; reuse, do not re-search",
        "items": evidence_cache(state, now, cfg),
    }
    snap["missing"] = missing
    cap_snapshot(snap, cfg.max_prompt_chars)
    body = json.dumps(snap, sort_keys=True, default=str)
    snap["snapshot_id"] = hashlib.sha256(body.encode()).hexdigest()[:16]
    return snap


# --- evidence cache (T14.3) -------------------------------------------------------------


def _naive(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.tz_convert("UTC").tz_localize(None) if ts.tzinfo else ts


def remember_evidence(state: dict[str, Any], dec: dict[str, Any], now: pd.Timestamp,
                      cfg: OilConfig) -> None:  # fmt: skip
    """Add the validated evidence of a decision to ``state["evidence_cache"]`` (one entry per
    URL, the latest sighting wins), then drop entries older than ``cfg.evidence_ttl_h``."""
    cache = {e["url"]: e for e in state.setdefault("evidence_cache", [])}
    seen = _naive(now).isoformat()
    for e in dec.get("evidence") or []:
        if not e.get("source_url"):
            continue
        cache.pop(e["source_url"], None)  # re-insert: newest last
        cache[e["source_url"]] = {
            "url": e["source_url"], "published_utc": e.get("published_utc"),
            "kind": e.get("kind"), "claim": (e.get("title") or "")[:200], "seen_utc": seen,
            "decision": dec.get("decision"),
        }  # fmt: skip
    state["evidence_cache"] = evidence_cache({"evidence_cache": list(cache.values())}, now, cfg)


def evidence_cache(state: dict[str, Any], now: pd.Timestamp, cfg: OilConfig) -> list[dict]:
    """Entries seen within the TTL, at most ``cfg.max_evidence``, oldest first."""
    lo = _naive(now) - pd.Timedelta(hours=cfg.evidence_ttl_h)
    items = [e for e in state.get("evidence_cache", []) if pd.Timestamp(e["seen_utc"]) >= lo]
    return items[-cfg.max_evidence :]


# --- prompt-size cap (T14.3) ------------------------------------------------------------

# (path, keep at least) in trim order: the oldest, least decision-relevant context goes first.
TRIM = (
    (("evidence_cache", "items"), 0),
    (("track_record", "recent_decisions"), 2),
    (("track_record", "closed_trades"), 1),
    (("track_record", "lessons"), 2),
    (("calendar", "events"), 4),
    (("active_hypotheses",), 1),
    (("track_record", "recent_decisions"), 0),
    (("track_record", "closed_trades"), 0),
    (("track_record", "lessons"), 0),
)


def prompt_chars(snap: dict[str, Any]) -> int:
    """Length of the snapshot as the agent prompt renders it."""
    return len(json.dumps(snap, indent=1, default=str))


def cap_snapshot(snap: dict[str, Any], limit: int) -> dict[str, Any]:
    """Trim list entries (oldest first, per :data:`TRIM`) until the rendered snapshot fits in
    ``limit`` characters; what was cut is listed in ``snap["truncated"]``. Market data,
    limits, position and G gates are never trimmed."""
    cut: dict[str, int] = {}
    for path, keep in TRIM:
        if prompt_chars(snap) <= limit:
            break
        parent = snap
        for k in path[:-1]:
            parent = parent.get(k) if isinstance(parent, dict) else None
        xs = parent.get(path[-1]) if isinstance(parent, dict) else None
        if not isinstance(xs, list):
            continue
        name = ".".join(path)
        while len(xs) > keep and prompt_chars(snap) > limit:
            xs.pop(0)
            cut[name] = cut.get(name, 0) + 1
    if cut:
        snap["truncated"] = {"limit_chars": limit, "dropped_oldest": cut}
    return snap
