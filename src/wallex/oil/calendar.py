"""Deterministic oil calendar for the snapshot: scheduled data releases, US holidays and US
equity hours, computed from rules in New York time (DST-safe). No network; the agent is told
to check on the web whether a release moved."""

from __future__ import annotations

from datetime import date, time, timedelta
from typing import Any

import pandas as pd

from wallex.oil.config import CONTEST_END_UTC, NY, OilConfig

# US federal holidays (EIA/API shift) and NYSE closures / early closes, 2026–2027.
FEDERAL = {
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 5, 25),
    date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7), date(2026, 10, 12),
    date(2026, 11, 11), date(2026, 11, 26), date(2026, 12, 25),
    date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15), date(2027, 5, 31),
}  # fmt: skip
NYSE_CLOSED = {
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
    date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7),
    date(2026, 11, 26), date(2026, 12, 25),
    date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15), date(2027, 3, 26),
}  # fmt: skip
NYSE_EARLY = {date(2026, 11, 27), date(2026, 12, 24)}  # 13:00 New York close
OPEC_NOTE = (
    "OPEC+ eight-country monthly meeting: usually the first Sunday of the month in 2025; "
    "2026 dates NOT verified, confirm on the web"
)


def _utc(d: date, t: time) -> pd.Timestamp:
    """New York wall time → naive UTC (the convention of the rest of the package)."""
    return pd.Timestamp.combine(d, t).tz_localize(NY).tz_convert("UTC").tz_localize(None)


def _naive(now: pd.Timestamp) -> pd.Timestamp:
    return now.tz_convert("UTC").tz_localize(None) if now.tzinfo else now


def _weekly(d: date, weekday: int, t: time) -> tuple[date, time, bool]:
    """Release day of the week of ``d``: one day later when a federal holiday falls on
    Monday..release day of that week (the usual EIA/API rule)."""
    monday = d - timedelta(days=d.weekday())
    day = monday + timedelta(days=weekday)
    shifted = any(monday + timedelta(days=i) in FEDERAL for i in range(weekday + 1))
    return (day + timedelta(days=1) if shifted else day), t, shifted


def nyse_hours(d: date) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """(open, close) in naive UTC for a New York trading day, None when the market is shut."""
    if d.weekday() >= 5 or d in NYSE_CLOSED:
        return None
    return _utc(d, time(9, 30)), _utc(d, time(13 if d in NYSE_EARLY else 16))


def us_session_open(now: pd.Timestamp) -> bool:
    now = _naive(now)
    h = nyse_hours(now.tz_localize("UTC").tz_convert(NY).date())
    return bool(h and h[0] <= now < h[1])


def events(now: pd.Timestamp, back_h: float = 24, ahead_h: float = 48) -> list[dict[str, Any]]:
    """Scheduled events in [now − back_h, now + ahead_h], sorted by time."""
    now = _naive(now)
    lo, hi = now - pd.Timedelta(hours=back_h), now + pd.Timedelta(hours=ahead_h)
    out: list[dict[str, Any]] = []

    def add(ts: pd.Timestamp, name: str, note: str = "") -> None:
        if lo <= ts <= hi and not any(e["name"] == name and e["ts"] == ts for e in out):
            out.append({"ts": ts, "name": name, "note": note})

    d0 = (lo - pd.Timedelta(days=1)).date()
    for i in range((hi - lo).days + 3):
        d = d0 + timedelta(days=i)
        for wd, t, name in (
            (1, time(16, 30), "API weekly crude inventories"),
            (2, time(10, 30), "EIA Weekly Petroleum Status Report"),
        ):
            day, at, shifted = _weekly(d, wd, t)
            add(_utc(day, at), name, "holiday week: moved one day (verify)" if shifted else "")
        if d.weekday() == 4 and d not in NYSE_CLOSED:
            add(_utc(d, time(13, 0)), "Baker Hughes US rig count")
        if d in FEDERAL or d in NYSE_CLOSED:
            flags = (("federal", d in FEDERAL), ("NYSE closed", d in NYSE_CLOSED))
            what = [w for w, on in flags if on]
            add(_utc(d, time(0, 0)), f"US holiday ({', '.join(what)})")
        if h := nyse_hours(d):
            early = " (early close)" if d in NYSE_EARLY else ""
            add(h[0], "US equity open (USO trades)")
            add(h[1], f"US equity close{early}")
        if d.weekday() == 6 and d.day <= 7:
            add(_utc(d, time(6, 0)), "OPEC+ monthly meeting (unverified)", OPEC_NOTE)
    out.sort(key=lambda e: e["ts"])
    return out


def calendar(now: pd.Timestamp, cfg: OilConfig) -> dict[str, Any]:
    """Snapshot block: events with hours from now (negative = past) and the contest clock."""
    now = _naive(now)

    def hours(ts: Any) -> float | None:
        return round((pd.Timestamp(ts) - now).total_seconds() / 3600, 2) if ts else None

    return {
        "source": "rules in wallex.oil.calendar (New York time, DST-aware); not fetched",
        "events": [
            {"time_utc": e["ts"].isoformat(), "in_hours": hours(e["ts"]), "name": e["name"]}
            | ({"note": e["note"]} if e["note"] else {})
            for e in events(now)
        ],
        "us_equity_session_open": us_session_open(now),
        "hours_to_flat_by": hours(cfg.end_utc),
        "hours_to_contest_end": hours(CONTEST_END_UTC),
    }
