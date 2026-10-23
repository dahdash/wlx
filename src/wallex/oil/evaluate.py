"""Learning evaluation (T14.5): the agent's logged decisions and the no-LLM :class:`Rule`
baseline re-decided on the same snapshots, both scored offline against the USOON bid path
pooled from every run's ``snapshots.jsonl`` (so stopped runs keep being scored).

Per analyst: decision mix, forward net at 1/2/4/8 h per decision (a long bought at the ask
then, sold at the bid later, fees included), hit rate (BUY right when the 4-h net > 0, any
other decision right when it is ≤ 0), missed moves (no buy but a > 0.5 % net was there
within 8 h), WATCH counterfactual entries, and calibration of the gross estimate against the
realised best bid move within the stated horizon."""

from __future__ import annotations

import bisect
import json
from pathlib import Path
from typing import Any

import pandas as pd

from wallex.oil import config
from wallex.oil.analyst import AgentError, Analyst, Rule, no_trade, read_calls, validate
from wallex.oil.config import DECISIONS, HORIZONS
from wallex.oil.memory import forward_net
from wallex.oil.ops import read_jsonl

MISSED = 0.005  # a net move above this within 8 h counts as a missed opportunity
LATE_H = 0.5  # a horizon mark more than this after its target time is left unscored


def _naive(ts: Any) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_convert(None) if t.tzinfo else t


def price_path(state_dir: Path) -> tuple[list[pd.Timestamp], list[float]]:
    """Every (as-of, USOON bid) logged by any run in ``state_dir``, time-sorted, deduped."""
    pts: dict[pd.Timestamp, float] = {}
    for d in state_dir.iterdir():
        if d.is_dir() and any(d.glob("snapshots*.jsonl")):
            for s in read_jsonl(d, "snapshots.jsonl"):
                bid = (s.get("usoon_book") or {}).get("bid")
                if bid and s.get("as_of_utc"):
                    pts[_naive(s["as_of_utc"])] = float(bid)
    ts = sorted(pts)
    return ts, [pts[t] for t in ts]


def score(ts: pd.Timestamp, ask: float, bid: float, entry: str, path, horizon_h=None):
    """Forward marks of one decision: net per horizon, best/worst net within 8 h and the
    best bid move within ``horizon_h`` (or 8 h) for calibration."""
    times, bids = path
    lo = bisect.bisect_right(times, ts)
    end = ts + pd.Timedelta(hours=max(HORIZONS))
    hi = bisect.bisect_right(times, end)
    out: dict[str, Any] = {"net": {}}
    for h in HORIZONS:
        i = bisect.bisect_left(times, ts + pd.Timedelta(hours=h), lo)
        if i < len(times) and times[i] <= ts + pd.Timedelta(hours=h + LATE_H):
            out["net"][str(h)] = forward_net(ask, bids[i], entry)
    window = bids[lo:hi]
    if window:
        nets = [forward_net(ask, b, entry) for b in window]
        out["best_net"], out["worst_net"] = max(nets), min(nets)
        cut = ts + pd.Timedelta(hours=horizon_h or max(HORIZONS))
        moves = [b / bid - 1 for t, b in zip(times[lo:hi], window, strict=True) if t <= cut]
        out["best_move"] = round(max(moves), 5) if moves else None
    return out


def _mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 5) if xs else None


def _right(decision: str, net4: float) -> bool:
    return net4 > 0 if decision == "BUY_CANDIDATE" else net4 <= 0


def stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Scores of one analyst's decisions (rows: decision, est_pct, horizon_h, marks)."""
    by: dict[str, Any] = {}
    for d in DECISIONS:
        xs = [r for r in rows if r["decision"] == d]
        if not xs:
            continue
        s4 = [r["net"]["4"] for r in xs if "4" in r["net"]]
        by[d] = {"n": len(xs), "n_scored_4h": len(s4),
                 "hit_rate_4h": round(sum(_right(d, x) for x in s4) / len(s4), 2) if s4 else None,
                 "mean_net": {str(h): _mean([r["net"][str(h)] for r in xs if str(h) in r["net"]])
                              for h in HORIZONS}}  # fmt: skip
    scored = [r for r in rows if "4" in r["net"]]
    missed = [r for r in rows if r["decision"] != "BUY_CANDIDATE"
              and (r.get("best_net") or 0) > MISSED]  # fmt: skip
    watch = [r for r in rows if r["decision"] == "WATCH"]
    cal = [r for r in rows if r.get("est_pct") is not None and r.get("best_move") is not None]
    err = [r["est_pct"] / 100 - r["best_move"] for r in cal]
    return {
        "decisions": {d: v["n"] for d, v in by.items()},
        "hit_rate_4h": round(sum(_right(r["decision"], r["net"]["4"]) for r in scored)
                             / len(scored), 2) if scored else None,
        "n_scored_4h": len(scored),
        "by_decision": by,
        "buys_mean_net_4h": (by.get("BUY_CANDIDATE") or {}).get("mean_net", {}).get("4"),
        "missed_moves": len(missed),
        "missed_at": [r["ts"][:16] for r in missed][-8:],
        "watch_counterfactual": {"n": len(watch), "mean_net": {
            str(h): _mean([r["net"][str(h)] for r in watch if str(h) in r["net"]])
            for h in HORIZONS}, "mean_best_net_8h": _mean(
            [r["best_net"] for r in watch if "best_net" in r])},
        "calibration": {"n": len(cal), "mean_est_gross": _mean([r["est_pct"] / 100 for r in cal]),
                        "mean_realised_best_move": _mean([r["best_move"] for r in cal]),
                        "bias": _mean(err), "mae": _mean([abs(e) for e in err])},
    }  # fmt: skip


SIDE = ("hit_rate_4h", "n_scored_4h", "buys_mean_net_4h", "missed_moves")


def evaluate(run_id: str, state_dir: Path | None = None, baseline: Analyst | None = None,
             path=None) -> dict[str, Any]:  # fmt: skip
    """Agent (logged decisions) vs ``baseline`` (default :class:`Rule`) on the run's recorded
    snapshots; ``side_by_side`` holds the headline metrics as [agent, baseline]."""
    state_dir = state_dir or config.STATE_DIR
    run_dir = state_dir / run_id
    baseline = baseline or Rule()
    path = path or price_path(state_dir)
    st_path = state_dir / f"{run_id}.json"
    st = json.loads(st_path.read_text()) if st_path.exists() else {}
    entry = {e["ts"][:19]: e.get("entry", "taker") for e in st.get("evals", [])}
    agent, base, same = [], [], 0
    for rec in read_calls(run_dir):
        snap = json.loads(rec["prompt"].split("Snapshot:\n", 1)[1])
        b = snap.get("usoon_book") or {}
        if not (b.get("ask") and b.get("bid")):
            continue
        now = _naive(rec["ts"])
        try:
            dec = validate(baseline(rec["prompt"], snap)["output"], pd.Timestamp(rec["ts"]),
                           bool(snap.get("position")))  # fmt: skip
        except AgentError as e:
            dec = no_trade(f"agent failed: AgentError: {e}"[:200])
        logged = rec["decision"]
        same += dec["decision"] == logged["decision"]
        kind = entry.get(rec["ts"][:19], "taker")
        for d, out in ((logged, agent), (dec, base)):
            marks = score(now, b["ask"], b["bid"], kind, path, d.get("horizon_hours"))
            out.append({"ts": rec["ts"], "decision": d["decision"],
                        "est_pct": d.get("estimated_gross_opportunity_pct"),
                        "horizon_h": d.get("horizon_hours")} | marks)  # fmt: skip
    a, r = stats(agent), stats(base)
    return {
        "baseline": baseline.name, "snapshots": len(agent),
        "same_decision": f"{same}/{len(agent)}",
        "price_points": len(path[0]),
        "path_end_utc": path[0][-1].isoformat() if path[0] else None,
        "side_by_side": {"columns": ["agent", baseline.name]}
        | {k: [a[k], r[k]] for k in SIDE}
        | {f"n_{d}": [a["decisions"].get(d, 0), r["decisions"].get(d, 0)] for d in DECISIONS},
        "agent": a, baseline.name: r,
    }  # fmt: skip
