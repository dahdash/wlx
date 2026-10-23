"""Build ``docs/results/results.json``: what the live account and the runs actually did.

Read-only. Sources (written by the engines while they ran, never edited afterwards):
- ``data/state/convert-*.jsonl``: TMN → USDT at the start, liquidation to TMN at the end;
- ``data/state/<g_run>/cycles.jsonl`` + ``alerts.log``: strategy G, one record per 30-min cycle
  (equity in USDT, signals, risk verdict, orders, fills);
- ``data/state/<oil_run>.json`` + ``<oil_run>/decisions*.jsonl`` + ``<oil_run>.out``: the oil
  agent (live ``oil-live-1``, shadow ``oil-s2``): decisions, forward scores, lessons, ticks;
- ``data/state/oil-probe/trades.csv``: the live micro round trip that tested the oil order path;
- Wallex public API (no key): USDTTMN hourly candles to value the account in toman, and the
  USDTTMN book for the mark at build time. ``--offline`` reuses the candles already in the
  output file and skips the book.

    python scripts/build_results.py                   # → docs/results/results.json
    python scripts/build_results.py --offline --check # no network; print the headline numbers
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_decision_log import _jsonl, _rotated, convert_rows, g_rows, oil_rows  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
START_TMN = 5_000_000  # challenge.md: dedicated 5M-toman account
TAKER_FEE = 0.003  # USDTTMN taker coefficient on the fills (convert-tmn.jsonl)
CYCLE_S = 30 * 60  # G cycle
TICK_S = 300  # oil tick (wallex.oil.config.TICK_S)
HORIZONS = ("1", "2", "4", "8")  # oil forward-score horizons, hours


def _ts(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s[:19]).replace(tzinfo=dt.UTC)


def _iso(t: dt.datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%S")


def _pct(x: float, nd: int = 3) -> float:
    return round(100 * x, nd)


def uptime(times: list[dt.datetime], step_s: float, until: dt.datetime) -> dict:
    expected = int((until - times[0]).total_seconds() // step_s) + 1
    done = min(len(times), expected)
    return {"first": _iso(times[0]), "last": _iso(times[-1]), "expected": expected,
            "done": len(times), "pct": round(100 * done / expected, 1)}  # fmt: skip


def usdttmn_candles(since: dt.datetime, until: dt.datetime) -> dict[str, float]:
    from wallex.client import WallexClient

    with WallexClient() as c:
        b = c.candles("USDTTMN", "60", int(since.timestamp()), int(until.timestamp()))
    return {
        _iso(dt.datetime.fromtimestamp(t, dt.UTC)): float(x)
        for t, x in zip(b["t"], b["c"], strict=True)
    }


def usdttmn_bid() -> float:
    from wallex.client import WallexClient

    with WallexClient() as c:
        return float(c.depth("USDTTMN")["bid"][0]["price"])


def rate_at(rates: dict[str, float], t: dt.datetime) -> float | None:
    """Close of the last hourly USDTTMN bar that had closed by ``t``."""
    keys = [k for k in rates if _ts(k) <= t - dt.timedelta(hours=1)]
    return rates[max(keys)] if keys else None


def max_dd(values: list[float]) -> float:
    peak, dd = float("-inf"), 0.0
    for v in values:
        peak = max(peak, v)
        dd = min(dd, v / peak - 1)
    return dd


def filled(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["filled_qty"] and float(r["filled_qty"]) > 0]


def oil_decisions(state: Path, run: str) -> list[dict]:
    return [d for f in _rotated(state / run, "decisions") for d in _jsonl(f)]


def forward(evals: list[dict]) -> dict:
    """Forward score of every decision, net of the round-trip cost at the time: a NO_TRADE was
    right when the trade it declined would have lost money at that horizon."""
    out = {}
    for h in HORIZONS:
        net = [e["net"][h] for e in evals if (e.get("net") or {}).get(h) is not None]
        n = len(net)
        out[h + "h"] = {
            "n": n,
            "entry_would_lose_pct": round(100 * sum(x <= 0 for x in net) / n, 1) if n else None,
            "mean_net_pct": _pct(sum(net) / n) if n else None,
        }
    return out


def oil_section(state: Path, run: str, now: dt.datetime) -> dict:
    st = json.loads((state / f"{run}.json").read_text())
    decs = oil_decisions(state, run)
    counts: dict[str, int] = {}
    for d in decs:
        k = d["decision"].get("decision", "?")
        counts[k] = counts.get(k, 0) + 1
    log = (state / f"{run}.out").read_text().splitlines()
    ticks = [_ts(ln) for ln in log if ln[:2] == "20"]
    trades = st.get("trades") or 0
    last = decs[-1] if decs else None
    return {
        "run_id": run,
        "mode": st["mode"],
        "profile": st.get("profile"),
        "decisions": len(decs),
        "by_decision": counts,
        "trades": len(trades) if isinstance(trades, list) else trades,
        "closed": st.get("closed", []),
        "killed": st.get("killed"),
        "paused": st.get("paused_reason"),
        "forward": forward(st.get("evals", [])),
        "lessons": [{"t": t[:19], "text": x} for t, x in st.get("lessons", [])],
        "last_decision": last
        and {
            "t": last["ts"][:19],
            "why": last["why"],
            "decision": last["decision"].get("decision"),
            "thesis": last["decision"].get("thesis", ""),
        },
        "uptime": uptime(ticks, TICK_S, now),
    }


def g_section(cycles: list[dict], state: Path, run: str, now: dt.datetime) -> dict:
    alerts = state / run / "alerts.log"
    best = max(
        (x["gate_ret"], s, c["bar_close"])
        for c in cycles
        for s, x in c["signals"].items()
        if x.get("gate_ret") is not None
    )
    return {
        "run_id": run,
        "mode": cycles[-1]["mode"],
        "cycles": len(cycles),
        "orders": sum(len(c.get("results", [])) for c in cycles),
        "cycles_any_gate_on": sum(any(x["gate_on"] for x in c["signals"].values()) for c in cycles),
        "best_28d": {"ret_pct": _pct(best[0], 2), "symbol": best[1], "t": best[2]},
        "risk_events": [e for c in cycles for e in c.get("risk_events", [])],
        "verdicts": sorted({c["risk"]["verdict"] for c in cycles if c["risk"]["verdict"]}),
        "last_signals": {s: x["reason"] for s, x in cycles[-1]["signals"].items()},
        "uptime": uptime([_ts(c["bar_close"]) for c in cycles], CYCLE_S, now),
        "alerts": alerts.read_text().splitlines() if alerts.exists() else [],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--g-run", default="live-1")
    ap.add_argument("--oil-live", default="oil-live-1")
    ap.add_argument("--oil-shadow", default="oil-s2")
    ap.add_argument("--state", default=str(ROOT / "data" / "state"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "results" / "results.json"))
    ap.add_argument("--offline", action="store_true", help="no network: reuse cached candles")
    ap.add_argument("--check", action="store_true", help="print the headline numbers")
    a = ap.parse_args(argv)
    state, out = Path(a.state), Path(a.out)
    now = dt.datetime.now(dt.UTC).replace(microsecond=0)
    prev = json.loads(out.read_text()) if out.exists() else {}

    # --- account: start conversion, TMN left over -----------------------------------------
    conv = convert_rows(state)
    buys = filled([r for r in conv if r["symbol"] == "USDTTMN" and r["action"] == "BUY"])
    sells = filled([r for r in conv if r["symbol"] == "USDTTMN" and r["action"] == "SELL"])
    bought = sum(float(r["filled_qty"]) for r in buys)
    spent = sum(float(r["filled_qty"]) * float(r["price"]) for r in buys)
    start = min(_ts(r["time_utc"]) for r in buys)
    tmn_left = START_TMN - spent  # TMN not converted (below USDTTMN's minimum order)

    # --- equity: G's account equity in USDT every 30 min, valued in toman -----------------
    if a.offline:
        rates = prev.get("usdttmn_1h", {})
        bid = (prev.get("mark") or {}).get("usdttmn_bid")
    else:
        rates = usdttmn_candles(start - dt.timedelta(hours=2), now)
        bid = usdttmn_bid()
    cycles = list(_jsonl(state / a.g_run / "cycles.jsonl"))
    curve = []
    for c in cycles:
        r = rate_at(rates, _ts(c["bar_close"]))
        curve.append({
            "t": c["bar_close"], "usdt": c["equity_after"], "usdttmn": r,
            "tmn": None if r is None else round(c["equity_after"] * r + tmn_left),
            "orders": len(c.get("results", [])),
        })  # fmt: skip
    usdt_now = cycles[-1]["equity_after"]
    mark = {"t": _iso(now), "usdttmn_bid": bid, "usdt": usdt_now, "tmn_left": round(tmn_left)}
    if bid:
        mark["tmn_mark"] = round(usdt_now * bid + tmn_left)
        mark["tmn_after_taker_fee"] = round(usdt_now * bid * (1 - TAKER_FEE) + tmn_left)
        mark["return_pct"] = _pct(mark["tmn_after_taker_fee"] / START_TMN - 1)
    final = None
    if sells:  # liquidation done: the official number is the TMN balance afterwards
        got = sum(float(r["filled_qty"]) * float(r["price"]) for r in sells)
        final = {"t": max(r["time_utc"] for r in sells),
                 "sold_usdt": sum(float(r["filled_qty"]) for r in sells),
                 "got_tmn": round(got)}  # fmt: skip

    # --- every filled order on the account ------------------------------------------------
    trades = conv + g_rows(state, a.g_run) + oil_rows(state, a.oil_live)
    if (state / "oil-probe" / "trades.csv").exists():
        trades += oil_rows(state, "oil-probe")
    trades = sorted(
        filled([r for r in trades if r["kind"] == "order"]), key=lambda r: r["time_utc"]
    )
    fees_usdt = sum(float(r["fee"] or 0) for r in trades if r["symbol"] != "USDTTMN")
    fees_usdt += sum(float(r["fee"] or 0) for r in buys)  # USDTTMN buys pay the fee in USDT
    fees_tmn = sum(float(r["fee"] or 0) for r in sells)  # USDTTMN sells pay it in TMN
    usdt_tmn = curve[-1]["usdttmn"] or bid or 0

    # --- oil agent: live vs shadow --------------------------------------------------------
    oil = {"live": oil_section(state, a.oil_live, now),
           "shadow": oil_section(state, a.oil_shadow, now)}  # fmt: skip
    lv = {d["ts"][:16]: d["decision"].get("decision") for d in oil_decisions(state, a.oil_live)}
    sh = {d["ts"][:16]: d["decision"].get("decision") for d in oil_decisions(state, a.oil_shadow)}
    both = sorted(set(lv) & set(sh))
    oil["live_vs_shadow"] = {"same_minute": len(both), "agree": sum(lv[k] == sh[k] for k in both)}

    tmn_series = [p["tmn"] for p in curve if p["tmn"] is not None]
    g = g_section(cycles, state, a.g_run, now)
    res = {
        "built_utc": _iso(now),
        "status": "final" if final else "interim (account still trading)",
        "start": {
            "t": _iso(start),
            "tmn": START_TMN,
            "usdt_bought": round(bought, 8),
            "tmn_spent": round(spent),
            "avg_usdttmn": round(spent / bought, 1),
            "tmn_left": round(tmn_left),
        },
        "mark": mark,
        "final": final,
        "equity": {
            "usdt_first_cycle": cycles[0]["equity_before"],
            "usdt_now": usdt_now,
            "max_dd_usdt_pct": _pct(max_dd([p["usdt"] for p in curve])),
            "max_dd_tmn_pct": _pct(max_dd(tmn_series)) if tmn_series else None,
        },
        "fees": {
            "usdt": round(fees_usdt, 6),
            "tmn": round(fees_tmn),
            "tmn_total_approx": round(fees_usdt * usdt_tmn + fees_tmn),
        },
        "trades": trades,
        "g": g,
        "oil": oil,
        "curve": curve,
        "usdttmn_1h": rates,
        "backtest": {
            "metrics": "docs/backtest/metrics.md",
            "validation": "docs/backtest/validation.html",
            "crosscheck": "docs/backtest/crosscheck.md",
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {out}: {len(curve)} G cycles, {len(trades)} fills, {res['status']}")
    if a.check:
        for k in ("start", "mark", "final", "equity", "fees"):
            print(k, json.dumps(res[k]))
        print("G", {k: g[k] for k in ("cycles", "orders", "cycles_any_gate_on", "best_28d")})
        print("G uptime", g["uptime"]["pct"], "%")
        for role in ("live", "shadow"):
            o = oil[role]
            up, f4 = o["uptime"]["pct"], o["forward"]["4h"]
            print(f"oil {role}: {o['decisions']} decisions {o['by_decision']}, "
                  f"trades {o['trades']}, up {up} %, 4h {f4}")  # fmt: skip
        print("live vs shadow", oil["live_vs_shadow"])
        for r in trades:
            print(" ", r["time_utc"], r["engine"], r["symbol"], r["action"], r["filled_qty"],
                  r["price"], r["fee"], r["status"])  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
