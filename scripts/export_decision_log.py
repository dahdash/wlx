"""Export the per-decision log of a run to ``decision-log.csv`` + ``decision-log.md``.

Sources (written by the engines while they run, never edited afterwards):
- ``data/state/<run_id>/cycles.jsonl`` + ``trades.csv``: strategy G, one record per 30-min
  cycle (signals, votes, 28-day gate, risk verdict, target, orders, fills);
- ``data/state/convert-*.jsonl``: TMN → USDT at the start (``convert_tmn.py``) and the
  end-of-contest liquidation to TMN (``liquidate_tmn.py``, live ``order`` records only);
- ``data/state/<oil_run>/decisions.jsonl`` + ``trades.csv``: oil agent decisions (optional).

    python scripts/export_decision_log.py --run-id live-1 --oil-run oil-probe --oil-run oil-live-1 \
        --out docs/decision-log
"""

from __future__ import annotations

import argparse
import csv
import json
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLS = ["time_utc", "engine", "run_id", "mode", "kind", "symbol", "action", "target",
        "filled_qty", "price", "fee", "status", "signal", "reason"]  # fmt: skip
PROBE = "live micro round trip (buy, then sell) to check the oil order path before going live"


def _num(x) -> str:
    if x in (None, ""):
        return ""
    d = Decimal(str(x)).normalize()
    return f"{d:f}"


def _jsonl(path: Path):
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                yield json.loads(line)


def _rotated(run_dir: Path, name: str) -> list[Path]:
    """``<name>.<date>.jsonl`` files (rotated daily by the oil runner) oldest first, then the
    current ``<name>.jsonl``."""
    return [*sorted(run_dir.glob(f"{name}.????-??-??.jsonl")), run_dir / f"{name}.jsonl"]


def g_rows(state: Path, run_id: str) -> list[dict]:
    rows = []
    for c in _jsonl(state / run_id / "cycles.jsonl"):
        slot = c["equity_before"] / max(len(c["signals"]), 1)
        sides = {i["symbol"]: i["side"] for i in c.get("intents", [])}
        for s, sig in c["signals"].items():
            tgt = c["risk"]["targets"].get(s, 0.0)
            votes = int(sum(sig["votes"].values()))
            gate = "on" if sig["gate_on"] else "off"
            ret = sig.get("gate_ret")
            ret = "n/a" if ret is None else f"{ret:+.2%}"
            rows.append({
                "time_utc": c["bar_close"], "engine": "G", "run_id": run_id, "mode": c["mode"],
                "kind": "decision", "symbol": s, "action": sides.get(s, "HOLD"),
                "target": f"{tgt / slot:.2f}" if slot > 0 else "0.00",
                "signal": f"votes {votes}/{len(sig['votes'])}, 28d {ret} gate {gate}",
                "reason": c.get("reasons", {}).get(s) or sig["reason"],
                "status": c["risk"]["verdict"] or "",
            })  # fmt: skip
        for r in c.get("results", []):
            rows.append({
                "time_utc": c["bar_close"], "engine": "G", "run_id": run_id, "mode": c["mode"],
                "kind": "order", "symbol": r["symbol"], "action": r["side"],
                "filled_qty": _num(r["filled"]), "price": _num(r["vwap"]), "fee": _num(r["fee"]),
                "status": r["status"] + (" (crossed)" if r["crossed"] else ""),
                "reason": c.get("reasons", {}).get(r["symbol"], ""),
                "signal": " ".join(r["orders"]),
            })  # fmt: skip
    return rows


def convert_rows(state: Path) -> list[dict]:
    rows = []
    for f in sorted(state.glob("convert-*.jsonl")):
        for e in _jsonl(f):
            if e["event"] == "order" and e.get("mode") == "live":  # scripts/liquidate_tmn.py
                rows.append({
                    "time_utc": e["ts"][:19], "engine": "convert", "run_id": f.stem,
                    "mode": "live", "kind": "order", "symbol": e["symbol"], "action": e["side"],
                    "filled_qty": _num(e["filled"]), "price": _num(e["vwap"]),
                    "fee": _num(e["fee"]), "status": e["status"] + (f" ({e['note']})" if
                    e.get("note") else ""), "signal": " ".join(e["orders"]), "reason": e["reason"],
                })  # fmt: skip
                continue
            resp = e.get("resp") or {}
            if not e["event"].endswith(("_placed", "_rejected")):
                continue
            qty = Decimal(resp.get("executedQty") or 0)
            px = Decimal(resp["executedSum"]) / qty if qty else e.get("price", "")
            rows.append({
                "time_utc": e["ts"][:19], "engine": "convert", "run_id": f.stem, "mode": "live",
                "kind": "order", "symbol": resp.get("symbol", "USDTTMN"),
                "action": resp.get("side", "BUY"), "filled_qty": _num(qty),
                "price": _num(px), "fee": _num(resp.get("fee")),
                "status": resp.get("status", e["event"]),
                "signal": e["event"],
                "reason": "contest start/end: move the account between TMN and USDT",
            })  # fmt: skip
    return rows


def oil_rows(state: Path, run_id: str) -> list[dict]:
    rows = []
    st = state / f"{run_id}.json"
    mode = json.loads(st.read_text()).get("mode", "") if st.exists() else ""
    for d in (d for f in _rotated(state / run_id, "decisions") for d in _jsonl(f)):
        dec = d["decision"]
        gate = d.get("gate") or {}
        rows.append({
            "time_utc": d["ts"][:19], "engine": "oil-agent", "run_id": run_id, "mode": mode,
            "kind": "decision", "symbol": "USOONUSDT", "action": dec.get("decision", ""),
            "status": gate.get("verdict", "") if isinstance(gate, dict) else str(gate),
            "signal": d.get("why", ""), "reason": (dec.get("thesis") or "")[:300],
        })  # fmt: skip
    trades = state / run_id / "trades.csv"
    if trades.exists():
        for t in csv.DictReader(trades.open()):
            rows.append({
                "time_utc": t["time"][:19], "engine": "oil-agent", "run_id": run_id,
                "mode": t["mode"], "kind": "order", "symbol": t["symbol"], "action": t["side"],
                "filled_qty": _num(t["qty"]), "price": _num(t["price"]), "fee": _num(t["fee"]),
                "status": "filled", "signal": t["hypothesis_id"],
                "reason": PROBE if t["hypothesis_id"] == "probe" else t["reason"],
            })  # fmt: skip
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", action="append", default=[], help="G run id (repeatable)")
    ap.add_argument("--oil-run", action="append", default=[])
    ap.add_argument("--state", default=str(ROOT / "data" / "state"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "decision-log"))
    a = ap.parse_args(argv)
    state = Path(a.state)
    rows = convert_rows(state)
    for r in a.run_id:
        rows += g_rows(state, r)
    for r in a.oil_run:
        rows += oil_rows(state, r)
    rows.sort(key=lambda r: (r["time_utc"], r["kind"] != "decision"))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.with_suffix(".csv").open("w", newline="") as f:
        w = csv.DictWriter(f, COLS)
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in COLS} for r in rows)
    orders = [r for r in rows if r["kind"] == "order"]
    acts = [r for r in rows if r["kind"] == "decision" and r["action"] not in ("HOLD", "NO_TRADE")]
    md = [
        "# Decision log",
        "",
        f"Runs: G {', '.join(a.run_id) or '—'}; oil agent {', '.join(a.oil_run) or '—'}."
        f" {len(rows)} rows: {len(rows) - len(orders)} decisions, {len(orders)} orders."
        " Every row of every cycle is in `decision-log.csv` (one row per coin per 30-min"
        " cycle, plus one row per order).",
        "",
        "## Orders",
        "",
        "| time (UTC) | engine | symbol | side | filled | price | fee | status | reason |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |",
        *[f"| {r['time_utc']} | {r['engine']} | {r['symbol']} | {r['action']} |"
          f" {r.get('filled_qty', '')} | {r.get('price', '')} | {r.get('fee', '')} |"
          f" {r['status']} | {r['reason']} |" for r in orders],
        "",
        "## Decisions that changed a position (or asked to)",
        "",
        "| time (UTC) | engine | symbol | action | target | signal | reason |",
        "| --- | --- | --- | --- | ---: | --- | --- |",
        *[f"| {r['time_utc']} | {r['engine']} | {r['symbol']} | {r['action']} |"
          f" {r.get('target', '')} | {r['signal']} | {r['reason']} |" for r in acts],
        "",
    ]  # fmt: skip
    out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print(f"wrote {out}.csv / .md: {len(rows)} rows, {len(orders)} orders")


if __name__ == "__main__":
    main()
