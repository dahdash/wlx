"""T10/T11: live probe on USDTTMN, then convert the remaining TMN to USDT in one order.

1. From ``--at`` (UTC), place a small crossing LIMIT buy (``--probe`` USDT at the best ask);
   while the market answers 422 "closed", retry every 3 s until ``--deadline``.
2. Only if the probe is fully filled within ~10 s: one LIMIT buy for all free TMN, price
   capped at best ask × (1 + ``--cap``). An unfilled remainder is cancelled after 60 s.
Every request/response goes to ``data/state/convert-tmn.jsonl``. Needs WALLEX_LIVE=1.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from decimal import ROUND_DOWN, ROUND_UP, Decimal
from pathlib import Path

from wallex.client import WallexClient, WallexError, load_env
from wallex.exec import LiveBroker, parse_order

SYM = "USDTTMN"
LOG = Path(__file__).resolve().parents[1] / "data" / "state" / "convert-tmn.jsonl"


def log(event: str, **kw) -> None:
    rec = {"ts": dt.datetime.now(dt.UTC).isoformat(), "event": event, **kw}
    print(json.dumps(rec, ensure_ascii=False, default=str), flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def free_tmn(c: WallexClient) -> Decimal:
    b = c.balances()
    t = b.get("balances", b).get("TMN", {})
    return Decimal(str(t.get("value") or 0)) - Decimal(str(t.get("locked") or 0))


def wait_fill(br: LiveBroker, c: WallexClient, cid: str, qty: Decimal, secs: float):
    end = time.time() + secs
    while True:
        raw = c.order(cid)
        st = parse_order(raw, cid)
        if st.filled >= qty or time.time() > end:
            return st, raw
        time.sleep(2)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--at", default="08:00:00", help="start time, UTC HH:MM:SS today")
    p.add_argument("--deadline", default="08:15:00", help="give up retrying, UTC")
    p.add_argument("--probe", default="0.21", help="probe size in USDT")
    p.add_argument("--skip-probe", action="store_true", help="probe already filled")
    p.add_argument("--cap", type=float, default=0.003, help="max price above best ask")
    a = p.parse_args()
    load_env()
    c = WallexClient()
    br = LiveBroker(c, live=True)
    today = dt.datetime.now(dt.UTC).date()
    at, dl = (
        dt.datetime.combine(today, dt.time.fromisoformat(x), dt.UTC) for x in (a.at, a.deadline)
    )
    log("start", at=at, deadline=dl, free_tmn=free_tmn(c))
    while dt.datetime.now(dt.UTC) < at:
        time.sleep(0.5)

    qty = Decimal(a.probe)
    while not a.skip_probe:
        bid, ask = br.best(SYM)
        cid = f"probe{int(time.time() * 1000)}"
        try:
            r = c.place_order(SYM, "BUY", "LIMIT", str(qty), str(ask), client_id=cid, live=True)
            log("probe_placed", cid=cid, price=ask, qty=qty, bid=bid, resp=r)
            break
        except WallexError as e:
            log("probe_rejected", status=e.status_code, error=str(e))
            if e.status_code != 422 or dt.datetime.now(dt.UTC) > dl:
                log("abort", reason="probe not accepted")
                return
            time.sleep(3)

    if not a.skip_probe:
        st, raw = wait_fill(br, c, cid, qty, 10)
        log("probe_status", state=st, raw=raw)
    if not a.skip_probe and st.filled < qty:
        log("probe_cancel", state=br.cancel(cid))
        log("abort", reason="probe not fully filled; no bulk order")
        return

    while True:  # the market flaps open/closed around the open: retry 422 until the deadline
        tmn = free_tmn(c)
        bid, ask = br.best(SYM)
        cap = (ask * (1 + Decimal(str(a.cap)))).to_integral_value(rounding=ROUND_UP)
        qty = (tmn * Decimal("0.999") / cap).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
        cid = f"bulk{int(time.time() * 1000)}"
        try:
            r = c.place_order(SYM, "BUY", "LIMIT", str(qty), str(cap), client_id=cid, live=True)
            break
        except WallexError as e:
            log("bulk_rejected", status=e.status_code, error=str(e), qty=qty, price=cap)
            if e.status_code != 422 or dt.datetime.now(dt.UTC) > dl:
                log("abort", reason="bulk not accepted")
                return
            time.sleep(3)
    log("bulk_placed", cid=cid, qty=qty, price=cap, best_ask=ask, free_tmn=tmn, resp=r)
    st, raw = wait_fill(br, c, cid, qty, 60)
    log("bulk_status", state=st, raw=raw)
    if st.active:
        log("bulk_cancel_remainder", state=br.cancel(cid))
    b = c.balances()
    log(
        "balances",
        balances={k: v for k, v in b.get("balances", b).items() if float(v.get("value") or 0) > 0},
    )


if __name__ == "__main__":
    main()
