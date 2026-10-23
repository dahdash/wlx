"""End of contest (plan §5): sell every coin to USDT, then USDT → TMN, algorithmically.

1. Waits until ``--at`` (UTC). Aborts if the account has open orders it did not place:
   stop ``live-1`` and the oil runs first, so nothing else trades.
2. Every pass: sell each free coin on ``<COIN>USDT`` (limit-through, the traded
   :class:`~wallex.exec.Executor`), then sell free USDT on ``USDTTMN``. A rejected order
   (e.g. 422 while the market is halted) is logged and retried every ``--retry`` s.
3. From ``--fallback-at`` (or ``--route btc``): USDT → BTC on ``BTCUSDT``, BTC → TMN on
   ``BTCTMN``, because USDTTMN halts at 23:30 Tehran.
4. Done when the free USDT is below USDTTMN's minimum order, or at ``--deadline``; then any
   order of its own still open is cancelled and the balances are logged.

Dust below a market's minimum (e.g. USOON < 0.0001) is logged and left.

Dry run is the default: real balances and books, :class:`~wallex.exec.PaperBroker` fills,
one pass, no waiting, log ``data/state/liquidate-dry.jsonl``. Live needs ``--live`` and
``WALLEX_LIVE=1``; it logs to ``data/state/convert-to-tmn.jsonl`` (read by the decision-log
export).

    python scripts/liquidate_tmn.py                       # dry run, direct route
    python scripts/liquidate_tmn.py --route btc           # dry run, BTC route
    WALLEX_LIVE=1 python scripts/liquidate_tmn.py --live --date 2026-10-23
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

from wallex.client import WallexClient, load_env
from wallex.exec import Executor, Intent, LiveBroker, MarketRules, PaperBroker

STATE = Path(__file__).resolve().parents[1] / "data" / "state"
CASH = ("TMN", "USDT")
TAG = "liq"
REASON = "contest end: realise the account in TMN (plan §5)"


def free_balances(client: WallexClient) -> dict[str, Decimal]:
    b = client.balances()
    out = {}
    for asset, v in b.get("balances", b).items():
        q = Decimal(str(v.get("value") or 0)) - Decimal(str(v.get("locked") or 0))
        if q > 0:
            out[asset] = q
    return out


class Liquidator:
    def __init__(
        self,
        client: WallexClient,
        broker,
        *,
        paper: bool,
        log: Callable[..., None],
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        poll_s: float = 5,
        through_wait_s: float = 20,
    ):
        self.client, self.broker, self.paper, self.log = client, broker, paper, log
        self.clock, self.sleep = clock, sleep
        self.poll_s, self.through_wait_s = poll_s, through_wait_s
        self.markets = client.markets()["symbols"]
        self.rules: dict[str, MarketRules] = {}
        self.n = 0

    def rule(self, sym: str) -> MarketRules | None:
        if sym not in self.rules and sym in self.markets:
            self.rules[sym] = MarketRules.from_market(self.markets[sym])
        return self.rules.get(sym)

    def sellable(self, sym: str, qty: Decimal) -> Decimal:
        """Quantised qty if it makes a valid order at the bid, else 0."""
        r = self.rule(sym)
        if r is None:
            return Decimal(0)
        q = r.q_qty(qty)
        bid, _ = self.broker.best(sym)
        return q if r.valid(q, bid) else Decimal(0)

    def trade(self, sym: str, side: str, *, qty=None, notional=None):
        self.n += 1
        ex = Executor(
            self.broker,
            self.rules,
            mode="through",
            clock=self.clock,
            sleep=self.sleep,
            poll_s=self.poll_s,
            through_wait_s=self.through_wait_s,
        )
        tag = f"{TAG}{int(self.clock() * 1000)}n{self.n}"
        (res,) = ex.execute([Intent(sym, side, qty=qty, notional=notional)], tag=tag)
        self.log(
            "order",
            symbol=sym,
            side=side,
            requested=res.requested,
            filled=res.filled,
            vwap=res.vwap,
            fee=res.fee,
            status=res.status,
            crossed=res.crossed,
            orders=res.orders,
            note=res.note,
        )
        return res

    def sweep(self, bal: dict[str, Decimal], keep: tuple[str, ...] = ()) -> None:
        for coin, q in sorted(bal.items()):
            if coin in CASH or coin in keep:
                continue
            sym = f"{coin}USDT"
            qty = self.sellable(sym, q)
            if qty > 0:
                self.trade(sym, "SELL", qty=qty)
            else:
                self.log("dust", asset=coin, qty=q, market=sym if sym in self.markets else None)

    def direct(self, bal: dict[str, Decimal]) -> None:
        qty = self.sellable("USDTTMN", bal.get("USDT", Decimal(0)))
        if qty > 0:
            self.trade("USDTTMN", "SELL", qty=qty)

    def via_btc(self, bal: dict[str, Decimal]) -> None:
        btc = self.sellable("BTCTMN", bal.get("BTC", Decimal(0)))
        if btc > 0:  # left from an earlier pass
            self.trade("BTCTMN", "SELL", qty=btc)
        usdt = bal.get("USDT", Decimal(0)) * Decimal("0.995")  # room for cap + fee
        r = self.rule("BTCUSDT")
        if r is None or usdt < max(r.min_notional, Decimal(1)):
            return
        res = self.trade("BTCUSDT", "BUY", notional=usdt)
        got = res.filled - (res.fee / res.vwap if res.vwap else 0)  # buy fee is in BTC
        if not self.paper:
            got = free_balances(self.client).get("BTC", Decimal(0))
        btc = self.sellable("BTCTMN", got)
        if btc > 0:
            self.trade("BTCTMN", "SELL", qty=btc)

    def done(self, bal: dict[str, Decimal]) -> bool:
        return self.sellable("USDTTMN", bal.get("USDT", Decimal(0))) == 0

    def open_ids(self) -> list[str]:
        oo = self.client.open_orders()
        orders = oo.get("orders")
        if orders is None:  # docs: keyed by symbol
            orders = [o for v in oo.values() if isinstance(v, list) for o in v]
        return [str(o.get("clientOrderId") or "") for o in orders]

    def foreign_orders(self) -> list[str]:
        return [i for i in self.open_ids() if not i.startswith(TAG)]

    def run(self, *, route: str, fallback_at: float, deadline: float, retry_s: float, once: bool):
        while True:
            btc = route == "btc" or (route == "auto" and self.clock() >= fallback_at)
            try:
                bal = free_balances(self.client)
                self.log("pass", route="btc" if btc else "direct", free=bal)
                self.sweep(bal, keep=("BTC",) if btc else ())
                bal = bal if self.paper else free_balances(self.client)
                (self.via_btc if btc else self.direct)(bal)
                if not self.paper and self.done(free_balances(self.client)):
                    self.log("done")
                    break
            except Exception as e:  # noqa: BLE001 — keep retrying until the deadline
                self.log("error", error=f"{type(e).__name__}: {e}")
            if once:
                break
            if self.clock() + retry_s > deadline:
                self.log("deadline", reason="USDT not fully converted; alert the user")
                break
            self.sleep(retry_s)
        if not self.paper:
            for cid in (i for i in self.open_ids() if i.startswith(TAG)):
                self.log("cancel_own", cid=cid, state=self.broker.cancel(cid))
            self.log("balances", free=free_balances(self.client))


def _at(date: dt.date, hhmm: str) -> float:
    return dt.datetime.combine(date, dt.time.fromisoformat(hhmm), dt.UTC).timestamp()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--live", action="store_true", help="real orders (also needs WALLEX_LIVE=1)")
    p.add_argument("--date", default=None, help="UTC date of the times below (default: today)")
    p.add_argument("--at", default="19:30", help="start, UTC (23:00 Tehran)")
    p.add_argument("--fallback-at", default="19:50", help="switch to the BTC route, UTC")
    p.add_argument("--deadline", default="20:15", help="give up, UTC (23:45 Tehran)")
    p.add_argument("--route", choices=["auto", "direct", "btc"], default="auto")
    p.add_argument("--retry", type=float, default=30, help="seconds between passes")
    a = p.parse_args(argv)
    load_env()
    client = WallexClient()
    log_path = STATE / ("convert-to-tmn.jsonl" if a.live else "liquidate-dry.jsonl")

    def log(event: str, **kw) -> None:
        rec = {"ts": dt.datetime.now(dt.UTC).isoformat(), "event": event,
               "mode": "live" if a.live else "paper", **kw}  # fmt: skip
        if event == "order":
            rec["reason"] = REASON
        line = json.dumps(rec, ensure_ascii=False, default=str)
        print(line, flush=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as f:
            f.write(line + "\n")

    date = dt.date.fromisoformat(a.date) if a.date else dt.datetime.now(dt.UTC).date()
    at, fb, dl = (_at(date, x) for x in (a.at, a.fallback_at, a.deadline))
    broker = LiveBroker(client, live=True) if a.live else PaperBroker(client.depth)
    liq = Liquidator(client, broker, paper=not a.live, log=log)
    log("start", at=a.at, fallback_at=a.fallback_at, deadline=a.deadline, route=a.route)
    if a.live:
        while time.time() < at:
            time.sleep(1)
    foreign = liq.foreign_orders()
    if foreign:
        log("abort", reason="open orders from another run; stop the engines first", ids=foreign)
        return 2
    liq.run(route=a.route, fallback_at=fb, deadline=dl, retry_s=a.retry, once=not a.live)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
