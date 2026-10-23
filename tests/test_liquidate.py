"""liquidate_tmn: coins → USDT → TMN with a fake exchange, a fake clock, no network."""

import importlib.util
from decimal import Decimal
from pathlib import Path

from wallex.client import WallexError
from wallex.exec import OrderState

spec = importlib.util.spec_from_file_location(
    "liquidate_tmn", Path(__file__).resolve().parents[1] / "scripts" / "liquidate_tmn.py"
)
liq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(liq)

MARKETS = {
    "USDTTMN": {"minQty": 0.01, "minNotional": 50000, "stepSize": 2, "tickSize": 0},
    "BTCUSDT": {"minQty": 1e-8, "minNotional": 1, "stepSize": 8, "tickSize": 2},
    "BTCTMN": {"minQty": 1e-8, "minNotional": 50000, "stepSize": 8, "tickSize": 0},
    "ETHUSDT": {"minQty": 1e-4, "minNotional": 1, "stepSize": 4, "tickSize": 2},
    "USOONUSDT": {"minQty": 0.0001, "minNotional": 1, "stepSize": 4, "tickSize": 2},
}
BOOK = {
    "USDTTMN": ("268000", "268100"),
    "BTCUSDT": ("100000", "100010"),
    "BTCTMN": ("26800000000", "26810000000"),
    "ETHUSDT": ("4000", "4001"),
    "USOONUSDT": ("160", "161"),
}


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


class Fake:
    """Client + broker: orders fill at their limit price at once; balances move."""

    def __init__(self, bal, reject=(), orders=()):
        self.bal = {k: Decimal(v) for k, v in bal.items()}
        self.reject = dict(reject)  # symbol -> number of 422s left
        self.open = list(orders)
        self.placed, self.states = [], {}

    # client
    def markets(self):
        return {"symbols": MARKETS}

    def balances(self):
        return {"balances": {k: {"value": str(v), "locked": "0"} for k, v in self.bal.items()}}

    def open_orders(self):
        return {"orders": [{"clientOrderId": c} for c in self.open]}

    # broker
    def best(self, sym):
        return tuple(Decimal(x) for x in BOOK[sym])

    def place(self, sym, side, qty, price, cid):
        if self.reject.get(sym, 0) > 0:
            self.reject[sym] -= 1
            raise WallexError("market closed", 422)
        base, quote = (sym[:-4], "USDT") if sym.endswith("USDT") else (sym[:-3], "TMN")
        sign = 1 if side == "BUY" else -1
        self.bal[base] = self.bal.get(base, Decimal(0)) + sign * qty
        self.bal[quote] = self.bal.get(quote, Decimal(0)) - sign * qty * price
        self.placed.append((sym, side, qty))
        self.states[cid] = OrderState(cid, qty, price, Decimal(0), active=False)

    def status(self, cid):
        return self.states[cid]

    def cancel(self, cid):
        return self.states[cid]


def make(fake, clock, paper=False):
    events = []
    lq = liq.Liquidator(
        fake, fake, paper=paper, log=lambda e, **kw: events.append((e, kw)),
        clock=clock, sleep=clock.sleep,
    )  # fmt: skip
    return lq, events


def test_direct_sweeps_coins_then_sells_usdt_and_leaves_dust():
    f, c = Fake({"USDT": "18.5", "ETH": "0.01", "USOON": "0.00008817", "TMN": "20704"}), Clock()
    lq, ev = make(f, c)
    lq.run(route="direct", fallback_at=1e12, deadline=c.t + 3600, retry_s=30, once=False)
    assert [p[:2] for p in f.placed] == [("ETHUSDT", "SELL"), ("USDTTMN", "SELL")]
    assert f.bal["USDT"] < Decimal("0.2") and f.bal["ETH"] == 0
    assert ("dust", {"asset": "USOON", "qty": Decimal("0.00008817"), "market": "USOONUSDT"}) in ev
    assert [e for e, _ in ev].count("done") == 1


def test_halted_market_is_retried_until_it_opens():
    f, c = Fake({"USDT": "18.5"}, reject={"USDTTMN": 3}), Clock()
    lq, ev = make(f, c)
    lq.run(route="direct", fallback_at=1e12, deadline=c.t + 3600, retry_s=30, once=False)
    orders = [kw for e, kw in ev if e == "order"]
    assert orders[0]["status"] == "none" and "place error" in orders[0]["note"]
    assert orders[-1]["status"] == "filled" and f.bal["USDT"] < Decimal("0.2")


def test_fallback_routes_via_btc_and_keeps_btc():
    f, c = Fake({"USDT": "18.5"}, reject={"USDTTMN": 99}), Clock()
    lq, _ = make(f, c)
    lq.run(route="auto", fallback_at=c.t + 100, deadline=c.t + 3600, retry_s=30, once=False)
    syms = [p[:2] for p in f.placed]
    assert ("BTCUSDT", "BUY") in syms and syms[-1] == ("BTCTMN", "SELL")
    assert ("BTCUSDT", "SELL") not in syms  # the sweep never sells BTC back
    assert f.bal["BTC"] < Decimal("0.000001") and f.bal["TMN"] > Decimal("4900000")


def test_deadline_stops_and_alerts():
    f, c = Fake({"USDT": "18.5"}, reject={"USDTTMN": 999}), Clock()
    lq, ev = make(f, c)
    lq.run(route="direct", fallback_at=1e12, deadline=c.t + 600, retry_s=30, once=False)
    assert ev[-2][0] == "deadline" and ev[-1][0] == "balances"
    assert f.bal["USDT"] == Decimal("18.5")


def test_foreign_open_orders_are_detected_own_are_not():
    f = Fake({"USDT": "1"}, orders=["oil-USOONUSDT-B1", "liq17923-USDTTMN-S1"])
    lq, _ = make(f, Clock())
    assert lq.foreign_orders() == ["oil-USOONUSDT-B1"]


def test_paper_pass_runs_once_without_cancel_or_balances():
    f, c = Fake({"USDT": "18.5"}), Clock()
    lq, ev = make(f, c, paper=True)
    lq.run(route="btc", fallback_at=1e12, deadline=c.t + 3600, retry_s=30, once=True)
    assert [e for e, _ in ev] == ["pass", "order", "order"]
    assert [kw["symbol"] for e, kw in ev if e == "order"] == ["BTCUSDT", "BTCTMN"]


def test_order_ids_are_unique_across_passes():
    f, c = Fake({"USDT": "18.5"}, reject={"USDTTMN": 99}), Clock()
    lq, _ = make(f, c)
    lq.run(route="auto", fallback_at=c.t + 100, deadline=c.t + 3600, retry_s=30, once=False)
    cids = list(f.states)
    assert len(cids) == len(set(cids)) >= 2
