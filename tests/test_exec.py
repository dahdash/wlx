"""Executor tests: a fake Wallex over httpx.MockTransport, a fake clock, no real sleeping."""

import json
from decimal import Decimal

import httpx
import pytest

from wallex import WallexClient
from wallex.exec import (
    Executor,
    Intent,
    LiveBroker,
    MarketRules,
    PaperBroker,
    parse_order,
)

RULES = {"BTCUSDT": MarketRules(tick=2, step=6, min_notional=Decimal(1))}


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


class FakeWallex:
    """Minimal exchange: depth from ``self.book``; orders rest until ``fill(cid)``."""

    def __init__(self):
        self.book = {"bid": "100.00", "ask": "100.20"}
        self.orders = {}
        self.calls = []

    def fill(self, cid, qty=None, price=None):
        o = self.orders[cid]
        o["executedQty"] = qty or o["quantity"]
        o["executedPrice"] = price or o["price"]
        o["fee"] = str(Decimal(o["executedQty"]) * Decimal("0.001"))
        o["active"] = False

    def __call__(self, req: httpx.Request) -> httpx.Response:
        self.calls.append((req.method, req.url.path))
        ok = lambda r: httpx.Response(200, json={"success": True, "message": "", "result": r})  # noqa: E731
        if req.url.path == "/v1/depth":
            return ok(
                {
                    "bid": [{"price": self.book["bid"], "quantity": "1"}],
                    "ask": [{"price": self.book["ask"], "quantity": "1"}],
                }
            )
        if req.method == "POST" and req.url.path == "/v1/account/orders":
            body = json.loads(req.content)
            cid = body["client_id"]
            self.orders[cid] = {
                **body,
                "clientOrderId": cid,
                "active": True,
                "executedQty": "0",
                "executedPrice": "0",
                "fee": "0",
            }
            return ok(self.orders[cid])
        if req.url.path.startswith("/v1/account/orders/"):
            cid = req.url.path.rsplit("/", 1)[1]
            if req.method == "DELETE":
                self.orders[cid]["active"] = False
            return ok(self.orders[cid])
        raise AssertionError(req.url.path)


@pytest.fixture
def live(monkeypatch):
    monkeypatch.setenv("WALLEX_LIVE", "1")
    fx = FakeWallex()
    client = WallexClient("k", transport=httpx.MockTransport(fx))
    return fx, LiveBroker(client, live=True)


def make(broker, clock, mode="maker"):
    return Executor(broker, RULES, mode=mode, clock=clock, sleep=clock.sleep)


# --- quantisation ---------------------------------------------------------------------


def test_market_rules_quantise_and_validate():
    r = MarketRules.from_market(
        {"tickSize": 2, "stepSize": 6, "minNotional": "1", "minQty": "0.00001"}
    )
    assert r.q_price("100.129", "BUY") == Decimal("100.12")
    assert r.q_price("100.121", "SELL") == Decimal("100.13")
    assert r.q_qty("0.0123456789") == Decimal("0.012345")
    assert r.valid(Decimal("0.01"), Decimal("100"))
    assert not r.valid(Decimal("0.005"), Decimal("100"))  # 0.5 < minNotional


def test_parse_order_converts_buy_fee_to_quote():
    st = parse_order(
        {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "executedQty": "2",
            "executedPrice": "10",
            "fee": "0.002",
            "active": False,
        },
        "x",
    )
    assert (st.filled, st.vwap, st.fee, st.active) == (
        Decimal(2),
        Decimal(10),
        Decimal("0.02"),
        False,
    )
    st = parse_order(
        {
            "symbol": "BTCUSDT",
            "executedQty": "1",
            "executedPrice": "10",
            "fills": [{"fee": "0.01", "feeAsset": "USDT", "price": "10"}],
            "active": True,
        },
        "y",
    )
    assert st.fee == Decimal("0.01") and st.active


# --- live broker guard ---------------------------------------------------------------


def test_live_broker_refuses_without_env(monkeypatch):
    monkeypatch.delenv("WALLEX_LIVE", raising=False)
    with pytest.raises(RuntimeError):
        LiveBroker(WallexClient("k", transport=httpx.MockTransport(FakeWallex())), live=True)


# --- maker-first ---------------------------------------------------------------------------


def test_maker_posts_at_bid_and_fills(live):
    fx, broker = live
    clock = Clock()
    ex = make(broker, clock)
    # fill right after the first poll
    orig_status = broker.status

    def status(cid):
        fx.fill(cid)
        return orig_status(cid)

    broker.status = status
    [res] = ex.execute([Intent("BTCUSDT", "BUY", notional=Decimal("3.7"))], tag="r1-c0")
    o = fx.orders["r1-c0-BTCUSDT-B1"]
    assert o["price"] == "100.00" and o["type"] == "LIMIT" and o["side"] == "BUY"
    assert res.status == "filled" and res.vwap == Decimal("100.00") and not res.crossed
    assert res.requested == Decimal("0.037000") and res.fee == Decimal("0.0037")


def test_maker_repegs_after_5_min_when_best_moves(live):
    fx, broker = live
    clock = Clock()
    ex = make(broker, clock)
    orig_status = broker.status

    def status(cid):
        if clock.t - 1000 > 300 and fx.book["bid"] == "100.00":
            fx.book["bid"] = "99.50"  # best moved → re-peg expected
        if cid.endswith("B2"):
            fx.fill(cid)
        return orig_status(cid)

    broker.status = status
    [res] = ex.execute([Intent("BTCUSDT", "BUY", notional=Decimal("3.7"))], tag="t")
    assert res.orders == ["t-BTCUSDT-B1", "t-BTCUSDT-B2"]
    assert fx.orders["t-BTCUSDT-B2"]["price"] == "99.50"
    assert ("DELETE", "/v1/account/orders/t-BTCUSDT-B1") in fx.calls
    assert res.status == "filled" and not res.crossed


def test_maker_crosses_at_20_min_with_cap(live):
    fx, broker = live
    clock = Clock()
    ex = make(broker, clock)
    orig_status = broker.status

    def status(cid):
        if cid.endswith("B2"):
            fx.fill(cid)
        return orig_status(cid)

    broker.status = status
    [res] = ex.execute([Intent("BTCUSDT", "BUY", notional=Decimal("3.7"))], tag="t")
    cross = fx.orders["t-BTCUSDT-B2"]
    assert cross["price"] == "100.40"  # ask 100.20 × 1.002 = 100.4004 → 100.40 (round down)
    assert res.crossed and res.status == "filled"
    assert clock.t - 1000 >= 1200


def test_maker_respects_allow_cross_false_and_keeps_partial(live):
    fx, broker = live
    clock = Clock()
    ex = make(broker, clock)
    orig_status = broker.status

    def status(cid):
        if fx.orders[cid]["executedQty"] == "0":
            fx.orders[cid]["executedQty"] = "0.010000"  # partial, still active
            fx.orders[cid]["executedPrice"] = "100.00"
        return orig_status(cid)

    broker.status = status
    [res] = ex.execute(
        [Intent("BTCUSDT", "BUY", notional=Decimal("3.7"))], tag="t", allow_cross=False
    )
    assert res.orders == ["t-BTCUSDT-B1"] and not res.crossed
    assert res.status == "partial" and res.filled == Decimal("0.01")
    assert res.note == "maker deadline, no cross"


# --- through -------------------------------------------------------------------------------


def test_through_prices_beyond_opposite_best_and_retries_once(live):
    fx, broker = live
    clock = Clock()
    ex = make(broker, clock, mode="through")
    [res] = ex.execute([Intent("BTCUSDT", "SELL", qty=Decimal("0.05"))], tag="t")
    assert fx.orders["t-BTCUSDT-S1"]["price"] == "99.80"  # bid 100 × 0.998, round up
    assert res.orders == ["t-BTCUSDT-S1", "t-BTCUSDT-S2"] and res.status == "none"
    assert res.crossed and clock.t - 1000 >= 120


def test_stop_exit_crosses_even_in_maker_executor(live):
    fx, broker = live
    clock = Clock()
    ex = make(broker, clock)
    orig_status = broker.status
    broker.status = lambda cid: (fx.fill(cid), orig_status(cid))[1]
    [res] = ex.execute([Intent("BTCUSDT", "SELL", qty=Decimal("0.05"))], tag="t", mode="through")
    assert res.crossed and res.status == "filled" and res.vwap == Decimal("99.80")


# --- paper broker -------------------------------------------------------------------------


def test_paper_broker_resting_buy_fills_when_ask_reaches_price():
    book = {"bid": [{"price": "100.00"}], "ask": [{"price": "100.20"}]}
    pb = PaperBroker(lambda s: book)
    clock = Clock()
    ex = make(pb, clock)
    polls = {"n": 0}
    orig = pb.status

    def status(cid):
        polls["n"] += 1
        if polls["n"] == 3:
            book["ask"] = [{"price": "99.90"}]
        return orig(cid)

    pb.status = status
    [res] = ex.execute([Intent("BTCUSDT", "BUY", notional=Decimal("3.7"))], tag="p")
    assert res.status == "filled" and res.vwap == Decimal("99.90") and not res.crossed
    assert res.fee == Decimal("0.037") * Decimal("99.90") * Decimal("0.0013")


def test_paper_broker_through_fills_immediately_at_touch():
    book = {"bid": [{"price": "100.00"}], "ask": [{"price": "100.20"}]}
    pb = PaperBroker(lambda s: book)
    [res] = make(pb, Clock(), mode="through").execute(
        [Intent("BTCUSDT", "SELL", qty=Decimal("0.05"))], tag="p"
    )
    assert res.status == "filled" and res.vwap == Decimal("100.00") and res.crossed


def test_below_min_notional_is_not_placed():
    pb = PaperBroker(lambda s: {"bid": [{"price": "100"}], "ask": [{"price": "101"}]})
    [res] = make(pb, Clock()).execute([Intent("BTCUSDT", "BUY", notional=Decimal("0.5"))], tag="p")
    assert res.status == "none" and res.orders == [] and res.note == "below min size"
