"""Runner tests: fake Wallex (UDF candles, depth, markets) over MockTransport, fake clock."""

import csv
import json

import httpx
import numpy as np
import pandas as pd

from wallex import WallexClient
from wallex.run import SYMBOLS, Runner, bound_ai, to_30m
from wallex.strategy import g
from wallex.validate_sma import PAIRS, sma_book

NOW = int(pd.Timestamp("2026-10-22 06:00").timestamp())  # a bar close


class Clock:
    def __init__(self, t):
        self.t = float(t)

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def price(sym, t):
    """Up-trend for BTC/ETH/XRP (gate on), down-trend for SOL/BNB (gate off)."""
    base = {"BTC": 80000, "ETH": 3000, "XRP": 2.0, "SOL": 150, "BNB": 600}[sym[:3]]
    drift = 0.02 if sym[:3] in ("BTC", "ETH", "XRP") else -0.02
    days = (t - NOW) / 86400
    return base * (1 + drift) ** days * (1 + 0.001 * np.sin(t / 3600))


class FakeWallex:
    def __init__(self, clock):
        self.clock = clock
        self.calls = 0

    def __call__(self, req: httpx.Request) -> httpx.Response:
        self.calls += 1
        q = req.url.params
        ok = lambda r: httpx.Response(200, json={"success": True, "message": "", "result": r})  # noqa: E731
        if req.url.path == "/v1/udf/history":
            step = 60 if q["resolution"] == "1" else 3600
            a, b = int(q["from"]), min(int(q["to"]), int(self.clock()) - step)
            ts = list(range(-(-a // step) * step, b + 1, step))
            c = [str(price(q["symbol"], t + step)) for t in ts]  # close at the bar end
            return httpx.Response(200, json={"s": "ok", "t": ts, "c": c, "o": c, "h": c, "l": c})
        if req.url.path == "/v1/depth":
            p = price(q["symbol"], self.clock())
            return ok(
                {
                    "bid": [{"price": f"{p * 0.9995:.4f}", "quantity": "10"}],
                    "ask": [{"price": f"{p * 1.0005:.4f}", "quantity": "10"}],
                }
            )
        if req.url.path == "/v1/markets":
            m = {"tickSize": 4, "stepSize": 6, "minNotional": "1", "minQty": "0"}
            return ok({"symbols": {s: m for s in SYMBOLS}})
        return httpx.Response(404, json={"success": False, "message": "nope"})


def make(tmp_path, run_id=None, clock=None):
    clock = clock or Clock(NOW + 20)
    client = WallexClient(api_key="k", transport=httpx.MockTransport(FakeWallex(clock)))
    r = Runner(
        client, run_id=run_id or "t1", state_dir=tmp_path, equity=100.0, clock=clock,
        sleep=clock.sleep,
    )  # fmt: skip
    return r, clock


def test_g_matches_validated_sma_book():
    idx = pd.date_range("2026-01-01", periods=2000, freq="30min")
    rng = np.random.default_rng(1)
    closes = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.004, len(idx)))), index=idx)
    w = g.votes(closes).sum(axis=1) / len(g.PAIRS)
    ref = sma_book(closes, PAIRS["G"][1], band=0.003, gate=False)
    np.testing.assert_allclose(w.to_numpy(), ref)


def test_gate_uses_daily_close():
    idx = pd.date_range("2026-09-20", "2026-10-22 05:00", freq="1h")
    hourly = pd.Series(np.linspace(100, 110, len(idx)), index=idx)
    on, ret = g.gate(hourly, pd.Timestamp("2026-10-22 06:00"))
    expect = hourly[pd.Timestamp("2026-10-21 23:00")] / hourly[pd.Timestamp("2026-09-23 23:00")]
    assert on and abs(ret - (expect - 1)) < 1e-12
    assert g.gate(hourly.iloc[-100:], pd.Timestamp("2026-10-22 06:00")) is None


def test_to_30m_drops_open_bar():
    idx = pd.date_range("2026-10-22 05:00", "2026-10-22 06:10", freq="1min")
    s = to_30m(pd.Series(range(len(idx)), index=idx, dtype=float), pd.Timestamp("2026-10-22 06:00"))
    assert list(s.index.strftime("%H:%M")) == ["05:00", "05:30"]
    assert s.iloc[-1] == 59  # close of the 05:59 minute


def test_shadow_cycle_trades_logs_and_resumes(tmp_path):
    r, clock = make(tmp_path)
    r.run(max_cycles=1)
    rec = json.loads((tmp_path / "t1" / "cycles.jsonl").read_text().splitlines()[0])
    assert rec["stale"] == []
    sig = rec["signals"]
    assert sig["BTCUSDT"]["gate_on"] and sig["BTCUSDT"]["weight"] == 1.0
    assert not sig["SOLUSDT"]["gate_on"] and sig["SOLUSDT"]["weight"] == 0.0
    buys = {i["symbol"] for i in rec["intents"]}
    assert buys == {"BTCUSDT", "ETHUSDT", "XRPUSDT"}
    assert "6/6 SMA pairs up" in rec["reasons"]["BTCUSDT"]
    rows = list(csv.DictReader(open(tmp_path / "t1" / "trades.csv")))
    assert {x["symbol"] for x in rows} == buys and all(x["status"] == "filled" for x in rows)
    st = json.loads((tmp_path / "t1.json").read_text())
    assert st["cycles"] == 1 and 0 < st["qty"]["BTCUSDT"] and st["cash"] < 45
    assert abs(rec["equity_after"] - 100) < 1  # fees + spread only

    # resume after a "kill": same run id, next bar; holdings carried, no re-buy
    clock.t = NOW + 1800 + 20
    r2, _ = make(tmp_path, run_id="t1", clock=clock)
    assert r2.state["cycles"] == 1 and r2.risk.slot("BTCUSDT").qty > 0
    r2.run(max_cycles=1)
    lines = (tmp_path / "t1" / "cycles.jsonl").read_text().splitlines()
    assert len(lines) == 2 and json.loads(lines[1])["intents"] == []
    assert "resumed run t1" in (tmp_path / "t1" / "alerts.log").read_text()


def test_kill_switch_flattens(tmp_path):
    r, clock = make(tmp_path)
    r.run(max_cycles=1)
    r.state["cash"] = 0.0  # equity drops > 10 % from the peak
    clock.t = NOW + 1800 + 20
    r.run(max_cycles=1)
    assert r.risk.killed
    rec = json.loads((tmp_path / "t1" / "cycles.jsonl").read_text().splitlines()[-1])
    assert {x["side"] for x in rec["results"]} == {"SELL"}
    assert all(v < 1e-9 for v in r.state["qty"].values())


def _sig(coin, up, gate_on):
    v = {f"p{i}": float(i < up) for i in range(6)}
    return g.Signal(coin, v, up / 6, gate_on, 0.01 if gate_on else -0.02, up / 6 * gate_on)


def test_bound_ai_caps_gate_off_entries():
    sigs = {
        "BTCUSDT": _sig("BTC", 6, False),
        "ETHUSDT": _sig("ETH", 6, False),
        "XRPUSDT": _sig("XRP", 5, False),
        "SOLUSDT": _sig("SOL", 4, False),  # < 5/6 votes: AI may not enter
        "BNBUSDT": _sig("BNB", 3, True),
    }
    prop = {s: (0.9, "trend") for s in SYMBOLS} | {"BNBUSDT": (1.5, "strong")}
    out = bound_ai(sigs, prop)
    assert out["BNBUSDT"][:2] == (1.0, "AI")  # gate on: ≤ 1 slot
    off = {s: out[s][0] for s in ("BTCUSDT", "ETHUSDT", "XRPUSDT")}
    assert all(w <= 0.5 for w in off.values()) and abs(sum(off.values()) - 1.5) < 1e-9
    assert out["SOLUSDT"][:2] == (0.0, "AI (no entry)") and "no entry: trend" in out["SOLUSDT"][2]
    assert bound_ai(sigs, {})["BNBUSDT"][0] == 0.5  # no proposal → pure G


def test_officer_failure_falls_back_to_pure_g(tmp_path):
    clock = Clock(NOW + 20)
    client = WallexClient(api_key="k", transport=httpx.MockTransport(FakeWallex(clock)))

    def boom(sigs, ctx):
        raise RuntimeError("llm down")

    r = Runner(client, run_id="t2", state_dir=tmp_path, equity=100.0, officer=boom,
               clock=clock, sleep=clock.sleep)  # fmt: skip
    r.run(max_cycles=1)
    rec = json.loads((tmp_path / "t2" / "cycles.jsonl").read_text())
    assert {i["symbol"] for i in rec["intents"]} == {"BTCUSDT", "ETHUSDT", "XRPUSDT"}
    assert "officer failed" in (tmp_path / "t2" / "alerts.log").read_text()


def test_live_equity_counts_oil_and_locked(tmp_path):
    """live-v2: USOON (oil agent) and locked balances are in the account equity; G sizes on
    its own share; a reading with anything locked never feeds the kill/pause switches."""
    clock = Clock(NOW + 20)
    fake = FakeWallex(clock)
    bal = {"USDT": {"value": "12", "locked": "0"}, "BTC": {"value": "0.0001", "locked": "0"},
           "USOON": {"value": "2", "locked": "0"}}  # fmt: skip

    def handler(req):
        if req.url.path == "/v1/account/balances":
            return httpx.Response(
                200, json={"success": True, "message": "", "result": {"balances": bal}}
            )
        if req.url.path == "/v1/depth" and req.url.params["symbol"] == "USOONUSDT":
            book = {
                "bid": [{"price": "2.99", "quantity": "9"}],
                "ask": [{"price": "3.01", "quantity": "9"}],
            }
            return httpx.Response(200, json={"success": True, "message": "", "result": book})
        return fake(req)

    client = WallexClient(api_key="k", transport=httpx.MockTransport(handler))
    r = Runner(client, run_id="lv", state_dir=tmp_path, clock=clock, sleep=clock.sleep)
    r.mode = "live"
    prices = {s: 1.0 for s in SYMBOLS} | {"BTCUSDT": 80000.0}
    qty, cash, eq = r.holdings(prices)
    assert cash == 12 and qty["BTCUSDT"] == 0.0001
    assert r.extra_value == 6.0 and not r.locked
    assert eq == 12 + 8 + 6
    # Wallex value is the total and includes locked (live probe 2026-10-22)
    bal["USDT"] = {"value": "12", "locked": "6"}  # oil buy resting: 6 USDT locked
    qty, cash, eq = r.holdings(prices)
    assert r.locked and cash == 6 and eq == 26
    bal["USDT"] = {"value": "12", "locked": "0"}
    bal["BTC"] = {"value": "0.0001", "locked": "0.00004"}  # G sell resting
    bal["USOON"] = {"value": "2", "locked": "2"}  # oil sell resting
    qty, cash, eq = r.holdings(prices)
    assert r.locked and qty["BTCUSDT"] == 0.0001 - 0.00004 and cash == 12
    assert r.extra_value == 6.0 and eq == 26
