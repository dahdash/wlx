from decimal import Decimal

import numpy as np

from wallex.exec import Executor, MarketRules, PaperBroker
from wallex.risk import RiskConfig, RiskState, Slot, flatten


def test_slot_vwap_and_stop_with_cooldown():
    rs = RiskState()
    s = rs.slot("BTCUSDT")
    s.add_fill("BUY", Decimal(1), Decimal(100))
    s.add_fill("BUY", Decimal(1), Decimal(110))
    assert s.entry_vwap == Decimal(105)
    assert rs.check_stops({"BTCUSDT": Decimal("103.0")}, bar=10) == []
    assert rs.check_stops({"BTCUSDT": Decimal("102.9")}, bar=10) == ["BTCUSDT"]
    assert rs.apply({"BTCUSDT": 3.0}, 18.5, bar=14) == {"BTCUSDT": 0.0}  # cool-down 4 bars
    assert rs.apply({"BTCUSDT": 3.0}, 18.5, bar=15) == {"BTCUSDT": 3.0}


def test_ai_can_only_tighten_stop():
    rs = RiskState()
    rs.tighten_stop("X", 0.015)
    assert rs.slot("X").stop == 0.015
    rs.tighten_stop("X", 0.05)  # loosening ignored
    assert rs.slot("X").stop == 0.015
    rs.tighten_stop("X", 0.001)  # floor 1 %
    assert rs.slot("X").stop == 0.01


def test_crash_gate():
    rs = RiskState()
    rng = np.random.default_rng(0)
    calm = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, 200)))
    assert not rs.crash_gate(calm)
    crash = calm.copy()
    crash[-24:] *= np.linspace(1, 0.9, 24)  # −10 % over the last 24 bars
    assert rs.crash_gate(crash)
    assert not rs.crash_gate(calm[:50])  # too short → no gate


def test_pause_and_kill():
    rs = RiskState(cfg=RiskConfig())
    for b in range(7):
        assert rs.update_equity(b, 100.0) is None
    assert rs.update_equity(7, 95.9) == "pause"  # −4.1 % vs 6 bars ago
    assert rs.apply({"A": 3.0}, 95.9, bar=9) == {"A": 0.0}
    assert rs.apply({"A": 3.0}, 95.9, bar=10) == {"A": 3.0}
    assert rs.update_equity(11, 89.9) == "kill"
    assert rs.killed and rs.apply({"A": 3.0}, 89.9, bar=12) == {"A": 0.0}


def test_apply_caps_position_and_slots_and_crash():
    rs = RiskState()
    targets = {s: 10.0 for s in "ABCDEF"}
    out = rs.apply(targets, 18.5, bar=0, crash={"B": True}, paused_symbols={"C"})
    assert len(out) == 5 and "F" not in out
    assert out["A"] == 18.5 * 0.2 and out["B"] == 0.0 and out["C"] == 0.0


def test_flatten_cancels_and_sells_through():
    book = {"bid": [{"price": "100.00"}], "ask": [{"price": "100.20"}]}
    pb = PaperBroker(lambda s: book)
    rules = {s: MarketRules(2, 6, Decimal(1)) for s in ("BTCUSDT", "ETHUSDT")}
    pb.place("BTCUSDT", "BUY", Decimal("0.01"), Decimal("99"), "old-1")  # resting order
    ex = Executor(pb, rules, clock=lambda: 0.0, sleep=lambda s: None)
    res = flatten(
        ex,
        {"BTCUSDT": Decimal("0.05"), "ETHUSDT": Decimal(0)},
        tag="kill",
        open_order_ids=("old-1",),
    )
    assert ("cancel", "old-1") in pb.log and not pb.status("old-1").active
    assert [r.symbol for r in res] == ["BTCUSDT"]
    assert res[0].status == "filled" and res[0].crossed and res[0].vwap == Decimal("100.00")
    assert Slot(qty=Decimal(1)).qty == 1
