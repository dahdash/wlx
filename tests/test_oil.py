"""Oil agent tests: fake feeds, fake ``claude -p``, fake clock; paper fills only, no network."""

import json
import subprocess
from decimal import Decimal

import httpx
import pandas as pd
import pytest

from wallex import oil
from wallex.exec import Executor, MarketRules, OrderState, PaperBroker

T0 = pd.Timestamp("2026-10-22 12:00").timestamp()  # Thursday, inside entry hours
RULES = MarketRules(tick=2, step=4, min_notional=Decimal(1), min_qty=Decimal("0.0001"))


class Clock:
    def __init__(self, feeds):
        self.t, self.feeds = T0, feeds

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s
        if self.feeds.next_book:  # the market moves while the executor waits
            self.feeds.book, self.feeds.next_book = self.feeds.next_book, None


class FakeFeeds:
    def __init__(self):
        self.book, self.next_book, self.clock, self.fail = (160.0, 160.4), None, None, False
        self.wti_lead = 0.0  # WTI 4-h return over the last 16 bars (entry gap floor)

    def depth(self):
        if self.fail:
            raise oil.WallexError("down")
        b, a = self.book
        return {
            "bid": [{"price": str(b), "quantity": "1"}],
            "ask": [{"price": str(a), "quantity": "1"}],
        }

    def usoon_15m(self, end):
        t = [int(self.clock()) - 900 * i for i in range(100, 0, -1)]
        return t, [160.0] * 99 + [self.book[0]]

    def ref_15m(self):
        t = [int(self.clock()) - 900 * i for i in range(100, 0, -1)]
        c = [99.0] * 84 + [99.0 * (1 + self.wti_lead)] * 16
        return t, c, {"mark": c[-1], "index": c[-1], "time_utc": "x"}


NO_REFLECTION = {"lesson_updates": [], "merges": [], "new_lessons": [], "summary": "nothing new"}


class FakeRun:
    """Scripted ``claude -p``: decision calls pop ``answers``; reflection calls (recognised by
    their schema) go to ``reflects`` and answer ``reflection``."""

    def __init__(self, *answers):
        self.answers, self.calls, self.reflects = list(answers), [], []
        self.reflection = NO_REFLECTION

    def __call__(self, cmd, **kw):
        reflecting = json.dumps(oil.REFLECT_SCHEMA) in cmd
        (self.reflects if reflecting else self.calls).append((cmd, kw))
        a = self.reflection if reflecting else self.answers.pop(0)
        if isinstance(a, Exception):
            raise a
        env = {"type": "result", "subtype": "success", "is_error": False, "structured_output": a,
               "num_turns": 3, "total_cost_usd": 0.1}  # fmt: skip
        return subprocess.CompletedProcess(cmd, 0, json.dumps(env), "")


def buy(hid=None, gross=1.2, published="2026-10-22T11:00:00Z", horizon=4):
    return {
        "decision": "BUY_CANDIDATE", "hypothesis_id": hid, "horizon_hours": horizon,
        "thesis": "EIA draw larger than expected, USOON lags WTI",
        "evidence": [{"id": "e1", "kind": "fact", "title": "EIA WPSR", "source_url": "https://eia.gov/x",
                      "published_utc": published}],
        "supporting_evidence_ids": ["e1"], "counter_evidence_ids": [], "known_unknowns": [],
        "catalyst_or_reason": "EIA draw", "invalidation_conditions": ["WTI below 98"],
        "expiry_utc": "2026-10-22T16:00:00Z", "entry_conditions": "", "exit_conditions": "",
        "estimated_gross_opportunity_pct": gross, "hypothesis_updates": [], "exit_now": False,
    }  # fmt: skip


def answer(decision, **kw):
    return {"decision": decision, "thesis": "t", "evidence": [], "supporting_evidence_ids": [],
            "counter_evidence_ids": [], "known_unknowns": [], "catalyst_or_reason": "",
            "invalidation_conditions": [], "estimated_gross_opportunity_pct": None,
            "exit_now": False} | kw  # fmt: skip


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(oil, "STATE_DIR", tmp_path)

    def write_g(gates_on=False, t=T0 - 600):
        d = tmp_path / "live-1"
        d.mkdir(exist_ok=True)
        sig = {"gate_on": gates_on, "gate_ret": -0.01, "votes": 0, "weight": 0, "reason": ""}
        rec = {"bar_close": pd.Timestamp(t, unit="s").isoformat(),
               "signals": {s: sig for s in oil.G_COINS}, "equity_after": 18.5}  # fmt: skip
        (d / "cycles.jsonl").write_text(json.dumps(rec) + "\n")

    write_g()

    def make(*answers, run_id="oil-t", cfg=None):
        feeds = FakeFeeds()
        clock = Clock(feeds)
        feeds.clock = clock
        run = FakeRun(*answers)
        a = oil.OilAgent(None, run_id=run_id, feeds=feeds, rules=RULES, run=run, clock=clock,
                         sleep=clock.sleep, state_dir=tmp_path, cfg=cfg)  # fmt: skip
        return a, feeds, clock, run

    return make, write_g, tmp_path


def test_round_trip_cost():
    assert oil.round_trip_cost(0.004) == pytest.approx(0.0010 + 0.0013 + 0.004)
    assert oil.round_trip_cost(0.0, "taker", "taker") == pytest.approx(0.0026)


def test_live_refused_unless_armed_capped_and_env(tmp_path, monkeypatch):
    kw = dict(rules=RULES, feeds=FakeFeeds(), state_dir=tmp_path)
    with pytest.raises(SystemExit, match="unknown mode"):
        oil.OilAgent(None, mode="micro_live", **kw)
    with pytest.raises(SystemExit, match="--arm"):
        oil.OilAgent(None, mode="live", **kw)
    with pytest.raises(SystemExit, match="hard cap"):
        oil.OilAgent(None, mode="live", armed=True, cfg=oil.OilConfig(notional=10.0), **kw)
    monkeypatch.delenv("WALLEX_LIVE", raising=False)
    with pytest.raises(RuntimeError, match="WALLEX_LIVE"):
        oil.OilAgent(None, mode="live", armed=True, **kw)
    with pytest.raises(SystemExit):
        oil.main(["--mode", "micro_live"])


class FakeAccount:
    """Wallex account for live-mode tests: USOON balance and open orders."""

    def __init__(self, usoon="0", open_cids=(), locked="0"):
        self.usoon, self.open = usoon, [{"clientOrderId": c} for c in open_cids]
        self.locked = locked  # Wallex: value is the total and includes locked

    def balances(self):
        return {"balances": {"USOON": {"value": self.usoon, "locked": self.locked}}}

    def open_orders(self, symbol):
        return {"orders": self.open}


class CancelLog(PaperBroker):
    def __init__(self, depth_fn):
        super().__init__(depth_fn)
        self.cancelled = []

    def cancel(self, cid):
        if cid not in self._state:
            self.cancelled.append(cid)
            return OrderState(cid, active=False)
        return super().cancel(cid)


LIVE_POS = {
    "qty": "0.0187",
    "entry": 160.0,
    "fee": 0.0,
    "opened_utc": "2026-10-22T11:00:00",
    "expiry_utc": None,
    "hypothesis_id": "h1",
    "thesis": "t",
    "invalidation": [],
}


def live_agent(tmp_path, acct, state=None):
    feeds = FakeFeeds()
    clock = Clock(feeds)
    feeds.clock = clock
    if state is not None:
        (tmp_path / "oil-l.json").write_text(json.dumps(state))
    br = CancelLog(lambda s: feeds.depth())
    ex = Executor(br, {oil.SYMBOL: RULES}, clock=clock, sleep=clock.sleep)
    a = oil.OilAgent(acct, mode="live", armed=True, run_id="oil-l", feeds=feeds, rules=RULES,
                     executor=ex, clock=clock, sleep=clock.sleep, state_dir=tmp_path)  # fmt: skip
    return a, br, feeds, clock


def test_live_reconcile_on_restart(tmp_path):
    pos = dict(LIVE_POS)
    # crash mid-order: our resting order is cancelled, the size follows the balance (coin fee)
    acct = FakeAccount("0.01868", ["oil-oil-l-h2-USOONUSDT-B1", "manual-1"])
    a, br, *_ = live_agent(tmp_path, acct, {"position": pos, "pending": "oil-oil-l-h2"})
    assert br.cancelled == ["oil-oil-l-h2-USOONUSDT-B1"]
    assert a.state["position"]["qty"] == "0.0186" and a.state["pending"] is None
    assert a.state["paused_reason"] is None
    # position in state but nothing on the account: cleared
    a, *_ = live_agent(tmp_path, FakeAccount("0"), {"position": pos})
    assert a.state["position"] is None
    # USOON on the account without a position: adopted at the bid, exits manage it
    (tmp_path / "oil-l.json").unlink()
    a, *_ = live_agent(tmp_path, FakeAccount("0.02"))
    assert a.state["position"]["hypothesis_id"] == "adopted"
    assert a.state["position"]["entry"] == 160.0
    a.client.locked = "0.02"  # a resting sell: value already counts it
    assert a.held() == Decimal("0.0200")


def test_live_exit_never_sells_more_than_held(tmp_path):
    pos = dict(LIVE_POS)
    acct = FakeAccount("0.0187")
    a, br, *_ = live_agent(tmp_path, acct, {"position": pos})
    acct.usoon = "0.0150"  # e.g. part sold by an earlier run that crashed before saving
    r = a.exit("test")
    assert Decimal(str(r["filled"])) == Decimal("0.0150") and a.state["position"] is None


def _gate(dec, book=None, age=1.0, g=None, st=None, symbol=oil.SYMBOL, cfg=None, gap=0.03):
    now = pd.Timestamp(T0, unit="s")
    book = book or oil.book_stats(FakeFeeds().depth(), 0.005)
    g = g or {"all_off": True, "status": "ok"}
    return oil.gate(oil.validate(dec, now, False), book, age, now - pd.Timedelta(minutes=10), g,
                    st or {}, cfg or oil.OilConfig(), RULES, now, symbol, gap)  # fmt: skip


def test_gate_accepts_good_candidate_and_sizes():
    ok, why, qty = _gate(buy())
    assert ok, why
    assert qty == Decimal("0.0187")  # 3 USDT / 160.4, step 4, ≥ minNotional


def test_gate_rejections():
    assert "symbol" in " ".join(_gate(buy(), symbol="BTCUSDT")[1])
    assert any("stale quote" in w for w in _gate(buy(), age=120)[1])
    wide = oil.book_stats({"bid": [{"price": "160", "quantity": "1"}],
                           "ask": [{"price": "161", "quantity": "1"}]}, 0.005)  # fmt: skip
    assert any("spread" in w for w in _gate(buy(), book=wide)[1])
    assert any("G not all" in w for w in _gate(buy(), g={"all_off": None, "status": "stale"})[1])
    assert any("net edge" in w for w in _gate(buy(gross=0.3))[1])
    assert any("horizon" in w for w in _gate(buy(horizon=24))[1])
    assert any("look-ahead" in w for w in _gate(buy(published="2026-10-22T13:00:00Z"))[1])
    assert any(
        "below market minimum" in w for w in _gate(buy(), cfg=oil.OilConfig(notional=0.5))[1]
    )
    day = {"date": "2026-10-22", "entries": 2, "pnl": 0.0}
    assert any("daily entry" in w for w in _gate(buy(), st={"day": day})[1])
    assert any("kill" in w for w in _gate(buy(), st={"killed": True})[1])


def test_gate_wti_gap_floor():
    assert any("gap unknown" in w for w in _gate(buy(), gap=None)[1])
    assert any("4h gap 1.00% < 2.0%" in w for w in _gate(buy(), gap=0.01)[1])
    ok, why, _ = _gate(buy(), gap=0.016, cfg=oil.PROFILES["aggressive"])
    assert not any("gap" in w for w in why), why
    assert any("gap" in w for w in _gate(buy(), gap=0.014, cfg=oil.PROFILES["aggressive"])[1])


def test_validate_rejects_malformed():
    now = pd.Timestamp(T0, unit="s")
    for bad in (None, {"decision": "SELL"}, "BUY"):
        with pytest.raises(oil.AgentError):
            oil.validate(bad, now, False)
    with pytest.raises(oil.AgentError):
        oil.validate(answer("EXIT_REVIEW"), now, holding=False)


def test_no_agent_call_while_g_gates_on(env):
    make, write_g, _ = env
    write_g(gates_on=True)
    a, *_, run = make()
    rec = a.tick()
    assert rec["analysis"] is None and run.calls == []


def test_stale_g_state_blocks_analysis(env):
    make, write_g, _ = env
    write_g(t=T0 - 7200)
    a, *_, run = make()
    assert a.tick()["analysis"] is None and run.calls == []


def test_agent_failure_is_no_trade(env):
    make, *_ = env
    a, *_ = make(subprocess.TimeoutExpired("claude", 300))
    rec = a.tick()
    assert rec["analysis"] == "scheduled sweep"
    assert rec["decision"]["decision"] == "NO_TRADE" and a.state["position"] is None


def test_buy_fill_stop_exit_and_idempotency(env):
    make, _, d = env
    a, feeds, clock, run = make(buy())
    feeds.wti_lead = 0.03  # WTI 3 % ahead of USOON → gap floor passes
    feeds.next_book = (159.9, 160.0)  # ask falls to our resting bid → maker fill at 160.0
    rec = a.tick()
    assert rec["gate"]["ok"], rec["gate"]
    pos = a.state["position"]
    assert pos["hypothesis_id"] == "h1" and Decimal(pos["qty"]) == Decimal("0.0187")
    assert "--allowedTools" in run.calls[0][0] and "WebSearch,WebFetch" in run.calls[0][0]
    cmd = run.calls[0][0]
    assert cmd[cmd.index("--tools") + 1] == "WebSearch,WebFetch"  # no Bash/Edit/Write
    assert "--strict-mcp-config" in cmd and "--mcp-config" not in cmd  # no Wallex MCP tools
    feeds.book = (157.0, 157.2)  # −1.9 % on the bid → deterministic stop
    clock.t += 300
    rec = a.tick()
    assert rec["exit"]["reason"].startswith("stop") and a.state["position"] is None
    assert a.state["loss_streak"] == 1 and a.state["day"]["pnl"] < 0
    assert (d / "oil-t" / "trades.csv").read_text().count("\n") == 3
    # the same thesis can never be bought twice
    feeds.book, clock.t = (160.0, 160.4), clock.t + 3600
    run.answers.append(buy(hid="h1"))
    rec = a.tick(force=True)
    assert not rec["gate"]["ok"] and any("already traded" in w for w in rec["gate"]["why"])


def test_maker_entry_not_filled_is_recorded(env):
    make, *_ = env
    a, feeds, *_ = make(buy())
    feeds.wti_lead = 0.03  # WTI 3 % ahead of USOON → gap floor passes
    rec = a.tick()
    assert rec["gate"]["ok"] and a.state["position"] is None
    assert "not filled" in rec["gate"]["why"][0]
    assert a.state["entered"] == ["h1"] and a.state["pending"] is None


def test_restart_with_pending_order_pauses(env):
    make, _, d = env
    a, *_ = make()
    a.state["pending"] = "oil-oil-t-h1"
    a._save()
    b, *_ = make(buy())
    assert "unreconciled" in b.state["paused_reason"]
    assert b.tick()["analysis"] is None  # paused and flat: no agent spend
    rec = b.tick(force=True)
    assert not rec["gate"]["ok"] and any("paused" in w for w in rec["gate"]["why"])
    assert "pending order" in (d / "oil-t" / "alerts.log").read_text()


def test_kill_file_flattens_and_stops(env):
    make, _, d = env
    a, feeds, clock, run = make(buy())
    feeds.wti_lead = 0.03  # WTI 3 % ahead of USOON → gap floor passes
    feeds.next_book = (159.9, 160.0)
    a.tick()
    assert a.state["position"]
    (d / "oil-t" / "KILL").touch()
    clock.t += 300
    rec = a.tick(force=True)
    assert rec["exit"]["reason"] == "kill switch" and a.state["position"] is None
    assert "analysis" not in rec and len(run.calls) == 1


def test_data_failures_pause_entries(env):
    make, *_ = env
    a, feeds, clock, run = make()
    feeds.fail = True
    for _ in range(3):
        a.tick()
        clock.t += 300
    assert "without data" in a.state["paused_reason"] and run.calls == []


def test_forward_eval_lessons_and_track_record(env):
    make, _, d = env
    a, feeds, clock, run = make(answer("NO_TRADE", lesson="EIA draws were priced in by 15 UTC"))
    a.tick(force=True)
    ev = a.state["evals"][0]
    assert ev["decision"] == "NO_TRADE" and ev["ask"] == 160.4 and ev["net"] == {}
    assert a.state["lessons"][0][1].startswith("EIA draws")
    feeds.book = (162.0, 162.4)
    for _ in range(13):  # 65 min later: the 1-h mark is filled from the current bid
        clock.t += 300
        a.tick()
    assert ev["net"]["1"] == oil.forward_net(160.4, 162.0, "maker") and "2" not in ev["net"]
    assert ev["best_net"] > 0.0070
    run.answers.append(answer("WATCH"))
    clock.t += 3 * 3600 + 600
    a.tick(force=True)  # 4 h after the first decision: scored, and shown to the agent
    tr = json.loads(run.calls[-1][1]["input"].split("Snapshot:\n", 1)[1])["track_record"]
    assert tr["by_decision_4h"]["NO_TRADE"]["n"] == 1 and tr["lessons"][0]["id"] == "L1"
    assert tr["lessons"][0]["text"].startswith("EIA") and "by_decision_4h_all_runs" in tr
    r = oil.report("oil-t", d)
    assert r["decisions"]["NO_TRADE"] == 1 and r["missed_moves_gt_0.5pct"] == 1


def test_aggressive_profile_taker_entry_and_closed_trade(env):
    make, *_ = env
    cfg = oil.AGGRESSIVE
    assert cfg.entry_kind == "taker" and cfg.notional == 6.0
    assert oil.round_trip_cost(0.004, cfg.entry_kind) == pytest.approx(0.0026 + 0.004)
    a, feeds, clock, run = make(buy(gross=1.5), cfg=cfg)
    feeds.wti_lead = 0.03  # WTI 3 % ahead of USOON → gap floor passes
    rec = a.tick()  # no maker fill: crosses after cross_after_s
    assert rec["gate"]["ok"], rec["gate"]
    assert a.state["position"] and Decimal(a.state["position"]["qty"]) == Decimal("0.0375")
    feeds.book = (165.0, 165.4)
    clock.t = pd.Timestamp("2026-10-22 16:00").timestamp()  # thesis expiry
    rec = a.tick()
    assert rec["exit"] and a.state["closed"][0]["pnl_usdt"] > 0
    assert a.state["closed"][0]["hypothesis_id"] == "h1"


def test_contest_end_blocks_entries_and_forces_exit(env):
    make, *_ = env
    now = pd.Timestamp(T0, unit="s")
    late = oil.OilConfig(end_utc=(now + pd.Timedelta(hours=1)).isoformat())
    ok, why, _ = _gate(buy(), cfg=late)
    assert not ok and any("contest end" in w for w in why)
    soon = oil.OilConfig(end_utc=(now + pd.Timedelta(hours=2.5)).isoformat())
    ok, why, _ = _gate(buy(), cfg=soon)  # expiry 16:00 > end 14:30 + 1 h
    assert not ok and any("expiry" in w for w in why)
    pos = {"entry": 160.0, "opened_utc": now.isoformat(), "expiry_utc": None}
    assert oil.exit_reason(pos, 160.0, now + pd.Timedelta(hours=1.1), late).startswith("contest")


# --- T14.6 reliability ------------------------------------------------------------------


def test_agent_quick_failure_retried_with_backoff(env):
    make, *_ = env
    a, _, clock, run = make(oil.AgentError("rate limited"), buy())
    t = clock.t
    rec = a.tick()
    assert len(run.calls) == 2 and clock.t - t >= 30  # one backoff nap, then success
    assert rec["decision"]["decision"] == "BUY_CANDIDATE" and a.state["agent_fail_streak"] == 0
    log = [json.loads(x) for x in (a.log_dir / "agent.jsonl").read_text().splitlines()]
    assert "error" not in log[0] and len(log[0]["attempts"]) == 1


def test_agent_retries_exhausted_fail_closed_and_alert(env):
    make, *_ = env
    cfg = oil.OilConfig(min_call_gap_s=0, sweep_s=0)
    a, _, clock, run = make(*[oil.AgentError("down")] * 9, cfg=cfg)
    for _ in range(3):
        rec = a.tick()
        assert rec["decision"]["decision"] == "NO_TRADE"
        clock.t += 300
    assert len(run.calls) == 9 and a.state["agent_fail_streak"] == 3
    assert "in a row failed" in (a.log_dir / "alerts.log").read_text()


def test_run_lock_single_instance(tmp_path):
    lock = tmp_path / "oil-t.lock"
    with oil.run_lock(lock):
        with pytest.raises(SystemExit, match="held by pid"):
            with oil.run_lock(lock):
                pass
    with oil.run_lock(lock):  # released on exit
        pass


def test_heartbeat_and_health(env):
    make, *_ = env
    a, _, clock, _ = make(answer("NO_TRADE"))
    assert oil.health(a.log_dir, clock.t)["status"] == "missing"
    a.loop(1)
    h = oil.health(a.log_dir, clock.t)
    assert h["ok"] and h["ticks"] == 1 and h["pid_alive"]
    assert oil.health(a.log_dir, clock.t + 3600)["status"] == "stale"
    hb = json.loads((a.log_dir / "heartbeat.json").read_text())
    (a.log_dir / "heartbeat.json").write_text(json.dumps(hb | {"pid": 2**22 + 1}))
    assert oil.health(a.log_dir, clock.t)["status"] == "dead"


def test_tick_failure_degrades_health_and_stall_alert(env, monkeypatch):
    make, *_ = env
    a, _, clock, _ = make(answer("NO_TRADE"))
    a.loop(1)
    monkeypatch.setattr(a, "tick", lambda force=False: 1 / 0)
    clock.t += 3600  # e.g. laptop suspended: the next tick comes an hour late
    a.loop(1)
    assert oil.health(a.log_dir, clock.t)["status"] == "degraded"
    alerts = (a.log_dir / "alerts.log").read_text()
    assert "loop stalled" in alerts and "ZeroDivisionError" in alerts


def test_state_migration_v1_and_newer_refused(env):
    make, _, d = env
    v1 = {"run_id": "oil-t", "mode": "shadow", "cash": -1.5, "trades": 2, "lessons": [["t", "x"]]}
    (d / "oil-t.json").write_text(json.dumps(v1))
    a, *_ = make()
    assert a.state["schema"] == oil.SCHEMA and a.state["trades"] == 2
    assert a.state["agent_fail_streak"] == 0 and a.state["evals"] == []
    a._save()
    assert json.loads((d / "oil-t.json").read_text())["schema"] == oil.SCHEMA
    (d / "oil-t.json").write_text(json.dumps(v1 | {"schema": oil.SCHEMA + 1}))
    with pytest.raises(SystemExit, match="newer"):
        make()


def test_daily_log_rotation_and_report(env):
    make, write_g, d = env
    a, _, clock, _ = make(answer("NO_TRADE"), answer("NO_TRADE"))
    a.tick()
    clock.t += 86400
    write_g(t=clock.t - 600)
    a.tick()
    logs = sorted(p.name for p in a.log_dir.glob("*.jsonl"))
    assert "agent.2026-10-22.jsonl" in logs and "snapshots.2026-10-22.jsonl" in logs
    assert "agent.jsonl" in logs
    assert len(oil.read_jsonl(a.log_dir, "agent.jsonl")) == 2
    assert oil.report("oil-t", d)["agent_calls"] == 2


def test_live_probe_round_trip(tmp_path):
    acct = FakeAccount("0")
    a, br, *_ = live_agent(tmp_path, acct)
    orig = br.place

    def place(symbol, side, qty, price, cid):  # the account holds what the buy filled
        orig(symbol, side, qty, price, cid)
        st = br._state[cid]
        if side == "BUY" and st.filled:
            acct.usoon = str(st.filled)
        elif side == "SELL" and st.filled:
            acct.usoon = "0"

    br.place = place
    out = a.probe(1.5)
    assert out["resting"]["after_cancel"]["filled"] == 0
    assert out["buy"]["filled"] > 0 and out["sell"]["filled"] == out["buy"]["filled"]
    assert out["held_after"] == "0.0000"
    with pytest.raises(SystemExit, match="hard cap"):
        a.probe(8.0)


# --- T14.2 analyst adapters ----------------------------------------------------------------


def snap_for(div4h=-0.025, position=None, cost=0.006):
    return {
        "as_of_utc": "2026-10-22T12:00:00", "snapshot_id": "s1", "position": position,
        "est_round_trip_cost": cost, "usoon_15m": {"ret_4h": 0.0},
        "wti_perp_15m": {"ret_4h": -div4h}, "divergence_usoon_minus_wti": {"ret_4h": div4h},
        "limits": {"min_wti_minus_usoon_gap_4h": 0.015, "min_net_edge": 0.001,
                   "horizon_h": [1.0, 12.0]},
    }  # fmt: skip


# --- T14.4 memory tiers ------------------------------------------------------------------

NOW = pd.Timestamp("2026-10-22 12:00")


def test_lesson_dedup_and_idempotent_import(tmp_path):
    mem = oil.Memory(tmp_path / "m.json")
    st = {"run_id": "r1", "evals": [], "closed": [], "lessons": [
        ["2026-10-22T10:00", "Off-hours a flat USOON quote is stale, not a lag to buy."],
        ["2026-10-22T11:00", "Off hours, a flat USOON quote is stale and not a lag to buy."],
        ["2026-10-22T12:00", "During US hours judge USOON against USO, not the CL perp."],
    ]}  # fmt: skip
    (tmp_path / "r1.json").write_text(json.dumps(st))
    assert mem.import_run(tmp_path / "r1.json", NOW)["lessons"] == 2  # near-duplicate → support
    assert mem.import_run(tmp_path / "r1.json", NOW)["lessons"] == 0  # re-import: no change
    st["run_id"] = "r2"  # a run that inherited r1's lessons: same (ts, text), no extra support
    (tmp_path / "r2.json").write_text(json.dumps(st))
    mem.import_run(tmp_path / "r2.json", NOW)
    ls = mem.load()["lessons"]
    assert [x["support"] for x in ls] == [2, 1] and ls[0]["id"] == "L1"
    assert mem.view()["lessons"][0]["id"] == "L1"  # ranked by support − contradict


def test_memory_carries_over_to_a_new_run(env):
    make, _, d = env
    a, *_ = make(answer("NO_TRADE", lesson="EIA draws were priced in by 15 UTC"), run_id="r-a")
    a.tick(force=True)
    b, _, _, run = make(answer("WATCH"), run_id="r-b")  # new run_id, same memory file
    b.tick(force=True)
    tr = json.loads(run.calls[-1][1]["input"].split("Snapshot:\n", 1)[1])["track_record"]
    assert tr["lessons"][0]["text"].startswith("EIA draws") and not b.state["lessons"]
    eps = oil.Memory(d / "memory.json").load()["episodes"]
    assert {e["run_id"] for e in eps} == {"r-a", "r-b"}


def test_reflection_retire_merge_and_auto_retire(tmp_path):
    mem = {**oil.memory.empty_memory(NOW)}
    for t in (
        "A rich USOON is no buy",
        "USOON rich vs WTI: skip",
        "Thursday EIA is priced",
        "Stale quote off-hours",
    ):
        oil.add_lesson(mem, t, None, NOW, "r1")
    mem["lessons"][3]["contradict"] = 1
    done = oil.apply_reflection(mem, {
        "lesson_updates": [{"id": "L3", "verdict": "retire", "note": "date-specific"},
                           {"id": "L4", "verdict": "contradict", "note": "moved after all"},
                           {"id": "L99", "verdict": "support", "note": "unknown id"}],
        "merges": [{"ids": ["L1", "L2"], "text": "USOON rich vs WTI is a reason not to buy",
                    "conditions": "4-h divergence > 0"}, {"ids": ["L1"], "text": "x"}],
        "new_lessons": [{"text": "Watch the EIA release time", "conditions": None}] * 3,
        "summary": "s",
    }, NOW, "r1")  # fmt: skip
    st = {x["id"]: x["status"] for x in mem["lessons"]}
    assert st == {"L1": "merged", "L2": "merged", "L3": "retired", "L4": "retired",
                  "L5": "active", "L6": "active"}  # fmt: skip
    assert mem["lessons"][4]["support"] == 2 and done["merged"] == [[["L1", "L2"], "L5"]]
    assert mem["lessons"][5]["support"] == 2  # the 3rd "n1" is over MAX_NEW, 2nd is support
    assert len(done["skipped"]) == 2 and "L4" in done["retire"]  # auto: 2 contradict > 1


def test_reflection_schedule_failure_and_limits_untouched(env):
    make, _, d = env
    a, feeds, clock, run = make(answer("NO_TRADE", lesson="Stale quote off-hours"))
    cfg0, limits = a.cfg, None
    a.tick(force=True)
    assert "reflection" not in a.tick()  # nothing due: memory just created
    for _ in range(3):
        clock.t += 3 * 3600
        run.answers.append(answer("NO_TRADE"))
        a.tick(force=True)
    new = [{"text": "Raise the notional to 50 USDT", "conditions": None}]
    run.reflection = {"lesson_updates": [], "merges": [], "summary": "s", "new_lessons": new}
    clock.t += 300
    r = a.tick()["reflection"]
    assert r["why"] == "scheduled reflection" and r["applied"]["new"] == ["L2"]
    assert (
        "Do not search the web"
        in run.reflects[0][0][run.reflects[0][0].index("--system-prompt") + 1]
    )
    clock.t += 300
    assert "reflection" not in a.tick()  # done: next one in 6 h
    run.answers.append(answer("NO_TRADE"))
    a.tick(force=True)
    tr = json.loads(run.calls[-1][1]["input"].split("Snapshot:\n", 1)[1])
    limits = tr["limits"]
    assert a.cfg is cfg0 and limits["notional_usdt"] == cfg0.notional == 3.0  # text only
    assert "Raise the notional" in json.dumps(tr["track_record"]["lessons"])
    m = oil.Memory(d / "memory.json")
    with m.edit(NOW) as mem:  # force a due reflection that fails: retry clock only
        mem["reflected_episodes"], mem["last_reflection_utc"] = 0, "2026-10-01T00:00:00"
    run.reflection = "not an object"
    clock.t += 3600  # any attempt (even a success) holds the next one off for 1 h
    r = a.tick()["reflection"]
    assert "invalid reflection" in r["error"] and m.load()["reflections"] == 1
    clock.t += 300
    assert "reflection" not in a.tick()  # failed: retried after 1 h, not every tick
    rows = oil.read_jsonl(a.log_dir, "reflections.jsonl")
    assert len(rows) == 2 and rows[0]["applied"]["new"] == ["L2"]
    assert oil.report("oil-t", d)["memory"]["reflections"] == 1


def test_reflector_follows_the_analyst_backend():
    assert oil.make_reflector(oil.Rule()) is None
    r = oil.make_reflector(oil.ClaudeCLI(timeout=300))
    assert r.schema is oil.REFLECT_SCHEMA and r.timeout == 180 and r.system == oil.REFLECT_SYSTEM
    api = oil.make_reflector(oil.AnthropicAPI())
    assert api.search is False and "submit_decision tool" in api.system
    fake = FakeAPI((200, msg("tool_use", submit(NO_REFLECTION))))
    api = oil.make_reflector(oil.AnthropicAPI(api_key="k", transport=httpx.MockTransport(fake)))
    assert api("x", {})["output"] == NO_REFLECTION
    assert [t["name"] for t in fake.bodies[0]["tools"]] == ["submit_decision"]
    assert fake.bodies[0]["tools"][0]["input_schema"] == oil.REFLECT_SCHEMA


def test_claude_cli_adapter_reply_and_timeout():
    an = oil.ClaudeCLI(model="m", run=FakeRun(buy()))
    r = an("x", {})
    assert r["output"]["decision"] == "BUY_CANDIDATE" and r["num_turns"] == 3
    assert an.name == "claude-cli:m" and "--model" in an.run.calls[0][0]
    with pytest.raises(oil.AgentTimeout):
        oil.ClaudeCLI(run=FakeRun(subprocess.TimeoutExpired("claude", 1)))("x", {})


class FakeAPI:
    """Anthropic Messages endpoint: a scripted list of (status, message) replies."""

    def __init__(self, *replies):
        self.replies, self.bodies = list(replies), []

    def __call__(self, req: httpx.Request) -> httpx.Response:
        assert req.url == oil.analyst.API_URL and req.headers["x-api-key"] == "k"
        assert req.headers["anthropic-version"] == "2023-06-01"
        self.bodies.append(json.loads(req.content))
        status, msg = self.replies.pop(0)
        return httpx.Response(status, json=msg)


def msg(stop, *content, searches=0):
    return {"model": "claude-x", "stop_reason": stop, "content": list(content),
            "usage": {"server_tool_use": {"web_search_requests": searches}}}  # fmt: skip


def submit(a):
    return {"type": "tool_use", "id": "t1", "name": oil.analyst.SUBMIT, "input": a}


def test_anthropic_api_pause_turn_then_submit():
    search = {"type": "server_tool_use", "id": "s", "name": "web_search", "input": {"query": "eia"}}
    api = FakeAPI((200, msg("pause_turn", search, searches=2)),
                  (200, msg("tool_use", {"type": "text", "text": "done"}, submit(buy()),
                            searches=1)))  # fmt: skip
    an = oil.AnthropicAPI(api_key="k", model="claude-x", transport=httpx.MockTransport(api))
    r = an("prompt", {})
    assert r["output"] == buy() and r["num_turns"] == 2 and r["searches"] == 3
    b0, b1 = api.bodies
    assert b0["tools"][0] == {"type": "web_search_20250305", "name": "web_search", "max_uses": 6}
    assert b0["tools"][1]["input_schema"] == oil.AGENT_SCHEMA and "tool_choice" not in b1
    assert b1["messages"][-1] == {"role": "assistant", "content": [search]}  # resent unchanged
    assert "submit_decision" in b0["system"] and an.system != oil.SYSTEM


def test_anthropic_api_forces_tool_after_end_turn_and_fails_closed():
    api = FakeAPI((200, msg("end_turn", {"type": "text", "text": "I think NO_TRADE"})),
                  (200, msg("tool_use", submit(answer("NO_TRADE")))))  # fmt: skip
    r = oil.AnthropicAPI(api_key="k", transport=httpx.MockTransport(api))("p", {})
    assert r["output"]["decision"] == "NO_TRADE"
    assert api.bodies[1]["tool_choice"] == {"type": "tool", "name": "submit_decision"}
    with pytest.raises(oil.AgentError, match="http 529"):
        api = FakeAPI((529, {"type": "error", "error": {"type": "overloaded_error"}}))
        oil.AnthropicAPI(api_key="k", transport=httpx.MockTransport(api))("p", {})
    with pytest.raises(oil.AgentError, match="no submit_decision"):
        api = FakeAPI(*[(200, msg("pause_turn"))] * 3)
        oil.AnthropicAPI(api_key="k", max_turns=3, transport=httpx.MockTransport(api))("p", {})
    with pytest.raises(oil.AgentTimeout):
        t = iter([0.0, 400.0])
        oil.AnthropicAPI(api_key="k", clock=lambda: next(t))("p", {})


def test_anthropic_api_needs_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(oil.AgentError, match="ANTHROPIC_API_KEY"):
        oil.AnthropicAPI()("p", {})


def test_rule_baseline_decisions_pass_validation_and_gate_inputs():
    as_of = pd.Timestamp("2026-10-22T12:00:00")
    rule = oil.Rule()
    d = oil.validate(rule("", snap_for(-0.025))["output"], as_of, False)
    assert d["decision"] == "BUY_CANDIDATE" and d["estimated_gross_opportunity_pct"] == 2.5
    assert d["supporting_evidence_ids"] and not d["lookahead_or_unsourced"]
    assert d["horizon_hours"] == 4.0 and d["expiry_utc"] == "2026-10-22T16:00:00"
    assert oil.validate(rule("", snap_for(-0.010))["output"], as_of, False)["decision"] == "WATCH"
    assert (
        oil.validate(rule("", snap_for(+0.010))["output"], as_of, False)["decision"] == "NO_TRADE"
    )
    held = oil.validate(rule("", snap_for(0.002, position={"entry": 1}))["output"], as_of, True)
    assert held["decision"] == "EXIT_REVIEW" and held["exit_now"] is True
    s = snap_for()
    s["divergence_usoon_minus_wti"]["ret_4h"] = None
    assert rule("", s)["output"]["decision"] == "NO_TRADE"  # missing data fails closed


def test_agent_with_rule_analyst_buys_on_gap(env):
    make, *_ = env
    a, feeds, *_ = make()
    a.analyst = oil.Rule()
    feeds.wti_lead = 0.03
    rec = a.tick()
    assert rec["decision"]["decision"] == "BUY_CANDIDATE"
    log = json.loads((a.log_dir / "agent.jsonl").read_text().splitlines()[0])
    assert log["model"] == "rule-v1" and log["raw"]["decision"] == "BUY_CANDIDATE"


def test_replay_reproduces_recorded_run(env):
    make, *_ = env
    cfg = oil.OilConfig(min_call_gap_s=0, sweep_s=0)
    a, _, clock, _ = make(buy(), oil.AgentError("x"), oil.AgentError("x"), oil.AgentError("x"),
                          answer("WATCH"), cfg=cfg)  # fmt: skip
    for _ in range(3):
        a.tick()
        clock.t += 300
    rows = list(oil.replay_run(a.log_dir, oil.Replay(a.log_dir)))
    assert len(rows) == 3 and all(r["same"] for r in rows)
    assert [r["replayed"]["decision"] for r in rows][0] == "BUY_CANDIDATE"
    rows = list(oil.replay_run(a.log_dir, oil.Rule()))  # baseline on the same snapshots
    assert len(rows) == 3 and all(r["replayed"]["decision"] in oil.DECISIONS for r in rows)
    with pytest.raises(ValueError):
        oil.make_analyst("gpt", timeout=1, max_turns=1)
    assert oil.make_analyst("replay:" + a.run_id, timeout=1, max_turns=1,
                            state_dir=a.state_dir).name == f"replay:{a.run_id}"  # fmt: skip


@pytest.mark.skipif(not (oil.STATE_DIR / "oil-s1" / "agent.jsonl").exists(), reason="no data/")
def test_replay_of_oil_s1_reproduces_its_decisions():
    rows = list(oil.replay_run(oil.STATE_DIR / "oil-s1", oil.Replay(oil.STATE_DIR / "oil-s1")))
    assert rows and all(r["same"] for r in rows)


def test_evaluate_scores_agent_vs_rule_on_same_snapshots(env):
    t0 = pd.Timestamp("2026-10-22T10:00:00")
    times = [t0 + pd.Timedelta(minutes=30 * i) for i in range(1, 19)]
    path = (times, [100.0 + i for i in range(1, 19)])  # steady rise: +1 bid per 30 min
    m = oil.evaluate_score(t0, 100.5, 100.0, "taker", path, horizon_h=2)
    assert m["net"]["1"] == oil.forward_net(100.5, 102.0, "taker") and set(m["net"]) == {
        "1", "2", "4", "8"}  # fmt: skip
    assert m["best_move"] == 0.04 and m["best_net"] == oil.forward_net(100.5, 116.0, "taker")
    gap = (times[:2], [101.0, 102.0])  # path ends after 1 h: later horizons stay unscored
    assert set(oil.evaluate_score(t0, 100.5, 100.0, "taker", gap)["net"]) == {"1"}
    rows = [
        {"ts": t0.isoformat(), "decision": d, "est_pct": e, "horizon_h": 2}
        | oil.evaluate_score(t0, 100.5, 100.0, "taker", path, 2)
        for d, e in (("BUY_CANDIDATE", 1.0), ("WATCH", None), ("NO_TRADE", None))
    ]
    s = oil.evaluate_stats(rows)
    assert s["hit_rate_4h"] == 0.33 and s["missed_moves"] == 2  # rise: only the buy was right
    assert s["watch_counterfactual"]["n"] == 1 and s["calibration"]["bias"] == -0.03

    make, *_ = env
    cfg = oil.OilConfig(min_call_gap_s=0, sweep_s=0)
    a, feeds, clock, _ = make(buy(), answer("WATCH"), cfg=cfg)
    for _ in range(2):
        a.tick()
        clock.t += 300
    e = oil.evaluate(a.run_id, a.state_dir)
    assert e["snapshots"] == 2 and e["baseline"] == "rule-v1"
    assert e["side_by_side"]["columns"] == ["agent", "rule-v1"]
    assert e["side_by_side"]["n_BUY_CANDIDATE"][0] == 1 and e["price_points"] >= 2
    assert oil.report(a.run_id, a.state_dir)["vs_baseline"]["snapshots"] == 2


def test_dashboard_renders_chart_cards_trades_and_escapes(env):
    from wallex.oil.dashboard import render

    make, *_ = env
    a, feeds, clock, _ = make(buy(gross=1.5), answer("WATCH", thesis="<script>x</script>"),
                              cfg=oil.AGGRESSIVE)  # fmt: skip
    feeds.wti_lead = 0.03
    a.tick()  # buy passes the gate and fills
    feeds.book = (165.0, 165.4)
    clock.t = pd.Timestamp("2026-10-22 16:00").timestamp()  # thesis expiry → exit
    a.tick(force=True)
    page = render(a.run_id, a.state_dir)
    assert page.startswith("<!doctype html>") and "<svg id=chart" in page
    assert page.count('class="mk ') == 2 and "id=d0" in page and "id=d1" in page
    assert "passed: qty" in page and "<td>SELL</td>" in page and "Agent vs Rule" in page
    assert "<script>x</script>" not in page and "&lt;script&gt;x" in page


def test_summary_uses_closed_bars_only():
    from wallex.oil.feeds import _summary

    now = 1_800_000_000
    t = [now - 900 * i for i in range(20, 0, -1)] + [now - 300]  # last bar still forming
    c = [100.0] * 20 + [103.0]  # +3 % spike inside the forming bar
    s = _summary(t, c, now)
    assert s["last"] == 103.0 and s["last_closed"] == 100.0
    assert s["ret_15m"] == 0.0 and s["ret_1h"] == 0.0 and s["ret_4h"] == 0.0
    assert s["last_bar_open_utc"] == pd.Timestamp(now - 900, unit="s").isoformat()
    assert _summary(t[-2:], c[-2:], now) == {"status": "unavailable"}  # one closed bar only


def test_calendar_releases_dst_holidays_and_contest_clock():
    ev = {(e["name"], e["ts"].isoformat()) for e in oil.events(pd.Timestamp("2026-10-20 12:00"))}
    assert ("API weekly crude inventories", "2026-10-20T20:30:00") in ev  # Tue 16:30 EDT
    assert ("EIA Weekly Petroleum Status Report", "2026-10-21T14:30:00") in ev  # Wed 10:30 EDT
    after_dst = oil.events(pd.Timestamp("2026-11-04 12:00"), 0, 6)
    assert [e["ts"].isoformat() for e in after_dst if e["name"].startswith("EIA")] == [
        "2026-11-04T15:30:00"
    ]  # EST
    holiday = [e for e in oil.events(pd.Timestamp("2026-11-11 00:00"), 0, 48) if "EIA" in e["name"]]
    assert holiday[0]["ts"].isoformat() == "2026-11-12T15:30:00" and "moved" in holiday[0]["note"]
    assert not oil.us_session_open(pd.Timestamp("2026-11-26 15:00"))  # Thanksgiving
    assert not oil.us_session_open(pd.Timestamp("2026-11-27 18:30"))  # early close 13:00 EST
    assert oil.us_session_open(pd.Timestamp("2026-11-27 17:30"))
    assert oil.us_session_open(pd.Timestamp("2026-10-22T14:00:00+00:00"))
    cal = oil.calendar(pd.Timestamp("2026-10-23 12:00"), oil.AGGRESSIVE)
    assert cal["hours_to_flat_by"] == 78.5 and cal["hours_to_contest_end"] == 79.5
    assert all(-24 <= e["in_hours"] <= 48 for e in cal["events"])


def test_evidence_cache_dedup_ttl_and_fed_back(env):
    make, *_ = env
    a, feeds, clock, run = make(answer("WATCH", evidence=buy()["evidence"]), answer("NO_TRADE"))
    a.tick(force=True)
    (e,) = a.state["evidence_cache"]
    assert e["url"] == "https://eia.gov/x" and e["claim"] == "EIA WPSR" and e["decision"] == "WATCH"
    clock.t += 3600
    a.tick(force=True)
    snap = json.loads(run.calls[-1][1]["input"].split("Snapshot:\n", 1)[1])
    assert [x["url"] for x in snap["evidence_cache"]["items"]] == ["https://eia.gov/x"]
    st, cfg, now = a.state, a.cfg, pd.Timestamp(clock.t, unit="s")
    oil.remember_evidence(st, buy(published="2026-10-22T12:30:00Z"), now, cfg)  # same URL
    assert len(st["evidence_cache"]) == 1 and st["evidence_cache"][0]["decision"] == "BUY_CANDIDATE"
    assert oil.evidence_cache(st, now + pd.Timedelta(hours=25), cfg) == []
    oil.remember_evidence(st, answer("NO_TRADE"), now + pd.Timedelta(hours=25), cfg)
    assert st["evidence_cache"] == []  # expired entries are dropped on the next write


def test_prompt_size_cap_trims_oldest_context_only():
    snap = {
        "usoon_book": {"bid": 1.0}, "limits": {"notional_usdt": 6},
        "evidence_cache": {"items": [{"url": f"u{i}", "claim": "x" * 300} for i in range(10)]},
        "track_record": {"recent_decisions": [{"i": i, "t": "y" * 300} for i in range(8)],
                         "closed_trades": [], "lessons": [{"text": "z" * 100}] * 3},
    }  # fmt: skip
    oil.cap_snapshot(snap, 10**6)
    assert "truncated" not in snap
    oil.cap_snapshot(snap, 2500)
    assert oil.prompt_chars(snap) <= 2500 + 200  # the marker itself is small
    assert snap["evidence_cache"]["items"] == [] and snap["usoon_book"] == {"bid": 1.0}
    rd = snap["track_record"]["recent_decisions"]
    assert rd[-1]["i"] == 7 and snap["truncated"]["dropped_oldest"]["evidence_cache.items"] == 10


def test_snapshot_has_calendar_and_cache_and_prompt_version_logged(env):
    make, _, d = env
    a, *_ = make(answer("NO_TRADE"))
    a.tick(force=True)
    rec = oil.read_jsonl(d / "oil-t", "agent.jsonl")[-1]
    snap = json.loads(rec["prompt"].split("Snapshot:\n", 1)[1])
    assert {"calendar", "evidence_cache"} <= snap.keys() and "truncated" not in snap
    assert rec["system_version"] == oil.SYSTEM_VERSION == "analyst-v2"
    assert oil.SYSTEM == oil.load_system("analyst-v2") and "evidence_cache" in oil.SYSTEM
    assert oil.prompt_version(oil.load_system("analyst-v1")) == "analyst-v1"
    assert oil.prompt_version(oil.AnthropicAPI().system) == "analyst-v2"
    assert oil.prompt_version("x") == "custom"
