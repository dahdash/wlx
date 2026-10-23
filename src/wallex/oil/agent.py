"""The runner: wake-up, analysis, gate, paper execution, ledger, state."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from wallex.client import WallexClient, WallexError
from wallex.exec import Executor, Intent, LiveBroker, MarketRules, PaperBroker, load_rules
from wallex.oil import config, ops
from wallex.oil.analyst import (
    AGENT_SCHEMA,
    AgentError,
    AgentTimeout,
    Analyst,
    ClaudeCLI,
    Run,
    _ts,
    no_trade,
    prompt_version,
    validate,
)
from wallex.oil.config import SYMBOL, TICK_S, OilConfig
from wallex.oil.feeds import Feeds, book_stats
from wallex.oil.gate import exit_reason, gate
from wallex.oil.memory import Memory, make_reflector
from wallex.oil.snapshot import remember_evidence, snapshot


class OilAgent:
    def __init__(
        self,
        client: WallexClient,
        *,
        mode: str = "paper",
        run_id: str | None = None,
        g_run: str | None = "live-1",
        cfg: OilConfig | None = None,
        feeds: Feeds | None = None,
        executor: Executor | None = None,
        rules: MarketRules | None = None,
        run: Run = subprocess.run,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        state_dir: Path | None = None,
        armed: bool = False,
        analyst: Analyst | None = None,
        memory: Memory | None = None,
    ):
        if mode not in ("paper", "shadow", "live"):  # paper/shadow: live data, synthetic fills
            raise SystemExit(f"oil: unknown mode {mode!r} (paper, shadow, live)")
        self.client, self.mode, self.g_run = client, mode, g_run
        self.cfg = cfg or OilConfig()
        if mode == "live":
            if not armed:
                raise SystemExit("oil: --mode live needs --arm (and WALLEX_LIVE=1)")
            if self.cfg.notional > config.LIVE_MAX_NOTIONAL:
                raise SystemExit(f"oil: live notional {self.cfg.notional} > hard cap "
                                 f"{config.LIVE_MAX_NOTIONAL} USDT")  # fmt: skip
        self.feeds = feeds or Feeds(client)
        self.rules = rules or load_rules(client, [SYMBOL])[SYMBOL]
        self.run, self.clock, self.sleep = run, clock, sleep
        if executor is None:
            broker = (
                LiveBroker(client, live=True)  # raises without WALLEX_LIVE=1
                if mode == "live"
                else PaperBroker(lambda s: self.feeds.depth())
            )
            executor = Executor(
                broker, {SYMBOL: self.rules}, mode="maker", clock=clock, sleep=self.nap,
                cross_at_s=self.cfg.cross_after_s,
            )  # fmt: skip
        self.executor = executor
        self.analyst = analyst or ClaudeCLI(
            model=os.environ.get("WALLEX_AI_MODEL") or None, timeout=self.cfg.agent_timeout_s,
            max_turns=self.cfg.agent_max_turns, run=run,
        )  # fmt: skip
        self.reflector = make_reflector(self.analyst)
        self.run_id = run_id or time.strftime("oil-%Y%m%d-%H%M%S", time.gmtime(clock()))
        self.state_dir = state_dir = state_dir or config.STATE_DIR
        self.state_path = state_dir / f"{self.run_id}.json"
        self.log_dir = state_dir / self.run_id
        self.memory = memory or Memory(
            config.MEMORY_PATH if state_dir == config.STATE_DIR else state_dir / "memory.json"
        )
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.state: dict[str, Any] = {
            "run_id": self.run_id, "mode": mode, "cash": 0.0, "position": None,
            "pending": None, "entered": [], "hypotheses": {}, "day": {}, "loss_streak": 0,
            "calls": [], "last_call": None, "killed": False, "paused_reason": None,
            "fail_ticks": 0, "trades": 0, "evals": [], "closed": [], "lessons": [],
            "profile": self.cfg.profile, "schema": ops.SCHEMA, "log_day": None,
            "agent_fail_streak": 0, "evidence_cache": [],
        }  # fmt: skip
        self.ticks, self.last_tick, self.last_error, self.busy = 0, None, None, None
        if self.state_path.exists():
            self.state.update(ops.migrate(json.loads(self.state_path.read_text())))
            self.state["profile"] = self.cfg.profile
            if self.state.get("pending") and mode != "live":  # crash mid-order: fail closed
                self.state["paused_reason"] = f"unreconciled order {self.state['pending']}"
                self.alert(f"restart with pending order {self.state['pending']}: entries paused")
        if mode == "live":
            self.reconcile()

    # -- io --
    def now(self) -> pd.Timestamp:
        return pd.Timestamp(self.clock(), unit="s")

    def _save(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1, default=str))
        tmp.replace(self.state_path)

    def _log(self, name: str, rec: dict[str, Any]) -> None:
        with (self.log_dir / name).open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def alert(self, msg: str) -> None:
        with (self.log_dir / "alerts.log").open("a") as f:
            f.write(f"{self.now().isoformat()} {msg}\n")
        print(f"ALERT {msg}", flush=True)

    def _trade(self, side: str, qty: Decimal, px: Decimal, fee: Decimal, reason: str, hid) -> None:
        path = self.log_dir / "trades.csv"
        new = not path.exists()
        with path.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "run_id", "mode", "symbol", "side", "qty", "price", "fee",
                            "hypothesis_id", "reason"])  # fmt: skip
            w.writerow([self.now().isoformat(), self.run_id, self.mode, SYMBOL, side, qty, px,
                        fee, hid, reason])  # fmt: skip

    def _day(self) -> dict[str, Any]:
        today = self.now().date().isoformat()
        if self.state["day"].get("date") != today:
            self.state["day"] = {"date": today, "entries": 0, "pnl": 0.0}
        return self.state["day"]

    # -- live: the exchange is the source of truth --
    def held(self) -> Decimal:
        """USOON on the account, rounded down to the market step. Wallex ``value`` is the total
        and already includes ``locked`` (live probe 2026-10-22)."""
        b = self.client.balances()["balances"].get(SYMBOL.removesuffix("USDT"), {})
        return self.rules.q_qty(Decimal(str(b.get("value") or 0)))

    def reconcile(self) -> dict[str, Any]:
        """Live start: cancel resting oil orders (a crash mid-order), then take the position
        size from the balance (buy fees are paid in USOON; partial fills). USOON held with no
        position in state is adopted at the bid so the deterministic exits close it."""
        out: dict[str, Any] = {"cancelled": []}
        res = self.client.open_orders(SYMBOL)  # live: {"orders": [...]}; docs: keyed by symbol
        for o in res.get("orders", res.get(SYMBOL)) or []:
            cid = str(o.get("clientOrderId") or o.get("client_id") or "")
            if cid.startswith("oil") and o.get("symbol", SYMBOL) == SYMBOL:
                self.executor.broker.cancel(cid)
                out["cancelled"].append(cid)
        held, pos = self.held(), self.state["position"]
        bid = book_stats(self.feeds.depth(), self.cfg.depth_band)["bid"]
        dust = held * Decimal(str(bid)) < self.rules.min_notional
        if pos and dust:
            self.alert(f"reconcile: state position {pos['qty']} but USOON {held}: cleared")
            self.state["position"] = None
        elif pos:
            pos["qty"] = str(held)
        elif not dust:
            now = self.now()
            self.alert(f"reconcile: USOON {held} held with no position: adopted at bid {bid}")
            self.state["position"] = {
                "qty": str(held), "entry": float(bid), "fee": 0.0, "opened_utc": now.isoformat(),
                "expiry_utc": None, "hypothesis_id": "adopted", "thesis": "adopted on restart",
                "invalidation": [],
            }  # fmt: skip
        if self.state.get("pending"):
            self.alert(f"reconcile: pending order {self.state['pending']} resolved")
            self.state["pending"] = None
        out |= {"held": str(held), "position": self.state["position"]}
        self._log("reconcile.jsonl", {"ts": self.now().isoformat(), **out})
        self._save()
        return out

    def probe(self, notional: float) -> dict[str, Any]:
        """Micro round trip before the live run (T14.10 c). (1) A buy resting 5 % under the
        bid: read the USDT balance while it rests (does "value" include "locked"?), cancel.
        (2) Taker buy of ``notional`` USDT, (3) sell everything the account then holds."""
        if self.state["position"]:
            raise SystemExit("oil: probe refused, a position is open")
        if notional > config.LIVE_MAX_NOTIONAL:
            raise SystemExit(f"oil: probe notional above the hard cap {config.LIVE_MAX_NOTIONAL}")
        br, rules, tag = self.executor.broker, self.rules, f"oilp{int(self.clock()) % 10**6}"
        usdt = lambda: self.client.balances()["balances"].get("USDT", {})  # noqa: E731
        out: dict[str, Any] = {"before": usdt(), "held_before": str(self.held())}
        bid, _ = br.best(SYMBOL)
        px = rules.q_price(bid * Decimal("0.95"), "BUY")
        qty = rules.q_qty(Decimal(str(notional)) / px)
        br.place(SYMBOL, "BUY", qty, px, f"{tag}-r")
        out["resting"] = {"price": str(px), "qty": str(qty), "balance": usdt()}
        out["resting"]["after_cancel"] = asdict(br.cancel(f"{tag}-r"))
        (b,) = self.executor.execute(
            [Intent(SYMBOL, "BUY", notional=Decimal(str(notional)))], tag=f"{tag}-b", mode="through"
        )  # fmt: skip
        out["buy"] = asdict(b) | {"status": b.status}
        if b.filled > 0:
            self._trade("BUY", b.filled, b.vwap, b.fee, "probe", "probe")
        held = self.held()
        out["held_after_buy"] = str(held)
        if held > 0 and rules.valid(held, bid):
            (x,) = self.executor.execute(
                [Intent(SYMBOL, "SELL", qty=held)], tag=f"{tag}-s", mode="through"
            )  # fmt: skip
            out["sell"] = asdict(x) | {"status": x.status}
            if x.filled > 0:
                self._trade("SELL", x.filled, x.vwap, x.fee, "probe", "probe")
        out |= {"after": usdt(), "held_after": str(self.held())}
        self._log("probe.jsonl", {"ts": self.now().isoformat(), **out})
        return out

    # -- analysis scheduling --
    def due(self, snap: dict[str, Any]) -> str | None:
        """Why an analysis should run now, or None (deterministic wake-up)."""
        now, cfg = self.clock(), self.cfg
        calls = [t for t in self.state["calls"] if now - t < 86400]
        if len(calls) >= cfg.max_calls_day:
            return None
        last = self.state["last_call"]
        since = now - last if last else float("inf")
        if self.state["position"]:
            return "position review" if since >= cfg.review_s else None
        if snap["strategy_g"].get("all_off") is not True or self.state["killed"]:
            return None
        if since < cfg.min_call_gap_s:
            return None
        u, w = snap.get("usoon_15m", {}), snap.get("wti_perp_15m", {})
        div = snap["divergence_usoon_minus_wti"].get("ret_4h")
        if any(abs(x.get("ret_1h") or 0) >= cfg.move_trigger for x in (u, w)):
            return "1-h move trigger"
        if div is not None and abs(div) >= cfg.div_trigger:
            return "USOON-WTI divergence trigger"
        if since >= cfg.sweep_s:
            return "scheduled sweep"
        return None

    def analyse(self, snap: dict[str, Any], why: str) -> dict[str, Any]:
        """One agent call → validated decision (NO_TRADE on any failure)."""
        now = self.now()
        self.state["calls"] = [t for t in self.state["calls"] if self.clock() - t < 86400]
        self.state["calls"].append(self.clock())
        self.state["last_call"] = self.clock()
        text = f"Wake-up reason: {why}\nSnapshot:\n" + json.dumps(snap, indent=1, default=str)
        rec: dict[str, Any] = {
            "ts": now.isoformat(), "snapshot_id": snap["snapshot_id"], "why": why,
            "model": self.analyst.name,
            "system_version": prompt_version(self.analyst.system),
            "system_sha256": hashlib.sha256(self.analyst.system.encode()).hexdigest(),
            "schema_sha256": hashlib.sha256(json.dumps(AGENT_SCHEMA).encode()).hexdigest(),
            "prompt": text,
        }  # fmt: skip
        t0 = self.clock()
        try:
            reply = self._call(text, snap, t0, rec)
            rec["raw"] = reply["output"]
            rec |= {k: reply.get(k) for k in ("num_turns", "duration_ms", "model_used")}
            dec = validate(rec["raw"], now, bool(self.state["position"]))
            remember_evidence(self.state, dec, now, self.cfg)
        except AgentError as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            dec = no_trade(f"agent failed: {rec['error'][:200]}")
        fails = self.state["agent_fail_streak"] + 1 if "error" in rec else 0
        self.state["agent_fail_streak"] = fails
        if self.state["agent_fail_streak"] == self.cfg.max_agent_fails:
            self.alert(f"{self.cfg.max_agent_fails} analyses in a row failed: {rec['error'][:200]}")
        rec["latency_s"] = round(self.clock() - t0, 1)
        rec["decision"] = dec
        self._log("agent.jsonl", rec)
        self._ledger(dec, now)
        self._learn(dec, snap, now, "error" not in rec)
        return dec

    def _call(self, text: str, snap: dict[str, Any], t0: float, rec: dict[str, Any]):
        """The analyst with backoff retries on quick failures; the last error propagates
        (→ NO_TRADE). A timeout already spent the budget, so it is never retried."""
        cfg, errors = self.cfg, []
        for attempt in range(cfg.agent_retries + 1):
            try:
                return self.analyst(text, snap)
            except AgentError as e:
                errors.append(f"{type(e).__name__}: {str(e)[:150]}")
                rec["attempts"] = errors
                wait = cfg.agent_backoff_s * 2**attempt
                spent = self.clock() - t0
                timeout = isinstance(e, AgentTimeout)
                if timeout or attempt == cfg.agent_retries or spent + wait >= cfg.agent_timeout_s:
                    raise
                self.nap(wait, "agent retry")
        raise AssertionError("unreachable")

    def _learn(self, dec: dict[str, Any], snap: dict[str, Any], now: pd.Timestamp, ok: bool):
        """Keep the decision for forward scoring and the agent's lesson for the next call."""
        b = snap.get("usoon_book", {})
        self.state["evals"].append({
            "ts": now.tz_localize(None).isoformat() if now.tzinfo else now.isoformat(),
            "decision": dec["decision"], "hypothesis_id": dec.get("hypothesis_id"),
            "gross_est_pct": dec.get("estimated_gross_opportunity_pct"),
            "cost": snap.get("est_round_trip_cost"), "ask": b.get("ask"), "bid": b.get("bid"),
            "entry": self.cfg.entry_kind, "net": {}, "thesis": dec["thesis"][:160], "ok": ok,
        })  # fmt: skip
        self.state["evals"] = self.state["evals"][-60:]
        if dec.get("lesson"):
            self.state["lessons"].append([now.isoformat(), dec["lesson"]])
            self.state["lessons"] = self.state["lessons"][-12:]

    def _ledger(self, dec: dict[str, Any], now: pd.Timestamp) -> None:
        """Hypothesis ledger: updates from the agent, new thesis for BUY/WATCH, auto-expiry."""
        hyps = self.state["hypotheses"]
        for u in dec.get("hypothesis_updates", []):
            h = hyps.get(u.get("hypothesis_id"))
            if h and h["status"] == "active":
                h["status"] = u["status"]
                h["history"].append([now.isoformat(), u["status"], str(u.get("note"))[:200]])
        if dec["decision"] in ("BUY_CANDIDATE", "WATCH"):
            hid = dec.get("hypothesis_id")
            if hid not in hyps:
                hid = f"h{len(hyps) + 1}"
                hyps[hid] = {"id": hid, "created_utc": now.isoformat(), "status": "active",
                             "history": []}  # fmt: skip
            hyps[hid] |= {
                "thesis": dec["thesis"], "decision": dec["decision"],
                "invalidation": dec["invalidation_conditions"], "expiry_utc": dec["expiry_utc"],
            }  # fmt: skip
            hyps[hid]["history"].append([now.isoformat(), dec["decision"], dec["thesis"][:200]])
            dec["hypothesis_id"] = hid
        for h in hyps.values():
            exp = _ts(h.get("expiry_utc"))
            if h["status"] == "active" and exp is not None and exp <= now:
                h["status"] = "expired"

    # -- trading --
    def enter(self, dec: dict[str, Any], snap: dict[str, Any]) -> dict[str, Any]:
        t_book = self.clock()
        try:
            book = book_stats(self.feeds.depth(), self.cfg.depth_band)  # fresh, not the briefing
            t, _ = self.feeds.usoon_15m(int(self.clock()))
            candle = pd.Timestamp(t[-1], unit="s") if t else None
        except (WallexError, httpx.HTTPError, KeyError, ValueError) as e:
            return {"ok": False, "why": [f"fresh data failed: {e}"]}
        div = snap.get("divergence_usoon_minus_wti", {}).get("ret_4h")  # USOON − WTI
        ok, why, qty = gate(
            dec, book, self.clock() - t_book, candle, snap["strategy_g"],
            self.state | {"day": self._day()}, self.cfg, self.rules, self.now(),
            gap_4h=None if div is None else -div,
        )  # fmt: skip
        verdict: dict[str, Any] = {"ok": ok, "why": why, "book": book, "qty": str(qty)}
        if not ok:
            return verdict
        if self.mode == "live" and self.cfg.notional > config.LIVE_MAX_NOTIONAL:
            return verdict | {"ok": False, "why": ["live notional above the hard cap"]}
        hid = dec["hypothesis_id"]
        tag = f"oil-{self.run_id}-{hid}"
        self.state["pending"] = tag  # write-ahead: a crash now blocks re-entry on restart
        self.state["entered"].append(hid)
        self._save()
        (r,) = self.executor.execute(
            [Intent(SYMBOL, "BUY", notional=Decimal(str(self.cfg.notional)))],
            tag=tag, allow_cross=self.cfg.entry_cross,
        )  # fmt: skip
        self.state["pending"] = None
        verdict["fill"] = asdict(r) | {"status": r.status}
        if r.filled > 0:
            day = self._day()
            day["entries"] += 1
            qty = self.held() if self.mode == "live" else r.filled  # live: net of coin fee
            self.state["position"] = {
                "qty": str(qty), "entry": float(r.vwap), "fee": float(r.fee),
                "opened_utc": self.now().isoformat(), "expiry_utc": dec["expiry_utc"],
                "hypothesis_id": hid, "thesis": dec["thesis"][:300],
                "invalidation": dec["invalidation_conditions"],
            }  # fmt: skip
            self.state["cash"] -= float(r.filled * r.vwap + r.fee)
            self._trade("BUY", r.filled, r.vwap, r.fee, dec["catalyst_or_reason"], hid)
        else:
            verdict["why"] = [f"maker entry not filled ({r.note or 'no fill'}), no cross"]
        self._save()
        return verdict

    def exit(self, reason: str) -> dict[str, Any]:
        pos = self.state["position"]
        qty = Decimal(pos["qty"])
        if self.mode == "live":
            qty = min(qty, self.held())  # never sell more than the account holds
        (r,) = self.executor.execute(
            [Intent(SYMBOL, "SELL", qty=qty)], tag=f"oilx-{self.run_id}-{pos['hypothesis_id']}",
            mode="through",
        )  # fmt: skip
        if r.filled <= 0:
            self.alert(f"exit not filled ({r.note}); retry next tick")
            return asdict(r)
        pnl = float(r.filled * r.vwap - r.fee) - (pos["entry"] * float(r.filled) + pos["fee"])
        self.state["cash"] += float(r.filled * r.vwap - r.fee)
        day = self._day()
        day["pnl"] = round(day["pnl"] + pnl, 6)
        self.state["loss_streak"] = self.state["loss_streak"] + 1 if pnl < 0 else 0
        self.state["trades"] += 1
        self._trade("SELL", r.filled, r.vwap, r.fee, reason, pos["hypothesis_id"])
        left = qty - r.filled
        if left > 0 and not self.rules.valid(left, r.vwap):
            left = Decimal(0)  # unsellable dust: the position is closed
        self.state["position"] = (pos | {"qty": str(left), "fee": 0.0}) if left > 0 else None
        self.state["closed"].append({
            "hypothesis_id": pos["hypothesis_id"], "opened_utc": pos["opened_utc"],
            "closed_utc": self.now().isoformat(), "entry": pos["entry"], "exit": float(r.vwap),
            "pnl_usdt": round(pnl, 6), "pnl_pct": round(pnl / (pos["entry"] * float(r.filled)), 5),
            "reason": reason[:150], "thesis": pos["thesis"][:160],
        })  # fmt: skip
        self.state["closed"] = self.state["closed"][-30:]
        h = self.state["hypotheses"].get(pos["hypothesis_id"])
        if h and h["status"] == "active" and left <= 0:
            h["status"] = "closed"
            h["history"].append([self.now().isoformat(), "closed", reason[:200]])
        self._save()
        return asdict(r) | {"pnl_usdt": round(pnl, 6)}

    # -- one tick --
    def tick(self, force: bool = False) -> dict[str, Any]:
        now = self.now()
        rec: dict[str, Any] = {"ts": now.isoformat(), "run_id": self.run_id, "mode": self.mode}
        today = now.date().isoformat()
        if self.state["log_day"] != today:  # daily rotation of the jsonl logs
            if self.state["log_day"]:
                ops.rotate(self.log_dir, self.state["log_day"])
            self.state["log_day"] = today
        if (self.log_dir / "KILL").exists() and not self.state["killed"]:
            self.state["killed"] = True
            self.alert("KILL file found: flatten, no new entries")
        snap = snapshot(self.feeds, now, self.state, self.cfg, self.g_run, self.state_dir,
                        self.memory.view())  # fmt: skip
        self._log("snapshots.jsonl", snap)
        rec["snapshot_id"] = snap["snapshot_id"]
        data_ok = "bid" in snap["usoon_book"] and "last" in snap["usoon_15m"]
        self.state["fail_ticks"] = 0 if data_ok else self.state["fail_ticks"] + 1
        if self.state["fail_ticks"] >= self.cfg.max_fail_ticks and not self.state["paused_reason"]:
            self.state["paused_reason"] = f"{self.state['fail_ticks']} ticks without data"
            self.alert(self.state["paused_reason"])
        pos = self.state["position"]
        if pos and data_ok:
            why = "kill switch" if self.state["killed"] else exit_reason(
                pos, snap["usoon_book"]["bid"], now, self.cfg
            )  # fmt: skip
            if why:
                rec["exit"] = self.exit(why) | {"reason": why}
        idle = bool(self.state["paused_reason"]) and not self.state["position"]
        if not self.state["killed"] and data_ok:
            why = "forced" if force else (None if idle else self.due(snap))
            rec["analysis"] = why
            if why:
                dec = self.analyse(snap, why)
                rec["decision"] = {k: dec.get(k) for k in ("decision", "hypothesis_id", "thesis")}
                if dec["decision"] == "BUY_CANDIDATE":
                    rec["gate"] = self.enter(dec, snap)
                elif (
                    dec["decision"] == "EXIT_REVIEW" and dec["exit_now"] and self.state["position"]
                ):
                    rec["exit"] = self.exit(f"agent: {dec['thesis'][:150]}")
                self._log("decisions.jsonl", {
                    "ts": now.isoformat(), "snapshot_id": snap["snapshot_id"], "why": why,
                    "decision": dec, "gate": rec.get("gate"),
                })  # fmt: skip
        self._save()
        self.memory.sync(self.run_id, self.state, now)
        if not rec.get("analysis") and self.reflector:  # never stacked on an analysis
            why = self.memory.due_reflection(now)
            if why:
                rec["reflection"] = self.reflect(why)
        return rec

    def reflect(self, why: str) -> dict[str, Any]:
        """Lesson curation call: judge, merge, retire, add. Failure changes nothing but the
        retry clock; the memory only ever reaches the analyst as text."""
        now, t0 = self.now(), self.clock()
        inp = self.memory.reflect_input(now)
        text = f"Reflection reason: {why}\nMemory:\n" + json.dumps(inp, indent=1, default=str)
        rec: dict[str, Any] = {
            "ts": now.isoformat(), "why": why, "model": self.reflector.name,
            "system_sha256": hashlib.sha256(self.reflector.system.encode()).hexdigest(),
            "prompt": text,
        }  # fmt: skip
        out = None
        try:
            reply = self.reflector(text, inp)
            rec["raw"] = out = reply["output"]
            rec |= {k: reply.get(k) for k in ("num_turns", "duration_ms", "model_used")}
        except AgentError as e:
            rec["error"] = f"{type(e).__name__}: {e}"
        try:
            rec["applied"] = self.memory.finish_reflection(now, out, self.run_id)
        except ValueError as e:
            rec["error"] = f"invalid reflection: {e}"
            self.memory.finish_reflection(now, None, self.run_id)
        rec["latency_s"] = round(self.clock() - t0, 1)
        self._log("reflections.jsonl", rec)
        return {k: rec.get(k) for k in ("why", "applied", "error")}

    def nap(self, s: float, why: str = "order wait") -> None:
        """Sleep inside a tick (executor re-peg, agent backoff) with a heartbeat, so a long
        maker wait does not look like a dead loop."""
        self.heartbeat(True, busy=why)
        self.sleep(s)

    def heartbeat(self, ok: bool, busy: str | None = None) -> None:
        st = self.state
        ops.write_heartbeat(self.log_dir, {
            "ts": self.now().isoformat(), "epoch": self.clock(), "run_id": self.run_id,
            "mode": self.mode, "profile": self.cfg.profile, "ticks": self.ticks, "busy": busy,
            "last_error": None if ok else self.last_error, "position": bool(st["position"]),
            "paused": st["paused_reason"], "killed": st["killed"],
            "agent_fail_streak": st["agent_fail_streak"],
        })  # fmt: skip

    def loop(self, max_ticks: int | None = None, force_first: bool = False) -> None:
        n = 0
        self.heartbeat(True, busy="start")
        while max_ticks is None or n < max_ticks:
            if n:
                self.sleep(TICK_S - self.clock() % TICK_S + 5)
            t = self.clock()  # gap since the previous tick ENDED: suspend, hang, clock jump
            if self.last_tick is not None and t - self.last_tick > self.cfg.stall_s:
                self.alert(f"loop stalled: {t - self.last_tick:.0f} s since the last tick")
            try:
                rec = self.tick(force=force_first and n == 0)
                self.ticks += 1
                self.heartbeat(True)
                d = rec.get("decision", {}).get("decision", "-")
                g = rec.get("gate") or {}
                print(f"{rec['ts']} analysis {rec.get('analysis')} decision {d} "
                      f"gate {g.get('ok')} {g.get('why', '')}", flush=True)  # fmt: skip
            except Exception as e:  # keep running; deterministic exits retry next tick
                self.last_error = f"{type(e).__name__}: {e}"
                self.alert(f"tick failed: {self.last_error}")
                self.heartbeat(False)
            self.last_tick = self.clock()
            n += 1
