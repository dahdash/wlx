"""Strategy-G runner: one decision cycle per 30-min bar close (T8).

``python -m wallex.run --mode shadow`` trades a paper portfolio against the live Wallex order
book; ``--mode live`` sends real orders (needs ``WALLEX_LIVE=1`` too, see :class:`LiveBroker`).

Each cycle (bar close T + a few seconds):
1. 30-min closes for the 5 coins from Wallex 1-min UDF candles (Wallex has no 30-min
   resolution), 14 days back; 28-day gate from 60-min candles (daily 00:00 UTC close).
2. Target weight per coin from :mod:`wallex.strategy.g`; slot = equity / 5.
3. Risk (:class:`RiskState`, windows scaled to 30-min bars): kill switch, pause, crash gate
   (off, `CRASH_GATE`),
   optional stop (off by default, decisions.md T8), per-slot cap. An optional AI officer
   (T9) may reweight within :func:`bound_ai`; without it, or if it fails, pure G.
   Trades only when |target − held| ≥ minNotional; full exits
   always go through.
4. Orders via :class:`Executor` (maker-first, cross at 20 min).
5. Decision log: ``cycles.jsonl`` (inputs, votes, reasons, risk, orders), ``trades.csv`` and
   ``alerts.log`` in ``data/state/<run_id>/``; state in ``data/state/<run_id>.json`` so a
   killed run resumes with ``--run-id``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections.abc import Callable
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from wallex import ai
from wallex.client import WallexClient, load_env
from wallex.exec import ExecResult, Executor, Intent, LiveBroker, PaperBroker, load_rules
from wallex.risk import RiskConfig, RiskState, Slot, flatten
from wallex.strategy import g

ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = ROOT / "data" / "state"
QUOTE = "USDT"
SYMBOLS = [f"{c}{QUOTE}" for c in g.COINS]
# other strategies' coins on the same account (oil agent): counted in the account equity that
# the kill/pause switches watch, never traded or sized by G
EXTRA = ("USOONUSDT",)
HISTORY_DAYS = 14  # ≥ 2 × the slowest SMA (336 bars = 7 d) so the band state is warmed up
BAR_S = 1800
SETTLE_S = 15  # wait after the bar close so the last 1-min candle is published
DAY_S = 86400
# risk windows in 30-min bars (same wall-clock spans as the 60-min defaults)
RISK_30M = dict(
    stop=None, cooldown_bars=8, crash_bars=48, vol_bars=144, pause_window_bars=12, pause_bars=4
)
# crash gate off for live (user decision 2026-10-22): it cut holdout CAGR 18.6 % → 11.8 %
# without lowering max drawdown (docs/strategy-g-results.md)
CRASH_GATE = False


# AI officer bounds (decisions.md, T8 + R6-Q04 b). Gate-on coins: any weight 0…1 slot.
# Gate-off coins: only with ≥ 5/6 SMA votes up, ≤ 0.5 slot each, ≤ 30 % of equity together.
AI_MIN_VOTES = 5 / 6
AI_GATE_OFF_MAX = 0.5
AI_GATE_OFF_TOTAL = 0.30
# (signals, ai.features context) → sym → (weight, reason)
Officer = Callable[[dict[str, g.Signal], dict[str, Any]], dict[str, tuple[float, str]]]


def bound_ai(
    sigs: dict[str, g.Signal], proposal: dict[str, tuple[float, str]]
) -> dict[str, tuple[float, str, str]]:
    """Clamp an AI proposal to the hard bounds → sym → (weight, source, reason).

    Coins the AI leaves out keep the pure-G weight. Gate-off entries are filled largest
    first until the 30 %-of-equity budget (1.5 slots of 5) is used up."""
    out = {s: (x.weight, "G", x.reason()) for s, x in sigs.items()}
    budget = AI_GATE_OFF_TOTAL * len(sigs)
    for s, (w, why) in sorted(proposal.items(), key=lambda kv: -kv[1][0]):
        if s not in sigs:
            continue
        x, w = sigs[s], min(max(float(w), 0.0), 1.0)
        if x.gate_on:
            out[s] = (w, "AI", f"{x.reason()}; AI w={w:.2f}: {why}")
        elif w > 0 and x.w_sma >= AI_MIN_VOTES - 1e-9:
            w = min(w, AI_GATE_OFF_MAX, budget)
            budget -= w
            out[s] = (w, "AI gate-off", f"{x.reason()}; AI gate-off entry w={w:.2f}: {why}")
        else:  # gate off and no (allowed) AI entry: G weight stays, keep the AI's view in the log
            out[s] = (x.weight, "AI (no entry)", f"{x.reason()}; AI w={w:.2f}, no entry: {why}")
    return out


# --- market data ------------------------------------------------------------------------


def fetch_1m(client: WallexClient, symbol: str, start: int, end: int) -> pd.Series:
    """1-min closes in [start, end) (Unix s), fetched one day per request."""
    parts = []
    for a in range(start, end, DAY_S):
        r = client.candles(symbol, "1", a, min(a + DAY_S, end) - 1)
        if r["t"]:
            parts.append(pd.Series([float(c) for c in r["c"]], index=r["t"]))
    if not parts:
        return pd.Series(dtype=float)
    s = pd.concat(parts)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s.index = pd.to_datetime(s.index, unit="s")
    return s[s.index < pd.Timestamp(end, unit="s")]


def to_30m(closes_1m: pd.Series, end: pd.Timestamp) -> pd.Series:
    """30-min closes labelled by open time, only bars closed by ``end``; gaps forward-filled."""
    if closes_1m.empty:
        return closes_1m
    bars = closes_1m.resample("30min", label="left", closed="left").last()
    return bars[bars.index + g.BAR <= end].ffill()


def fetch_hourly(client: WallexClient, symbol: str, end: int, days: int = 29) -> pd.Series:
    r = client.candles(symbol, "60", end - days * DAY_S, end)
    return pd.Series([float(c) for c in r["c"]], index=pd.to_datetime(r["t"], unit="s"))


def mid(depth: dict[str, list[dict[str, str]]]) -> float:
    return (float(depth["bid"][0]["price"]) + float(depth["ask"][0]["price"])) / 2


# --- runner ------------------------------------------------------------------------------


class Runner:
    def __init__(
        self,
        client: WallexClient,
        *,
        mode: str = "shadow",
        run_id: str | None = None,
        state_dir: Path = STATE_DIR,
        equity: float = 50.0,
        stop: float | None = None,
        executor: Executor | None = None,
        officer: Officer | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if mode not in ("shadow", "live"):
            raise ValueError(mode)
        self.client, self.mode, self.clock, self.sleep = client, mode, clock, sleep
        self.run_id = run_id or time.strftime("g-%Y%m%d-%H%M%S", time.gmtime(clock()))
        self.state_path = state_dir / f"{self.run_id}.json"
        self.log_dir = state_dir / self.run_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.rules = load_rules(client, SYMBOLS)
        if executor is None:
            broker = LiveBroker(client, live=True) if mode == "live" else PaperBroker(client.depth)
            executor = Executor(broker, self.rules, mode="maker", clock=clock, sleep=sleep)
        self.executor, self.officer = executor, officer
        self.bars: dict[str, pd.Series] = {}
        self.hourly: dict[str, pd.Series] = {}
        self.hourly_day: pd.Timestamp | None = None
        self.risk = RiskState(RiskConfig(**(RISK_30M | {"stop": stop})))
        self.state: dict[str, Any] = {
            "run_id": self.run_id,
            "mode": mode,
            "started": pd.Timestamp(clock(), unit="s").isoformat(),
            "last_bar": None,
            "cycles": 0,
            "cash": equity,
            "qty": {s: 0.0 for s in SYMBOLS},
            "gates": {},
        }
        if self.state_path.exists():
            self._load()

    # -- state --
    def _load(self) -> None:
        st = json.loads(self.state_path.read_text())
        rk = st.pop("risk")
        self.state.update(st)
        r = self.risk
        r.slots = {
            s: Slot(Decimal(v["qty"]), Decimal(v["entry_vwap"]), v["cooldown_until"], v["stop"])
            for s, v in rk["slots"].items()
        }
        r.peak_equity, r.paused_until, r.killed = rk["peak"], rk["paused_until"], rk["killed"]
        r.equity_hist = [tuple(x) for x in rk["equity_hist"]]
        self.alert(f"resumed run {self.run_id} at cycle {self.state['cycles']}")

    def _save(self) -> None:
        r = self.risk
        slots = {
            s: {
                "qty": str(v.qty),
                "entry_vwap": str(v.entry_vwap),
                "cooldown_until": v.cooldown_until,
                "stop": v.stop,
            }
            for s, v in r.slots.items()
        }
        risk = {
            "slots": slots,
            "peak": r.peak_equity,
            "paused_until": r.paused_until,
            "killed": r.killed,
            "equity_hist": r.equity_hist[-100:],
        }
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state | {"risk": risk}, indent=1))
        tmp.replace(self.state_path)

    # -- logs --
    def now(self) -> pd.Timestamp:
        return pd.Timestamp(self.clock(), unit="s")

    def alert(self, msg: str) -> None:
        with open(self.log_dir / "alerts.log", "a") as f:
            f.write(f"{self.now().isoformat()} {msg}\n")

    def _log_cycle(self, rec: dict[str, Any]) -> None:
        with open(self.log_dir / "cycles.jsonl", "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def _log_trade(self, bar: pd.Timestamp, r: ExecResult, reason: str) -> None:
        path = self.log_dir / "trades.csv"
        new = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(
                    ["time", "bar", "run_id", "mode", "symbol", "side", "requested", "filled",
                     "vwap", "fee", "crossed", "status", "orders", "reason"]
                )  # fmt: skip
            w.writerow(
                [self.now().isoformat(), bar.isoformat(), self.run_id, self.mode, r.symbol,
                 r.side, r.requested, r.filled, r.vwap, r.fee, r.crossed, r.status,
                 " ".join(r.orders), reason]
            )  # fmt: skip

    # -- data --
    def refresh_bars(self, end: pd.Timestamp) -> None:
        end_s = int(end.timestamp())
        for s in SYMBOLS:
            old = self.bars.get(s)
            start = end_s - HISTORY_DAYS * DAY_S
            if old is not None and len(old):
                start = max(start, int(old.index[-1].timestamp()))  # refetch the last bar
            new = to_30m(fetch_1m(self.client, s, start, end_s), end)
            merged = new if old is None else new.combine_first(old)
            self.bars[s] = merged[merged.index >= end - pd.Timedelta(days=HISTORY_DAYS)]
        day = end.floor("1D")
        if self.hourly_day != day:
            self.hourly = {s: fetch_hourly(self.client, s, end_s) for s in SYMBOLS}
            self.hourly_day = day

    def holdings(self, prices: dict[str, float]) -> tuple[dict[str, float], float, float]:
        """(G qty per symbol, free quote, account equity) in USDT.

        Live, the equity is the whole account: G coins, EXTRA coins (oil) and locked balances,
        so an oil fill never looks like a G loss. Wallex ``value`` is the total balance and
        already includes ``locked`` (live probe 2026-10-22), so free = value − locked. Sets
        ``self.extra_value`` (EXTRA coins, kept out of G's sizing) and ``self.locked`` (any
        locked balance: an order is in flight)."""
        self.extra_value, self.locked = 0.0, False
        if self.mode != "live":
            qty, cash = dict(self.state["qty"]), float(self.state["cash"])
            return qty, cash, cash + sum(qty[s] * prices[s] for s in SYMBOLS)
        bal = self.client.balances()["balances"]
        amt = lambda a, k="value": float(bal.get(a, {}).get(k, 0) or 0)  # noqa: E731
        coins = [s.removesuffix(QUOTE) for s in [*SYMBOLS, *EXTRA]]
        locked = {a: amt(a, "locked") for a in [QUOTE, *coins]}
        qty = {s: amt(s.removesuffix(QUOTE)) - locked[s.removesuffix(QUOTE)] for s in SYMBOLS}
        cash = amt(QUOTE) - locked[QUOTE]
        equity = amt(QUOTE) + sum(amt(s.removesuffix(QUOTE)) * prices[s] for s in SYMBOLS)
        for s in EXTRA:
            if (n := amt(s.removesuffix(QUOTE))) > 0:
                self.extra_value += n * mid(self.client.depth(s))
        self.locked = any(v > 0 for v in locked.values())
        return qty, cash, equity + self.extra_value

    # -- one cycle --
    def cycle(self, bar_close: pd.Timestamp) -> dict[str, Any]:
        t0 = self.clock()
        bar_no = int(bar_close.timestamp()) // BAR_S
        self.refresh_bars(bar_close)
        last = bar_close - g.BAR
        stale = [s for s in SYMBOLS if self.bars[s].empty or self.bars[s].index[-1] != last]
        prices = {s: mid(self.client.depth(s)) for s in SYMBOLS}
        qty, cash, equity = self.holdings(prices)
        rec: dict[str, Any] = {
            "bar": last.isoformat(),
            "bar_close": bar_close.isoformat(),
            "run_id": self.run_id,
            "mode": self.mode,
            "prices": prices,
            "equity_before": round(equity, 4),
            "cash": round(cash, 4),
            "qty": qty,
            "stale": stale,
        }

        # signal
        sigs: dict[str, g.Signal] = {}
        for s, c in zip(SYMBOLS, g.COINS, strict=True):
            gt = g.gate(self.hourly.get(s, pd.Series(dtype=float)), bar_close)
            if gt is None:
                prev = self.state["gates"].get(s)
                gt = (prev[0], prev[1]) if prev else (False, None)
                self.alert(f"{s}: 28-day gate unavailable, using {gt}")
            self.state["gates"][s] = list(gt)
            sigs[s] = g.signal(c, self.bars[s], *gt)
        rec["signals"] = {s: asdict(x) | {"reason": x.reason()} for s, x in sigs.items()}
        proposal: dict[str, tuple[float, str]] = {}
        if self.officer is not None:
            try:
                slot0 = (equity - self.extra_value) / len(SYMBOLS)
                held_w = {s: qty[s] * prices[s] / slot0 for s in SYMBOLS} if slot0 > 0 else {}
                ctx = ai.features(
                    sigs, self.bars, held_w, bar_close=bar_close, equity=equity,
                    peak_equity=max(self.risk.peak_equity or 0.0, equity),
                )  # fmt: skip
                proposal = self.officer(sigs, ctx)
            except Exception as e:  # fallback = pure G
                self.alert(f"officer failed, pure G: {type(e).__name__}: {e}")
        final = bound_ai(sigs, proposal)
        rec["ai_proposal"], rec["final_weights"] = proposal, final

        # risk
        # Wallex does not document whether "value" includes "locked": with an order in flight
        # the equity reading is ambiguous, so the kill/pause switches skip it
        verdict = None if self.locked else self.risk.update_equity(bar_no, equity)
        rec["equity_extra"], rec["locked"] = round(self.extra_value, 4), self.locked
        if self.locked:
            self.alert(f"balances locked: equity {equity:.2f} not used for kill/pause")
        if verdict:
            self.alert(f"risk {verdict}: equity {equity:.2f} peak {self.risk.peak_equity:.2f}")
        crash = {s: CRASH_GATE and self.risk.crash_gate(self.bars[s].to_numpy()) for s in SYMBOLS}
        dprices = {s: Decimal(str(p)) for s, p in prices.items()}
        stops = self.risk.check_stops(dprices, bar_no)
        slot = (equity - self.extra_value) / len(SYMBOLS)  # G sizes on its own share
        raw = {s: final[s][0] * slot for s in SYMBOLS}
        targets = self.risk.apply(
            raw, equity - self.extra_value, bar_no, crash=crash, paused_symbols=set(stale)
        )
        for s in stale:  # no fresh bar: hold, never trade on old data
            targets[s] = qty[s] * prices[s]
        rec["risk"] = {"verdict": verdict, "crash": crash, "stops": stops, "targets": targets}

        # orders
        intents, reasons = [], {}
        if self.risk.killed:
            hold = {s: Decimal(str(q)) for s, q in qty.items() if q > 0}
            results = flatten(self.executor, hold, tag=f"k{bar_no}")
            reasons = {s: "kill switch: flatten" for s in hold}
        else:
            for s in SYMBOLS:
                held = qty[s] * prices[s]
                delta, mn = targets[s] - held, float(self.rules[s].min_notional)
                why = self._why(final[s], targets[s], raw[s], held, crash[s], s in stops)
                if targets[s] <= 0 and held >= mn:  # full exit always executes
                    intents.append(Intent(s, "SELL", qty=Decimal(str(qty[s]))))
                elif (buy := min(delta, cash * 0.995)) >= mn:
                    intents.append(Intent(s, "BUY", notional=Decimal(str(buy))))
                    cash -= buy
                elif -delta >= mn:
                    intents.append(Intent(s, "SELL", qty=Decimal(str(-delta / prices[s]))))
                else:
                    continue
                reasons[s] = why
            results = (
                self.executor.execute(intents, tag=f"g{bar_no}", cycle_start=t0) if intents else []
            )
        rec["intents"] = [asdict(i) for i in intents]
        rec["reasons"] = reasons
        rec["results"] = [asdict(r) | {"status": r.status} for r in results]
        for r in results:
            self._apply_fill(r)
            self._log_trade(last, r, reasons.get(r.symbol, ""))
        qty, cash, equity = self.holdings(prices)
        rec["equity_after"] = round(equity, 4)
        rec["risk_events"] = self.risk.events[-5:]
        rec["elapsed_s"] = round(self.clock() - t0, 1)
        self.state["last_bar"] = bar_close.isoformat()
        self.state["cycles"] += 1
        self._log_cycle(rec)
        self._save()
        return rec

    @staticmethod
    def _why(final, target, raw, held, crash, stopped) -> str:
        w, src, reason = final
        why = f"[{src}] {reason}; w={w:.2f} target {target:.2f} held {held:.2f} USDT"
        if crash:
            why += "; crash gate → 0"
        if stopped:
            why += "; stop hit → cooldown"
        if target < raw - 1e-9 and not crash:
            why += "; capped by risk"
        return why

    def _apply_fill(self, r: ExecResult) -> None:
        if r.filled <= 0:
            return
        self.risk.slot(r.symbol).add_fill(r.side, r.filled, r.vwap)
        if self.mode == "live":
            return
        q, notional, fee = float(r.filled), float(r.filled * r.vwap), float(r.fee)
        sign = 1 if r.side == "BUY" else -1
        self.state["qty"][r.symbol] = max(0.0, self.state["qty"][r.symbol] + sign * q)
        self.state["cash"] = float(self.state["cash"]) - sign * notional - fee

    # -- loop --
    def run(self, max_cycles: int | None = None) -> None:
        n = 0
        while not self.risk.killed and (max_cycles is None or n < max_cycles):
            now = self.clock()
            bar_close = pd.Timestamp((int(now) // BAR_S) * BAR_S, unit="s")
            if self.state["last_bar"] == bar_close.isoformat() or now - bar_close.timestamp() > 600:
                nxt = bar_close.timestamp() + BAR_S + SETTLE_S
                self.sleep(max(0.0, nxt - now))
                continue
            try:
                rec = self.cycle(bar_close)
                print(
                    f"{rec['bar_close']} equity {rec['equity_after']:.2f} "
                    f"orders {len(rec['results'])} stale {rec['stale']}",
                    flush=True,
                )
            except Exception as e:  # keep the run alive; the next bar retries
                self.alert(f"cycle {bar_close.isoformat()} failed: {type(e).__name__}: {e}")
                self.state["last_bar"] = bar_close.isoformat()
                self._save()
            n += 1


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=["shadow", "live"], default="shadow")
    p.add_argument("--run-id", help="resume this run (data/state/<run_id>.json)")
    p.add_argument("--equity", type=float, default=50.0, help="paper start equity (USDT)")
    p.add_argument("--stop", type=float, default=None, help="per-slot stop, e.g. 0.05 (off)")
    p.add_argument("--cycles", type=int, default=None, help="stop after N cycles")
    p.add_argument("--ai", action="store_true", help="enable the Claude officer (T9)")
    a = p.parse_args(argv)
    load_env()
    if a.mode == "live" and os.environ.get("WALLEX_LIVE") != "1":
        p.error("live mode needs WALLEX_LIVE=1")
    with WallexClient() as client:
        r = Runner(client, mode=a.mode, run_id=a.run_id, equity=a.equity, stop=a.stop)
        if a.ai:
            r.officer = ai.ClaudeOfficer(r.log_dir)
        print(f"run {r.run_id} mode {a.mode} ai {a.ai} state {r.state_path}", flush=True)
        r.run(a.cycles)


if __name__ == "__main__":
    main()
