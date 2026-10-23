"""AI officer (T9): Claude reweights strategy G's candidates once per 30-min cycle.

The rule (:mod:`wallex.strategy.g`) builds the candidates and the hard caps; the officer only
proposes a weight in [0, 1] slot per coin plus a reason, and :func:`wallex.run.bound_ai`
clamps it (gate-off entries only with ≥ 5/6 votes, ≤ 0.5 slot each, ≤ 30 % of equity in total;
decisions.md R6-Q04, T8). Coins the officer leaves out keep the pure-G weight.

Call: ``claude -p`` (Max login, CLI default model unless ``WALLEX_AI_MODEL``; R5-Q05) with a
fixed system prompt, no tools, no project context, a JSON schema for the answer and a 45-s
timeout. Every call is logged to ``ai.jsonl`` (prompt, sha256, raw answer, parsed decision,
model, latency, error). A failed or invalid answer reuses the last valid decision for up to
2 cycles, then raises so the runner falls back to pure G (R4-Q05).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from wallex.strategy import g

TIMEOUT_S = 45
MAX_STALE = 2  # cycles a last valid decision may be reused
REASON_MAX = 200

SYSTEM = """\
You are the risk officer of an automated long-only spot crypto strategy (5 coins vs USDT,
one equal slot = 20 % of equity per coin, no leverage, no shorts). Every 30 minutes the rule
"G" gives each coin a weight in [0, 1] slot: votes of six SMA-cross pairs on 30-min closes
(12 h to 7 d), but only while the coin's 28-day return is positive (gate on); gate off → 0.

Your job: propose a weight in [0, 1] slot per coin, with a short reason. Hard limits are
enforced in code after you answer, so stay inside them or your number is clipped:
- gate on: any weight 0..1;
- gate off: an entry is allowed only with >= 5 of 6 SMA votes up, at most 0.5 slot per coin,
  and at most 1.5 slots for all gate-off coins together. Gate-off entries were never
  backtested: use them only when short-term momentum clearly confirms the trend.
Guidance:
- When unsure, follow G or prefer cash. The competition score rewards net return in toman
  AND low drawdown; a loss hurts more than a missed gain.
- Each trade costs about 0.1-0.3 % in fees and spread. Do not move a weight by less than
  ~0.2 slot from the currently held weight unless risk demands it (avoid churn).
- Reduce weight on sharp short-term drops, very high volatility, or strong divergence
  between the 30-min/6-h moves and the slow SMA votes.
- Omit a coin to keep G's weight. Never invent coins.
Answer only with the JSON object required by the schema."""


def schema(symbols: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "regime": {"type": "string", "enum": ["trend_up", "range", "trend_down", "crash"]},
            "weights": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "enum": symbols},
                        "weight": {"type": "number", "minimum": 0, "maximum": 1},
                        "reason": {"type": "string", "maxLength": REASON_MAX},
                    },
                    "required": ["symbol", "weight", "reason"],
                    "additionalProperties": False,
                },
            },
            "reason": {"type": "string", "maxLength": 300},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["regime", "weights", "reason", "confidence"],
        "additionalProperties": False,
    }


class AIError(RuntimeError):
    pass


def _r(x: float, nd: int = 4) -> float | None:
    return None if x is None or not math.isfinite(x) else round(float(x), nd)


def _ret(c: pd.Series, n: int) -> float | None:
    return _r(c.iloc[-1] / c.iloc[-1 - n] - 1) if len(c) > n else None


def features(
    sigs: dict[str, g.Signal],
    bars: dict[str, pd.Series],
    held_w: dict[str, float],
    *,
    bar_close: pd.Timestamp,
    equity: float,
    peak_equity: float,
) -> dict[str, Any]:
    """JSON-safe snapshot the officer decides on: G's view plus short-term market context."""
    coins = {}
    for s, x in sigs.items():
        c = bars.get(s, pd.Series(dtype=float))
        r30 = c.pct_change().iloc[-48:]
        coins[s] = {
            "sma_votes_up": int(sum(x.votes.values())),
            "gate_on": x.gate_on,
            "ret_28d": _r(x.gate_ret) if x.gate_ret is not None else None,
            "g_weight": _r(x.weight, 3),
            "held_weight": _r(held_w.get(s, 0.0), 3),
            "ret_30m": _ret(c, 1),
            "ret_6h": _ret(c, 12),
            "ret_24h": _ret(c, 48),
            "vol_24h": _r(float(r30.std()) * math.sqrt(48)) if len(r30) > 2 else None,
            "close_vs_sma24h": _r(c.iloc[-1] / c.iloc[-48:].mean() - 1) if len(c) >= 48 else None,
        }
    return {
        "bar_close_utc": bar_close.isoformat(),
        "equity_usdt": _r(equity, 2),
        "drawdown_from_peak": _r(equity / peak_equity - 1) if peak_equity > 0 else 0.0,
        "coins": coins,
    }


def prompt(feats: dict[str, Any]) -> str:
    return "Current state (weights in slots, returns as fractions):\n" + json.dumps(
        feats, indent=1, sort_keys=True
    )


Run = Callable[..., subprocess.CompletedProcess]


def call_claude(text: str, sch: dict, *, model: str | None, timeout: float, run: Run) -> dict:
    """One isolated ``claude -p`` call → the CLI's JSON result envelope."""
    cmd = [
        "claude", "-p", "--output-format", "json", "--json-schema", json.dumps(sch),
        "--system-prompt", SYSTEM, "--tools", "", "--strict-mcp-config",
        "--no-session-persistence", "--setting-sources", "",
    ]  # fmt: skip
    if model:
        cmd += ["--model", model]
    with tempfile.TemporaryDirectory() as cwd:  # no project CLAUDE.md in context
        p = run(cmd, input=text, capture_output=True, text=True, timeout=timeout, cwd=cwd)
    if p.returncode != 0:
        raise AIError(f"claude exit {p.returncode}: {(p.stderr or p.stdout)[-300:]}")
    try:
        env = json.loads(p.stdout)
    except json.JSONDecodeError as e:
        raise AIError(f"claude output not JSON: {p.stdout[:200]!r}") from e
    if env.get("is_error") or env.get("subtype") != "success":
        raise AIError(f"claude error: {str(env.get('result'))[:300]}")
    return env


def validate(out: Any, symbols: list[str]) -> dict[str, Any]:
    """Check the structured answer independently of the CLI → normalised decision."""
    if not isinstance(out, dict) or not isinstance(out.get("weights"), list):
        raise AIError(f"bad answer shape: {str(out)[:200]}")
    weights: dict[str, tuple[float, str]] = {}
    for it in out["weights"]:
        s, w = it.get("symbol"), it.get("weight")
        if s not in symbols or s in weights:
            raise AIError(f"bad or duplicate symbol {s!r}")
        if not isinstance(w, (int, float)) or isinstance(w, bool) or not 0 <= w <= 1:
            raise AIError(f"{s}: weight {w!r} outside [0, 1]")
        weights[s] = (float(w), str(it.get("reason", ""))[:REASON_MAX])
    conf = out.get("confidence")
    return {
        "regime": str(out.get("regime", "")),
        "reason": str(out.get("reason", ""))[:300],
        "confidence": float(conf) if isinstance(conf, (int, float)) else None,
        "weights": weights,
    }


class ClaudeOfficer:
    """``Runner(officer=ClaudeOfficer(log_dir))``: ``officer(sigs, ctx) → sym → (w, reason)``."""

    def __init__(
        self,
        log_dir: Path,
        *,
        model: str | None = None,
        timeout: float = TIMEOUT_S,
        run: Run = subprocess.run,
        clock: Callable[[], float] = time.time,
    ):
        self.log_path = Path(log_dir) / "ai.jsonl"
        self.model = model or os.environ.get("WALLEX_AI_MODEL") or None
        self.timeout, self.run, self.clock = timeout, run, clock
        self.last: dict[str, Any] | None = None
        self.stale = 0

    def __call__(self, sigs: dict[str, g.Signal], ctx: dict[str, Any]) -> dict:
        symbols = list(sigs)
        text = prompt(ctx)
        rec: dict[str, Any] = {
            "bar_close": ctx.get("bar_close_utc"),
            "model": self.model or "cli-default",
            "system_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
            "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "prompt": text,
        }
        t0 = self.clock()
        try:
            env = call_claude(
                text, schema(symbols), model=self.model, timeout=self.timeout, run=self.run
            )
            rec["raw"] = env.get("structured_output", env.get("result"))
            rec["model_used"] = list(env.get("modelUsage", {})) or None
            dec = validate(rec["raw"], symbols)
            self.last, self.stale = dec, 0
            rec["decision"], rec["source"] = dec, "fresh"
        except (AIError, subprocess.SubprocessError, OSError) as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            if self.last is None or self.stale >= MAX_STALE:
                rec["source"] = "fallback: pure G"
                self._log(rec, t0)
                raise AIError(rec["error"]) from e
            self.stale += 1
            dec = self.last
            rec["decision"], rec["source"] = dec, f"last valid ({self.stale}/{MAX_STALE})"
        self._log(rec, t0)
        tag = f"{dec['regime']}, conf {dec['confidence']}"
        if rec["source"] != "fresh":
            tag += f", {rec['source']}"
        return {s: (w, f"{why} [{tag}]") for s, (w, why) in dec["weights"].items()}

    def _log(self, rec: dict[str, Any], t0: float) -> None:
        rec["latency_s"] = round(self.clock() - t0, 2)
        rec["ts"] = pd.Timestamp(self.clock(), unit="s").isoformat()
        with self.log_path.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
