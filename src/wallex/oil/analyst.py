"""The analyst: system prompt, decision contract, validation, and the AI-agnostic
:class:`Analyst` adapters (``claude -p``, Anthropic Messages API, replay, rule baseline)."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx
import pandas as pd

from wallex.oil.config import DECISIONS, KINDS, STATE_DIR, OilConfig
from wallex.oil.ops import read_jsonl

PROMPTS = Path(__file__).with_name("prompts")
SYSTEM_VERSION = "analyst-v2"  # prompts/<version>.txt; a new text = a new file, never an edit


def load_system(version: str = SYSTEM_VERSION) -> str:
    return (PROMPTS / f"{version}.txt").read_text()


SYSTEM = load_system()

_STR = {"type": "string"}
_STRS = {"type": "array", "items": _STR}
AGENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": list(DECISIONS)},
        "hypothesis_id": {"type": ["string", "null"]},
        "horizon_hours": {"type": ["number", "null"]},
        "thesis": _STR,
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": _STR,
                    "kind": {"type": "string", "enum": KINDS},
                    "title": _STR,
                    "source_url": _STR,
                    "published_utc": _STR,
                },
                "required": ["id", "kind", "title", "source_url", "published_utc"],
            },
        },
        "supporting_evidence_ids": _STRS,
        "counter_evidence_ids": _STRS,
        "known_unknowns": _STRS,
        "catalyst_or_reason": _STR,
        "invalidation_conditions": _STRS,
        "expiry_utc": {"type": ["string", "null"]},
        "entry_conditions": _STR,
        "exit_conditions": _STR,
        "estimated_gross_opportunity_pct": {"type": ["number", "null"]},
        "hypothesis_updates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "hypothesis_id": _STR,
                    "status": {"type": "string", "enum": ["active", "confirmed", "abandoned"]},
                    "note": _STR,
                },
                "required": ["hypothesis_id", "status", "note"],
            },
        },
        "exit_now": {"type": "boolean"},
        "lesson": {"type": ["string", "null"]},
    },
    "required": [
        "decision", "thesis", "evidence", "supporting_evidence_ids", "counter_evidence_ids",
        "known_unknowns", "catalyst_or_reason", "invalidation_conditions",
        "estimated_gross_opportunity_pct", "exit_now",
    ],
}  # fmt: skip

Run = Callable[..., subprocess.CompletedProcess]


class AgentError(RuntimeError):
    pass


class AgentTimeout(AgentError):
    """The whole time budget is spent: never retried."""


def call_agent(text: str, *, model: str | None, timeout: float, max_turns: int, run: Run,
               system: str = SYSTEM, schema: dict[str, Any] = AGENT_SCHEMA) -> dict:  # fmt: skip
    """One isolated ``claude -p`` call with web tools only → the CLI's JSON envelope."""
    cmd = [
        "claude", "-p", "--output-format", "json", "--json-schema", json.dumps(schema),
        "--system-prompt", system, "--tools", "WebSearch,WebFetch",
        "--allowedTools", "WebSearch,WebFetch", "--strict-mcp-config",
        "--no-session-persistence", "--setting-sources", "", "--max-turns", str(max_turns),
    ]  # fmt: skip
    if model:
        cmd += ["--model", model]
    with tempfile.TemporaryDirectory() as cwd:  # no project files or CLAUDE.md in reach
        p = run(cmd, input=text, capture_output=True, text=True, timeout=timeout, cwd=cwd)
    if p.returncode != 0:
        raise AgentError(f"claude exit {p.returncode}: {(p.stderr or p.stdout)[-300:]}")
    try:
        env = json.loads(p.stdout)
    except json.JSONDecodeError as e:
        raise AgentError(f"claude output not JSON: {p.stdout[:200]!r}") from e
    if env.get("is_error") or env.get("subtype") != "success":
        raise AgentError(f"claude error: {str(env.get('result'))[:300]}")
    return env


def _ts(x: Any) -> pd.Timestamp | None:
    try:
        t = pd.Timestamp(x)
    except (ValueError, TypeError):
        return None
    if t is pd.NaT:
        return None
    return t.tz_convert("UTC").tz_localize(None) if t.tzinfo else t


def validate(out: Any, as_of: pd.Timestamp, holding: bool) -> dict[str, Any]:
    """Independent check of the structured answer. Raises :class:`AgentError` if unusable."""
    if not isinstance(out, dict) or out.get("decision") not in DECISIONS:
        raise AgentError(f"bad decision: {str(out)[:200]}")
    d = out["decision"]
    if d == "EXIT_REVIEW" and not holding:
        raise AgentError("EXIT_REVIEW without a position")
    if d == "BUY_CANDIDATE" and holding:
        raise AgentError("BUY_CANDIDATE while holding")
    as_of = as_of.tz_localize(None) if as_of.tzinfo else as_of
    ev = {}
    for e in out.get("evidence") or []:
        t = _ts(e.get("published_utc"))
        ok_url = str(e.get("source_url", "")).startswith(("http://", "https://"))
        e = dict(e) | {"valid": bool(t is not None and t <= as_of and ok_url)}
        ev[str(e.get("id"))] = e
    sup = [i for i in out.get("supporting_evidence_ids") or [] if i in ev]
    gross = out.get("estimated_gross_opportunity_pct")
    gross = (
        float(gross) if isinstance(gross, (int, float)) and not isinstance(gross, bool) else None
    )
    dec = {
        "decision": d,
        "hypothesis_id": out.get("hypothesis_id"),
        "horizon_hours": out.get("horizon_hours"),
        "thesis": str(out.get("thesis", ""))[:600],
        "evidence": list(ev.values()),
        "supporting_evidence_ids": sup,
        "counter_evidence_ids": [i for i in out.get("counter_evidence_ids") or [] if i in ev],
        "known_unknowns": [str(x)[:200] for x in out.get("known_unknowns") or []][:8],
        "catalyst_or_reason": str(out.get("catalyst_or_reason", ""))[:300],
        "invalidation_conditions": [str(x)[:200] for x in out.get("invalidation_conditions")
                                    or []][:6],
        "expiry_utc": out.get("expiry_utc"),
        "entry_conditions": str(out.get("entry_conditions", ""))[:300],
        "exit_conditions": str(out.get("exit_conditions", ""))[:300],
        "estimated_gross_opportunity_pct": gross,
        "hypothesis_updates": list(out.get("hypothesis_updates") or [])[:6],
        "exit_now": bool(out.get("exit_now", False)),
        "lesson": str(out["lesson"])[:240] if out.get("lesson") else None,
        "lookahead_or_unsourced": [i for i, e in ev.items() if not e["valid"]],
    }  # fmt: skip
    return dec


def no_trade(reason: str) -> dict[str, Any]:
    return {"decision": "NO_TRADE", "thesis": reason, "evidence": [], "supporting_evidence_ids": [],
            "exit_now": False, "estimated_gross_opportunity_pct": None}  # fmt: skip


# --- analyst adapters --------------------------------------------------------------------


class Analyst(Protocol):
    """``(prompt text, snapshot) → reply``; reply = {"output": raw answer dict, "num_turns",
    "duration_ms", "model_used"}. Raises :class:`AgentError` (→ NO_TRADE) on any failure."""

    name: str  # logged with every call
    system: str  # system prompt actually sent (its sha256 is logged)

    def __call__(self, text: str, snap: dict[str, Any]) -> dict[str, Any]: ...


@dataclass
class ClaudeCLI:
    """``claude -p`` with WebSearch + WebFetch (the analyst of oil-s1 … oil-live-1)."""

    model: str | None = None
    timeout: float = 300
    max_turns: int = 20
    run: Run = subprocess.run
    system: str = SYSTEM
    schema: dict[str, Any] = field(default_factory=lambda: AGENT_SCHEMA)

    @property
    def name(self) -> str:
        return f"claude-cli:{self.model or 'default'}"

    def __call__(self, text: str, snap: dict[str, Any]) -> dict[str, Any]:
        try:
            env = call_agent(text, model=self.model, timeout=self.timeout,
                             max_turns=self.max_turns, run=self.run, system=self.system,
                             schema=self.schema)  # fmt: skip
        except subprocess.TimeoutExpired as e:
            raise AgentTimeout(f"claude timeout after {self.timeout:.0f} s") from e
        except (subprocess.SubprocessError, OSError) as e:
            raise AgentError(f"claude: {type(e).__name__}: {e}") from e
        return {
            "output": env.get("structured_output", env.get("result")),
            "num_turns": env.get("num_turns"),
            "duration_ms": env.get("duration_ms"),
            "model_used": list(env.get("modelUsage", {})) or None,
        }


API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
SUBMIT = "submit_decision"


def api_system(system: str) -> str:
    """The claude -p prompt adapted to the Messages API tools."""
    return system.replace("WebSearch/WebFetch", "web_search").replace(
        "Answer only with the\nJSON object required by the schema.",
        f"Deliver the answer only by calling the {SUBMIT} tool once.",
    )


API_SYSTEM = api_system(SYSTEM)


def prompt_version(system: str) -> str:
    """``prompts/<version>.txt`` whose text (or its API variant) is ``system``, else "custom"."""
    for f in sorted(PROMPTS.glob("*.txt")):
        if system in (t := f.read_text(), api_system(t)):
            return f.stem
    return "custom"


@dataclass
class AnthropicAPI:
    """Messages API with the server-side web_search tool; the answer arrives as the input of
    a client tool whose input_schema is :data:`AGENT_SCHEMA`. Handles ``pause_turn`` (resend
    the paused turn) and forces the tool once if the model ends its turn without it.
    Key: ``ANTHROPIC_API_KEY``. Basic search version: works on every model, no code exec."""

    model: str = "claude-opus-5-5"
    timeout: float = 300
    max_turns: int = 20  # API round trips, not searches
    max_searches: int = 6
    max_tokens: int = 8000
    api_key: str | None = None
    transport: httpx.BaseTransport | None = None
    search_tool: str = "web_search_20250305"
    clock: Callable[[], float] = time.monotonic
    system: str = API_SYSTEM
    schema: dict[str, Any] = field(default_factory=lambda: AGENT_SCHEMA)
    search: bool = True  # False: no web_search tool (the reflection call)

    @property
    def name(self) -> str:
        return f"anthropic-api:{self.model}"

    def __call__(self, text: str, snap: dict[str, Any]) -> dict[str, Any]:
        key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise AgentError("ANTHROPIC_API_KEY not set")
        body: dict[str, Any] = {
            "model": self.model, "max_tokens": self.max_tokens, "system": self.system,
            "tools": [
                {"type": self.search_tool, "name": "web_search", "max_uses": self.max_searches},
            ][: int(self.search)] + [
                {"name": SUBMIT, "description": "Submit the final answer (once, at the end).",
                 "input_schema": self.schema},
            ],
        }  # fmt: skip
        headers = {"x-api-key": key, "anthropic-version": API_VERSION}
        msgs: list[dict[str, Any]] = [{"role": "user", "content": text}]
        t0, searches = self.clock(), 0
        with httpx.Client(transport=self.transport) as c:
            for turn in range(1, self.max_turns + 1):
                left = self.timeout - (self.clock() - t0)
                if left <= 0:
                    raise AgentTimeout(f"anthropic api: {self.timeout:.0f} s budget spent")
                try:
                    r = c.post(API_URL, headers=headers, json=body | {"messages": msgs},
                               timeout=left)  # fmt: skip
                except httpx.TimeoutException as e:
                    raise AgentTimeout(f"anthropic api timeout: {e}") from e
                except httpx.HTTPError as e:
                    raise AgentError(f"anthropic api: {type(e).__name__}: {e}") from e
                if r.status_code >= 400:
                    raise AgentError(f"anthropic api http {r.status_code}: {r.text[:300]}")
                m = r.json()
                use = (m.get("usage") or {}).get("server_tool_use") or {}
                searches += int(use.get("web_search_requests") or 0)
                content = m.get("content") or []
                for b in content:
                    if b.get("type") == "tool_use" and b.get("name") == SUBMIT:
                        return {
                            "output": b.get("input"), "num_turns": turn,
                            "duration_ms": round((self.clock() - t0) * 1000),
                            "model_used": [m.get("model", self.model)], "searches": searches,
                        }  # fmt: skip
                msgs.append({"role": "assistant", "content": content})
                stop = m.get("stop_reason")
                if stop == "pause_turn":  # server loop paused: send the turn back unchanged
                    continue
                if stop in ("end_turn", "max_tokens", "stop_sequence"):
                    msgs.append({"role": "user", "content": f"Call {SUBMIT} now."})
                    body["tool_choice"] = {"type": "tool", "name": SUBMIT}
                    continue
                raise AgentError(f"anthropic api: unexpected stop_reason {stop!r}")
        raise AgentError(f"anthropic api: no {SUBMIT} after {self.max_turns} turns")


def read_calls(run_dir: Path) -> list[dict[str, Any]]:
    """All agent calls of a run (rotated days first)."""
    return read_jsonl(run_dir, "agent.jsonl")


@dataclass
class Replay:
    """Answers with the raw reply recorded for the same ``snapshot_id`` in a run's agent log;
    a recorded failure or an unknown snapshot raises (→ NO_TRADE, as it did live)."""

    run_dir: Path
    system: str = SYSTEM
    recs: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        for r in read_calls(self.run_dir):  # last call per snapshot wins
            self.recs[r["snapshot_id"]] = r

    @property
    def name(self) -> str:
        return f"replay:{self.run_dir.name}"

    def __call__(self, text: str, snap: dict[str, Any]) -> dict[str, Any]:
        r = self.recs.get(snap.get("snapshot_id", ""))
        if r is None:
            raise AgentError(f"replay: no call for snapshot {snap.get('snapshot_id')}")
        if "error" in r or "raw" not in r:
            raise AgentError(f"replay: recorded failure {r.get('error')}")
        keys = ("num_turns", "duration_ms", "model_used")
        return {"output": r["raw"]} | {k: r.get(k) for k in keys}


RULE_SYSTEM = "rule-v1: buy when WTI 4-h return − USOON 4-h return ≥ limit and beats costs"


@dataclass
class Rule:
    """Deterministic, no-LLM baseline on the same snapshot (the only rule that was weakly
    positive after costs on history, docs/oil-estimate.md): BUY when WTI led USOON over 4 h
    by at least the profile's gap floor and the gap minus the round-trip cost clears the
    edge limit; WATCH at half the floor; while holding, exit once the gap has closed."""

    system: str = RULE_SYSTEM
    name: str = "rule-v1"

    def __call__(self, text: str, snap: dict[str, Any]) -> dict[str, Any]:
        as_of = str(snap.get("as_of_utc"))
        lim = snap.get("limits") or {}
        div = (snap.get("divergence_usoon_minus_wti") or {}).get("ret_4h")
        u = (snap.get("usoon_15m") or {}).get("ret_4h")
        w = (snap.get("wti_perp_15m") or {}).get("ret_4h")
        cost = snap.get("est_round_trip_cost")
        base = OilConfig()  # snapshots before the gap floor (oil-s1) lack the limit
        floor = lim.get("min_wti_minus_usoon_gap_4h", base.min_gap_4h)
        edge = lim.get("min_net_edge", base.min_net_edge)
        out: dict[str, Any] = {
            "decision": "NO_TRADE", "hypothesis_id": None, "horizon_hours": None,
            "thesis": "", "evidence": [], "supporting_evidence_ids": [],
            "counter_evidence_ids": [], "known_unknowns": ["news flow (rule reads prices only)"],
            "catalyst_or_reason": "WTI−USOON 4-h gap", "invalidation_conditions": [],
            "expiry_utc": None, "entry_conditions": "", "exit_conditions": "",
            "estimated_gross_opportunity_pct": None, "hypothesis_updates": [],
            "exit_now": False, "lesson": None,
        }  # fmt: skip
        if None in (div, u, w, cost, floor, edge):
            out["thesis"] = "rule: inputs missing (gap, cost or limits), fail closed"
            return {"output": out, "num_turns": 0, "duration_ms": 0, "model_used": None}
        gap = -div  # WTI 4-h return − USOON 4-h return
        out["evidence"] = [
            {"id": "wti4h", "kind": "fact", "title": f"CL perp 4-h return {w:+.2%}",
             "source_url": "https://fapi.binance.com/fapi/v1/klines?symbol=CLUSDT",
             "published_utc": as_of},
            {"id": "usoon4h", "kind": "fact", "title": f"USOON/USDT 4-h return {u:+.2%}",
             "source_url": "https://api.wallex.ir/v1/udf/history?symbol=USOONUSDT",
             "published_utc": as_of},
        ]  # fmt: skip
        ids = ["wti4h", "usoon4h"]
        if snap.get("position"):
            out |= {"decision": "EXIT_REVIEW", "supporting_evidence_ids": ids,
                    "exit_now": gap <= 0,
                    "thesis": f"rule: gap {gap:+.2%}" + (" closed, exit" if gap <= 0 else
                                                         ", still open, hold")}  # fmt: skip
        elif gap >= floor and gap - cost >= edge:
            h0, h1 = lim.get("horizon_h") or [2.0, 8.0]
            h = min(max(4.0, h0), h1)
            exp = pd.Timestamp(as_of) + pd.Timedelta(hours=h)
            out |= {
                "decision": "BUY_CANDIDATE", "hypothesis_id": f"rule-gap-{as_of[:13]}",
                "horizon_hours": h, "supporting_evidence_ids": ids,
                "thesis": f"rule: WTI led USOON by {gap:.2%} over 4 h (floor {floor:.1%}); "
                          f"expect USOON to catch up, net of cost {cost:.2%}",
                "invalidation_conditions": ["gap closes by WTI falling back",
                                            "stop or thesis expiry"],
                "expiry_utc": exp.isoformat(), "estimated_gross_opportunity_pct":
                round(gap * 100, 3),
            }  # fmt: skip
        elif gap >= floor / 2:
            out |= {
                "decision": "WATCH",
                "thesis": f"rule: gap {gap:+.2%} below the floor {floor:.1%} or the cost",
            }
        else:
            out["thesis"] = f"rule: gap {gap:+.2%} < {floor / 2:.2%}"
        return {"output": out, "num_turns": 0, "duration_ms": 0, "model_used": None}


def make_analyst(spec: str, *, timeout: float, max_turns: int, run: Run = subprocess.run,
                 state_dir: Path | None = None) -> Analyst:  # fmt: skip
    """``cli`` | ``api`` | ``rule`` | ``replay:<run_id>`` (model from ``WALLEX_AI_MODEL``)."""
    model = os.environ.get("WALLEX_AI_MODEL") or None
    if spec == "cli":
        return ClaudeCLI(model=model, timeout=timeout, max_turns=max_turns, run=run)
    if spec == "api":
        kw = {"model": model} if model else {}
        return AnthropicAPI(timeout=timeout, max_turns=max_turns, **kw)
    if spec == "rule":
        return Rule()
    if spec.startswith("replay:") and spec[7:]:
        return Replay((state_dir or STATE_DIR) / spec[7:])
    raise ValueError(f"unknown analyst {spec!r} (cli, api, rule, replay:<run_id>)")


def replay_run(run_dir: Path, analyst: Analyst) -> Iterator[dict[str, Any]]:
    """Re-decide every recorded call of a run with ``analyst`` on the same snapshot and as-of
    time → {ts, snapshot_id, logged, replayed, same, same_decision}. With :class:`Replay` of
    the same run, ``same`` must hold everywhere (the validation path is deterministic)."""
    for rec in read_calls(Path(run_dir)):
        snap = json.loads(rec["prompt"].split("Snapshot:\n", 1)[1])
        now = pd.Timestamp(rec["ts"])
        try:
            dec = validate(analyst(rec["prompt"], snap)["output"], now, bool(snap.get("position")))
        except AgentError as e:
            dec = no_trade(f"agent failed: AgentError: {e}"[:200])
        logged = rec["decision"]
        same = (
            json.dumps(dec, sort_keys=True, default=str)
            == json.dumps(logged, sort_keys=True, default=str)
            if "error" not in rec
            else dec["decision"] == logged["decision"]
        )
        yield {"ts": rec["ts"], "snapshot_id": rec["snapshot_id"], "logged": logged,
               "replayed": dec, "same": same,
               "same_decision": dec["decision"] == logged["decision"]}  # fmt: skip
