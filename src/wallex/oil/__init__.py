"""Oil intelligence agent (T13): event-driven long-only USOON/USDT trades while G is sidelined.

One Claude agent, no megatool:

1. **Wake-up** (deterministic, every 5 min): exits first (stop, expiry, max hold, kill file),
   then decide whether an analysis is due (G's five gates all off, call budget, spacing, and a
   trigger: scheduled sweep, oil/USOON move, USOON-vs-WTI divergence, open position review).
2. **Snapshot** (:func:`snapshot`): timestamped as-of packet: Wallex USOONUSDT book + 15-min
   candles, Binance CLUSDT perp (WTI reference, keyless), divergence, US equity session flag,
   G state from the live run's ``cycles.jsonl``, position, active hypotheses, missing fields.
   USO itself is ``unavailable`` (Yahoo rate-limits); USOON tracks USO, not spot WTI.
3. **Agent** (:class:`Analyst`, ``--analyst``): ``claude -p`` with only WebSearch + WebFetch
   (:class:`ClaudeCLI`, default), the Anthropic Messages API with server web_search
   (:class:`AnthropicAPI`), a recorded run (:class:`Replay`) or the no-LLM :class:`Rule`
   baseline; one fixed system prompt and a JSON schema (decision contract, brief §6). Claude
   chooses sources, reconciles reports, looks for counter-evidence and manages a time-limited
   thesis. Invalid or look-ahead output → ``NO_TRADE`` (fail closed).
4. **Gate** (:func:`gate`): deterministic cost / liquidity / freshness / limit checks on a
   freshly fetched book. The agent's gross estimate must beat our own round-trip cost; its
   confidence is never used for sizing (fixed notional).
5. **Execution**: :class:`wallex.exec.Executor` on :class:`wallex.exec.PaperBroker` (maker entry,
   no cross → non-fills are recorded; taker exit). Only ``--mode paper`` exists: micro-live is
   refused until the live G runner counts USOON in its equity (``run.py`` ``holdings``),
   otherwise an oil buy looks like a G loss and trips its pause / kill switch.

Logs in ``data/state/<run_id>/``: ``snapshots.jsonl``, ``agent.jsonl`` (prompt, raw answer,
turns, latency), ``decisions.jsonl`` (decision + gate verdict + snapshot id),
``trades.csv``, ``alerts.log``; state in ``data/state/<run_id>.json`` (restart-safe).
Kill switch: create ``data/state/<run_id>/KILL`` → flatten, no new entries.
Reliability (:mod:`wallex.oil.ops`): one process per run_id (flock on ``<run_id>.lock``),
``heartbeat.json`` each tick and during in-tick waits (``--health --run-id X`` exits 1 unless
ok), quick agent failures retried with backoff then NO_TRADE, versioned state (``schema``,
migrated on load), jsonl logs rotated daily to ``<name>.<day>.jsonl``, stale-loop alert.

Run: ``python -m wallex.oil --mode paper --run-id oil-p1`` (``--once`` for one traced
analysis, ``--force-analysis`` to skip the trigger check).
"""

from wallex.client import WallexError  # noqa: F401  (re-export for callers and tests)
from wallex.oil.agent import OilAgent
from wallex.oil.analyst import (
    AGENT_SCHEMA,
    SYSTEM,
    SYSTEM_VERSION,
    AgentError,
    AgentTimeout,
    Analyst,
    AnthropicAPI,
    ClaudeCLI,
    Replay,
    Rule,
    call_agent,
    load_system,
    make_analyst,
    no_trade,
    prompt_version,
    replay_run,
    validate,
)
from wallex.oil.calendar import calendar, events, us_session_open
from wallex.oil.cli import main
from wallex.oil.config import (
    AGGRESSIVE,
    BINANCE_FAPI,
    CONTEST_END_UTC,
    DECISIONS,
    G_COINS,
    HORIZONS,
    KINDS,
    MEMORY_PATH,
    PROFILES,
    REF,
    ROOT,
    STATE_DIR,
    SYMBOL,
    TICK_S,
    OilConfig,
    round_trip_cost,
)  # fmt: skip
from wallex.oil.evaluate import evaluate, price_path
from wallex.oil.evaluate import score as evaluate_score
from wallex.oil.evaluate import stats as evaluate_stats
from wallex.oil.feeds import Feeds, book_stats, read_g
from wallex.oil.gate import exit_reason, gate
from wallex.oil.memory import (
    REFLECT_SCHEMA,
    REFLECT_SYSTEM,
    Memory,
    add_lesson,
    apply_reflection,
    forward_net,
    make_reflector,
    similarity,
    track_record,
    update_evals,
)  # fmt: skip
from wallex.oil.ops import SCHEMA, health, migrate, read_jsonl, rotate, run_lock
from wallex.oil.report import report
from wallex.oil.snapshot import (
    cap_snapshot,
    evidence_cache,
    prompt_chars,
    remember_evidence,
    snapshot,
)

__all__ = [
    "AGENT_SCHEMA", "AGGRESSIVE", "BINANCE_FAPI", "DECISIONS", "G_COINS", "HORIZONS", "KINDS",
    "MEMORY_PATH", "PROFILES", "REF", "ROOT", "SCHEMA", "STATE_DIR", "SYMBOL", "SYSTEM", "TICK_S",
    "REFLECT_SCHEMA", "REFLECT_SYSTEM", "Memory", "add_lesson", "apply_reflection",
    "make_reflector", "similarity", "evaluate", "evaluate_score", "evaluate_stats", "price_path",
    "AgentError", "AgentTimeout", "Analyst", "AnthropicAPI", "ClaudeCLI", "Replay", "Rule",
    "Feeds", "OilAgent", "OilConfig", "WallexError", "book_stats", "call_agent", "make_analyst",
    "replay_run", "exit_reason", "forward_net", "gate", "health", "main", "migrate", "no_trade",
    "read_g", "read_jsonl", "report", "rotate", "round_trip_cost", "run_lock", "snapshot",
    "track_record", "update_evals", "us_session_open", "validate", "calendar", "events",
    "cap_snapshot", "evidence_cache", "remember_evidence", "prompt_chars", "prompt_version",
    "load_system", "SYSTEM_VERSION", "CONTEST_END_UTC",
]  # fmt: skip
