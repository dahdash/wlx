"""Forward shadow evaluation report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from wallex.oil import config
from wallex.oil.config import DECISIONS
from wallex.oil.evaluate import evaluate
from wallex.oil.memory import Memory, track_record
from wallex.oil.ops import read_jsonl


def report(run_id: str, state_dir: Path | None = None) -> dict[str, Any]:
    """Forward shadow evaluation of a run: decisions scored, trades, agent spend."""
    state_dir = state_dir or config.STATE_DIR
    st = json.loads((state_dir / f"{run_id}.json").read_text())
    calls = read_jsonl(state_dir / run_id, "agent.jsonl")
    evals = st.get("evals", [])
    closed = st.get("closed", [])
    missed = [e for e in evals if e["decision"] in ("NO_TRADE", "WATCH")
              and e.get("best_net", 0) > 0.005]  # fmt: skip
    return {
        "run_id": run_id, "profile": st.get("profile"), "mode": st.get("mode"),
        "agent_calls": len(calls), "agent_errors": sum("error" in c for c in calls),
        "decisions": {d: sum(e["decision"] == d for e in evals) for d in DECISIONS},
        "scored_4h": track_record(st)["by_decision_4h"],
        "missed_moves_gt_0.5pct": len(missed),
        "trades_closed": len(closed),
        "pnl_usdt": round(sum(t["pnl_usdt"] for t in closed), 4),
        "wins": sum(t["pnl_usdt"] > 0 for t in closed),
        "open_position": st.get("position"),
        "paused": st.get("paused_reason"), "killed": st.get("killed"),
        "lessons": [x[1] for x in st.get("lessons", [])[-5:]],
        "memory": _memory(state_dir),
        "vs_baseline": evaluate(run_id, state_dir),
    }  # fmt: skip


def _memory(state_dir: Path) -> dict[str, Any]:
    """Shared memory status: lesson counts by status and reflections so far."""
    path = config.MEMORY_PATH if state_dir == config.STATE_DIR else state_dir / "memory.json"
    if not path.exists():
        return {}
    mem = Memory(path).load()
    status = [x["status"] for x in mem["lessons"]]
    return {"lessons": {s: status.count(s) for s in sorted(set(status))},
            "episodes": len(mem["episodes"]), "reflections": mem["reflections"],
            "last_reflection_utc": mem["last_reflection_utc"]}  # fmt: skip
