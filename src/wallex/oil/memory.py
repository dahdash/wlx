"""Learning: forward evaluation of every decision, the track record fed back, and the shared
memory tiers (episodic outcomes, semantic lessons, scheduled reflection)."""

from __future__ import annotations

import fcntl
import hashlib
import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from wallex.exec import MAKER_FEE, TAKER_FEE
from wallex.oil.config import DECISIONS, HORIZONS


def forward_net(ask0: float, bid: float, entry: str) -> float:
    """Net return of a hypothetical long bought at ``ask0`` and sold at ``bid``, both fees."""
    fee = MAKER_FEE if entry == "maker" else TAKER_FEE
    return round(bid / ask0 * (1 - fee) * (1 - TAKER_FEE) - 1, 5)


def update_evals(evals: list[dict[str, Any]], bid: float, now: pd.Timestamp) -> None:
    """Mark every past decision to the current bid: net at 1/2/4/8 h and best net within 8 h."""
    now = now.tz_localize(None) if now.tzinfo else now
    for e in evals:
        age_h = (now - pd.Timestamp(e["ts"])).total_seconds() / 3600
        if age_h > max(HORIZONS) + 0.25 or e.get("ask") is None:
            continue
        n = forward_net(e["ask"], bid, e["entry"])
        e["best_net"] = max(e.get("best_net", n), n)
        e["worst_net"] = min(e.get("worst_net", n), n)
        for h in HORIZONS:
            if str(h) not in e["net"] and age_h >= h:
                e["net"][str(h)] = n


def track_record(state: dict[str, Any], memory: dict[str, Any] | None = None) -> dict[str, Any]:
    """What the agent sees of its own past: scored decisions, closed trades, its lessons.
    With ``memory`` (:meth:`Memory.view`) the lessons are the shared ranked ones and the
    decision stats of all runs are added."""
    evals = state.get("evals", [])
    by: dict[str, Any] = {}
    for d in DECISIONS:
        xs = [e["net"]["4"] for e in evals if e["decision"] == d and "4" in e["net"]]
        if xs:
            by[d] = {"n": len(xs), "mean_net_4h": round(sum(xs) / len(xs), 5),
                     "share_positive": round(sum(x > 0 for x in xs) / len(xs), 2)}  # fmt: skip
    recent = [
        {k: e.get(k) for k in ("ts", "decision", "gross_est_pct", "cost", "net", "best_net",
                               "worst_net", "thesis")}
        for e in evals[-8:]
    ]  # fmt: skip
    out = {
        "note": "net = long bought at the ask then, sold at the bid later, fees included; "
        "small sample, calibrate but do not overfit",
        "by_decision_4h": by,
        "recent_decisions": recent,
        "closed_trades": state.get("closed", [])[-6:],
        "lessons": state.get("lessons", [])[-8:],
    }
    return out | memory if memory else out


# --- shared memory tiers (data/memory/oil.json) -------------------------------------------
#
# Episodic: every scored decision and closed trade of every run. Semantic: lessons with
# support/contradict counts and a status, curated by a scheduled reflection call. The memory
# only ever reaches the analyst as text in snapshot.track_record; limits, sizing, stops and
# the gate are code (OilConfig) and never read it.

MEM_SCHEMA = 1
PROMPT_LESSONS = 8  # active lessons shown to the analyst
REFLECT_EVERY_H = 6.0
REFLECT_RETRY_H = 1.0  # after a failed reflection
MIN_NEW_EPISODES = 3  # scheduled reflection only when there is something new to judge
SIMILAR = 0.6  # token Jaccard above which a new lesson counts as support for an old one
MAX_NEW = 2

REFLECT_SYSTEM = """\
You curate the lesson memory of an automated long-only USOON/USDT oil analyst. You get the
current lessons (id, text, conditions, support, contradict) and the scored episodes of all
runs: each past decision with its net return of a hypothetical long (bought at the ask,
sold at the bid 1/2/4/8 h later, fees included) and the closed trades. Do not search the web.
For every lesson give a verdict: support (the outcomes since agree with it), contradict (they
disagree), retire (obsolete, specific to one day or one price level, or wrong), keep (no new
evidence). Merge lessons that say the same thing into one general rule and state when it
applies in `conditions`. Add at most 2 new lessons, only for patterns the episodes show and
no lesson covers. Lessons are about reading the market and the evidence; never about limits,
position size, stops, fees or the gate, which are fixed by code. Small sample: prefer keep
over a verdict you cannot back with an episode. Answer only with the JSON object required by
the schema."""

REFLECT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "lesson_updates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "verdict": {"type": "string",
                                "enum": ["support", "contradict", "retire", "keep"]},
                    "note": {"type": "string"},
                },
                "required": ["id", "verdict", "note"],
            },
        },
        "merges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ids": {"type": "array", "items": {"type": "string"}},
                    "text": {"type": "string"},
                    "conditions": {"type": ["string", "null"]},
                },
                "required": ["ids", "text", "conditions"],
            },
        },
        "new_lessons": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"text": {"type": "string"},
                               "conditions": {"type": ["string", "null"]}},
                "required": ["text", "conditions"],
            },
        },
        "summary": {"type": "string"},
    },
    "required": ["lesson_updates", "merges", "new_lessons", "summary"],
}  # fmt: skip


def _tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2}


def similarity(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


def _key(*parts: Any) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16]


def _iso(now: pd.Timestamp) -> str:
    return (now.tz_localize(None) if now.tzinfo else now).isoformat()


def _rank(lesson: dict[str, Any]) -> tuple:
    return (lesson["support"] - lesson["contradict"], lesson["updated_utc"])


def empty_memory(now: pd.Timestamp) -> dict[str, Any]:
    return {"schema": MEM_SCHEMA, "created_utc": _iso(now), "episodes": [], "trades": [],
            "lessons": [], "seen": [], "next_id": 1, "last_reflection_utc": None,
            "last_reflection_try_utc": None, "reflected_episodes": 0, "reflected_trades": 0,
            "reflections": 0}  # fmt: skip


def add_lesson(mem: dict[str, Any], text: str, conditions: str | None, now: pd.Timestamp,
               source: str, seen_key: str | None = None) -> dict[str, Any] | None:  # fmt: skip
    """New lesson, or +1 support on an active lesson that says nearly the same thing. A
    ``seen_key`` (the original decision time + text) makes re-imports idempotent."""
    text = text.strip()[:240]
    if not text or (seen_key and seen_key in mem["seen"]):
        return None
    if seen_key:
        mem["seen"] = (mem["seen"] + [seen_key])[-2000:]
    ts = _iso(now)
    act = [x for x in mem["lessons"] if x["status"] == "active"]
    best = max(act, key=lambda x: similarity(x["text"], text), default=None)
    if best and similarity(best["text"], text) >= SIMILAR:
        best["support"] += 1
        best["updated_utc"] = ts
        best["sources"] = (best["sources"] + [source])[-20:]
        best["history"].append([ts, "support", f"restated by {source}"])
        return best
    return _new_lesson(mem, text, conditions, ts, [source], [[ts, "created", source]])


def _new_lesson(mem, text, conditions, ts, sources, history, support=1, contradict=0):
    lesson = {"id": f"L{mem['next_id']}", "text": text, "conditions": conditions,
              "support": support, "contradict": contradict, "status": "active",
              "created_utc": ts, "updated_utc": ts, "sources": sources[-20:],
              "history": history}  # fmt: skip
    mem["next_id"] += 1
    mem["lessons"].append(lesson)
    return lesson


def apply_reflection(mem: dict[str, Any], out: Any, now: pd.Timestamp, source: str) -> dict:
    """Apply a reflection answer; unknown ids and malformed items are skipped, never fatal.
    A lesson contradicted at least twice and more often than supported is auto-retired."""
    if not isinstance(out, dict):
        raise ValueError(f"reflection is not an object: {str(out)[:120]!r}")
    ts = _iso(now)
    by = {x["id"]: x for x in mem["lessons"] if x["status"] == "active"}
    done: dict[str, list] = {"support": [], "contradict": [], "retire": [], "merged": [],
                             "new": [], "skipped": []}  # fmt: skip
    for u in out.get("lesson_updates") or []:
        x = by.get(u.get("id")) if isinstance(u, dict) else None
        v = u.get("verdict") if isinstance(u, dict) else None
        if x is None or v not in ("support", "contradict", "retire", "keep"):
            done["skipped"].append(u)
            continue
        if v == "keep":
            continue
        note = str(u.get("note") or "")[:200]
        if v == "retire":
            x["status"] = "retired"
        else:
            x[v] += 1
        x["updated_utc"] = ts
        x["history"].append([ts, v, note])
        done[v].append(x["id"])
    for m in out.get("merges") or []:
        ids = [i for i in dict.fromkeys(m.get("ids") or []) if i in by and
               by[i]["status"] == "active"] if isinstance(m, dict) else []  # fmt: skip
        if len(ids) < 2 or not str(m.get("text") or "").strip():
            done["skipped"].append(m)
            continue
        new = _new_lesson(
            mem, str(m["text"]).strip()[:240], m.get("conditions"), ts,
            [x for i in ids for x in by[i]["sources"]], [[ts, "merged", ",".join(ids)]],
            sum(by[i]["support"] for i in ids), sum(by[i]["contradict"] for i in ids),
        )  # fmt: skip
        for i in ids:
            by[i]["status"] = "merged"
            by[i]["history"].append([ts, "merged", f"into {new['id']}"])
        done["merged"].append([ids, new["id"]])
    for n in (out.get("new_lessons") or [])[:MAX_NEW]:
        if isinstance(n, dict) and str(n.get("text") or "").strip():
            x = add_lesson(mem, str(n["text"]), n.get("conditions"), now, source)
            done["new"].append(x["id"] if x else None)
    for x in mem["lessons"]:
        if x["status"] == "active" and x["contradict"] >= 2 and x["contradict"] > x["support"]:
            x["status"] = "retired"
            x["history"].append([ts, "retired", "auto: contradicted more than supported"])
            done["retire"].append(x["id"])
    return done


class Memory:
    """The shared memory file. Every write is read-modify-write under an exclusive flock, so
    parallel runs (shadow + live) never lose each other's updates."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self, now: pd.Timestamp | None = None) -> dict[str, Any]:
        if self.path.exists():
            mem = json.loads(self.path.read_text())
            if mem.get("schema", 1) > MEM_SCHEMA:
                raise SystemExit(f"oil: memory schema {mem['schema']} is newer than this code")
            return mem
        return empty_memory(now if now is not None else pd.Timestamp.now())

    @contextmanager
    def edit(self, now: pd.Timestamp) -> Iterator[dict[str, Any]]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(".lock").open("a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                mem = self.load(now)
                yield mem
                tmp = self.path.with_suffix(".tmp")
                tmp.write_text(json.dumps(mem, indent=1, default=str))
                tmp.replace(self.path)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def sync(self, run_id: str, state: dict[str, Any], now: pd.Timestamp) -> None:
        """Upsert the run's scored decisions and closed trades; take in its new lessons."""
        with self.edit(now) as mem:
            eps = {e["key"]: e for e in mem["episodes"]}
            for e in state.get("evals", []):
                k = _key(run_id, e["ts"])
                eps[k] = {"key": k, "run_id": run_id} | {
                    f: e.get(f) for f in ("ts", "decision", "gross_est_pct", "cost", "net",
                                          "best_net", "worst_net", "thesis")
                }  # fmt: skip
            mem["episodes"] = sorted(eps.values(), key=lambda e: e["ts"])[-300:]
            trs = {t["key"]: t for t in mem["trades"]}
            for t in state.get("closed", []):
                k = _key(run_id, t["closed_utc"])
                trs[k] = {"key": k, "run_id": run_id} | t
            mem["trades"] = sorted(trs.values(), key=lambda t: t["closed_utc"])[-100:]
            for ts, text in state.get("lessons", []):
                add_lesson(mem, text, None, now, f"{run_id}@{str(ts)[:16]}", _key(ts, text))

    def import_run(self, state_path: Path, now: pd.Timestamp) -> dict[str, int]:
        st = json.loads(Path(state_path).read_text())
        before = self.load(now)
        self.sync(st["run_id"], st, now)
        after = self.load(now)
        return {k: len(after[k]) - len(before[k]) for k in ("episodes", "trades", "lessons")}

    def view(self, mem: dict[str, Any] | None = None) -> dict[str, Any]:
        """The analyst's view: top active lessons + decision stats across all runs."""
        mem = mem or self.load()
        act = sorted((x for x in mem["lessons"] if x["status"] == "active"), key=_rank,
                     reverse=True)[:PROMPT_LESSONS]  # fmt: skip
        return {
            "lessons": [
                {k: x[k] for k in ("id", "text", "conditions", "support", "contradict")}
                for x in act
            ],  # fmt: skip
            "by_decision_4h_all_runs": _by_decision(mem["episodes"]),
        }

    def due_reflection(self, now: pd.Timestamp) -> str | None:
        mem = self.load(now)
        if not mem["lessons"] and not mem["episodes"]:
            return None
        t = pd.Timestamp(_iso(now))
        last_try = mem["last_reflection_try_utc"]
        if last_try and (t - pd.Timestamp(last_try)).total_seconds() < REFLECT_RETRY_H * 3600:
            return None
        if len(mem["trades"]) > mem["reflected_trades"]:
            return "closed trade"
        since = pd.Timestamp(mem["last_reflection_utc"] or mem["created_utc"])
        new = len(mem["episodes"]) - mem["reflected_episodes"]
        if (t - since).total_seconds() >= REFLECT_EVERY_H * 3600 and new >= MIN_NEW_EPISODES:
            return "scheduled reflection"
        return None

    def reflect_input(self, now: pd.Timestamp) -> dict[str, Any]:
        mem = self.load(now)
        return {
            "as_of_utc": _iso(now),
            "lessons": [
                {
                    k: x[k]
                    for k in ("id", "text", "conditions", "support", "contradict", "created_utc")
                }
                for x in mem["lessons"]
                if x["status"] == "active"
            ],  # fmt: skip
            "episodes": [
                {
                    k: e.get(k)
                    for k in (
                        "run_id",
                        "ts",
                        "decision",
                        "gross_est_pct",
                        "cost",
                        "net",
                        "best_net",
                        "worst_net",
                        "thesis",
                    )
                }
                for e in mem["episodes"][-40:]
            ],  # fmt: skip
            "closed_trades": mem["trades"][-10:],
            "by_decision_4h_all_runs": _by_decision(mem["episodes"]),
        }

    def finish_reflection(self, now: pd.Timestamp, out: Any, source: str) -> dict[str, Any]:
        """Record the attempt; apply ``out`` when it is not None (None = the call failed)."""
        with self.edit(now) as mem:
            mem["last_reflection_try_utc"] = _iso(now)
            if out is None:
                return {}
            done = apply_reflection(mem, out, now, source)
            mem |= {"last_reflection_utc": _iso(now), "reflections": mem["reflections"] + 1,
                    "reflected_episodes": len(mem["episodes"]),
                    "reflected_trades": len(mem["trades"])}  # fmt: skip
            return done


def _by_decision(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    by: dict[str, Any] = {}
    for d in DECISIONS:
        xs = [e["net"]["4"] for e in episodes if e["decision"] == d and "4" in (e["net"] or {})]
        if xs:
            by[d] = {"n": len(xs), "mean_net_4h": round(sum(xs) / len(xs), 5),
                     "share_positive": round(sum(x > 0 for x in xs) / len(xs), 2)}  # fmt: skip
    return by


def make_reflector(analyst: Any) -> Any:
    """The same LLM backend with the reflection prompt and schema; None for the rule and
    replay analysts (no model to reflect with)."""
    from wallex.oil.analyst import SUBMIT, AnthropicAPI, ClaudeCLI

    if isinstance(analyst, ClaudeCLI):
        return replace(analyst, system=REFLECT_SYSTEM, schema=REFLECT_SCHEMA, max_turns=6,
                       timeout=min(analyst.timeout, 180))  # fmt: skip
    if isinstance(analyst, AnthropicAPI):
        system = REFLECT_SYSTEM.replace(
            "Answer only with the JSON object required by\nthe schema.",
            f"Deliver the answer only by calling the {SUBMIT} tool once.",
        )
        return replace(analyst, system=system, schema=REFLECT_SCHEMA, search=False,
                       max_turns=4, timeout=min(analyst.timeout, 180))  # fmt: skip
    return None
