"""Reliability: single-instance lock, heartbeat + health check, state schema, log rotation."""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from wallex.oil.config import TICK_S

SCHEMA = 3  # state file version; bump + add a step to MIGRATIONS on any layout change
STALE_S = 2 * TICK_S + 120  # heartbeat older than this → the loop is stuck or dead


@contextmanager
def run_lock(path: Path) -> Iterator[None]:
    """Exclusive non-blocking flock: a second process on the same run_id exits at once.
    The kernel drops the lock when the holder dies, so a crash never leaves it stale."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            f.seek(0)
            raise SystemExit(f"oil: {path.name} is held by pid {f.read().strip() or '?'}") from None
        f.truncate(0)
        f.write(str(os.getpid()))
        f.flush()
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# --- state schema -----------------------------------------------------------------------


def _v1_to_v2(st: dict[str, Any]) -> dict[str, Any]:
    """v1 (T13, oil-s1): no version field, no log day, no agent failure streak."""
    return st | {"log_day": None, "agent_fail_streak": 0}


def _v2_to_v3(st: dict[str, Any]) -> dict[str, Any]:
    """v2 (T14.6): no evidence cache (T14.3)."""
    return st | {"evidence_cache": []}


MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {1: _v1_to_v2, 2: _v2_to_v3}


def migrate(st: dict[str, Any]) -> dict[str, Any]:
    """Bring a loaded state file up to SCHEMA; refuse a state written by newer code."""
    v = st.get("schema", 1)
    if v > SCHEMA:
        raise SystemExit(f"oil: state schema {v} is newer than this code ({SCHEMA})")
    while v < SCHEMA:
        st = MIGRATIONS[v](st)
        v += 1
    return st | {"schema": SCHEMA}


# --- heartbeat / health -----------------------------------------------------------------


def write_heartbeat(log_dir: Path, rec: dict[str, Any]) -> None:
    tmp = log_dir / "heartbeat.tmp"
    tmp.write_text(json.dumps(rec | {"pid": os.getpid()}, default=str))
    tmp.replace(log_dir / "heartbeat.json")


def _alive(pid: Any) -> bool | None:
    try:
        os.kill(int(pid), 0)
    except (TypeError, ValueError):
        return None
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def health(log_dir: Path, now: float) -> dict[str, Any]:
    """ok / stale (no tick for STALE_S) / dead (pid gone) / missing; ``ok`` drives the exit code."""
    path = log_dir / "heartbeat.json"
    if not path.exists():
        return {"status": "missing", "ok": False, "path": str(path)}
    hb = json.loads(path.read_text())
    age = now - hb["epoch"]
    alive = _alive(hb.get("pid"))
    status = "dead" if alive is False else "stale" if age > STALE_S else "ok"
    if status == "ok" and hb.get("last_error"):
        status = "degraded"  # alive and ticking, but the last tick raised
    return hb | {"status": status, "ok": status == "ok", "age_s": round(age), "pid_alive": alive}


# --- daily log rotation -----------------------------------------------------------------


def rotate(log_dir: Path, day: str) -> list[Path]:
    """Move every current ``<name>.jsonl`` to ``<name>.<day>.jsonl`` (the day it was written)."""
    moved = []
    for p in sorted(log_dir.glob("*.jsonl")):
        if p.stem.count(".") == 0:  # current file, not an already rotated one
            dst = p.with_name(f"{p.stem}.{day}.jsonl")
            p.replace(dst)
            moved.append(dst)
    return moved


def read_jsonl(log_dir: Path, name: str) -> list[dict[str, Any]]:
    """All records of a log in time order: rotated days first, then the current file."""
    stem = Path(name).stem
    paths = sorted(log_dir.glob(f"{stem}.*.jsonl")) + [log_dir / name]
    return [json.loads(x) for p in paths if p.exists() for x in p.read_text().splitlines() if x]
