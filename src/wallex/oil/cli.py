"""Oil intelligence agent (T13): event-driven long-only USOON/USDT trades while G is sidelined."""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import ExitStack
from dataclasses import replace

import pandas as pd

from wallex.client import WallexClient, load_env
from wallex.oil import config
from wallex.oil.agent import OilAgent
from wallex.oil.analyst import make_analyst, replay_run
from wallex.oil.config import PROFILES
from wallex.oil.memory import Memory
from wallex.oil.ops import health, run_lock
from wallex.oil.report import report


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=["paper", "shadow", "live"], default="shadow")
    p.add_argument("--arm", action="store_true", help="allow real orders (with WALLEX_LIVE=1)")
    p.add_argument("--profile", choices=list(PROFILES), default="safe")
    p.add_argument("--report", action="store_true", help="print the run's evaluation and exit")
    p.add_argument(
        "--health",
        action="store_true",
        help="print the run's heartbeat status; exit 1 unless ok (for cron)",
    )
    p.add_argument("--run-id", help="resume this run (data/state/<run_id>.json)")
    p.add_argument("--g-run", default="live-1", help="live G run whose gates authorise oil")
    p.add_argument("--notional", type=float, help="USDT per entry (overrides the profile)")
    p.add_argument("--ticks", type=int, default=None, help="stop after N 5-min ticks")
    p.add_argument("--once", action="store_true", help="one tick with a forced analysis")
    p.add_argument("--probe", type=float, metavar="USDT", help="live micro round trip, then exit")
    p.add_argument("--force-analysis", action="store_true", help="force analysis on tick 1")
    p.add_argument(
        "--analyst",
        default="cli",
        help="cli (claude -p), api (Anthropic Messages + web_search, ANTHROPIC_API_KEY), "
        "rule (no-LLM baseline), replay:<run_id>",
    )
    p.add_argument(
        "--replay",
        metavar="RUN_ID",
        help="re-decide RUN_ID's recorded snapshots with --analyst (default replay of the "
        "same run: must reproduce every logged decision), print the comparison and exit",
    )
    p.add_argument(
        "--import-lessons",
        nargs="+",
        metavar="RUN_ID",
        help="merge these runs' decisions, trades and lessons into the shared memory "
        "(data/memory/oil.json; idempotent), print the active lessons and exit",
    )
    a = p.parse_args(argv)
    cfg = PROFILES[a.profile]

    def analyst(spec):
        try:
            return make_analyst(spec, timeout=cfg.agent_timeout_s, max_turns=cfg.agent_max_turns)
        except ValueError as e:
            p.error(str(e))

    if a.replay:
        spec = a.analyst if a.analyst != "cli" else f"replay:{a.replay}"
        exact = spec == f"replay:{a.replay}"  # self-replay must match the whole decision
        key = "same" if exact else "same_decision"
        rows = list(replay_run(config.STATE_DIR / a.replay, analyst(spec)))
        for r in rows:
            print(
                f"{r['ts'][:19]} {r['snapshot_id']} logged {r['logged']['decision']:<13} "
                f"replayed {r['replayed']['decision']:<13} {'same' if r[key] else 'DIFF'}"
            )
        n = sum(r[key] for r in rows)
        print(f"{spec} on {a.replay}: {n}/{len(rows)} {'identical' if exact else 'same decision'}")
        sys.exit(0 if not exact or n == len(rows) else 1)
    if a.import_lessons:
        mem, now = Memory(config.MEMORY_PATH), pd.Timestamp.now()
        for rid in a.import_lessons:
            print(rid, mem.import_run(config.STATE_DIR / f"{rid}.json", now))
        print(json.dumps(mem.view(), indent=1, default=str))
        return
    if a.report:
        print(json.dumps(report(a.run_id), indent=1, default=str))
        return
    if a.health:
        if not a.run_id:
            p.error("--health needs --run-id")
        h = health(config.STATE_DIR / a.run_id, time.time())
        print(json.dumps(h, indent=1, default=str))
        sys.exit(0 if h["ok"] else 1)
    if a.notional:
        cfg = replace(cfg, notional=a.notional)
    run_id = a.run_id or time.strftime("oil-%Y%m%d-%H%M%S", time.gmtime())
    load_env()
    with ExitStack() as stack:
        stack.enter_context(run_lock(config.STATE_DIR / f"{run_id}.lock"))
        if a.mode == "live":  # one live oil run per account, whatever its run_id
            stack.enter_context(run_lock(config.STATE_DIR / config.LIVE_LOCK))
        client = stack.enter_context(WallexClient())
        agent = OilAgent(client, mode=a.mode, run_id=run_id, g_run=a.g_run, cfg=cfg,
                         armed=a.arm, analyst=analyst(a.analyst))  # fmt: skip
        print(f"oil run {agent.run_id} mode {a.mode} profile {cfg.profile} g {a.g_run} "
              f"analyst {agent.analyst.name} state {agent.state_path}", flush=True)  # fmt: skip
        if a.probe:
            if a.mode != "live":
                p.error("--probe needs --mode live --arm")
            print(json.dumps(agent.probe(a.probe), indent=1, default=str))
        elif a.once:
            print(json.dumps(agent.tick(force=True), indent=1, default=str))
        else:
            agent.loop(a.ticks, force_first=a.force_analysis)
