"""Constants, operator limits (profiles) and the cost model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from wallex.exec import MAKER_FEE, TAKER_FEE

ROOT = Path(__file__).resolve().parents[3]
STATE_DIR = ROOT / "data" / "state"
MEMORY_PATH = ROOT / "data" / "memory" / "oil.json"  # shared by all oil runs (T14.4)
SYMBOL = "USOONUSDT"  # the only market this module may touch
REF = "CLUSDT"  # Binance USDT-M WTI perp (reference only, never traded)
BINANCE_FAPI = "https://fapi.binance.com"
G_COINS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "BNBUSDT"]
TICK_S = 300
CONTEST_END_UTC = "2026-10-26T19:30:00"  # Mon 23:00 Tehran (deadline extended to Monday)
LIVE_MAX_NOTIONAL = 7.0  # hard cap on one live entry (USDT); code-only, no flag raises it
LIVE_LOCK = "USOONUSDT.live.lock"  # in STATE_DIR: one live oil run per account
DECISIONS = ("BUY_CANDIDATE", "WATCH", "NO_TRADE", "EXIT_REVIEW")
KINDS = ["fact", "forecast", "rumor", "interpretation"]
NY = ZoneInfo("America/New_York")


@dataclass
class OilConfig:
    """Operator limits. The agent cannot change these; only the CLI / code can."""

    notional: float = 3.0  # USDT per entry (fixed; ≥ minNotional 1 USDT)
    max_entries_day: int = 2
    max_losses: int = 2  # consecutive losing trades → no more entries this run
    day_loss_cap: float = 0.10  # USDT realised loss per UTC day → no more entries that day
    stop_pct: float = 0.015  # deterministic price stop on the bid
    max_hold_h: float = 8.0
    min_horizon_h: float = 2.0
    max_spread: float = 0.0040  # 0.40 % of mid
    depth_band: float = 0.005  # depth counted within 0.5 % of the best
    depth_mult: float = 2.0  # book depth on both sides ≥ 2 × notional
    min_net_edge: float = 0.0020  # agent gross estimate − our round-trip cost ≥ 0.20 %
    min_gap_4h: float = 0.020  # WTI 4-h return − USOON 4-h return ≥ 2 % (docs/oil-estimate.md)
    max_quote_age_s: float = 60
    max_candle_age_s: float = 1800  # last 15-min bar must have opened within 30 min
    max_g_age_s: float = 3000  # G cycle record ≤ 50 min old
    entry_hours_utc: tuple[int, int] = (6, 21)  # liquid hours (docs: USOON volume by hour)
    # agent budget / wake-up
    min_call_gap_s: float = 2700  # ≥ 45 min between analyses
    sweep_s: float = 7200  # scheduled sweep every 2 h while flat and eligible
    review_s: float = 3600  # open position review every hour
    max_calls_day: int = 16
    move_trigger: float = 0.010  # |1-h return| of USOON or WTI ≥ 1 %
    div_trigger: float = 0.010  # |USOON − WTI| 4-h return ≥ 1 %
    agent_timeout_s: float = 300
    agent_max_turns: int = 20
    agent_retries: int = 2  # quick failures only: no retry once agent_timeout_s is spent
    agent_backoff_s: float = 30  # doubles per retry
    max_agent_fails: int = 3  # consecutive failed analyses → alert
    stall_s: float = 3 * TICK_S  # gap between ticks above this → stale-loop alert
    max_fail_ticks: int = 3  # consecutive data failures → pause entries
    entry_cross: bool = False  # maker entry may cross after cross_after_s (taker cost)
    cross_after_s: float = 1200
    end_utc: str | None = "2026-10-26T18:30:00"  # flat 1 h before CONTEST_END_UTC
    max_prompt_chars: int = 20000  # snapshot JSON in the prompt; oldest context trimmed first
    evidence_ttl_h: float = 24  # evidence cache shown back to the agent
    max_evidence: int = 20
    profile: str = "safe"

    @property
    def entry_kind(self) -> str:
        return "taker" if self.entry_cross else "maker"


# "Almost risky" (user, 2026-10-22): one third of the account per trade, looser cost gate,
# more frequent analysis, taker entry allowed. Still: one position, fixed size, hard stop.
AGGRESSIVE = OilConfig(
    notional=6.0, max_entries_day=4, max_losses=3, day_loss_cap=0.50, stop_pct=0.025,
    max_hold_h=12.0, min_horizon_h=1.0, max_spread=0.0080, min_net_edge=0.0010, min_gap_4h=0.015,
    entry_hours_utc=(2, 23), min_call_gap_s=1800, sweep_s=3600, review_s=1800,
    max_calls_day=30, move_trigger=0.006, div_trigger=0.006, entry_cross=True,
    cross_after_s=600, profile="aggressive",
)  # fmt: skip
PROFILES = {"safe": OilConfig(), "aggressive": AGGRESSIVE}
HORIZONS = (1, 2, 4, 8)  # hours for the forward evaluation of every decision


# --- costs ------------------------------------------------------------------------------


def round_trip_cost(spread: float, entry: str = "maker", exit: str = "taker") -> float:
    """Fraction of notional: both fees + the full spread (entry may give up the maker fill,
    exit crosses at the bid). Conservative on purpose; slippage at our size is ~0."""
    fee = {"maker": MAKER_FEE, "taker": TAKER_FEE}
    return fee[entry] + fee[exit] + spread


def _last_entry(cfg: OilConfig) -> str | None:
    if not cfg.end_utc:
        return None
    return (pd.Timestamp(cfg.end_utc) - pd.Timedelta(hours=cfg.min_horizon_h)).isoformat()
