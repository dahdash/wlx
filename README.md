# WLX: an automated trading engine for Wallex

Built for the Wallex AI Algorithmic Trading Challenge. It trades a dedicated 5,000,000-toman
Wallex sub-account from Thursday 2026-10-22 to the official end on Monday 2026-10-26 (the
organizers extended the trading period) without any manual orders. [فارسی](README.fa.md)

The engine has two parts that share one account:

- **Strategy G** is a deterministic, validated trend follower on five liquid USDT spot
  markets (BTC, ETH, XRP, SOL, BNB).
- **The oil agent** is an AI agent built on Claude. It researches the oil market on the web
  and trades USOON/USDT, but only while Strategy G is fully in cash. Its proposals pass a
  deterministic gate before any order is placed (see [AI use](#4-ai-use-the-oil-agent)).

Every decision is logged with its inputs and its reason. Every order is logged with its
fill. See [Decision log](#7-decision-log).

**Results and evidence**

- [RESULTS.md](RESULTS.md): account value in toman, how the return splits into engine result,
  conversion fees and the USDT/TMN rate, every order, uptime.
- [Dashboard](docs/demo/dashboard.html): the same numbers as charts, plus the oil agent's
  decisions, their forward scores and its lessons (one self-contained HTML file).
- [Decision log](docs/decision-log.md) ([CSV](docs/decision-log.csv)): every G cycle per coin
  and every oil-agent decision, with signal and reason, and every order with its fill.
- [Backtest report](docs/backtest/metrics.md): tearsheets, permutation test and an
  independent cross-check; full tables in [strategy-g-results.md](docs/strategy-g-results.md).

## 1. Account flow

1. **Start (TMN → USDT).** `scripts/convert_tmn.py` first places a small probe order on
   USDTTMN. Only if the probe fills does it place one capped LIMIT buy for the remaining
   toman. USDT pairs cost 0.10 % maker / 0.13 % taker, against 0.25 % / 0.30 % on TMN pairs.
2. **Trading.** Strategy G runs one decision cycle every 30 minutes on BTCUSDT, ETHUSDT,
   XRPUSDT, SOLUSDT and BNBUSDT. Each coin gets one equal slot (equity / 5). While all five
   G gates are off, the oil agent may use part of the idle USDT on USOONUSDT.
3. **End (USDT → TMN).** The oil agent goes flat by itself one hour before the end. Then
   `scripts/liquidate_tmn.py` (Monday 2026-10-26, 23:05 Tehran = 19:35 UTC) sells every coin to USDT and USDT to toman on
   USDTTMN, retrying every 30 s while the market rejects orders. From 23:20 Tehran it routes
   USDT → BTC → TMN instead, because USDTTMN halts at 23:30. So the result is realised in
   toman before the official end (the challenge measures net value in toman).

## 2. Architecture

```
                  ┌───────────────────────────── every 30-min bar close ─────────────────┐
Wallex REST ──►   │ wallex.run: candles (1-min UDF → 30-min, 60-min for the 28-day gate) │
(api.wallex.ir)   │   → strategy.g: votes / weight per coin                             │
                  │   → risk.RiskState: kill switch, loss pause, per-slot cap            │
                  │   → exec.Executor: maker-first limit orders, re-peg, cross at 20 min │
                  │   → data/state/<run>/cycles.jsonl, trades.csv, alerts.log            │
                  └──────────────────────────────────────────────────────────────────────┘
                  ┌───────────────────────────── every 5 min (oil agent) ────────────────┐
Wallex + Binance  │ wallex.oil: exits first → is an analysis due? (G all off, triggers)  │
CLUSDT reference  │   → as-of snapshot (book, closed 15-min bars, WTI, calendar, memory) │
web (news, EIA)   │   → analyst: claude -p with WebSearch / WebFetch → JSON decision     │
                  │   → deterministic gate (cost, spread, depth, freshness, limits)       │
                  │   → exec.Executor on LiveBroker → data/state/<run>/decisions.jsonl … │
                  └──────────────────────────────────────────────────────────────────────┘
```

| module | role |
| --- | --- |
| `src/wallex/client.py` | `WallexClient`, an `httpx` wrapper for the Wallex REST API (auth via `X-API-Key`). It unwraps the `{success, message, result}` envelope and raises `WallexError` on errors. Order calls are dry-run unless `live=True`. |
| `src/wallex/strategy/g.py` | Strategy G: votes, 28-day gate, target weights. |
| `src/wallex/risk.py` | `RiskState`: kill switch, loss pause, optional stop and crash gate, per-slot cap. Also `flatten()`. |
| `src/wallex/exec.py` | `MarketRules` (tick, step, minNotional), `Executor` (maker-first execution), `PaperBroker` (simulated fills against the real book) and `LiveBroker` (real orders, double opt-in). |
| `src/wallex/run.py` | The 30-min G runner, in `shadow` or `live` mode. It is restart-safe: state is kept in `data/state/<run_id>.json`. Equity counts every balance of the account (free and locked), including USOON held by the oil agent, so an oil trade never looks like a G loss. |
| `src/wallex/oil/` | The oil agent: `cli` (entry point), `agent` (5-min loop), `feeds` (Wallex USOON, Binance WTI perp, G gates), `snapshot`, `calendar` (scheduled oil data releases, US hours), `analyst` (prompt, decision contract, validation, model adapters), `prompts/` (versioned system prompts), `gate` (entry gate and exits), `memory` (forward scoring, lessons), `evaluate`, `report`, `dashboard`, `ops` (lock, heartbeat, health). |
| `src/wallex/sim.py`, `validate_sma.py`, `report_g.py`, `data.py` | Backtest, validation and reports on Binance history. |
| `scripts/convert_tmn.py` | TMN ↔ USDT conversion with a live probe. |
| `scripts/liquidate_tmn.py` | End-of-contest liquidation: coins → USDT → TMN (BTC route fallback). Dry run unless `--live` and `WALLEX_LIVE=1`. |
| `scripts/export_decision_log.py` | Exports the decision log (CSV and Markdown). |
| `scripts/build_results.py`, `build_dashboard.py` | Results: `docs/results/results.json` from the run logs and public USDTTMN prices, then `RESULTS.md`, `RESULTS.fa.md` and `docs/demo/dashboard.html` from it. Read-only. |
| `scripts/backtest_report.py` | Backtest report: tearsheets, permutation test, cross-check (`docs/backtest/`). |

`run.py` also has an optional AI reweighting hook (`ai.py`, `--ai`). It is not enabled in the
live run, so G trades as pure rules.

The G runner uses only Wallex market data (UDF candles, depth) and the Wallex account
endpoints (balances, orders). The oil agent also reads the Binance CLUSDT perpetual as a
WTI reference (read-only, never traded).

## 3. Strategy G: entry and exit rules

All rules are fixed in code (`strategy/g.py`, `run.py`). Nothing is tuned while the engine
runs.

**Signal (per coin, at each 30-min bar close):**

- Six fast×slow SMA pairs on 30-min closes vote: (24,72) (36,108) (48,144) (72,192) (96,240)
  (144,336), which spans 12 hours to 7 days.
- Each vote uses a ±0.3 % band:
  - it turns **on** when SMA_fast > SMA_slow × 1.003;
  - it turns **off** when SMA_fast < SMA_slow × 0.997;
  - otherwise it keeps its previous state, which avoids whipsaw.
- **Trend gate.** Weight = votes / 6 while the coin's 28-day return is positive (daily close,
  00:00 UTC). Otherwise the weight is 0.

**Orders:**

- Target position = weight × slot (slot = equity / 5).
- **Entry / add:** buy when the target exceeds the held value by at least minNotional
  (1 USDT ≈ 0.27 slot).
- **Exit / reduce:** sell when the target is lower by at least minNotional. A full exit
  (weight 0, for example when the gate turns off) always executes.
- The engine is long-only spot: no leverage, no shorts, no margin.

**Execution (`Executor`):**

1. A LIMIT order is posted at the best bid (buy) or best ask (sell), to get the 0.10 % maker
   fee.
2. It is re-pegged every 5 minutes.
3. After 20 minutes the remainder crosses the spread, capped at 0.2 % beyond the opposite
   best price.
4. Quantities and prices are rounded to the market's `stepSize` and `tickSize`. Orders below
   `minNotional` are skipped.

## 4. AI use: the oil agent

Strategy G is in cash whenever no coin passes its 28-day gate. The oil agent puts that idle
capital to work on a market with its own news flow: **USOON/USDT** (Ondo tokenized United
States Oil Fund, an allowed Wallex market). The agent decides *whether* there is a trade; code
decides *whether it is allowed* and *how big* it is.

Each 5-minute tick runs these steps:

1. **Exits come first**, and they are deterministic: price stop, thesis expiry, maximum
   holding time, contest end, kill file. An open position is also reviewed by the agent every
   30 minutes (`EXIT_REVIEW`).
2. **Wake-up check** (deterministic). An analysis runs only if all of the following hold:
   - all five G gates are off (read from the live G run's own log);
   - the daily call budget and the minimum spacing between calls allow it;
   - a trigger fired: an hourly sweep, a 1-hour move of USOON or WTI of at least 0.6 %, a
     USOON-vs-WTI 4-hour divergence of at least 0.6 %, or a position review. Triggers use
     closed 15-minute bars only, so a forming bar never wakes the agent.
3. **Snapshot.** The agent receives a timestamped as-of packet with:
   - the Wallex USOONUSDT book and closed 15-min candles;
   - the Binance CLUSDT perpetual as the WTI reference, and the USOON-vs-WTI divergence;
   - the oil calendar (scheduled EIA/API/Baker Hughes releases, US holidays, US equity
     hours), computed from rules, so the agent knows what is due and what is already out;
   - the G state, its open position and active hypotheses;
   - an evidence cache: sources it cited in the last 24 h, so it does not search them again;
   - its **own track record**: past decisions scored 1/2/4/8 h later net of fees, closed
     trades, and ranked lessons from its long-term memory.
4. **Analyst.** `claude -p` runs with only the WebSearch and WebFetch tools and a versioned
   system prompt (`oil/prompts/analyst-v2.txt`). It must:
   - research the current drivers (EIA weekly report vs consensus, OPEC+, supply
     disruptions, sanctions, shipping);
   - use only sources published before the snapshot time, each with a URL and publication
     time;
   - separate fact, forecast, rumor and interpretation;
   - ask whether the news is already priced in, and look for counter-evidence;
   - answer with a JSON object against a strict schema. The decision is `BUY_CANDIDATE`,
     `WATCH`, `NO_TRADE` or `EXIT_REVIEW`, together with a thesis, horizon, expiry,
     evidence, invalidation conditions, an estimated gross move, and one `lesson` for its
     next call.

   Invalid output, a timeout, or look-ahead evidence becomes `NO_TRADE` (fail closed).
5. **Gate** (deterministic, on a freshly fetched book). The agent's proposal is not an order.
   Code rejects it unless all of these hold:
   - spread and book depth are within limits;
   - quotes and candles are fresh;
   - the horizon and expiry are valid;
   - there is supporting evidence and there are invalidation conditions;
   - the estimated gross move minus our own round-trip cost (both fees plus the full spread)
     clears a minimum net edge;
   - WTI has led USOON by at least 1.5 % over 4 hours (the only setup that beat the
     round-trip cost on 58 days of history). Wallex describes USOON as a tokenized oil ETF
     whose price follows world oil prices and the fund's value, kept in line by Wallex's own
     market making, with temporary gaps. Buying a WTI lead bets that this market making
     closes the gap;
   - the daily entry, loss-streak and daily-loss limits are not exceeded.

   The agent's confidence never sets the size: the notional is fixed, there is one position
   at a time, and a hard stop applies.
6. **Execution** uses the same `Executor` with `LiveBroker`.

**Learning.** Every decision, including `NO_TRADE`, is scored forward against what the price
did. The scores and each call's `lesson` go into a shared memory (`data/memory/oil.json`):
similar lessons are merged and ranked by how often later outcomes supported or contradicted
them, and a scheduled reflection call condenses them. The top lessons are fed back into
every snapshot.

**Model-agnostic and reproducible.** The analyst is an adapter: `--analyst cli` (`claude -p`),
`api` (Anthropic Messages API with server-side web search), `rule` (a no-LLM baseline) or
`replay:<run>`. Every snapshot and raw answer is logged, so `--replay <run>` re-decides a
run's recorded snapshots and must reproduce every logged decision. `--report` and the
`evaluate` module score the agent against the rule baseline on the same snapshots, without
calling the model.

## 5. Risk management

All limits are hard code paths. Nothing (the AI included) can loosen them.

| rule | value | where |
| --- | --- | --- |
| Position size | one equal slot per coin, maximum 20 % of equity each, 5 slots | `run.py`, `RiskConfig.max_pos_frac` |
| Leverage / shorts | none (spot, long-only) | strategy |
| Kill switch | equity ≤ 90 % of the run's peak → flatten everything and stop | `RiskConfig.kill_dd` |
| Loss pause | rolling 6-h loss ≥ 4 % → exposure 0 for 2 h | `RiskConfig.pause_loss` |
| Trend gate | 28-day return ≤ 0 → coin weight 0 | `strategy/g.py` |
| Per-slot stop, crash gate | implemented, **off** in the live run: both lowered the backtest return (crash gate: results §5; 2 % stop: validation robustness table) | `risk.py`, `run.py --stop` |
| Execution | price cap 0.2 % beyond the opposite best; tick/step rounding; minNotional | `exec.py` |
| Stale data | a coin with stale candles or book is not traded in that cycle | `run.py` |
| Live opt-in (G) | real orders need `--mode live` **and** `WALLEX_LIVE=1`; otherwise paper | `run.py`, `LiveBroker` |
| Live opt-in (oil) | `--mode live --arm` **and** `WALLEX_LIVE=1`; one live oil run per account (lock file); restart reconciles open orders and balances | `oil/cli.py`, `oil/ops.py` |
| Oil size | 6 USDT per entry; a code constant caps any live entry at 7 USDT (no flag raises it) | `oil/config.py` |
| Oil limits | one position; 2.5 % hard stop; max hold 12 h; ≤ 4 entries and ≤ 0.50 USDT realised loss per day; 3 losing trades in a row → no more entries; ≤ 30 analyses per day, ≥ 30 min apart; entries only while G is all off and between 02:00 and 23:00 UTC; flat at 18:30 UTC on the last day; `KILL` file | `oil/config.py` (`aggressive` profile), `oil/gate.py` |
| Oil exits | may only sell what the agent itself holds; never touches G's coins | `oil/agent.py` |

## 6. Backtest and validation

The data is Binance 1-minute klines, resampled to 30 minutes, from 2020-09 to 2026-09:

- the **discovery** span is 2020-09 → 2023-12;
- the **holdout** span is 2024-01 → 2026-09.

Decisions are made at the bar close and filled at that close. Costs are charged per side.
The pass bar was written before the run (`src/wallex/validate_sma.py` docstring).

Holdout results (from `docs/strategy-g-results.md`):

| holdout 2024-01 → 2026-09 | total | CAGR | Sharpe | max drawdown | mean / 36 h | trades / 36 h |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G, maker 0.10 % | +59.9 % | +18.6 % | 0.81 | 22.3 % | +0.08 % | 2.9 |
| G, taker 0.13 % + half spread | +27.7 % | +9.3 % | 0.48 | 30.2 % | +0.05 % | 2.9 |
| buy & hold N5, no cost | +108.0 % | +30.5 % | 0.75 | 63.3 % | +0.17 % | 0 |

- **Is it luck?** We shifted each coin's weight series in time (150 circular shifts), which
  keeps the exposure and the trade count but breaks the timing. The real strategy beats
  these shifts with one-sided **p = 0.026** (the validation run in `docs/validation-g.md`
  gives p = 0.033).
- **What G is for.** G does not try to beat buy & hold in a bull market. It aims to stay
  positive with a third of the drawdown. In 2022 G returned −6 % while buy & hold lost 71 %.
- **Robustness.** 27 of 27 parameter cells (band × gate length, pair scale × gate length)
  have a positive holdout CAGR.
- **Caveats.** The mean 36-h return is small (+0.08 %), and most 36-h windows are flat.
  In a 3-day contest, G's main job is to avoid losses and trade only when a trend is
  confirmed.
- The full tables are in `docs/strategy-g-results.md` (per year, per coin, contest-length
  windows, sensitivity, gate states) and `docs/validation-g.md` (pass bar, null, ablations).

**Shadow test.** `python -m wallex.run --mode shadow` runs the full live loop against the
real Wallex book with paper fills (`PaperBroker`). It was run before going live. The oil
agent was forward-tested the same way (`--mode shadow`) before its live run, and a shadow
copy keeps running beside the live one.

## 7. Decision log

The engines write logs while they run and never edit them afterwards:

- **G** writes `data/state/<run>/cycles.jsonl`, one record per 30-min cycle. Each record
  holds:
  - prices, equity, holdings and stale flags;
  - per coin: every SMA vote, the 28-day return and gate, the weight, and a reason;
  - the risk verdict and targets;
  - the order intents and their fills (price, quantity, fee, status).
- **Oil agent** writes `data/state/<run>/decisions.jsonl` (decision, thesis, evidence,
  gate verdict, snapshot id), `snapshots.jsonl` (the exact input), `agent.jsonl` (prompt and
  raw answer), `trades.csv` and `reconcile.jsonl`.
- **Conversions** write `data/state/convert-tmn.jsonl` (every request and response).

To export one table with time, engine, symbol, action, target, fill, fee, signal and reason:

```
python scripts/export_decision_log.py --run-id live-1 --oil-run oil-probe --oil-run oil-live-1 --oil-run oil-s2 \
    --out docs/decision-log
```

This writes `docs/decision-log.csv` and `docs/decision-log.md` (both shipped here). `oil-s2`
is the shadow run of the oil agent: same code and profile, paper fills, `mode` = `shadow`.

## 8. How to re-run

```
python -m venv .venv && .venv/bin/pip install -e '.[dev]'
PYTHONPATH=src .venv/bin/pytest -q          # unit tests, no network (httpx.MockTransport)
cp .env.example .env                         # WALLEX_API_KEY=...; keep WALLEX_LIVE=0
```

**Strategy G, shadow (paper) run against the live Wallex book:**

```
.venv/bin/python -m wallex.run --mode shadow --run-id shadow-1 --equity 18.5
```

**Strategy G, live run** (real money; both switches are required):

```
WALLEX_LIVE=1 .venv/bin/python -m wallex.run --mode live --run-id live-1
```

- `--run-id` resumes a stopped run from `data/state/<run_id>.json`.
- `--cycles N` stops after N cycles.

**Oil agent** (needs the `claude` CLI logged in, or `--analyst api` with `ANTHROPIC_API_KEY`):

```
.venv/bin/python -m wallex.oil --mode shadow --profile aggressive --run-id oil-shadow-1 --g-run live-1
WALLEX_LIVE=1 .venv/bin/python -m wallex.oil --mode live --arm --profile aggressive --run-id oil-live-1
.venv/bin/python -m wallex.oil --health --run-id oil-live-1      # heartbeat; exit 1 unless ok
.venv/bin/python -m wallex.oil --report --run-id oil-live-1      # decisions, forward scores, lessons
.venv/bin/python -m wallex.oil --replay oil-live-1               # re-decide logged snapshots
.venv/bin/python -m wallex.oil.dashboard --run-id oil-live-1     # static HTML dashboard
touch data/state/oil-live-1/KILL                                 # stop: flatten, no new entries
```

**Backtest and validation.** Put the Binance 1-minute klines for BTC, ETH, XRP, SOL and BNB
vs USDT at `data/parquets/<SYMBOL>USDT.parquet.gz`. Each file needs a naive-UTC index named
`time` and open/high/low/close/volume columns (public data from data.binance.vision). Then:

```
.venv/bin/python -m wallex.validate_sma      # → docs/validation-g.md
.venv/bin/python -m wallex.report_g          # → docs/strategy-g-results.md
```

**Backtest report** (extra packages in `requirements-report.txt`):

```
.venv/bin/pip install -r requirements-report.txt
.venv/bin/python scripts/backtest_report.py series && \
  .venv/bin/python scripts/backtest_report.py tearsheets && \
  .venv/bin/python scripts/backtest_report.py charts && \
  .venv/bin/python scripts/backtest_report.py crosscheck   # → docs/backtest/
```

**Results and dashboard** (from the run logs in `data/state/`; read-only):

```
.venv/bin/python scripts/build_results.py --state data/state --out docs/results/results.json
.venv/bin/python scripts/build_dashboard.py --results docs/results/results.json \
    --out docs/demo/dashboard.html --md-dir .
```

## 9. Reproducibility

The files in this repository are byte-identical copies of the code that traded, taken from
tagged commits of the development repository. `MANIFEST.sha256` lists each file's hash and
its source commit.

- **Strategy G** (runtime, strategy, risk, execution, runner, conversion and their tests)
  comes from the commit `b3e4b98`, which ran live from 2026-10-22 14:45 UTC to the end.
- **The oil agent** (`src/wallex/oil/`, `tests/test_oil.py`) comes from the commit
  `84ac0e0`, which ran live from 2026-10-23 10:08 UTC to the end. It differs from the
  previous live commit `be2ecb2` only in the contest end date (`oil/config.py`).
- **The liquidation** (`scripts/liquidate_tmn.py`) comes from the commit `84c6cf8`.

Every commit that ran live during the contest (times in UTC, 2026-10-22 unless dated):

| run | commit | from | what it did |
| --- | --- | --- | --- |
| `convert-tmn` | `ee3495a` | 08:00 | TMN → USDT probe order |
| `convert-tmn` | `3872720` | 08:01 | TMN → USDT bulk order (retries the market-closed reply; final) |
| G `live-1` | `b7ebca2` | 09:30 | Strategy G, first version |
| G `live-1` | `f380585` | 14:01 | equity counts USOON and locked balances |
| G `live-1` | `b3e4b98` | 14:45 | locked balances counted once (final) |
| oil `oil-probe` | `1b40cd1` | 14:43 | one live micro round trip on USOONUSDT to check the order path |
| oil `oil-live-1` | `b3e4b98` | 14:49 | oil agent, first live version |
| oil `oil-live-1` | `4a2b84a` | 15:22 | analyst adapters, shared memory, offline evaluation |
| oil `oil-live-1` | `be2ecb2` | 16:22 | versioned prompt, oil calendar, evidence cache, closed-bar triggers |
| oil `oil-live-1` | `84ac0e0` | 10-23 10:08 | contest end moved to Monday 2026-10-26 (final) |
| `liquidate` | `84c6cf8` | 10-26 19:35 | coins → USDT → TMN at the end |

Validation, report and export modules were added for the deliverables. They do not change any
traded code path. Credentials come only from the environment or an untracked `.env`.
