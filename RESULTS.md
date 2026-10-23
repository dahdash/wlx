# Results

[فارسی](RESULTS.fa.md) · [dashboard](docs/demo/dashboard.html) · [decision log](docs/decision-log.md) · data: `docs/results/results.json`

**Interim, built 2026-10-23T19:09:00 UTC.** The account keeps trading until the official end; the score uses the toman balance after the end-of-contest sale to toman. Until then the value is marked at the USDTTMN bid minus the 0.3 % taker fee.

| | |
| --- | --- |
| Start | 5,000,000 TMN (2026-10-22T08:00:06 UTC) |
| Value now | 4,966,888 TMN |
| Net return in toman | -0.66 % |
| of which: equity in USDT | -0.075 % |
| of which: TMN ↔ USDT conversion fees | -0.60 % |
| of which: USDT/TMN rate | +0.01 % (268,137 → 268,160) |
| Max drawdown (toman / USDT) | -1.74 % / -0.079 % |
| Filled orders | 4 |
| Fees paid | ≈ 15,985 TMN |
| Uptime (G / oil live) | 91.2 % / 88.5 % |

The value is USDT equity × USDTTMN plus the 20,704 TMN that could not be converted at the start (below the market's minimum order). The engines trade in USDT, so the toman return is the USDT result times the USDT/TMN rate move, less the fees of converting in and out; the lines above split it.

## Strategy G (trend, BTC/ETH/XRP/SOL/BNB)

62 cycles of 30 min, 0 orders. G only buys a coin whose 28-day return is positive; 0 cycles had any gate open (best 28-day return seen: -0.53 %, BTCUSDT). In a falling market it holds USDT by design, which is the drawdown control the backtest selected. Risk events: none.

## Oil agent (USOON, AI analyst + deterministic risk gate)

| run | decisions | trades | 4-h: entry would have lost | 4-h mean net | uptime |
| --- | --- | ---: | ---: | ---: | ---: |
| live (armed) | 36 (35 NO_TRADE, 1 WATCH) | 0 | 63.3 % | -0.053 % | 88.5 % |
| shadow (paper) | 36 (2 WATCH, 34 NO_TRADE) | 0 | 63.3 % | -0.061 % | 85.8 % |

Every decision is scored 1, 2, 4 and 8 h later against the price, net of the round-trip cost (forward score). A NO_TRADE was right when the entry it declined would have lost. Live and shadow decided in the same minute 28 times and agreed 26 times. The agent wrote 12 lessons from its own scores, which are fed back into its next calls.

## Every filled order

| time (UTC) | engine | market | side | qty | price | fee | why |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 2026-10-22 08:00:06 | convert | USDTTMN | BUY | 0.21 | 267049 | 0.00063 | contest start/end: move the account between TMN and USDT |
| 2026-10-22 08:01:19 | convert | USDTTMN | BUY | 18.36 | 268149 | 0.05508 | contest start/end: move the account between TMN and USDT |
| 2026-10-22 14:43:10 | oil-agent | USOONUSDT | BUY | 0.0091 | 163.26 | 0.0019131476 | live micro round trip (buy, then sell) to check the oil order path before going live |
| 2026-10-22 14:43:10 | oil-agent | USOONUSDT | SELL | 0.009 | 160.21 | 0.001878201 | live micro round trip (buy, then sell) to check the oil order path before going live |

## Backtest and shadow test

- [Metrics by period](docs/backtest/metrics.md), [full tearsheet](docs/backtest/tearsheet-full.html)
- [Validation charts](docs/backtest/validation.html): permutation test, holdout, sensitivity
- [Independent cross-check](docs/backtest/crosscheck.md)
- Shadow test: the oil agent ran in paper mode next to the live run (table above).

## Rebuild

```
python scripts/build_results.py      # data/state + public USDTTMN prices → results.json
python scripts/build_dashboard.py    # → dashboard.html, RESULTS.md, RESULTS.fa.md
```
