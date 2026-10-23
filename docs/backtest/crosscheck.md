# Cross-check: sim.py vs vectorbt

Same bars, same target weights, same trade bars, maker fee 0.10 % per side; one slot per coin, equal slots (portfolio = mean of slot returns, as `sim.stats`).

| case | span | engine | total | CAGR | Sharpe | MDD |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| buy & hold (no fee) | discovery | sim.py | +1149.97% | +113.30% | 1.32 | 79.7% |
| buy & hold (no fee) | discovery | vectorbt | +1149.97% | +113.30% | 1.32 | 79.7% |
| buy & hold (no fee) | holdout | sim.py | +108.00% | +30.51% | 0.75 | 63.3% |
| buy & hold (no fee) | holdout | vectorbt | +108.00% | +30.51% | 0.75 | 63.3% |
| G 0/1 weights, maker | discovery | sim.py | +1454.95% | +127.73% | 1.90 | 41.0% |
| G 0/1 weights, maker | discovery | vectorbt | +1455.13% | +127.74% | 1.90 | 41.0% |
| G 0/1 weights, maker | holdout | sim.py | +47.99% | +15.32% | 0.59 | 37.0% |
| G 0/1 weights, maker | holdout | vectorbt | +48.01% | +15.32% | 0.59 | 37.0% |
| G traded weights, maker | discovery | sim.py | +984.89% | +104.43% | 2.09 | 27.2% |
| G traded weights, maker | discovery | vectorbt | +965.84% | +103.34% | 2.07 | 27.1% |
| G traded weights, maker | holdout | sim.py | +59.87% | +18.60% | 0.81 | 22.3% |
| G traded weights, maker | holdout | vectorbt | +59.16% | +18.41% | 0.80 | 22.4% |

Largest CAGR gap per case: buy & hold (no fee) 0.00%, G 0/1 weights, maker 0.01%, G traded weights, maker 1.08%

Reading: with whole slots (buy & hold, 0/1 weights) the two engines agree to rounding, so prices, bar alignment, trade timing and fees are the same. With G's partial weights (votes / 6) `sim.py` keeps the slot at a fixed fraction between trades (an implicit free rebalance each bar) while vectorbt holds a fixed coin amount that drifts with the price; that is the whole remaining gap, a fraction of a CAGR point on the holdout. Run: `pip install vectorbt` (optional), then `python scripts/backtest_report.py crosscheck`.
