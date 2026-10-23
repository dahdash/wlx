# Decision log

Runs: G live-1; oil agent oil-probe, oil-live-1, oil-s2. 389 rows: 382 decisions, 7 orders. Every row of every cycle is in `decision-log.csv` (one row per coin per 30-min cycle, plus one row per order).

## Orders

| time (UTC) | engine | symbol | side | filled | price | fee | status | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| 2026-10-22T08:00:00 | convert | USDTTMN | BUY | 0 |  |  | probe_rejected | contest start/end: move the account between TMN and USDT |
| 2026-10-22T08:00:03 | convert | USDTTMN | BUY | 0 |  |  | probe_rejected | contest start/end: move the account between TMN and USDT |
| 2026-10-22T08:00:06 | convert | USDTTMN | BUY | 0.21 | 267049 | 0.00063 | FILLED | contest start/end: move the account between TMN and USDT |
| 2026-10-22T08:00:06 | convert | USDTTMN | BUY | 0 | 267851 |  | bulk_rejected | contest start/end: move the account between TMN and USDT |
| 2026-10-22T08:01:19 | convert | USDTTMN | BUY | 18.36 | 268149 | 0.05508 | FILLED | contest start/end: move the account between TMN and USDT |
| 2026-10-22T14:43:10 | oil-agent | USOONUSDT | BUY | 0.0091 | 163.26 | 0.0019131476 | filled | live micro round trip (buy, then sell) to check the oil order path before going live |
| 2026-10-22T14:43:10 | oil-agent | USOONUSDT | SELL | 0.009 | 160.21 | 0.001878201 | filled | live micro round trip (buy, then sell) to check the oil order path before going live |

## Decisions that changed a position (or asked to)

| time (UTC) | engine | symbol | action | target | signal | reason |
| --- | --- | --- | --- | ---: | --- | --- |
| 2026-10-22T13:50:19 | oil-agent | USOONUSDT | WATCH |  | 1-h move trigger | The oil rally is real, but it is driven by headlines and is losing steam at about $100. On Oct 22 WTI broke above $100 (CNBC says +2.8% to $99.88 at 9:15 ET) on reports of renewed US strikes on Iran and Houthi attacks on Saudi targets. This came after the Oct 21 EIA report showed a bigger-than-expec |
| 2026-10-22T19:10:05 | oil-agent | USOONUSDT | WATCH |  | USOON-WTI divergence trigger | This trigger looks like a real gap on paper, but I can't confirm USOON is cheap. Over 4h the WTI perp is up 0.92% and USOON is down 1.15%, a gap of -2.08%. That is past the 1.5% limit. Over 24h the gap is -1.37%, about 1.6× the 0.86% round-trip cost. Three things stop me buying. (1) USOON's 15m and  |
| 2026-10-23T17:35:05 | oil-agent | USOONUSDT | WATCH |  | USOON-WTI divergence trigger | Watch, no entry. Following L41, I compared USOON with live USO rather than the CL perp. USO was $168.85 (+3.90%) at 17:34:59 UTC and the USOON ask is 167.99, so USOON sits about 0.51% below USO. The round-trip cost is 0.385%, so a full catch-up would leave only about 0.13% net, barely above the 0.10 |

