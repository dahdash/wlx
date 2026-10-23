"""Strategy G backtest report.

``python -m wallex.report_g`` writes ``docs/strategy-g-results.md``.

Conventions are those of :mod:`wallex.sim` / :mod:`wallex.validate_sma`: Binance 30-min bars
of the five coins, one equal slot per coin, signal at the bar close traded at that close, cost
per side as a fraction of the traded slot fraction, trade only when the weight change is
≥ minNotional / slot, holdout 2024-01 → 2026-09. Nothing here is used to choose parameters:
G was fixed in ``validate_sma``; sensitivity grids are reported only.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from wallex import sim as v
from wallex import validate_sma as vs
from wallex.strategy import g

ROOT = v.ROOT
BPH = 2  # 30-min bars per hour
DAY = 48
SPANS = {
    "full 2020-09→2026-09": (v.START, v.END),
    **{f"{k} {a[:7]}→{b[:7]}": (a, b) for k, (a, b) in v.SPANS.items()},
}
HOLD = v.SPANS["holdout"]
MAKER = v.COSTS["maker 0.10%"]
RUN = v.Run(hysteresis=0.0, stop=None)


# ---------------------------------------------------------------- books


def votes_on(closes: pd.Series, scale: float = 1.0, band: float = g.BAND) -> pd.Series:
    """Number of SMA pairs voting up (0…6)."""
    total = 0
    for f, s in g.PAIRS:
        pair = [(max(1, round(f * scale)), max(2, round(s * scale)))]
        total = total + g.votes(closes, pair, band).iloc[:, 0]
    return total


def _gate(closes: pd.Series, days: int) -> pd.Series:
    """28-day return > 0 at the 00:00 UTC close (bar labelled 23:30), held for the day."""
    up = (closes / closes.shift(days * DAY) > 1).astype(float)
    daily = up.where((up.index.hour == 23) & (up.index.minute == 30)).ffill().fillna(0.0)
    return daily


def g_book(closes, scale=1.0, band=g.BAND, days=g.GATE_DAYS, gate=True) -> np.ndarray:
    w = votes_on(closes, scale, band) / len(g.PAIRS)
    if gate:
        w = w * _gate(closes, days)
    return w.to_numpy()


def crash(closes: pd.Series, k=1.5, n=48, vol=144) -> np.ndarray:
    """Live crash gate on 30-min bars: 48-bar log return < −k·σ (σ of 144 bars × √48)."""
    lr = np.log(closes).diff()
    sigma = lr.rolling(vol).std() * np.sqrt(n)
    return (lr.rolling(n).sum() < -k * sigma).to_numpy()


# ---------------------------------------------------------------- metrics


def window_dist(slot_rets: pd.DataFrame, span, hours: int, kill: float | None = None) -> dict:
    """Rolling windows of ``hours`` (1-h step), equal slots restart each window (live run).
    ``kill``: flatten the rest of the window once the window path is ≥ kill below its peak."""
    r = slot_rets.loc[span[0] : span[1]].to_numpy()
    gr = np.vstack([np.ones(r.shape[1]), np.cumprod(1 + r, axis=0)])
    L = hours * BPH
    starts = np.arange(0, len(gr) - L, BPH)
    idx = starts[:, None] + np.arange(L + 1)[None, :]
    path = np.zeros(idx.shape)
    for c in range(r.shape[1]):
        path += gr[idx, c] / gr[starts, c][:, None]
    path /= r.shape[1]
    peak = np.maximum.accumulate(path, axis=1)
    dd = 1 - path / peak
    if kill is not None:
        hit = dd >= kill
        first = np.where(hit.any(axis=1), hit.argmax(axis=1), L)
        final = path[np.arange(len(path)), first]
        mdd = np.array([dd[i, : first[i] + 1].max() for i in range(len(dd))])
    else:
        final, mdd = path[:, -1], dd.max(axis=1)
    ret = final - 1
    return {
        "mean": ret.mean(), "median": np.median(ret), "win": (ret > 0).mean(),
        "p10": np.quantile(ret, 0.10), "p05": np.quantile(ret, 0.05), "worst": ret.min(),
        "best": ret.max(), "mdd_mean": mdd.mean(), "mdd_p95": np.quantile(mdd, 0.95),
        "p_mdd5": (mdd > 0.05).mean(),
    }  # fmt: skip


def block_ci(slot_rets: pd.DataFrame, span, block_days=7, n=2000, seed=11) -> tuple:
    """90 % moving-block bootstrap CI of the mean 36-h return (from mean bar return)."""
    p = slot_rets.loc[span[0] : span[1]].mean(axis=1).to_numpy()
    L = block_days * DAY
    rng = np.random.default_rng(seed)
    nb = len(p) // L
    means = np.empty(n)
    for i in range(n):
        s = rng.integers(0, len(p) - L, nb)
        means[i] = p[(s[:, None] + np.arange(L)).ravel()].mean()
    k = 36 * BPH
    return np.quantile(means, 0.05) * k, np.quantile(means, 0.95) * k, (means <= 0).mean()


def pct(x):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.2%}"


def row(cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def table(head, rows) -> list[str]:
    return [row(head), row(["---"] + ["---:"] * (len(head) - 1)), *[row(r) for r in rows], ""]


def core(s) -> list:
    return [pct(s["total"]), pct(s["cagr"]), f"{s['sharpe']:.2f}", f"{s['mdd']:.1%}",
            pct(s["mean36"]), f"{s['win36']:.0%}", pct(s["p10_36"]), f"{s['trades36']:.1f}",
            f"{s['exposure']:.0%}"]  # fmt: skip


CORE_HEAD = ["total", "CAGR", "Sharpe", "MDD", "mean 36 h", "win 36 h", "p10 36 h",
             "trades / 36 h", "exposure"]  # fmt: skip


# ---------------------------------------------------------------- report


def build(quick: bool = False) -> list[str]:
    bars = vs.load(30)
    closes = {c: b["close"] for c, b in bars.items()}
    books = {c: g_book(closes[c]) for c in v.N5}
    ones = {c: np.ones(len(b)) for c, b in bars.items()}
    bh = v.simulate(bars, ones, RUN, None)
    sims = {k: v.simulate(bars, books, RUN, row_) for k, row_ in v.COSTS.items()}
    out = [
        "# Strategy G: backtest tables",
        "",
        f"Generated by `python -m wallex.report_g` on {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M}"
        " UTC."
        " Binance 30-min bars, N5 = BTC ETH XRP SOL BNB vs USDT, one equal slot (20 %) per"
        " coin, decision at the bar close, fill at that close, trade only when |Δweight| ≥"
        f" {1 / 3.7:.2f} slot (minNotional / slot) or on a full exit. Costs per side: frictionless,"
        " maker 0.10 %, taker 0.13 % + half spread (BTC 0.06 %, ETH 0.075 %, others 0.15 %),"
        " stress 0.20 %. Holdout = 2024-01-01 → 2026-09-30.",
        "",
        "## 1. Main result",
        "",
    ]
    rows = []
    for sname, span in SPANS.items():
        for cname in v.COSTS:
            rows.append([sname, cname, *core(v.stats(sims[cname], span, BPH))])
        rows.append([sname, "buy & hold, no cost", *core(v.stats(bh, span, BPH))])
    out += table(["span", "cost / side", *CORE_HEAD], rows)

    print("null…", flush=True)
    null = v.circular_null(bars, books, RUN, MAKER, HOLD, n=40 if quick else v.N_PERM, bph=BPH)
    real = v.stats(sims["maker 0.10%"], HOLD, BPH)["mean_h"]
    p = v.p_value(real, null)
    lo, hi, p_neg = block_ci(sims["maker 0.10%"], HOLD)
    k = 36 * BPH
    out += [
        "## 2. Is it luck? (holdout, maker)",
        "",
        f"- Circular-shift null ({len(null)} shifts ≥ 30 days of each coin's weight book, keeps"
        f" exposure and trade count): real {real * k:+.3%} / 36 h, null mean"
        f" {null.mean() * k:+.3%}, null p95 {np.quantile(null, 0.95) * k:+.3%} → one-sided"
        f" **p = {p:.3f}**.",
        f"- Moving-block bootstrap (7-day blocks, 2000 draws) of the mean 36-h return: 90 % CI"
        f" [{lo:+.3%}, {hi:+.3%}]; share of draws ≤ 0: {p_neg:.1%}.",
        "",
    ]

    # per year / per coin
    maker = sims["maker 0.10%"]
    rows = []
    for y in sorted(set(maker.index.year)):
        span = (max(f"{y}-01-01", v.START), f"{y}-12-31 23:30")
        s, b = v.stats(maker, span, BPH), v.stats(bh, span, BPH)
        rows.append([y, pct(s["total"]), f"{s['mdd']:.1%}", pct(b["total"]), f"{b['mdd']:.1%}",
                     f"{s['trades36']:.1f}", f"{s['exposure']:.0%}"])  # fmt: skip
    out += ["## 3. Per year (maker)", ""]
    out += table(["year", "G", "G MDD", "buy & hold", "B&H MDD", "trades / 36 h", "exposure"], rows)
    rows = []
    for c in v.N5:
        one_g, one_b = maker[[c]].copy(), bh[[c]].copy()
        one_g.attrs["trades"] = maker.attrs["trades"][[c]]
        one_b.attrs["trades"] = bh.attrs["trades"][[c]]
        s, b = v.stats(one_g, HOLD, BPH), v.stats(one_b, HOLD, BPH)
        rows.append([c, pct(s["total"]), f"{s['mdd']:.1%}", f"{s['sharpe']:.2f}",
                     pct(b["total"]), f"{b['mdd']:.1%}", f"{s['trades36'] * 5:.1f}",
                     f"{s['exposure']:.0%}"])  # fmt: skip
    out += ["## 4. Per coin, one slot alone (holdout, maker)", ""]
    out += table(["coin", "G total", "G MDD", "G Sharpe", "B&H total", "B&H MDD",
                  "trades / 36 h (×5 slots)", "exposure"], rows)  # fmt: skip

    # horizons + live risk overlay
    crash_books = {c: np.where(crash(closes[c]), 0.0, books[c]) for c in v.N5}
    crash_sim = v.simulate(bars, crash_books, RUN, MAKER)
    variants = [
        ("G", maker, None), ("G + crash gate", crash_sim, None),
        ("G + crash gate + 10 % kill", crash_sim, 0.10), ("buy & hold", bh, None),
    ]  # fmt: skip
    rows = []
    for hours in (24, 36, 72):
        for name, sim, kill in variants:
            d = window_dist(sim, HOLD, hours, kill)
            rows.append([f"{hours} h", name, pct(d["mean"]), pct(d["median"]), f"{d['win']:.0%}",
                         pct(d["p05"]), pct(d["worst"]), pct(d["best"]), f"{d['mdd_mean']:.2%}",
                         f"{d['mdd_p95']:.2%}", f"{d['p_mdd5']:.1%}"])  # fmt: skip
    out += [
        "## 5. Contest-length windows (holdout, maker, every start hour)",
        "",
        "Each window starts flat-equal like the live run. Crash gate = live rule on 30-min bars"
        " (48-bar log return < −1.5 σ → weight 0). Kill = flatten the rest of the window at"
        " 10 % below the window peak (flatten cost not charged). MDD = drawdown inside the window.",
        "",
    ]
    out += table(["window", "variant", "mean", "median", "win", "p5", "worst", "best", "mean MDD",
                  "p95 MDD", "P(MDD > 5 %)"], rows)  # fmt: skip
    s = v.stats(crash_sim, HOLD, BPH)
    out += [f"G + crash gate, full holdout (maker): {' | '.join(core(s))}"
            f" ({' | '.join(CORE_HEAD)}).", ""]  # fmt: skip

    # sensitivity
    print("sensitivity…", flush=True)
    out += [
        "## 6. Parameter sensitivity (holdout, maker CAGR / MDD; reported, not selected from)",
        "",
    ]
    bands, days, scales = (0.0, 0.0015, 0.003, 0.005), (14, 28, 56), (0.5, 0.75, 1.0, 1.5, 2.0)
    if quick:
        bands, scales = (0.0, 0.003), (0.5, 1.0, 2.0)
    cells, rows = [], []
    for d in days:
        r_ = [f"gate {d} d"]
        for b_ in bands:
            s = v.stats(v.simulate(bars, {c: g_book(closes[c], 1.0, b_, d) for c in v.N5}, RUN,
                                   MAKER), HOLD, BPH)  # fmt: skip
            cells.append(s["cagr"])
            r_.append(f"{pct(s['cagr'])} / {s['mdd']:.0%}")
        rows.append(r_)
    out += ["Band × gate length (pairs as specified):", ""]
    out += table(["", *[f"band {b_:.2%}" for b_ in bands]], rows)
    rows = []
    for d in days:
        r_ = [f"gate {d} d"]
        for sc in scales:
            s = v.stats(v.simulate(bars, {c: g_book(closes[c], sc, g.BAND, d) for c in v.N5}, RUN,
                                   MAKER), HOLD, BPH)  # fmt: skip
            cells.append(s["cagr"])
            r_.append(f"{pct(s['cagr'])} / {s['mdd']:.0%}")
        rows.append(r_)
    out += ["Pair-length scale × gate length (band 0.30 %):", ""]
    out += table(["", *[f"pairs ×{sc}" for sc in scales]], rows)
    s_ng = v.stats(v.simulate(bars, {c: g_book(closes[c], gate=False) for c in v.N5}, RUN, MAKER),
                   HOLD, BPH)  # fmt: skip
    out += [
        f"{sum(c > 0 for c in cells)} of {len(cells)} cells have a positive holdout CAGR."
        f" Without the 28-day gate: CAGR {pct(s_ng['cagr'])}, MDD {s_ng['mdd']:.1%}.",
        "",
    ]

    # gate states + conditional forward returns
    hold_ix = maker.loc[HOLD[0] : HOLD[1]].index
    gates = pd.DataFrame({c: _gate(closes[c], g.GATE_DAYS) for c in v.N5}).loc[hold_ix]
    n_on = gates.sum(axis=1)
    all_off = (n_on == 0).astype(int)
    runs = all_off.groupby((all_off != all_off.shift()).cumsum()).agg(["first", "size"])
    off_len = runs[runs["first"] == 1]["size"] / DAY
    rows = []
    for c in v.N5:
        cl = closes[c]
        fwd = (cl.shift(-36 * BPH) / cl - 1).loc[hold_ix]
        gt = gates[c] > 0
        vt = votes_on(cl).loc[hold_ix]
        rows.append([c, pct(fwd.mean()), pct(fwd[gt & (vt >= 5)].mean()),
                     pct(fwd[~gt & (vt >= 5)].mean()),
                     pct(fwd[~gt & (vt < 5)].mean())])  # fmt: skip
    out += [
        "## 7. Gate states (holdout)",
        "",
        f"- All five gates off: {all_off.mean():.0%} of the time; {len(off_len)} spells, median"
        f" {off_len.median():.1f} days, longest {off_len.max():.1f} days.",
        "- Gates on (count of 5), share of time: "
        + ", ".join(f"{i}: {(n_on == i).mean():.0%}" for i in range(6)),
        "",
        "Mean forward 36-h return of the coin (no cost) by state at the bar close.",
        "",
    ]
    out += table(["coin", "all bars", "gate on, ≥ 5 votes", "gate off, ≥ 5 votes",
                  "gate off, < 5 votes"], rows)  # fmt: skip
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "docs" / "strategy-g-results.md"))
    ap.add_argument("--quick", action="store_true", help="fewer null shifts / grid cells")
    a = ap.parse_args(argv)
    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(build(a.quick)) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
