"""Strategy G backtest report with charts (QuantStats tearsheets + Plotly validation charts).

The engine is :mod:`wallex.sim` (same model and numbers as ``docs/strategy-g-results.md``);
this script only exports its series and draws them. Extra packages (not needed to trade):
``pip install -r requirements-report.txt``.

    python scripts/backtest_report.py series      # data/report/series.parquet + check vs §1
    python scripts/backtest_report.py tearsheets  # docs/backtest/metrics.md + tearsheet-*.html
    python scripts/backtest_report.py charts      # docs/backtest/validation.html (Plotly)
    python scripts/backtest_report.py crosscheck  # docs/backtest/crosscheck.md (vectorbt)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from wallex import report_g as rg
from wallex import sim as v

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "report"
BPH = rg.BPH
BARS_PER_YEAR = v.HOURS_PER_YEAR * BPH
COLS = {  # series column → cost row (None = frictionless) or variant
    "G maker": "maker 0.10%",
    "G taker": "taker",
    "G frictionless": "frictionless",
    "G stress": "stress 0.20%",
}


def spans(index: pd.DatetimeIndex) -> dict[str, tuple[str, str]]:
    """Report spans: full, discovery, holdout, each calendar year, last 90 days."""
    out = dict(rg.SPANS)
    for y in sorted(set(index.year)):
        out[str(y)] = (max(f"{y}-01-01", v.START), f"{y}-12-31 23:30")
    end = pd.Timestamp(v.END)
    out["last 90 d"] = (f"{end - pd.Timedelta(days=90):%Y-%m-%d %H:%M}", v.END)
    return out


def build_series() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Portfolio net return per 30-min bar (equal slots, as ``sim.stats``) for each variant."""
    bars = rg.vs.load(30)
    closes = {c: b["close"] for c, b in bars.items()}
    books = {c: rg.g_book(closes[c]) for c in v.N5}
    sims = {col: v.simulate(bars, books, rg.RUN, v.COSTS[row]) for col, row in COLS.items()}
    crash_books = {c: np.where(rg.crash(closes[c]), 0.0, books[c]) for c in v.N5}
    sims["G maker + crash gate"] = v.simulate(bars, crash_books, rg.RUN, rg.MAKER)
    ones = {c: np.ones(len(b)) for c, b in bars.items()}
    sims["buy & hold"] = v.simulate(bars, ones, rg.RUN, None)
    btc = bars["BTC"]["close"]
    port = pd.DataFrame({k: s.mean(axis=1) for k, s in sims.items()})
    port["BTC buy & hold"] = btc.pct_change().fillna(0.0)
    return port.loc[v.START :], sims  # keep the last bar like sim.stats year spans


def metrics(r: pd.Series) -> dict[str, float]:
    """total, CAGR, Sharpe, MDD from a per-bar return series (same formulas as ``sim.stats``)."""
    eq = (1 + r).cumprod()
    years = len(r) / BARS_PER_YEAR
    sd = r.std()
    return {
        "total": eq.iloc[-1] - 1,
        "cagr": eq.iloc[-1] ** (1 / years) - 1,
        "sharpe": r.mean() / sd * np.sqrt(BARS_PER_YEAR) if sd > 0 else 0.0,
        "mdd": float((1 - eq / eq.cummax()).max()),
    }


def check(port: pd.DataFrame, sims: dict[str, pd.DataFrame]) -> int:
    """Series metrics must equal ``sim.stats`` (the numbers in strategy-g-results.md §1)."""
    bad = 0
    for sname, span in rg.SPANS.items():
        for col in [*COLS, "buy & hold"]:
            a = metrics(port[col].loc[span[0] : span[1]])
            b = v.stats(sims[col], span, BPH)
            diff = max(abs(a[k] - b[k]) for k in a)
            bad += diff > 1e-9
            print(f"{sname:28} {col:16} total {a['total']:+9.2%} CAGR {a['cagr']:+8.2%}"
                  f" Sharpe {a['sharpe']:5.2f} MDD {a['mdd']:5.1%}"
                  f"{'' if diff <= 1e-9 else f'  MISMATCH {diff:.2e}'}")  # fmt: skip
    return bad


def cmd_series(_args) -> int:
    port, sims = build_series()
    DATA.mkdir(parents=True, exist_ok=True)
    port.to_parquet(DATA / "series.parquet")
    print(f"wrote {DATA / 'series.parquet'} {port.shape}")
    bad = check(port, sims)
    print("CHECK", "OK" if not bad else f"FAILED ({bad})")
    return 1 if bad else 0


def load_series() -> pd.DataFrame:
    f = DATA / "series.parquet"
    if not f.exists():
        raise SystemExit(f"{f} missing: run `series` first")
    return pd.read_parquet(f)


def daily(r: pd.Series) -> pd.Series:
    return (1 + r).resample("1D").prod() - 1


TABLE_COLS = ["G maker", "G taker", "G maker + crash gate", "buy & hold", "BTC buy & hold"]
TEARSHEET_SPANS = ["full", "discovery", "holdout", "last 90 d"]


def metrics_table(port: pd.DataFrame) -> pd.DataFrame:
    """One row per (span, variant). total/CAGR/Sharpe/MDD on 30-min bars (= sim.stats);
    Sortino, volatility, best/worst/positive days on daily returns (QuantStats, 365 d/yr)."""
    import quantstats as qs

    rows = []
    for sname, (a, b) in spans(port.index).items():
        for col in TABLE_COLS:
            r = port[col].loc[a:b]
            m, d = metrics(r), daily(r)
            rows.append({
                "span": sname, "variant": col, **m,
                "calmar": m["cagr"] / m["mdd"] if m["mdd"] > 0 else np.nan,
                "years": len(r) / BARS_PER_YEAR,
                "sortino": float(qs.stats.sortino(d, periods=365)),
                "vol": float(qs.stats.volatility(d, periods=365)),
                "best_day": d.max(), "worst_day": d.min(), "pos_days": (d > 0).mean(),
                "flat_days": (d == 0).mean(),
            })  # fmt: skip
    return pd.DataFrame(rows)


def table_md(t: pd.DataFrame) -> list[str]:
    head = ["span", "variant", "total", "CAGR", "Sharpe", "Sortino", "Calmar", "MDD",
            "vol / yr", "best day", "worst day", "up days", "flat days"]  # fmt: skip
    out = ["| " + " | ".join(head) + " |", "| --- | --- |" + " ---: |" * (len(head) - 2)]
    for r in t.itertuples():
        short = r.years < 0.99  # annualised CAGR / Calmar mislead on short spans
        cagr, calmar = ("—", "—") if short else (f"{r.cagr:+.2%}", f"{r.calmar:.2f}")
        out.append(
            f"| {r.span} | {r.variant} | {r.total:+.2%} | {cagr} | {r.sharpe:.2f} |"
            f" {r.sortino:.2f} | {calmar} | {r.mdd:.1%} | {r.vol:.1%} | {r.best_day:+.2%} |"
            f" {r.worst_day:+.2%} | {r.pos_days:.0%} | {r.flat_days:.0%} |"
        )
    return out


def span_key(name: str) -> str:
    return next((k for k in TEARSHEET_SPANS if name.startswith(k)), name)


def cmd_tearsheets(args) -> int:
    import warnings

    import quantstats as qs

    warnings.filterwarnings("ignore")
    port = load_series()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t = metrics_table(port)
    t.to_csv(out / "metrics.csv", index=False)
    lines = [
        "# Strategy G: metrics by span",
        "",
        "Generated by `python scripts/backtest_report.py tearsheets`. Engine and costs as"
        " `strategy-g-results.md` (maker = 0.10 % per side, taker = 0.13 % + half spread)."
        " total / CAGR / Sharpe / MDD on 30-min bars (identical to that file); Sortino,"
        " volatility and day statistics on daily returns, 365 days per year; CAGR and Calmar"
        " left out for spans under a year. buy & hold ="
        " equal slots of the five coins, no cost. Flat days = days with no position.",
        "",
        *table_md(t),
        "",
        "Tearsheets (QuantStats, daily returns, benchmark = buy & hold of the five coins): "
        + ", ".join(f"[{s}](tearsheet-{s.replace(' ', '-')}.html)" for s in TEARSHEET_SPANS),
        "",
        "Validation charts (permutation test, discovery vs holdout, 72-h windows, parameter"
        " sensitivity): [validation.html](validation.html); independent engine check:"
        " [crosscheck.md](crosscheck.md)",
        "",
    ]
    (out / "metrics.md").write_text("\n".join(lines))
    for sname, (a, b) in spans(port.index).items():
        key = span_key(sname)
        if key not in TEARSHEET_SPANS:
            continue
        g = daily(port["G maker"].loc[a:b]).rename("Strategy G (maker)")
        bench = daily(port["buy & hold"].loc[a:b]).rename("Buy & hold (5 coins)")
        f = out / f"tearsheet-{key.replace(' ', '-')}.html"
        qs.reports.html(g, benchmark=bench, output=str(f), periods_per_year=365,
                        match_dates=False, title=f"Strategy G, {sname}")  # fmt: skip
        print("wrote", f)
    print("wrote", out / "metrics.md", out / "metrics.csv")
    return 0


# ---------------------------------------------------------------- validation charts

HOURS_FAN = 72  # contest length
PCTS = (5, 25, 50, 75, 95)
BANDS, DAYS, SCALES = (0.0, 0.0015, 0.003, 0.005), (14, 28, 56), (0.5, 0.75, 1.0, 1.5, 2.0)


def window_paths(slot_rets: pd.DataFrame, span, hours: int) -> np.ndarray:
    """Equal-slot equity path of every ``hours`` window (1-h step), as ``report_g.window_dist``."""
    r = slot_rets.loc[span[0] : span[1]].to_numpy()
    gr = np.vstack([np.ones(r.shape[1]), np.cumprod(1 + r, axis=0)])
    L = hours * BPH
    starts = np.arange(0, len(gr) - L, BPH)
    idx = starts[:, None] + np.arange(L + 1)[None, :]
    return sum(gr[idx, c] / gr[starts, c][:, None] for c in range(r.shape[1])) / r.shape[1]


def compute_validation() -> dict:
    """Everything the charts need that is slow to build (null, grids); cached as a pickle."""
    bars = rg.vs.load(30)
    closes = {c: b["close"] for c, b in bars.items()}
    books = {c: rg.g_book(closes[c]) for c in v.N5}
    maker = v.simulate(bars, books, rg.RUN, rg.MAKER)
    ones = {c: np.ones(len(b)) for c, b in bars.items()}
    bh = v.simulate(bars, ones, rg.RUN, None)
    print("null…", flush=True)
    null = v.circular_null(bars, books, rg.RUN, rg.MAKER, rg.HOLD, n=v.N_PERM, bph=BPH)
    real = v.stats(maker, rg.HOLD, BPH)["mean_h"]
    lo, hi, p_neg = rg.block_ci(maker, rg.HOLD)
    print("sensitivity…", flush=True)

    def grid(param, values, mk):
        cagr, mdd = np.empty((len(DAYS), len(values))), np.empty((len(DAYS), len(values)))
        for i, d in enumerate(DAYS):
            for j, x in enumerate(values):
                bk = {c: mk(closes[c], x, d) for c in v.N5}
                s = v.stats(v.simulate(bars, bk, rg.RUN, rg.MAKER), rg.HOLD, BPH)
                cagr[i, j], mdd[i, j] = s["cagr"], s["mdd"]
        return {"param": param, "values": list(values), "cagr": cagr, "mdd": mdd}

    sens = [
        grid("band", BANDS, lambda cl, b, d: rg.g_book(cl, 1.0, b, d)),
        grid("pairs ×", SCALES, lambda cl, sc, d: rg.g_book(cl, sc, rg.g.BAND, d)),
    ]
    fan = {}
    for name, sim in (("G maker", maker), ("buy & hold", bh)):
        path = window_paths(sim, rg.HOLD, HOURS_FAN)
        fan[name] = {"pct": np.percentile(path, PCTS, axis=0), "final": path[:, -1] - 1}
    gates = pd.DataFrame({c: rg._gate(closes[c], rg.g.GATE_DAYS) for c in v.N5}).loc[v.START :]
    coin = {}
    for c in v.N5:
        g1, b1 = maker[c].loc[rg.HOLD[0] : rg.HOLD[1]], bh[c].loc[rg.HOLD[0] : rg.HOLD[1]]
        coin[c] = ((1 + g1).prod() - 1, (1 + b1).prod() - 1)
    return {
        "null": null, "real": real, "p": v.p_value(real, null), "ci": (lo, hi, p_neg),
        "sens": sens, "fan": fan, "gates": gates.resample("1D").mean(), "coin": coin,
    }  # fmt: skip


# Colors are role tokens ("@s1" …) resolved in the page for light / dark (dataviz palette,
# first three categorical slots, validated all-pairs in both modes).
THEME = {
    "light": {
        "s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a",
        "s1a": "rgba(42,120,214,0.14)", "s1b": "rgba(42,120,214,0.30)",
        "s2a": "rgba(235,104,52,0.14)", "s2b": "rgba(235,104,52,0.30)",
        "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
        "axis": "#c3c2b7", "surface": "#fcfcfb", "page": "#f9f9f7", "null": "#b5b3ab",
        "shade": "rgba(11,11,11,0.04)", "neg": "#ee8f8e", "mid": "#f0efec", "pos": "#86b6ef",
        "seq0": "#eef4fc", "seq1": "#104281",
    },
    "dark": {
        "s1": "#3987e5", "s2": "#d95926", "s3": "#199e70",
        "s1a": "rgba(57,135,229,0.16)", "s1b": "rgba(57,135,229,0.34)",
        "s2a": "rgba(217,89,38,0.16)", "s2b": "rgba(217,89,38,0.34)",
        "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
        "axis": "#383835", "surface": "#1a1a19", "page": "#0d0d0d", "null": "#5d5c57",
        "shade": "rgba(255,255,255,0.05)", "neg": "#8a3434", "mid": "#383835", "pos": "#1c5cab",
        "seq0": "#1f2733", "seq1": "#86b6ef",
    },
}  # fmt: skip
NAMES = {"G maker": "Strategy G (maker)", "buy & hold": "Buy & hold, 5 coins",
         "BTC buy & hold": "BTC buy & hold"}  # fmt: skip


def base_layout(**kw) -> dict:
    ax = {"gridcolor": "@grid", "linecolor": "@axis", "zerolinecolor": "@axis",
          "tickfont": {"color": "@muted"}, "title": {"font": {"color": "@ink2"}}}  # fmt: skip
    lay = {
        "paper_bgcolor": "@surface", "plot_bgcolor": "@surface",
        "font": {"family": "system-ui, -apple-system, Segoe UI, sans-serif", "color": "@ink2",
                 "size": 13},
        "margin": {"l": 60, "r": 20, "t": 30, "b": 50}, "height": 380,
        "legend": {"orientation": "h", "y": 1.12, "x": 0, "font": {"color": "@ink2"}},
        "hoverlabel": {"bgcolor": "@surface", "font": {"color": "@ink"}, "bordercolor": "@axis"},
        "xaxis": dict(ax), "yaxis": dict(ax), "barcornerradius": 4,
    }  # fmt: skip
    for k, val in kw.items():
        if isinstance(val, dict) and isinstance(lay.get(k), dict):
            lay[k] = {**lay[k], **val}
        else:
            lay[k] = val
    return lay


def fig_equity(port: pd.DataFrame) -> list[dict]:
    eq = (1 + port).cumprod()
    d = eq.resample("1D").last()
    dd = (1 - eq / eq.cummax()).resample("1D").max()
    x = [f"{t:%Y-%m-%d}" for t in d.index]
    roles = {"G maker": "@s1", "buy & hold": "@s2", "BTC buy & hold": "@s3"}
    lines = [{"type": "scatter", "mode": "lines", "x": x, "y": d[k].round(4).tolist(),
              "name": NAMES[k], "line": {"color": c, "width": 2},
              "hovertemplate": "%{y:.2f}×"} for k, c in roles.items()]  # fmt: skip
    a, b = rg.HOLD[0], v.SPANS["discovery"][1][:10]
    shapes = [
        {
            "type": "rect",
            "xref": "x",
            "yref": "paper",
            "x0": a,
            "x1": x[-1],
            "y0": 0,
            "y1": 1,
            "fillcolor": "@shade",
            "line": {"width": 0},
            "layer": "below",
        }
    ]
    notes = [
        {
            "x": x0,
            "y": 1,
            "xref": "x",
            "yref": "paper",
            "text": txt,
            "xanchor": side,
            "showarrow": False,
            "font": {"color": "@muted"},
        }
        for x0, txt, side in ((b, "discovery ", "right"), (a, " holdout", "left"))
    ]
    f1 = {
        "data": lines,
        "layout": base_layout(
            yaxis={
                "type": "log",
                "title": {"text": "growth of 1 (log scale)"},
                "tickvals": [1, 2, 5, 10, 20, 50],
                "ticktext": ["1×", "2×", "5×", "10×", "20×", "50×"],
            },
            hovermode="x unified",
            shapes=shapes,
            annotations=notes,
            height=420,
        ),
    }
    under = [{"type": "scatter", "mode": "lines", "x": x, "y": (-dd[k]).round(4).tolist(),
              "name": NAMES[k], "line": {"color": c, "width": 1.5},
              "fill": "tozeroy", "fillcolor": c + "a", "hovertemplate": "%{y:.1%}"}
             for k, c in list(roles.items())[:2]]  # fmt: skip
    f2 = {
        "data": under,
        "layout": base_layout(
            yaxis={"tickformat": ".0%", "title": {"text": "drawdown from peak"}},
            hovermode="x unified",
            shapes=shapes,
            height=300,
        ),
    }
    return [f1, f2]


def bars2(x, g, b, fmt=".1%", names=NAMES, **lay) -> dict:
    data = [{"type": "bar", "x": x, "y": list(map(float, ys)), "name": names[k],
             "marker": {"color": c},
             "hovertemplate": f"%{{x}}: %{{y:{fmt}}}<extra>{names[k]}</extra>"}
            for k, c, ys in (("G maker", "@s1", g), ("buy & hold", "@s2", b))]  # fmt: skip
    return {
        "data": data,
        "layout": base_layout(barmode="group", bargap=0.3, bargroupgap=0.08, **lay),
    }


def fig_years(t: pd.DataFrame) -> dict:
    yrs = sorted(s for s in t["span"].unique() if s.isdigit())
    get = lambda col: [t[(t.span == y) & (t.variant == col)]["total"].iloc[0] for y in yrs]  # noqa: E731
    return bars2(yrs, get("G maker"), get("buy & hold"),
                 yaxis={"tickformat": ".0%", "title": {"text": "return in the year"}})  # fmt: skip


def fig_split(t: pd.DataFrame) -> dict:
    """Discovery vs holdout: CAGR, Sharpe, MDD as three small panels (one scale each)."""
    names = [k for k in rg.SPANS if not k.startswith("full")]
    data, lay = [], base_layout(barmode="group", bargap=0.3, bargroupgap=0.08, height=340)
    for i, (m, fmt, label) in enumerate((("cagr", ".0%", "CAGR"), ("sharpe", ".2f", "Sharpe"),
                                          ("mdd", ".0%", "max drawdown")), start=1):  # fmt: skip
        for k, c in (("G maker", "@s1"), ("buy & hold", "@s2")):
            ys = [float(t[(t.span == s) & (t.variant == k)][m].iloc[0]) for s in names]
            data.append({"type": "bar", "x": [s.split()[0] for s in names], "y": ys,
                         "name": NAMES[k], "marker": {"color": c}, "showlegend": i == 1,
                         "legendgroup": k, "xaxis": f"x{i}", "yaxis": f"y{i}",
                         "hovertemplate": f"%{{x}}: %{{y:{fmt}}}<extra>{NAMES[k]}</extra>",
                         })  # fmt: skip
        dom = [(i - 1) / 3 + 0.04, i / 3 - 0.04]
        lay[f"xaxis{i}"] = {**lay["xaxis"], "domain": dom, "anchor": f"y{i}"}
        lay[f"yaxis{i}"] = {
            **lay["yaxis"],
            "anchor": f"x{i}",
            "tickformat": fmt,
            "title": {"text": label, "font": {"color": "@ink2"}},
        }
    del lay["xaxis"], lay["yaxis"]
    lay["xaxis"], lay["yaxis"] = lay.pop("xaxis1"), lay.pop("yaxis1")
    for tr in data:
        if tr["xaxis"] == "x1":
            tr["xaxis"], tr["yaxis"] = "x", "y"
    return {"data": data, "layout": lay}


def fig_null(val: dict) -> dict:
    k = 36 * BPH
    null, real = val["null"] * k, val["real"] * k
    lo, hi, _ = val["ci"]
    counts, edges = np.histogram(null, bins=24)
    top = float(counts.max())
    data = [
        {"type": "bar", "x": ((edges[:-1] + edges[1:]) / 2).tolist(), "y": counts.tolist(),
         "width": float(edges[1] - edges[0]) * 0.9, "name": "random timing (150 shifts)",
         "marker": {"color": "@null"}, "hovertemplate": "%{x:.3%}: %{y} shifts<extra></extra>"},
        {"type": "scatter", "mode": "markers", "x": [real], "y": [top * 1.18],
         "name": "bootstrap 90 % CI of real", "marker": {"color": "@s1", "size": 10},
         "error_x": {"type": "data", "symmetric": False, "array": [hi - real],
                     "arrayminus": [real - lo], "color": "@s1", "thickness": 2, "width": 6},
         "hovertemplate": f"real %{{x:.3%}}<br>90 % CI [{lo:+.3%}, {hi:+.3%}]<extra></extra>"},
    ]  # fmt: skip
    shapes = [
        {"type": "line", "x0": real, "x1": real, "y0": 0, "y1": top * 1.05,
         "line": {"color": "@s1", "width": 2}},
        {"type": "line", "x0": 0, "x1": 0, "y0": 0, "y1": top * 1.3,
         "line": {"color": "@axis", "width": 1, "dash": "dot"}},
    ]  # fmt: skip
    label = f"real G: {real:+.3%} / 36 h, p = {val['p']:.3f}"
    notes = [{"x": real, "y": top * 1.05, "text": label,
              "showarrow": False, "yanchor": "bottom", "xanchor": "left",
              "font": {"color": "@ink"}}]  # fmt: skip
    return {
        "data": data,
        "layout": base_layout(
            xaxis={
                "tickformat": ".2%",
                "title": {"text": "mean net return per 36 h (holdout, maker)"},
            },
            yaxis={"title": {"text": "shifts"}},
            shapes=shapes,
            annotations=notes,
            bargap=0,
        ),
    }


def fig_sens(val: dict) -> list[dict]:
    out = []
    lim = max(float(np.abs(s["cagr"]).max()) for s in val["sens"])
    for s, chosen in zip(val["sens"], (rg.g.BAND, 1.0), strict=True):
        xs = [f"{x:.2%}" if s["param"] == "band" else f"×{x:g}" for x in s["values"]]
        ys = [f"gate {d} d" for d in DAYS]
        text = [[f"{c:+.1%}<br>MDD {m:.0%}" for c, m in zip(rc, rm, strict=True)]
                for rc, rm in zip(s["cagr"], s["mdd"], strict=True)]  # fmt: skip
        j, i = s["values"].index(chosen), DAYS.index(rg.g.GATE_DAYS)
        ring = {"type": "rect", "x0": j - 0.5, "x1": j + 0.5, "y0": i - 0.5, "y1": i + 0.5,
                "line": {"color": "@ink", "width": 2}}  # fmt: skip
        out.append({"data": [{
            "type": "heatmap", "z": s["cagr"].round(4).tolist(), "x": xs, "y": ys, "text": text,
            "texttemplate": "%{text}", "textfont": {"size": 12, "color": "@ink"}, "zmid": 0,
            "zmin": -lim, "zmax": lim, "colorscale": [[0, "@neg"], [0.5, "@mid"], [1, "@pos"]],
            "xgap": 2, "ygap": 2, "colorbar": {"tickformat": ".0%", "title": {"text": "CAGR"},
                                    "tickfont": {"color": "@muted"}},
            "hovertemplate": "%{y}, %{x}<br>holdout CAGR %{z:.1%}<extra></extra>",
        }], "layout": base_layout(
            xaxis={"type": "category", "title": {"text": "SMA band" if s["param"] == "band"
                                                 else "SMA pair-length scale"}},
            yaxis={"type": "category"}, shapes=[ring], height=300,
            margin={"l": 80, "r": 10, "t": 20, "b": 50})})  # fmt: skip
    return out


def fig_fan(val: dict) -> list[dict]:
    hrs = (np.arange(HOURS_FAN * BPH + 1) / BPH).tolist()
    data, lay = [], base_layout(hovermode="x unified", height=380)
    for i, (k, c) in enumerate((("G maker", "s1"), ("buy & hold", "s2")), start=1):
        p = val["fan"][k]["pct"] - 1
        ax = {"xaxis": "x" if i == 1 else "x2", "yaxis": "y" if i == 1 else "y2"}
        for lo, hi, fill, nm in ((0, 4, "a", "5–95 %"), (1, 3, "b", "25–75 %")):
            data.append(
                {
                    "type": "scatter",
                    "x": hrs,
                    "y": p[lo].round(5).tolist(),
                    "mode": "lines",
                    "line": {"width": 0},
                    "showlegend": False,
                    "hoverinfo": "skip",
                    **ax,
                }
            )
            data.append({"type": "scatter", "x": hrs, "y": p[hi].round(5).tolist(), "mode": "lines",
                         "line": {"width": 0}, "fill": "tonexty", "fillcolor": f"@{c}{fill}",
                         "name": f"{NAMES[k]} {nm}", "showlegend": False, "hoverinfo": "skip",
                         **ax})  # fmt: skip
        data.append({"type": "scatter", "x": hrs, "y": p[2].round(5).tolist(), "mode": "lines",
                     "line": {"color": f"@{c}", "width": 2}, "name": f"{NAMES[k]}, median",
                     "showlegend": False,
                     "customdata": np.stack([p[0], p[4]], axis=1).round(5).tolist(),
                     "hovertemplate": "median %{y:.2%} · 5–95 %: %{customdata[0]:.2%} to"
                                      " %{customdata[1]:.2%}", **ax})  # fmt: skip
    lo_, hi_ = (
        float(min(val["fan"][k]["pct"][0].min() for k in val["fan"])) - 1,
        float(max(val["fan"][k]["pct"][4].max() for k in val["fan"])) - 1,
    )
    pad = 0.05 * (hi_ - lo_)
    lim = [lo_ - pad, hi_ + pad]
    for i, dom in ((1, [0, 0.47]), (2, [0.53, 1])):
        sx, sy = ("", "") if i == 1 else ("2", "2")
        lay[f"xaxis{sx}"] = {
            **lay["xaxis"],
            "domain": dom,
            "anchor": f"y{sy}",
            "title": {"text": "hours into the window"},
            "dtick": 12,
        }
        lay[f"yaxis{sy}"] = {
            **lay["yaxis"],
            "anchor": f"x{sx}",
            "tickformat": ".0%",
            "range": lim,
            "matches": "y" if i == 2 else None,
        }
    lay["yaxis"]["title"] = {"text": "return since window start"}
    lay["annotations"] = [
        {
            "x": x,
            "y": 1.06,
            "xref": "paper",
            "yref": "paper",
            "text": NAMES[k],
            "showarrow": False,
            "font": {"color": "@ink2"},
        }
        for x, k in ((0.235, "G maker"), (0.765, "buy & hold"))
    ]
    f1 = {"data": data, "layout": lay}
    edges = np.linspace(-0.25, 0.25, 51)
    hist = []
    for k, c in (("G maker", "s1"), ("buy & hold", "s2")):
        fin = np.clip(val["fan"][k]["final"], edges[0], edges[-1])
        cnt, _ = np.histogram(fin, bins=edges)
        hist.append({"type": "scatter", "x": ((edges[:-1] + edges[1:]) / 2).tolist(),
                     "y": (cnt / cnt.sum()).round(5).tolist(), "mode": "lines",
                     "line": {"shape": "hvh", "color": f"@{c}", "width": 2}, "fill": "tozeroy",
                     "fillcolor": f"@{c}a", "name": NAMES[k],
                     "hovertemplate": "%{x:.0%}: %{y:.1%} of windows"})  # fmt: skip
    f2 = {"data": hist, "layout": base_layout(
        hovermode="x unified", height=320,
        xaxis={"tickformat": ".0%", "title": {"text": "return over 72 h (clipped at ±25 %)"}},
        yaxis={"tickformat": ".0%", "title": {"text": "share of windows"}},
        shapes=[{"type": "line", "x0": 0, "x1": 0, "y0": 0, "y1": 1, "yref": "paper",
                 "line": {"color": "@axis", "dash": "dot", "width": 1}}])}  # fmt: skip
    return [f1, f2]


def fig_gates(val: dict) -> dict:
    gt = val["gates"]
    return {"data": [{
        "type": "heatmap", "z": gt.T.round(3).values.tolist(), "y": list(gt.columns),
        "x": [f"{t:%Y-%m-%d}" for t in gt.index], "zmin": 0, "zmax": 1, "ygap": 2,
        "colorscale": [[0, "@seq0"], [1, "@seq1"]], "showscale": False,
        "hovertemplate": "%{y} %{x}: gate %{z:.0%} on<extra></extra>",
    }], "layout": base_layout(height=260, yaxis={"autorange": "reversed"})}  # fmt: skip


def fig_coins(val: dict) -> dict:
    cs = list(val["coin"])
    names = {**NAMES, "buy & hold": "Buy & hold (the coin)"}
    return bars2(
        cs,
        [val["coin"][c][0] for c in cs],
        [val["coin"][c][1] for c in cs],
        names=names,
        yaxis={"tickformat": ".0%", "title": {"text": "holdout return, one slot alone"}},
    )


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Strategy G Validation</title>
<style>
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;
--muted:#898781;--border:rgba(11,11,11,.10)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;
--page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--border:rgba(255,255,255,.10)}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#fff;
--ink2:#c3c2b7;--border:rgba(255,255,255,.10)}
body{margin:0;background:var(--page);color:var(--ink);
font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:18px;margin:0 0 4px}
p{color:var(--ink2);margin:0 0 12px}.sub{color:var(--muted);margin-bottom:24px}
section{background:var(--surface);border:1px solid var(--border);border-radius:12px;
padding:18px 18px 8px;margin:0 0 20px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;
margin:0 0 20px}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:14px}
.tile b{display:block;font-size:24px;color:var(--ink)}.tile span{color:var(--muted);font-size:13px}
.row{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media (max-width:760px){.row{grid-template-columns:1fr}}
a{color:inherit}
</style></head><body><main>
<h1>Strategy G: validation charts</h1>
<p class="sub">__SUB__</p>
<div class="tiles">__TILES__</div>
__SECTIONS__
<p class="sub">Numbers as tables: <a href="metrics.md">metrics.md</a> ·
tearsheets: __TEARS__</p>
</main>
<script>__PLOTLY__</script>
<script>
const THEME = __THEME__, FIGS = __FIGS__;
function mode(){const t=document.documentElement.dataset.theme;
 if(t==="dark"||t==="light")return t;
 return matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light";}
function paint(o,T){if(typeof o==="string")return o[0]==="@"&&T[o.slice(1)]?T[o.slice(1)]:o;
 if(Array.isArray(o))return o.map(x=>paint(x,T));
 if(o&&typeof o==="object"){const r={};for(const k in o)r[k]=paint(o[k],T);return r}return o;}
function draw(){const T=THEME[mode()];
 for(const id in FIGS){const f=paint(FIGS[id],T);
  Plotly.react(id,f.data,f.layout,{responsive:true,displayModeBar:false});}}
draw();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change",draw);
new MutationObserver(draw).observe(document.documentElement,{attributes:true,
 attributeFilter:["data-theme"]});
</script></body></html>
"""


def cmd_charts(args) -> int:
    import json
    import pickle

    import plotly
    from plotly.offline import get_plotlyjs

    cache = DATA / "validation.pkl"
    if args.recompute or not cache.exists():
        val = compute_validation()
        cache.write_bytes(pickle.dumps(val))
    else:
        val = pickle.loads(cache.read_bytes())
    k = 36 * BPH
    print(f"real {val['real'] * k:+.3%} / 36 h, null mean {val['null'].mean() * k:+.3%},"
          f" p = {val['p']:.3f}, CI [{val['ci'][0]:+.3%}, {val['ci'][1]:+.3%}]")  # fmt: skip
    port = load_series()
    t = metrics_table(port)
    figs = {"equity": fig_equity(port)[0], "under": fig_equity(port)[1], "split": fig_split(t),
            "years": fig_years(t), "null": fig_null(val), "coins": fig_coins(val),
            "gates": fig_gates(val)}  # fmt: skip
    figs["fan"], figs["hist"] = fig_fan(val)
    figs["sens1"], figs["sens2"] = fig_sens(val)
    row = lambda span, col: t[(t.span == span) & (t.variant == col)].iloc[0]  # noqa: E731
    hold = row(next(s for s in rg.SPANS if s.startswith("holdout")), "G maker")
    hold_bh = row(next(s for s in rg.SPANS if s.startswith("holdout")), "buy & hold")
    s72 = val["fan"]["G maker"]["final"]
    tiles = [
        (f"{hold.cagr:+.1%}", "holdout CAGR, maker fees (2024-01 → 2026-09)"),
        (f"{hold.mdd:.1%}", f"holdout max drawdown (buy & hold {hold_bh.mdd:.1%})"),
        (f"p = {val['p']:.3f}", "timing vs 150 random shifts of the same positions"),
        (
            f"{np.percentile(s72, 5):+.1%}",
            "worst 5 % of 72-h windows, holdout (buy & hold"
            f" {np.percentile(val['fan']['buy & hold']['final'], 5):+.1%})",
        ),
    ]
    div = lambda i: f'<div id="{i}"></div>'  # noqa: E731
    sec = lambda h, p, body: f"<section><h2>{h}</h2><p>{p}</p>{body}</section>"  # noqa: E731
    sections = [
        sec(
            "Equity, discovery and holdout",
            "Growth of 1 at maker fees vs holding the same five coins (and BTC alone). The shaded"
            " part is the holdout: parameters were fixed on the discovery span before it was run.",
            div("equity") + div("under"),
        ),
        sec(
            "Discovery vs holdout",
            "The same rules on both spans. G gives up part of the upside to cut the drawdown.",
            div("split"),
        ),
        sec(
            "Is the timing luck? Permutation test",
            "Each coin's position series is shifted in time by a random 30+ days, 150 times: same"
            " exposure, same number of trades, timing broken. Gray: what random timing earns."
            " Blue: the real strategy, with its moving-block bootstrap 90 % interval.",
            div("null"),
        ),
        sec("Per year", "Calendar-year return, G at maker fees vs buy & hold.", div("years")),
        sec(
            "Contest-length windows (72 h, holdout)",
            "Every 72-h window starting on the hour, each starting with equal slots as the live"
            " run does. Bands: 5–95 % and 25–75 % of windows; line: median.",
            div("fan") + div("hist"),
        ),
        sec(
            "Parameter sensitivity (holdout CAGR, maker)",
            "Neighbouring parameter values, reported only (not used to choose). The outlined cell"
            " is the traded setting.",
            '<div class="row">' + div("sens1") + div("sens2") + "</div>",
        ),
        sec(
            "28-day trend gate per coin",
            "Strong blue = gate on (the coin may be held); faint = gate off, when G stays out"
            " of a falling market.",
            div("gates"),
        ),
        sec("Per coin (holdout)", "Each coin's slot alone, G vs holding it.", div("coins")),
    ]
    enc = lambda o: json.dumps(o, cls=plotly.utils.PlotlyJSONEncoder)  # noqa: E731
    html = (PAGE.replace("__SUB__", "Backtest of the traded strategy on Binance 30-min bars,"
                         " 2020-09 → 2026-09, five coins in equal slots, fees per side included."
                         " Generated by <code>scripts/backtest_report.py charts</code>.")
            .replace("__TILES__", "".join(f'<div class="tile"><b>{a}</b><span>{b}</span></div>'
                                          for a, b in tiles))
            .replace("__SECTIONS__", "\n".join(sections))
            .replace("__TEARS__", " · ".join(f'<a href="tearsheet-{s.replace(" ", "-")}.html">{s}'
                                             "</a>" for s in TEARSHEET_SPANS))
            .replace("__THEME__", enc(THEME)).replace("__FIGS__", enc(figs))
            .replace("__PLOTLY__", get_plotlyjs()))  # fmt: skip
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation.html").write_text(html)
    print("wrote", out / "validation.html", f"{len(html) / 1e6:.1f} MB")
    return 0


def vbt_slots(bars, books, sims, fee: float) -> pd.DataFrame:
    """Same trades as ``sim.simulate`` replayed in vectorbt, one cash-separate column per coin.

    An order is sent at the close of bar ``t-1`` exactly where ``sim`` flags a trade on bar
    ``t``, sized to the same target weight (``targetpercent``); between trades vectorbt holds
    a fixed coin amount (``sim`` keeps a fixed fraction of the slot)."""
    import vectorbt as vbt

    close = pd.DataFrame({c: b["close"] for c, b in bars.items()})
    traded = sims.attrs["trades"]
    size = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    for c in close:
        w = np.nan_to_num(books[c], nan=0.0)
        idx = np.flatnonzero(traded[c].to_numpy()) - 1
        size.iloc[idx, size.columns.get_loc(c)] = w[idx]
    pf = vbt.Portfolio.from_orders(
        close,
        size,
        size_type="targetpercent",
        price=close,
        fees=fee,
        init_cash=1.0,
        cash_sharing=False,
        freq="30min",
    )
    return pf.returns()


def cmd_crosscheck(args) -> int:
    """vectorbt vs sim.py on buy & hold, a binary G (0/1 weights) and the traded G."""
    bars = rg.vs.load(30)
    closes = {c: b["close"] for c, b in bars.items()}
    books = {c: rg.g_book(closes[c]) for c in v.N5}
    cases = {
        "buy & hold (no fee)": ({c: np.ones(len(b)) for c, b in bars.items()}, 0.0),
        "G 0/1 weights, maker": (
            {c: (np.nan_to_num(w) > 0).astype(float) for c, w in books.items()},
            rg.MAKER,
        ),
        "G traded weights, maker": (books, rg.MAKER),
    }
    lines = [
        "# Cross-check: sim.py vs vectorbt",
        "",
        "Same bars, same target weights, same trade bars, maker fee 0.10 % per side; one slot"
        " per coin, equal slots (portfolio = mean of slot returns, as `sim.stats`).",
        "",
        "| case | span | engine | total | CAGR | Sharpe | MDD |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    worst = {}
    for name, (bk, fee) in cases.items():
        sims = v.simulate(bars, bk, rg.RUN, fee)
        both = {
            "sim.py": sims.mean(axis=1),
            "vectorbt": vbt_slots(bars, bk, sims, fee).mean(axis=1),
        }
        for span in ("discovery", "holdout"):
            a, b = v.SPANS[span]
            m = {e: metrics(r.loc[a:b]) for e, r in both.items()}
            worst[name] = max(worst.get(name, 0), abs(m["sim.py"]["cagr"] - m["vectorbt"]["cagr"]))
            for e, s in m.items():
                lines.append(
                    f"| {name} | {span} | {e} | {s['total']:+.2%} | {s['cagr']:+.2%} |"
                    f" {s['sharpe']:.2f} | {s['mdd']:.1%} |"
                )
    lines += [
        "",
        "Largest CAGR gap per case: " + ", ".join(f"{k} {g:.2%}" for k, g in worst.items()),
        "",
    ]
    lines += [
        "Reading: with whole slots (buy & hold, 0/1 weights) the two engines agree to rounding,"
        " so prices, bar alignment, trade timing and fees are the same. With G's partial"
        " weights (votes / 6) `sim.py` keeps the slot at a fixed fraction between trades"
        " (an implicit free rebalance each bar) while vectorbt holds a fixed coin amount"
        " that drifts with the price; that is the whole remaining gap, a fraction of a"
        " CAGR point on the holdout. Run: `pip install vectorbt` (optional), then"
        " `python scripts/backtest_report.py crosscheck`.",
        "",
    ]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "crosscheck.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("series", help="export portfolio series and check them against sim.stats")
    p = sub.add_parser("tearsheets", help="metrics table + QuantStats tearsheets per span")
    p.add_argument("--out", default=str(ROOT / "docs" / "backtest"))
    p = sub.add_parser("charts", help="Plotly validation page (permutation, windows, grids)")
    p.add_argument("--out", default=str(ROOT / "docs" / "backtest"))
    p.add_argument("--recompute", action="store_true", help="ignore data/report/validation.pkl")
    p = sub.add_parser("crosscheck", help="replay the same trades in vectorbt and compare")
    p.add_argument("--out", default=str(ROOT / "docs" / "backtest"))
    args = ap.parse_args(argv)
    cmds = {
        "series": cmd_series,
        "tearsheets": cmd_tearsheets,
        "charts": cmd_charts,
        "crosscheck": cmd_crosscheck,
    }
    return cmds[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
