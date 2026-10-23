"""Static HTML dashboard of one oil run (T14.8).

``python -m wallex.oil.dashboard --run-id oil-s1 [--out path.html]`` writes a self-contained
page (no network, no JS libraries): USOON bid and WTI perp indexed to the window start with
the run's decision markers, headline cards, agent vs Rule scores (:func:`evaluate`), one card
per decision (thesis, evidence for and against, gate verdict, forward outcome), trades, the
shared lessons and the architecture diagram."""

from __future__ import annotations

import argparse
import csv
import json
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

from wallex.oil import config
from wallex.oil.config import HORIZONS
from wallex.oil.evaluate import _naive, price_path, score
from wallex.oil.ops import read_jsonl
from wallex.oil.report import report

MARK = {"BUY_CANDIDATE": ("B", "buy"), "WATCH": ("W", "watch"),
        "NO_TRADE": ("N", "none"), "EXIT_REVIEW": ("X", "exit")}  # fmt: skip
W, H, PAD = 960, 320, (16, 56, 34, 16)  # chart size; padding top, right, bottom, left


def e(x: Any) -> str:
    return escape("" if x is None else str(x))


def pct(x: float | None, d: int = 2) -> str:
    return "–" if x is None else f"{x * 100:+.{d}f}%"


def market(state_dir: Path) -> list[dict[str, Any]]:
    """Every snapshot of every run (time-sorted, deduped): USOON bid/ask and WTI perp last."""
    pts: dict[pd.Timestamp, dict[str, Any]] = {}
    for d in state_dir.iterdir():
        if d.is_dir() and any(d.glob("snapshots*.jsonl")):
            for s in read_jsonl(d, "snapshots.jsonl"):
                b, w = s.get("usoon_book") or {}, s.get("wti_perp_15m") or {}
                if b.get("bid") and w.get("last") and s.get("as_of_utc"):
                    t = _naive(s["as_of_utc"])
                    pts[t] = {"t": t, "bid": b["bid"], "ask": b["ask"], "wti": w["last"],
                              "id": s.get("snapshot_id")}  # fmt: skip
    return [pts[t] for t in sorted(pts)]


def decisions(run_dir: Path, st: dict[str, Any], snaps: list[dict[str, Any]], path) -> list:
    """The run's decisions with their book, eval entry kind and forward marks."""
    by_id = {s["id"]: s for s in snaps if s["id"]}
    evals = st.get("evals", [])
    out = []
    for r in read_jsonl(run_dir, "decisions.jsonl"):
        t = _naive(r["ts"])
        s = by_id.get(r.get("snapshot_id")) or min(snaps, key=lambda x: abs(x["t"] - t))
        ev = min(evals, key=lambda x: abs(_naive(x["ts"]) - t), default={})
        near = ev and abs(_naive(ev["ts"]) - t).total_seconds() < 900
        kind = ev.get("entry", "taker") if near else "taker"
        dec = r["decision"]
        marks = score(t, s["ask"], s["bid"], kind, path, dec.get("horizon_hours"))
        out.append(r | {"t": t, "bid": s["bid"], "ask": s["ask"], "entry": kind, "marks": marks})
    return out


def chart(snaps: list[dict[str, Any]], decs: list[dict[str, Any]]) -> str:
    """Inline SVG: both series as % change from the window's first point, one y axis."""
    if not snaps:
        return "<p class=muted>No snapshots logged yet.</p>"
    t0 = (decs[0]["t"] if decs else snaps[0]["t"]) - pd.Timedelta(hours=1)
    pts = [s for s in snaps if s["t"] >= t0] or snaps
    b0, w0 = pts[0]["bid"], pts[0]["wti"]
    ys = [v for s in pts for v in (s["bid"] / b0 - 1, s["wti"] / w0 - 1)]
    lo, hi = min(ys + [0.0]), max(ys + [0.0])
    span = (hi - lo) or 0.01
    lo, hi = lo - span * 0.08, hi + span * 0.08
    ta, tb = pts[0]["t"].timestamp(), max(pts[-1]["t"].timestamp(), pts[0]["t"].timestamp() + 1)
    top, right, bottom, left = PAD

    def x(t: pd.Timestamp) -> float:
        return left + (t.timestamp() - ta) / (tb - ta) * (W - left - right)

    def y(v: float) -> float:
        return top + (hi - v) / (hi - lo) * (H - top - bottom)

    def line(key: str, base: float) -> str:
        return " ".join(f"{x(s['t']):.1f},{y(s[key] / base - 1):.1f}" for s in pts)

    g = []
    step = _nice((hi - lo) / 5)
    v = step * int(lo / step)
    while v <= hi:
        if v >= lo:
            yv = f"{y(v):.1f}"
            g.append(f'<line class=grid x1={left} x2={W - right} y1="{yv}" y2="{yv}"/>'
                     f'<text class=tick x={W - right + 6} y="{y(v) + 4:.1f}">'
                     f"{v * 100:+.1f}%</text>")  # fmt: skip
        v += step
    hours = pd.date_range(pts[0]["t"].ceil("2h"), pts[-1]["t"], freq="2h")
    for h in hours:
        g.append(f'<text class=tick x="{x(h):.1f}" y={H - 12} text-anchor=middle>'
                 f'{h:%d %H:%M}</text>')  # fmt: skip
    g.append(f'<line class=base x1={left} x2={W - right} y1="{y(0):.1f}" y2="{y(0):.1f}"/>')
    g.append(f'<polyline class="s s2" points="{line("wti", w0)}"/>')
    g.append(f'<polyline class="s s1" points="{line("bid", b0)}"/>')
    for i, d in enumerate(decs):
        if d["t"] < pts[0]["t"]:
            continue
        letter, cls = MARK.get(d["decision"]["decision"], ("?", "none"))
        cx, cy = x(d["t"]), y(d["bid"] / b0 - 1)
        g.append(f'<a href="#d{i}"><g class="mk {cls}"><title>{e(d["ts"][:16])} '
                 f'{e(d["decision"]["decision"])}</title>'
                 f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="9"/>'
                 f'<text x="{cx:.1f}" y="{cy + 4:.1f}" text-anchor=middle>{letter}</text>'
                 "</g></a>")  # fmt: skip
    data = json.dumps([[round(x(s["t"]), 1), s["t"].strftime("%d %H:%M"), s["bid"], s["ask"],
                        s["wti"], round(s["bid"] / b0 - 1, 5), round(s["wti"] / w0 - 1, 5)]
                       for s in pts]).replace("</", "<\\/")  # fmt: skip
    return (
        f'<div class=chartwrap><svg id=chart viewBox="0 0 {W} {H}" role=img aria-label="USOON '
        f'bid and WTI perp, % change since {pts[0]["t"]:%d %H:%M} UTC">{"".join(g)}'
        f"<line id=xh class=xh y1={top} y2={H - bottom} x1='-9' x2='-9'/></svg>"
        f"<div id=tip class=tip hidden></div></div><script>const PTS={data};</script>"
    )


def _nice(x: float) -> float:
    for m in (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1):
        if x <= m:
            return m
    return 0.25


def cards(r: dict[str, Any]) -> str:
    ev = r["vs_baseline"]
    kv = [
        ("Mode", f"{r['mode']} · {r['profile']}"),
        ("Analyses", f"{r['agent_calls']} ({r['agent_errors']} errors)"),
        (
            "Decisions",
            " · ".join(f"{k.split('_')[0].title()} {v}" for k, v in r["decisions"].items() if v)
            or "none",
        ),  # fmt: skip
        ("Closed trades", f"{r['trades_closed']} ({r['wins']} wins)"),
        ("Realised PnL", f"{r['pnl_usdt']:+.4f} USDT"),
        ("Open position", "yes" if r["open_position"] else "flat"),
        ("Same as Rule", ev["same_decision"]),
        ("Status", "killed" if r["killed"] else (r["paused"] or "running/ok")),
    ]
    return (
        "<div class=cards>"
        + "".join(
            f"<div class=card><div class=k>{e(k)}</div><div class=v>{e(v)}</div></div>"
            for k, v in kv
        )
        + "</div>"
    )


def baseline(ev: dict[str, Any]) -> str:
    sbs = ev["side_by_side"]
    cols = sbs["columns"]
    rows = []
    for k, v in sbs.items():
        if k == "columns":
            continue
        fmt = (lambda z: pct(z)) if k == "buys_mean_net_4h" else (lambda z: "–" if z is None else z)
        rows.append(f"<tr><th>{e(k)}</th>" + "".join(f"<td>{e(fmt(z))}</td>" for z in v) + "</tr>")
    cal = ev["agent"]["calibration"]
    return (
        f"<table class=t><thead><tr><th></th>{''.join(f'<th>{e(c)}</th>' for c in cols)}"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
        f"<p class=muted>{ev['snapshots']} logged analyses re-decided by {e(ev['baseline'])} on "
        f"the same snapshots; scored on {ev['price_points']} pooled USOON bids up to "
        f"{e((ev['path_end_utc'] or '')[:16])} UTC. Agent calibration: n={cal['n']}, "
        f"bias {pct(cal['bias'])}, MAE {pct(cal['mae'])}.</p>"
    )


def decision_cards(decs: list[dict[str, Any]]) -> str:
    out = []
    for i, r in reversed(list(enumerate(decs))):
        d = r["decision"]
        name = d["decision"]
        cls = MARK.get(name, ("", "none"))[1]
        ev = {x.get("id"): x for x in d.get("evidence") or []}
        sup, con = d.get("supporting_evidence_ids") or [], d.get("counter_evidence_ids") or []

        def lst(ids: list[str], ev=ev) -> str:
            items = []
            for k in ids:
                x = ev.get(k) or {}
                url = str(x.get("source_url") or "")
                t = e(x.get("title") or k)
                link = f'<a href="{e(url)}" rel=noreferrer>{t}</a>' if url.startswith("http") else t
                items.append(f"<li><span class=tag>{e(x.get('kind', ''))}</span>{link}</li>")
            return "<ul>" + "".join(items) + "</ul>" if items else "<p class=muted>none</p>"

        g = r.get("gate")
        if g is None:
            gate = "not reached (no buy candidate)"
        elif g.get("ok"):
            gate = f"passed: qty {g.get('qty', '')}"
        else:
            gate = "rejected: " + "; ".join(map(str, g.get("why") or []))
        m = r["marks"]
        nets = " · ".join(f"{h} h {pct(m['net'].get(str(h)))}" for h in HORIZONS)
        extra = [("Catalyst", d.get("catalyst_or_reason")),
                 ("Known unknowns", "; ".join(d.get("known_unknowns") or [])),
                 ("Invalidation", "; ".join(d.get("invalidation_conditions") or [])),
                 ("Entry / exit", " / ".join(x for x in (d.get("entry_conditions"),
                                                        d.get("exit_conditions")) if x)),
                 ("Est. gross / horizon",
                  f"{d['estimated_gross_opportunity_pct']}% / {d.get('horizon_hours')} h"
                  if d.get("estimated_gross_opportunity_pct") is not None else None),
                 ("Lesson", d.get("lesson"))]  # fmt: skip
        out.append(
            f'<article class=dec id=d{i}><header><span class="badge {cls}">'
            f"{e(name)}</span><b>{e(r['ts'][:16].replace('T', ' '))} UTC</b>"
            f"<span class=muted>{e(r.get('why'))} · bid {r['bid']} ask {r['ask']}</span></header>"
            f"<p>{e(d.get('thesis'))}</p><div class=two><div><h4>Supporting</h4>{lst(sup)}</div>"
            f"<div><h4>Counter</h4>{lst(con)}</div></div><dl>"
            + "".join(f"<dt>{e(k)}</dt><dd>{e(v)}</dd>" for k, v in extra if v)
            + f"<dt>Gate</dt><dd>{e(gate)}</dd><dt>Outcome (long at ask, {e(r['entry'])})</dt>"
            f"<dd>{nets} · best 8 h {pct(m.get('best_net'))} · worst {pct(m.get('worst_net'))}"
            f"</dd></dl></article>"
        )
    return "".join(out) or "<p class=muted>No decisions yet.</p>"


def trades(run_dir: Path, closed: list[dict[str, Any]]) -> str:
    rows = []
    f = run_dir / "trades.csv"
    if f.exists():
        with f.open() as fh:
            for t in csv.DictReader(fh):
                rows.append(f"<tr><td>{e(t['time'][:19])}</td><td>{e(t['side'])}</td>"
                            f"<td>{float(t['qty']):.6g}</td><td>{float(t['price']):.2f}</td>"
                            f"<td>{float(t['fee']):.4f}</td><td>{e(t['reason'])}</td>"
                            "</tr>")  # fmt: skip
    pnl = "".join(f"<li>{e(c['hypothesis_id'])}: {c['pnl_usdt']:+.4f} USDT ({pct(c['pnl_pct'])}),"
                  f" {e(c['reason'])}</li>" for c in closed)  # fmt: skip
    if not rows:
        return "<p class=muted>No orders filled in this run.</p>"
    return ("<table class=t><thead><tr><th>Time UTC</th><th>Side</th><th>Qty</th><th>Price</th>"
            f"<th>Fee</th><th>Reason</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
            + (f"<ul>{pnl}</ul>" if pnl else ""))  # fmt: skip


def lessons(path: Path) -> str:
    if not path.exists():
        return "<p class=muted>No shared memory yet.</p>"
    mem = json.loads(path.read_text())
    rows = "".join(
        f"<tr><td>{e(x['id'])}</td><td><span class=tag>{e(x['status'])}</span></td>"
        f"<td>{x['support']}/{x['contradict']}</td><td>{e(x['text'])}</td></tr>"
        for x in mem.get("lessons", [])
    )
    return (f"<p class=muted>{len(mem.get('episodes', []))} episodes, "
            f"{mem.get('reflections', 0)} reflections, last "
            f"{e(mem.get('last_reflection_utc') or 'never')}.</p><table class=t><thead><tr>"
            f"<th>ID</th><th>Status</th><th>For/against</th><th>Lesson</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>")  # fmt: skip


ARCH = """<svg viewBox="0 0 960 250" class=arch role=img aria-label="Oil agent architecture">
<defs><marker id=ar viewBox="0 0 10 10" refX=9 refY=5 markerWidth=7 markerHeight=7 orient=auto>
<path d="M0,0L10,5L0,10z" class=ah /></marker></defs>
<g class=box><rect x=10 y=20 width=170 height=78 rx=8 /><text x=95 y=44>Feeds</text>
<text x=95 y=64 class=sm>Wallex USOON depth+candles</text><text x=95 y=80 class=sm>
Binance CL perp · G gates</text></g>
<g class=box><rect x=220 y=20 width=150 height=78 rx=8 /><text x=295 y=44>Snapshot</text>
<text x=295 y=64 class=sm>returns, divergence, cost</text><text x=295 y=80 class=sm>
limits · track record</text></g>
<g class=box><rect x=410 y=20 width=130 height=78 rx=8 /><text x=475 y=44>Triggers</text>
<text x=475 y=64 class=sm>1-h move, divergence,</text><text x=475 y=80 class=sm>
sweep, position</text></g>
<g class="box ai"><rect x=580 y=20 width=170 height=78 rx=8 /><text x=665 y=44>Analyst (LLM)</text>
<text x=665 y=64 class=sm>web search, evidence,</text><text x=665 y=80 class=sm>
JSON decision · or Rule</text></g>
<g class=box><rect x=790 y=20 width=160 height=78 rx=8 /><text x=870 y=44>Validate + Gate</text>
<text x=870 y=64 class=sm>schema, sources, edge,</text><text x=870 y=80 class=sm>
gap floor, caps, G off</text></g>
<g class=box><rect x=790 y=150 width=160 height=78 rx=8 /><text x=870 y=174>Executor</text>
<text x=870 y=194 class=sm>taker/maker, stop, expiry</text><text x=870 y=210 class=sm>
flat by contest end</text></g>
<g class=box><rect x=580 y=150 width=170 height=78 rx=8 /><text x=665 y=174>Logs</text>
<text x=665 y=194 class=sm>snapshots · decisions</text><text x=665 y=210 class=sm>
agent calls · trades</text></g>
<g class=box><rect x=360 y=150 width=180 height=78 rx=8 /><text x=450 y=174>Memory</text>
<text x=450 y=194 class=sm>episodes, lessons,</text><text x=450 y=210 class=sm>
reflection (all runs)</text></g>
<g class=box><rect x=120 y=150 width=200 height=78 rx=8 /><text x=220 y=174>Evaluate</text>
<text x=220 y=194 class=sm>forward net 1/2/4/8 h</text><text x=220 y=210 class=sm>
agent vs Rule baseline</text></g>
<g class=ln><path d="M180,59H218"/><path d="M370,59H408"/><path d="M540,59H578"/>
<path d="M750,59H788"/><path d="M870,98V148"/><path d="M790,189H752"/>
<path d="M580,189H542"/><path d="M360,189H322"/><path d="M450,150V120H295V100"/></g>
</svg>"""

CSS = """:root{color-scheme:light;--bg:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;
--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;
--buy:#0ca30c;--watch:#eda100;--exit:#d03b3b;--none:#898781}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){color-scheme:dark;--bg:#0d0d0d;
--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;
--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--watch:#c98500}}
:root[data-theme=dark]{color-scheme:dark;--bg:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;
--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--watch:#c98500}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1000px;margin:0 auto;padding:24px 16px 64px}h1{font-size:22px;margin:0 0 4px}
h2{font-size:16px;margin:32px 0 10px}h4{margin:8px 0 4px;font-size:13px;color:var(--ink2)}
section,.dec,.card{background:var(--surface);border:1px solid var(--ring);border-radius:10px}
section{padding:16px}.muted{color:var(--muted)}a{color:var(--s1)}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:10px}
.card{padding:10px 12px}.k{font-size:12px;color:var(--muted)}.v{font-size:16px;font-weight:600}
.chartwrap{position:relative;overflow-x:auto}svg{width:100%;height:auto;display:block}
#chart{min-width:640px}
.grid{stroke:var(--grid);stroke-width:1}.base{stroke:var(--axis);stroke-width:1}
.tick{fill:var(--muted);font-size:11px}.s{fill:none;stroke-width:2;stroke-linejoin:round}
.s1{stroke:var(--s1)}.s2{stroke:var(--s2)}.xh{stroke:var(--axis);stroke-width:1}
.mk circle{stroke:var(--surface);stroke-width:2}.mk text{fill:#fff;font-size:11px;font-weight:700}
.mk.buy circle{fill:var(--buy)}.mk.watch circle{fill:var(--watch)}.mk.exit circle{fill:var(--exit)}
.mk.none circle{fill:var(--none)}.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;
color:var(--ink2);margin-bottom:6px}.sw{display:inline-block;width:14px;height:3px;
vertical-align:middle;margin-right:5px;border-radius:2px}
.tip{position:absolute;pointer-events:none;background:var(--surface);border:1px solid var(--ring);
border-radius:6px;padding:6px 8px;font-size:12px;box-shadow:0 2px 8px rgba(0,0,0,.15);
white-space:nowrap}.t{width:100%;border-collapse:collapse;font-size:13px}
.t th,.t td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--grid);vertical-align:top}
.tablewrap{overflow-x:auto}.dec{padding:14px 16px;margin:0 0 12px}
.dec header{display:flex;gap:10px;flex-wrap:wrap;align-items:baseline}
.badge{font-size:11px;font-weight:700;padding:2px 8px;border-radius:99px;color:#fff;
background:var(--none)}.badge.buy{background:var(--buy)}.badge.watch{background:var(--watch)}
.badge.exit{background:var(--exit)}.two{display:grid;grid-template-columns:1fr 1fr;gap:12px}
ul{margin:0;padding-left:18px}
.tag{font-size:11px;color:var(--ink2);border:1px solid var(--ring);border-radius:4px;
padding:0 4px;margin-right:6px}dl{display:grid;grid-template-columns:minmax(80px,max-content) 1fr;
gap:2px 12px;margin:10px 0 0;font-size:13px}dt{color:var(--muted)}dd{margin:0}
@media (max-width:640px){.two{grid-template-columns:1fr}
dl{grid-template-columns:1fr}dd{margin-bottom:6px}}
.arch .box rect{fill:var(--surface);stroke:var(--axis)}.arch .ai rect{stroke:var(--s1);
stroke-width:2}.arch text{fill:var(--ink);font-size:14px;font-weight:600;text-anchor:middle}
.arch text.sm{fill:var(--ink2);font-size:11px;font-weight:400}
.arch .ln path{fill:none;stroke:var(--muted);stroke-width:1.5;marker-end:url(#ar)}
.arch .ah{fill:var(--muted)}"""

JS = """const svg=document.getElementById('chart'),tip=document.getElementById('tip'),
xh=document.getElementById('xh');
if(svg&&typeof PTS!=="undefined"&&PTS.length){
svg.addEventListener('mousemove',ev=>{const r=svg.getBoundingClientRect(),
k=r.width/svg.viewBox.baseVal.width,mx=(ev.clientX-r.left)/k;let b=PTS[0];
for(const p of PTS)if(Math.abs(p[0]-mx)<Math.abs(b[0]-mx))b=p;
xh.setAttribute('x1',b[0]);xh.setAttribute('x2',b[0]);
const f=v=>(v>=0?'+':'')+(v*100).toFixed(2)+'%';
tip.innerHTML=`<b>${b[1]} UTC</b><br>USOON bid ${b[2]} / ask ${b[3]} (${f(b[5])})<br>`+
`WTI perp ${b[4]} (${f(b[6])})`;tip.hidden=false;const x=b[0]*k;
tip.style.left=Math.min(x+12,r.width-tip.offsetWidth-4)+'px';tip.style.top='8px'});
svg.addEventListener('mouseleave',()=>{tip.hidden=true;xh.setAttribute('x1',-9);
xh.setAttribute('x2',-9)})}"""


def render(run_id: str, state_dir: Path | None = None, memory_path: Path | None = None) -> str:
    """The dashboard HTML of ``run_id``."""
    state_dir = state_dir or config.STATE_DIR
    if memory_path is None:
        memory_path = (config.MEMORY_PATH if state_dir == config.STATE_DIR
                       else state_dir / "memory.json")  # fmt: skip
    run_dir = state_dir / run_id
    st = json.loads((state_dir / f"{run_id}.json").read_text())
    rep = report(run_id, state_dir)
    snaps = market(state_dir)
    decs = decisions(run_dir, st, snaps, price_path(state_dir)) if snaps else []
    legend = ("<div class=legend><span><i class=sw style='background:var(--s1)'></i>USOON bid "
              "(Wallex)</span><span><i class=sw style='background:var(--s2)'></i>WTI perp "
              "(Binance CLUSDT)</span><span>Markers: B buy · W watch · N no trade · X exit "
              "(click to open the card)</span></div>")  # fmt: skip
    now = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M")
    return f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>Oil Run {e(run_id)}</title>
<style>{CSS}</style></head><body><main>
<h1>Oil agent · {e(run_id)}</h1><p class=muted>USOON/USDT on Wallex, long only, event-driven.
Generated {now} UTC from data/state/{e(run_id)}.</p>
{cards(rep)}
<h2>Price and decisions</h2><section>{legend}{chart(snaps, decs)}
<p class=muted>Both series as % change from the first point shown (one shared axis).</p></section>
<h2>Agent vs Rule baseline</h2><section class=tablewrap>{baseline(rep["vs_baseline"])}</section>
<h2>Decisions ({len(decs)})</h2>{decision_cards(decs)}
<h2>Trades</h2><section class=tablewrap>{trades(run_dir, st.get("closed", []))}</section>
<h2>Shared lessons</h2><section class=tablewrap>{lessons(memory_path)}</section>
<h2>Architecture</h2><section>{ARCH}<p class=muted>Every 5 min: feeds → snapshot → trigger
check. Only on a trigger does the analyst run; its JSON decision is validated and a buy must
pass the hard gate before any order. Everything is logged; memory and the scores feed back into
the next snapshot.</p></section>
</main><script>{JS}</script></body></html>"""


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-id", required=True)
    p.add_argument("--out", type=Path, help="default data/state/<run_id>/dashboard.html")
    a = p.parse_args(argv)
    out = a.out or config.STATE_DIR / a.run_id / "dashboard.html"
    out.write_text(render(a.run_id))
    print(out)


if __name__ == "__main__":
    main()
