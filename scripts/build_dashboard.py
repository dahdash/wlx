"""Render ``docs/results/results.json`` (``build_results.py``) as the results pages.

- ``docs/demo/dashboard.html``: one self-contained page, data inlined, charts drawn as SVG by a
  few lines of JS, no network;
- ``RESULTS.md`` and ``RESULTS.fa.md``: the same numbers as text.

    python scripts/build_dashboard.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Live Results</title>
<style>
:root{--bg:#f7f7f5;--card:#fff;--fg:#1d1d1f;--mut:#6b6b70;--line:#e3e3e0;--acc:#2563eb;
--acc2:#d97706;--pos:#15803d;--neg:#b91c1c}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#111214;--card:#1b1c1f;
--fg:#ececee;--mut:#9a9aa2;--line:#2c2d31;--acc:#60a5fa;--acc2:#fbbf24;--pos:#4ade80;
--neg:#f87171}}
:root[data-theme="dark"]{--bg:#111214;--card:#1b1c1f;--fg:#ececee;--mut:#9a9aa2;--line:#2c2d31;
--acc:#60a5fa;--acc2:#fbbf24;--pos:#4ade80;--neg:#f87171}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:18px;margin:0 0 12px}
.mut{color:var(--mut)}.pos{color:var(--pos)}.neg{color:var(--neg)}
.banner{border:1px solid var(--acc2);border-radius:8px;padding:8px 12px;margin:12px 0 20px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}
.kpi,.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.kpi b{display:block;font-size:22px;font-variant-numeric:tabular-nums}
.kpi>span{font-size:13px;color:var(--mut)}
.card{margin-top:16px}.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.two .card{margin-top:0}
@media (max-width:760px){.two{grid-template-columns:1fr}}
svg{width:100%;height:auto;display:block}
table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--mut);font-weight:600}.scroll{overflow-x:auto}
.tag{display:inline-block;padding:1px 7px;border-radius:9px;border:1px solid var(--line);
font-size:12px}
blockquote{margin:8px 0;padding:8px 12px;border-left:3px solid var(--acc);background:var(--bg)}
a{color:var(--acc)}.legend span{margin-right:14px;font-size:13px}
.sw{display:inline-block;width:12px;height:3px;vertical-align:middle;margin-right:5px}
</style></head><body><main>
<h1>Live results</h1>
<div class="mut" id="sub"></div>
<div class="banner" id="status"></div>
<div class="kpis" id="kpis"></div>

<div class="card"><h2>Account value in toman</h2>
<div class="legend"><span><i class="sw" style="background:var(--acc)"></i>value in TMN
(USDT × USDTTMN hourly close + unconverted TMN)</span>
</div>
<div id="eq"></div>
<div class="legend"><span><i class="sw" style="background:var(--acc2)"></i>equity in USDT (what the
strategy controls)</span></div>
<div id="usdt"></div></div>

<div class="two">
<div class="card"><h2>Strategy G (trend, 5 coins)</h2><div id="g"></div></div>
<div class="card"><h2>Oil agent (USOON)</h2><div id="oil"></div></div>
</div>

<div class="card"><h2>Oil agent: last decision and what it learned</h2><div id="oildec"></div></div>
<div class="card"><h2>Every filled order</h2><div class="scroll" id="trades"></div></div>
<div class="card"><h2>Backtest and validation</h2><div id="bt"></div></div>
</main>
<script>
const R = __DATA__;
const $ = id => document.getElementById(id);
const fmt = (x, d = 0) => x == null ? "—" : Number(x).toLocaleString("en-US",
  {minimumFractionDigits: d, maximumFractionDigits: d});
const sgn = (x, d = 2) => x == null ? "—" : (x > 0 ? "+" : "") + fmt(x, d) + " %";
const cut = (s, n) => esc(s) + (String(s ?? "").length >= n ? " …" : "");
const cls = x => x > 0 ? "pos" : x < 0 ? "neg" : "";
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",
  '"':"&quot;"}[c]));
const table = (head, rows) => `<table><tr>${head.map(h => `<th>${h}</th>`).join("")}</tr>` +
  rows.map(r => `<tr>${r.map(c => `<td>${c}</td>`).join("")}</tr>`).join("") + "</table>";

const fin = R.final, m = R.mark, st = R.start;
$("sub").textContent = `Built ${R.built_utc} UTC · account started ${st.t} UTC with ` +
  `${fmt(st.tmn)} TMN`;
$("status").innerHTML = fin
  ? `<b>Final.</b> Liquidated to toman at ${fin.t} UTC: ${fmt(fin.got_tmn)} TMN.`
  : `<b>Interim.</b> The account is still trading; the official result is the toman balance ` +
    `after the end-of-contest liquidation. The mark below values the USDT at the USDTTMN bid ` +
    `minus the 0.3 % taker fee.`;
const endTmn = fin ? fin.got_tmn + st.tmn_left : m.tmn_after_taker_fee;
const ret = endTmn / st.tmn * 100 - 100;
const kp = [
  [fmt(endTmn) + " TMN", fin ? "final value" : "value now (after sell fee)"],
  [`<span class="${cls(ret)}">${sgn(ret)}</span>`, "net return in toman"],
  [sgn(R.equity.max_dd_tmn_pct), "max drawdown (toman)"],
  [sgn(R.equity.max_dd_usdt_pct, 3), "max drawdown (USDT)"],
  [fmt(R.trades.length), "filled orders"],
  [fmt(R.fees.tmn_total_approx) + " TMN", "fees paid (≈)"],
];
$("kpis").innerHTML = kp.map(([b, s]) => `<div class="kpi"><b>${b}</b><span>${s}</span></div>`)
  .join("");

function chart(el, pts, key, color, dec) {
  const W = 1000, H = 260, L = 70, B = 26, T = 10;
  const ts = pts.map(p => Date.parse(p.t + "Z")), ys = pts.map(p => p[key]);
  const ok = ys.filter(y => y != null);
  if (!ok.length) { el.innerHTML = "<p class='mut'>no data</p>"; return; }
  let lo = Math.min(...ok), hi = Math.max(...ok); const pad = (hi - lo) * 0.1 || hi * 0.001;
  lo -= pad; hi += pad;
  const t0 = ts[0], t1 = ts[ts.length - 1] || t0 + 1;
  const X = t => L + (W - L - 10) * (t - t0) / (t1 - t0 || 1);
  const Y = y => T + (H - T - B) * (1 - (y - lo) / (hi - lo));
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img">`;
  for (let i = 0; i <= 4; i++) {
    const y = lo + (hi - lo) * i / 4;
    s += `<line x1="${L}" x2="${W - 10}" y1="${Y(y)}" y2="${Y(y)}" stroke="var(--line)"/>` +
      `<text x="${L - 6}" y="${Y(y) + 4}" text-anchor="end" font-size="12" fill="var(--mut)">` +
      `${fmt(y, dec)}</text>`;
  }
  const day = 6 * 3600e3;
  for (let t = Math.ceil(t0 / day) * day; t <= t1; t += day) {
    const d = new Date(t).toISOString();
    s += `<text x="${X(t)}" y="${H - 6}" text-anchor="middle" font-size="12" ` +
      `fill="var(--mut)">${d.slice(5, 10)} ${d.slice(11, 16)}</text>`;
  }
  let path = "", pen = false;
  pts.forEach((p, i) => {
    const y = p[key];
    if (y == null) pen = false;
    if (y != null) { path += (pen ? "L" : "M") + X(ts[i]).toFixed(1) + "," + Y(y).toFixed(1);
      pen = true; }
  });
  s += `<path d="${path}" fill="none" stroke="${color}" stroke-width="2"/>`;
  pts.forEach((p, i) => { if (p.orders && p[key] != null) s += `<circle cx="${X(ts[i])}" ` +
    `cy="${Y(p[key])}" r="4" fill="${color}"/>`; });
  el.innerHTML = s + "</svg>";
}
chart($("eq"), R.curve, "tmn", "var(--acc)", 0);
chart($("usdt"), R.curve, "usdt", "var(--acc2)", 3);

const g = R.g;
$("g").innerHTML = `<p>${g.cycles} cycles of 30 min, ${g.orders} orders. ` +
  `${g.cycles_any_gate_on} cycles had any coin's 28-day trend gate open; the best 28-day ` +
  `return seen was ${sgn(g.best_28d.ret_pct)} (${g.best_28d.symbol.replace("USDT", "")}), ` +
  `so the strategy stayed in USDT by design. Uptime ${g.uptime.pct} %.</p>` +
  table(["coin", "last signal"], Object.entries(g.last_signals).map(([k, v]) =>
    [k.replace("USDT", ""), esc(v.split(": ")[1] || v)])) +
  `<p class="mut">Risk events: ${g.risk_events.length || "none"}.</p>`;

const role = (o, name) => {
  const f = o.forward;
  return `<h3 style="font-size:15px;margin:10px 0 4px">${name} <span class="tag">${o.mode}` +
    `</span></h3><p>${o.decisions} decisions: ${Object.entries(o.by_decision).map(([k, v]) =>
    `${v} ${k}`).join(", ")}; ${o.trades} trades; uptime ${o.uptime.pct} %.</p>` +
    table(["horizon", "scored", "entry would have lost", "mean net"],
      Object.entries(f).map(([h, x]) => [h, x.n, x.entry_would_lose_pct == null ? "—" :
        x.entry_would_lose_pct + " %", `<span class="${cls(x.mean_net_pct)}">` +
        sgn(x.mean_net_pct, 3) + "</span>"]));
};
const lvs = R.oil.live_vs_shadow;
$("oil").innerHTML = role(R.oil.live, "Live (armed)") + role(R.oil.shadow, "Shadow (paper)") +
  `<p class="mut">Forward score: each decision is checked 1–8 h later against the price, net of ` +
  `the round-trip cost. Live and shadow decided in the same minute ${lvs.same_minute} times and ` +
  `agreed ${lvs.agree} times.</p>`;

const ld = R.oil.live.last_decision;
$("oildec").innerHTML = (ld ? `<p><span class="tag">${ld.decision}</span> ${ld.t} UTC · ` +
  `wake-up: ${esc(ld.why)}</p><blockquote>${cut(ld.thesis, 600)}</blockquote>` +
  `<p class="mut">Answers are capped at 600 characters (thesis) and 240 (lesson).</p>` : "") +
  `<h3 style="font-size:15px">Lessons it wrote for itself (fed back into the next calls)</h3>` +
  `<ul>${R.oil.live.lessons.slice(-6).map(l => `<li><span class="mut">${l.t}</span> ` +
  `${cut(l.text, 240)}</li>`).join("")}</ul>`;

$("trades").innerHTML = table(["time (UTC)", "engine", "market", "side", "qty", "price", "fee",
  "why"], R.trades.map(r => [r.time_utc.replace("T", " "), r.engine, r.symbol, r.action,
  r.filled_qty, r.price, r.fee, esc(r.reason)]));

const bt = R.backtest;
$("bt").innerHTML = `<ul><li><a href="../${bt.metrics.replace("docs/", "")}">Metrics by ` +
  `period</a></li><li><a href="../${bt.validation.replace("docs/", "")}">Validation charts ` +
  `(permutation test, holdout, sensitivity)</a></li><li><a href="../backtest/` +
  `tearsheet-full.html">Full tearsheet</a></li><li><a href="../${bt.crosscheck.replace("docs/",
  "")}">Independent cross-check</a></li></ul>`;
</script></body></html>
"""


def _f(x, d=0) -> str:
    return "—" if x is None else f"{x:,.{d}f}"


def _s(x, d=2) -> str:
    return "—" if x is None else f"{x:+.{d}f} %"


def numbers(r: dict) -> dict:
    st, m, fin, eq = r["start"], r["mark"], r["final"], r["equity"]
    end = fin["got_tmn"] + st["tmn_left"] if fin else m["tmn_after_taker_fee"]
    rate_end = fin["got_tmn"] / fin["sold_usdt"] if fin else m["usdttmn_bid"]
    return {
        "end": end,
        "ret": 100 * (end / st["tmn"] - 1),
        "usdt_ret": 100 * (eq["usdt_now"] / eq["usdt_first_cycle"] - 1),
        "rate_ret": 100 * (rate_end / st["avg_usdttmn"] - 1),
        "rate_end": rate_end,
        # TMN → USDT at the start (fee taken in USDT) and USDT → TMN at the end (0.3 % taker)
        "conv_fee": 100 * (eq["usdt_first_cycle"] / st["usdt_bought"] - 1)
        + (100 * -r["fees"]["tmn"] / (fin["got_tmn"] + r["fees"]["tmn"]) if fin else -0.3),
    }


def results_md(r: dict) -> str:
    st, fin, eq, g, oil = r["start"], r["final"], r["equity"], r["g"], r["oil"]
    n = numbers(r)
    status = (
        f"**Final**: the account was sold back to toman at {fin['t']} UTC."
        if fin
        else f"**Interim, built {r['built_utc']} UTC.** The account keeps trading until the "
        "official end; the score uses the toman balance after the end-of-contest sale to toman. "
        "Until then the value is marked at the USDTTMN bid minus the 0.3 % taker fee."
    )
    rows = [
        ("Start", f"{_f(st['tmn'])} TMN ({st['t']} UTC)"),
        ("Value " + ("at the end" if fin else "now"), f"{_f(n['end'])} TMN"),
        ("Net return in toman", _s(n["ret"])),
        ("of which: equity in USDT", _s(n["usdt_ret"], 3)),
        ("of which: TMN ↔ USDT conversion fees", _s(n["conv_fee"])),
        ("of which: USDT/TMN rate", f"{_s(n['rate_ret'])} ({_f(st['avg_usdttmn'])} → "
         f"{_f(n['rate_end'])})"),
        ("Max drawdown (toman / USDT)", f"{_s(eq['max_dd_tmn_pct'])} / "
         f"{_s(eq['max_dd_usdt_pct'], 3)}"),
        ("Filled orders", str(len(r["trades"]))),
        ("Fees paid", f"≈ {_f(r['fees']['tmn_total_approx'])} TMN"),
        ("Uptime (G / oil live)", f"{g['uptime']['pct']} % / {oil['live']['uptime']['pct']} %"),
    ]  # fmt: skip
    oil_rows = []
    for name, o in (("live (armed)", oil["live"]), ("shadow (paper)", oil["shadow"])):
        f4 = o["forward"]["4h"]
        dec = ", ".join(f"{v} {k}" for k, v in o["by_decision"].items())
        oil_rows.append(
            f"| {name} | {o['decisions']} ({dec}) | {o['trades']} | "
            f"{f4['entry_would_lose_pct']} % | {_s(f4['mean_net_pct'], 3)} | "
            f"{o['uptime']['pct']} % |"
        )
    trades = [
        f"| {t['time_utc'].replace('T', ' ')} | {t['engine']} | {t['symbol']} | {t['action']} | "
        f"{t['filled_qty']} | {t['price']} | {t['fee']} | {t['reason']} |"
        for t in r["trades"]
    ]
    lvs = oil["live_vs_shadow"]
    return "\n".join([
        "# Results",
        "",
        "[فارسی](RESULTS.fa.md) · [dashboard](docs/demo/dashboard.html) · "
        "[decision log](docs/decision-log.md) · data: `docs/results/results.json`",
        "",
        status,
        "",
        "| | |",
        "| --- | --- |",
        *(f"| {k} | {v} |" for k, v in rows),
        "",
        "The value is USDT equity × USDTTMN plus the "
        f"{_f(st['tmn_left'])} TMN that could not be converted at the start (below the market's "
        "minimum order). The engines trade in USDT, so the toman return is the USDT result "
        "times the USDT/TMN rate move, less the fees of converting in and out; the lines "
        "above split it.",
        "",
        "## Strategy G (trend, BTC/ETH/XRP/SOL/BNB)",
        "",
        f"{g['cycles']} cycles of 30 min, {g['orders']} orders. G only buys a coin whose "
        f"28-day return is positive; {g['cycles_any_gate_on']} cycles had any gate open (best "
        f"28-day return seen: {_s(g['best_28d']['ret_pct'])}, {g['best_28d']['symbol']}). In a "
        "falling market it holds USDT by design, which is the drawdown control the backtest "
        "selected. Risk events: " + (str(len(g["risk_events"])) if g["risk_events"] else "none")
        + ".",
        "",
        "## Oil agent (USOON, AI analyst + deterministic risk gate)",
        "",
        "| run | decisions | trades | 4-h: entry would have lost | 4-h mean net | uptime |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
        *oil_rows,
        "",
        "Every decision is scored 1, 2, 4 and 8 h later against the price, net of the "
        "round-trip cost (forward score). A NO_TRADE was right when the entry it declined would "
        f"have lost. Live and shadow decided in the same minute {lvs['same_minute']} times and "
        f"agreed {lvs['agree']} times. The agent wrote {len(oil['live']['lessons'])} lessons "
        "from its own scores, which are fed back into its next calls.",
        "",
        "## Every filled order",
        "",
        "| time (UTC) | engine | market | side | qty | price | fee | why |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
        *trades,
        "",
        "## Backtest and shadow test",
        "",
        "- [Metrics by period](docs/backtest/metrics.md), "
        "[full tearsheet](docs/backtest/tearsheet-full.html)",
        "- [Validation charts](docs/backtest/validation.html): permutation test, holdout, "
        "sensitivity",
        "- [Independent cross-check](docs/backtest/crosscheck.md)",
        "- Shadow test: the oil agent ran in paper mode next to the live run (table above).",
        "",
        "## Rebuild",
        "",
        "```",
        "python scripts/build_results.py      # data/state + public USDTTMN prices → results.json",
        "python scripts/build_dashboard.py    # → dashboard.html, RESULTS.md, RESULTS.fa.md",
        "```",
        "",
    ])  # fmt: skip


def results_fa(r: dict) -> str:
    st, fin, eq, g, oil = r["start"], r["final"], r["equity"], r["g"], r["oil"]
    n = numbers(r)
    status = (
        f"**نهایی**: حساب در {fin['t']} UTC به تومان فروخته شد."
        if fin
        else f"**موقت، ساخته‌شده در {r['built_utc']} UTC.** حساب تا پایان رسمی مسابقه معامله "
        "می‌کند و امتیاز بر اساس موجودی تومانی پس از فروش پایانی به تومان است. تا آن زمان ارزش "
        "حساب با قیمت خرید USDTTMN منهای کارمزد ۰٫۳ درصدی taker حساب می‌شود."
    )
    rows = [
        ("شروع", f"{_f(st['tmn'])} TMN ({st['t']} UTC)"),
        ("ارزش " + ("پایانی" if fin else "فعلی"), f"{_f(n['end'])} TMN"),
        ("بازده خالص تومانی", _s(n["ret"])),
        ("سهم ارزش دلاری (USDT)", _s(n["usdt_ret"], 3)),
        ("سهم کارمزد تبدیل تومان ↔ USDT", _s(n["conv_fee"])),
        ("سهم نرخ USDT/TMN", f"{_s(n['rate_ret'])} ({_f(st['avg_usdttmn'])} → "
         f"{_f(n['rate_end'])})"),
        ("بیشترین افت (تومان / USDT)", f"{_s(eq['max_dd_tmn_pct'])} / "
         f"{_s(eq['max_dd_usdt_pct'], 3)}"),
        ("سفارش‌های پرشده", str(len(r["trades"]))),
        ("کارمزد پرداختی", f"≈ {_f(r['fees']['tmn_total_approx'])} TMN"),
        ("پایداری اجرا (G / عامل نفت)",
         f"{g['uptime']['pct']} % / {oil['live']['uptime']['pct']} %"),
    ]  # fmt: skip
    oil_rows = []
    for name, o in (("زنده (مسلح)", oil["live"]), ("سایه (کاغذی)", oil["shadow"])):
        f4 = o["forward"]["4h"]
        dec = ", ".join(f"{v} {k}" for k, v in o["by_decision"].items())
        oil_rows.append(
            f"| {name} | {o['decisions']} ({dec}) | {o['trades']} | "
            f"{f4['entry_would_lose_pct']} % | {_s(f4['mean_net_pct'], 3)} | "
            f"{o['uptime']['pct']} % |"
        )
    trades = [
        f"| {t['time_utc'].replace('T', ' ')} | {t['engine']} | {t['symbol']} | {t['action']} | "
        f"{t['filled_qty']} | {t['price']} | {t['fee']} |"
        for t in r["trades"]
    ]
    lvs = oil["live_vs_shadow"]
    return "\n".join([
        '<div dir="rtl">',
        "",
        "# نتایج",
        "",
        "[English](RESULTS.md) · [داشبورد](docs/demo/dashboard.html) · "
        "[لاگ تصمیم‌ها](docs/decision-log.md) · داده: `docs/results/results.json`",
        "",
        status,
        "",
        "| | |",
        "| --- | --- |",
        *(f"| {k} | {v} |" for k, v in rows),
        "",
        "ارزش حساب = موجودی USDT × نرخ USDTTMN به‌علاوهٔ "
        f"{_f(st['tmn_left'])} تومانی که در شروع قابل تبدیل نبود (کمتر از حداقل سفارش بازار). "
        "موتورها با USDT معامله می‌کنند، پس بازده تومانی حاصل نتیجهٔ دلاری ضرب در تغییر نرخ "
        "USDT/TMN منهای کارمزد تبدیل ورود و خروج است و سطرهای بالا این سهم‌ها را جدا نشان می‌دهند.",
        "",
        "## استراتژی G (روند؛ BTC/ETH/XRP/SOL/BNB)",
        "",
        f"{g['cycles']} چرخهٔ ۳۰ دقیقه‌ای، {g['orders']} سفارش. G فقط کوینی را می‌خرد که بازده "
        f"۲۸ روزه‌اش مثبت باشد؛ در {g['cycles_any_gate_on']} چرخه دروازه‌ای باز بود (بهترین بازده "
        f"۲۸ روزهٔ دیده‌شده: {_s(g['best_28d']['ret_pct'])}، {g['best_28d']['symbol']}). در بازار "
        "نزولی عمداً در USDT می‌ماند؛ این همان کنترل افت سرمایه است که بک‌تست انتخاب کرد. "
        "رویداد ریسک: " + (str(len(g["risk_events"])) if g["risk_events"] else "هیچ") + ".",
        "",
        "## عامل نفت (USOON؛ تحلیلگر هوش مصنوعی + دروازهٔ ریسک قطعی)",
        "",
        "| اجرا | تصمیم‌ها | معامله | ۴ ساعت: ورود ضرر می‌داد | میانگین خالص ۴ ساعت | پایداری |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
        *oil_rows,
        "",
        "هر تصمیم ۱، ۲، ۴ و ۸ ساعت بعد با قیمت واقعی و پس از کسر هزینهٔ رفت‌وبرگشت سنجیده "
        "می‌شود. NO_TRADE وقتی درست بوده که ورودِ ردشده ضرر می‌داد. اجرای زنده و سایه "
        f"{lvs['same_minute']} بار در یک دقیقه تصمیم گرفتند و {lvs['agree']} بار هم‌نظر بودند. "
        f"عامل از امتیازهای خودش {len(oil['live']['lessons'])} درس نوشت که به فراخوانی‌های "
        "بعدی‌اش داده می‌شود.",
        "",
        "## همهٔ سفارش‌های پرشده",
        "",
        "| زمان (UTC) | موتور | بازار | سمت | مقدار | قیمت | کارمزد |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
        *trades,
        "",
        "دلیل هر سفارش در [لاگ تصمیم‌ها](docs/decision-log.md) و [نسخهٔ انگلیسی](RESULTS.md) "
        "آمده است.",
        "",
        "## بک‌تست و آزمون سایه",
        "",
        "- [معیارها به تفکیک دوره](docs/backtest/metrics.md)، "
        "[گزارش کامل](docs/backtest/tearsheet-full.html)",
        "- [نمودارهای اعتبارسنجی](docs/backtest/validation.html): آزمون جایگشت، داده‌های کنار "
        "گذاشته، حساسیت",
        "- [بررسی مستقل](docs/backtest/crosscheck.md)",
        "- آزمون سایه: عامل نفت در حالت کاغذی کنار اجرای زنده کار کرد (جدول بالا).",
        "",
        "</div>",
        "",
    ])  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--results", default=str(ROOT / "docs" / "results" / "results.json"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "demo" / "dashboard.html"))
    ap.add_argument("--md-dir", default=str(ROOT), help="where RESULTS.md / RESULTS.fa.md go")
    a = ap.parse_args(argv)
    res = json.loads(Path(a.results).read_text())
    res.pop("usdttmn_1h", None)  # already folded into the curve
    data = json.dumps(res, ensure_ascii=False).replace("</", "<\\/")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(PAGE.replace("__DATA__", data))
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
    md = Path(a.md_dir)
    (md / "RESULTS.md").write_text(results_md(res))
    (md / "RESULTS.fa.md").write_text(results_fa(res))
    print(f"wrote {md}/RESULTS.md, RESULTS.fa.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
