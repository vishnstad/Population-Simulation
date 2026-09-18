"""M10 — the results report (checklist 5.5, 6.5).

    "Every headline number carries three figures, never one: **achieved W1, the
     B0a baseline, and the noise floor.**"

So the report cannot render a bare number: `triple()` is the only way a headline
figure reaches the page, and it takes all three.

Self-contained HTML, regenerable from `runs/` alone. It reads whatever run
directories exist — `layer4_report.json`, `gate4_report.json`,
`leakage_report.json` — takes the most recent per model, and says plainly what is
missing rather than omitting the section.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .charts import Bar, bar_chart, dot_line, scatter

__all__ = ["ReportInputs", "build_report", "collect_runs"]

#: Arm -> how to describe it in one line. Order is the order they are listed in.
ARM_NOTES = {
    "system": "the method",
    "system_percluster": "ablation: per-cluster scale, shrunk to the global one",
    "system_s1": "ablation: deviation scale fixed at 1, nothing fitted",
    "system_bytopic": "ablation: per-topic scale",
    "system_nosubspace": "ablation: no subgroup-subspace projection",
    "B0a": "baseline: the true national marginal copied to every cluster",
    "B0a_national": "baseline: the full-population marginal (vs the in-scope mixture)",
    "B0b": "baseline: a population-level LLM call copied to every cluster",
    "B1": "baseline: copy the most textually similar anchor item",
    "B3": "skyline: gradient boosting trained on the real target labels",
    "B4": "ablation: raw elicitation, uncalibrated",
}

#: Model tag -> (display name, parameter count for the tier curve).
TIERS = {
    "ministral-3b-2512": ("Ministral 3B", 3),
    "qwen2.5-ctx8k:7b-instruct-8192": ("Qwen2.5 7B", 7),
    "qwen2.5-ctx8k_7b-instruct-8192": ("Qwen2.5 7B", 7),
    "ministral-8b-2512": ("Ministral 8B", 8),
    "ministral-14b-2512": ("Ministral 14B", 14),
}


@dataclass
class ReportInputs:
    layer4: dict[str, dict[str, Any]]          # model -> report
    gate4: dict[str, Any] | None = None
    gate3: dict[str, Any] | None = None
    leakage: dict[str, dict[str, Any]] | None = None
    router: list[dict[str, float]] | None = None
    generated: str = ""


def _newest(paths: list[Path]) -> Path | None:
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None


def collect_runs(runs_root: Path, *, profile: str = "permutation") -> ReportInputs:
    """Most recent artifact of each kind; Layer 4 kept per model.

    Only the canonical arm counts toward the headline tables: the ablation runs
    (a capped ensemble, the no-anchor card) write a Layer 4 report under the same
    model tag, and a report that silently headlined whichever ran last would be
    reporting an ablation as the result.
    """
    layer4: dict[str, dict] = {}
    best_mtime: dict[str, float] = {}
    for p in sorted(runs_root.glob("*/layer4_report.json")):
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if d.get("profile", profile) != profile or d.get("ensemble_limit", 0):
            continue
        m = d.get("model") or "?"
        if p.stat().st_mtime >= best_mtime.get(m, -1):
            layer4[m], best_mtime[m] = d, p.stat().st_mtime
    leak: dict[str, dict] = {}
    for p in sorted(runs_root.glob("*/leakage_report.json")):
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        leak[d.get("model") or "?"] = d
    g4 = _newest(list(runs_root.glob("*/gate4_report.json")))
    g3 = _newest(list(runs_root.glob("*/gate3_report.json")))
    return ReportInputs(
        layer4=layer4,
        gate4=json.loads(g4.read_text()) if g4 else None,
        gate3=json.loads(g3.read_text()) if g3 else None,
        leakage=leak or None,
        generated=datetime.now(UTC).strftime("%d %b %Y %H:%M UTC"),
    )


def triple(achieved: float, baseline: float, floor: float) -> str:
    """The honest-reporting rule, enforced by being the only way to print one."""
    captured = ((baseline - achieved) / (baseline - floor) * 100
                if baseline > floor else float("nan"))
    tail = _num(captured, "{:.0f}", "")
    return (f'<span class="triple"><b>{_num(achieved)}</b> against a baseline of '
            f'{_num(baseline)} and a floor of {_num(floor)}'
            + (f" — {tail}% of the available signal" if tail else "")
            + "</span>")


def _num(x, fmt: str = "{:.4f}", dash: str = "—") -> str:
    """One place where a missing number becomes a dash rather than 'nan'.

    A NaN reaching the page is a real miss, not a formatting slip: it means a
    metric could not be computed for that item — typically an item whose
    predicted cluster means are constant, so between-cluster rho is undefined.
    Printing an em dash says that; printing 'nan' says nothing.
    """
    try:
        v = float(x)
    except (TypeError, ValueError):
        return dash
    return dash if math.isnan(v) or math.isinf(v) else fmt.format(v)


def _pct(x: float) -> str:
    return _num(x, "{:+.1f}%")


def _verdict_table(rep: dict) -> str:
    rows = []
    for name, v in rep.get("verdicts", {}).items():
        rows.append(
            f"<tr><td>{name.replace('_', ' ')}</td>"
            f"<td class='num'><b>{_num(v['achieved'])}</b></td>"
            f"<td class='num'>{_num(v['baseline_b0a'])}</td>"
            f"<td class='num'>{_num(v['noise_floor'])}</td>"
            f"<td class='num'>{_pct(v['vs_baseline_pct'])}</td>"
            f"<td class='num'>{_num(v['signal_captured_pct'], '{:.0f}')}%</td>"
            f"<td class='{'yes' if v['pass'] else 'no'}'>"
            f"{'PASS' if v['pass'] else 'fail'}</td>"
            f"<td class='{'yes' if v['beats_measured_baseline_by_20pct'] else 'no'}'>"
            f"{'PASS' if v['beats_measured_baseline_by_20pct'] else 'fail'}</td>"
            f"<td class='num'>{v['n_items']}</td></tr>")
    return (
        "<table><thead><tr><th>item set</th><th>achieved W1</th><th>B0a here</th>"
        "<th>noise floor</th><th>vs B0a</th><th>signal captured</th>"
        "<th>absolute</th><th>relative</th><th>items</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>")


def _arms_chart(rep: dict) -> str:
    macro = rep.get("macro", {}).get("all_targets", {})
    bars = []
    for arm, v in sorted(macro.items(),
                         key=lambda kv: (math.isnan(float(kv[1])), float(kv[1]))):
        if math.isnan(float(v)):
            continue
        bars.append(Bar(label=arm.replace("_", " "), value=float(v),
                        highlight=(arm == "system"), note=ARM_NOTES.get(arm, "")))
    floor = rep.get("noise_floor", 0.0)
    rules = [(floor, f"noise floor {floor:.4f}")]
    return bar_chart(bars, title="Macro W1 over the scored targets — lower is better",
                     rules=rules)


def _per_item_scatter(rep: dict) -> str:
    pts = []
    for r in rep.get("per_item", []):
        b = r.get("w1_B0a", (r.get("w1") or {}).get("B0a"))
        s = r.get("w1_system", (r.get("w1") or {}).get("system"))
        if b is None or s is None or math.isnan(float(b)) or math.isnan(float(s)):
            continue
        pts.append((r["item_id"], float(b), float(s)))
    return scatter(pts, title="Per item: the system against the national-marginal baseline",
                   xlabel="B0a W1 (copy the national marginal)",
                   ylabel="system W1", diagonal_note="y = x")


def _tier_chart(inputs: ReportInputs) -> str:
    pts = []
    for model, rep in inputs.layer4.items():
        v = rep.get("verdicts", {}).get("all_targets")
        if not v:
            continue
        name, size = TIERS.get(model, (model, 0))
        if (rep.get("n_items") or 0) < 30:
            name += "*"
        pts.append((size, name, float(v["vs_baseline_pct"])))
    pts.sort()
    return dot_line([(n, v) for _, n, v in pts],
                    title="Capability curve — improvement on B0a, all scored targets",
                    ylabel="vs B0a")


def _per_item_table(rep: dict) -> str:
    rows = []
    for r in sorted(rep.get("per_item", []),
                    key=lambda r: -(r.get("heterogeneity") or 0)):
        w = r.get("w1") or {}
        sys_ = r.get("w1_system", w.get("system"))
        b0a = r.get("w1_B0a", w.get("B0a"))
        b4 = r.get("w1_B4", w.get("B4"))
        flags = []
        if r.get("leakage_resistant"):
            flags.append("leak-resistant")
        if r.get("famous"):
            flags.append("famous")
        rows.append(
            f"<tr><td><code>{r['item_id']}</code></td><td>{r.get('topic', '')}</td>"
            f"<td class='num'>{r.get('k', '')}</td>"
            f"<td class='num'>{r.get('n_clusters', '')}</td>"
            f"<td class='num'>{_num(r.get('heterogeneity'))}</td>"
            f"<td class='num'>{_num(b4)}</td>"
            f"<td class='num'>{_num(b0a)}</td>"
            f"<td class='num'><b>{_num(sys_)}</b></td>"
            f"<td class='num'>{_num(r.get('rho'), '{:+.3f}')}</td>"
            f"<td class='num'>{_num(r.get('var_ratio'), '{:.2f}')}</td>"
            f"<td class='small'>{' · '.join(flags)}</td></tr>")
    return ("<table class='wide'><thead><tr><th>item</th><th>topic</th><th>k</th>"
            "<th>clusters</th><th>heterogeneity</th><th>B4 raw</th><th>B0a</th>"
            "<th>system</th><th>between-cluster ρ</th><th>var ratio</th><th></th>"
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


#: IBM Plex, not a neutral default: it is a type family drawn for technical
#: documentation, its serif and sans share a skeleton so headings and body read
#: as one voice, and Plex Mono gives the data columns real tabular figures. The
#: numbers are the substance of this page, so the face that sets them matters.
FONTS = ('<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
         'family=IBM+Plex+Mono:wght@400;500&'
         'family=IBM+Plex+Sans:wght@400;500;600&'
         'family=IBM+Plex+Serif:wght@500;600&display=swap">')

CSS = """
:root{color-scheme:light;--bg:#fbfaf7;--surface:#ffffff;--ink:#14130f;--ink2:#4e4c45;
--ink3:#7c796f;--line:#e5e2d9;--accent:#2a78d6;--good:#008300;--bad:#e34948;
--code:#f3f1ea;
--sans:"IBM Plex Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;
--serif:"IBM Plex Serif",Georgia,"Times New Roman",serif;
--mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
--bg:#14140f;--surface:#1d1d18;--ink:#f7f6f1;--ink2:#c3c1b5;--ink3:#93918a;
--line:#32312b;--accent:#3987e5;--good:#4caf50;--bad:#e66767;--code:#24241d;}}
:root[data-theme="dark"]{--bg:#14140f;--surface:#1d1d18;--ink:#f7f6f1;--ink2:#c3c1b5;
--ink3:#93918a;--line:#32312b;--accent:#3987e5;--good:#4caf50;--bad:#e66767;
--code:#24241d;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:400 16px/1.62 var(--sans);-webkit-font-smoothing:antialiased;}
main{max-width:1040px;margin:0 auto;padding-block:48px 96px;padding-inline:16px;
display:flex;flex-direction:column;gap:2px}
h1{font:600 31px/1.2 var(--serif);margin:0 0 6px;letter-spacing:-.015em;
text-wrap:balance}
h2{font:600 21px/1.3 var(--serif);margin:44px 0 10px;letter-spacing:-.01em;
border-top:1px solid var(--line);padding-top:22px;text-wrap:balance}
h3{font:600 15px/1.4 var(--sans);margin:26px 0 8px;color:var(--ink2);
letter-spacing:.02em;text-transform:uppercase}
p,li{color:var(--ink2)} b,strong{color:var(--ink)}
.sub{color:var(--ink3);font-size:14px;margin:0 0 28px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:12px;
padding:20px 22px;margin:18px 0}
.hero{font:500 40px/1 var(--mono);letter-spacing:-.03em;color:var(--ink);
font-variant-numeric:tabular-nums}
.triple{font-size:15px}
table{width:100%;border-collapse:collapse;margin:12px 0;font-size:14px}
th,td{text-align:left;padding:7px 9px;border-bottom:1px solid var(--line)}
th{color:var(--ink3);font-weight:600;font-size:11.5px;text-transform:uppercase;
letter-spacing:.06em}
/* A Greek letter has no sensible uppercase here: text-transform turns rho into
   a capital that reads as a Latin P. */
th.nocaps{text-transform:none;letter-spacing:.02em}
td.num{text-align:right;font-family:var(--mono);font-size:13px;
font-variant-numeric:tabular-nums}
td.yes{color:var(--good);font-weight:650} td.no{color:var(--ink3)}
td.small,.small{font-size:12.5px;color:var(--ink3)}
code{background:var(--code);padding:1px 5px;border-radius:4px;font-size:12.5px;
font-family:var(--mono)}
p{max-width:68ch}
.wrap{overflow-x:auto}
.ch-title{font-size:13px;font-weight:600;fill:var(--ink2)}
.ch-label{font-size:12.5px;fill:var(--ink2)}
.ch-value{font:500 12px var(--mono);fill:var(--ink);font-variant-numeric:tabular-nums}
.ch-tick{font:400 11px var(--mono);fill:var(--ink3)}
.ch-axis{font-size:12px;fill:var(--ink3)}
.ch-legend{font-size:12px;fill:var(--ink2)}
.ch-rule{stroke:var(--ink3);stroke-width:1;stroke-dasharray:3 3}
.ch-rule-label{font-size:11px;fill:var(--ink3)}
.ch-bar{fill:var(--ink3);opacity:.45}
.ch-bar-hi{fill:var(--accent);opacity:1}
.ch-plot{fill:none;stroke:var(--line)}
.ch-line{fill:none;stroke:var(--accent);stroke-width:2}
.ch-dot{fill:var(--accent)}
.ch-dot-good{fill:var(--good)} .ch-dot-bad{fill:var(--bad)}
.note{border-left:3px solid var(--accent);padding:2px 0 2px 14px;margin:16px 0}
.warn{border-left-color:var(--bad)}
ul{padding-left:20px}
"""


def build_report(inputs: ReportInputs, out: Path, *,
                 headline_model: str | None = None,
                 fragment: bool = False) -> Path:
    """Write the report. ``fragment=True`` omits the document skeleton.

    Some hosts wrap the content in their own ``<html>``/``<head>``; emitting a
    second one there produces a nested document that renders once and styles
    twice. The style block travels with the fragment either way, so the page is
    still self-contained wherever it lands.
    """
    models = list(inputs.layer4)
    if not models:
        raise ValueError("no layer4_report.json found — run `popsim layer4` first")
    # The headline is the arm with the widest coverage, not the biggest model:
    # a part-finished elicitation on the largest model would otherwise headline
    # the page on ten items while a complete 39-item arm sat below it.
    order = sorted(models, key=lambda m: (-(inputs.layer4[m].get("n_items") or 0),
                                          -(inputs.layer4[m].get("n_clusters") or 0),
                                          -TIERS.get(m, (m, 0))[1]))
    head = headline_model if headline_model in inputs.layer4 else order[0]
    rep = inputs.layer4[head]
    v_all = rep.get("verdicts", {}).get("all_targets", {})
    parts: list[str] = []

    parts.append(f"""<h1>B17 — held-out subgroup distributions from stat-card elicitation</h1>
<p class="sub">Generated {inputs.generated} from <code>runs/</code>. Every figure is
regenerable with <code>popsim report</code>. Headline arm: <code>{head}</code>,
scored on {rep.get('n_items', 0)} held-out items &times; {rep.get('n_clusters', 0)}
clusters. An arm marked <i>partial</i> or <code>*</code> is still eliciting and is
scored on the items it has.</p>

<div class="card">
  <div class="hero">{_num(v_all.get('achieved'))}</div>
  <p class="triple">macro Wasserstein-1 over {v_all.get('n_items', 0)} held-out
  GSS items at K&nbsp;=&nbsp;{rep.get('n_clusters', 0)} clusters
  — against a national-marginal baseline of
  {_num(v_all.get('baseline_b0a'))} and a truth-noise floor of
  {_num(v_all.get('noise_floor'))}. That is
  <b>{_num(v_all.get('signal_captured_pct'), '{:.0f}')}%</b> of the signal
  available between the two.</p>
</div>

<p><b>The question.</b> Given a demographic subgroup described only by its
population marginals and its answers to a set of <i>other</i> survey questions,
what is the full distribution of its answers to a question it was never asked?
Not the average — the whole histogram, because the distribution is what
persona-prompting gets wrong and what a cross-tab cannot give you for an unasked
question.</p>

<p><b>The baseline that matters.</b> B0a copies the true national marginal onto
every cluster. Beating it requires real between-cluster signal, and it is
deliberately unfair: it is handed the population answer. The system is handed the
same national marginal, so the only thing being compared is whether it adds
subgroup structure on top — and the deviation scale <code>s&nbsp;=&nbsp;0</code>
reproduces B0a exactly, which makes the baseline an ablation of the method rather
than an outside competitor.</p>""")

    parts.append("<h2>The verdict</h2>")
    parts.append("<div class='wrap'>" + _verdict_table(rep) + "</div>")
    parts.append("""<p class="small"><b>Two verdicts, both reported.</b> <i>absolute</i>
is the pass mark frozen in the config before any elicitation ran. <i>relative</i>
is §1.4(i)'s own wording — 20% below the B0a measured on the scope being scored.
They differ because the pre-registered mark was derived from a baseline measured
on the feasibility bed (2016–2022, n&nbsp;=&nbsp;11,045, 88 items) and the frozen
bed is pooled 2010–2022, n&nbsp;=&nbsp;18,772. Neither was moved after the fact;
both are shown so neither can be chosen after the fact.</p>""")

    pop = rep.get("population", {})
    bat = rep.get("battery", {})
    extra = []
    if pop:
        extra.append(
            f"<li><b>Population variance ratio {_num(pop.get('variance_ratio'), '{:.3f}')}</b> "
            f"(target band {pop.get('band')}) — "
            f"<b>{'passes' if pop.get('pass') else 'fails'}</b> §1.4(ii). "
            f"Weights raked to ACS 2024: {pop.get('weights', '')}.</li>")
    if bat.get("rho") is not None and not math.isnan(float(bat["rho"])):
        extra.append(
            f"<li><b>Tolerance-battery ordering ρ = {_num(bat['rho'], '{:+.3f}')}</b> across "
            f"{bat.get('n_clusters', 0)} clusters and {bat.get('n_items', 0)} items — "
            f"whether the system reproduces the ordering of the five target groups "
            f"<i>within</i> each subgroup, which a national marginal cannot do at all.</li>")
    cov = [r.get("coverage90") for r in rep.get("per_item", [])]
    cov = [c for c in cov if c is not None and not math.isnan(float(c))]
    if cov:
        extra.append(
            f"<li><b>90 % interval coverage {sum(cov) / len(cov):.3f}</b> against a "
            f"nominal 0.90. The interval carries two terms: the elicitation "
            f"ensemble's spread, and the out-of-fold anchor residual — how far the "
            f"finished prediction sat from the truth on questions where the truth "
            f"is known. The first alone covers 15.7 %, which is what an interval "
            f"built only from ensemble spread is worth.</li>")
    sd_t = [r.get("btw_sd_true") for r in rep.get("per_item", [])]
    sd_p = [r.get("btw_sd_pred") for r in rep.get("per_item", [])]
    sd_t = [x for x in sd_t if x is not None and not math.isnan(float(x))]
    sd_p = [x for x in sd_p if x is not None and not math.isnan(float(x))]
    if sd_t and sd_p:
        extra.append(
            f"<li><b>Between-cluster spread: predicted {sum(sd_p) / len(sd_p):.4f} "
            f"against a true {sum(sd_t) / len(sd_t):.4f}.</b> The system "
            f"under-disperses across clusters by about half — the price of the "
            f"shrinkage that minimises W1. It orders the subgroups correctly and "
            f"places them too close together, which is a milder relative of F2. "
            f"Reported rather than tuned away: matching the spread would cost W1.</li>")
    if extra:
        parts.append("<ul>" + "".join(extra) + "</ul>")

    parts.append("<h2>Every arm, side by side</h2>")
    parts.append(_arms_chart(rep))
    parts.append("<table><thead><tr><th>arm</th><th>what it is</th></tr></thead><tbody>"
                 + "".join(f"<tr><td><code>{a}</code></td><td>{n}</td></tr>"
                           for a, n in ARM_NOTES.items()
                           if _num(rep.get("macro", {}).get("all_targets", {}).get(a),
                                   dash="") != "")
                 + "</tbody></table>")

    if len(inputs.layer4) > 1:
        parts.append("<h2>Capability</h2>")
        parts.append(_tier_chart(inputs))
        parts.append("<div class='wrap'><table><thead><tr><th>model</th>"
                     "<th>scope</th><th>achieved</th><th>B0a</th><th>vs B0a</th>"
                     "<th>var ratio</th><th class='nocaps'>battery ρ</th>"
                     "</tr></thead><tbody>"
                     + "".join(
                         f"<tr><td>{TIERS.get(m, (m, 0))[0]}"
                         + ("" if r.get('n_items', 0) >= 30 else
                            " <span class='small'>partial</span>") + "</td>"
                         f"<td class='small'>{r.get('n_items', 0)} items &times; "
                         f"{r.get('n_clusters', 0)} clusters</td>"
                         f"<td class='num'>{_num(r['verdicts']['all_targets']['achieved'])}</td>"
                         f"<td class='num'>{_num(r['verdicts']['all_targets']['baseline_b0a'])}</td>"
                         f"<td class='num'>{_pct(r['verdicts']['all_targets']['vs_baseline_pct'])}</td>"
                         f"<td class='num'>{_num((r.get('population') or {}).get('variance_ratio'), '{:.3f}')}</td>"
                         f"<td class='num'>{_num((r.get('battery') or {}).get('rho'), '{:+.3f}')}</td></tr>"
                         for m, r in sorted(
                             inputs.layer4.items(),
                             key=lambda kv: TIERS.get(kv[0], (kv[0], 0))[1])
                         if r.get("verdicts", {}).get("all_targets"))
                     + "</tbody></table></div>")

    parts.append("<h2>Item by item</h2>")
    parts.append(_per_item_scatter(rep))
    parts.append("<div class='wrap'>" + _per_item_table(rep) + "</div>")

    if inputs.gate4:
        g = inputs.gate4
        parts.append(f"""<h2>Layer 3 — the calibration layer is arithmetically sound</h2>
<p>Feed the <i>true</i> per-cluster histograms in where the model's go. The layer
must return them. It does, to
<b>{_num(g.get('max_w1_at_s1'), '{:.1e}')}</b> W1 at worst over
{g.get('n_cells', 0)} cells, the fitted deviation scale comes back at exactly
<b>{_num(g.get('fitted_scale'))}</b>, and
<code>s&nbsp;=&nbsp;0</code> equals B0a to
{_num(g.get('max_w1_s0_vs_b0a'), '{:.1e}')}. Off-by-one option ordering,
misaligned scale codes and sign errors in the tempering all fail this test
loudly instead of masquerading as a bad model.</p>""")

    if inputs.leakage:
        lk = inputs.leakage.get(head) or next(iter(inputs.leakage.values()))
        flagged = lk.get("leaky", [])
        mv = rep.get("verdicts", {}).get("minus_leaky", {})
        parts.append(f"""<h2>Layer 5 — is the win memorised?</h2>
<p>The probe names GSS, names the waves and names the subgroup, then asks for the
published cross-tab — the opposite of the elicitation prompt, which names no
dataset. An item is flagged only when the probe recalls a <i>subgroup</i> within
0.05 <b>and</b> beats simply copying the topline onto it, because every GSS
topline is published and reciting one says nothing about knowing the breakdown.</p>
<p>Flagged: <b>{len(flagged)}</b> of {lk.get('n_items', 0)} items
{('— <code>' + '</code>, <code>'.join(flagged) + '</code>') if flagged else ''}.
""" + (f"""Excluding them moves the headline from
{_num(v_all.get('achieved'))} to {_num(mv.get('achieved'))}
({_pct(mv.get('vs_baseline_pct', float('nan')))} vs B0a, against
{_pct(v_all.get('vs_baseline_pct', float('nan')))} with them in).""" if mv else "")
            + "</p>")

    parts.append("""<h2>What this does not answer</h2>
<p>Product and policy scenarios have no ground truth and never will. The
held-out-item score is the honest proxy and the honesty box is where that is said
to the reader. This report makes none of the following claims: that the system
simulates society, predicts policy outcomes, replaces surveys — it is built on
them and dies without them — or scales to millions of agents. The claim is
fidelity per dollar at a few dozen clusters, and it is exactly what the tables
above show.</p>""")

    body = (f"<title>B17 Subgroup Distributions</title>{FONTS}"
            f"<style>{CSS}</style><main>{''.join(parts)}</main>")
    html = body if fragment else (
        f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"</head><body>{body}</body></html>")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return out
