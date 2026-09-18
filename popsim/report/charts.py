"""Inline SVG charts for the results report.

Self-contained on purpose: the report has to open from a thesis folder, an email
attachment or a USB stick with no network, so there is no chart library and no
font download. Every figure is regenerable from `runs/` (spec §M10).

Colours come from a validated categorical palette; both light and dark steps are
declared so the report is readable either way. Marks follow the house rules the
figures need: one axis only, thin marks, recessive grid, a legend whenever two
things are being distinguished, and text in ink rather than in the series colour.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["PALETTE", "bar_chart", "dot_line", "scatter", "svg_open"]

PALETTE = {
    "series1": ("#2a78d6", "#3987e5"),
    "series2": ("#eb6834", "#d95926"),
    "series3": ("#1baf7a", "#199e70"),
    "good": ("#008300", "#008300"),
    "bad": ("#e34948", "#e66767"),
}


def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def svg_open(w: int, h: int, title: str) -> str:
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" height="auto" '
            f'role="img" aria-label="{_esc(title)}" '
            f'style="max-width:{w}px;font-family:inherit">')


@dataclass
class Bar:
    label: str
    value: float
    highlight: bool = False
    note: str = ""


def bar_chart(bars: list[Bar], *, title: str, unit: str = "",
              rules: list[tuple[float, str]] | None = None,
              width: int = 720, row: int = 30, pad_left: int = 190) -> str:
    """Horizontal bars, sorted by the caller. Direct-labelled, no legend needed.

    Horizontal because the labels are arm names, not dates: a vertical bar chart
    with "system without the subspace" under it is a rotated-label chart, which
    is on every anti-pattern list for good reason.
    """
    rules = rules or []
    vmax = max([b.value for b in bars] + [r[0] for r in rules] + [1e-9]) * 1.18
    h = row * len(bars) + 54
    plot_w = width - pad_left - 74
    out = [svg_open(width, h, title)]
    out.append(f'<text x="0" y="14" class="ch-title">{_esc(title)}</text>')
    for x, lab in rules:
        px = pad_left + plot_w * x / vmax
        out.append(f'<line x1="{px:.1f}" y1="26" x2="{px:.1f}" y2="{h - 26}" '
                   f'class="ch-rule"/>')
        out.append(f'<text x="{px:.1f}" y="{h - 10}" class="ch-rule-label" '
                   f'text-anchor="middle">{_esc(lab)}</text>')
    for i, b in enumerate(bars):
        y = 34 + i * row
        bw = max(plot_w * b.value / vmax, 1.5)
        cls = "ch-bar ch-bar-hi" if b.highlight else "ch-bar"
        out.append(f'<text x="{pad_left - 8}" y="{y + 13}" class="ch-label" '
                   f'text-anchor="end">{_esc(b.label)}</text>')
        out.append(f'<rect x="{pad_left}" y="{y}" width="{bw:.1f}" height="17" '
                   f'rx="4" class="{cls}"><title>{_esc(b.label)}: '
                   f'{b.value:.4f}{_esc(unit)}{" — " + _esc(b.note) if b.note else ""}'
                   f'</title></rect>')
        out.append(f'<text x="{pad_left + bw + 7:.1f}" y="{y + 13}" '
                   f'class="ch-value">{b.value:.4f}</text>')
    out.append("</svg>")
    return "\n".join(out)


def dot_line(points: list[tuple[str, float]], *, title: str, ylabel: str = "",
             width: int = 620, height: int = 240, zero_rule: bool = True,
             fmt: str = "{:+.1f}%") -> str:
    """One series over an ordinal axis. No legend — the title names the series.

    The right-hand gutter is sized from the longest label, and the first and last
    labels anchor to their own ends rather than to their points, so a long series
    name at either extreme stays inside the viewBox instead of being clipped by
    it — which is what happened to "Ministral 14B*" at the first draw.
    """
    if not points:
        return ""
    vals = [v for _, v in points]
    lo, hi = min(vals + ([0.0] if zero_rule else [])), max(vals + ([0.0] if zero_rule else []))
    span = (hi - lo) or 1.0
    lo, hi = lo - span * 0.18, hi + span * 0.18
    longest = max(len(lab) for lab, _ in points)
    gutter = max(24, int(longest * 3.4))
    pad_l, pad_b, pad_t = 62, 40, 30
    pw, ph = width - pad_l - gutter, height - pad_b - pad_t

    def X(i):
        return pad_l + (pw * i / max(len(points) - 1, 1))

    def Y(v):
        return pad_t + ph * (hi - v) / (hi - lo)

    out = [svg_open(width, height, title)]
    out.append(f'<text x="0" y="14" class="ch-title">{_esc(title)}</text>')
    if zero_rule and lo < 0 < hi:
        out.append(f'<line x1="{pad_l}" y1="{Y(0):.1f}" x2="{pad_l + pw}" '
                   f'y2="{Y(0):.1f}" class="ch-rule"/>')
        out.append(f'<text x="{pad_l - 8}" y="{Y(0) + 4:.1f}" class="ch-rule-label" '
                   f'text-anchor="end">0</text>')
    d = " ".join(f"{'M' if i == 0 else 'L'}{X(i):.1f},{Y(v):.1f}"
                 for i, (_, v) in enumerate(points))
    out.append(f'<path d="{d}" class="ch-line"/>')
    for i, (lab, v) in enumerate(points):
        anchor = ("start" if i == 0 and len(points) > 1 else
                  "end" if i == len(points) - 1 and len(points) > 1 else "middle")
        lx = X(i) + (-6 if anchor == "start" else 6 if anchor == "end" else 0)
        out.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="5" class="ch-dot">'
                   f'<title>{_esc(lab)}: {fmt.format(v)}</title></circle>')
        out.append(f'<text x="{X(i):.1f}" y="{Y(v) - 12:.1f}" class="ch-value" '
                   f'text-anchor="middle">{fmt.format(v)}</text>')
        out.append(f'<text x="{lx:.1f}" y="{height - 14}" class="ch-label" '
                   f'text-anchor="{anchor}">{_esc(lab)}</text>')
    if ylabel:
        out.append(f'<text x="14" y="{pad_t + ph / 2:.0f}" class="ch-axis" '
                   f'text-anchor="middle" '
                   f'transform="rotate(-90 14 {pad_t + ph / 2:.0f})">'
                   f'{_esc(ylabel)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def scatter(points: list[tuple[str, float, float]], *, title: str,
            xlabel: str, ylabel: str, width: int = 520, height: int = 460,
            diagonal_note: str = "") -> str:
    """Per-item: the system against the baseline, with the y = x reference.

    Two encodings, so a legend is present: colour says which side of the diagonal
    the item fell on, and the diagonal itself is labelled.
    """
    if not points:
        return ""
    vals = [v for _, a, b in points for v in (a, b)]
    hi = max(vals) * 1.1
    pad_l, pad_b, pad_t, pad_r = 74, 46, 34, 16
    pw, ph = width - pad_l - pad_r, height - pad_b - pad_t

    def X(v):
        return pad_l + pw * v / hi

    def Y(v):
        return pad_t + ph * (1 - v / hi)

    out = [svg_open(width, height, title)]
    out.append(f'<text x="0" y="14" class="ch-title">{_esc(title)}</text>')
    out.append(f'<rect x="{pad_l}" y="{pad_t}" width="{pw}" height="{ph}" class="ch-plot"/>')
    out.append(f'<line x1="{X(0)}" y1="{Y(0)}" x2="{X(hi)}" y2="{Y(hi)}" class="ch-rule"/>')
    if diagonal_note:
        out.append(f'<text x="{X(hi) - 6:.0f}" y="{Y(hi) + 16:.0f}" '
                   f'class="ch-rule-label" text-anchor="end">{_esc(diagonal_note)}</text>')
    for name, x, y in points:
        cls = "ch-dot-good" if y < x else "ch-dot-bad"
        out.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="4.5" class="{cls}">'
                   f'<title>{_esc(name)}: baseline {x:.4f}, system {y:.4f}</title></circle>')
    out.append(f'<text x="{pad_l + pw / 2:.0f}" y="{height - 12}" class="ch-axis" '
               f'text-anchor="middle">{_esc(xlabel)}</text>')
    out.append(f'<text x="18" y="{pad_t + ph / 2:.0f}" class="ch-axis" '
               f'text-anchor="middle" transform="rotate(-90 18 {pad_t + ph / 2:.0f})">'
               f'{_esc(ylabel)}</text>')
    for frac in (0.0, 0.5, 1.0):
        v = hi * frac
        out.append(f'<text x="{X(v):.0f}" y="{height - 30}" class="ch-tick" '
                   f'text-anchor="middle">{v:.2f}</text>')
        out.append(f'<text x="{pad_l - 8}" y="{Y(v) + 4:.0f}" class="ch-tick" '
                   f'text-anchor="end">{v:.2f}</text>')
    g, b = PALETTE["good"][0], PALETTE["bad"][0]
    out.append(f'<g transform="translate({pad_l + 8},{pad_t + 14})">'
               f'<circle cx="0" cy="0" r="4.5" fill="{g}"/>'
               f'<text x="10" y="4" class="ch-legend">system better</text>'
               f'<circle cx="0" cy="18" r="4.5" fill="{b}"/>'
               f'<text x="10" y="22" class="ch-legend">baseline better</text></g>')
    out.append("</svg>")
    return "\n".join(out)
