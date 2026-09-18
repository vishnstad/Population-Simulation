"""
Figures for the B17 GSS 2024 preprocessing run.

Palette: validated categorical slots 1-4 (blue / orange / aqua / yellow),
sequential blue ramp, from the dataviz reference instance. Aqua and yellow sit
below 3:1 on the light surface, so every chart using them carries visible
direct labels (the relief rule).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#83827c"
GRID = "#e6e5e1"

S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SERIES = [S1, S2, S3, S4]
STATUS_BAD = "#e34948"

SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ)


def use_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": GRID,
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def _title(ax, title, subtitle=None, lift=0):
    """Title + subtitle stacked above the axes, offset in POINTS so the gap is
    identical regardless of figure height (axes-fraction offsets collapse on
    tall figures and the two lines overlap)."""
    ax.set_title("", loc="left")
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction",
                    textcoords="offset points", xytext=(0, 12 + lift),
                    ha="left", va="bottom", fontsize=9.5, color=INK_2)
        ax.annotate(title, xy=(0, 1), xycoords="axes fraction",
                    textcoords="offset points", xytext=(0, 28 + lift),
                    ha="left", va="bottom", fontsize=12.5,
                    fontweight="600", color=INK)
    else:
        ax.annotate(title, xy=(0, 1), xycoords="axes fraction",
                    textcoords="offset points", xytext=(0, 12),
                    ha="left", va="bottom", fontsize=12.5,
                    fontweight="600", color=INK)


def _fig_title(fig, ax_left, title, subtitle):
    """Same, anchored to the leftmost axes of a multi-panel figure."""
    _title(ax_left, title, subtitle)


_LEVEL_ORDER = {
    "age_band": ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
    "education": ["none", "primary", "secondary", "higher_secondary", "tertiary"],
    "income_band": ["q1", "q2", "q3", "q4", "q5"],
    "srcbelt": ["lg_central_city", "md_central_city", "lg_suburb", "md_suburb",
                "other_urban", "other_rural"],
}


def _ordered_levels(field, present):
    """Ordinal fields keep their substantive order; everything else falls back
    to alphabetical. Plotting `higher_secondary` between `primary` and
    `secondary` because 'h' < 'p' would misrepresent an ordered variable."""
    order = _LEVEL_ORDER.get(field)
    if order:
        return [v for v in order if v in present] + \
               sorted((v for v in present if v not in order), key=str)
    return sorted(present, key=str)


def _clean(ax, y_only=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if y_only:
        ax.xaxis.grid(False)


# ---------------------------------------------------------------------------
# 1. K vs scoreable cells -- the headline constraint
# ---------------------------------------------------------------------------

def min_cell_of(k: int, sweep: pd.DataFrame) -> int:
    """min_cell for the main run, inferred by interpolating the sweep (the
    driver passes K, and K is monotone decreasing in min_cell)."""
    d = sweep.sort_values("K")
    return int(np.interp(k, d["K"], d["min_cell"]))


def fig_k_vs_scoreable(sweep: pd.DataFrame, chosen_k: int, chosen_pct: float,
                       min_neff: int, path: str, min_cell: int | None = None):
    use_style()
    fig, ax = plt.subplots(figsize=(9.0, 5.4))

    # the main run is just another point on the same curve -- plot one series
    d = pd.concat([
        sweep[["min_cell", "K", "pct_scoreable"]],
        pd.DataFrame([{"min_cell": min_cell if min_cell is not None
                       else min_cell_of(chosen_k, sweep),
                       "K": chosen_k, "pct_scoreable": chosen_pct}]),
    ]).drop_duplicates("K").sort_values("K")
    # a sweep point within 10% of the main run's K would collide with it
    d = d[(d["K"] == chosen_k) | (abs(d["K"] - chosen_k) > 0.1 * chosen_k)]

    ax.axvspan(100, 200, color=STATUS_BAD, alpha=0.07, zorder=0)
    ax.plot(d["K"], d["pct_scoreable"], color=S1, lw=2, zorder=2)
    ax.scatter(d["K"], d["pct_scoreable"], s=70, color=S1, ec=SURFACE, lw=2,
               zorder=3, clip_on=False)
    ax.scatter([chosen_k], [chosen_pct], s=130, color=S2, ec=SURFACE, lw=2,
               zorder=4, clip_on=False)

    for _, r in d.iterrows():
        if int(r.K) == chosen_k:
            continue
        ax.annotate(f"{r.pct_scoreable:.0f}%\nmin_cell {int(r.min_cell)}",
                    (r.K, r.pct_scoreable), textcoords="offset points",
                    xytext=(9, 12), ha="left", fontsize=9, color=INK,
                    fontweight="600", linespacing=1.5)

    ax.annotate(f"main run  ·  K = {chosen_k}\nmin_cell {min_cell}  ·  "
                f"{chosen_pct:.1f}% scoreable",
                (chosen_k, chosen_pct), textcoords="offset points",
                xytext=(12, 26), ha="left", fontsize=9.5, color=S2,
                fontweight="700", linespacing=1.5,
                arrowprops=dict(arrowstyle="-", color=S2, lw=1.2,
                                shrinkA=0, shrinkB=7))

    ax.text(150, 104, "spec target  K ≈ 150", ha="center", va="top",
            fontsize=10, color=STATUS_BAD, fontweight="700")
    ax.text(150, 96, "§1.4 / §2.2", ha="center", va="top",
            fontsize=8.8, color=STATUS_BAD)

    ax.set_xlabel("K  (number of leaf clusters)")
    ax.set_ylabel(f"% of (cluster × item) cells with n_eff ≥ {min_neff}")
    ax.set_ylim(-11, 112)
    ax.set_xlim(0, 215)
    _clean(ax)
    _title(ax, "Cluster count trades directly against scoreable cells",
           f"GSS 2024, n = 3,309, 507 items. §5.2 excludes any cell below "
           f"n_eff = {min_neff} from scoring.")
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Leaf size distribution
# ---------------------------------------------------------------------------

def fig_leaf_sizes(nodes: list[dict], min_neff: int, min_cell: int, path: str):
    use_style()
    leaves = [n for n in nodes if n["level"] == 2]
    raw = np.array([n["n_raw"] for n in leaves])
    neff = np.array([n["n_eff"] for n in leaves])

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), sharey=True)
    for ax, vals, col, lab, ref, refname in (
        (axes[0], raw, S1, "respondents per leaf", min_cell, f"min_cell = {min_cell}"),
        (axes[1], neff, S2, "Kish n_eff per leaf", min_neff, f"scoring gate = {min_neff}"),
    ):
        bins = np.linspace(0, max(vals.max(), ref) * 1.05, 26)
        ax.hist(vals, bins=bins, color=col, edgecolor=SURFACE, linewidth=1.2)
        ax.axvline(ref, color=STATUS_BAD, lw=2, ls=(0, (4, 3)), zorder=5)
        ax.text(ref, ax.get_ylim()[1] * 0.95, f"  {refname}", color=STATUS_BAD,
                fontsize=9.5, fontweight="600", va="top")
        ax.set_xlabel(lab)
        _clean(ax)
        med = np.median(vals)
        ax.text(0.97, 0.95, f"median {med:.0f}\n{100*np.mean(vals>=ref):.0f}% ≥ line",
                transform=ax.transAxes, ha="right", va="top", fontsize=9.5,
                color=INK, fontweight="600")
    axes[0].set_ylabel("number of leaf clusters")
    _title(axes[0], "Leaves clear the size floor; almost none clear the scoring gate",
           f"K = {len(leaves)} leaves. Survey weights push n_eff well below raw n.")
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Heterogeneity: naive vs debiased
# ---------------------------------------------------------------------------

def fig_heterogeneity_bias(split: pd.DataFrame, path: str):
    use_style()
    d = split.dropna(subset=["h_raw", "h_icc"])
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    lim = [min(d.h_icc.min(), 0) - 0.02, max(d.h_raw.max(), d.h_icc.max()) + 0.02]
    ax.plot(lim, lim, color=INK_MUTED, lw=1.4, ls=(0, (4, 3)), zorder=1)
    ax.axhline(0, color=INK_MUTED, lw=1, zorder=1)
    ax.scatter(d.h_raw, d.h_icc, s=26, color=S1, alpha=0.55,
               ec=SURFACE, lw=0.6, zorder=3)

    shrink = float(np.median(1 - d.h_icc / d.h_raw))
    rank_shift = int((d.h_raw.rank(ascending=False) <= len(d) / 4).ne(
        d.h_icc.rank(ascending=False) <= len(d) / 4).sum())
    ax.text(0.035, 0.965,
            f"median item loses {100*shrink:.0f}% of its apparent\n"
            f"heterogeneity once the within-cluster\n"
            f"mean square is subtracted\n\n"
            f"{rank_shift} items change top-quartile membership",
            transform=ax.transAxes, va="top", fontsize=10, color=INK,
            fontweight="600", linespacing=1.45)
    ax.text(lim[1] * 0.90, lim[1] * 0.955, "y = x", color=INK_MUTED,
            fontsize=9.5, ha="right", va="top")

    ax.set_xlabel("h_raw  —  naive weighted between-cluster variance share")
    ax.set_ylabel("h_icc  —  debiased (random-effects ANOVA)")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.xaxis.grid(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    _title(ax, "Naive heterogeneity roughly doubles the real signal",
           f"Each point is one of {len(d)} items. Every point sits below y = x: "
           f"cluster means differ partly because cells are small.")
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Top heterogeneity items
# ---------------------------------------------------------------------------

def fig_top_items(split: pd.DataFrame, codebook: dict, topn: int, path: str):
    use_style()
    d = split.dropna(subset=["h_icc"]).nlargest(topn, "h_icc").iloc[::-1]
    labels = [f"{r.item_id}  ·  {codebook[r.item_id]['text'][:44]}"
              for r in d.itertuples()]
    fig, ax = plt.subplots(figsize=(10.4, 0.34 * topn + 2.0))
    y = np.arange(len(d))
    ax.barh(y, d.h_icc, color=S1, height=0.72)
    for yi, (v, t) in enumerate(zip(d.h_icc, d.topic)):
        ax.text(v + 0.002, yi, f"{v:.3f}   {t}", va="center", fontsize=8.8,
                color=INK_2)
    ax.set_yticks(y, labels, fontsize=8.6)
    ax.set_xlabel("h_icc  —  debiased between-cluster variance share")
    ax.set_xlim(0, d.h_icc.max() * 1.42)
    ax.yaxis.grid(False); ax.xaxis.grid(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    _title(ax, f"The {topn} items where clusters genuinely differ",
           "§1.4 claim (i) is scored on the top-heterogeneity quartile — these "
           "are the items that actually carry subgroup signal.")
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Item coverage (ballot rotation)
# ---------------------------------------------------------------------------

def fig_coverage(codebook: dict, path: str):
    use_style()
    cov = np.array([v["coverage"] for v in codebook.values()])
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    n, _, _ = ax.hist(cov, bins=np.linspace(0, 1, 41), color=S1,
                      edgecolor=SURFACE, linewidth=1.2)
    ax.set_ylim(0, n.max() * 1.22)
    for x, lab in ((1 / 3, "⅓"), (2 / 3, "⅔"), (1.0, "all")):
        ax.axvline(x, color=INK_MUTED, lw=1.3, ls=(0, (3, 3)), zorder=1)
        ax.text(x, n.max() * 1.19, lab, fontsize=9.5, color=INK_2,
                va="top", ha="center", fontweight="600")
    med = float(np.median(cov))
    ax.axvline(med, color=S2, lw=2, zorder=4)
    ax.annotate(f"median item is answered by\n{100*med:.0f}% of the sample",
                xy=(med, n.max() * 0.72), xytext=(12, 0),
                textcoords="offset points", fontsize=9.5, color=S2,
                fontweight="600", va="center")
    ax.set_xlabel("weighted share of the analytic sample answering the item")
    ax.set_ylabel("items")
    _clean(ax)
    _title(ax, "Item coverage is multi-modal — ballot and module assignment",
           f"{len(cov)} retained items. Most are asked of a fraction of "
           f"respondents, thinning every cluster cell by the same factor.")
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Anchor / target split by topic, both regimes
# ---------------------------------------------------------------------------

def fig_split(split: pd.DataFrame, adv_topics: list[str], path: str):
    use_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.6), sharey=True)
    order = split.topic.value_counts().index[::-1]

    for ax, col, name in ((axes[0], "role_standard", "Standard (stratified random)"),
                          (axes[1], "role_adversarial", "Adversarial (whole topics held out)")):
        ct = split.groupby(["topic", col], observed=True).size().unstack(fill_value=0)
        ct = ct.reindex(order).fillna(0)
        y = np.arange(len(ct))
        a = ct.get("anchor", pd.Series(0, index=ct.index)).to_numpy()
        t = ct.get("target", pd.Series(0, index=ct.index)).to_numpy()
        ax.barh(y, a, color=S1, height=0.7, label="anchor")
        ax.barh(y, t, left=a + 0.9, color=S2, height=0.7, label="target")
        for yi, (ai, ti) in enumerate(zip(a, t)):
            if ai: ax.text(ai / 2, yi, str(int(ai)), ha="center", va="center",
                           fontsize=8.4, color="white", fontweight="700")
            if ti: ax.text(ai + ti / 2, yi, str(int(ti)), ha="center", va="center",
                           fontsize=8.4, color="white", fontweight="700")
        ax.set_yticks(y, ct.index, fontsize=9)
        ax.set_xlabel("items")
        ax.yaxis.grid(False); ax.xaxis.grid(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        if col == "role_adversarial":
            for yi, tp in enumerate(ct.index):
                if tp in adv_topics:
                    ax.text(-0.012, yi, "▸", transform=ax.get_yaxis_transform(),
                            ha="right", va="center", color=S2, fontsize=11)

    axes[0].legend(loc="lower right", fontsize=9.5)
    for a in axes:
        a.title.set_visible(False)
    _title(axes[0], "Anchor / target split under both regimes",
           "60/40 anchor/target. The adversarial regime holds out entire "
           "topics (▸) to measure out-of-domain transfer.")
    axes[0].annotate("Standard (stratified random)", xy=(0, 1),
                     xycoords="axes fraction", textcoords="offset points",
                     xytext=(0, -2), ha="left", va="bottom", fontsize=10.5,
                     color=INK, fontweight="600")
    axes[1].annotate("Adversarial (whole topics held out)", xy=(0, 1),
                     xycoords="axes fraction", textcoords="offset points",
                     xytext=(0, -2), ha="left", va="bottom", fontsize=10.5,
                     color=INK, fontweight="600")
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Between-cluster spread on real items
# ---------------------------------------------------------------------------

def fig_between_cluster(cstats: pd.DataFrame, pstats: pd.DataFrame,
                        split: pd.DataFrame, codebook: dict,
                        min_neff: int, path: str, n_items: int = 14,
                        k_label: str = "", het_col: str = "h_icc_coarse"):
    """Cluster means as deviations from the national marginal, expressed as a
    share of each item's response range so all items share one axis. This is
    exactly the quantity §1.4 claim (i) is about: how far real subgroups sit
    from 'everyone answers like the nation'."""
    use_style()
    ok = cstats[cstats.n_eff >= min_neff].copy()
    counts = ok.groupby("item_id").size()
    hc = het_col if het_col in split.columns else "h_icc"

    cand = (split.dropna(subset=[hc])
            .assign(n_cells=split.item_id.map(counts).fillna(0))
            .query("n_cells >= 5")
            .nlargest(n_items, hc)
            .iloc[::-1])

    pop = pstats.set_index("item_id")
    fig, ax = plt.subplots(figsize=(10.0, 0.42 * len(cand) + 2.4))

    ylabels = []
    for yi, r in enumerate(cand.itertuples()):
        sub = ok[ok.item_id == r.item_id]
        codes = codebook[r.item_id]["scale"]["codes"]
        rng_ = max(codes) - min(codes)
        dev = 100 * (sub["mean"].to_numpy() - pop.loc[r.item_id, "mean"]) / rng_
        jit = np.random.default_rng(11 + yi).normal(0, 0.085, len(dev))
        ax.scatter(dev, np.full(len(dev), yi) + jit, s=44, color=S1, alpha=0.7,
                   ec=SURFACE, lw=0.7, zorder=3)
        ax.plot([dev.min(), dev.max()], [yi, yi], color=S1, lw=1.2, alpha=0.35,
                zorder=2)
        ax.text(dev.max() + 1.4, yi, f"spread {dev.max()-dev.min():.0f} pts",
                va="center", fontsize=8.4, color=INK_2)
        ylabels.append(f"{r.item_id}  ·  {codebook[r.item_id]['text'][:40]}")

    ax.axvline(0, color=S2, lw=2.2, zorder=4)
    ax.text(0.6, len(cand) - 0.35, "national marginal (B0a)", color=S2,
            fontsize=9.5, fontweight="700", va="center")

    ax.set_yticks(np.arange(len(cand)), ylabels, fontsize=8.6)
    ax.set_ylim(-0.8, len(cand) - 0.2)
    ax.set_xlabel("cluster mean − national mean,  as % of the item's response range")
    ax.yaxis.grid(False); ax.xaxis.grid(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    _title(ax, "What claim (i) has to beat",
           f"Each dot is one cluster with n_eff ≥ {min_neff}{k_label}. "
           f"Highest-heterogeneity items with ≥5 scoreable cells.")
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 8. Weighted vs unweighted demographics
# ---------------------------------------------------------------------------

def fig_weighting(table: pd.DataFrame, path: str):
    use_style()
    fields = ["age_band", "education", "region", "income_band"]
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 4.6))
    for ax, f in zip(axes, fields):
        col = table[f"demo_{f}"]
        m = col.notna()
        uw = col[m].value_counts(normalize=True)
        ww = (table.loc[m].groupby(col[m], observed=True)["weight"].sum())
        ww = ww / ww.sum()
        idx = _ordered_levels(f, set(uw.index) | set(ww.index))
        x = np.arange(len(idx))
        u = [100 * uw.get(i, 0) for i in idx]
        w = [100 * ww.get(i, 0) for i in idx]
        ax.bar(x - 0.21, u, width=0.38, color=S1, label="unweighted")
        ax.bar(x + 0.21, w, width=0.38, color=S2, label="weighted")
        biggest = max(range(len(idx)), key=lambda i: abs(w[i] - u[i]))
        ax.annotate(f"{w[biggest]-u[biggest]:+.1f} pp",
                    (biggest + 0.21, w[biggest]), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=8.8, color=S2,
                    fontweight="700")
        ax.set_xticks(x, [str(i).replace("_", "\n") for i in idx],
                      fontsize=8.2, rotation=0)
        ax.annotate(f, xy=(0, 1), xycoords="axes fraction",
                    textcoords="offset points", xytext=(0, -2), ha="left",
                    va="bottom", fontsize=10, color=INK, fontweight="600")
        _clean(ax)
    axes[0].set_ylabel("% of respondents")
    axes[0].legend(fontsize=9, loc="upper left")
    for a in axes:
        a.title.set_visible(False)
    _title(axes[0], "Weighting moves the marginals — every cross-tab must use it",
           "wtssnrps, the post-stratified nonresponse-adjusted person weight. "
           "Largest shift annotated per panel.")
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 9. Data availability heatmap: cluster x topic
# ---------------------------------------------------------------------------

def fig_availability(cstats: pd.DataFrame, nodes: list[dict],
                     topic_of: dict, min_neff: int, path: str,
                     cstats_coarse: pd.DataFrame | None = None,
                     nodes_coarse: list[dict] | None = None,
                     k_main: int | None = None, k_coarse: int | None = None):
    """Two partitions side by side. The contrast IS the finding: the partition
    that satisfies the spec's K target has almost no usable ground truth; the
    one that has ground truth is 5x coarser."""
    use_style()

    panels = []
    if cstats_coarse is not None:
        panels.append((cstats_coarse, nodes_coarse or [],
                       f"Coarse partition  ·  K = {k_coarse}"))
    panels.append((cstats, nodes, f"Main partition  ·  K = {k_main}"))

    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 6.4))
    axes = np.atleast_1d(axes)

    pivs = []
    for cs, _, _ in panels:
        d = cs.copy()
        d["topic"] = d.item_id.map(topic_of)
        d["ok"] = d.n_eff >= min_neff
        pivs.append(d.pivot_table(index="cluster_id", columns="topic",
                                  values="ok", aggfunc="mean") * 100)

    col_order = sorted(pivs[0].columns, key=lambda t: -pivs[0][t].mean())

    im = None
    for ax, piv, (cs, nds, name) in zip(axes, pivs, panels):
        size = {n["cluster_id"]: n["n_raw"] for n in nds if n["level"] == 2}
        piv = piv.reindex(sorted(piv.index, key=lambda c: -size.get(c, 0)))
        piv = piv[[c for c in col_order if c in piv.columns]]
        im = ax.imshow(piv.to_numpy(), aspect="auto", cmap=SEQ_CMAP,
                       vmin=0, vmax=100, interpolation="nearest")
        ax.set_xticks(np.arange(len(piv.columns)), piv.columns, rotation=42,
                      ha="right", fontsize=8.6)
        ax.set_yticks([])
        pct = 100 * (cs["n_eff"] >= min_neff).mean()
        ax.set_ylabel(f"{len(piv)} leaf clusters  →  largest at top", fontsize=9)
        ax.grid(False)
        ax.annotate(f"{name}  ·  {pct:.1f}% scoreable",
                    xy=(0, 1), xycoords="axes fraction",
                    textcoords="offset points", xytext=(0, 6), ha="left",
                    va="bottom", fontsize=10.5, color=INK, fontweight="600")

    cb = fig.colorbar(im, ax=axes.tolist(), fraction=0.026, pad=0.02)
    cb.set_label(f"% of that topic's items reaching n_eff ≥ {min_neff}",
                 fontsize=9.5)
    cb.outline.set_visible(False)
    _title(axes[0], "Where usable ground truth actually exists",
           "Palest cells have no scoreable (cluster × item) pair at all.",
           lift=16)
    fig.savefig(path)
    plt.close(fig)
