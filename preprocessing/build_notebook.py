#!/usr/bin/env python3
"""Generate B17_GSS2024_preprocessing.ipynb from source cells."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
co = lambda s: C.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# B17 — GSS 2024 preprocessing (M1 → M2 → item split)

Everything before the first LLM call: harmonised microdata, an interpretable
cluster tree, per-cluster ground-truth histograms, and the anchor/target split
that §5.1 of the architecture spec calls for.

**Data.** GSS 2024 cross-section, Release 3 (NORC, March 2026). The spec names
GSS 2022 as the primary bed but §5.6 says to swap in a newer wave if one exists
— it does, and the newer wave is the better choice on leakage grounds (F6).

**What to look at first.** Figures 1, 2 and 9. They show a constraint the spec
does not account for: at the K the spec asks for, almost no (cluster × item)
cell clears the n_eff ≥ 30 gate that §5.2 imposes on scoring.

---

## Pipeline

| Module | Output | Spec |
|---|---|---|
| M1 ingest/harmonise | `individual_table.parquet`, `item_codebook.yaml` | §3.3 M1 |
| M2 cluster tree | `cluster_tree.json`, `cluster_stats.parquet` | §3.3 M2 |
| — level-0 marginals | `population_stats.parquet` | §5.4 B0a |
| item split | `item_split.csv` | §5.1 |
| K sweep | `k_sweep.csv` | §5.5 / §5.7 |
""")

md("## 0 · Setup")
co(r"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
import matplotlib.pyplot as plt

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from popsim.report import plots
plots.use_style()

CFG = ROOT / "configs" / "gss_main.yaml"
cfg = yaml.safe_load(CFG.read_text())
RUN = ROOT / cfg["output"]["dir"]
print("run dir:", RUN)
""")

md("""
### Re-run the pipeline from scratch

Takes about 10 seconds. Skip it if `runs/gss2024_main/` is already populated.
""")
co(r"""
# import run_pipeline; run_pipeline.main(str(CFG))
""")

co(r"""
table   = pd.read_parquet(RUN / "individual_table.parquet")
cstats  = pd.read_parquet(RUN / "cluster_stats.parquet")
cstats_c= pd.read_parquet(RUN / "cluster_stats_coarse.parquet")
pstats  = pd.read_parquet(RUN / "population_stats.parquet")
split   = pd.read_csv(RUN / "item_split.csv")
sweep   = pd.read_csv(RUN / "k_sweep.csv")
nodes   = json.loads((RUN / "cluster_tree.json").read_text())
nodes_c = json.loads((RUN / "cluster_tree_coarse.json").read_text())
manifest= json.loads((RUN / "manifest.json").read_text())
codebook= {c["item_id"]: c for c in yaml.safe_load((RUN / "item_codebook.yaml").read_text())}
topic_of= {k: v["topic"] for k, v in codebook.items()}

MIN_NEFF = cfg["scoring"]["min_neff"]
K, K_C   = manifest["K"], manifest["K_coarse"]
MIN_CELL = manifest["min_cell"]
print(f"{len(table):,} respondents · {len(codebook)} items · K = {K} (coarse {K_C})")
""")

md("## 1 · M1 — did the harmonisation do what it claims?")
co(r"""
for line in manifest["log"][:6]:
    print(line)
print()
print("contract checks:")
for k, v in manifest["checks"].items():
    flag = "" if v in (True, 0) or isinstance(v, int) else "   <-- FAILED"
    print(f"  {k:32s} {v}{flag}")
""")

co(r"""
# Harmonised demographics, weighted vs unweighted
demo_cols = [c for c in table.columns if c.startswith("demo_")]
rows = []
for c in demo_cols:
    m = table[c].notna()
    rows.append({"field": c.replace("demo_", ""), "levels": table[c].nunique(),
                 "missing": int((~m).sum()),
                 "missing_pct": round(100 * (~m).mean(), 1)})
pd.DataFrame(rows).sort_values("missing_pct", ascending=False)
""")

md("""
Income is the binding constraint: `CONINC` non-response puts >10% of
respondents outside any income-conditioned cell. They are **not** dropped —
they land in an explicit `unclassified` leaf, so population marginals stay
unbiased.
""")

md("## 2 · M2 — the cluster tree")
co(r"""
leaves = [n for n in nodes if n["level"] == 2]
print("level-1 nodes")
for n in [x for x in nodes if x["level"] == 1]:
    print(f"   n={n['n_raw']:5d}   {n['definition_text']}")

print(f"\n{len(leaves)} leaves — 8 largest")
for n in sorted(leaves, key=lambda x: -x["n_raw"])[:8]:
    print(f"   n={n['n_raw']:4d}  n_eff={n['n_eff']:6.1f}  share={n['pop_share']:.3f}")
    print(f"        {n['definition_text']}")
""")

md("""
Every leaf definition is a literal conjunction of demographic constraints, so
the stat card in M3 can be generated rather than authored — that is the whole
reason for rejecting embedding clustering in §2.2.
""")

md("## 3 · Figures")
co(r"""
from IPython.display import Image, display
FIG = ROOT / cfg["output"]["figures"]

def show(name, note=""):
    if note: print(note)
    display(Image(filename=str(FIG / name)))
""")

md("""
### 3.1 The binding constraint

The spec asks for K ≈ 150 (§1.4, §2.2) *and* excludes any cell below
n_eff = 30 from scoring (§5.2). On a 3,309-respondent sample those two
requirements are close to mutually exclusive.
""")
co('show("fig01_k_vs_scoreable.png")')
co('show("fig02_leaf_sizes.png")')
co('show("fig09_availability.png")')

md("""
### 3.2 Item heterogeneity

§1.4 claim (i) is scored on the *top-quartile heterogeneity* items. Which items
those are depends entirely on whether heterogeneity is debiased first.
""")
co('show("fig03_heterogeneity_bias.png")')
co('show("fig04_top_heterogeneity_items.png")')
co('show("fig07_between_cluster_spread.png")')

md("### 3.3 Item supply and the split")
co('show("fig05_item_coverage.png")')
co('show("fig06_anchor_target_split.png")')

md("### 3.4 Weighting")
co('show("fig08_weighting_effect.png")')

md("## 4 · The split, as a table")
co(r"""
sm = (split.groupby(["topic"])
      .agg(items=("item_id", "size"),
           anchors=("role_standard", lambda s: (s == "anchor").sum()),
           median_h=("h_icc", "median"),
           max_h=("h_icc", "max"))
      .sort_values("median_h", ascending=False).round(4))
sm
""")

co(r"""
# Items available for §1.4 claim (i): held-out targets in the top heterogeneity
# quartile that actually have scoreable cells.
q75 = split.h_icc.quantile(0.75)
cells_main   = cstats[cstats.n_eff >= MIN_NEFF].groupby("item_id").size()
cells_coarse = cstats_c[cstats_c.n_eff >= MIN_NEFF].groupby("item_id").size()

for name, cells, kk in (("main", cells_main, K), ("coarse", cells_coarse, K_C)):
    elig = split[(split.role_standard == "target") & (split.h_icc >= q75)]
    elig = elig.assign(n_cells=elig.item_id.map(cells).fillna(0))
    ok = elig[elig.n_cells >= 5]
    print(f"{name:7s} K={kk:4d}   top-quartile held-out targets with >=5 "
          f"scoreable clusters: {len(ok):3d} of {len(elig)}   "
          f"(spec §1.4 asks for >= 40)")
""")

md("""
**This is the number that decides whether the study can run as written.**
§1.4 requires ≥ 40 held-out items. Compare the two partitions: the one that
meets the spec's K target and the one that has usable ground truth.
""")

md("## 5 · Lookups")
co(r"""
def cluster(query, n=5):
    hits = [x for x in nodes if x["level"] == 2
            and query.lower() in x["definition_text"].lower()]
    for h in sorted(hits, key=lambda x: -x["n_raw"])[:n]:
        print(f"n={h['n_raw']:4d} n_eff={h['n_eff']:6.1f}  {h['definition_text']}")
        print(f"   id: {h['cluster_id']}")
    print(f"({len(hits)} matches)")

cluster("college graduates, income q4")
""")

co(r"""
def item(item_id):
    c = codebook[item_id]
    r = split[split.item_id == item_id].iloc[0]
    print(f"{item_id}  —  {c['text']}")
    print(f"  topic {c['topic']} · coverage {c['coverage']:.2f} · "
          f"h_icc {c['heterogeneity']} (raw {c['heterogeneity_raw']})")
    print(f"  role: standard={r.role_standard}  adversarial={r.role_adversarial}")
    pop = pstats[pstats.item_id == item_id].iloc[0]
    for lab, p in zip(c["scale"]["labels"], pop["hist"]):
        print(f"    {100*p:5.1f}%  {lab}")

item("polviews")
""")

co(r"""
def cluster_vs_nation(item_id, min_neff=None, coarse=True):
    src = cstats_c if coarse else cstats
    mn = MIN_NEFF if min_neff is None else min_neff
    d = src[(src.item_id == item_id) & (src.n_eff >= mn)].copy()
    tree = {n["cluster_id"]: n["definition_text"]
            for n in (nodes_c if coarse else nodes)}
    d["cluster"] = d.cluster_id.map(tree)
    pop = pstats[pstats.item_id == item_id].iloc[0]["mean"]
    d["vs_nation"] = (d["mean"] - pop).round(3)
    return d[["cluster", "n_raw", "n_eff", "mean", "sd", "vs_nation"]] \
             .sort_values("vs_nation")

cluster_vs_nation("polviews").head(10)
""")

md("""
## 6 · What M3 needs next

`cluster_tree.json` + `cluster_stats.parquet` are exactly the inputs the stat
card builder takes. The one contract M3 must enforce that this stage cannot:
a stat card for item Y must exclude Y, its near-duplicates, and every anchor in
Y's cross-fit fold (§3.3 M3). The fold assignment is not yet written here
because it depends on M9's cross-fit plan (`F = 3`).
""")

nb["cells"] = C
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}
nbf.write(nb, "B17_GSS2024_preprocessing.ipynb")
print("wrote B17_GSS2024_preprocessing.ipynb", len(C), "cells")
