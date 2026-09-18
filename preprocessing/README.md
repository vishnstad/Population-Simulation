# B17 — GSS 2024 preprocessing (M1 → M2 → item split)

Everything the architecture spec calls for **before the first LLM call**:
harmonised microdata, an interpretable cluster tree, per-cluster ground-truth
histograms, and the anchor/target split of §5.1.

```
python run_pipeline.py configs/gss_main.yaml    # ~10 s, writes runs/gss2024_main/
python make_figures.py configs/gss_main.yaml    # writes figures/
python verify.py /tmp/cb.txt                    # external check vs the codebook
jupyter lab B17_GSS2024_preprocessing.ipynb     # the walkthrough
```

## Data

**GSS 2024 cross-section, Release 3a** (NORC, March 2026) — `data_raw/GSS2024.dta`.

The spec (§1.4) names GSS 2022, but §5.6 says to swap in a newer wave if one is
released, because a wave postdating most pretraining cutoffs is the cheapest
available mitigation for leakage (F6). GSS 2024 exists, so it is used here.
The adapter is wave-agnostic — point `data.path` at `2022_stata.zip` or the
cumulative file and it runs unchanged.

- 3,986 rows in the file → **3,309 analytic respondents** (677 carry no
  cross-section weight: follow-on and oversample cases whose inclusion would
  corrupt every population estimate). 3,309 is also NORC's own base in the
  published codebook.
- Weight: `wtssnrps` (post-stratified, nonresponse-adjusted).
- 989 variables → 646 with a 2–7 point labelled scale → **507 attitudinal /
  behavioural items** after hand review; 139 dropped as biography, admin,
  duplicated demographics, or knowledge quizzes (`codebooks/item_taxonomy_gss.yaml`).

## Three findings that bear on the study design

### 1. K ≈ 150 and the n_eff ≥ 30 scoring gate are close to incompatible

§1.4 and §2.2 ask for K ≈ 150 clusters. §5.2 excludes any (cluster × item) cell
below Kish n_eff = 30 from scoring. On 3,309 respondents:

| K | min_cell | cells clearing n_eff ≥ 30 |
|---:|---:|---:|
| 12 | 200 | **96%** |
| 26 | 90 | **51%** |
| 57 | 40 | 5% |
| **122** | **17** | **0.8%** |

The headline consequence, computed in notebook §4:

> At K = 122, **0 of 51** top-quartile-heterogeneity held-out target items have
> ≥ 5 scoreable clusters. At K = 26, **32 of 51** do.
> §1.4 requires **≥ 40** held-out items.

So the claim as written is not evaluable at the spec's K, and only *nearly*
evaluable at a K roughly 5× coarser. Options, in rough order of cost: relax the
n_eff floor and report interval coverage instead of dropping cells; score claim
(i) at the coarse partition and use K = 122 only for the cost curve; pool
GSS 2022 + 2024 to roughly double n; or scale the falsifiable claim down from
40 items. Worth settling before any elicitation budget is spent —
`figures/fig01`, `fig02` and `fig09` are the three plots to show a supervisor.

### 2. Naive between-cluster heterogeneity is inflated ~2×

Claim (i) is scored on "top-quartile heterogeneity items". With ~27 respondents
per cluster, the naive weighted between-cluster variance share is badly upward
biased — cluster means differ partly *because* cells are small. `evalx/split.py`
therefore reports two numbers per item:

- `h_raw` — the naive share, what a plain groupby gives you;
- `h_icc` — a one-way random-effects ANOVA estimate that subtracts the
  within-cluster mean square (can legitimately go negative).

The median item loses roughly half its apparent heterogeneity, and top-quartile
membership changes for a substantial number of items. **The split and the
top-quartile selection use `h_icc`.** Selecting on `h_raw` would select on noise
(`figures/fig03`).

### 3. Two spec-vs-file deviations, both documented in the recode tables

- **`region` has 4 levels, not 9.** The 9 census divisions live in the
  cumulative file; the single-year public file ships Northeast/Midwest/South/West.
  Level-1 granularity drops accordingly. Re-run against `GSS_stata.zip` to recover.
- **`SRCBELT` is absent.** Reconstructed from `XNORCSIZ` using NORC's published
  derivation, so `urban` and the 6-level `srcbelt` are exactly reproducible.

## Design choices worth knowing about

**The partition is built by recursive coarsening, not greedy cell merging.**
The obvious implementation — "merge any cell below min_cell into its nearest
sibling" — was implemented first and produced two bugs that are easy to ship
without noticing:

1. Widening each axis in turn eventually yields a cell meaning *"anyone at all
   in this region"*. Two such cells have byte-identical definitions, collide on
   `cluster_id`, and silently fuse in every downstream groupby. On GSS 2024 this
   quietly merged 8 of 21 clusters.
2. Restricting merges to cells differing on exactly one axis fixes the collision
   but strands ~98% of cells below `min_cell`.

`clustering/partition.py` instead splits recursively, axis by axis, coarsening
one axis's *levels* at a time and descending only where the data supports a
further split. Ordinal axes (age, education, income) merge adjacent levels only,
so no leaf spans "18-24 or 65+". Every leaf is a plain conjunction:

> *women, aged 25-34, less than HS to some college, income q3–q5,
> urban/suburban, the South*

Sparse branches stop early and carry the remaining axes as unconstrained —
which is §2.2's adaptive refinement, arrived at because the data forced it.
`leaf_ids_unique` is now a hard contract check.

**Recode tables are self-verifying.** Every code in `codebooks/recodes_gss.yaml`
carries a `label_contains` guard checked against the value labels inside the
`.dta` at load time. If NORC renumbers a variable, the load raises instead of
silently producing a wrong cross-tab (§3.3 M1's "silent recode errors").

**Income non-response is not imputed.** ~11% of respondents lack `CONINC`. They
are not dropped — they land in an explicit `unclassified` leaf, so population
marginals stay unbiased and the cost is visible rather than hidden.

## Verification

`verify.py` checks the pipeline against NORC's own published frequency tables,
parsed out of the codebook PDF (`pdftotext -layout`), which is the only genuinely
external reference available:

```
[PASS] analytic sample = 3309
[PASS] 2,202 (item × code) unweighted counts matched the codebook exactly
[PASS] demographic recode level totals reconcile (0 mismatches)
[PASS] 507 population histograms sum to 1
[PASS] weights sum to 3309.0
[PASS] every respondent in exactly one leaf · leaf ids unique
[PASS] pop shares sum to 1 · leaf n_raw sums to analytic n
11/11 checks passed
```

The codebook prints unweighted counts, so that comparison validates the
analytic-sample filter, the Stata extended-missing handling (`.d`/`.i`/`.n`/`.s`),
and every value-code mapping. Weighting is checked separately.

To regenerate the reference:
`pdftotext -layout "GSS 2024 Codebook R3a.pdf" /tmp/cb.txt`

## Outputs — `runs/gss2024_main/`

| File | Contract |
|---|---|
| `individual_table.parquet` | `IndividualTable` — §3.3 M1 |
| `item_codebook.yaml` | `Item` records with topic, coverage, heterogeneity, role |
| `cluster_tree.json` | `ClusterTree`, K = 122 — §3.3 M2 |
| `cluster_stats.parquet` | `ClusterStats`: weighted hist, Kish n_eff, mean, sd |
| `cluster_tree_coarse.json` / `cluster_stats_coarse.parquet` | K = 26 — the finest partition where a majority of cells are scoreable |
| `population_stats.parquet` | Level-0 marginals — the B0a oracle of §5.4 |
| `item_split.csv` | Heterogeneity + roles under both split regimes — §5.1 |
| `k_sweep.csv` | K vs scoreable cells — §5.5 / §5.7 |
| `manifest.json` | Config snapshot, audit log, contract checks |
| `verification.json` | External codebook check |

Both split regimes are emitted: **standard** (stratified by topic ×
heterogeneity quartile, 60/40) and **adversarial** (whole topics held out —
here `health_wellbeing`, `media_technology`, `politics`).

## Figures — `figures/`

| | |
|---|---|
| `fig01_k_vs_scoreable` | The K vs scoreable-cells trade-off |
| `fig02_leaf_sizes` | Leaf raw-n and n_eff vs the two thresholds |
| `fig03_heterogeneity_bias` | `h_raw` vs `h_icc` |
| `fig04_top_heterogeneity_items` | The 25 items where clusters genuinely differ |
| `fig05_item_coverage` | Ballot/module rotation |
| `fig06_anchor_target_split` | Both split regimes by topic |
| `fig07_between_cluster_spread` | Cluster means vs the national marginal |
| `fig08_weighting_effect` | Weighted vs unweighted demographics |
| `fig09_availability` | Scoreable cells, main vs coarse partition |

## What M3 needs next

`cluster_tree.json` + `cluster_stats.parquet` are the stat-card builder's inputs.
The one contract M3 must enforce that this stage cannot: a stat card for item Y
must exclude Y, its near-duplicates, and every anchor in Y's cross-fit fold
(§3.3 M3). Fold assignment is not written here because it depends on M9's
cross-fit plan (`F = 3`).

## Layout

```
configs/gss_main.yaml            every run fully specified by one file (§3.2)
codebooks/recodes_gss.yaml       reviewed demographic recodes + label guards
codebooks/item_taxonomy_gss.yaml 646 candidates → 507 items, hand-classified
popsim/data/adapters/gss.py      M1
popsim/clustering/partition.py   M2 — recursive coarsening
popsim/clustering/stats.py       weighted histograms, Kish n_eff, Hellinger
popsim/evalx/split.py            heterogeneity + both split regimes
popsim/report/plots.py           figures
run_pipeline.py  make_figures.py  verify.py  build_notebook.py
```

Python 3.11+; `pandas pyarrow pyreadstat pyyaml numpy scipy scikit-learn matplotlib`.
