# B17 — pipeline reference

Cluster-level distributional elicitation with cross-fitted anchor calibration.
For setup and the things a human has to do, see [`../manual.md`](../manual.md).
For the scientific argument, see [`../Docs/B17_architecture_spec.md`](../Docs/B17_architecture_spec.md).

---

## The idea in one paragraph

Instead of prompting one LLM persona per simulated person — expensive, and it
collapses each person onto a single modal answer — we group the population into
~26 interpretable demographic clusters and ask the model, once per cluster, *what
percentage of this group would choose each option*. The model is shown the
cluster's real answers to questions we have data for (its **anchor** items), and
asked about a question we don't. Raw model histograms are systematically
mis-dispersed, so a **calibration map is learned from the anchors** and applied to
the unseen question. The whole thing is scored on held-out survey items, against
the baseline that matters: "every subgroup answers like the country".

---

## Pipeline

```
preprocessing/runs/gss2024_main/        (M1, M2 — already built and verified)
  individual_table.parquet     3,309 respondents, weights, demographics, leaf id
  cluster_tree_coarse.json     K = 26 interpretable leaves
  cluster_stats_coarse.parquet per-cluster weighted histograms + Kish n_eff
  item_codebook.yaml           507 items: wording, scale, topic
  item_split.csv               anchor/target roles, heterogeneity (h_icc)
  population_stats.parquet     national marginals (the B0a oracle)
                    |
                    v
run_elicitation.py  ---- M3 stat cards ----> M4 LLM ensemble ----> runs/<name>/
                                                    raw_anchor_elicitation.parquet
                                                    raw_target_elicitation.parquet
                                                    crossfit_plan.json
                    |
                    v
benchmark.py        ---- M5 calibration ---> M9 metrics + baselines
                                                    benchmark_summary_<split>.json
                                                    benchmark_per_item_<split>.csv
                                                    calibrated_target_distributions.parquet
                    |
                    v
simulate_region.py  ---- M7 router --------> M6 segment aggregation + bootstrap
M10_app/app.py                                      answer + 90% CI + honesty box
```

## Layout

| Path | Module | Role |
|---|---|---|
| `config/run_gss2024.yaml` | — | One file fully specifies a run |
| `shared/paths.py` | — | Config and path resolution, `.env` loading |
| `shared/llm_client.py` | — | Anthropic / OpenAI / Gemini / mock, caching, budget |
| `shared/cache.py` | — | SQLite prompt cache — resumability and no double-spend |
| `shared/budget_guard.py` | — | Hard dollar ceiling, enforced before each call |
| `M3_statcards/builder.py` | M3 | Stat cards + the leakage contract |
| `M4_elicitation/` | M4 | Prompts, schemas, the ensemble runner |
| `M5_calibration/crossfit.py` | M5 | **Cross-fitted calibration — the core mechanism** |
| `M5_calibration/isotonic.py` | M5 | Ordinal CDF recalibration |
| `M5_calibration/variance.py` | M5 | Variance restoration with hierarchical shrinkage |
| `M6_aggregation/scope.py` | M6 | Segment/region resolution with fractional membership |
| `M6_aggregation/raking.py` | M6 | Iterative proportional fitting to census margins |
| `M6_aggregation/bootstrap.py` | M6 | Nonparametric intervals |
| `M7_router/router.py` | M7 | Oracle router: observed vs simulated |
| `M9_evaluation/metrics.py` | M9 | W1, JS, variance ratio, between-cluster SD, ECE |
| `M9_evaluation/baselines.py` | M9 | B0a, B0b, B1, B2, B3, B4 |
| `M10_app/app.py` | M10 | Streamlit demo |
| `run_elicitation.py` | — | Stage 1: spend money, get raw histograms |
| `benchmark.py` | — | Stage 2: calibrate and score |
| `simulate_region.py` | — | Stage 3: answer a question about a region |
| `build_acs_margins.py` | — | Build raking margins from ACS PUMS |
| `tests/test_pipeline.py` | — | Contract tests (`pytest tests/ -v`) |

---

## Three design decisions worth knowing

### 1. Cross-fitting is what makes the calibrator honest

If anchor *a* is elicited from a card that already displays *a*'s true histogram,
the model can copy it. The calibrator then learns "the model is nearly perfect" —
true on anchors, false on targets — and becomes a no-op that inflates the score.

So anchors are split into 3 folds. To produce the training pair for anchor *a* in
fold *f*, the card is built from anchors in folds **other than** *f*: *a* is
absent, and so is every anchor sharing its fold. The resulting
(prediction, truth) pairs are generated under exactly the conditions a held-out
target faces.

`StatCardBuilder.verify_card_contract` asserts this on every single call and
raises rather than letting a leak through. The rule is tested in
`test_card_excludes_same_fold_anchors`.

### 2. Segment membership is fractional, not all-or-nothing

After the recursive coarsening in M2, a leaf may span several levels of an axis —
a sparse branch stops early and carries the remaining axes unconstrained. So for a
query like "women aged 25–34", a leaf is not simply in or out: part of it is in.

Counting such a leaf as wholly in over-counts the segment; dropping it
under-counts. `SegmentResolver` instead computes membership exactly from the
individual table:

```
membership(leaf, query) = weight of respondents in leaf satisfying query
                          / total weight of respondents in leaf
```

For a region query this reduces to clean 0/1 selection; for anything finer it
stays exact.

### 3. The ensemble members must be distinct draws

Nine draws per cell — 3 hand-written paraphrases × 3 samples at temperature 0.7.
The sample index is part of the cache key, so the members are cached
independently and remain independent samples.

This was previously broken: without the sample index in the key, repeats 2 and 3
of each paraphrase were cache hits of repeat 1, so a "3×3 ensemble" was really 3
draws with zero within-paraphrase variance — and the bootstrap over ensemble
members was resampling three identical points. `run_elicitation.py` now prints
`mean ensemble SD` after every stage; if it is ever 0.0, the ensemble is not
sampling. Pinned by `test_ensemble_members_are_distinct_draws`.

---

## Reading the benchmark

```
                               system      W1  var_ratio  between_SD_ratio  rank_rho
      Calibrated cluster agent (ours)     ...        ...               ...       ...
       B0a  national marginal, oracle     ...        ...             0.000       NaN
    B0b  national marginal, predicted     ...        ...             0.000       NaN
        B1   nearest-anchor heuristic     ...        ...               ...       ...
B3   supervised skyline (uses labels)     ...        ...               ...       ...
   B4   uncalibrated agent (ablation)     ...        ...               ...       ...
```

- **W1** — Wasserstein-1 to the true cluster histograms. Lower is better. Primary.
- **var_ratio** — predicted / true SD *within* cluster. Target [0.8, 1.2]. Never
  read alone: a uniform histogram scores well here and terribly on W1.
- **between_SD_ratio** — predicted / true spread *across* clusters. **The F2
  detector.** B0a and B0b score exactly 0 by construction — they give every
  cluster the same answer. If our row is near zero too, cluster conditioning has
  failed regardless of what W1 says.
- **rank_rho** — Spearman correlation of cluster means. Does it order subgroups
  correctly, even if levels are off? `NaN` for B0a/B0b is correct: constant
  predictions have no ordering.

Sanity checks the harness should always satisfy: B3 (which uses real labels)
should win on W1; B0a/B0b should show `between_SD_ratio = 0`; B1 should show a
between-cluster ratio near 1 because it copies real subgroup data.

---

## Cost control

Three independent mechanisms:

1. **SQLite prompt cache** (`shared/cache.py`) keyed on model + prompts +
   temperature + sample index. A completed call is never paid for twice, across
   runs.
2. **Resumable batches** — `elicit_batch` skips cells already in the output
   parquet, flushing every 25 cells. Ctrl-C and re-run to continue.
3. **Hard ceiling** (`shared/budget_guard.py`) — `preflight()` refuses a call
   whose estimated cost would breach `llm.max_usd`, *before* the tokens are spent.

Always `--dry-run` first.

---

## Tests

```bash
pytest tests/ -v
```

Contract tests, not smoke tests. Each pins a property that was either broken
before or is load-bearing for the claim:

| Test | Pins |
|---|---|
| `test_real_model_never_silently_becomes_a_mock` | Asking for Claude gives Claude or an error — never a mock |
| `test_ensemble_members_are_distinct_draws` | 9 draws are 9 draws |
| `test_card_excludes_same_fold_anchors` | Cross-fit leakage contract |
| `test_card_excludes_near_duplicates` | Near-duplicate leakage contract |
| `test_calibrator_restores_collapsed_variance` | The mechanism actually moves dispersion toward truth |
| `test_between_cluster_sd_ratio_detects_collapse` | The F2 detector returns 0 for a collapsed system |
| `test_partial_membership_is_fractional` | Segment membership is not all-or-nothing |
| `test_budget_ceiling_stops_before_spending` | The ceiling is a ceiling |

---

## Known limitations

- **K = 26.** The spec asks for ~150. On 3,309 respondents that leaves ~27 people
  per cell and no scoreable ground truth. Pooling GSS 2022 is the fix
  (`../manual.md` §6).
- **Region has 4 levels, not 9.** The single-year GSS public file ships
  Northeast/Midwest/South/West. `Dataset/GSS_stata.zip` (cumulative) has the 9
  census divisions.
- **Pretraining leakage is not yet probed.** GSS cross-tabs are plausibly in the
  model's training data. The leakage probe (spec §5.6) is not implemented; the
  adversarial split partially bounds it.
- **No opinion dynamics (M8).** Stage B. The panel file is already on disk.
- **Scenario accuracy is not validated and cannot be.** No ground truth exists for
  a novel question. Held-out-item accuracy is the honest proxy, which is exactly
  what the honesty box reports.
