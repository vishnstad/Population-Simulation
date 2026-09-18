# B17 — Hierarchical Multi-Agent Population Simulation: Architecture & Methodology Specification

**Role of this document.** Complete structural and methodological specification. A coding model implements from this; no design decisions are left open. No implementation code appears here — schemas, signatures, and pseudocode for novel algorithms only.

**Date:** 2026-08-06. Research citations current to early 2026; where release status of a dataset or paper is uncertain, it is flagged rather than asserted.

---

## 1. Problem reformulation

### 1.1 Attacking the framing

**The stated gaps are not the hard problem.**

- **G1 (cost)** is not a research problem. Clustering people and querying one agent per cluster is the obvious cost move, and Chopra et al. (2025) already run one-query-per-archetype at million scale. If the project's contribution is "fewer agents," it is a re-implementation.
- **G3 (hierarchy)** as stated ("flat simulations lack group dynamics") is decoration. A hierarchy of agents is only worth having if it *does statistical work*: partial pooling of sparse subgroups toward parents, and adaptive refinement where the model is uncertain. "Agents arranged in a tree" is not a mechanism.
- **G2 (representation)** is the real problem, and the presentation understates it.

**The actual hard technical problem, stated precisely:**

> **Conditional distribution transfer.** Given a cluster *c* described by sufficient statistics **S**_c (demographic marginals + response distributions on *anchor* items observed in microdata), produce a predicted distribution **P̂**(Y | c) for a *target* question Y that is **not observed in any dataset**, such that:
> 1. **Between-cluster structure is right** — the predicted ordering and spread of cluster-level means across clusters matches reality (not every cluster collapsing to one "safe" modal answer);
> 2. **Within-cluster variance is right** — the predicted distribution has the dispersion of real people in that cluster, not the near-zero variance of a roleplayed individual (Bisbee et al. 2024);
> 3. **Cost is O(K·I)**, K clusters × I items, not O(N·I) over N individuals;
> 4. Every property above is **measurable on held-out data**.
>
> A second, harder problem sits on top: **checkable influence** — making inter-cluster interaction *change* those distributions in a way that can be validated against observed distribution shifts (panel data), rather than by vibes.

This confirms and sharpens the suspicion in the brief: the contribution is not cheapness; it is **distributional faithfulness at fixed cheapness**, verified out-of-sample.

### 1.2 Where the naive version fails, mechanically

| # | Failure mode | Mechanism | Consequence |
|---|---|---|---|
| F1 | **Within-cluster variance collapse** | Prompting "you are a 30-year-old rural Tamil farmer, what do you think of X?" samples one modal token path. Bisbee et al. measured synthetic SDs at a fraction of ANES SDs. | Aggregate distributions are far too confident; decision-support outputs overstate consensus. |
| F2 | **Between-cluster collapse** | RLHF-tuned models give hedged, socially-safe answers regardless of persona → all clusters emit near-identical distributions. | System output ≈ national marginal; clusters add nothing. This is the *primary* way the whole project silently fails while looking like it works. |
| F3 | **Cross-tab redundancy** | For any item present in the survey, P(Y\|demographics) is a weighted cross-tab — zero LLM calls needed, exactly correct. | Any evaluation on in-survey items where the model saw those items is theater. |
| F4 | **Cluster-boundary artefacts** | Hard partitions assign a person entirely to one cluster; a 29-year-old and a 31-year-old land in different personas with discontinuous outputs. | Jagged, unstable segment predictions; results change under re-clustering. |
| F5 | **Sycophancy / false consensus under interaction** | Free-form LLM-vs-LLM dialogue converges: models agree with interlocutors (documented across multi-agent debate literature). | "Opinion dynamics" that always drift to consensus — an artifact, not a finding; echo-chamber runaway in the other direction if prompted adversarially. |
| F6 | **Pretraining leakage** | GSS/ANES/Pew cross-tabs are in every frontier model's training data. | Inflated scores that measure memory, not transfer. Must be probed and designed around (§5.6). |
| F7 | **Unfalsifiable scenario outputs** | For "launch product X in market Y," no ground truth exists. | Demo-ware. The validation must run on proxies (held-out survey items) that share the *mechanism* (zero-shot conditional distribution prediction) with the product use case. |
| F8 | **Prompt instability** | Paraphrasing the persona or question shifts outputs by many points (Bisbee; Dominguez-Olmedo et al. 2024). | Any single-prompt number is noise; all elicitation must be ensembled and spread must be reported. |

### 1.3 The killer objection and our answer

> *"You built an expensive way to reproduce a cross-tab you already had."*

**Answer, in three parts, built into the architecture:**

1. **The system never re-predicts what the data answers.** An *oracle router* (§3, module M7) checks every incoming question against the harmonized codebook. If the item (or a near-duplicate) exists in microdata, the answer is the weighted cross-tab, served directly, labeled "observed." The LLM path fires **only** for questions outside the data. By construction, the LLM's job is exclusively the thing the cross-tab cannot do.
2. **Validation conditions on that split.** Anchor items go into the agent's context; target items are held out entirely. The reported metric is performance on items the "cross-tab machine" has no access to. The trivial cross-tab is retained as the *skyline* on anchors and as an input, never as a competitor we pretend to beat on its own turf.
3. **The product case is the same mechanism.** A novel scenario ("UPI-based micro-pension for gig workers: adoption intent") is formally an unseen item. Held-out-item accuracy is the honest, measurable proxy for scenario accuracy — and we say so explicitly rather than claiming validated scenario prediction.

If we could not make this answer, the correct move would be to change the design. We can, so the design stands — but note what it commits us to: **the entire evaluation lives on held-out items, and the trivial baseline we must beat is the national marginal** (predicting every cluster answers like the whole population). Beating it is not automatic — it is exactly what F2 destroys.

### 1.4 The single falsifiable claim

> **Claim.** On ≥ 40 held-out attitudinal items from GSS 2022 (US primary) and ≥ 15 held-out items from NFHS-5 + Pew *Religion in India* (India transfer), calibrated distribution-valued cluster agents at K ≈ 150 clusters will:
> **(i)** achieve mean cluster-weighted Wasserstein-1 distance to true per-cluster response distributions **≥ 20 % lower than the national-marginal baseline** on the top-quartile heterogeneity items (items where clusters genuinely differ);
> **(ii)** achieve population-level variance ratio (predicted SD / true SD) in **[0.8, 1.2]**, where per-individual silicon sampling at ≥ 20× our query cost sits below 0.7;
> **(iii)** match or beat per-individual silicon sampling at equal query budget on (i).
>
> **If (i) fails — if cluster agents cannot beat "everyone answers like the average" on the items where subgroups demonstrably differ — the approach contributes nothing over the survey itself and is abandoned.** (ii) failing alone kills the distributional claim but leaves a means-only result; (iii) failing alone kills the efficiency claim.

---

## 2. Core novel mechanism

### 2.1 Candidate mechanisms — the search

**C1. Calibrated distribution-valued cluster agents (anchor-calibrated distributional elicitation).**
Agents never answer *as* a person. Each agent is asked, for each question, to output a **histogram over the response scale** ("what percentage of this group chooses each option"), conditioned on a *stat card* of the cluster's sufficient statistics. Raw LLM histograms are then passed through a **thin statistical calibration layer fit on anchor items via cross-fitting** (the agent predicts anchor items it was *not shown*, and the mapping from its predictions to the true anchor cross-tabs is learned and applied to target items).
- *Novelty:* moderate-to-good. Distribution elicitation from LLMs exists (Argyle et al. used token probabilities; Santurkar et al. 2023 *OpinionQA*; Meister et al. ~2024 benchmark verbalized vs. sampled vs. logprob distribution elicitation against Pew — cite after verifying exact title). **What is not in the literature:** eliciting at the *cluster* level conditioned on explicit sufficient statistics, and *learning a transfer calibration from anchors to held-out items* as a first-class mechanism with a variance-restoration term. Bisbee et al. diagnose variance collapse; nobody, to our knowledge (uncertain — re-check at writeup time), repairs it with a cross-fitted calibration map and validates distributionally.
- *Implementability:* high. Prompting + structured output + isotonic/Dirichlet calibration. No fine-tuning.
- *Falsifiability:* excellent — the claim in §1.4 is directly about this mechanism.

**C2. Mechanistic influence with LLM-elicited parameters (sufficient-statistics message passing).**
Inter-cluster influence is *not* roleplayed dialogue. The dynamical system is a transparent Friedkin–Johnsen model over cluster-level distribution summaries; only sufficient statistics (mean, dispersion, histogram) cross edges. The LLM's role is confined to **setting interpretable parameters** — susceptibility s_i per cluster×topic, edge salience — with written justifications. Validated against panel-wave distribution shifts (ANES 2020→2022 reinterviews; GSS 2016–2020 panel).
- *Novelty:* good. "LLM parameterizes a mechanistic model" is an emerging pattern (LLM-augmented ABM surveys, 2024–25) but has not been done for population opinion dynamics with panel validation.
- *Implementability:* medium. The dynamics are trivial; the *validation* is hard — persistence ("nothing changes") is a brutally strong baseline on 2-year panels, and we may lose to it.
- *Falsifiability:* good in principle, high risk of a null result.

**C3. Adaptive cluster granularity with an LLM budget allocator.**
Start coarse (K≈12); recursively split a cluster only where (a) elicitation ensemble disagreement is high or (b) anchor-item calibration error is high; allocate LLM calls where they buy accuracy. Produces the cost-vs-accuracy curve as a *mechanism* rather than a chart.
- *Novelty:* low-moderate — active-learning flavored engineering; reviewers will see a heuristic.
- *Implementability:* high.
- *Falsifiability:* fine (curve dominates fixed-K or it doesn't) but the result is an efficiency delta, not a scientific claim.

**C4. Calibration-as-training-signal (distillation).** Fine-tune a small open model (7–8B, LoRA) to map stat cards → distributions using anchor items as supervision.
- *Novelty:* moderate. *Implementability:* the risk item — GPU access, training instability, and it converts a prompting project into an ML-training project mid-semester. *Falsifiability:* fine.
- **Rejected** as core: too much schedule risk for four undergrads; and if C1's calibration layer works, C4 is an optimization of it, not a new claim. Keep as a stretch appendix experiment only if Stage B finishes early.

**C5. Pure archetype simulation à la Chopra et al. with better personas.** Rejected without ceremony: it is the naive version; owns none of F1–F3.

### 2.2 Decision

**Core mechanism = C1** (calibrated distribution-valued cluster agents). It attacks F1/F2/F3 head-on, is executable by the team, and carries the falsifiable claim.
**C3 is adopted as a supporting component** (it *is* the hierarchy earning its keep — see below), not headlined.
**C2 is the Stage B research extension**, pre-registered with an explicit fallback: if dynamics cannot beat persistence on panel shifts, we report the negative result in one section and the paper stands on C1.

**Why the hierarchy survives at all (sharpened G3):** the tree is not an org chart of agents. It does three statistical jobs: (1) **partial pooling** — leaf clusters with thin anchor data get calibration parameters shrunk toward their parent (hierarchical-Bayes style), which is what makes K≈150 viable on a 3.5k-respondent survey; (2) **adaptive refinement** (C3) — split only where uncertain; (3) **the influence graph** (C2) lives on the tree + homophily edges. Any hierarchy claim in the writeup must be one of these three, nothing fuzzier.

**One-line alternative rejections at the decision level:**
- Token-logprob elicitation instead of verbalized histograms → rejected as *sole* method: unavailable on several frontier APIs and conflates token probability with population proportion; we use it as an ablation arm where available.
- Free-form multi-agent dialogue for interaction → rejected: F5 makes it unfalsifiable; mechanistic dynamics with elicited parameters instead.
- Embedding-space k-means clustering → rejected: uninterpretable clusters can't be rendered into stat cards or replicated by reviewers; interpretable cross-partitioning instead.

---

## 3. System architecture

### 3.1 Pipeline overview

```
raw microdata ──▶ M1 ingest/harmonize ──▶ M2 cluster tree + stats ──▶ M3 stat cards
                                                                        │
scenario/question ──▶ M7 oracle router ──(observed)──▶ cross-tab answer │
                          │(unseen)                                     ▼
                          └──────────────▶ M4 distributional elicitation (LLM)
                                                    │ raw histograms
                                                    ▼
                                          M5 calibration layer
                                                    │ calibrated per-cluster dists
                                   ┌────────────────┤
                                   ▼                ▼
                          M8 dynamics (Stage B)   M6 aggregation + uncertainty
                                   │                │
                                   └───────▶ M9 evaluation harness
                                                    │
                                                    ▼
                                          M10 decision-support report
```

### 3.2 Directory layout

```
popsim/
  configs/                  # YAML experiment configs; every run fully specified by one file
    gss_main.yaml, nfhs_transfer.yaml, ablation_*.yaml
  popsim/
    data/
      adapters/gss.py  anes.py  nfhs.py  ihds.py  pew_india.py  census_margins.py
      harmonize.py  codebook.py
    clustering/partition.py  stats.py  pooling.py
    agents/statcard.py  elicit.py  paraphrase.py
    prompts/statcard.jinja  elicit_hist.jinja  elicit_params.jinja  paraphrases/
    calibration/fit.py  apply.py  crossfit.py
    dynamics/graph.py  fj.py  elicit_params.py          # Stage B
    scenarios/schema.py  compile.py  router.py
    aggregate/mixture.py  uncertainty.py
    evalx/metrics.py  baselines.py  ablations.py  cost.py  leakage.py
    report/build.py  plots.py
    llm/client.py  cache.py  budget.py
  codebooks/                # harmonized item metadata YAML per dataset
  runs/                     # outputs: one dir per run, config snapshot + parquet + metrics.json
  app/                      # Stage A demo UI (Streamlit)
  paper/
```

Language: Python 3.11+. Storage: Parquet via pyarrow; all tabular contracts below are Parquet schemas. LLM access through one `llm/client.py` wrapper with mandatory response caching (`cache.py`, keyed on model+prompt hash) and a hard budget guard (`budget.py`) that kills a run past its configured token ceiling. Statistics: numpy/scipy/scikit-learn only; no deep-learning framework in Stages A/B.

### 3.3 Modules

#### M1 — Ingestion & harmonization (`data/`)
- **Purpose:** per-dataset adapters producing one canonical individual-level table + one canonical codebook, with survey weights preserved.
- **Inputs:** raw distribution files (GSS SAS/Stata export, DHS recode files for NFHS, ICPSR files for IHDS, Pew SPSS, CSV census margin tables).
- **Outputs — `IndividualTable` (Parquet):**

| field | type | notes |
|---|---|---|
| person_id | string | dataset-prefixed, e.g. `gss2022:12345` |
| dataset | string | enum: gss, anes, nfhs5, ihds2, pew_india |
| wave | string | e.g. `2022` |
| weight | float64 | final person-level survey weight |
| demo | struct | see harmonized demographic schema below |
| responses | map<string,int8> | item_id → response code; −1 = missing/refused |

Harmonized `demo` struct (nullable fields; country-specific fields null elsewhere):
`age_band` (str: 18-24/25-34/35-44/45-54/55-64/65+), `sex` (str), `education` (str: none/primary/secondary/higher_secondary/tertiary), `income_band` (str, within-country quintile), `urban` (bool), `region` (str: US census division / India state), `religion` (str), `race` (str, US), `caste_group` (str, India: SC/ST/OBC/General, from self-report fields where available), `employment` (str), `marital` (str).

- **Codebook — `Item` (YAML per item):** `item_id`, `dataset`, `text` (exact question wording), `scale: {type: ordinal|nominal|binary, labels: [str], codes: [int]}`, `topic` (tag), `wave`, `role: anchor|target|excluded` (assigned by M9's split, not by hand), `heterogeneity` (float, filled by M2: weighted between-cluster variance share of the item).
- **Algorithm choices:** deterministic recode maps per dataset checked into `codebooks/`; refuse to auto-map — every recode is an explicit reviewed table. Missing-data rule: listwise per item (respondent excluded from that item's histogram only); weights renormalized per cell.
- **Failure modes:** silent recode errors (mitigation: per-item assertion that harmonized marginals match published toplines within 0.5 pp); weight misuse (all cross-tabs must use `weight`; a unit test compares one published GSS cross-tab exactly).

#### M2 — Cluster tree (`clustering/`)
- **Purpose:** interpretable hierarchical partition of the population + per-cluster sufficient statistics.
- **Inputs:** `IndividualTable`, clustering config (axes, min cell size, max leaves).
- **Method:** **constrained interpretable cross-partitioning**, not k-means. Level 0 = population. Level 1 = region × urban. Level 2 = leaf clusters formed by crossing (age_band × education × income_band × sex [+ religion or caste_group for India]) within each level-1 node, then **greedy sibling merge**: any cell with weighted n < `min_cell` (default 40 respondents) merges with its nearest sibling (Hellinger distance on anchor-item histograms) until all leaves ≥ min_cell. Target K after merging: 100–200 leaves for GSS; 300–600 for NFHS (larger n allows it).
  - *Rejected alternative:* k-prototypes / embedding clustering — uninterpretable stat cards, unreproducible clusters.
- **Outputs — `ClusterTree` (JSON) and `ClusterStats` (Parquet):**
  - `ClusterTree`: nodes `{cluster_id: str, level: int, parent: str|null, definition: {field: value|value-set}, n_weighted: float, pop_share: float}` — `definition` is a conjunction of demographic constraints, human-readable by construction.
  - `ClusterStats`: `cluster_id`, `item_id`, `hist` (list<float>, sums to 1, weighted), `n_eff` (float, Kish effective sample size for that cell), `mean`, `sd`.
- **Failure modes:** F4 boundary artefacts (mitigation: report metrics under a re-clustering seed perturbation; soft assignment is *not* used — instead robustness is demonstrated); tiny effective n in leaves (mitigation: min_cell + partial pooling in M5).

#### M3 — Stat cards (`agents/statcard.py`)
- **Purpose:** render a cluster's sufficient statistics into the agent's conditioning context.
- **Inputs:** `ClusterTree`, `ClusterStats`, anchor fold assignment (from M9 cross-fit plan).
- **Output — `StatCard` (JSON):**

```json
{
  "cluster_id": "gss:south_atlantic:urban:35-44:tertiary:q4:f",
  "definition_text": "Urban women aged 35-44 in the South Atlantic division, college-educated, household income in the 4th national quintile",
  "pop_share": 0.011,
  "demo_marginals": {"religion": {"protestant": 0.44, "...": 0.0}},
  "anchor_items": [
    {"text": "…exact GSS wording…", "labels": ["…"], "hist": [0.12, 0.31, 0.4, 0.17]}
  ],
  "anchor_fold": 2
}
```

Rendered to prose via `statcard.jinja` (numbers stated as percentages; no invented color). **Contract:** a stat card for elicitation of item Y must not contain Y, any item flagged as its near-duplicate (M7 similarity check), nor any anchor in Y's cross-fit fold.
- **Failure modes:** context-length pressure with many anchors (cap: 12 anchor items per card, selected by topic diversity — max-coverage over topic tags); leaking target semantics through topically-identical anchors (mitigation: the near-duplicate exclusion above).

#### M4 — Distributional elicitation (`agents/elicit.py`)
- **Purpose:** the LLM call. For (cluster, item): produce raw predicted histograms.
- **Protocol:** prompt = rendered stat card + exact item text + labeled options; instruction: *"Estimate the percentage of adults in this specific group who would choose each option. Output JSON only."* Ensemble: `n_paraphrase = 3` prompt paraphrases (fixed, checked-in, written once by hand — not model-generated at runtime) × `n_repeat = 3` samples at temperature 0.7 → 9 raw histograms per (cluster, item). Structured output (JSON schema enforced). Model tiers: primary = small frontier model (Haiku-class) for all main runs; one frontier model (Sonnet/GPT-class) on a 10-item subset for a model-scale ablation.
- **Output — `RawElicitation` (Parquet):** `cluster_id`, `item_id`, `model`, `paraphrase_id` (int8), `repeat_id` (int8), `hist` (list<float>), `rationale` (string, ≤ 60 tokens, logged not used), `prompt_tokens`, `completion_tokens`, `cached` (bool).
- **Failure modes:** malformed JSON (retry ≤ 2, then mark failed; failure rate is a reported metric); refusals on sensitive items — expected for caste/religion-conditioned items (mitigation: neutral statistical framing "estimate the percentage," never "speak as"; log refusal rate per item×cluster; items with > 10 % refusal are excluded and the exclusion is reported); histogram not summing to 1 (renormalize, log).

#### M5 — Calibration layer (`calibration/`) — **the core novel component**
- **Purpose:** learn the map from raw LLM histograms to faithful cluster distributions, using anchors as supervision, and transfer it to targets.
- **Cross-fitting plan:** anchors are split into `F = 3` folds. For fold *f*: elicit every anchor item in *f* using stat cards built from folds ≠ *f*. This yields, for every anchor, a (raw prediction, ground-truth cross-tab) pair produced under exactly the conditions targets face.
- **Calibrator (fit on pooled anchor pairs, hierarchically pooled per cluster):** three components, applied in order:
  1. **Ordinal recalibration:** isotonic regression on cumulative probabilities (predicted CDF → true CDF), pooled across clusters per scale-length; nominal scales use Dirichlet calibration (Kull et al. 2019) instead.
  2. **Variance restoration:** per-cluster dispersion multiplier — model true SD as `sd_true = α + β·sd_raw` fit by weighted least squares on anchors; adjust each calibrated histogram to hit the predicted SD by power-tempering the histogram (exponent found by 1-D root-finding). Cluster-level (α, β) shrunk toward parent-node estimates with weight `n_eff/(n_eff + τ)`, τ = 100 (this is the hierarchy doing partial pooling).
  3. **Ensemble aggregation:** the 9 raw histograms are averaged *after* step 1, and their spread is retained as `elicit_sd` for uncertainty (M6).
- **Pseudocode (novel algorithm — cross-fitted anchor calibration):**

```
fit_calibration(anchors, clusters, F):
  for f in 1..F:
    for (c, a) in clusters × anchors[f]:
      raw[c,a] ← elicit(statcard(c, anchors_excluding_fold(f)), a)   # M4
  pairs ← {(raw[c,a], truth[c,a])}
  iso   ← isotonic_fit(cdf(pairs))                 # per scale-type
  (α,β) ← wls_fit(sd_true ~ sd_raw, per cluster, shrunk to parent by n_eff/(n_eff+τ))
  return Calibrator(iso, α, β)

apply(cal, raw_hist, cluster):
  h ← iso_map(raw_hist)
  target_sd ← cal.α[cluster] + cal.β[cluster] · sd(h)
  return temper(h, solve λ: sd(h^λ/Z) = target_sd)
```

- **Output — `CalibratedDistribution` (Parquet):** `cluster_id`, `item_id`, `hist_cal` (list<float>), `elicit_sd` (float), `calibrator_version` (str).
- **Failure modes:** anchors unrepresentative of targets (topic mismatch — mitigation: anchor/target splits stratified by topic; plus an ablation with adversarial topic-disjoint splits to measure transfer honestly); overfitting per-cluster (mitigation: pooling above); variance restoration gamed by predicting uniform (mitigation: W1 metric punishes it — report both jointly, never variance ratio alone).

#### M6 — Aggregation & uncertainty (`aggregate/`)
- **Purpose:** population/segment answers with honest intervals.
- **Method:** population distribution = Σ_c w_c · hist_cal(c), with w_c = census-raked population shares — cluster weights are raked (iterative proportional fitting) to **external census margins** (ACS for US, Census of India 2011 projections for India) so composition matches the real population, not the survey sample. Uncertainty: nonparametric bootstrap over (a) elicitation ensemble members and (b) cluster resampling; report 90 % intervals.
- **Output — `PopulationAnswer` (JSON):** `question_id`, `scope` (population | segment definition), `hist` , `ci90_per_option` (list<[lo,hi]>), `top_segments` (list of cluster_ids with largest Hellinger distance from population answer), `provenance: observed|simulated`, `nearest_validated_items` (list<item_id, W1 score> — the honesty box, §M10).
- **Failure modes:** raking non-convergence on sparse margins (cap iterations, fall back to survey weights with a logged warning).

#### M7 — Scenario compiler & oracle router (`scenarios/`)
- **Purpose:** turn a decision question into an item; route observed questions to data.
- **`Scenario` schema (YAML, user-authored):** `scenario_id`, `decision_question` (str), `context` (str ≤ 500 words — product/policy description), `response_scale: {type, labels}`, `segments_of_interest` (list of demographic constraint dicts), `country`.
- **Router:** embed question text (any local sentence embedder; choice is not load-bearing) and compare against codebook items; cosine > 0.85 → flag as observed/near-duplicate → answer from cross-tab, and **exclude such items from ever being LLM-elicited**. This is the architectural answer to F3.
- **Compiler:** wraps scenario into the same `Item` structure with `synthetic: true`; synthetic items always carry `provenance: simulated` downstream.
- **Failure modes:** embedding similarity missing a paraphrased duplicate (mitigation: threshold tuned on a hand-labeled duplicate set of ~100 pairs; report duplicate-detection recall).

#### M8 — Influence dynamics (`dynamics/`) — Stage B only
- **Purpose:** checkable inter-cluster influence (mechanism C2).
- **Graph:** nodes = leaf clusters. Edge weight w_ij = normalized product of (geographic adjacency: same region 1.0 / adjacent 0.5 / else 0.1) × (homophily kernel: exp(−d_H(demo_i, demo_j)/σ), d_H = Hamming distance over demographic fields). Fixed, transparent, checked in.
- **Dynamics:** Friedkin–Johnsen on item means: `m_i(t+1) = s_i·m_i(0) + (1−s_i)·Σ_j w_ij·m_j(t)`; dispersion propagated by moment matching (histogram re-tempered to the FJ-updated mean holding calibrated SD, then SD relaxed toward neighborhood mixture SD at rate ρ). **Susceptibility s_i per (cluster, topic)** and ρ elicited from the LLM with the stat card + topic (`elicit_params.jinja`), output JSON `{s: float in [0,1], justification: str}`, ensembled ×5, median taken.
- **Validation target:** predict wave-2 per-cluster distributions given wave-1, on ANES 2020→2022 reinterview panel and GSS 2016–2020 panel. Baselines: persistence (no change), uniform national shift. Pre-registered: if FJ-with-elicited-parameters does not beat persistence in W1 on ≥ 55 % of panel items, report as negative result.
- **Failure modes:** F5 is designed out (no dialogue); real risk is the null result — accepted and pre-registered.

#### M9 — Evaluation harness (`evalx/`) — full protocol in §5.

#### M10 — Reporting (`report/`, `app/`)
- **Purpose:** the Stage A product. Streamlit app: pick country → author scenario → see population + segment distributions with intervals, top divergent segments, and a mandatory **honesty box**: "This is a simulated estimate. On the *k* most similar validated questions, this system's error was W1 = x (baseline y)." No report renders without it.
- **Output:** HTML/PDF report per scenario; all figures regenerable from `runs/` parquet.

### 3.4 Interface signatures (names, args, returns only)

```python
# data
def load_individuals(dataset: DatasetID, raw_dir: Path, codebook: Codebook) -> IndividualTable
def build_codebook(dataset: DatasetID, raw_dir: Path) -> Codebook
def published_topline_check(table: IndividualTable, checks: list[ToplineCheck]) -> CheckReport

# clustering
def build_cluster_tree(table: IndividualTable, axes: list[str], min_cell: int, max_leaves: int) -> ClusterTree
def compute_cluster_stats(tree: ClusterTree, table: IndividualTable, items: list[ItemID]) -> ClusterStats
def pool_toward_parent(stats: ClusterStats, tree: ClusterTree, tau: float) -> ClusterStats

# agents
def make_statcard(tree: ClusterTree, stats: ClusterStats, cluster_id: str,
                  anchor_items: list[ItemID], exclude: set[ItemID]) -> StatCard
def elicit_distribution(card: StatCard, item: Item, model: ModelID,
                        n_paraphrase: int, n_repeat: int, client: LLMClient) -> list[RawElicitation]

# calibration
def make_crossfit_plan(anchors: list[ItemID], n_folds: int, stratify_by: str) -> CrossfitPlan
def fit_calibrator(pairs: list[tuple[RawElicitation, Histogram]], tree: ClusterTree,
                   scheme: CalibScheme) -> Calibrator
def apply_calibrator(cal: Calibrator, raw: list[RawElicitation], cluster_id: str) -> CalibratedDistribution

# scenarios
def route(question: str, codebook: Codebook, threshold: float) -> RouteDecision  # observed | simulate
def compile_scenario(sc: Scenario) -> Item

# aggregation
def rake_weights(tree: ClusterTree, margins: CensusMargins, max_iter: int) -> dict[str, float]
def aggregate(dists: list[CalibratedDistribution], weights: dict[str, float],
              scope: SegmentQuery | None) -> PopulationAnswer
def bootstrap_ci(dists: list[CalibratedDistribution], weights: dict[str, float],
                 n_boot: int, alpha: float) -> PopulationAnswer

# dynamics (Stage B)
def build_influence_graph(tree: ClusterTree, sigma: float) -> InfluenceGraph
def elicit_susceptibility(card: StatCard, topic: str, model: ModelID, client: LLMClient) -> SusceptibilityParam
def run_fj(graph: InfluenceGraph, init: list[CalibratedDistribution],
           params: list[SusceptibilityParam], n_steps: int, rho: float) -> list[CalibratedDistribution]

# evaluation
def evaluate(preds: list[CalibratedDistribution], truth: ClusterStats,
             weights: dict[str, float], metrics: list[MetricID]) -> MetricsReport
def run_baseline(name: BaselineID, cfg: RunConfig) -> list[CalibratedDistribution]
def leakage_probe(items: list[Item], model: ModelID, client: LLMClient) -> LeakageReport
def cost_accuracy_curve(cfgs: list[RunConfig]) -> CurveReport

# report
def build_report(answer: PopulationAnswer, eval_context: MetricsReport, out: Path) -> Path
```

---

## 4. Data

### 4.1 Honest verdict up front

India-first was the brief. The honest assessment after going dataset by dataset: **Indian microdata is fully adequate for the demographic backbone and adequate-but-thin for attitudes; it is not adequate as the *primary* distributional-validation bed**, for two reasons: (1) no Indian source offers the *hundreds of repeated attitudinal items* that held-out-item validation feeds on (GSS alone carries ~hundreds per wave), and (2) no Indian source gives panel reinterviews for dynamics validation. Also decisive: primary validation on GSS/ANES gives **direct comparability with Argyle, Bisbee, Park, Santurkar** — reviewers can see our numbers next to theirs.

**Therefore: primary validation = US (GSS + ANES, ACS margins). India = full secondary replication and the Stage A product demo (NFHS-5 + Pew India + IHDS-II, Census 2011 margins).** The India replication is itself a claimable contribution — to our knowledge (verify at writeup) no silicon-sampling-style distributional evaluation exists on Indian microdata — but it rides on a validation design proven on US data. This ordering is a strength, not a retreat: we do not pretend NFHS's ~20 usable attitudinal items can do the job of GSS's item bank.

### 4.2 Dataset table

| Dataset | Contents | Unit / n | Access & license | Consumed by |
|---|---|---|---|---|
| **GSS 2022 cross-section** (NORC) | ~hundreds of attitudinal/behavioral items + full demographics | person; n ≈ 3,500 | free direct download, no gate | M1; **primary anchor/target bed** |
| **GSS 2016–2020 panel** | reinterviews of 2016/2018 respondents in 2020 | person-wave | free | M8 dynamics validation |
| **ANES 2020 Time Series** (+ 2020→2022 reinterview) | political attitudes, feeling thermometers | person; n ≈ 8,280 | free registration | M1 targets; M8 panel; Bisbee comparability |
| **ACS PUMS 2023** (IPUMS-USA) | demographics only, big n | person; ~3.4 M/yr | free registration (IPUMS terms: no redistribution of microdata) | M6 raking margins US |
| **NFHS-5** 2019–21 (DHS program) | health, empowerment, media exposure; attitudes: decision-making autonomy, attitudes toward IPV (7-scenario battery), fertility preferences | woman 15–49, n ≈ 724k; men n ≈ 102k | free, DHS registration with stated purpose; ~1-week approval | M1 India; **India anchor/target bed** (large n → K up to 600 clusters, district-level geography) |
| **Pew "Religion in India" 2019–20** | rich attitudes: religion, nationalism, gender norms, politics | adult, n ≈ 29,999 | free download after registration | M1 India; **the main India attitudinal target set** |
| **IHDS-II 2011–12** (ICPSR 36151) | income, consumption, employment + some attitude/practice items (gender practices, confidence in institutions) | 42k households / ~204k persons | free ICPSR registration | M1 India; income/consumption grounding for stat cards. **Caveat: 2011–12 vintage.** IHDS-3 fieldwork completed ~2024; public release status uncertain as of early 2026 — check, do not plan on it |
| **HCES 2022–23** (MoSPI) | consumption expenditure detail | household, ~260k | free, MoSPI microdata portal | optional stat-card enrichment (spending patterns for product scenarios) |
| **PLFS 2023–24** (MoSPI) | employment/labor detail | ~100k households | free | optional stat-card enrichment; no attitudes |
| **Census of India 2011 tables** (+ MoSPI projections) | population margins by state × urban × age × sex | tables only | free | M6 raking margins India. **Caveat: 2011; no 2021 census exists.** Use official projections; state the limitation |
| **WVS wave 7 India** | values battery | n ≈ 1,600–2,000 | free | optional extra India targets; small n → cluster-level truth is noisy; use at coarse clusters only |

Rejected: CMIE Consumer Pyramids (not free), NSSO older rounds (superseded by HCES/PLFS for our needs), IPUMS-International India (census extracts — demographics only, and Census tables suffice for margins).

**Data-flow rule:** stat cards for India runs are built from NFHS-5 + IHDS-II (+ HCES enrichment); targets come from Pew India and held-out NFHS attitudinal items. US stat cards from GSS anchors + ACS margins; targets = held-out GSS items and ANES items (cross-dataset transfer: GSS-anchored agents predicting ANES items is a strong generalization test — same population, different instrument).

---

## 5. Validation protocol

### 5.1 Item split
Pool all attitudinal/behavioral items with scale length 2–7. Compute per-item heterogeneity h = weighted between-cluster variance share. Stratify by topic × heterogeneity quartile; assign 60 % anchors / 40 % targets. Two split regimes: **standard** (stratified random) and **adversarial** (entire topics held out — measures true out-of-domain transfer). All results reported under both.

### 5.2 Ground truth
Per-cluster weighted histograms from the microdata itself (`ClusterStats` on target items) — never shown to any LLM path. Cells with n_eff < 30 are excluded from scoring (truth too noisy).

### 5.3 Metrics (all distributional; means alone are banned)
- **W1** — Wasserstein-1 between predicted and true histograms on the normalized ordinal scale (Santurkar et al. 2023 use the same family → comparability). Primary. Reported cluster-weighted, per item, and macro-averaged.
- **JS** — Jensen–Shannon divergence for nominal scales.
- **Variance ratio** — predicted SD / true SD, at cluster level and population level. Target 1.0; Bisbee-style collapse shows as ≪ 1.
- **Between-cluster fidelity** — Spearman correlation between predicted and true cluster means across clusters, per item ("does the model order subgroups correctly"); plus predicted vs. true between-cluster SD (detects F2 directly).
- **Calibration curves** — pool (cluster × item × option) cells; predicted proportion vs. empirical; report ECE.
- **Interval coverage** — fraction of true values inside bootstrap 90 % intervals.

### 5.4 Baselines
| ID | Baseline | What it kills if it wins |
|---|---|---|
| B0 | **National marginal**: every cluster = population distribution of that target item's *true* population marginal is unknown for targets, so B0 = population-level prediction from… | see note |
| B0a | National marginal *oracle* (true population marginal copied to every cluster) — an unfair-strong variant; beating it on cluster-level W1 requires real between-cluster signal | the entire thesis |
| B0b | National marginal *predicted* (LLM predicts one population-level histogram; copied to all clusters) — the fair version | cluster conditioning |
| B1 | Nearest-anchor heuristic: copy each cluster's histogram from its most semantically similar anchor item | LLM reasoning adds nothing over item similarity |
| B2 | **Per-individual silicon sampling** (Argyle-style): sample respondents, one persona prompt each, sampled answers → cluster histograms; run at (a) equal query budget, (b) 20× budget | the efficiency claim |
| B3 | **Supervised skyline**: gradient boosting on demographics → response, trained on 50 % of individuals' *actual* target-item labels | nothing — it uses labels we assume absent; it locates the ceiling |
| B4 | Uncalibrated cluster agent (raw M4 output) | the calibration mechanism (ablation) |
| B5 | Point-valued cluster agent (forced single answer, ×9 samples → histogram) | distribution-valued elicitation (ablation) |

**Note on B0:** for *cluster-level* scoring the operative trivial baseline is B0a/B0b — "no subgroup differentiation." §1.4's claim is against exactly this.

### 5.5 Ablations
No-calibration (B4); point-valued (B5); no-anchor stat card (persona text only — the Argyle regime at cluster level); no-hierarchy (flat K clusters, no pooling: isolates partial pooling's contribution); K sweep ∈ {12, 50, 150, 400}; ensemble size sweep {1, 3, 9}; model tier (Haiku-class vs Sonnet-class on 10-item subset); logprob vs verbalized elicitation (where API permits).

### 5.6 Leakage control (F6)
(1) **Direct probe:** ask each model for the actual cross-tab of each target item by name/wording ("In the 2022 GSS, what fraction of college-educated women said…"); items where the probe is accurate within W1 < 0.05 are flagged `leaky` and all headline metrics are reported with and without them. (2) Prefer newest waves (GSS 2022 is likely partially in pretraining; GSS 2024, if released by the time of the run — status uncertain as of early 2026 — becomes the preferred target bed and should be swapped in). (3) The adversarial topic-held-out split bounds memorization gains. Leakage is discussed as a limitation regardless — this is a known unsolved problem for the whole silicon-sampling literature, and saying so plainly buys credibility.

### 5.7 Cost–accuracy curves
X-axis: total LLM tokens (measured, from `cost.py`). Y-axis: macro W1. Curves: cluster agents at K ∈ {12, 50, 150, 400}; per-individual sampling at {200, 1k, 5k} respondents. The scalability claim in the writeup is exactly and only what these curves show.

### 5.8 Abandonment / downgrade criteria (pre-registered)
- Core kill: claim (i) of §1.4 fails on the standard split → approach abandoned as scientific contribution; Stage A demo survives as engineering only and the writeup pivots to a negative-result analysis of *why* cluster conditioning fails (still a defensible paper).
- Calibration kill: B4 ≥ calibrated on W1 and variance ratio → mechanism is dead weight; report and drop.
- Dynamics kill: §M8 pre-registered persistence test fails → dynamics reported as negative result; paper stands on statics.

---

## 6. Staged roadmap

### Stage A — working prototype (weeks 1–8) — demo to panel
**Essential path (must-have):** M1 (GSS + NFHS-5 adapters only), M2, M3, M4, M5 (isotonic + variance restoration only, single fold), M6, M7 router (threshold, no tuned recall), M10 app. Minimal eval: W1 + variance ratio vs B0b on a 20-item standard split, one country.
**Explicitly deferred:** M8 entirely; B2/B3; ablations; adversarial split; leakage probe; ANES/IHDS/Pew adapters.

| Task | Owner | Depends on | Effort (person-weeks) |
|---|---|---|---|
| A1 GSS adapter + codebook + topline checks | P1 | — | 2 |
| A2 NFHS-5 adapter (access request week 1!) | P1 | — | 2 |
| A3 Cluster tree + stats + merge rule | P2 | A1 | 1.5 |
| A4 LLM client, cache, budget guard | P4 | — | 1 |
| A5 Stat cards + elicitation + paraphrase set | P2 | A3, A4 | 2 |
| A6 Calibration v1 (crossfit, isotonic, variance) | P3 | A5 | 2 |
| A7 Router + scenario compiler | P4 | A1 | 1 |
| A8 Aggregation + raking + bootstrap | P3 | A6 | 1.5 |
| A9 Streamlit app + report + honesty box | P4 | A8 | 2 |
| A10 Mini-eval (W1, variance ratio, B0b) | P3 | A6 | 1 |

Panel demo script: author one India product scenario + one US policy scenario live; show segment divergences and the honesty box.

### Stage B — research contribution (weeks 7–16, overlapping)

| Task | Owner | Depends on | Effort (pw) |
|---|---|---|---|
| B1 ANES + Pew India + IHDS adapters; ACS/Census margins | P1 | A1 | 2.5 |
| B2 Full eval harness: all metrics, both splits | P3 | A10 | 2 |
| B3 Baselines B0–B5 incl. per-individual sampling | P3 + P2 | B2 | 2.5 |
| B4 Leakage probe + leaky-item reporting | P2 | B2 | 1 |
| B5 Ablation battery + K/ensemble/model sweeps | P2 | B3 | 2 |
| B6 Cost–accuracy curves | P4 | B3 | 1 |
| B7 Dynamics: graph, FJ, param elicitation | P4 | A6 | 2.5 |
| B8 Panel validation (GSS panel, ANES 2020→22) | P4 + P1 | B7, B1 | 2 |
| B9 India replication run (NFHS/Pew targets) | P1 + P3 | B1, B2 | 1.5 |
| B10 Writeup: claims, figures, limitations | all | B5, B6, B8, B9 | 3 |

Total ≈ 18 pw (Stage A) + 20 pw (Stage B) ≈ 38 person-weeks against 4 people × 14–16 weeks ≈ 56–64 available — leaves honest slack for the 30 % overrun that always happens, mid-semester exams, and the mentor-review cadence.

---

## 7. Risks and honest assessment

### 7.1 Compute / API cost (realistic numbers, assumptions stated)
Main US run: 150 clusters × 60 targets × 9 ensemble ≈ 81k calls; ~1.3k input + 250 output tokens each ≈ 105 M in / 20 M out. At Haiku-class pricing (~$1/M in, $5/M out): **≈ $205**. Cross-fit anchor elicitation ≈ same again. India run at K = 400, 25 targets: ≈ $150. Per-individual baseline (5k respondents × 20 items): ≈ $120. Ablations + sweeps ≈ 1.5× main run. Frontier-model subset ≈ $60. Dynamics params ≈ $30. **Realistic total: $700–1,000; hard cap $1,200 enforced by `budget.py`; batch APIs (50 % discount) bring the expected spend to ~$400–600.** This is fundable by a department or a single API grant; say the number in the review rather than hand-waving.

### 7.2 Where it most likely fails
1. **F2 (between-cluster collapse) survives calibration.** Most likely failure. Calibration can restore variance but cannot invent subgroup signal the base model refuses to produce. Early-warning: run the 20-item mini-eval in week 6; if predicted between-cluster SD < 50 % of truth, escalate (frontier model for elicitation, richer anchor cards) before Stage B commits.
2. **Dynamics null result.** Probable (persistence is strong over 2-year panels). Pre-registered fallback exists; the project does not depend on it.
3. **NFHS access or IHDS-3 timing.** Mitigation: request NFHS week 1; IHDS-2 (already public) is the committed fallback; IHDS-3 is upside only.
4. **Leakage discount.** Some headline gains may vanish on the leaky-item-excluded view. We report both views; the adversarial split is the defense.
5. **Sensitive-attribute simulation (India).** Conditioning on caste/religion and publishing "what this group thinks" is ethically loaded. Rules baked in: aggregate-only outputs, no synthetic individuals, neutral statistical elicitation framing, refusal-rate reporting, and a limitations section written with the mentors. This also matters for the review panel.

### 7.3 What "revolutionise the industry" actually supports
It does not. Here is what is defensible at this scale, and it is worth having:

- **Defensible claim 1 (the paper's spine):** *Cluster-level, distribution-valued LLM elicitation with cross-fitted anchor calibration recovers held-out subgroup response distributions better than persona sampling at a fraction of the cost, and restores the variance that persona methods collapse.* — New mechanism, new evaluation, direct numbers against Argyle/Bisbee-style baselines.
- **Defensible claim 2:** *First distributional-fidelity evaluation of LLM population simulation on Indian microdata (NFHS-5/Pew), with documented failure modes.* — Under-served angle; genuinely useful to the field even where results are mixed.
- **Defensible claim 3 (conditional):** *LLM-parameterized mechanistic influence dynamics evaluated against panel ground truth* — publishable even as a well-executed negative result.

**Claims that will not survive review — do not make them:** "simulates society"; "predicts policy outcomes"; "replaces surveys" (the system is *built on* surveys and dies without them); "scales to millions of agents" (Chopra owns that; ours is a fidelity-per-dollar claim at K ≤ a few hundred, shown by §5.7 curves only); any dynamics claim not backed by the panel test; any accuracy claim on synthetic scenarios (only the honesty-box proxy is defensible).

Realistic venue for the Stage B result: an IC2S2 / NeurIPS-workshop / CSCW-adjacent submission or a solid applied preprint — not a main-track ML paper on the first pass. One sharp, pre-registered, distributionally-validated result with honest leakage handling will read better than any grand framing, including to the review panel.

---

## Appendix — pinned config defaults (single source: `configs/*.yaml`)
`min_cell: 40` · `K_target: 150 (US) / 400 (India)` · `anchors_per_card: 12` · `n_paraphrase: 3` · `n_repeat: 3` · `temperature: 0.7` · `crossfit_folds: 3` · `tau_pooling: 100` · `router_threshold: 0.85` · `n_boot: 500` · `truth_min_neff: 30` · `budget_cap_usd: 1200` · primary elicitation model: current Haiku-class; comparison: current Sonnet-class; all model IDs pinned per run in config.
