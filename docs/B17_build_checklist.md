# B17 — The question, the verification, and the build checklist

**15 Sep 2026.** Companion to `B17_architecture_spec.md`, `B17_feasibility_verdict.md`.

Part 1 says what the project actually answers. Part 2 says how you know an answer is
right, with the pass marks fixed in advance. Part 3 is the build, step by step, with a
gate at every point where being wrong is still cheap.

All numbers measured from `data/` on 15 Sep. Method in `B17_feasibility_verdict.md` §6.

---

# Part 1 — What question does this answer?

## 1.1 The scientific question — one sentence

> **Given a demographic subgroup described only by its population marginals and its
> answers to a set of *other* survey questions, what is the full distribution of its
> answers to a question it was never asked?**

Not "what does this group think on average" — the **whole histogram**, because the
distribution is the thing persona-prompting gets wrong (Bisbee's variance collapse), and
the thing a cross-tab cannot give you for an unasked question.

Everything else in the spec is scaffolding for this one sentence.

## 1.2 The operational form — fixed by measurement

The feasibility numbers pick the config; do not treat these as tunable:

| Setting | Value | Why this and not the spec's |
|---|---|---|
| **Bed** | GSS pooled **2010–2022** | 74 items clear the noise gate vs 59 on 2016–2022. 2024 excluded — `region_7222` is 0 % populated, complete-demo n = 0 |
| **Partition** | `age_band × degree × sex`, min_cell 60 → **K ≈ 56** | Adding urban/rural drops usable items from 74 to **35**. Geography goes in the stat card as a marginal, never in the partition |
| **Scorable pool** | **74 items** at SNR ≥ 1.5 | Measured. 56 clear SNR ≥ 2.0 |
| **Split** | **40 targets / 34 anchors**, + 14 low-SNR items as anchor-only context | §5.1's 60/40 would leave 29 targets — short of the ≥ 40 the §1.4 claim needs. Flip the ratio; anchors need not be high-SNR because they are context, not scored |
| **Dynamics bed** | `age3 × edu3 × sex`, **K ≈ 18** | 79 items clear SNR ≥ 1.5 at coarse K. Matches `B17_data_review.md`'s coarse-cluster recommendation |

**Change to `configs/gss_main.yaml`:** `K_target: 150` → `56` · `min_cell: 40` → `60` ·
add `min_item_snr: 1.5` · partition axes lose `region_7222` and `urban`.

## 1.2a What this costs to run — and why it is free

One call ≈ 1.3k input (stat card + question) + ~250 output ≈ **1.55k tokens**.

| Stage | Calls | Tokens |
|---|---|---|
| Permutation test (Gate 3) | ~1,700 | 2.6M |
| Mini-eval, 20 items (Gate 5) | 10,080 | 15.6M |
| Full core run (40 targets + 34 anchors, ×9) | 37,296 | 58M |
| Everything incl. ablations | ~110k | ~175M |

Mistral's free tier alone (~1B tokens/month) covers the last row five times over.

**Stage the ensemble — it is the biggest multiplier.** `n_paraphrase 3 × n_repeat 3 = 9`:

| Phase | paraphrase × repeat | why |
|---|---|---|
| plumbing / development | **1 × 1** | local model, unlimited |
| permutation test (Gate 3) | **1 × 3** | enough to see collapse |
| mini-eval (Gate 5) | 3 × 3 | the verdict needs the real spread |
| headline run | **3 × 3** | as specified |

That is a **9× swing** on quota between development and the final run.

## 1.3 The best single item to lead the demo with

Top of the measured table, K = 53, pooled 2016–2022:

| item | what it asks | baseline W1 | noise | SNR | gradient |
|---|---|---|---|---|---|
| **`natroad`** | spending on highways and bridges | 0.087 | 0.023 | **3.61** | age 0.71 |
| `finrela` | family income vs. average | 0.073 | 0.022 | 3.21 | **education 0.93** |
| `satfin` | satisfaction with finances | 0.098 | 0.029 | 3.18 | education 0.80 |
| `pray` | frequency of prayer | 0.109 | 0.026 | 4.04 | age 0.60 |
| `homosex` | same-sex relations wrong? | 0.122 | 0.037 | 3.15 | age 0.66 |

**Lead with `natroad`, not `pray`.** `pray` has the highest SNR, but prayer, abortion,
`homosex` and `polviews` are the most cross-tabbed items in the history of social science
— maximum §5.6 leakage exposure. A reviewer's first question about a strong result on
`pray` is "did it know, or did it infer?", and you cannot fully answer that.

`natroad` has almost the same SNR, a clean age gradient, and nobody has ever written a
think-piece about highway-spending attitudes by education band. When the system gets
`natroad` right, inference is the only available explanation. **That is what makes it the
best demo item: not that it is the most impressive, but that it is the least deniable.**

## 1.4 Which items the model will actually score best on — and the trap in that list

Ranked by *learnability*: strong monotone demographic gradient, low truth-noise, few
answer options. This predicts where the model should do well; it is a proxy, not a
measurement, until the first elicitation run.

| item | what it asks | gradient | noise | note |
|---|---|---|---|---|
| `finrela` | family income vs. average | **edu 0.93** | 0.022 | ⚠️ see below |
| `satfin` | satisfied with your finances? | edu 0.80 | 0.029 | ⚠️ see below |
| **`spkhomo`** | should a gay man be allowed to speak? | edu 0.85 | 0.024 | binary, sharp split |
| **`colhomo`** | …allowed to teach at a college? | edu 0.80 | 0.023 | |
| **`natroad`** | too much/too little spending on highways? | age 0.71 | **0.023** | lowest noise in the set |
| **`consci`** | confidence in the scientific community | edu 0.82 | 0.031 | |
| **`spkcom`** | should a Communist be allowed to speak? | edu 0.82 | 0.036 | |

### ⚠️ The `finrela` / `satfin` trap

They will score highest and they are the **worst** items to headline. Neither is really an
opinion question — both are income restated, and the clusters are defined partly by
education, which the model maps to income cold. A strong result there invites exactly the
§F3 objection the oracle router exists to answer: *you predicted income from education and
called it population simulation.*

**Use them as a pipeline sanity check — "if we cannot get `finrela` right, something is
broken" — and never as evidence.** Mark them `role: sanity` in the codebook so they cannot
leak into a headline table by accident.

### ✅ The tolerance battery — build on this

Fifteen items with identical structure: should [an atheist · a racist · a Communist · a
militarist · a gay man] be allowed to **speak** (`spk*`), **teach** (`col*`), or **have a
book in the library** (`lib*`).

`spkath colath libath · spkrac colrac librac · spkcom colcom libcom · spkmil colmil libmil · spkhomo colhomo libhomo`

Why this is the strongest family available:

- **12 of the 15 clear the SNR ≥ 1.5 gate**, all binary, all low-noise
- The model must get the **ordering across the five target groups** right, not just each
  level — a second signal, free, and far harder to fake than one histogram
- The **3 × 5 grid structure** means a systematic error shows up as a visible pattern
  (a whole row or column off) rather than as scattered noise

Measured, K = 53:

| target group | `spk*` | `col*` | `lib*` | education gradient |
|---|---|---|---|---|
| atheist | 1.67 | 2.73 | 1.85 | 0.73–0.74 |
| Communist | **2.97** | 1.90 | 2.35 | 0.79–0.82 |
| militarist | 2.16 | **1.48** ✗ | 1.73 | 0.53–0.69 |
| gay man | 2.19 | 2.16 | 2.33 | **0.78–0.85** |
| racist | 1.59 | **0.64** ✗ | **1.21** ✗ | **0.15–0.42** |

*(cells are SNR; ✗ = below the 1.5 gate)*

**Drop `colrac`, `librac`, `colmil`** and say why. Note the substantive pattern in the
last row: tolerance toward racists barely tracks education (0.15–0.42) where tolerance
toward atheists, Communists and gay people tracks it strongly (0.73–0.85). That is a real
finding about the item, not a defect — but it means **the racist column is the one place
the battery has little subgroup signal to predict**, so do not read a weak result there as
model failure.

**Add a battery-level metric:** Spearman correlation between predicted and true group
ordering, per cluster. A model that reproduces "atheist > Communist > racist" tolerance
ordering *within* each education band is doing something a national marginal cannot.

## 1.5 The leakage-resistant target set — use this

Of the 74 that clear the gate, these 32 are outside the famous battery:

`natroad, finrela, satfin, spkcom, colath, natarms, natsoc, libcom, pornlaw, libhomo,
consci, spkhomo, colhomo, spkmil, conmedic, conlegis, suicide1, natspac, natenvir,
sexeduc, spanking, postlife, colcom, libath, natmass, libmil, spkath, conlabor, spkrac,
nateduc, fair, life`

Report headline numbers **both ways** — all 40 targets, and this leakage-resistant subset.
If the two agree, §5.6 stops being a weakness and becomes a strength.

## 1.6 What this does *not* answer, said out loud

Product and policy scenarios ("would gig workers adopt a UPI micro-pension?") have no
ground truth and never will. The held-out item score is the honest proxy, the honesty box
in M10 is where you say so, and §7.3's list of forbidden claims stands. Do not let a good
W1 number drift into a claim about scenarios.

---

# Part 2 — How you verify a prediction is correct

Six layers. Each one catches a failure the layer below cannot see. **Layers 0–2 are gates:
if one fails, stop and fix — do not proceed and report.**

## Layer 0 — Does the data pipeline reproduce reality? *(gate)*

Match harmonized marginals against the published GSS toplines in the codebook PDFs on
disk, within 0.5 pp, on at least 10 items plus one full cross-tab.

*Catches:* recode errors, weight misuse, scale reversals. These invalidate everything
downstream and are invisible later — a reversed 5-point scale produces plausible,
completely wrong numbers forever.

## Layer 1 — Is the ground truth itself trustworthy? *(gate)*

For every (cluster, item) cell, estimate sampling noise in the **truth** by split-half
resampling. Gate on SNR ≥ 1.5. Measured floors:

| Config | K | mean noise W1 | items clearing gate |
|---|---|---|---|
| 2010–2022, age × edu × sex, mc 60 | 56 | 0.0273 | **74** |
| 2016–2022, same | 53 | 0.0345 | 59 |
| coarse, age3 × edu3 × sex | 18 | 0.0219 | 79 |
| **+ urban/rural, mc 80** | 71 | 0.0428 | **35** |

Note row 4 again: geography in the partition costs you more than half your item bank.

*Catches:* scoring cells where truth is noise, which produces a confident wrong verdict in
either direction. Absent from the spec — `truth_min_neff: 30` is a crude proxy for it.

## Layer 2 — Is the model reading the stat card at all? *(gate — run in week 4)*

**The permutation test.** Shuffle which stat card goes to which cluster. Re-run on 10
items. If W1 does not degrade sharply, the model is ignoring the conditioning and
producing one hedged answer per item — **F2, the failure mode §7.2 calls the most likely
one.**

*Expected:* permuted W1 should land at or above the B0a baseline (≈ 0.070). If permuted
and real W1 are within noise of each other, the project's core premise is not working and
you have ten weeks to respond, not two. **This is the single most valuable test in the
document and it costs about 40 LLM calls.**

## Layer 3 — Is the calibration layer arithmetically sound?

**The oracle pass-through.** Feed the *true* per-cluster histogram in where the LLM
histogram goes. Calibrated output must come back ≈ truth.

*Catches:* off-by-one option ordering, misaligned scale codes, sign errors in the variance
tempering, isotonic fits on the wrong axis. These masquerade as "the model is bad at this
item" and can burn weeks. Ten lines of code.

## Layer 4 — Does it beat the baseline on held-out items? *(the actual claim)*

Pre-registered pass marks, measured at K ≈ 56 on pooled 2010–2022:

| Item set | B0a baseline W1 | **pass mark (−20 %)** | ceiling (noise floor) |
|---|---|---|---|
| All 40 targets | 0.0695 | **≤ 0.0556** | 0.0273 |
| Top-quartile heterogeneity | 0.1043 | **≤ 0.0834** | 0.0402 |
| Leakage-resistant subset | 0.0767 | **≤ 0.0614** | 0.0324 |

Plus §1.4(ii): population variance ratio in **[0.8, 1.2]**.

Write these into `configs/gss_main.yaml` **before** the first elicitation run. A pass mark
chosen after seeing results is not a pass mark.

## Layer 5 — Is the win real, or memorized?

Direct leakage probe (§5.6), the leakage-resistant subset from §1.4 above, and the
adversarial topic-disjoint split. New and cheap: **leakage vs. wave age** across 35 GSS
waves, 1972–2024 — a test nobody else in this literature can run.

## The honest reporting rule

Every headline number carries three figures, never one: **achieved W1, the B0a baseline,
and the noise floor.** "W1 = 0.052" means nothing. "W1 = 0.052 against a baseline of 0.070
and a floor of 0.027" says you captured 42 % of the available signal — and that is a
sentence a reviewer can check.

---

# Part 3 — The build checklist

Phases are sequential; steps inside a phase can run in parallel. **Each gate must pass
before the next phase starts.**

## Phase 0 — Scaffold *(before any pipeline code)*

- [ ] **0.1** Repo skeleton per §3.2. `pyproject.toml`, `popsim/`, `configs/`, `runs/`, `tests/`
- [ ] **0.2** Config loader — one YAML fully specifies a run; snapshot it into `runs/<id>/`
- [ ] **0.3** Pin deps. Verified working: `pyreadstat 1.3.6 · pandas 2.3.3 · numpy 2.2.6 · pyarrow 25.0.1 · scipy 1.15.3 · scikit-learn 1.7.2 · jinja2 3.0.3 · pyyaml 6.0.3 · streamlit 1.63.0`
- [ ] **0.4** `llm/client.py` + `cache.py` (keyed on model+prompt hash) + `budget.py`
  - *Build this before anything that calls an LLM.*
  - [ ] **0.4a** **Provider router.** The project runs on free tiers (§0.7), which are **rate-capped per day, not per dollar**. The client picks a provider by remaining daily quota and falls through on 429
  - [ ] **0.4b** **Per-provider daily counter**, persisted to disk, reset on local midnight
  - [ ] **0.4c** **Checkpoint after every single call.** A 37k-call run at ~1k/day takes weeks — the runner must survive being stopped and resumed indefinitely. `cache.py` is what makes that free, so it is load-bearing, not a nicety
  - [ ] **0.4d** `budget.py` still enforces a **call-count** ceiling per run even at zero cost — it is the guard against a loop bug burning a day's quota in ten minutes
- [ ] **0.5** Test harness + CI that runs Layer 0 and Layer 1 on every commit
- [ ] **0.6** **Local model for development.** Ollama + Qwen 2.5 7B or Llama 3.1 8B. Every pipeline bug — JSON parsing, cache keys, cross-fit plumbing, calibration math — is found with a free unlimited local model, not with quota
- [ ] **0.7** **Register the free API tiers** (verify current limits in each console; they move):

| Provider | Free allowance | Role |
|---|---|---|
| **Mistral** | ~1B tokens/month, no card | **Primary.** Covers the whole project several times |
| **Groq** | ~1,000 req/day | Second model → tier ablation |
| **Google AI Studio** | up to ~1,500 req/day | Third arm |
| Cerebras / GitHub Models / NVIDIA NIM | ~1M tok/day · 150–1k req/day · ~1k req/day | Overflow |

  **Budget required: $0.** Set `budget_cap_usd: 0`.

- [ ] **0.8** **Ask the FYP supervisor about department GPU access.** A 14–32B model on real GPUs removes the rate-limit problem entirely *and* cuts the F2 risk that a 7B carries

**Gate 0:** a no-op run writes a config snapshot to `runs/`, the call-count guard kills a deliberate overrun, and the provider router fails over cleanly on a simulated 429.

## Phase 1 — Data foundation

- [ ] **1.1** GSS adapter → `IndividualTable`. `encoding='latin1'`, `usecols=` subsets — never read the 598 MB file whole
- [ ] **1.2** Harmonized `demo` struct. Drop `caste_group` (no source on disk). Alias `region_7222`→`region`, log which was used per row
- [ ] **1.3** **Exclude GSS 2024** with a logged reason, and assert the exclusion in a test
- [ ] **1.4** Item codebook YAML: `item_id`, scale, labels, topic, wave coverage, `role: anchor|target|sanity|excluded`
- [ ] **1.4a** **Exact question wording transcribed from the GSS 2022 codebook PDF — never paraphrased.** Elicitation prompts must carry the real instrument text; a reworded item is a different item and silently invalidates the comparison to published work
- [ ] **1.4b** Mark `finrela` and `satfin` as `role: sanity` so they cannot reach a headline table (§1.4)
- [ ] **1.5** 🚦 **Layer 0 gate** — toplines match published within 0.5 pp on ≥ 10 items + 1 cross-tab
- [ ] **1.6** GFS India adapter (`COUNTRY = 6`, weight `ANNUAL_WEIGHT_R2`, `_Y1` items). India keeps geography — it can afford it
- [ ] **1.7** OpinionQA adapter — 15 ATP waves, individual-level. Buys Santurkar comparability free

**Gate 1:** `published_topline_check` green on GSS and GFS.

## Phase 2 — Clustering and the item split

- [ ] **2.1** **Replace build-then-merge.** Partition targets K directly — supervised tree with a min-leaf constraint, splitting on anchor-response variance
  - *Why:* the spec merges on Hellinger distance between histograms built from 1.6-person cells. Undefined, not a tuning issue
- [ ] **2.2** `ClusterStats` — weighted histograms, Kish `n_eff`, mean, SD per (cluster, item)
- [ ] **2.3** **`evalx/noise.py`** — split-half noise floor per cell. *New module, not in the spec*
- [ ] **2.4** Item split: 40 targets / 34 anchors, SNR-gated, topic-stratified. Flag famous vs leakage-resistant. Persist the split — it must never be regenerated
- [ ] **2.5** Partial pooling toward parent, `τ = 100`
- [ ] **2.6** 🚦 **Layer 1 gate** — ≥ 40 targets clear SNR ≥ 1.5 (measured: 74 available)

**Gate 2:** K ≈ 56, ≥ 40 scorable targets, split frozen to disk.

## Phase 3 — Elicitation

- [ ] **3.1** `statcard.jinja`. Geography as a **marginal line** ("58 % South, 31 % urban"), not a constraint. Leaves are disjunctions — render them honestly
- [ ] **3.2** 3 hand-written paraphrases, checked in, never model-generated at runtime
- [ ] **3.3** `elicit.py` — structured JSON out, 3 paraphrases × 3 repeats, temp 0.7, retry ≤ 2, log refusals
- [ ] **3.4** Contract test: a card for item Y contains no Y, no near-duplicate, no same-fold anchor
- [ ] **3.5** 🚦 **Layer 2 gate — the permutation test.** 10 items, shuffled cards. ~40 calls
  - **If permuted ≈ real: stop. Escalate before Phase 4.** Frontier model, richer cards, more anchors — in that order

**Gate 3:** permuted W1 ≥ baseline; real W1 measurably better. This is the project's first real evidence.

## Phase 4 — Calibration *(the novel contribution)*

- [ ] **4.1** Cross-fit plan, F = 3, stratified by topic
- [ ] **4.2** Anchor elicitation under target conditions (cards built from folds ≠ f)
- [ ] **4.3** Isotonic recalibration on cumulative probabilities; Dirichlet for nominal
- [ ] **4.4** Variance restoration — `sd_true = α + β·sd_raw`, power-tempering, per-cluster (α, β) shrunk to parent
- [ ] **4.5** 🚦 **Layer 3 gate — oracle pass-through.** True histogram in → ≈ truth out
- [ ] **4.6** Ensemble aggregation after step 4.3; retain spread as `elicit_sd`

**Gate 4:** oracle round-trips within 0.005 W1.

## Phase 5 — Aggregation, baselines, the verdict

- [ ] **5.1** B0a (true marginal copied) and B0b (LLM-predicted marginal copied)
- [ ] **5.2** Metrics: W1, JS, variance ratio, between-cluster Spearman, ECE, coverage
- [ ] **5.2a** **Tolerance-battery ordering metric** (§1.4): Spearman between predicted and true ordering of the five target groups, per cluster. Exclude `colrac` (SNR 0.64) and say why
- [x] **5.3a** ✅ **ACS 2024 margins acquired and verified** (15 Sep). `data/acs/acs2024_margins.parquet` — 60 cells on `age_band × degree × sex`, 267.2M adults; plus `_division` (540 cells) and `_geo` (9 divisions) for the stat-card geography marginal
  - SCHL→`degree` recode validated against GSS 2016–2022: **max gap 2.9 pp** (education), 2.3 pp (age), 0.2 pp (sex) — all in the direction the 2-to-6-year time lag predicts. The recode is correct
  - ⚠️ Margins **must** stay on the same category scale as the cluster axes or raking is meaningless. If the GSS adapter changes its `degree` scale, regenerate these
- [ ] **5.3** Aggregation + bootstrap CIs, raking to `acs2024_margins.parquet` via IPF. Keep the survey-weight fallback for non-convergence
- [ ] **5.4** 🚦 **Layer 4 gate — the mini-eval.** 20 items. Pass marks from Part 2, written into the config *first*
- [ ] **5.5** Report achieved / baseline / floor as a triple, always

**Gate 5 — the go/no-go.** Pass → Stage B. Fail → §5.8 pivot to the negative-result paper, which is still a paper. **Reach this gate by week 7.**

## Phase 6 — The product surface

- [ ] **6.1** Router. **Start with TF-IDF + cosine** — HuggingFace is blocked in the sandbox and §M7 says the embedder is not load-bearing
- [ ] **6.2** Hand-label ~100 duplicate pairs; tune threshold; report recall
- [ ] **6.3** Scenario schema + compiler; synthetic items carry `provenance: simulated`
- [ ] **6.4** Streamlit app
- [ ] **6.5** **Honesty box — no report renders without it.** Nearest validated items + their W1 + the baseline
- [ ] **6.6** Panel demo script: one US policy scenario, one India scenario, both showing the box

## Phase 7 — Stage B

- [ ] **7.1** B1 nearest-anchor, B2 per-individual silicon sampling (equal + 20× budget), B3 supervised skyline, B4 uncalibrated, B5 point-valued
- [ ] **7.2** SubPOP as a published fine-tuned reference point (§5.4 has none)
- [ ] **7.3** Ablations: no-calibration, no-anchor, no-hierarchy, K ∈ {18, 36, 56, 100}, ensemble {1,3,9}
- [ ] **7.3a** **Model-tier ablation — open 7B vs Groq-class vs Mistral-large-class.** Running on free tiers means you get this for free, and it answers a question reviewers care about: *does the method need frontier access, or does the calibration layer rescue a weak model?* A yes is a stronger result than making a strong model slightly better
- [ ] **7.3b** **Logprob vs verbalized elicitation.** §5.5 notes logprobs are unavailable on several frontier APIs — a **local model exposes full token logprobs**, so the local arm *enables* this experiment rather than merely being cheaper
- [ ] **7.4** Leakage probe + **leakage vs. wave age across 35 waves** — the novel side-finding
- [ ] **7.5** Adversarial topic-disjoint split
- [ ] **7.6** Cost–accuracy curves from measured tokens
- [ ] **7.7** Dynamics on **repeated cross-sections with horizon as a variable**, K ≈ 18. Not the 2-year panel
- [ ] **7.8** India replication on GFS. Upgrade to NFHS/Pew if the registrations land
- [ ] **7.9** Writeup: claims, figures, limitations, and §7.3's list of claims *not* made

---

## Ordering notes

**Three things are early because being wrong about them later is expensive:**
the budget guard (0.4) before any LLM call · the noise floor (2.3) before any scoring ·
the permutation test (3.5) before any calibration work.

**One thing changed shape:** the project runs on free rate-capped tiers, not a dollar budget, so the constraint is *calls per day*, not spend. That makes resumability (0.4c) and ensemble staging (§1.2a) the two decisions that determine whether the schedule works.

**Two things are deliberately late:** raking, because ACS PUMS is not on disk and the
fallback is documented; and the neural embedder, because TF-IDF is sufficient and more
defensible.

**One thing is not on the critical path at all:** the India registrations. Start them —
DHS is 24–48 h and has been open since 31 Aug — but build on GFS so nothing waits.
