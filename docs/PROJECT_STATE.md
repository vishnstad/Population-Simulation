# B17 — PROJECT STATE

**The living record for this build. Read this first; it supersedes `HANDOFF.md`.**
`STATUS.md` is the long-form history, `PREREGISTRATION.md` is the register of every
configuration that has been run and why. This file is what you paste into a new
session.

Last updated: **17 Sep 2026** — session 2, complete. All arms scored.

---

## 1. What the project is, in one page

Split the US adult population into ~56 demographic groups (`age_band × degree × sex`,
min 60 respondents per cell, from GSS pooled 2010–2022). Give each group a **stat
card**: its population share, its demographic marginals, and its *real* answers to 12
survey questions we have data for (the **anchors**). Show that card to an LLM and ask it
to estimate, as percentages, how that group would answer a **different** survey question
it was never shown (a **target**). Pass the raw output through a **calibration layer**
fitted on questions where the truth is known, then score against the real GSS answers
for those groups, which were held out.

**The falsifiable claim** (checklist Part 2, Layer 4 — pre-registered, never moved):

| item set | B0a baseline W1 | pass mark (−20 %) | noise floor |
|---|---|---|---|
| all 40 targets | 0.0695 | **≤ 0.0556** | 0.0273 |
| top-quartile heterogeneity | 0.1043 | **≤ 0.0834** | 0.0402 |
| leakage-resistant subset | 0.0767 | **≤ 0.0614** | 0.0324 |

plus population variance ratio in **[0.8, 1.2]**.

**B0a** = the true national marginal copied to every cluster. Beating it means the
system adds real subgroup structure, which is the only thing a cross-tab cannot give
you for an unasked question.

**The honest-reporting rule:** every headline number carries three figures — achieved
W1, the B0a baseline, and the noise floor. Never one.

---

## 2. Where the project stands — 16 Sep 2026, session 2

| gate | state |
|---|---|
| 0 config snapshot, call-count guard, 429 failover | ✅ |
| 1 Layer 0 — 142/142 published toplines within 0.5 pp | ✅ |
| 2 Layer 1 — K = 56, 140/148 items clear SNR 1.5, split frozen | ✅ |
| 3 Layer 2 — conditioning, **raw** statistic (frozen) | ❌ RED, +0.0011 (14B) |
| 3 Layer 2 — conditioning, **level-matched** statistic | ✅ GREEN, +0.0689 (14B) |
| **4 Layer 3 — the oracle pass-through** | ✅ **GREEN, round-trip 1.1e-16** |
| **5 Layer 4 — the verdict, 8B arm** | 🟢 **relative rule passes on all four subsets; the absolute mark on two** |
| **5 Layer 4 — the verdict, 14B arm** | 🟢 **−25.3 % on all 39 targets; −30.9 % on the top quartile** |

**174 tests pass, lint clean.** Five commits on `main`.

### What was wrong, and what fixed it

Gate 3 was red twice and the cause was thought to be the frozen item split. It is
not. Decomposing the 960 records already on disk (`scripts/forensics.py`, no new
calls) shows the model's error splits into two parts that behave completely
differently:

* **level** — badly wrong and shared across clusters. The 14B puts `spkcom`'s
  group mean at 0.72 where the truth is 0.29; `homosex` at 0.24 against 0.59.
* **structure** — largely right. Between-cluster Spearman **+0.597** macro on the
  14B, 8 of 10 items significant. `xmovie` 0.865, `news` 0.817, `colath` 0.793.

That is the opposite of F2. Raw W1 is dominated by the level error, so the frozen
Gate 3 statistic — a difference between two large, mostly-level numbers — could
never see the structure underneath. Repairing the level is what Phase 4 is for,
and Gate 3 sits *before* Phase 4 in the checklist.

### What the system predicts

```
pred_cdf(c) = level_cdf  +  s · P_r ( raw_cdf(c) − Σ_c w_c · raw_cdf(c) )
```

* **`level_cdf`** — the weighted mixture of the scored cells' truths. **B0a is
  handed the same object**, so `s = 0` reproduces B0a identically and the
  deviation term is its own ablation. (A second mode replaces it with a
  population-level LLM call, for scenario questions that have no topline.)
* **`P_r`** — projection onto the top-`r` principal directions of the anchor
  deviation matrix, built from anchor truths only. Three ensemble draws make the
  deviation mostly noise, and noise is isotropic, so the component outside the
  space real subgroups occupy is discarded. Two directions carry 68 % of the
  anchor variance on this bed.
* **`s`** — one scalar, or one per scale length. The only thing fitted, and it is
  fitted on **anchors**, out of fold.

`r`, the scale scheme and the variance-restoration switch are **all** chosen by
out-of-fold anchor W1 on the F = 3 cross-fit folds. No target truth touches any
of them. `PREREGISTRATION.md` §8.10 froze this before the 14B arm was read.

### Layer 4 — the headline (14B, 12,540 elicitations, 100 % JSON ok)

39 targets × 56 clusters, 1 × 3, observed-level, ACS-raked population weights.

| subset | achieved | B0a here | absolute mark | absolute | relative (−20 %) |
|---|---|---|---|---|---|
| all targets | **0.0602** | 0.0807 | 0.0556 | fail | **PASS** (−25.3 %) |
| top-quartile heterogeneity | **0.0787** | 0.1139 | 0.0834 | **PASS** | **PASS** (−30.9 %) |
| leakage-resistant | **0.0565** | 0.0757 | 0.0614 | **PASS** | **PASS** (−25.4 %) |
| all targets, leaky removed | 0.0611 | 0.0822 | 0.0556 | fail | **PASS** (−25.7 %) |

Population variance ratio **1.009** → §1.4(ii) passes. Interval coverage
**0.899** against a nominal 0.90. Between-cluster Spearman **+0.651** macro,
positive on **36 of 39** items. Tolerance-battery ordering ρ = **+0.879**.
ECE 0.0227.

| arm | W1 | |
|---|---|---|
| B3 supervised skyline | 0.0282 | the ceiling — essentially the noise floor |
| **system** | **0.0602** | |
| system, topic-disjoint calibration | 0.0606 | the adversarial split costs 0.0004 |
| system, s = 1 unfitted | 0.0609 | |
| system, per-cluster scale | 0.0612 | |
| system, no subspace projection | 0.0626 | |
| B0a national marginal | 0.0807 | the baseline |
| B0b LLM-predicted marginal | 0.1787 | |
| B4 uncalibrated | 0.1926 | **the layer cuts the error by 69 %** |
| B1 nearest anchor | 0.2343 | |

**Two verdicts are reported for every subset, always.** The pre-registered
absolute mark (0.0556) came from a B0a of 0.0695 measured on the *feasibility*
bed — 2016–2022, n = 11,045, 88 items. The frozen bed is pooled 2010–2022,
n = 18,772, scored on the 40-item target half, where B0a is 0.0807. Neither is
wrong; they are different estimators. §1.4(i)'s claim is stated relatively, so
both rules are reported and neither can be chosen after the fact
(`PREREGISTRATION.md` §8.8).

### Layer 4 — the 8B development arm

| subset | achieved | B0a here | absolute | relative |
|---|---|---|---|---|
| all targets | 0.0634 | 0.0807 | fail | **PASS** (−21.4 %) |
| top-quartile heterogeneity | 0.0832 | 0.1139 | **PASS** | **PASS** |
| leakage-resistant | 0.0584 | 0.0757 | **PASS** | **PASS** |

Variance ratio 1.008, battery ordering ρ = +0.898. This was the **development
arm**: every design decision was made by looking at it, and `PREREGISTRATION.md`
§8.10 froze the design on it before the 14B was read.

### The capability curve

| arm | achieved | vs B0a | var ratio | battery ρ |
|---|---|---|---|---|
| Ministral 3B | 0.0689 | −14.6 % | 1.008 | +0.876 |
| Ministral 8B | 0.0638 | **−20.9 %** | 1.009 | +0.897 |
| Ministral 14B¹ | 0.0651 | **−27.8 %** | 1.011 | +0.879 |

¹ anchors complete (5,712 records); targets still eliciting, 28 of 39 items done.
Scored at the **full 56 clusters** on those items, where B0a is 0.0901 rather
than the 0.0807 of the complete 39 — the elicitation runs in SNR order, so the
finished items are the most heterogeneous ones and the *level* is not yet
comparable with the rows above. On that scope: top-quartile 0.0810 (**passes
both rules**), leakage-resistant 0.0607 (**passes both rules**), 40 % of the
available signal.

### The K sweep (3B arm, §5.7)

| K | calls | achieved | B0a at that K | vs B0a | `s = 1` unfitted |
|---|---|---|---|---|---|
| 18 | 3,942 | 0.0744 | 0.0744 | −0.1 % | **0.0639 (−14 %)** |
| 36 | 7,884 | **0.0654** | 0.0772 | −15.2 % | 0.0688 |
| 56 | 12,264 | 0.0686 | 0.0807 | −15.0 % | 0.0705 |

**K = 36 is the efficiency point** — the same relative gain for 64 % of the calls.
**At K = 18 the anchor-side selection overfits**: the CV preferred the
per-scale-length scheme, which fitted `s = 2.6` for binary items on a handful of
coarse anchors. No anchor-side rule could catch it, because anchors and targets
differ *by construction* — the 2.4 split put the high-SNR items in the target
half. Partial pooling of the per-k scale (`m/(m+3)`) was added for this and took
K = 18 from +8.7 % to −0.1 %, and improved K = 56 too, but does not close the gap.
This is the sharpest limitation the project has.

Monotone so far, and the 8B's shortfall was predictable from its **anchor** fit
alone before any target was scored — the useful property of an anchor-side
selection rule.

### Ablations (all free from the existing store unless noted)

| sweep | result | reading |
|---|---|---|
| ensemble size 1 / 2 / 3 | 0.0659 / 0.0639 / 0.0638 | the second draw is worth −3 %, the third nothing. The *paraphrase* axis is untested and is the one §F8 cares about |
| subspace rank 0 → chosen | 0.0688 → 0.0638 | the projection is load-bearing |
| standard vs topic-disjoint | 0.0638 vs 0.0641 | **removing every same-topic anchor costs 0.0003 W1** — the transfer result the scenario story needs |
| variance restoration on/off | anchor W1 0.0636 vs 0.0589 | **F1 does not occur here**; the repair has nothing to repair |
| no anchors on the card (new calls) | 0.0707 vs 0.0659 | the demographics are the larger half of the card; anchors add −6.8 % |

### Layer 5 — leakage

2 of 39 items flagged (`conlegis`, `polviews` — the most cross-tabbed item in
social science, which is the sanity check). Removing them moves the headline by
0.0001. The flag requires the probe to recall a *subgroup* within 0.05 **and**
beat copying the topline onto it; a naive best-over-subgroups rule flagged 22 of
39 almost entirely by chance.

### Six places the spec was wrong, each found by measuring

1. Gate 3's statistic is computed on the raw histogram — the *input* to Phase 4,
   not the prediction.
2. M5's variance-restoration step makes anchor W1 worse and is dropped.
3. The router's 0.85 threshold has a recall of **zero** on hand-labeled
   paraphrases. Now 0.25, with recall 0.81 at precision 0.96.
4. The pre-registered pass marks were derived from a baseline measured on a
   different bed. Both verdicts now reported.
5. The subspace projection must be **restricted** to the clusters actually
   present, not zero-padded: a zero asserts a cluster is average. On the 14B at
   16 of 56 clusters, padding turned the projection from a gain into a loss.
6. Amendment 1 to the anchor set was misapplied (fold reshuffle) and is retired.

## 3. Architecture decisions taken this session

1. **The calibration layer works on a level/structure decomposition**, not on the raw
   histogram. Level from the national marginal, structure from the LLM, one scale
   parameter fitted on anchors. Reason: measured, above.
2. **Two reported modes.** *Observed-level* (primary) is the small-area-estimation
   framing — you have a national topline, which is cheap, and want the subgroup
   breakdown, which is not. *Predicted-level* (secondary) replaces the national marginal
   with an LLM population-level call (this is B0b) for genuinely unasked scenario
   questions. Both are reported; neither is presented as the other.
3. **Gate 3 keeps both verdicts on the record.** The frozen raw statistic stays RED in
   every table. The level-matched statistic is reported beside it with the reason they
   differ. Nothing is retracted.
4. **The elicitation prompt and the stat card are frozen as they are.** Any improvement
   there is an ablation arm, never the main path.

## 4. Decisions inherited — do not re-litigate

- Layer 0 gates **unweighted** marginals against the codebook, plus a separate weighted
  check against ACS 2024.
- GSS 2021 stays in the bed; interview mode recorded per respondent.
- Pooled waves use `wtssps` rescaled to equal per-wave mass.
- `natroad` is the demo item; mode travels as a stat-card marginal.
- Split-ballot `*Y` twins: base variable only (§1.4a).
- `codebooks/item_split.frozen.yaml` is **FROZEN**. Never regenerate it.
- Response contract is a **JSON object keyed by option label**, never a positional array.
- 3.4's fold exclusion is enforced and stays enforced.
- `dwelown` excluded from Gate 3 item selection as a household fact; `xmovie` and `news`
  stay (§5.1 says attitudinal/**behavioural**).
- **Never select an anchor by closeness to the target's level.** That is the held-out
  quantity.

## 4a. How to run it

```bash
uv venv --python 3.11 $HOME/venv-b17
uv pip install --python $HOME/venv-b17/bin/python -e ".[dev,pdf]"
PY=$HOME/venv-b17/bin/python

$PY -m popsim.cli doctor --live --set 'llm.active_providers=["mistral_8b"]'
$PY -m popsim.cli gate4                       # Layer 3, no LLM calls
./scripts/run_on_mac.sh mistral_8b            # the whole elicitation, resumable
$PY -m popsim.cli layer4 --model ministral-8b-2512 --profile permutation
```

Inside a session whose shell freezes between calls, use one bounded window at a
time instead: `./scripts/run_elicitation.sh mistral_8b permutation anchors 6 156`,
then the same for `targets`, then `population`. `scripts/store_progress.py
<model> <profile>` says how far it got.

Useful scripts, all read-only and free: `forensics.py` (the level/structure
decomposition), `structure_test.py` (Gate 3 on the level-matched prediction),
`deviation_probe.py` (the scale sweep), `subspace_probe.py` (the rank sweep),
`import_gate3_records.py` (fold an old Gate 3 run's real arm into the store).

## 5. Environment

- Repo `~/Downloads/fyp/popsim`; data at `~/Downloads/fyp/data`, outside it.
- Rebuild the venv (it does not persist):
  `uv venv --python 3.11 $HOME/venv-b17 && uv pip install --python $HOME/venv-b17/bin/python -e ".[dev,pdf]"`
  then run everything with `$HOME/venv-b17/bin/python`.
- `load_gss` over the 598 MB `.dta` takes ~6 s with `usecols=`. No caching needed.
- **Shells freeze between tool calls** — in the cloud container *and* in the desktop
  Linux VM. Background jobs do not progress. Long runs must be chunked into ≤170 s
  pieces; the response cache makes each chunk resume for free. A multi-hour run is
  better handed to Gaurav as a one-command script for his own Terminal.
- Mistral capacity is allocated **per model**: `ministral-3b-2512` 750/min,
  `ministral-8b-2512` 188/min, `ministral-14b-2512` 30/min, everything larger 0/min.
- Google's key works but `gemini-2.0-flash` in the config is retired — **still unfixed**.
- Groq returns 403/Cloudflare 1010 from the sandbox; may work from Gaurav's own terminal.
- Ollama on Gaurav's Mac is **not** reachable from either shell (localhost there is not
  localhost here). The dev arm must be run by him.

## 6. Task list — what has to be done

| # | task | state |
|---|---|---|
| 1 | Diagnose the red gate from existing data | ✅ done |
| 2 | Properly-powered conditioning test | ✅ done — GREEN |
| 3 | This file + `PREREGISTRATION.md` §8 | ✅ done |
| 4 | Phase 4 — calibration layer (M5) + Gate 4 oracle pass-through | 🔨 |
| 5 | Phase 5 — baselines B0a/B0b/B1/B4/B5, metrics, raking, bootstrap | ⏳ |
| 6 | Elicitation capacity + resumable chunked runner | ⏳ |
| 7 | Headline run: 40 targets × 56 clusters, the Layer 4 verdict | ⏳ |
| 8 | Ablations, 3B/7B/14B tier curve, leakage probe | ⏳ |
| 9 | Report, figures, honesty box, demo app | ⏳ |

## 6a. Reproducing any of this

```bash
uv venv --python 3.11 $HOME/venv-b17
uv pip install --python $HOME/venv-b17/bin/python -e ".[dev,pdf]"
PY=$HOME/venv-b17/bin/python

$PY -m popsim.cli gate1                       # Layer 0, no LLM calls
$PY -m popsim.cli gate2                       # Layer 1
$PY -m popsim.cli gate4                       # Layer 3, no LLM calls
./scripts/run_on_mac.sh mistral               # the whole elicitation, resumable
$PY -m popsim.cli leakage --set 'llm.active_providers=["mistral"]'
$PY -m popsim.cli layer4 --model ministral-14b-2512 --adversarial \
     --leakage runs/<that run>/leakage_report.json
$PY -m popsim.cli report --model ministral-14b-2512
$PY scripts/verify_headline.py ministral-14b-2512
```

The store is resumable and the response cache is keyed per draw, so any of these
can be stopped and re-run. `scripts/store_progress.py <model> <profile>` says how
far an elicitation got.

## 6b. Verification — the two checks that exist because a silent bug is the risk

Both bugs this project has hit produced a *plausible number*, not an error, so
two independent re-derivations are part of the repo:

* **`scripts/verify_headline.py`** recomputes the headline W1 from first
  principles, sharing nothing with `evalx/harness.py` but the stored calibrator
  and the bed. Current state: **agrees to 0.00e+00.** Run it before quoting a
  number anywhere.
* **`scripts/verify_toplines.py`** re-parses the GSS codebook PDF with a
  from-scratch parser and compares cell by cell. **1,147 of 1,148 entries
  reproduce exactly**; the one that does not is `relactiv`, whose entry carries
  the wrong scale entirely — not in the codebook, not an anchor, not a target, so
  it never touched a result. `verified:` in the toplines YAML now means "a second
  independent parser reproduced this", and the failure names itself in
  `verified_note`.

## 7. Open and not blocking

- 1.6 GFS India adapter (`COUNTRY = 6`, weight `ANNUAL_WEIGHT_R2`, `_Y1`).
- 1.7 OpinionQA adapter — 15 ATP waves.
- ~~Spot-check `codebooks/gss2022_published_toplines.yaml`~~ — **done**, see §6b.
- Decide the 6 dropped tolerance items.
- Re-check the §1.2 item counts against base-only n.
- Fix the retired `gemini-2.0-flash` tag.
