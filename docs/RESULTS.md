# B17 — results, claims, and the things this does not claim

**Draft, 17 Sep 2026.** Numbers regenerate from `runs/` via `popsim report`; this
document is the prose around them. Where a figure appears here it also appears in
`runs/report/b17_results.html`, and the two are generated from the same JSON.

Every headline number carries three figures — **achieved W1, the B0a baseline,
the noise floor** — because "W1 = 0.066" means nothing and "0.066 against a
baseline of 0.081 and a floor of 0.027" is a sentence a reviewer can check.

---

## 1. The question, and why the obvious objection does not land

> Given a demographic subgroup described only by its population marginals and its
> answers to a set of *other* survey questions, what is the full distribution of
> its answers to a question it was never asked?

Not the mean — the whole histogram, because the distribution is what
persona-prompting gets wrong (Bisbee et al.'s variance collapse) and what a
cross-tab cannot supply for an unasked question.

**The killer objection is F3:** *you built an expensive way to reproduce a
cross-tab you already had.* Three things answer it structurally rather than
rhetorically.

1. **The oracle router.** Every incoming question is checked against the
   harmonized codebook first. If it is there, or near-duplicate of something
   there, the answer is the weighted cross-tab and no model is called. The LLM
   path fires only for questions outside the data, so by construction the model's
   only job is the thing the cross-tab cannot do.
2. **Validation lives entirely on held-out items.** Anchors go into the card;
   targets are held out. The frozen 34/40 split was written before any
   elicitation ran and has never been regenerated.
3. **The baseline is the hard one.** B0a copies the *true* national marginal onto
   every cluster. Beating it needs real between-cluster signal. The spec calls it
   "unfair-strong" because it is handed the population answer — so the system is
   handed the same national marginal, and the deviation scale `s = 0` reproduces
   B0a *identically*. The baseline is an ablation of the method, not an outside
   competitor, and what is being compared is only whether subgroup structure was
   added on top.

## 2. What the system computes

```
pred_cdf(c) = level_cdf  +  s · P_r ( raw_cdf(c) − Σ_c w_c · raw_cdf(c) )
```

| term | what it is | where it comes from |
|---|---|---|
| `level_cdf` | the item's overall level | the weighted mixture of the scored cells' truths (**observed-level**), or a population-level LLM call (**predicted-level**) |
| `raw_cdf(c) − Σ w raw_cdf(c)` | the model's subgroup *deviation* | the stat-card elicitation, with whatever constant offset it applied to this item removed |
| `P_r` | projection onto the subgroup subspace | the top-`r` principal directions of the **anchor** deviation matrix |
| `s` | how much of that deviation to believe | fitted on **anchors**, out of fold |

Everything fitted is fitted on anchors, by the F = 3 cross-fitting the checklist
specifies, and nothing — not the rank, not the scale, not whether to apply
variance restoration — is chosen using a target's truth.

### 2.1 Why the decomposition, and not the spec's M5

The spec's calibration layer is isotonic-on-CDF, then a variance multiplier, then
ensemble averaging, all applied to the raw histogram. That assumes the raw
histogram is roughly in the right place and needs its shape repaired. Measured on
960 records across two model sizes, it is not. The error splits into two parts
that behave completely differently:

| | 7B | 14B |
|---|---|---|
| raw W1 | 0.1852 | 0.2074 |
| W1 after the item level is repaired (oracle ceiling) | 0.0756 | **0.0603** |
| between-cluster Spearman ρ, macro | +0.411 | **+0.597** |
| items with ρ significantly > 0 | 6/10 | **8/10** |

The 14B puts `spkcom`'s group mean at 0.718 where the truth is 0.293, and
`homosex`'s at 0.241 against 0.591 — while ordering the clusters correctly on
`xmovie` at ρ = 0.865 and `news` at 0.817. A calibration map fitted on anchors
cannot repair that level error, because it is **per item** and a target item has
no truth to fit against; pooling isotonic across items removes the average offset
and leaves each item's own offset untouched. Separating the two is the only thing
that makes the anchor-fitted part a quantity that transfers.

### 2.2 Why the subspace projection

A deviation measured from three ensemble draws is mostly elicitation noise, and
noise is isotropic. Real subgroup variation is not: across the 34 anchor items
whose per-cluster truths are observed, the cluster-level deviations live in a
low-dimensional subspace — two directions carry 68 % of the variance on this bed,
and they are recognisably the age and education gradients. Any component of the
model's deviation outside that subspace cannot be subgroup structure.

This is the hierarchy doing statistical work in the §2.2 sense — pooling
information across clusters — rather than being an org chart of agents. Measured
contribution on the 8B arm: **0.0738 → 0.0661**.

## 3. The gates

| layer | what it asks | verdict |
|---|---|---|
| 0 | does the pipeline reproduce published reality? | ✅ 142/142 toplines within 0.5 pp |
| 1 | is the ground truth itself trustworthy? | ✅ K = 56, noise floor 0.0273, 140/148 items clear SNR 1.5 |
| 2 | is the model reading the card? | ❌ on the raw statistic · ✅ on the level-matched one — §3.1 |
| 3 | is the calibration layer arithmetically sound? | ✅ oracle round-trip 1.1 × 10⁻¹⁶ |
| 4 | does it beat the baseline on held-out items? | §4 |
| 5 | is the win memorised? | §6 |

### 3.1 Gate 2, and the one place this project changed its mind

Gate 3 (the permutation test) was run four times and was red every time, on two
model sizes, and the cause was diagnosed as the frozen item split. **That
diagnosis was wrong, and the record of it stands.**

The frozen statistic is the pooled real-vs-permuted W1 gap computed on the **raw**
histogram. The raw histogram is not what the system predicts — it is the *input*
to Phase 4 — and its level error (≈ 0.20 W1) is four times the entire
between-cluster spread of the truth (≈ 0.12). The gap is therefore a small
difference between two large, mostly-level numbers, and cannot see the structure
underneath. Gate 3 sits before Phase 4 in the checklist, so it was measuring,
correctly and as specified, a pipeline the checklist itself says is incomplete at
that point.

Asked on the level-matched prediction — same records, same clusters, same items,
same 0.0273 floor, same derangement seed 17, `s = 1` unfitted, no new calls:

| | 7B | 14B |
|---|---|---|
| real W1 | 0.0974 | 0.0799 |
| card-permuted W1 (the model really saw another cluster's card) | 0.1376 | 0.1488 |
| **gap** (gate needs > 0.0273) | **+0.0402** | **+0.0689** |
| items p < 0.05 against the derangement null | 6/10 | 8/10 |

**Both verdicts are reported in every table.** The frozen RED is not withdrawn,
and the level-anchoring mechanism measured alongside it (r ≈ −0.8 between
|anchor level − target level| and ΔW1, replicated on two model sizes) stands as a
finding about **raw verbalized elicitation**, which is what it was measured on.
`PREREGISTRATION.md` §8 records the whole reinterpretation, written before the
calibration layer existed.

## 4. Layer 4 — the verdict

39 held-out GSS items × 56 clusters, 1 paraphrase × 3 repeats, observed-level
mode, ACS-2024-raked population weights.

### 4.1 The headline: the 14B arm (12,540 elicitations, 100 % JSON ok)

| item set | achieved | B0a here | absolute mark | absolute | relative (−20 %) |
|---|---|---|---|---|---|
| all targets | **0.0602** | 0.0807 | 0.0556 | fail | **PASS** (−25.3 %) |
| top-quartile heterogeneity | **0.0787** | 0.1139 | 0.0834 | **PASS** | **PASS** (−30.9 %) |
| leakage-resistant subset | **0.0565** | 0.0757 | 0.0614 | **PASS** | **PASS** (−25.4 %) |
| all targets, leaky items removed | 0.0611 | 0.0822 | 0.0556 | fail | **PASS** (−25.7 %) |

- **Population variance ratio 1.009**, inside the pre-registered [0.8, 1.2] band —
  §1.4(ii) **passes**.
- **90 % interval coverage 0.899** against a nominal 0.90.
- **Between-cluster Spearman +0.651** macro; **36 of 39 items positive**.
- **Tolerance-battery ordering ρ = +0.879** over 49 clusters and 12 items.
- ECE 0.0227.

| arm | W1 | |
|---|---|---|
| B3 supervised skyline | 0.0282 | the ceiling, essentially the noise floor |
| **system** | **0.0602** | |
| system, topic-disjoint calibration | 0.0606 | the adversarial split costs 0.0004 |
| system, `s = 1` unfitted | 0.0609 | |
| system, per-cluster scale | 0.0612 | |
| system, no subspace projection | 0.0626 | |
| B0a national marginal | 0.0807 | **the baseline the claim is written against** |
| B0b LLM-predicted marginal | 0.1787 | |
| B4 uncalibrated | 0.1926 | **the layer cuts the error by 69 %** |
| B1 nearest anchor | 0.2343 | |

§1.4's claim, in its own words, is that calibrated cluster agents beat the
national-marginal baseline by ≥ 20 % on the items where subgroups genuinely
differ. On the top-quartile heterogeneity set that is **−30.9 %**, and it clears
the pre-registered absolute mark as well. On all 39 targets it is **−25.3 %**,
which clears the relative rule and misses the absolute mark — a mark derived from
a baseline measured on a bed the project later replaced (§7, item 4).

### 4.2 The 8B arm (12,486 elicitations, 99.9 % JSON ok)

| item set | achieved | B0a here | absolute mark | absolute | relative (−20 %) |
|---|---|---|---|---|---|
| all targets | 0.0638 | 0.0807 | 0.0556 | fail | **PASS** (−20.9 %) |
| top-quartile heterogeneity | 0.0832 | 0.1139 | 0.0834 | **PASS** | **PASS** (−26.9 %) |
| leakage-resistant subset | 0.0584 | 0.0757 | 0.0614 | **PASS** | **PASS** (−22.8 %) |
| all targets, leaky items removed | 0.0646 | 0.0819 | 0.0556 | fail | **PASS** (−21.1 %) |

- **Population variance ratio 1.009**, inside the pre-registered [0.8, 1.2] band —
  §1.4(ii) **passes**.
- **Tolerance-battery ordering ρ = +0.897** over 49 clusters and 12 items: the
  system reproduces the ordering of the five target groups *within* each subgroup,
  which a national marginal cannot do at all.
- **90 % interval coverage 0.896** against a nominal 0.90 — see §4.5.
- **Between-cluster spread: predicted 0.0550 against a true 0.0979.** The system
  under-disperses across clusters by about half. See §4.6; this is reported, not
  tuned away.

### 4.2a Every arm, 8B

| arm | W1 | what it kills if it wins |
|---|---|---|
| B3 supervised skyline | 0.0282 | nothing — it uses the labels we assume absent; it locates the ceiling, essentially at the noise floor |
| **system** | **0.0638** | |
| system, topic-disjoint calibration | 0.0641 | *(adversarial split, §5.1)* |
| system, per-cluster scale | 0.0646 | *(ablation: hierarchical pooling of `s`)* |
| system, `s = 1` unfitted | 0.0646 | *(ablation: is the anchor fit doing anything?)* |
| system, no subspace projection | 0.0688 | the projection |
| system, no anchors on the card | 0.0707¹ | the stat card's conditioning |
| B0a national marginal | 0.0807 | **the entire thesis** |
| B0b LLM-predicted marginal | 0.1647 | cluster conditioning |
| B4 uncalibrated | 0.1655 | the calibration mechanism |
| B1 nearest anchor | 0.2343 | LLM reasoning over item similarity |

¹ measured at a matched 1-draw ensemble, against 0.0659 for the anchored card at
the same ensemble size — see §5.

Three things this table settles.

1. **The calibration layer is not dead weight.** B4 is 0.1655 — raw elicitation is
   *twice as bad* as copying the national marginal. Calibrated it is 0.0661. The
   §5.8 calibration-kill criterion is not met; the mechanism is load-bearing.
2. **The anchor fit transfers.** `s` fitted on anchors is 0.84 and beats
   nothing-fitted by a small margin (0.0638 against 0.0646). The margin is small
   because the fitted value lands near 1 anyway; what matters is that a quantity
   fitted only on anchors does not *hurt* on items whose truth it never saw,
   which is the cross-fitting claim.
3. **Nothing here is item similarity.** B1 copies the most textually similar
   anchor item and scores 0.2343, worse than raw elicitation.

### 4.3 The capability curve (§7.3a)

| arm | achieved | vs B0a | variance ratio | battery ρ |
|---|---|---|---|---|
| Ministral 3B | 0.0686 | −15.0 % | 1.007 | +0.870 |
| Ministral 8B | 0.0634 | −21.4 % | 1.008 | +0.898 |
| **Ministral 14B** | **0.0602** | **−25.3 %** | 1.009 | +0.879 |

All three arms at the same 39 targets × 56 clusters, same cards, same split, same
selection rule — only the model changes. The curve is monotone in capability, and
each arm's strength was predictable from its **anchor** fit before a single target
was scored: anchor W1 at the fitted scale is 0.0616 (3B), 0.0589 (8B), 0.0558
(14B).

Monotone in capability, and each arm's strength was predictable from its **anchor**
fit alone before any target was scored. That is the useful property of an anchor-side selection rule: it tells you
what the arm will do before you spend the target half of the quota.

### 4.5 Honest intervals, and what an interval made of ensemble spread is worth

§M6 asks for "honest intervals". The obvious construction — bootstrap the
elicitation ensemble, take per-option quantiles — is not one. Measured on the 8B
arm, its nominal 90 % intervals covered **15.7 %** of true values. Three draws of
the same prompt agree with each other; that agreement says nothing about how far
they all are from the truth.

The missing term is estimable without any target's truth: the **out-of-fold anchor
residual**, the SD of (truth − prediction) at each interior CDF position, measured
on the anchors after the whole pipeline has run on them. Adding it (per scale
length: k = 2 → 0.122, k = 3 → 0.077, k = 4 → 0.074, k = 5 → 0.116, k = 6 → 0.094)
takes coverage to **0.896**, essentially nominal. It is not a widening fudge; it is
the term the first construction left out, and it is fitted on anchors like
everything else.

### 4.6 Where the system is still wrong: between-cluster spread

| | predicted | true |
|---|---|---|
| between-cluster SD of the item mean, averaged over items | 0.0550 | 0.0979 |

The system places the subgroups in the right **order** — ρ = +0.879 on the
battery, +0.651 macro across all targets, positive on 36 of 39 items — and too close **together**, by about a
factor of two. That is the direct price of the shrinkage that minimises W1: `s`
below 1 and a rank-restricted projection both pull the clusters toward the
population, and W1 rewards that because being conservative is cheap when the
deviation is noisy.

It is a milder relative of F2 and it is reported rather than corrected. Matching
the spread instead — choosing `s` so predicted between-cluster SD equals the
anchors' — is a one-line change and it costs W1. A user who needs *ranking* of
segments is well served; a user who needs the magnitude of the gap between two
segments should read the interval, not the point estimate.

### 4.7 The K sweep and the cost–accuracy curve (§5.7, §7.3)

On the 3B arm, the same 39 targets and the same anchors, re-clustered:

| K | calls (anchors + targets, 1×3) | achieved | B0a at that K | vs B0a | `s = 1` unfitted |
|---|---|---|---|---|---|
| 18 | 3,942 | 0.0744 | 0.0744 | −0.1 % | **0.0639 (−14 %)** |
| 36 | 7,884 | **0.0654** | 0.0772 | −15.2 % | 0.0688 |
| 56 | 12,264 | 0.0686 | 0.0807 | −15.0 % | 0.0705 |

Two readings, and the second is a limitation rather than a result.

**K = 36 is the efficiency point.** It gets the same relative improvement as
K = 56 for 64 % of the calls, and the lowest *absolute* W1 of the three — coarser
clusters have less true variation to miss. The scalability claim in the writeup is
exactly and only this table.

**At K = 18 the anchor-side selection overfits, and no anchor-side rule could have
caught it.** The out-of-fold CV preferred the per-scale-length scheme (0.0532
against 0.0557 on anchors) and that scheme fitted `s = 2.6` for binary items on a
handful of coarse-cluster anchors; the targets paid for it. The unfitted `s = 1`
arm beats B0a by 14 % on the same records. The structural reason is that anchors
and targets differ **by construction** — the 2.4 split put the high-SNR items in
the target half — so a scheme that exploits a property the anchors have and the
targets do not will win every anchor-side comparison there is. Partial pooling of
the per-k scale toward the global one (`m/(m + 3)`, the same form the cluster
pooling uses) was added for this and helps — it took K = 18 from +8.7 % to
−0.1 % and improved K = 56 as well — but it does not remove the gap. This is the
sharpest limitation the project has and it is not specific to K.

## 5. Ablations

| sweep | result | what it means |
|---|---|---|
| ensemble size 1 / 2 / 3 | 0.0659 / 0.0639 / 0.0638 | The second draw is worth −3 %; the third is worth nothing. The *paraphrase* axis is untested here and is the one §F8 actually cares about. |
| subspace rank 0 vs chosen | 0.0688 → 0.0638 | the projection is worth −7 % |
| **no-anchor stat card** (§5.5, the Argyle regime at cluster level) | 0.0707 vs 0.0659 at a matched 1-draw ensemble | **The demographics alone get most of the way** — −12.0 % against B0a with no anchors at all — and the anchor items add a further −6.8 % relative. Honest reading: the stat card's *conditioning* is real but is the smaller half of what the card does. The fitted scale also drops from 0.84 to 0.29, which is the anchor-side rule correctly shrinking a noisier deviation harder. |
| standard vs topic-disjoint split | 0.0638 vs 0.0641 | **removing every same-topic anchor costs 0.0003 W1**, which is what a layer whose only fitted quantity is one scalar should do — and it is the transfer result the scenario story needs, since no anchor shares a novel question's subject |
| variance restoration on vs off | anchor W1 0.0636 vs 0.0589 | **F1 does not occur in this design.** The prediction inherits the national histogram's *shape* instead of generating it token by token, so the population variance ratio is 1.000 with no restoration at all. The M5 step-2 repair has nothing to repair and only distorts. §5.8's kill criterion, applied to a component: report and drop. |

## 6. Layer 5 — leakage

The probe names GSS, names the waves and names the subgroup, then asks for the
published cross-tab: the opposite of the elicitation prompt, which names no
dataset and asks the model to reason from a group's other answers.

An item is flagged only when the probe recalls a **subgroup** within W1 < 0.05
**and** beats simply copying the topline onto that subgroup. Both halves matter:
every GSS topline is published, so a model reciting one says nothing about whether
it knows the breakdown, and on a binary item W1 is |Δp|, so with four subgroups
probed one landing inside 0.05 by chance is not unlikely. A naive
best-over-subgroups rule flagged 22 of 39 items on the 8B arm, almost entirely on
that.

The corrected rule flags **2 of 39 on each arm** — `conlegis` and `polviews` on
the 8B, `postlife` and `spanking` on the 14B. Removing them moves the headline
from −25.3 % to −25.7 % on the 14B and from −18.0 % to −18.1 % on the 8B. The two
views agree, so §5.6 stops being a weakness.

One thing worth reporting from the probe itself: the 14B returns usable JSON on
**100 %** of elicitation calls and **50 %** of recall calls, the rest being
malformed or hedged. It is markedly less willing to *claim* a published
cross-tab than to *estimate* a subgroup distribution — which is mild evidence
against memorisation, and is the opposite of what a model reciting a memorised
table would look like.

A clean probe is **necessary, not sufficient**: a model can absorb an item's shape
from the literature without recalling its cross-tab. That is why the
leakage-resistant subset and the topic-disjoint split are reported beside it, and
why leakage is discussed as a limitation regardless.

## 7. Where the spec was wrong, and how we know

Six places, all found by measuring rather than by reasoning. Each is a
methodological contribution in its own right, because each was a defensible design
choice that turns out to be wrong in a way a reader would not have guessed.

1. **Gate 3's statistic is computed on the wrong object** — the raw histogram
   rather than the prediction. §3.1.
2. **M5's variance-restoration step is dead weight in this design** and makes
   anchor W1 worse. §5.
3. **The router's 0.85 threshold has a recall of zero.** No human paraphrase of a
   GSS item shares enough vocabulary with its verbatim instrument wording to clear
   a 0.85 TF-IDF cosine. Taking that number on faith would have routed every real
   question to the LLM and quietly retired the F3 defence. Stripping the
   parenthetical battery stems, adding character n-grams and folding in the option
   labels takes recall from 0.46 to 0.81 at precision 0.96; the threshold is 0.25.
4. **The pre-registered pass marks were derived from a baseline measured on a
   different bed** (2016–2022, n = 11,045, 88 items) than the one the project
   froze (pooled 2010–2022, n = 18,772, the 40-item target half). B0a on the frozen
   bed is 0.0807, not 0.0695. Nothing was moved; both the absolute and the
   relative verdict are reported for every subset. §8.8 of the pre-registration.
5. **The subspace projection must be restricted, not zero-padded.** Any given item
   is scored on the clusters whose truth clears `truth_min_neff`, which differs
   per item. Padding the absent ones with zeros and projecting in the full space
   asserts that those clusters sit exactly at the population mean, and the
   projection spends its budget honouring that claim: on the 14B at 16 of 56
   clusters it turned a gain into a loss (0.0769 → 0.0944). The right object is
   the basis restricted to the present rows and re-orthonormalised.
6. **Amendment 1 to the anchor set was misapplied** and is retired. `make_crossfit_plan`
   derives folds from the anchor list, so adding five anchors re-derived every fold
   and put the spending battery's outlier back onto the spending cards, confounding
   the rule being tested with the thing it was meant to remove.

## 7a. Two independent re-derivations

Both bugs above produced a *plausible number*, not an error, so the number does
not get quoted until a second implementation agrees with it.

**The headline.** `scripts/verify_headline.py` recomputes the macro W1 from first
principles: truths straight from `ClusterStats`, the level built by hand as the
weighted mixture, the projection and scale applied by hand from the calibrator's
own JSON, W1 written out as the sum of absolute CDF differences. It shares
nothing with the harness but the stored calibrator and the bed. Agreement:
**0.00e+00**. It earned its keep on its first run by failing — against a report
that predated the projection fix, because it picked the comparison by item count
rather than by modification time.

**The toplines.** `scripts/verify_toplines.py` re-parses the GSS 2022 codebook PDF
with a parser written from scratch and compares counts, percentages and reserved
codes cell by cell against `gss2022_published_toplines.yaml`. **1,147 of 1,148
entries reproduce exactly.** The one that does not, `relactiv`, carries an
agree/disagree scale where the real item is a frequency scale — the extractor
took a neighbouring table. It is not in the item codebook, not an anchor and not
a target, so it never touched a result, but it is precisely the failure the
Layer 0 gate cannot see: an item that parsed wrongly *and* sits outside the 142
items Layer 0 covers. Twenty-one further entries differ only in where a long
option label wraps; the counts are identical and the YAML's version is the more
complete.

`verified:` in that file used to mean "no human read the PDF". It now means a
second independent parser reproduced the entry, and the failure names itself.

## 8. Limitations, stated before anyone asks

- **One country, one instrument.** GSS only. The India replication (GFS,
  `COUNTRY = 6`) is built for but not run; NFHS-5 and Pew India remain gated.
- **K = 56, not 150.** The GSS supports ~53 leaves at the median item. The spec's
  150 was reasoned, not measured.
- **Geography is a stat-card marginal, not a partition axis.** Adding urban/rural
  to the partition drops the usable item bank from 74 to 35.
- **The top of the SNR ranking is behaviour-heavy.** `xmovie` and `news` are
  reported behaviours; §5.1 says "attitudinal/**behavioral**", so they stay, and
  this is a limitations line rather than a silent inclusion.
- **The system under-disperses between clusters by about half** (§4.6). Ordering
  is right; the magnitude of subgroup gaps is understated.
- **The paraphrase axis of the ensemble is untested.** Every number here is one
  paraphrase × three repeats. §F8 says prompt instability is the thing to ensemble
  over, and this project measured the wrong axis of it.
- **The observed-level mode needs a national topline.** That is a realistic
  deployment — a national poll is cheap, a subgroup-powered one is not — but it is
  *not* the scenario case. The predicted-level mode is what a genuinely unasked
  question has to use, and it is reported separately and never merged.
- **Which same-topic exemplar reaches a card is decided by an arbitrary fold
  seed.** `make_crossfit_plan` shuffles anchors into folds and `_pick_anchors`
  breaks ties alphabetically; nothing in that chain knows whether the surviving
  exemplars are representative of their battery, and the measured consequence on
  raw elicitation is a W1 swing of 0.10–0.41 on a single item. That is a real
  criticism of stat-card conditioning and it belongs in the writeup whatever the
  gate says.

## 9. Claims not made

Not "simulates society". Not "predicts policy outcomes". Not "replaces surveys" —
the system is built on them and dies without them. Not "scales to millions of
agents" — Chopra et al. own that; this is a fidelity-per-dollar claim at K ≤ a few
dozen. Not any accuracy claim about a product or policy scenario: those have no
ground truth and never will, the held-out-item score is the honest proxy, and the
honesty box is where that is said to the reader rather than buried here.
