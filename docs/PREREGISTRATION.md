# Pre-registration — Gate 3 (Layer 2, the permutation test)

Written **16 Sep 2026, before the §7.2 rung-1 run**, and before any frontier-model
call has been made. Its purpose is to make the degrees of freedom already spent
visible, and to fix what remains before the next number exists.

The checklist's rule is the reason this file exists:

> *"Write these into `configs/gss_main.yaml` **before** the first elicitation
> run. A pass mark chosen after seeing results is not a pass mark."*

That applies to the Layer 4 pass marks, which were pre-registered and have not
moved. It applies with equal force to the **card configuration**, which was not
pre-registered, and which has now been changed twice with the gate result
visible in between. This file records that honestly rather than quietly.

---

## 1. What has already been run and inspected

Four Gate 3 runs on the dev model `qwen2.5-ctx8k:7b-instruct-8192`, all local,
all free:

| # | run id | scope | card / contract change | seeded gap | verdict |
|---|---|---|---|---|---|
| 1 | `…20260916T070311Z` | 4 items × 8 clusters, 1×1 | baseline; positional response array | +0.0173 | RED |
| 2 | `…20260916T075245Z` | 10 × 16, 1×3 | label-keyed response contract | **+0.0271** | RED |
| 3 | `…20260916T081708Z` | 10 × 16, 1×3 | 3.4 fold exclusion enforced | +0.0125 | RED |

Noise floor 0.0273 throughout. Run 2's gap is the largest and run 3's the
smallest; **the configuration was not chosen on that basis** and run 3's change
is retained. See §3.

## 2. What each change was, and why it stands or falls independently of the gate

**Run 2 — label-keyed response contract.** 19 codebook items (the whole `nat*`
spending battery plus `polviews`) carry verbatim wording that enumerates their
answer options in an order differing from `codes`. With a positional response
array a model answering in the wording's order was scored in the wrong bins.
Justification is independent of any gate number: the `textorder` fake in
`scripts/gate3_dryrun.py` scores 0.0132 under the keyed contract and would have
scored 0.349 under the positional one. **Retained.**

*Correction on the record:* when this change was made it was believed to be the
cause of `natroad`'s 0.42 W1. It was not — `natroad` came back 0.4137 under the
keyed contract, essentially unchanged. The reversal evidence fitted two
hypotheses and the wrong one was backed. The change stands on the dry-run
evidence above, not on that reasoning.

**Run 3 — 3.4 fold exclusion enforced.** Checklist 3.4 specifies "no same-fold
anchor"; `assert_card_contract`'s own docstring claimed it; nothing checked it,
and `anchors_for()` returned all 34 anchors for a target. **Retained, and it
would be retained even though it halved the gap**, because reverting a clause the
checklist specifies in order to recover a better number is the failure this file
exists to prevent.

## 3. The configuration frozen as of now

In `configs/gss_main.yaml`, `elicitation:`

| key | value |
|---|---|
| `anchors_per_card` | 12 |
| `enforce_fold_exclusion` | true |
| `anchor_pick` | `topic_diverse_alpha` |
| `temperature` | 0.7 |
| profile for Gate 3 | `permutation` = 1 paraphrase × 3 repeats |
| scope for Gate 3 | `--n-items 10 --n-clusters 16` |
| item selection | highest-SNR targets, minus `POST_FREEZE_DOCTRINE_FLAGS` |
| gate statistic | seeded `permuted_w1 - real_w1 > noise_floor` (0.0273) |
| derangement seed | 17 |
| frontier model | **`ministral-14b-2512`**, pinned 16 Sep 2026 (entry 4a) |

`item_split.frozen.yaml` is unchanged and remains frozen.

**Any further change to these values is a new numbered entry in this file,
written before the run that uses it.**

## 4. The next run, declared in advance

**§7.2 rung 1 — the frontier arm.** Same 10 items, same 16 clusters, same cards,
same statistic. Only the model changes.

```
popsim doctor --set 'llm.active_providers=["mistral"]'     # expect: all good
popsim gate3 --n-items 10 --n-clusters 16 --profile permutation \
  --set 'llm.active_providers=["mistral"]'
```

The first attempt at this run (`runs/gss_main__20260916T094338Z`) returned NO
DATA: a per-second throttle on the first call was read as the day's allowance
being gone, and all 960 calls failed with `calls_today: 0`. That was a client
bug, not a result, and it is recorded here so the run directory is not mistaken
for evidence about the model. Fixed (throttle vs exhaustion, plus request
pacing); **nothing about the gate, the cards, the items or the statistic
changed**, and this entry is not a new configuration.

The model string is pinned to **`mistral-small-2603`** before the run — the only
dated small tag of the 46 the account is offered — not left on
`mistral-small-latest`: `cache.py` keys responses on that string, and the
provider repoints the alias without notice, so a resumed run would mix two sets
of weights into one ensemble and report the difference as elicitation variance.
The exact tag used is recorded in the run's `config.snapshot.yaml` and in
`gate3_report.json`.

The hypothesis being tested is specific, and it is not "is the premise true":

> The dev model's failures are explained by it using a single same-topic
> anchor's **level** as its prior for the target's level — measured at
> r = −0.853 between |anchor level − target level| and the change in real W1
> across the nine anchor removals run 3 produced. **Does a frontier model do
> the same?**

Declared readings, before the number exists:

- **Gap > 0.0273 (GREEN).** The card carries usable conditioning for a capable
  model, the 7B's level-anchoring is a small-model artifact, and Phase 4 starts.
  Record that Gate 3 passed on the frontier arm and not on the dev arm, and
  carry both numbers in every subsequent table.
- **Gap ≤ 0.0273 (RED) with per-item structure like run 3** — a majority of items
  significant against the derangement null, failures concentrated on items whose
  one same-topic anchor is far from them in level. Then the binding constraint is
  the item split, not the model, and §7.2's rungs 2 and 3 cannot reach it: the
  frozen split holds 2 civil-liberties anchors for 12 civil-liberties targets,
  and 0 anchors for religion, crime, family and politics. The next step is then
  the §5 decision about amending the anchor set, taken with this evidence in
  hand — not before it.
- **Gap ≤ 0.0273 with per-cluster spread near zero** — that is actual F2, and the
  premise is in trouble. It is also the one reading none of the four runs so far
  supports: the `hedger` fake produces a gap of exactly 0.0000, and runs 2 and 3
  show per-option spreads of 0.13–0.19 across clusters.

**Neither the gate statistic nor the noise floor moves on the basis of what
comes back.**

## 5. Known open items that are NOT being changed before that run

1. **`_pick_anchors` gives a battery target one arbitrary same-battery exemplar**,
   chosen alphabetically. The measured mechanism says several exemplars would fix
   it; the frozen split does not contain several for most topics. Not changed.
2. **The split's anchor/target topic imbalance.** 12 civil-liberties targets vs 2
   civil-liberties anchors; 3 topics with targets and no anchors; `other` holds
   11 anchors for 3 targets. Promotable from the 148-item codebook: 5 spending
   items, and for civil liberties only `colmslm` — a third `*mslm` item with the
   same referent and the same outlying level. Not changed.
3. **No anchor may ever be selected by closeness to the target's level.** That is
   the held-out quantity. Recorded here because the measured mechanism makes it a
   tempting and completely invalid fix.
4. **`dwelown`, `xmovie`, `news`.** `dwelown` is excluded from item selection as a
   household fact (`POST_FREEZE_DOCTRINE_FLAGS`); `xmovie` and `news` are reported
   behaviours and stay, since §5.1 says "attitudinal/**behavioral**". The top of
   the SNR ranking is behaviour-heavy and that is a limitations line.

## 4a. Entry: the rung-1 model changed, on availability not on results

`mistral-small-2603` **cannot be run on this workspace.** Capacity is allocated
per model, not per workspace as Mistral's docs imply, and measured on 16 Sep
2026 from `x-ratelimit-limit-req-minute`:

| model | req/min |
|---|---|
| `mistral-medium-2604` | **0** |
| `magistral-medium-latest` | **0** |
| `mistral-small-2603` | **0** |
| `ministral-14b-2512` | 30 |
| `ministral-8b-2512` | 188 |
| `ministral-3b-2512` | 750 |

So the first attempt at this run was refused with zero capacity allocated, which
no amount of pacing or retrying could have changed. **This change is forced by
availability and was made before any result existed** — no Gate 3 number has
ever been produced on any Mistral model.

**`ministral-14b-2512`** is the rung-1 arm: the largest tag with capacity, and
14B is the size class STATUS.md names as the meaningful step up from the 7B dev
arm (*"a 14–32B model ... cuts the F2 risk that a 7B carries"*). Paced at 2.1s
for the 30/min cap.

`ministral-3b-2512` was suggested and does work, but it is **smaller** than the
dev model, so a red result on it would say nothing about whether capability
fixes the level-anchoring — which is the only question rung 1 asks. It is
recorded instead as a genuinely useful *downward* point for §7.3a: 3B / 7B / 14B
is a clean capability curve on one question, which is a better tier ablation than
the checklist budgeted for. Not run now.

A JSON-contract note, since M4 reports malformed-JSON rate as a metric: asked for
`{"ok": 1}`, `ministral-3b` returned `{"response": {"status": "success", ...}}`
and `ministral-14b` returned `{"ok": 1}`. The 14B honours the shape; the 3B
would inflate that metric for reasons of size rather than of elicitation.

## 5a. A confound this arm introduces, declared now

`ministral-14b-2512` is a 2026-trained model. The dev arm was chosen partly
*because* it is 2024-era, which makes §5.6's leakage story cleaner: a 2026 model
has seen more GSS cross-tabs, more OpinionQA, and more silicon-sampling papers
**with their result tables** than a 2024 one.

So a GREEN on this arm is not interchangeable with a GREEN on the dev arm. If it
comes back green, the §5.6 leakage probe and the leakage-resistant subset become
load-bearing for interpreting it, not later nice-to-haves — and the writeup
carries both arms' numbers side by side rather than the better one. Recorded here
so that is a commitment made before the result, not a caveat added after it.

## 4b. Outcome of the rung-1 run, against the declared readings

Run 16 Sep, `runs/gss_main__20260916T134458Z`, `ministral-14b-2512`, 960 records,
100% JSON ok, 0 throttles. **Gap +0.0011 against a 0.0273 floor: RED.**

This is **reading 2 of the three declared in §4**, and nothing was reinterpreted
to make it fit: red, 4 of 10 items significant against the derangement null, and
the failures concentrated on items whose one same-topic anchor is far from them
in level. Reading 3 (actual F2, per-cluster spread near zero) is again not what
happened — `xmovie` alone has a gap of +0.1132 at p = 0.000.

The pre-registered consequence therefore applies as written: *"the binding
constraint is the item split, not the model, and §7.2's rungs 2 and 3 cannot
reach it."*

The mechanism from §4 replicated rather than weakening with capability:

| | 7B (2024-era) | 14B (2026-era) |
|---|---|---|
| corr(\|anchor level − target level\|, ΔW1) | −0.853 | −0.799 |
| pooled gap | +0.0125 | +0.0011 |
| items p < 0.05 | 3/10 | 4/10 |

and the two arms' responses to the same card change correlate at **+0.915**. The
level-anchoring result promised in §6 as reportable "whether the gate passes or
fails" is now a two-model finding, which is stronger than it was as one.

Neither the gate statistic nor the noise floor was changed. §5's open items are
unchanged and are now the live decision.

## 7. Entry: amendment 1 to the anchor set — the rule, stated before the run

Written **before the amendment exists and before any run uses it.**

### The rule

> **Every topic with at least one target reserves at least three anchors, where
> the item pool contains enough items to do so. Anchors are added from the
> unassigned pool only. No target ever becomes an anchor, and
> `item_split.frozen.yaml` is not edited.**

The rule references only topic membership and assignment status. It references
**no** measurement of any target — not its truth, not its level, not its SNR,
not any Gate 3 result. That is the property that makes it legitimate: the
mechanism found in §4 (a same-topic anchor's level becomes the target's level
prior, r ≈ −0.8 on both model arms) makes "pick the anchor closest to the
target's level" an obvious fix and a completely invalid one, because the
target's level is the held-out quantity. Showing *several* same-battery
exemplars requires knowing nothing about the target.

### What it does, mechanically

The frozen split stays byte-identical. The amendment is a separate file,
`codebooks/item_split.amendment_1.yaml`, layered on top at load time, so the
frozen artifact is preserved literally and the amendment is auditable on its
own.

Only `spending_priorities` can be helped: the 5 unassigned spending items
(`natdrug`, `natcity`, `natfare`, `natheal`, `natsci`) join the anchor set,
taking spending from 6 anchors to 11 for 7 targets. Targets stay at 40, so the
≥40 requirement in START_HERE holds untouched.

### What it cannot do, stated now so a red is not a surprise

**Nothing changes for 12 of the 40 targets.** Civil liberties has 2 anchors and
no unassigned items but `colmslm` — a third `*mslm`, the same referent and the
same outlying level as the two already there, so it adds no spread and is **not**
promoted. Religion, sexual morality, crime, family and politics have zero spare
items. The rule therefore cannot reach them, and saying so before the run is the
point: **this amendment is expected to move the spending items and little else.**

### Declared readings, before the number exists

- **`natroad` and `natsoc` improve materially and the pooled gap clears 0.0273.**
  Then the diagnosis is confirmed *and* sufficient, and Phase 4 starts.
- **The spending items improve but the pooled gap stays under 0.0273.** The
  expected outcome. It confirms the diagnosis and localises what remains to the
  topics the frozen split cannot serve. That is a complete, defensible Gate 3
  finding: the method conditions where the card can supply representative
  same-topic evidence and fails where it cannot, independent of model size. The
  next step is then **not another card change** — it is a supervisor decision
  about reopening the 2.4 split against the ≥40-target claim, or accepting the
  limitation and carrying it into the writeup.
- **The spending items do not improve.** Then the level-anchoring account is
  wrong or incomplete and everything above needs revisiting, including the
  two-model correlations.

Neither the gate statistic (seeded gap > 0.0273) nor the noise floor moves on
what comes back. The model stays `ministral-14b-2512`; the dev arm will be
re-run on the same amendment for the two-arm comparison.

## 7a. Outcome of amendment 1 — it made the gate worse, and why

Run 16 Sep, `runs/gss_main__20260916T152836Z`, `ministral-14b-2512`, 960 records,
100% JSON ok.

```
real      W1 0.2231   (was 0.2074 before the amendment)
permuted  W1 0.2204
seeded gap  -0.0027   (was +0.0011)
null mean W1 0.2281   -> vs-null +0.0051 (was +0.0075)
=> RED, and worse than before.
```

| item | before | after | |
|---|---|---|---|
| `natroad` | 0.1095 | **0.2056** | worse |
| `natsoc` | 0.0849 | **0.1599** | worse |
| `xmovie` | 0.0876 | 0.0825 | ~same |
| `pray` | 0.1118 | 0.1162 | ~same |
| `spkcom` | 0.4242 | 0.4018 | ~same |
| `colath` | 0.2851 | 0.2702 | ~same |
| `libcom` | 0.3105 | 0.2934 | ~same |

**The two spending targets — the only ones the amendment could affect — got
materially worse. This run is not a clean test of the rule, and the reason is an
error in how it was applied.**

`make_crossfit_plan` derives fold assignments *from the anchor list*. Adding five
anchors re-derived every fold, which changed which fold each target excludes,
which put **`nataid` back onto the spending cards**:

```
BEFORE amendment  natroad excluded fold 0
                  surviving spending anchors: natchld natcrime natenrgy natpark
                  nataid present? no
AFTER amendment   natroad excluded fold 0
                  surviving: nataid natchld natcity natcrime natdrug natrace natsci
                  nataid present? YES
```

So the run confounds two changes — five extra exemplars (intended) and the
battery's outlier returning (not intended) — and the second dominated. That was
not anticipated when §7's rule was written, and it is the author's error, not a
property of the rule.

**It does not refute the level-anchoring account; it is further evidence for
it.** Putting `nataid` back moved `natroad` 0.1095 → 0.2056 and `natsoc`
0.0849 → 0.1599, in the direction and roughly the magnitude r ≈ −0.8 predicts.
Three independent manipulations have now moved these items the same way.

### The finding this actually produces

**Which same-topic exemplar reaches a card is decided by an arbitrary fold seed,
and the method's conditioning quality rides on that lottery.** `make_crossfit_plan`
shuffles anchors into folds with `seed=17` and picks each target's excluded fold
as the one densest in same-topic anchors; `_pick_anchors` then breaks ties
alphabetically. Nothing in that chain knows or controls whether the surviving
exemplars are representative of their battery, and the measured consequence is a
W1 swing of 0.10–0.41 on a single item. That is a real methodological criticism
of stat-card conditioning and belongs in the writeup whatever happens next.

### Why iteration stops here

Four card configurations have now been run and inspected:

| # | configuration | seeded gap |
|---|---|---|
| 1 | positional contract, `nataid` present (4 items) | +0.0173 |
| 2 | keyed contract, `nataid` present | +0.0271 |
| 3 | 3.4 fold exclusion enforced (7B / 14B) | +0.0125 / +0.0011 |
| 4 | amendment 1, `nataid` back via fold reshuffle (14B) | **−0.0027** |

Continuing to adjust the card and re-reading the gap is how a pass mark gets
chosen after seeing results, which §1 of this document exists to prevent. **No
further card change will be made in pursuit of a green gate.** What remains is
not a tuning question:

- reopening the 2.4 split against the ≥40-target claim in START_HERE, or
- accepting the limitation and carrying it into the writeup, or
- making the fold/anchor selection deterministic and *reporting the lottery* as
  a measured property rather than trying to win it.

All three are supervisor decisions about what the thesis claims. The evidence for
that conversation is complete: two model sizes, four card configurations, a
replicated mechanism at r ≈ −0.8, and an anchor pool that cannot serve 12 of the
40 targets.

## 6. What gets reported regardless of outcome

Per the checklist's honest-reporting rule, every headline number carries three
figures — achieved W1, the B0a baseline, and the noise floor — and for Gate 3
additionally: the derangement null mean and p, the per-item table, the
`positional_response` and `label_mismatch` rates, and **which model produced it**.

The level-anchoring result (§4) is reported as a finding in its own right whether
the gate passes or fails. It is a measurement about verbalized elicitation from
stat cards, and it does not depend on the gate's verdict.

## 8. Entry: the Gate 3 reinterpretation, and the Phase 4 design declared before it is built

Written **16 Sep 2026, session 2**, after re-analysing records already on disk and
**before the calibration layer exists**, before any new elicitation call has been made,
and before any Layer 4 number exists.

### 8.1 What was re-analysed, and what was not

No new call. No card change. No prompt change. No new model. The frozen split is
byte-identical, the derangement seed is still 17, the noise floor is still 0.0273, and
the Layer 4 pass marks in `configs/gss_main.yaml` are untouched. Amendment 1 stays
recorded as the failure §7a describes and is **not** used: the runs re-analysed here are
the pre-amendment ones, `…20260916T081708Z` (7B) and `…20260916T134458Z` (14B).

The re-analysis is `scripts/forensics.py` and `scripts/structure_test.py`, both of which
read `permutation_raw.parquet` and the bed, and neither of which touches a provider.

### 8.2 The measurement

Decomposing the real arm's 480 usable records per model into an item-level component and
a between-cluster component:

| | 7B | 14B |
|---|---|---|
| raw W1 (the frozen Gate 3 arm) | 0.1852 | 0.2074 |
| W1 after the item level is repaired (oracle isotonic, a ceiling not a score) | 0.0756 | 0.0603 |
| between-cluster Spearman ρ, macro | +0.411 | +0.597 |
| items with ρ > 0 | 9/10 | 9/10 |
| items with ρ significantly > 0 at p < 0.05 | 6/10 | 8/10 |

Worked cases of the level error, 14B: `spkcom` predicted group mean 0.718 against a true
0.293; `homosex` 0.241 against 0.591; `colath` 0.624 against 0.339. These are item-level
offsets shared across clusters, not failures of subgroup ordering — `xmovie` ρ = 0.865,
`colath` ρ = 0.793, `news` ρ = 0.817 on the same records.

**This is the opposite of F2.** F2 is one hedged answer per item regardless of who the
card describes, and it shows up as ρ ≈ 0 and per-cluster spread ≈ 0. What is measured is
strong correct ordering sitting under a large shared offset.

### 8.3 Why the frozen statistic could not see it

The frozen Gate 3 statistic is the pooled real-vs-permuted W1 gap. The level error enters
both arms identically, so it cancels from the difference — but it sets the scale of both
terms, and W1 is not additive in the two components. With a level error of ~0.20 W1 and a
true between-cluster spread of ~0.12, the gap is a small difference between two large,
mostly-level numbers. The statistic is the right idea measured on the wrong object: it is
computed on the **raw** histogram, and the raw histogram is not what the system predicts.
It is the input to Phase 4.

Gate 3 sits before Phase 4 in the checklist. It was therefore run, correctly and as
specified, against a pipeline the checklist itself says is incomplete at that point.

**The frozen verdict is not withdrawn.** Gate 3 on the raw statistic is RED at +0.0011
(14B) and +0.0125 (7B) and that number appears in every subsequent table beside the one
below. The level-anchoring mechanism recorded in §4 — r ≈ −0.8 between |anchor level −
target level| and ΔW1, replicated on two model sizes — is unaffected and remains a
reportable finding: it is a statement about **raw verbalized elicitation**, which is what
it was measured on.

### 8.4 The statistic this entry adds, stated in full before its number is used downstream

The system's prediction for cluster *c* on target item *Y* is

```
pred_cdf(c) = level_cdf  +  s · ( model_cdf(c) − Σ_c w_c · model_cdf(c) )
```

with `s ≥ 0` a scalar. `s = 0` reduces exactly to B0a. Gate 3's question — *is the model
reading the card* — is then asked on `pred`, with everything else unchanged: same
records, same clusters, same items, same 0.0273 floor, same seed 17, both the
card-permuted arm (the model was really shown another cluster's card) and the derangement
null.

At `s = 1`, **unfitted**:

| | 7B | 14B |
|---|---|---|
| real W1 | 0.0974 | 0.0799 |
| card-permuted W1 | 0.1376 | 0.1488 |
| gap (needs > 0.0273) | **+0.0402** | **+0.0689** |
| derangement-null gap | +0.0403 | +0.0667 |
| items p < 0.05 | 6/10 | 8/10 |

GREEN on both arms, by 1.5× and 2.5× the floor respectively.

`s = 1` is used for this test **because it is the value that requires no fitting**, not
because it is the best of several tried. The per-item optimum ranges 0.0–2.5 and a
best-fit global `s` on these ten targets would be 0.8; that number is reported in
`scripts/deviation_probe.py` output and is **not** used anywhere, because it is fitted on
target truth and target truth is the held-out quantity.

### 8.5 What `s` will be in the real system, fixed now

`s` is fitted on **anchor** items only, by the same F = 3 cross-fitting the checklist
specifies for the rest of the calibrator (4.1–4.2): for fold *f*, anchors in *f* are
elicited from cards built from folds ≠ *f*, and `s` is chosen to minimise weighted anchor
W1 out of fold. It is therefore fitted on (raw prediction, known truth) pairs produced
under exactly the conditions targets face, and no target truth enters it.

Declared now, before the calibrator exists:

- **One global `s` per scale-length** is the primary parameterisation.
- A per-cluster `s_c` shrunk toward the global value by `n_eff/(n_eff + τ)`, τ = 100, is
  the hierarchical variant, reported as an ablation. It is the partial pooling the
  hierarchy exists to do (spec §2.2).
- **A per-item `s` is forbidden for targets**, because there is no target truth to fit it
  on. If a per-topic `s` is used it is fitted on that topic's anchors only, and topics
  with no anchors fall back to the global value — a fallback the frozen split makes
  necessary for religion, crime, family and politics.

### 8.6 Two reported modes, both declared now

- **Observed-level (primary).** `level_cdf` is the true weighted national marginal from
  the bed. This is the small-area-estimation framing — a national topline is cheap, a
  subgroup-powered one is not — and it is the *fair* comparison, because B0a is handed
  exactly the same national marginal. The difference between the two is then only whether
  the system adds subgroup structure, which is what §1.4(i) claims.
- **Predicted-level (secondary).** `level_cdf` comes from an LLM population-level call.
  This is B0b, and it is the mode a genuine scenario question has to use, because an
  unasked question has no national marginal either. Reported beside the primary mode,
  never merged with it, and it is the number the M10 honesty box quotes for scenarios.

Reporting only the primary mode and calling it scenario accuracy would be the §1.6 error
the checklist names. Both appear in every headline table.

### 8.7 Declared readings for the Layer 4 run, before the number exists

The pass marks are those already in `configs/gss_main.yaml` and are not touched:
all-40 ≤ 0.0556 against B0a 0.0695; top-quartile ≤ 0.0834 against 0.1043;
leakage-resistant ≤ 0.0614 against 0.0767; variance ratio in [0.8, 1.2].

- **Passes on all 40 targets, observed-level.** The claim in §1.4(i) holds; Stage B
  proceeds; the writeup leads with the level/structure decomposition as the mechanism and
  carries the raw Gate 3 RED as the measurement that produced it.
- **Passes on the top-quartile heterogeneity subset but not on all 40.** The honest
  reading is that the method works where clusters genuinely differ and adds nothing where
  they do not, which is a narrower claim and still a result. It is reported as that, not
  as a pass.
- **Fails on all three subsets.** The ten-item probe did not generalise, the §5.8 pivot
  applies, and the paper is the negative-result analysis — with the ρ ≈ +0.6 ordering
  measurement as the thing that makes it a mechanism rather than a shrug.
- **Predicted-level mode fails while observed-level passes.** Then the contribution is
  small-area estimation from a known topline, and every scenario claim is dropped. Said
  now so it is not a caveat added afterwards.

Neither the pass marks, the noise floors, the baselines nor the frozen split move on what
comes back.

## 8.8 Entry: the pre-registered baseline was measured on a different bed — both verdicts are reported

Written **16 Sep 2026, session 2**, with the 8B arm's number in hand and before
the 14B arm's exists. It is recorded as a discrepancy found, not as a pass mark
being moved.

The Layer 4 pass marks in `configs/gss_main.yaml` are absolute W1 numbers:
`all_targets ≤ 0.0556`, derived as 80 % of a B0a baseline of 0.0695. That 0.0695
comes from `B17_feasibility_verdict.md` §appendix, which states its own method:
**waves 2016/2018/2021/2022, n = 11,045, 88–89 classic attitudinal items.**

The bed the project actually froze is **pooled 2010–2022, n = 18,772**, and the
scored set is the **40-item target half of the frozen split** minus `dwelown`,
with cells below `truth_min_neff = 30` dropped. B0a measured on that scope is
**0.0807**, not 0.0695. Neither number is wrong; they are different estimators on
different item sets, and the difference was not visible until Phase 5 existed to
compute the second one.

**Nothing moves.** The frozen mark stays exactly where it is and every table
reports the verdict against it. But §1.4(i)'s claim is stated *relatively* —

> "achieve mean cluster-weighted Wasserstein-1 distance to true per-cluster
> response distributions **≥ 20 % lower than the national-marginal baseline**"

— so the relative rule is what the claim says, and the absolute number was one
materialisation of it using a baseline from a bed that was later replaced. Both
are therefore reported side by side for every subset, in every table, labelled
**absolute** (against the frozen mark) and **relative** (20 % below the B0a
measured on the scope being scored):

| | rule | source |
|---|---|---|
| absolute | achieved ≤ 0.0556 / 0.0834 / 0.0614 | frozen in the config before any elicitation |
| relative | achieved ≤ 0.8 × B0a on this scope | §1.4(i)'s own wording |

A run that passes one and not the other is reported as exactly that, and the
writeup leads with whichever is *harder*, which on the measured baselines is the
absolute one. Declaring this before the 14B number exists is the point: it is not
available afterwards as a choice between two framings.

### 8.9 Result of the 8B arm, against both rules

`ministral-8b-2512`, 39 targets × 56 clusters, 1 × 3, observed-level mode,
12,429 records, 99.9 % JSON ok.

| subset | achieved | B0a here | absolute mark | verdict |
|---|---|---|---|---|
| all targets | 0.0729 | 0.0807 | 0.0556 | fail both (−9.7 %) |
| top-quartile heterogeneity | 0.1032 | 0.1139 | 0.0834 | fail both (−9.4 %) |
| leakage-resistant | 0.0662 | 0.0757 | 0.0614 | fail both (−12.6 %) |

Population variance ratio **0.972**, inside [0.8, 1.2] — §1.4(ii) **passes**.
Tolerance-battery ordering ρ = **+0.870** across 49 clusters and 12 items.

The arms, macro W1: B3 supervised skyline 0.0282 (the ceiling, essentially the
noise floor) · **system 0.0729** · system at s = 1 unfitted 0.0785 · B0a 0.0807 ·
B0b 0.1647 · B4 uncalibrated 0.1655 · B1 nearest-anchor 0.2343.

Three things this fixes in place before the 14B arm runs:

1. **The calibration layer is doing the work it was built to do.** Raw elicitation
   (B4) is 0.1655, twice as bad as copying the national marginal. Calibrated it is
   0.0729. That is the §5.5 B4 ablation answered: the mechanism is not dead weight.
2. **The anchor-fitted scale transfers.** `s` fitted on anchors is 0.46 and beats
   the unfitted `s = 1` on targets (0.0729 vs 0.0785). Fitting on anchors, which
   is the whole cross-fitting claim, generalises to items whose truth was never
   seen.
3. **The 8B does not clear either rule**, and it was predictable from the anchor
   fit alone: the anchor W1 moves only 0.0658 → 0.0616. That is the §7.3a
   capability question answered downward at 8B, and it is reported as a tier
   point, not as a failure of the method — the 14B's 10-item probe on the same
   statistic gave −25.6 % where the 8B gives −9.7 %.

## 8.10 Entry: the calibration layer's design is frozen here, on the 8B arm, before the 14B arm is read

Written **16 Sep 2026, session 2**. The 8B elicitation is complete and has been
used to settle the layer's design. The 14B elicitation is **still running** and
its target records have not been scored. Everything below is fixed before that
number exists.

### The arms, and what each one is for

* **`ministral-8b-2512` — the development arm.** Checklist 0.6 gives the dev
  model this job ("every pipeline bug ... is found with a free unlimited local
  model, not with quota"); Ollama is unreachable from either shell this session
  drives, so the cheapest hosted model with real capacity takes that role. Every
  design decision below was made by looking at it.
* **`ministral-14b-2512` — the confirmatory arm.** Same items, same clusters,
  same cards, same statistic. Only the model changes. Nothing about the layer
  moves after its numbers exist.
* **`qwen2.5-ctx8k:7b-instruct-8192`** — the 2024-era point on the §7.3a tier
  curve, from records already on disk.

### What the layer does, fixed

```
pred_cdf(c) = level_cdf + s · P_r ( raw_cdf(c) − Σ_c w_c raw_cdf(c) )
```

1. **`level_cdf`** — the weighted mixture of the scored cells' truths in
   observed-level mode, the isotonically-corrected population-level LLM call in
   predicted-level mode (§8.6). B0a is handed the same object, so `s = 0` is B0a
   identically.
2. **`P_r`** — orthogonal projection onto the top `r` principal directions of the
   anchor deviation matrix: one column per (anchor item, interior CDF position),
   centred across clusters, built from **anchor truths only**. A deviation
   measured from three ensemble draws is mostly noise and noise is isotropic, so
   the component outside the space real subgroups occupy is discarded. On this
   bed two directions carry 68 % of the anchor variance.
3. **`s`** — one scalar, or one per scale length. Nothing else.

### How every free parameter is chosen, fixed

**All of it out of fold on the anchors, using the F = 3 cross-fit folds that
already exist.** Fit on two folds, score on the third, take the best:

| parameter | candidates | chosen by |
|---|---|---|
| rank `r` | 0, 1, 2, 3, 4, 6, 8, 12 | out-of-fold anchor W1 |
| scale scheme | `global`, `by_k` | out-of-fold anchor W1 |
| variance restoration | on, off | out-of-fold anchor W1 |

`by_topic` and `by_cluster` are **not** candidates for the primary arm and are
reported as ablation arms. §8.5 fixed that before any Phase 5 number existed:
per-cluster is the hierarchical variant, and per-topic is fitted on exactly the
topics the frozen split cannot serve. In-sample anchor W1 is not used to choose
anything — it always prefers the richest parameterisation, and on this split some
scale lengths carry one or two anchors.

**No target truth enters any of these choices, and no per-item `s` exists.**

### Two findings the 8B arm already fixes, reported whatever the 14B does

**F1 does not occur in this design, so the step that repairs it is dropped.**
Within-cluster variance collapse is the failure this whole method family is known
for (Bisbee et al.), and M5 step 2 exists to repair it. Measured: the population
variance ratio is **1.000 with no restoration at all**, because the prediction
inherits the national histogram's shape instead of being generated token by
token. Switching restoration on makes anchor W1 *worse* (0.0636 against 0.0589),
so the anchor-side rule switches it off. This is §5.8's calibration-kill
criterion applied to a component rather than the whole layer: report and drop.

**The subspace projection earns its place.** On the 8B, rank 3 against rank 0 is
0.0738 → 0.0661 macro W1 on the targets, and the rank was chosen on anchors.
That is the hierarchy doing statistical work in the §2.2 sense — pooling
information across clusters — rather than being an org chart of agents.

### The 8B arm's numbers, for the record and as the thing the 14B is compared to

39 targets × 56 clusters, 1 × 3, observed-level, 12,486 records:

| subset | achieved | B0a here | absolute mark | absolute | relative |
|---|---|---|---|---|---|
| all targets | 0.0661 | 0.0807 | 0.0556 | fail | fail (−18.0 %) |
| top-quartile heterogeneity | 0.0844 | 0.1139 | 0.0834 | fail | **PASS** (−25.9 %) |
| leakage-resistant | 0.0609 | 0.0757 | 0.0614 | **PASS** | fail (−19.6 %) |

Population variance ratio **1.013** → §1.4(ii) passes. Tolerance-battery
ordering ρ = **+0.893** over 49 clusters.

Arms: B3 skyline 0.0282 (the ceiling — essentially the noise floor) ·
system 0.0661 · system per-cluster 0.0645 · system at s = 1 0.0646 ·
system without the subspace 0.0738 · B0a 0.0807 · B0b 0.1647 ·
B4 uncalibrated 0.1655 · B1 nearest-anchor 0.2343.

### Declared readings for the 14B arm

Unchanged from §8.7, plus one: **if the 14B's numbers would be improved by a
different rank, scheme or variance setting, that is not a reason to change any of
them.** The selection rule is anchor-side and it runs on the 14B's own anchors;
whatever it picks there is what is reported.

## 8.11 Outcome of the confirmatory arm, against the readings declared in §8.7

Run 17 Sep 2026, `ministral-14b-2512`, 39 held-out targets × 56 clusters,
12,540 elicitations, **100 % JSON ok, 0 refusals**. Observed-level mode, ACS-raked
population weights. Nothing in the calibration layer moved after §8.10 was
written: the rank, the scale scheme and the variance-restoration switch were
chosen by the same out-of-fold anchor rule, running on this arm's own anchors.

| item set | achieved | B0a here | absolute mark | absolute | relative (−20 %) |
|---|---|---|---|---|---|
| all targets | **0.0602** | 0.0807 | 0.0556 | fail | **PASS** (−25.3 %) |
| top-quartile heterogeneity | **0.0787** | 0.1139 | 0.0834 | **PASS** | **PASS** (−30.9 %) |
| leakage-resistant subset | **0.0565** | 0.0757 | 0.0614 | **PASS** | **PASS** (−25.4 %) |
| all targets, leaky removed | 0.0611 | 0.0822 | 0.0556 | fail | **PASS** (−25.7 %) |

Population variance ratio **1.009**, inside the pre-registered [0.8, 1.2] band, so
**§1.4(ii) passes**. Interval coverage 0.899 against a nominal 0.90.

### Which declared reading this is

§8.7 declared four. This is the **second**: *"Passes on the top-quartile
heterogeneity subset but not on all 40."* — with the qualification that it passes
the relative rule everywhere and the absolute mark on two of four subsets.

Read literally against §1.4(i), which says the claim is about "the top-quartile
heterogeneity items", the claim **holds**: −30.9 %, and the pre-registered
absolute mark for that subset is cleared too. Read against the all-targets
absolute mark of 0.0556, it does not: 0.0602. §8.8, written before this number
existed, is why both are on the page and why neither could be chosen afterwards.

The honest summary is the one §8.7 wrote in advance for this case: **the method
works where clusters genuinely differ and adds less where they do not.** That is a
narrower claim than the §1.4 headline and it is the one the numbers support.

### What did not happen

* Not reading 3 (fail on all three subsets), so §5.8's pivot to a negative-result
  paper does not apply.
* Not reading 4 either — the predicted-level mode is genuinely worse (B0b 0.1787
  against B0a 0.0807), which is *not* a surprise and was declared: a scenario
  question has no topline and the system must predict one, and the model's own
  population estimate is poor. Every scenario claim therefore rests on the
  observed-level number only through the honesty box, which quotes the measured
  items rather than the scenario.

### The §5.6 view, and one thing the probe showed that was not asked for

The direct probe flags **2 of 39** items on this arm (`postlife`, `spanking`) —
recalled within 0.05 on a subgroup *and* better than copying the topline onto it.
Removing them moves the headline from −25.3 % to −25.7 %.

Unasked for, and reported because it is evidence: the 14B returns usable JSON on
**100 %** of elicitation calls and **50 %** of recall calls, the remainder being
malformed or hedged. It is markedly less willing to claim a published cross-tab
than to estimate a subgroup distribution. That is the opposite of what a model
reciting a memorised table would look like, and it is a weaker argument than the
probe itself — it is recorded as a remark, not as a result.

### The record, complete

Gate 3 on the frozen raw statistic is **RED** at +0.0011 and stays red in every
table. Gate 3 on the level-matched prediction is **GREEN** at +0.0689. The
level-anchoring mechanism (r ≈ −0.8, two model sizes) stands as a finding about
raw verbalized elicitation. Four card configurations were run and inspected before
any of this, and no card change was made in pursuit of any of these numbers: the
cards, the split, the folds and the prompts are byte-identical to the ones the red
gate was measured on.
