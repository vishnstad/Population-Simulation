# B17 — build status

**Gate 0: green. Gate 1 (Layer 0): green.** Nothing is built past Gate 1 yet, per
`START_HERE.md`.

Run everything: `PYTHON=<your python> scripts/gates.sh`

---

## Done

### Phase 0 — scaffold
| step | state | notes |
|---|---|---|
| 0.1 repo skeleton | ✅ | spec §3.2 layout, at `~/Downloads/fyp/popsim` |
| 0.2 config loader | ✅ | `popsim/config.py`; snapshots the resolved config + source + git provenance into `runs/<id>__<utc>/` |
| 0.3 pinned deps | ✅ | exact 0.3 set installed and verified |
| 0.4 llm client / cache / budget | ✅ | `popsim/llm/` |
| 0.4a provider router | ✅ | picks by remaining quota, 429 falls through, never retried on the same provider |
| 0.4b daily counters | ✅ | `quota.json`, local-midnight buckets, token caps too |
| 0.4c checkpoint every call | ✅ | cache write is unconditional and atomic |
| 0.4d call-count ceiling | ✅ | checked *before* the send; survives restart |
| 0.5 harness + CI | ✅ | pytest + `.github/workflows/ci.yml`; data gates run locally via `scripts/gates.sh` |

**The config loader refuses the spec's stale numbers.** `min_cell: 40`,
`K_target: 150`, `budget_cap_usd: 1200`, geography in `partition.axes`, 2024 in
the bed, missing or token pass marks, `checkpoint_every_call: false` — each is a
load-time error with the measurement that overrides it. Tests in
`tests/test_config.py` assert every one.

### Phase 1 — data foundation
| step | state | notes |
|---|---|---|
| 1.1 GSS adapter | ✅ | `latin1`, `usecols` always, never reads the 598 MB file whole |
| 1.2 harmonized demo | ✅ | `caste_group` dropped; `region_7222`→`region` with per-row `region_source`; **`mode` added** |
| 1.3 exclude 2024 | ✅ | logged reason, asserted in `tests/test_gss_adapter.py` |
| 1.4 item codebook | ✅ | `codebooks/gss_items.yaml`, 38 items |
| 1.4a exact wording | ✅ | 36 of 38 verified from the ballot questionnaires; the 2 unverified are `role: excluded` |
| 1.4b sanity roles | ✅ | `finrela`, `satfin` → `role: sanity`; `colrac`/`librac`/`colmil` → `excluded` |
| **1.5 Layer 0 gate** | ✅ **GREEN** | 32/32 items within 0.5 pp, worst 0.190 pp; 1 cross-tab; weights checked |

---

## Layer 0, as run

Three checks, because the gate names three bugs and they do not show up in the
same comparison.

**A. Recodes and scale reversals** — unweighted marginals vs the codebook's
published frequencies. The GSS codebook is titled *"Codebook and Unweighted
Frequencies"*, and that is a feature: an unweighted marginal is a direct
function of the recode with no weighting step in between, so a mismatch is a
recode or reversal bug and nothing else. **32/32 within 0.5 pp, worst 0.190 pp.**
Run across the whole survey rather than just the pool, 858 of 869 admissible
published tables pass; the 11 that do not are household count variables whose
published tables collapse their upper tail, plus two ballot-version splits —
none in the pool.

**B. Cross-tab machinery** — `degree × sex`, both margins reproducing their
published toplines (worst 0.085 pp). The codebook prints no cross-tabs, so this
is what is actually checkable: it exercises the real groupby-and-renormalize
path every `ClusterStats` histogram will use, against published numbers rather
than a second copy of our own arithmetic.

**C. Weight misuse** — against ACS 2024, not the codebook, which publishes no
weighted figures. `sex` is held to 1 pp (no two-year drift story exists for the
adult sex ratio); `degree` and `age_band` to 3.5 pp, the lag the checklist
measured at 5.3a.

A deliberately reversed `finrela` scale is injected in
`tests/test_layer0_gate.py` and must turn the gate red. If it does not, the gate
is not testing anything.

---

## Things found on the way that are not in START_HERE's eleven traps

1. **Mode is nearly collinear with wave in this bed.** 2010–2018 are
   essentially all in-person; 2021 is 87 % web; 2022 is 46 %. NORC publishes a
   mode-sensitivity classification, and **`natroad` — the intended demo item —
   is on the "Likely mode sensitive" list**, along with `spkrac`, `conlegis` and
   `conmedic`; `pray` and `suicide1` are "requires further investigation".
   `mode` is now carried per respondent and per item in the codebook.

2. **The `homo` and `mil` tolerance sub-batteries were dropped from GSS after
   2021.** `spkhomo colhomo libhomo spkmil colmil libmil` have **n = 0 in 2022
   and 2024**. They are still fine in the pooled 2010–2022 bed, but 5 of the 32
   leakage-resistant items are among them, and none has 2022 instrument wording.

3. **`wtssall` is empty from 2021 on.** `wtssps` is 100 % populated across
   2010–2024 and is the only weight spanning the bed; the configured fallback
   never fires here.

4. **GSS missing values arrive as letter codes** (`d` `i` `n` `s` …) in the same
   column as integer responses, so every item column is `object` dtype. In *this*
   release pyreadstat maps them to NaN, but the adapter coerces defensively
   either way and keeps refusal separable from inapplicability.

5. **The .dta value-label map is cumulative across 35 waves.** `homosex` still
   labels code 5 ("other"), which has zero respondents in every wave of the bed.
   The scale used downstream is the fielded one — offering an option nobody ever
   chose would put probability mass where no truth histogram can score it.

6. **`fair` and `postlife` differ from the codebook by 12 and 3 cases.** The
   codebook PDF is Release 4 (Nov 2024); the `.dta` files on disk are an earlier
   release, and the standalone `GSS2022.dta` agrees with the cumulative file.
   Shapes match to 0.19 pp. The gate is on the distribution, as the checklist
   specifies; n is reported, with a 2 % bound that still catches a filtering bug.

7. **`acs2024_margins.parquet` is correct.** Recomputed independently from
   `psam_pusa/b.csv`: shares 0.1026 / 0.4707 / 0.0847 / 0.2104 / 0.1316 over
   267,241,508 adults — exact match, and the SCHL→`degree` recode is confirmed
   as 1–15→0, 16–19→1, 20→2, 21→3, 22–24→4.

8. **Weighting moves `degree` and `age_band` slightly *away* from ACS** (L1
   1.53→9.23 pp and 7.39→7.72 pp) while pulling `sex` sharply toward it
   (5.65→0.59 pp). Every category is inside the 2.9 pp the checklist measured
   and accepted as time lag, so the gate passes and flags it. Worth watching
   across waves before raking is relied on.

---

## The wording problem (1.4a) — resolved

The checklist says: *exact question wording transcribed from the GSS 2022
codebook PDF.* That is not possible from that PDF. **The codebook stores each
question in a SAS label, and SAS labels cap at 256 bytes**, so long questions are
cut off mid-sentence *in the published codebook itself* — `SPKATH` ended at
`"...against churches and rel"`. Five more items were not fielded in 2022 at all.
This is the dangerous kind of bad data: it reads as complete right up to where it
stops.

**The instrument is the questionnaire, not the codebook.**
`scripts/extract_questionnaire.py` parses the six ballot PDFs in
`data/gss/quex/` and now supplies verbatim wording for all 13 previously blocked
items. `Item.assert_elicitable()` still refuses to build a prompt from anything
unverified, so the guard stays useful as the pool grows.

Three things the extractor does that are worth knowing:

- **Two instrument generations, two formats.** 2021 prints `SPKATH: RadioButton`
  with quoted text and `[1] label` codes; 2022 prints
  `SPKATH: Categorical (Single)` with bare text and `{token} Label` categories.
  The 2022 ballot wins for items still fielded in 2022, the 2021 ballot only for
  the five dropped after it — the wording is not identical between them.
- **Battery preambles are reconstructed, scoped to their group.** `COLATH` reads
  only "Should such a person be allowed to teach…"; the preamble is printed once
  on `SPKATH`. The codebook's convention is to restate it in parentheses, and the
  extractor reproduces that, keyed on the tolerance group (`ath`, `rac`, `com`,
  `mil`, `homo`) so a preamble cannot leak into an unrelated item.
- **Anything that does not come out clean is refused, not written.** Three checks
  — length, a question mark present, no leading lowercase. They caught real
  breakage: `SPKMIL` extracted as *"Consider a person who advocates doing away
  with elections and letting the military run the country."* — a fluent, complete
  sentence missing the thing being asked.

It was checked against the codebook for all 18 items whose codebook entry was not
truncated. Two differences are real and neither is a bug:

- the `con*` battery restates the full intro in the codebook ("these
  institutions") where the ballot asks per item ("this institution");
- **`spkrac` wording changed.** The codebook says "Blacks are genetically
  inferior"; the 2022 instrument says "**Black people** are genetically
  inferior". The codebook label lags the instrument. The instrument text is what
  respondents saw, so that is what is in the codebook now.

### Two traps found while doing it

**The tolerance battery is not coded consistently.** §5.2a's metric is a Spearman
correlation of the ordering of five target groups, so a misaligned column
silently inverts:

| | permissive answer | |
|---|---|---|
| `spk*` | code 1 ("yes, allowed to speak") | low code |
| `col*` | code 4 ("yes, allowed to teach") | low code |
| **`colcom`** | code 5 ("not fired") | **high code** |
| **`lib*` (all five)** | code 2 ("not remove") | **high code** |

One cell *and one whole row* run the other way. Every battery item now carries
`tolerant_code`, derived from its option labels rather than hard-coded, and
`reverse_coded`. Aligning on `tolerant_code` is not optional for §5.2a.

**`COL*` items are asked of both split-ballot arms.** `SPKHOMO` and `SPKHOMOY`
branch on `VERXY`, but `COLHOMO` follows with no branch and refers to "such a
person" — so half the respondents answer it having read "a man who admits that he
is homosexual" and half having read "a gay person". Its truth histogram mixes two
referent framings. NORC publishes it as one variable; worth a limitations line.

---

## The split-ballot wording twins

16 of 38 pool items have a `*Y` twin: the same construct fielded in a second
wording on the other half of the ballot, published by NORC as a separate
variable with its own topline. **Decision: base variable only**, since §1.4a
makes a reworded item a different item. `build_codebook` raises if a twin ever
enters the pool, so the two can never be silently merged.

The cost is recorded per item per wave in the codebook, and it is not small:

| family | waves affected | effect |
|---|---|---|
| `natarms` `nateduc` `natspac` | **all 7** | ~half the sample in *every* wave |
| `spk/col/lib` civil liberties | 2021, 2022 | ~half the sample in those two |

Thirteen of the fourteen affected items are in the §1.5 leakage-resistant 32, so
this bites exactly where the headline-both-ways reporting lives. `spkath` has
n = 1,147 in 2022, not the ~2,300 the ballot design alone suggests — and *that*
is the n the `min_cell` reasoning and the Layer 1 noise floor have to use.

`natroad`, the demo item, has no twin and is unaffected.

---

## Compute and quota

**No department GPU access.** Two consequences worth carrying into the writeup:

- The F2 risk (between-cluster collapse) that §7.2 calls the most likely failure
  is *higher* with a small open model than it would be on a 14–32B local model.
  The week-4 permutation test (Gate 3) is now the early-warning that matters
  most, and the §7.2 escalation ladder — frontier model, richer cards, more
  anchors — is the only response available.
- §7.3b (logprob vs verbalized elicitation) depended on a local model exposing
  full token logprobs. On CPU-only local inference this is still possible but
  slow; treat it as optional rather than planned.

**Ollama context window is a silent trap.** Ollama's default `num_ctx` is 2048.
A stat card plus question is ~1,300 tokens and grows with `anchors_per_card`,
and Ollama **truncates rather than erroring** — the model would answer on a
half-read card and the failure would look like F2 (between-cluster collapse)
rather than a config bug. Build a variant with a bigger window before any
elicitation:

```
printf 'FROM qwen2.5:7b-instruct\nPARAMETER num_ctx 8192\n' > Modelfile
ollama create qwen2.5-7b-ctx8k -f Modelfile
```

then point `llm.providers[ollama].model` at `qwen2.5-7b-ctx8k`.

Two guards now make this impossible to hit silently:

- **`popsim doctor`** asks the running Ollama what window the model actually has
  and compares it to `llm.num_ctx`. Also reports which API keys are set.
- **The client refuses to send** a prompt that would not fit, counting the
  reserved reply tokens as well as the prompt. `ContextWindowExceeded` is fatal
  rather than retried, because a retry would fail identically.

**Ollama could not be installed from here.** `ollama.com` and
`registry.ollama.ai` are both refused by the network allowlist
(`403 ... policy denial`), and the sandbox has 3 GB RAM and 6.7 GB free disk,
which would not run a 7B model even if the download succeeded. It has to go on
your own machine — and if it does, the pipeline's LLM phases have to run there
too, since the sandbox cannot reach your machine's `localhost:11434`.

**Why the dev model is Qwen 2.5 and not a current Qwen.** Checklist 0.6 names
Qwen 2.5 7B / Llama 3.1 8B, and for the *dev* slot that still holds — its job is
finding plumbing bugs for free, where boring and predictable beats capable.
Three specific reasons not to swap a 2026 reasoning model into that slot:

1. **Thinking mode breaks two things the project measures.** §1.2a's cost table
   assumes ~250 output tokens per call; a reasoning model emits far more. And
   M4 *reports malformed-JSON rate as a metric* — a model that sometimes emits a
   reasoning preamble inflates a number that appears in the writeup. Forcing
   thinking off has its own documented structured-output failures.
2. **The local arm is also an experiment, not just a tool.** §7.3a asks *does
   the calibration layer rescue a weak model?* That needs a genuinely weak arm to
   be worth answering. A current frontier-adjacent open model is not it.
3. **Leakage (F6, §5.6).** A 2026-trained model has seen more GSS cross-tabs,
   more OpinionQA, and more silicon-sampling papers *with their result tables*
   than a 2024-trained one. The older open arm makes §5.6's leakage story
   cleaner, and the "leakage vs wave age" side-finding is one of the novel bits.

So: **keep Qwen 2.5 7B as the dev model, and add a current open model as a
second, separate arm** for §7.3a/§7.3b. `configs/gss_main.yaml` now carries both
(`ollama` and a parked `ollama_modern`). Having a 2024-era *and* a 2026-era open
model in the tier ablation is a better result than either alone.

**Pin the exact tag.** `cache.py` keys responses on the model string, so a
floating tag (`qwen3`, which Ollama resolves to `:latest`) lets two different
sets of weights share one cache key — a resumed run would mix responses from two
models into one ensemble and report the spread as elicitation variance. The
config loader now refuses an unpinned Ollama model.

**Groq's real cap is tokens, not requests.** Advertised as 1,000 requests/day,
but also 200,000 tokens/day — at this project's ~1,550 tokens per elicitation
call that is about **129 calls/day, not 1,000**. An 8× difference between the
number on the tin and the number that stops the run. `quota.py` now tracks daily
token caps alongside call caps and `calls_left_today()` schedules on the minimum
of the two.

---

## Open items — need you

- **0.6 local model — must be on your machine.** I could not install it: the
  Ollama hosts are blocked by the network allowlist and the sandbox has 3 GB RAM.
  On your Mac: `brew install ollama && ollama serve`, then
  `ollama pull qwen2.5:7b`. The client is already wired for it
  (`llm.active_providers: [ollama]`, base URL `http://localhost:11434/v1`).
- **0.7 register the free tiers.** Mistral (primary), Groq, Google AI Studio,
  then Cerebras / GitHub Models / NVIDIA NIM as overflow. Export
  `MISTRAL_API_KEY`, `GROQ_API_KEY`, `GOOGLE_API_KEY`, `CEREBRAS_API_KEY` and
  confirm the per-day caps in `configs/gss_main.yaml` against each console —
  they move.
- ~~0.8 department GPU access~~ — **none available.** Consequences recorded above.
- ~~1.4a wording~~ — **done.** All 13 extracted from the ballots; 36 of 38 items
  verified, the 2 remaining are `role: excluded` and need no wording.
- **Spot-check the extracted toplines.** Every entry in
  `codebooks/gss2022_published_toplines.yaml` is `verified: false`. Ten minutes
  against the PDF on the pool items would make it a reviewed artifact.
- **Decide on the 6 dropped tolerance items.** They are in the pooled bed but
  have no 2022 data, so they can never be gated against a published 2022 topline
  and cannot appear in a 2022-only comparison.
- **Re-check the §1.2 item counts against base-only n.** The checklist's "74
  items clear SNR ≥ 1.5" was measured before the split-ballot twins were
  accounted for. If that measurement pooled base + `*Y`, the effective n for 14
  items is half what it assumed, and the noise floor for those items rises.
  Phase 2.3 will settle it — worth knowing the number may move.

## Phase 2 — done, Gate 2 green

| step | state | notes |
|---|---|---|
| candidate pool | ✅ | 148 items. The named 38 is what the checklist *discusses*; Layer 1 has to search the whole repeated item bank |
| 2.1 partition | ✅ | top-down supervised tree, **K = 56**, min leaf exactly 60 |
| 2.2 ClusterStats | ✅ | weighted histograms, Kish n_eff, per-item answered n |
| 2.3 noise floor | ✅ | `evalx/noise.py`, split-half, **mean noise W1 0.0265** vs the checklist's measured 0.0273 |
| 2.4 item split | ✅ | 40 targets / 34 anchors, frozen to `codebooks/item_split.frozen.yaml` |
| 2.5 pooling | ✅ | τ = 100, median shrinkage λ 0.48 |
| **2.6 Layer 1 gate** | ✅ **GREEN** | 140 of 148 clear SNR ≥ 1.5 |

### How the pool was built

`scripts/build_pool.py`, four filters: **Replicating Core** (the GSS codebook
assigns every variable a section; the core is the set repeated every round — the
rest are topical modules asked once, which cannot support a pooled-wave truth
histogram) → scale 2–7 with a complete published topline → wave coverage
measured from the file → attitudinal, by an explicit exclusion list grouped so a
reviewer can disagree with any single group. 5,701 value-labelled variables →
470 core → 148.

### The partition

Top-down is the whole point of 2.1. The spec builds the full cross and merges
what came out too small, which needs a distance between histograms estimated
from ~1.6 people — undefined, not noisy. A top-down tree checks the constraint
*before* splitting, so the tiny cell never exists and there is nothing to merge.

**K = 56, minimum leaf exactly 60.** The split criterion is weighted
within-group variance on normalized item scores, with contiguous cuts on ordered
axes so every leaf is a describable range.

Splitting on responses could in principle tune the partition to the targets.
Measured rather than argued: **adjusted Rand 0.9962** against the plain
demographic cross (60 cells). The tree is the cross with four cells merged, so
the freedom to overfit is not there.

### The noise floor, and a factor of two

SNR = between-cluster signal / sampling noise in the truth. Signal is the
cluster-weighted W1 from the national marginal — which is exactly baseline B0a,
so the thing measured is the thing the project has to beat.

The signal reproduces the checklist almost exactly (`natroad` 0.087 vs 0.087,
`satfin` 0.095 vs 0.098, `finrela` 0.070 vs 0.073). The noise did not, at first:
mean 0.0375 against their 0.0273, a ratio of 1.37.

That ratio is √2, and it was the correction constant. Two independent halves of
a cell differ by **2×** the error of a full-size sample, not √2×: each half has
SD σ√2, and the difference of two independent halves has SD √2·σ√2 = 2σ. With
the right constant the mean noise floor is **0.0265 against their 0.0273** — a
3% match, which is about as good as two independent implementations get. The
wrong constant moves every SNR by 40% and the item count with it.

### Gate 2 results

- **140 of 148 items clear SNR ≥ 1.5.** The checklist expected "74 available",
  but that is a claim about *their* pool, not a ceiling — ours is wider. Worth
  saying plainly: at K = 56 on the pooled bed the SNR gate is no longer the
  binding constraint. Item *quality* is — leakage, mode sensitivity, the sanity
  roles.
- **40 targets, 30 of them leakage-resistant** (the §1.5 list is 32; the two
  missing are `finrela` and `satfin`, which are `role: sanity` — exactly right).
  11 topics represented, target SNR 2.04–5.89.
- **441 of 8,288 (cluster, item) cells have n_eff below 30** and are excluded
  from scoring per spec §5.2. That is 5%, and it is why n_eff is recorded rather
  than raw n.
- Only **7 anchor-only low-SNR items**, not the 14 the checklist budgeted —
  there simply are not 14 eligible items below the gate in this pool.

### A third trap, caught by Layer 0

**`racopen` in the cumulative file is not the variable the 2022 codebook
documents.** GSS ran a three-arm wording experiment on it; the codebook
documents arms of 588, 554 and 1,153, while the cumulative column carries 1,173
answers over three codes. My code was silently dropping the 14 respondents in
the third option, because "absent from the published table" had been treated as
"nobody chose it". Those are different statements and conflating them discards
people. The codebook builder now checks observed usage against the bed and
refuses; `racopen` is excluded with that reason recorded.

I checked whether this affects the other 25 items with `v`/`nv` siblings — it
does not. For `fair`, `postlife`, `courts` and the rest the base column is its
own variable and the small n differences really are release vintage, as
originally recorded.

## Phase 3 — built, Gate 3 ready to run

| step | state | notes |
|---|---|---|
| 3.1 stat card | ✅ | `agents/statcard.py` + `prompts/statcard.jinja` |
| 3.2 paraphrases | ✅ | three, hand-written, checked in, no roleplay framing |
| 3.3 `elicit.py` | ✅ | structured JSON, retry ≤ 2, refusals counted separately |
| 3.4 contract + cross-fit | ✅ | F = 3 folds stratified by topic; card contract enforced |
| **3.5 Gate 3** | ⏳ | smoke run RED, and the red was **partly our bug** — see below. Fixed; needs a re-run |

### What the card looks like

A real card renders at **~1,195 tokens** (against `num_ctx` 8192), carrying the
group definition, its population share, demographic marginals, and 12 anchor
items with full instrument wording and histograms.

Two things the checklist is specific about, and why:

**Geography and mode are marginals, never constraints.** The card says the group
is "Men, aged 65+, whose highest qualification is high school", and then
*separately* that 22% of them live in the South Atlantic and 67% were
interviewed in person. Writing "men in the South" would assert a constraint the
cluster does not have, and the model would condition on it.

**Disjunctive leaves are rendered as disjunctions.** A merged leaf reads "whose
highest qualification is a bachelor's or a graduate degree", not
"college-educated".

### Near-duplicate exclusion

`spkath`, `colath` and `libath` are one proposition in three venues — speak,
teach, keep a book in the library. A card built to elicit one of them may not
show the other two, or it hands the model most of the answer. The `nat*`
spending battery is deliberately *not* treated this way: foreign aid and highway
spending are different questions that happen to share a stem, and excluding them
from each other's cards would throw away real conditioning. The proper
similarity router (M7, 6.1) replaces this later.

### The gate was dry-run against three fake models

Before spending hours of local inference, the harness was checked against models
whose behaviour is known:

| fake model | behaviour | verdict |
|---|---|---|
| oracle | returns the true histogram of whoever the card describes | **GREEN**, reading the card |
| hedger | one fixed answer per item, ignores the card | **RED — F2**, correctly diagnosed |
| refuser | refuses everything | **RED — NO DATA**, refusal rate flagged per item |

That exercise changed the gate. The first version also required permuted W1 ≥
B0a, as the checklist expects — and it failed the *oracle*, because adding noise
to the oracle's histograms flattened them toward uniform, and a flattened
histogram can sit closer to a cluster's truth than another cluster's sharp one.

That is not a hypothetical: flattening toward uniform **is** F1, the variance
collapse this whole method family is documented to suffer and the calibration
layer exists to repair. Failing a model for it on the gate whose job is to
detect F2 would be the wrong answer to the wrong question. So Gate 3 now turns
on the real-vs-permuted gap alone, measured against the truth's own noise floor,
and permuted-vs-baseline is reported as a diagnostic with both readings named
(variance collapse, or §5.6 leakage — the leakage probe distinguishes them).

With exact per-cluster truths, permuted does land above B0a as the checklist
predicts: **0.140 vs 0.106** on this bed.

## Next, once those land

**Phase 3 — elicitation.** This is the first phase that needs an LLM, and the
first that can fail for a reason worth learning from:

- 3.1 `statcard.jinja` — geography as a marginal line, never a constraint
- 3.2 three hand-written paraphrases, checked in, never model-generated
- 3.3 `elicit.py` — structured JSON, temp 0.7, retry ≤ 2, refusals logged
- 3.4 contract test: a card for item Y contains no Y, no near-duplicate, no
  same-fold anchor
- **3.5 Gate 3, the permutation test** — shuffle which stat card goes to which
  cluster, re-run on 10 items, ~40 calls. If permuted ≈ real, the core premise
  is not working and there are ten weeks to respond, not two. The checklist
  calls it "the single most valuable test in the document".

Before that: `popsim doctor` green, which needs Ollama installed and the three
API keys exported.


---

# Gate 3 — the smoke run, and what was wrong with it

`popsim gate3 --n-items 4 --n-clusters 8 --profile dev` on
`qwen2.5-ctx8k:7b-instruct-8192`, 16 Sep 2026, 64 calls:

```
real      W1 0.2577
permuted  W1 0.2750   (+6.7% vs real)
B0a       W1 0.1250
noise floor  0.0273
=> RED: F2 — the model is not reading the stat card
```

**That verdict was wrong about the mechanism, and one of the four items was
measuring a bug in our own prompt.** Re-analysed from
`runs/gss_main__20260916T070311Z/permutation_raw.parquet` with no new calls:

## 1. The model is not hedging

F2's signature is one answer per item regardless of the card — that is what the
`hedger` fake does, and its predictions have zero spread across clusters. The
7B's do not:

| item | prediction range across 8 clusters | per-option SD |
|---|---|---|
| `xmovie` | 0.15 → 0.72 | 0.185 |
| `dwelown` | 0.12 → 0.65 | 0.176 |
| `natroad` | 0.14 → 0.72 (opt 3) | 0.133 |
| `pray` | — | 0.032–0.064 |

And on two of the four items the variation tracks the truth, significantly,
against the full derangement null: **`xmovie` p = 0.038, `pray` p < 0.001**. The
other two, `dwelown` p = 0.78 and `natroad` p = 0.95, carry no signal at all.
So the card is being read; the aggregate says otherwise because it is a mean
over items that behave oppositely.

## 2. `natroad` and the positional contract

> **Corrected 16 Sep, after the clean run.** The conclusion below — that
> `natroad`'s error was the positional response format — was **wrong about the
> cause.** With the keyed contract in place and 0% positional replies, `natroad`
> came back at 0.4137 against the smoke's 0.4199: essentially unchanged. The
> reversal evidence was consistent with two hypotheses and this note backed the
> wrong one. The real cause is the anchor, and it is in the next section. The
> contract change stands on its own merits — the ambiguity was real, 19 items
> carry it, and `textorder` in the dry-run proves it would have bitten — but it
> was not what was breaking `natroad`.

Decomposing the real-arm W1 into the model's *level* error (its population
rollup vs the national marginal) and what is left:

| item | level error | raw W1 | after level correction | B0a |
|---|---|---|---|---|
| `xmovie` | 0.048 | 0.178 | 0.134 | 0.206 |
| `dwelown` | 0.207 | 0.251 | 0.083 | 0.065 |
| `pray` | 0.129 | 0.182 | 0.144 | 0.166 |
| **`natroad`** | **0.384** | **0.420** | 0.109 | 0.064 |
| pooled | 0.192 | 0.258 | 0.118 | 0.125 |

`natroad`'s level error is 0.384 on a three-point scale, which is close to the
maximum available. The cause:

```
truth national   [0.482 too little, 0.418 about right, 0.100 too much]
model mean       [0.159, 0.366, 0.475]
W1 as-is            0.3489
W1 vs REVERSED      0.0327      <- ten-fold collapse, and inside the noise floor
```

The item's verbatim wording is *"(... are we spending **too much, too little,**
or about the right amount on) Highways and bridges"* — it enumerates the options
in a different order from `labels` / `codes`, which is the order the truth
histogram is built in. The response contract was a positional array
(`{"percentages": [n, n, n]}`), so when the model answered in the *wording's*
order nothing in the pipeline could tell. It was not a wrong answer being scored
as wrong; it was a right answer being scored in the wrong bins.

`dwelown`'s level error is real, not this (reversing makes it worse: 0.19 →
0.43). The model puts home ownership at 38% against a true 64%.

**19 codebook items have this conflict — the whole `nat*` spending battery plus
`polviews`.** 8 of the 40 targets, 6 of the 34 anchors:

```
targets  natarms nateduc natenvir natmass natroad natsoc natspac polviews
anchors  nataid natchld natcrime natenrgy natpark natrace
```

Neither side of the conflict can be edited away: the wording is the instrument
(§1.4a) and the codes carry the published toplines Gate 0 is built on. Note that
the *cards* were never affected — anchor histograms are rendered with a label
next to every figure. Only the target item's reply was positional.

**Fix: the response contract is now a JSON object keyed by the option label.**

```
{"percentages": {"too little": 48, "about right": 42, "too much": 10}}
```

The parser returns the histogram in `codes` order whatever order the keys came
in, so the conflict is harmless by construction rather than by the model
cooperating. A positional array is no longer accepted — it is recorded as a new
failure kind `positional_response`, and an unrecognised option name as
`label_mismatch`, both reported in `failure_rates`. Accepting an array "to be
lenient" would reintroduce this exact bug on the 19 items where it does the most
damage. `data.codebook.wording_option_order_conflicts()` measures the conflict
list from the codebook rather than hard-coding it, and
`tests/test_phase3.py` pins both halves: that the conflict still exists and is
still ≥ 15 `nat*` items plus `polviews`, and that a reply in the wording's order
now lands in the right bins.

## 3. Seed 17 drew a favourable derangement

The permuted arm as specified is *one* derangement out of 8!. Reassigning
already-collected histograms is free, so the whole distribution is now computed:

```
seeded (seed 17)   permuted W1 0.2750     gap vs real +0.0173
expected null      permuted W1 0.2840     gap vs real +0.0262
noise floor                               0.0273
```

Seed 17 handed over 0.009 of apparent degradation by itself — a third of the
noise floor the gap is compared against. **The gate still turns on the seeded
real-vs-permuted gap, as frozen.** `null_mean_w1`, `null_p_value` and
`n_derangements` are reported alongside it as diagnostics, so how much of any
future gap the seed chose is visible rather than inferred. The dry-run shows the
same effect on the oracle (seeded 0.0992 against an expected 0.1860), so it is a
property of single-derangement estimation, not of this model.

## 4. Two bugs in the Gate 3 driver

- **Cards were built once per cluster with `target_item=items[0]` and reused for
  every item.** 3.4's contract is per target item — no near-duplicate of *that*
  item, no anchor from *that* item's fold — so the cards were only contract-
  correct for the first of the four. Now built per `(item, cluster)`. Costs no
  extra calls; a call is made per (item, cluster, arm) either way.
- **`dwelown` should never have been in the item selection.** It is a household
  fact — "do you own your home, pay rent, or what" — and `pool.py`'s own
  exclusion list exists to remove exactly that: *"the respondent's situation,
  not an opinion — predicting it from demographics is the §F3 objection in its
  purest form"*. The `household_composition` group caught counts of people
  (BABIES, PRETEEN, TEENS, AGED) but never tenure. It ranked **second of forty**
  at SNR 4.91, which is the tell: Gate 3 takes the highest-SNR targets, and the
  highest-SNR targets are the most demographically determined ones, so anything
  that slipped the attitudinal filter surfaces right at the front.

  `item_split.frozen.yaml` is **not** reopened for this (2.4). `dwelown` is
  flagged in `pool.POST_FREEZE_DOCTRINE_FLAGS`, held out of item selection, and
  is to be reported as its own stratum. `xmovie` and `news` are reported
  behaviours rather than attitudes, but §5.1 says "attitudinal/**behavioral**",
  so they stay — worth a limitations line that the top of the SNR ranking is
  behaviour-heavy.

## 5. The dry-run is now checked in

STATUS.md recorded that the three-fake exercise is what changed the gate's
shape, but it was not in the repo, so that reasoning could not be re-run.
`scripts/gate3_dryrun.py` now holds it, with two fakes added for the response
contract:

| fake | behaviour | verdict |
|---|---|---|
| `oracle` | true histogram of whoever the card describes | GREEN |
| `hedger` | one fixed answer per item | RED — F2 |
| `refuser` | refuses everything | RED — no data |
| `positional` | right answer, as a JSON array | RED, 100% `positional_response` |
| `textorder` | right answer, keyed, in the **wording's** order | GREEN, real 0.0132 |

`textorder` is the behaviour the 7B actually showed on `natroad`. It now scores
identically to the oracle. Under the old contract it scored 0.349 against a
truth it was 0.033 away from — which is the regression this guards.

## What this does and does not license

It does **not** turn the smoke run green. The gate is red on the numbers that
exist, no Phase 4 work starts, and the level-corrected column above is a
diagnostic — a pass mark read off after seeing results is not a pass mark.

What it says is that **the red has not been measured yet on a clean run.** One
of four items was scored in the wrong bins, the cards violated the 3.4 contract
for three of the four, the item set included one out-of-doctrine fact, the
ensemble was 1×1 so every histogram was a single temp-0.7 draw, and the null was
a single derangement that happened to flatter the permuted arm. §7.2's ladder
spends frontier quota to answer a question about the model; none of that should
be spent until the question is actually about the model.

**Next: re-run at the specified scope on the fixed pipeline, still on the local
7B.** The prompt changed, so `.llm_cache` will not hit and these are fresh
calls; the permuted arm reuses the real arm's prompts, so ~480 of the ~960 are
unique — about 22 minutes at the smoke's measured 2.8 s/call.

```
cd ~/Downloads/fyp/popsim
popsim gate3 --n-items 10 --n-clusters 16 --profile permutation
```

If that comes back red with the card being read and the level error gone, the
red is about the model and §7.2 starts — frontier model first, Mistral and Groq
keys already set.

---

# Gate 3 — the clean run, and the anchor that broke the spending battery

`--n-items 10 --n-clusters 16 --profile permutation`, 960 calls, 16 Sep.
**100% JSON ok, 0% refusals, 0% positional, 0% label mismatch** — the keyed
contract holds and the model complies with it.

```
real      W1 0.2038
permuted  W1 0.2309   (+13.3% vs real)
B0a       W1 0.1074
null mean W1 0.2400   (1414 derangements, real beat 71%)
noise floor  0.0273
=> RED — by 0.0002.
```

The seeded gap is **+0.0271 against a 0.0273 floor.** Against the expected null
it is +0.0361, which clears it. The gate as frozen reads the seeded gap, so:
red. It is as thin as a margin gets.

## The pooled number is hiding two populations

| item | real | perm | null | B0a | gap | p | r(model, truth) |
|---|---|---|---|---|---|---|---|
| `xmovie` | 0.0605 | 0.1979 | 0.2423 | 0.1614 | +0.1374 | **0.000** | +0.91 |
| `homosex` | 0.1005 | 0.1820 | 0.1904 | 0.1091 | +0.0816 | **0.000** | +0.88 |
| `spkcom` | 0.0971 | 0.1410 | 0.1506 | 0.1207 | +0.0439 | **0.004** | +0.65 |
| `news` | 0.1547 | 0.1890 | 0.1968 | 0.1422 | +0.0343 | **0.000** | +0.77 |
| `colath` | 0.0946 | 0.1227 | 0.1343 | 0.0993 | +0.0282 | **0.014** | +0.79 |
| `fear` | 0.1166 | 0.1383 | 0.1513 | 0.0978 | +0.0218 | 0.115 | +0.29 |
| `pray` | 0.2275 | 0.2290 | 0.2454 | 0.0957 | +0.0015 | **0.038** | +0.31 |
| `libcom` | 0.2732 | 0.2447 | 0.2518 | 0.0948 | −0.0285 | 0.743 | +0.73 |
| `natsoc` | 0.4998 | 0.4772 | 0.4675 | 0.0553 | −0.0226 | 0.963 | **−0.58** |
| `natroad` | 0.4137 | 0.3870 | 0.3692 | 0.0974 | −0.0267 | 1.000 | **−0.21** |

**Six of ten items are significant at p < 0.05**, with cluster-level
correlations to the truth of +0.65 to +0.91. Four of them (`xmovie`, `homosex`,
`spkcom`, `colath`) also beat B0a outright. Three items carry no signal, and the
pooled mean is dragged to the threshold by the two with W1 above 0.4.

**This is not F2.** F2 is the model emitting one hedged answer per item; that is
the `hedger` fake, whose per-cluster spread is zero and whose gap is exactly
0.0000. Six items refute it directly.

## What actually broke `natroad` and `natsoc`

The model was reading the card *too faithfully*, off one anchor:

```
national truth, share saying "too much", all 18 nat* items
  nataid     57.7%   <-- the only spending anchor on the natroad/natsoc cards
  natfare    41.7%
  natarms    31.8%
  natspac    28.4%
  ... twelve items between 5% and 16% ...
  natroad    10.0%   <-- target
  natsoc      6.6%   <-- target
  natpark     5.4%

              r(model, card's nataid)   r(model, truth)
  natroad            +0.845                 −0.206
  natsoc             +0.896                 −0.639
```

Foreign aid is the one thing Americans say too much is spent on. It sat alone on
every spending card at 57.7% "too much", the model copied it across to highways
and Social Security cluster by cluster at r ≈ +0.87, and its predictions came out
*anti-correlated* with the truth. Model mean "too much" for `natroad` was 0.463
against a true 0.097. That is over-conditioning on an unrepresentative exemplar
— the opposite failure from the one the gate is named for, and it reads as F2
because both arms score alike when the level is this wrong.

`libcom` is a milder version: `libmslm` ("a Muslim clergyman preaching hatred of
the US") sits at 50.2% "remove from the library" where the rest of the `lib*`
battery runs 18–27%, and `libcom` came back with good conditioning (r = +0.73)
on a badly wrong level.

## Why that anchor was there: the 3.4 clause that was never checked

`make_crossfit_plan` sets each target's excluded fold to **the fold densest in
same-topic anchors**, in its own words *"so a target's excluded fold is the one
most likely to contain a near duplicate of it."* That is right. But
`anchors_for()` ignored it for targets and returned all 34 anchors, and
`assert_card_contract` — whose docstring reads *"§3.4: no Y, no near-duplicate
of Y, no same-fold anchor"* — never checked the third clause. So `target_fold`
was computed with care, written onto every card, and enforced nowhere.

`_pick_anchors` breaks ties with `min()`, i.e. alphabetically. `natroad`'s
excluded fold is 0. `nataid` is in fold 0 and sorts first among the spending
anchors. **The battery's statistical outlier was on every spending card because
of alphabetical order, out of the one fold that was supposed to be gone.**

Fixed: `anchors_for` now excludes `target_fold` for targets as it already did
for anchors, and `assert_card_contract` takes `fold_of` and raises on a same-fold
anchor. Two tests pin it. Enforcing the specified contract removes the bad
anchors on its own, with nothing tuned toward the result:

| target | before | after |
|---|---|---|
| `natroad`, `natsoc` | `nataid` (57.7% too much) | `natchld` (9%), `natcrime` (4%) |
| `libcom`, `colath`, `spkcom` | `libmslm` (50.2% remove) | `spkmslm` |

Two representative spending anchors now, where there was one outlier.

## Still open, and a decision

`_pick_anchors` still takes one arbitrary exemplar per topic, chosen by name.
That worked out here — `natchld` is representative — but it is luck, not design,
and a battery target arguably needs several of its own battery to see the spread.
That is §7.2's rung 2 ("richer cards") and it is not done.

**Next: re-run on the fixed cards, same local 7B, same scope.** One variable
changed, so the effect is attributable. The diagnosis predicts `natroad` and
`natsoc` move a long way; if they do, the pooled gap clears 0.0273 without
anything having been tuned to make it.

```
cd ~/Downloads/fyp/popsim
popsim gate3 --n-items 10 --n-clusters 16 --profile permutation
```


---

# Gate 3, run 3 — the fold fix worked exactly as predicted, and made the gate worse

`--n-items 10 --n-clusters 16 --profile permutation`, 960 calls, 100% JSON ok.

```
real      W1 0.1852   (was 0.2038)
permuted  W1 0.1977
null mean W1 0.1984   (real beat 53% of reassignments, p 0.469)
noise floor  0.0273
seeded gap +0.0125   (was +0.0271)
=> RED
```

Real improved. **The gap halved and p went to chance.** The three items the
diagnosis named were fixed, and four that had been working broke:

| item | real before | real after | Δ | removed same-topic anchor |
|---|---|---|---|---|
| `natsoc` | 0.4998 | **0.0855** | −0.414 | `nataid` |
| `natroad` | 0.4137 | **0.1182** | −0.296 | `nataid` |
| `libcom` | 0.2732 | 0.1626 | −0.111 | `libmslm` |
| `pray` | 0.2275 | 0.2064 | −0.021 | `reborn` |
| `fear` | 0.1166 | 0.1052 | −0.011 | — |
| `news` | 0.1547 | 0.1738 | +0.019 | `divorce` |
| `xmovie` | 0.0605 | 0.1050 | +0.045 | `divorce` |
| `spkcom` | 0.0971 | 0.2214 | +0.124 | `libmslm` |
| `colath` | 0.0946 | 0.2280 | +0.133 | `libmslm` |
| `homosex` | 0.1005 | **0.4457** | +0.345 | `premarsx` |

`natroad` fell 3.5× and `natsoc` 5.8×, confirming the `nataid` diagnosis
outright. `homosex` rose 4.4× on losing `premarsx`.

## The mechanism, measured

Whether removing an anchor helped is **not** explained by how informative it was
about the target's cluster ordering (r = +0.36, weak). It is explained almost
entirely by the **distance between the anchor's level and the target's**:

```
target    anchor     level(tgt) level(anc)  |gap|      dW1   effect
natsoc    nataid        0.247     0.735     0.488   -0.414  removing HELPED
natroad   nataid        0.309     0.735     0.426   -0.296  removing HELPED
libcom    libmslm       0.735     0.498     0.237   -0.111  removing HELPED
pray      reborn        0.401     0.613     0.212   -0.021  removing HELPED
                        --- crossover near 0.20 ---
spkcom    libmslm       0.311     0.498     0.187   +0.124  removing HURT
news      divorce       0.571     0.754     0.183   +0.019  removing HURT
colath    libmslm       0.341     0.498     0.157   +0.133  removing HURT
homosex   premarsx      0.587     0.727     0.140   +0.345  removing HURT
xmovie    divorce       0.713     0.754     0.041   +0.045  removing HURT

corr(|level gap|, change in real W1) = -0.853
```

**The 7B treats a same-topic anchor's level as its prior for the target's
level.** One exemplar near the target's level is worth a great deal; one far
from it is actively harmful. That is a real finding about verbalized
elicitation and it belongs in the writeup regardless of how the gate lands.

It also says why each fix traded one set of items for another: with exactly one
same-topic anchor on the card, every target's level rides on whether that single
item happens to be representative. **And no anchor may be selected by closeness
to the target — that is the held-out quantity.** The only selection-free remedy
is to show *several* same-battery exemplars so no single one sets the level.

## Why that remedy is not available: the split has no anchors to give

| topic | anchors | targets |
|---|---|---|
| **civil_liberties** | **2** | **12** |
| spending_priorities | 6 | 7 |
| institutional_confidence | 8 | 4 |
| other | 11 | 3 |
| wellbeing | 3 | 2 |
| sexual_morality | 2 | 3 |
| religion | 1 | 2 |
| life_and_death | 1 | 2 |
| **crime_and_guns** | **0** | 2 |
| **family_childrearing** | **0** | 2 |
| **politics** | **0** | 1 |

Same-topic anchors actually available to each target after the fold exclusion:
`natroad`/`natsoc` 4, `xmovie`/`news` 7, `homosex` 1, `spkcom`/`colath`/`libcom`
**1**, `pray` **0**, `fear` **0**.

Twelve civil-liberties targets share two civil-liberties anchors, and both are
`*mslm` items — the battery's intolerant extreme (`libmslm` 50.2% "remove"
against 18–27% for the rest of `lib*`). Three topics that have targets have no
anchors at all. Meanwhile `other` holds 11 anchors for 3 targets.

Promotable from the 148-item codebook: **5 spending items** (`natdrug`,
`natcity`, `natfare`, `natheal`, `natsci`) and for civil_liberties only
`colmslm` — a third `*mslm` item, so the same referent and the same outlying
level. Religion, sexual morality, crime, family and politics have **zero** spare
items. The rest of the civil-liberties battery is already targets, and the
near-duplicate rule blocks `spk/col/lib` of the same group from each other's
cards in any case.

**So §7.2's ladder cannot reach this.** Rungs 2 and 3 — richer cards, more
anchors — require anchors that do not exist in the bed. The binding constraint
is the 2.4 split: it stratified *targets* by topic without reserving same-topic
anchors for them.

## The methodological problem this run creates

Three card configurations have now been run and looked at:

| configuration | seeded gap |
|---|---|
| `nataid` present, positional contract (4 items) | +0.0173 |
| `nataid` present, keyed contract | +0.0271 |
| fold contract enforced, keyed | +0.0125 |

Choosing among these by which gap is largest is precisely "a pass mark chosen
after seeing results". **The fold fix stays regardless** — it implements a
clause the checklist specifies and the code already claimed, and reverting a
correct fix because the number got worse would not be defensible. Any further
card change has to be settled on stated principle, written into
`configs/gss_main.yaml` before the run, and accepted as it comes.

## Decision taken: §7.2 rung 1, cards frozen

**Next run is the frontier arm, cards unchanged.** Reasoning: the ladder's own
ordering, it touches no frozen artifact, and it tests a hypothesis that is worth
an answer either way — *is a frontier model also level-anchored to a single
same-topic exemplar?* If it is, the binding constraint is provably the split and
there is evidence to justify amending it. If it is not, the 7B's behaviour is a
small-model artifact and Phase 4 starts.

The card configuration is now **frozen in `configs/gss_main.yaml`** under
`elicitation:` (`anchors_per_card: 12`, `enforce_fold_exclusion: true`,
`anchor_pick: topic_diverse_alpha`), and **`PREREGISTRATION.md`** records all
four dev-model runs, what each change was, why the fold fix is retained even
though it halved the gap, and the three declared readings of the frontier
result — written before the number exists. Neither the gate statistic nor the
noise floor moves on what comes back.

### Two guards added, because the provider switch had teeth

Both were live footguns on exactly this run, where a wasted attempt costs a day
of a rate-capped tier:

- **`--set llm.active_providers=[mistral]` is not JSON.** `_parse_override`
  falls through to the raw string, and the router then iterates it one
  *character* at a time. The config now refuses a string, refuses a provider
  name that is not configured, and prints the working syntax.
- **`mistral-small-latest` is a floating alias**, and the pin check only ever
  covered Ollama. The provider repoints it without notice and `cache.py` keys on
  the model string, so a resumed run would mix two models into one ensemble and
  report the difference as elicitation variance. Now refused for any provider in
  `active_providers`, with the command to list dated tags in the error. Parked
  arms keep their aliases as documentation.

### A third guard, and the bug it caught

Trying to pin the tag surfaced the one that would actually have cost the run.

**Nothing in the codebase loaded `.env`.** `.env.example` says to copy it to
`.env`; checklist 0.7 says to export the four keys; `client.py` read
`os.environ` only. So the keys sat on disk, gitignored exactly as intended, and
were invisible to every run — and an interactive shell does not source `.env`
either, which is why `curl ... -H "Authorization: Bearer $MISTRAL_API_KEY"`
returned nothing at all.

The failure mode is the dangerous kind. `LLMClient` treats a missing key as an
empty bearer token and sends the request anyway; the provider returns 401; the
elicitation loop records it per cell as a `provider` failure — which is *correct*
behaviour for a run of thousands of calls over days that must survive a provider
having a bad afternoon. The result would have been all 960 calls recorded as
failures, looking exactly like Mistral being down.

`popsim/llm/env.py` now loads the nearest `.env` at CLI entry, before anything
reads a key. **The environment always wins**, so an exported key still overrides
the file and CI — which has no `.env` — is unaffected. Four tests.

`popsim doctor` now also (a) reports which keys came from `.env` versus the
shell, and (b) lists the **pinnable tags** for each active hosted provider, using
the same key path a run uses. It loads the config with `validate=False` on
purpose and reports the errors at the end: doctor's job is to diagnose a setup
that is not yet right, and validating first would make the one command that can
list tags refuse to run until the tag was already pinned.

`api.mistral.ai` is not on the sandbox's egress allowlist (403 from the tunnel),
so **the tag still has to be read on your Mac** — but via `popsim doctor`, not
curl.

Nothing is built past this gate.


---

# One 429 cost the whole frontier run — and it was ours

First attempt at §7.2 rung 1, 16 Sep, `runs/gss_main__20260916T094338Z`:

```
WARNING popsim.llm: 429 from mistral; burning it for today and failing over
  ... all ten items: real nan  permuted nan
  JSON ok 0.0%
  => RED: NO DATA — every call failed
```

960 calls recorded as `provider` failures. The ledger afterwards:

```json
"mistral": {
  "calls_today": 0, "tokens_today": 0,
  "exhausted_until_day": "2026-09-17",
  "last_error": "mistral 429: {\"message\":\"Rate limit exceeded\",
                 \"type\":\"rate_limited\",\"code\":\"1300\"}"
}
```

**`calls_today: 0`.** Nothing was consumed. Mistral's free tier allows about one
request per second, the client sent as fast as the network allowed, and the
*first* call drew a per-second throttle — which the router read as the day's
allowance being gone, burned until local midnight, and with `mistral` the only
active provider there was nothing to fail over to.

## Two bugs, both in 0.4a's reading of "rate-capped"

The checklist says the free tiers are *"rate-capped per day, not per dollar"*.
True, and incomplete: **they are rate-capped per second as well.** Everything
downstream followed from taking only the first half.

**1. Every 429 burned the day.** There is one status code for two unrelated
events:

| | provider means | correct response |
|---|---|---|
| throttle | slow down | back off, retry the **same** provider |
| exhaustion | allowance gone | burn until midnight, fail over |

`RateLimited` now means only the first; `QuotaExhausted` (a subclass, so
existing handlers still catch both) means the second. `classify_429()` decides
on evidence — `Retry-After` when the provider sends one, else the response body
against a narrow marker list (`quota`, `credit`, `requests per day`, `billing`,
…), and anything over ten minutes of `Retry-After` is an exhaustion whatever the
body says. **An ambiguous 429 resolves to a throttle**, because reading a
throttle as exhaustion costs a whole day and the reverse costs a few retries.
A throttle honours `Retry-After`, retries the same provider, and never touches
the ledger.

**2. Nothing paced the sends.** Retrying alone would just re-earn the 429, so
`ProviderSpec.min_interval_s` is now enforced before every send, seeded from
config and defaulted per provider (`mistral 1.1s`, `groq 2.2s`, `google 4.1s`,
`ollama 0`). When a provider throttles anyway, its pace **doubles from its
current value** — not from the configured one, which would settle at twice a
number already proven too fast — capped at 10s. The gate sleeps *before* the
send, so a cached call pays nothing (the cache is checked before the router
picks a provider at all), and a resumed run is unaffected.

## The ledger needed a way out

A burn is deliberately irreversible within the day — that is what stops a run
rediscovering a dead provider on every call. But a burn can be *wrong*, as this
one was, and there was no way to undo it:

```
popsim quota --clear mistral
```

Never called automatically. Mistral has been cleared; `calls_today: 0`,
`exhausted_until_day: null`, and `popsim doctor` now reports **all good**.

## What this changes about the run

Gate 0's third clause was *"the provider router fails over cleanly on a
simulated 429"*, and its test asserted the burn. That test encoded the bug, so it
is now two tests — one per kind of 429 — plus the exact Mistral body as a named
regression, the classification table, the pacing arithmetic, and the monotonic
escalation. **129 tests pass.**

Wall time goes up: 480 unique calls at 1.1s pacing is ~9 minutes of deliberate
waiting on top of latency, so expect 20–30 minutes rather than ~22. That is the
price of not spending the attempts on 429s.

Worth keeping in view for Phase 4 and beyond: at 1.1s/call a 37k-call headline
run is ~11 hours of pacing alone on a single provider, which is what the
multi-provider router and the resumable cache exist for. The pacing numbers are
starting points from each provider's published limits, not measurements — if the
run throttles anyway, the escalation will find the real pace and the log records
it.


---

# §7.2 rung 1 — run, and answered: capability is not the constraint

`ministral-14b-2512` (mistral), 10 items x 16 clusters x 1x3, 960 records.
**100% JSON ok, 0% refusals, 0% malformed, 0% positional, 0 throttles.**
Run in ~10 resumable chunks from the session's own shell; `runs/gss_main__20260916T134458Z`.

```
real      W1 0.2074
permuted  W1 0.2085   (+0.5% vs real)
B0a       W1 0.1074
null mean W1 0.2149   (1414 derangements, real beat 54%, p 0.462)
noise floor  0.0273
=> RED. Gap +0.0011.
```

## Side by side with the 7B dev arm

| item | 14B real | 14B gap | p | 7B real | 7B gap | p | B0a |
|---|---|---|---|---|---|---|---|
| `xmovie` | 0.0876 | **+0.1132** | 0.000 | 0.1050 | +0.1029 | 0.000 | 0.1614 |
| `pray` | 0.1118 | **+0.0531** | 0.001 | 0.2064 | +0.0000 | 0.281 | 0.0957 |
| `news` | 0.2015 | **+0.0419** | 0.038 | 0.1738 | +0.0407 | 0.009 | 0.1422 |
| `fear` | 0.1055 | **+0.0559** | 0.033 | 0.1052 | +0.0620 | 0.036 | 0.0978 |
| `natsoc` | 0.0849 | +0.0258 | 0.056 | 0.0855 | +0.0249 | 0.296 | 0.0553 |
| `natroad` | 0.1095 | −0.0124 | 0.822 | 0.1182 | −0.0232 | 0.904 | 0.0974 |
| `homosex` | 0.3534 | −0.0353 | 0.933 | 0.4457 | −0.0421 | 0.961 | 0.1091 |
| `colath` | 0.2851 | −0.0698 | 0.868 | 0.2280 | −0.0076 | 0.839 | 0.0993 |
| `spkcom` | 0.4242 | −0.0714 | 0.925 | 0.2214 | −0.0165 | 0.729 | 0.1207 |
| `libcom` | 0.3105 | −0.0899 | 0.945 | 0.1626 | −0.0157 | 0.632 | 0.0948 |
| **pooled** | **0.2074** | **+0.0011** | 0.462 | 0.1852 | +0.0125 | 0.469 | 0.1074 |

**Twice the parameters and two years of training move nothing.** The 14B is
marginally *worse* on the pooled gap than the 7B. It redistributes which items
work — `pray` goes from no signal to p = 0.001, the civil-liberties items get
substantially worse — but the aggregate is unchanged, and 4 of 10 items are
still significant against the derangement null, which is still not the hedging
signature F2 names.

## The mechanism replicates, and that is the finding

Re-running the level-anchoring test on the 14B, using the nine anchor removals
the fold fix produced:

```
corr(|anchor level - target level|, change in real W1)
    7B   -0.853
    14B  -0.799

corr(7B change, 14B change)                 = +0.915
corr(7B per-item real W1, 14B per-item W1)  = +0.682
```

The two models respond to the *same* card change in the *same* direction with
correlation **+0.915**, and both are level-anchored at r ≈ −0.8. So:

**The 7B's behaviour was never a small-model artifact.** A single same-topic
anchor sets the model's prior for the target's level, at any capability tested,
and when that anchor is unrepresentative the prediction inherits its level. That
is a property of *this elicitation design*, not of the model — which is exactly
what §7.2's rung 1 was built to distinguish, and it answers it in the negative.

## What that licenses, per the pre-registered readings

This is `PREREGISTRATION.md` §4's **second** declared reading, written before the
number existed: red, with a substantial minority of items significant and the
failures concentrated where the one same-topic anchor is far from the target in
level. The pre-registered consequence is that **the binding constraint is the
item split, not the model**, and §7.2's remaining rungs cannot reach it:

- rung 2 (richer cards) and rung 3 (more anchors) both require same-topic
  anchors that **do not exist** — 2 civil-liberties anchors for 12
  civil-liberties targets, and 0 anchors for religion, crime, family, politics;
- the only promotable items in the 148-item codebook are 5 spending items and
  `colmslm`, a third `*mslm` item with the same referent and the same outlying
  level as the two already there.

So the next decision is the one §5 of the pre-registration parked: whether to
amend the anchor set, and under what stated rule. **It is now taken with
evidence rather than in anticipation** — which is what rung 1 was spent to buy.

Nothing is built past this gate. Phase 4 has not started.


---

# Amendment 1 ran, and made the gate worse — an error worth recording

`runs/gss_main__20260916T152836Z`, `ministral-14b-2512`, 960 records, 100% ok.
**real 0.2231, seeded gap −0.0027** (was +0.0011). `natroad` 0.1095 → 0.2056,
`natsoc` 0.0849 → 0.1599.

**Cause: `make_crossfit_plan` derives folds from the anchor list.** Adding five
anchors re-derived every fold, changed which fold each target excludes, and put
`nataid` — the battery's 57.7%-"too much" outlier — back on the spending cards:

```
BEFORE  natroad excludes fold 0 -> natchld natcrime natenrgy natpark   (no nataid)
AFTER   natroad excludes fold 0 -> nataid natchld natcity natcrime natdrug natrace natsci
```

So the run confounds five extra exemplars with the outlier returning, and the
second dominated. Not a property of the rule — an error in applying it.

**It is more evidence for the level-anchoring account, not less.** Three
independent manipulations have now moved `natroad` and `natsoc` in the direction
r ≈ −0.8 predicts.

## The finding this produces

**Which same-topic exemplar reaches a card is decided by an arbitrary fold seed.**
`make_crossfit_plan` shuffles anchors into folds at `seed=17`, picks each
target's excluded fold as the one densest in same-topic anchors, and
`_pick_anchors` breaks ties alphabetically. Nothing in that chain knows whether
the survivors are representative, and the measured consequence is a 0.10–0.41 W1
swing on one item. That is a methodological criticism of stat-card conditioning
in its own right.

## Iteration stops here

Four card configurations run and inspected: +0.0173, +0.0271, +0.0125/+0.0011,
−0.0027. Continuing to adjust the card and re-read the gap is choosing a pass
mark after seeing results. **No further card change in pursuit of a green gate.**
`PREREGISTRATION.md` §7a records all four and the three remaining options, all
of which are supervisor decisions about what the thesis claims.
