# Session handoff — paste this whole file into a new session

`STATUS.md` is the living record and is far more detailed. `PREREGISTRATION.md`
records every card/model configuration that has been run and why, and is the
reason the Gate 3 numbers are trustworthy. Read both. Everything below is true
as of **16 Sep 2026**.

---

## Where the project stands

**Gates 0, 1, 2 GREEN. Gate 3 RED — and the red is understood, reproducible and
model-independent.** 131 tests pass, lint clean. Nothing is built past Gate 3.

| gate | state |
|---|---|
| 0 config snapshot, call-count guard, 429 failover | ✅ |
| 1 Layer 0 — 142/142 published toplines within 0.5 pp, worst 0.369 pp | ✅ |
| 2 Layer 1 — K = 56, 140/148 items clear SNR 1.5, split frozen | ✅ |
| **3 Layer 2 — the permutation test** | ❌ **RED, twice, on two model sizes** |

### The Gate 3 result, in one table

| | 7B (`qwen2.5-ctx8k:7b-instruct-8192`) | 14B (`ministral-14b-2512`) |
|---|---|---|
| real W1 | 0.1852 | 0.2074 |
| permuted W1 | 0.1977 | 0.2085 |
| **gap** (gate needs > 0.0273) | **+0.0125** | **+0.0011** |
| null mean over 1414 derangements | 0.1984 | 0.2149 |
| items p < 0.05 of 10 | 3 | 4 |
| B0a baseline / noise floor | 0.1074 / 0.0273 | same |
| JSON ok / refusals / malformed | 100% / 0% / 0% | 100% / 0% / 0% |

**§7.2 rung 1 is spent and answered: capability is not the constraint.** Twice
the parameters and two years of training changed nothing (the 14B is marginally
worse). It is not F2 either — `xmovie` alone has a gap of +0.1132 at p = 0.000,
and the `hedger` fake scores exactly 0.0000.

### Why it is red — the finding to carry into the writeup

**A single same-topic anchor sets the model's prior for the target's LEVEL.**
Measured across the nine anchor removals the 3.4 fold fix produced:

```
corr(|anchor level - target level|, change in real W1)
    7B   -0.853
    14B  -0.799
corr(7B change, 14B change)  = +0.915   <- the two arms fail on the same items
```

When that one anchor is unrepresentative of its battery, the prediction inherits
its level and comes out anti-correlated with truth. The clearest case: `nataid`
(foreign aid, 57.7% "too much" against a battery median near 10%) was the only
spending anchor on every spending card, and the 7B tracked it at r = +0.845 on
`natroad` and +0.896 on `natsoc` while correlating *negatively* with the truth.
Removing it (by enforcing 3.4) moved `natsoc` 0.4998 → 0.0855 and `natroad`
0.4137 → 0.1182.

**The selection-free remedy is several same-battery exemplars, so no single one
sets the level. The frozen split does not contain them:**

| topic | anchors | targets |
|---|---|---|
| **civil_liberties** | **2** (both `*mslm`, the battery's intolerant extreme) | **12** |
| spending_priorities | 6 | 7 |
| religion / life_and_death | 1 each | 2 each |
| **crime_and_guns / family_childrearing / politics** | **0** | 2 / 2 / 1 |
| other | 11 | 3 |
| institutional_confidence | 8 | 4 |

Promotable from the 148-item codebook: 5 spending items, and for civil liberties
only `colmslm` — a third `*mslm`, same referent, same outlying level. Religion,
sexual morality, crime, family and politics have **zero** spare items.

**So §7.2's rungs 2 and 3 (richer cards, more anchors) cannot reach this** — they
require anchors that do not exist in the bed. The binding constraint is the 2.4
split, which stratified *targets* by topic without reserving same-topic anchors
for them.

**Never select an anchor by closeness to the target's level.** That is the
held-out quantity. It is a tempting and completely invalid fix.

---

## STOP: four card configurations have been run. Do not run a fifth.

Gaps, in order: **+0.0173 → +0.0271 → +0.0125 (7B) / +0.0011 (14B) → −0.0027**
(amendment 1). The gate needs > 0.0273 and has never been close.

Amendment 1 (5 extra spending anchors, `PREREGISTRATION.md` §7) made it *worse*,
because `make_crossfit_plan` derives folds from the anchor list — adding anchors
re-derived every fold and put `nataid` back on the spending cards. That was an
error in applying the rule, recorded in §7a, and it is further evidence for the
level-anchoring account rather than against it.

**Adjusting the card again and re-reading the gap is choosing a pass mark after
seeing results.** The remaining options are not tuning questions and all three
are supervisor decisions about what the thesis claims:

1. Reopen the 2.4 split against the ≥40-target claim in START_HERE.
2. Accept the limitation and carry it into the writeup; run Phase 4 knowing raw
   elicitation is level-biased, which is what §4.4 exists to repair. **This
   overrides Gaurav's own "do not build past a red gate" rule** and needs his
   explicit say-so.
3. Make the fold/anchor selection deterministic and **report the lottery** as a
   measured property instead of trying to win it.

The evidence for that conversation is complete. Do not spend more calls on it.

## The original framing of that decision, for reference

`item_split.frozen.yaml` is frozen (2.4: "must never be regenerated"), and the
split is structurally unable to condition its own targets. Options, none taken:

1. **Amend the anchor set additively** under a rule stated first (e.g. "every
   topic with ≥1 target reserves ≥3 anchors where the pool allows"). Promote the
   5 unassigned spending items. Targets stay at 40, so the ≥40 requirement holds
   and no target moves. **Fixes spending only** — civil liberties, religion and
   sexual morality cannot be fixed this way.
2. **Reopen the split** with same-topic anchor reservation as a constraint. The
   only option that fixes the 12 civil-liberties targets — but it moves items
   from target to anchor, dropping below 40 targets, which collides with the ≥40
   claim in START_HERE. Breaks the freeze.
3. **Keep the freeze**, report the anchor/target imbalance as *the* measured
   limitation, and take Phase 4 forward knowing raw elicitation is level-biased —
   which is what the calibration layer exists to repair.

Option 3 has a real argument: the level bias is exactly what §4.4's variance
restoration and isotonic recalibration are for, and Gate 3's job was to detect
F2 (no conditioning), which the per-item results say is *not* what is happening
on 4 of 10 items. But **the checklist says do not build past a red gate**, so
this cannot be assumed. Get his answer before touching Phase 4.

---

## Environment — read before running anything

- **Repo** `~/Downloads/fyp/popsim`. Data at `~/Downloads/fyp/data` (outside it).
- **Rebuild the venv** (does not persist):
  `uv venv --python 3.11 $HOME/venv-b17 && uv pip install --python $HOME/venv-b17/bin/python -e ".[dev,pdf]"`
  then run everything with `$HOME/venv-b17/bin/python`.
- **`popsim doctor --live`** makes one real call per active provider and reports
  latency, or the full 429 with its rate-limit headers. Run it before any batch.
- **Mistral capacity is allocated PER MODEL**, not per workspace as the docs
  imply. Measured from `x-ratelimit-limit-req-minute`:
  `mistral-small-2603` **0**, `mistral-medium-2604` **0**,
  `magistral-medium-latest` **0**, `ministral-14b-2512` **30**,
  `ministral-8b-2512` **188**, `ministral-3b-2512` **750**.
  Anything showing 0 can never run, whatever the pacing.
- **Groq** returns `HTTP 403 / Cloudflare 1010` from the session's shell (edge
  block on the proxied path, not auth). It may work from Gaurav's own terminal.
- **Google** is reachable and the key works, but the config's `gemini-2.0-flash`
  is **retired** — the API says use a current tag. 41 models support
  `generateContent`, up to `gemini-3.8-flash`; all have 1M-token context.
  **This is still unfixed in the config.**
- **Ollama is on Gaurav's Mac at `localhost:11434`** and is unreachable from the
  session's shell (the Linux VM's localhost is not his Mac's). The dev arm must
  be run by him. `popsim doctor` confirms it up at num_ctx 8192, max 32768.
- **The session's own shell freezes between tool calls**, so background jobs do
  not progress. A long run must be done in ~165-second chunks; the cache makes
  each chunk resume for free. The 14B Gate 3 run took ~10 chunks.
- `.env` is loaded automatically now (`popsim/llm/env.py`); the shell always wins.

## Decisions already made — do not re-litigate

- Layer 0 gates **unweighted** marginals against the codebook, plus a separate
  weighted check against ACS 2024.
- GSS 2021 stays in the bed, interview mode recorded per respondent.
- Pooled waves use `wtssps` rescaled to equal per-wave mass.
- `natroad` stays the demo item; mode travels as a stat-card marginal.
- Split-ballot `*Y` twins: **base variable only** (§1.4a).
- `codebooks/item_split.frozen.yaml` is **FROZEN**. Never regenerate it.
- **Gate 3 turns on the seeded real-vs-permuted gap alone**, not on
  permuted ≥ B0a. The derangement null is reported beside it as a diagnostic.
- The response contract is a **JSON object keyed by option label**, never a
  positional array. 19 codebook items (all `nat*` plus `polviews`) have wording
  that enumerates their options in a different order from `codes`; a positional
  array let the model answer in the wording's order and be scored in the wrong
  bins. A positional reply is now the recorded failure `positional_response`.
- **3.4's fold exclusion is enforced** — `anchors_for()` excludes the target's
  own fold, and `assert_card_contract` checks it. Retained even though it halved
  the Gate 3 gap, because reverting a specified clause to recover a better
  number is not defensible.
- `dwelown` is excluded from Gate 3 item selection as a household fact
  (`pool.POST_FREEZE_DOCTRINE_FLAGS`); the frozen split is untouched.
  `xmovie` and `news` stay — §5.1 says "attitudinal/**behavioral**".
- `ministral-3b-2512` works but is *smaller* than the dev arm, so it cannot
  answer rung 1. It is a useful downward point for §7.3a's 3B/7B/14B curve.

## How Gaurav wants you to work

- Ask before any design decision the checklist leaves open.
- Do not build past a red gate.
- Config values come from `B17_build_checklist.md`, never the spec.
- Free rate-capped tiers, not a dollar budget — the client stays resumable and
  checkpoints every call.
- Tell him explicitly when something needs doing on his machine.

## Open, and NOT blocked by the red gate

- **1.6** GFS India adapter (`COUNTRY = 6`, weight `ANNUAL_WEIGHT_R2`, `_Y1`).
- **1.7** OpinionQA adapter — 15 ATP waves, individual-level.
- Spot-check `codebooks/gss2022_published_toplines.yaml` — every entry is
  `verified: false`.
- Decide the 6 dropped tolerance items (in the pooled bed, no 2022 data).
- Re-check the §1.2 item counts against base-only n.
- Fix the retired `gemini-2.0-flash` tag in `configs/gss_main.yaml`.
- **`git init` — the repo has NO commits.** 131 tests and four design documents
  exist only as working-tree files.
