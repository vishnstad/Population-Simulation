# B17 — start here

**Read this before anything else in this folder.** 15 Sep 2026.

---

## What the project is, in three sentences

Split the US adult population into ~56 demographic groups (age × education × sex) and give
each one a **stat card**: its population share, its demographic marginals, and its real
answers to ~12 survey questions we have data for. Show that card to an LLM and ask it to
estimate, as percentages, how that group would answer a **different** survey question it
was never shown — then pass the raw output through a **calibration layer** fitted on
questions where the true answer is known, to repair the LLM's systematic biases (above all
its collapsed variance). Score the prediction against the real GSS answers for those
groups, which were held out.

**Input:** stat card + exact question wording + answer options.
**Output:** a probability distribution over the options, per group, plus a population
rollup raked to census margins.
**Verified by:** Wasserstein-1 distance to the true held-out histogram, against the
baseline of "every group answers like the national average."

---

## ⚠️ Precedence — read this before trusting any number

The documents were written at different times and **they contradict each other**. Later
measurement beats earlier design. In a conflict:

```
B17_build_checklist.md  >  B17_feasibility_verdict.md  >  B17_data_review.md
                                                       >  B17_architecture_spec.md
```

The spec is the authority on **architecture** — module boundaries, schemas, interfaces,
the falsifiable-claim structure. It is **out of date on every tuning parameter**, because
its numbers were reasoned rather than measured, and the measurements came later.

### The specific traps

| The spec says | Reality (measured) | Why |
|---|---|---|
| `K_target: 150` | **K ≈ 56** | GSS supports ~53 leaves at the median item, not 150 |
| `min_cell: 40` | **60** | At 40 the truth is noisier than the effect |
| Level-1 = `region × urban` | **no geography in the partition** | Adding it drops usable items from 74 to 35. Geography goes in the stat card as a *marginal* |
| Bed = GSS 2022, or pooled 2016–2024 | **pooled 2010–2022** | 74 items clear the noise gate vs 59 |
| Use GSS 2024 as preferred bed | **GSS 2024 is excluded** | `region_7222` is 0 % populated → complete-demo n = 0 |
| 60 % anchors / 40 % targets | **34 anchors / 40 targets** | 60/40 leaves only 29 targets; the claim needs ≥ 40 |
| M2 greedy sibling merge | **replace it** | Merge criterion is undefined on 1.6-person cells |
| M8 on 2-year panels | **repeated cross-sections, horizon as a variable** | 2-year signal is 1.01× the noise floor — no power |
| India = NFHS-5 + Pew India | **India = GFS** | No Indian microdata is on disk; all of it is gated |
| `budget_cap_usd: 1200` | **$0** | Runs on free API tiers; the constraint is calls/day |
| CES = second item bank | **CES has ~8–10 attitudinal items** | It is a vote-choice file, not an item bank |

---

## Read order

**For writing code — these four are at the root, and they are all you need:**

1. **`START_HERE.md`** — this file
2. **`B17_build_checklist.md`** — the build. Phases 0–7, a gate between each, the
   pre-registered pass marks, item selection, the free-tier plan. **The operative doc**
3. **`B17_architecture_spec.md`** — module definitions, Parquet schemas, function
   signatures, the M1–M10 pipeline. Architecture only; **ignore its parameters**
4. **`B17_location_audit.md`** — variable-level traps for the adapters

**`reference/`** — still accurate, not needed to write code. Go here when you want the
*evidence* behind a decision:

- `B17_feasibility_verdict.md` — why K = 56 and not 150; the headroom measurement that
  justifies building this at all; the full cell-count tables
- `B17_data_review.md` — the noise-floor and horizon analysis behind the M8 redesign
- `DOWNLOADED.md` — long-form inventory of every file and how it was verified
- `LOGIN_REQUIRED.md` · `DOWNLOAD_LINKS.md` — the gated India sources, for that phase

**`archive/`** — superseded. **Do not build from these.**

`fetch_data.sh` at the root re-fetches the ungated data.

---

## What is on disk

**US — complete. Nothing left to download.**

| | |
|---|---|
| `data/gss/` | cumulative 1972–2024 (75,699 × 6,943, 35 waves) · 2022 · 2024 · 2016–20 panel · codebook PDFs |
| `data/anes/` | 2020 Time Series · 2016-20-24 panel · social media study |
| `data/ces/` | cumulative 2006–2025, 718,955 respondents |
| `data/benchmarks/` | OpinionQA — 15 Pew ATP waves, individual-level · SubPOP repo + paper |
| `data/acs/` | **`acs2024_margins.parquet`** — raking margins, 60 cells, 267.2M adults ✅ |
| `data/gfs/` | Global Flourishing Study — India n = 12,765, the working India bed |

`acs2024_margins.parquet` is on the **GSS `degree` scale** (0 lt-HS, 1 HS, 2 associate,
3 bachelor, 4 graduate), validated against GSS 2016–22 to within 2.9 pp. **If the GSS
adapter changes its education scale, regenerate the margins** or raking is meaningless.

### File facts the adapters need

Verified by reading the files. Getting any of these wrong costs an afternoon.

| File | Fact |
|---|---|
| `gss/data/GSS_stata/gss7224_r3a.dta` | 598 MB, 75,699 × 6,943, 35 waves. **Needs `encoding='latin1'`** — the default UTF-8 decode throws on its value labels. Read with `usecols=` or `metadataonly=True`, **never whole** |
| GSS weights | `wtssps`, falling back to `wtssall`. No `wtsscomp` in this release |
| GSS format | **No CSV exists in any GSS release** — Stata/SPSS/SAS only. `pyreadstat` reads `.dta`/`.sav` with variable *and* value labels |
| GSS ballot design | Per-item answered n runs 1,117–3,526 in 2022. **Reason about `min_cell` from per-item answered n, never sample n** |
| GSS 2024 | `region_7222` and `srcbelt` are **0 % populated**. Excluded from the bed |
| `gfs/...wave2_with_midyear.sav` | India = **`COUNTRY` code 6**, n = 12,765. Weight `ANNUAL_WEIGHT_R2`. Items suffixed `_Y1`. `REGION2_Y1` (5 zones) is the safe India level-1; `AGE_Y1` can contain dates — coerce |
| `anes/...2016-2020-2024panel` | **`V243002` (2024 FIPS) is an empty string for all rows** — use `V243001` (postal) and map it yourself. Assert >1 distinct state |
| `ces/cumulative_2006-2025` | Take `.dta` for value labels, `.feather` for speed. 718,955 × 109, but only ~8–10 attitudinal items |
| `benchmarks/opinionqa/data/human_resp/` | 15 ATP waves, individual-level. Demographics are the `F_*` columns. `F_METRO` absent in W26, W27, W29, W34, W36 |
| `wvs/Trends_VS_1981_2022...dta` | 500 MB, 442,473 × 732. India = `S003` 356. **Geography is broken for India** — no wave has both region and urban/rural |

**India** — gated. NFHS-5, Pew India, IHDS-II, MoSPI, Census C-14 all need logins. Build
US-first; GFS is the India bed that works today.

---

## Where to start

**Phase 0**, then **Phase 1 up to the Layer-0 topline gate**. That gate — harmonized GSS
marginals matching the published codebook toplines within 0.5 pp — is the one that catches
recode and weight bugs, and those are the bugs that silently invalidate everything
downstream. Nothing should be built past it until it is green.

Language Python 3.11+, storage Parquet, stats numpy/scipy/scikit-learn only, no deep
learning in Stages A/B.
