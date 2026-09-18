# B17 — Location-variable audit

**1 Sep 2026.** Every file under `data/` checked for respondent geography. Every number
below was measured directly from the files on disk, not read off a codebook.

The spec (`B17_architecture_spec.md` §demo struct, §clustering) requires two geographic
fields per respondent: **`region`** (US census division / India state) and **`urban`**
(bool). Level-1 of the cluster tree is literally `region × urban`, so a dataset that is
missing either one cannot be partitioned as specified.

---

## Bottom line

**Region is fine almost everywhere. Urban/rural is the field that breaks.**

Five things will bite you if you build the adapters against the obvious variable names:

1. **GSS 2024 has no census division and no urban/rural belt code.** In `gss7224_r3a.dta`
   the 9-division variable `region_7222` and the belt code `srcbelt` are **0 % populated
   for all 3,309 respondents in 2024**. Only 4-category `region` and `xnorcsiz` survive.
   This hits exactly the 2016–2024 pooling strategy `B17_data_review.md` §Part 1
   recommends.
2. **`region` in the GSS cumulative file is 4 categories, not 9.** The 9-division variable
   was renamed `region_7222`. If you code `region` expecting divisions you silently get a
   4-way split and ~2.2× coarser level-1 nodes. (Verified: the 4-cat variable is a clean
   rollup of the 9 — crosstab is perfectly block-diagonal, 0 leakage.)
3. **The ANES–GSS joint file (n = 1,164) has no geography at all** — no state, no region.
   Its case IDs do **not** join to the ANES time-series file (0 of 1,164 overlap). Recovery
   route below; it works and gives 100 %.
4. **`V243002` (ANES 2024 sample FIPS state) is an empty string for all 2,839 rows.** Use
   `V243001` (postal abbreviation) instead. A pipeline keyed on the FIPS field gets a
   silent, total loss of 2024 state.
5. **WVS India wave 7 (2023) has 0 % sub-national region**, and India has no urban/rural
   for 1995–2012. The India panel story in WVS is geographically broken; **GFS is not** —
   use GFS for the India bed.

---

## 1. Per-dataset results

Coverage = share of respondents with a usable, non-missing, non-negative-coded value.

### US — respondent-level

| File | n | Region | Urban/rural | Verdict |
|---|---|---|---|---|
| `gss/data/GSS_stata/gss7224_r3a.dta` | 75,699 | `region` (4) **100 %**; `region_7222` (9) **95.6 %** | `xnorcsiz` 100 %, `srcbelt` 95.6 %, `size` 95.6 % | OK **except 2024** |
| `gss/data/2022/GSS2022.dta` | 4,149 | 4-cat 100 %, 9-cat 100 % | `xnorcsiz` 99.7 %, `srcbelt` 98.3 % | Good |
| `gss/data/2024/GSS2024.dta` | 3,986 | 4-cat 100 %, **no 9-cat variable** | `xnorcsiz` 100 %, **no `srcbelt`** | Degraded |
| `gss/.../gss2020panel_r1a.dta` | 5,215 | `region_1a/1b` (9) **100 %** combined; `region_2` **100 % of the 1,823 wave-2 respondents** | baseline `srcbelt_1a/1b` 100 %; **`srcbelt_2` and `xnorcsiz_2` 0 %** | Region good, wave-2 urban absent |
| `anes/anes_timeseries_2020_*` | 8,280 | `V203000` state FIPS **100 %** (51 incl. DC), `V203003` region **100 %** | `V202355` self-report **89.2 %** (10.8 % = no post-election interview) | Good |
| `anes/anes_mergedfile_...panel.sav/.csv` | 2,839 | 2016 `V163001a` 100 %; 2020 `V203000` 100 %; 2024 `V243001` 100 % of the 2,171 who took 2024 (**`V243002` empty**) | `V202355` 93.5 %; `V242341` (2024) 95.8 % of 2024 wave | Good, with the V243002 trap |
| `anes/..._gss_stata_20220408.dta` (joint) | 1,164 | **none** | `V202355` 96.8 % | **Broken as shipped** |
| `anes/..._gss_bridge_20220408.dta` | 9,444 | none (weights only) | — | Link table, not data |
| `anes/anes_specialstudy_...socialmedia.csv` | 5,750 | `profile_state` 100 % (51), `profile_region9` **100 %**, `profile_region4` 100 % | `profile_metro` **100 %** | **Best in the set** |
| `ces/cumulative_2006-2025.feather/.dta` | 718,955 | `st`/`state` **100 %** all 20 years; `cd` 100 %; `county_fips` 99.94 % | **none** — no urban/rural variable exists | Region excellent, urban missing |
| `benchmarks/opinionqa/.../responses.csv` (15 waves) | 2,537–10,221 | `CREGION` (4) **100 % in all 15 waves**; 9-division `F_CDIVISION` only in W82, W92 | `F_METRO` 100 % in 9 of 15 waves; **absent in W26, W27, W29, W34, W36** | Mixed |
| `benchmarks/subpop/.../opinionqa.csv` | 30,324 rows | `CREGION` used as a subpopulation attribute (4 levels, 1,992 rows) | not a dimension | Aggregate benchmark, fine |

### Cross-national — respondent-level

| File | n | Region | Urban/rural | Verdict |
|---|---|---|---|---|
| `gfs/gfs_all_countries_wave2...sav/.csv` | 207,919 | `REGION1_Y1` **100 % in all 23 countries** (state-level: 51 for US, 23 states for India); `REGION2_Y1` 95.5 % | `URBAN_RURAL_Y1` **99.8 %** (100 % for India and US) | **Clean. Use this for India.** |
| `wvs/Trends_VS_1981_2022_Stata_v4_1.dta` | 442,473 | `X048WVS` 90.4 %; `X048ISO` 71.2 % | `X049A` size-of-town 73.5 %; `X050C` habitat **30.2 %** | **Patchy — see §2** |
| `benchmarks/.../global_opinions.csv` | 2,556 questions | country-level aggregates only (138 keys, some flagged "Non-national sample") | n/a | No individuals; not a location problem |

### No respondent-level data on disk (documentation only)

`nfhs5/` (3 PDF reports), `census_india/` (projections PDF), `mospi/hces_docs/`
(layout + state-code XLSX, no unit records), `pew/` (codebook PDF),
`voter_nationscape/` (guide PDF). Location cannot be assessed — the microdata is not here.
`mospi/hces_docs/tabulation_state_code.xlsx` and `Layout_HCES_2023-24.xlsx` do confirm
HCES unit data carries state and sector (rural/urban) codes **when you obtain it**.

---

## 2. WVS geography, in detail

`X050C` (urban/rural habitat) is missing for 69.8 % of all WVS rows. Worst affected
countries by sub-national region coverage (n > 1,000): Tanzania 0 %, Kenya 0 %,
Croatia 0 %, Singapore 36.6 %, UK 45.0 %, Poland 46.8 %, Norway 47.6 %, Hungary 53.1 %.

India, by wave:

| Wave | n | Region | Size of town | Urban/rural |
|---|---|---|---|---|
| 1990 | 2,500 | 100 % | 90.0 % | 100 % |
| 1995 | 2,040 | 100 % | 100 % | **0 %** |
| 2001 | 2,002 | 100 % | 99.8 % | **0 %** |
| 2006 | 2,001 | 100 % | 99.6 % | **0 %** |
| 2012 | 4,078 | 100 % | 96.6 % | **0 %** |
| **2023** | 1,692 | **0 %** | 100 % | 100 % |

USA: region 96.4–100 % across all five waves, but `X049A` is **0 % in 2011** and
urban/rural exists **only in 2017**.

So there is no single WVS wave pair for India with both region and urban/rural present.
Any India longitudinal work on WVS has to fall back to `X049CS` size-of-town as a proxy
for `urban`, and 2023 has no `region` at all.

---

## 3. Fixes

**F1 — GSS 2024 division.** Not recoverable from the file (NORC suppressed it in the
7224 release). Options: (a) run level-1 at 4 regions for the pooled 2016–2024 bed and
accept the coarser split; (b) build the tree on 2016–2022 with 9 divisions and treat 2024
as an out-of-sample year at 4 regions. (b) is cleaner and keeps §M8's horizon curve intact.

**F2 — GSS `region` naming.** In the adapter, read `region_7222` and alias it to `region`;
fall back to 4-cat `region` only where `region_7222` is null (2024). Log which one was
used per row — otherwise a 2024 leaf is silently 2× the population of a 2022 leaf.

**F3 — ANES–GSS joint file.** Its `YEARID` is `year*10000 + gss_id`. Split it and join to
`gss2020panel_r1a.dta` on the baseline cohort id (`id_1a` for YEARID/10000 == 2016,
`id_1b` for 2018). **Verified: 1,164 of 1,164 match, giving 100 % 9-division region and
100 % baseline `srcbelt`.** 529 respondents are 2016-cohort, 635 are 2018-cohort.

**F4 — ANES 2024 state.** Read `V243001` (postal), map to FIPS yourself. Add an assertion
that the state column has > 1 distinct value — that alone would have caught `V243002`.

**F5 — CES urban/rural.** Derive it from `county_fips` (99.94 % populated, 3,101 counties)
against an NCHS urban–rural classification or a county-level RUCC crosswalk. This is a
one-off join and turns CES into a fully spec-compliant bed.

**F6 — India.** Use GFS as the India bed, not WVS: 12,765 respondents, 100 % state and
100 % urban/rural. Caveat: only **23 of 36 states/UTs** are represented, so state-level
leaves will have holes — the smaller states and UTs are simply not sampled. `REGION2_Y1`
(North/East/West/South/Central, 5 levels, 100 %) is the safe level-1 for India.

**F7 — OpinionQA waves without metro.** W26, W27, W29, W34, W36 have no urban/rural at
all. Either restrict level-1 to region for those waves or exclude them from any
`region × urban` comparison; do not let them fall into an "unknown urban" bucket that
gets merged across waves.

---

## 4. Two consistency checks run

- **GSS**: crosstab of `region_7222` (9) against `region` (4) is perfectly block-diagonal
  — the 4-category variable is an exact rollup, no misassignment, 0 rows with 9-cat
  present and 4-cat missing.
- **ANES 2020**: derived census region from `V203000` state FIPS agrees with `V203003` for
  **8,280 of 8,280** rows. So the 9 divisions are derivable at 100 % for ANES even though
  no division variable ships.
