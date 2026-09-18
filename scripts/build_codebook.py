#!/usr/bin/env python
"""Build the harmonized GSS item codebook (checklist 1.4 / 1.4a / 1.4b).

Sources, each for what it is authoritative about:

* exact question wording      <- GSS 2022 codebook PDF (transcribed, never paraphrased)
* option labels and codes     <- gss7224_r3a.dta value-label map
* per-wave answered n         <- the .dta itself, counted
* NORC mode-sensitivity class <- the codebook PDF's three published lists
* role                        <- checklist 1.4b and §1.4 (sanity / excluded)

Usage:  python scripts/build_codebook.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from popsim.data.adapters.gss import read_gss_columns
from popsim.data.codebook import build_codebook

DTA = REPO / ".." / "data" / "gss" / "data" / "GSS_stata" / "gss7224_r3a.dta"
PDF = REPO / ".." / "data" / "gss" / "codebooks" / "GSS_2022_Codebook.pdf"
TOPLINES = REPO / "codebooks" / "gss2022_published_toplines.yaml"
OUT = REPO / "codebooks" / "gss_items.yaml"

BED_WAVES = [2010, 2012, 2014, 2016, 2018, 2021, 2022]

# §1.5, the leakage-resistant target set: 32 items outside the famous battery.
LEAKAGE_RESISTANT = ["natroad", "finrela", "satfin", "spkcom", "colath", "natarms", "natsoc", "libcom", "pornlaw", "libhomo", "consci", "spkhomo", "colhomo", "spkmil", "conmedic", "conlegis", "suicide1", "natspac", "natenvir", "sexeduc", "spanking", "postlife", "colcom", "libath", "natmass", "libmil", "spkath", "conlabor", "spkrac", "nateduc", "fair", "life"]

# §1.4, the tolerance battery: identical structure, a 3x5 grid. The Muslim
# clergyman items (spk/col/lib mslm) are a later addition to the same battery
# and are carried alongside it.
TOLERANCE_BATTERY = ["spkath", "colath", "libath", "spkrac", "colrac", "librac", "spkcom", "colcom", "libcom", "spkmil", "colmil", "libmil", "spkhomo", "colhomo", "libhomo", "spkmslm", "colmslm", "libmslm"]

POOL_FILE = REPO / "codebooks" / "candidate_pool.yaml"


def load_pool() -> tuple[list[str], dict[str, dict[int, int]]]:
    """The measured candidate pool (scripts/build_pool.py), not a hand list.

    Layer 1 has to search the whole repeated item bank to find which items clear
    SNR >= 1.5 — the checklist's "74 available" is a claim about that bank, not
    about the 38 items it happens to discuss by name.
    """
    import yaml as _yaml
    doc = _yaml.safe_load(POOL_FILE.read_text())
    counts = {k: {int(w): int(n) for w, n in v.items()}
              for k, v in doc["answered_n_by_wave"].items()}
    return list(doc["pool"]), counts


def norc_mode_sensitivity(pdf_path: Path) -> dict[str, str]:
    """Scrape NORC's three published mode-sensitivity lists.

    GSS 2022 is multi-mode (2021 is 87% web, 2022 is 46% web, 2010-2018 are
    essentially all in-person), so mode is very nearly collinear with wave in
    this bed. NORC publishes which variables shift between face-to-face and web.
    Several of them are in this project's pool — NATROAD, the intended demo
    item, is on the "likely" list — so the classification travels with the item.
    """
    import pdfplumber

    PAGE_FURNITURE_RE = re.compile(
        r"^CODEBOOK\s*\||^General Social Survey Codebook|^Page\s+\d+|^\d+\s*$"
    )
    headings = {
        "Likely mode sensitive": "likely",
        "Requires further investigation": "investigate",
        "Less likely to be mode sensitive": "less_likely",
    }
    # The three lists run on past the page they start on — "Less likely to be
    # mode sensitive" alone is a few hundred variable names — so once a heading
    # is seen, keep reading the following pages while they still look like
    # columns of variable names. Stopping at the heading page silently
    # classifies most of the pool as "unknown".
    text = ""
    with pdfplumber.open(str(pdf_path)) as pdf:
        capturing = False
        for page in pdf.pages[:120]:
            t = page.extract_text() or ""
            if any(h in t for h in headings):
                capturing = True
            elif capturing:
                body = [ln.strip() for ln in t.splitlines() if ln.strip()]
                name_rows = sum(
                    1 for ln in body if re.fullmatch(r"(?:[A-Z][A-Z0-9_]{2,}\s*)+", ln)
                )
                if not body or name_rows < max(3, len(body) // 3):
                    capturing = False
            if capturing:
                text += t + "\n"
    out: dict[str, str] = {}
    current: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        matched = next((v for k, v in headings.items() if stripped.startswith(k)), None)
        if matched:
            current = matched
            continue
        if not current or not stripped:
            continue
        # Running page furniture ("CODEBOOK | 53", the footer line) sits in the
        # middle of a list that spans pages. Skipping it rather than treating it
        # as the end of the section is what keeps the second page of "Less
        # likely to be mode sensitive" — which is where SPKATH, SATFIN, SEXEDUC
        # and SPANKING live — from being dropped.
        if PAGE_FURNITURE_RE.search(stripped):
            continue
        if re.fullmatch(r"(?:[A-Z][A-Z0-9_]{2,}\s*)+", stripped):
            for tok in stripped.split():
                out[tok.lower()] = current
            continue
        # A line of ordinary prose is the end of the list.
        if re.search(r"[a-z]{3,}", stripped):
            current = None
    return out


def answered_n_by_wave(dta: Path, items: list[str], waves: list[int]) -> dict[str, dict[int, int]]:
    frame, _ = read_gss_columns(dta, ["year", *items])
    frame = frame[frame["year"].isin(waves)]
    out: dict[str, dict[int, int]] = {}
    for it in items:
        answered = pd.to_numeric(frame[it], errors="coerce").notna()
        counts = answered.groupby(frame["year"]).sum()
        out[it] = {int(k): int(v) for k, v in counts.items() if v > 0}
    return out


def observed_codes(dta: Path, items: list[str], waves: list[int],
                   chunk: int = 110) -> dict[str, set[int]]:
    """Which numeric codes real respondents actually chose, in the bed.

    Needed because "not in the published 2022 table" and "nobody chose it" are
    different statements, and conflating them drops respondents.
    """
    out: dict[str, set[int]] = {}
    for k in range(0, len(items), chunk):
        sub = items[k:k + chunk]
        frame, _ = read_gss_columns(dta, ["year", *sub])
        frame = frame[frame["year"].isin(waves)]
        for c in sub:
            vals = pd.to_numeric(frame[c], errors="coerce").dropna()
            out[c] = {int(v) for v in vals.unique() if v >= 0}
        del frame
    return out


def main() -> int:
    POOL, pool_counts = load_pool()
    print(f"pool: {len(POOL)} items (from {POOL_FILE.name})")
    modes = norc_mode_sensitivity(PDF)
    print(f"NORC mode-sensitivity classes scraped for {len(modes)} variables")

    counts = pool_counts

    # The 2021 split-ballot wording experiment. Base-only is the decision; this
    # measures what that costs so the Layer 1 noise floor is read with it in view.
    from popsim.data.adapters.gss import gss_metadata
    known = set(gss_metadata(DTA).column_names)
    twin_names = [i + "y" for i in POOL if i + "y" in known]
    twin_raw = answered_n_by_wave(DTA, twin_names, BED_WAVES) if twin_names else {}
    twin_counts = {n[:-1]: v for n, v in twin_raw.items() if v}
    cb = build_codebook(
        dta_path=DTA, toplines_path=TOPLINES, items=POOL,
        answered_n_by_wave=counts, twin_answered_n_by_wave=twin_counts,
        observed_codes=observed_codes(DTA, POOL, BED_WAVES),
        mode_sensitivity=modes,
        leakage_resistant=set(LEAKAGE_RESISTANT),
    )
    cb.to_yaml(OUT)
    print(f"wrote {len(cb)} items -> {OUT}")

    # --- what a human needs to look at before Phase 2 --------------------
    blocked = cb.needing_wording()
    if blocked:
        by_status: dict[str, list[str]] = {}
        for i in blocked:
            by_status.setdefault(i.wording_status, []).append(i.item_id)
        print(f"\n{len(blocked)} items are BLOCKED for elicitation on wording (1.4a):")
        for status, ids in sorted(by_status.items()):
            print(f"  {status:15s} {len(ids):2d}  {' '.join(sorted(ids))}")
        print("  -> fill with: python scripts/set_wording.py --item <id> --text '<verbatim>'")

    ms = {}
    for i in cb.items.values():
        ms.setdefault(i.mode_sensitivity, []).append(i.item_id)
    print("\nNORC mode sensitivity across the pool:")
    for k in ("likely", "investigate", "less_likely", "unknown"):
        if k in ms:
            print(f"  {k:12s} {len(ms[k]):2d}  {' '.join(sorted(ms[k]))}")

    print("\nroles:")
    for role in ("sanity", "excluded", "unassigned"):
        ids = [i.item_id for i in cb.items.values() if i.role == role]
        if ids:
            print(f"  {role:11s} {len(ids):2d}  {' '.join(sorted(ids))}")

    cost = cb.wording_experiment_cost()
    if cost:
        print(f"\n2021 wording experiment — base-only costs sample on {len(cost)} items:")
        print(f"  {'item':10s} {'wave':>5s} {'base':>6s} {'twin':>6s} {'base share':>11s}")
        for item_id, rec in sorted(cost.items()):
            for wave, w in rec["by_wave"].items():
                if w["twin_n"]:
                    print(f"  {item_id:10s} {wave:5d} {w['base_n']:6d} {w['twin_n']:6d} "
                          f"{w['base_share']:11.3f}")

    thin = {i.item_id: i.answered_n_by_wave for i in cb.items.values()
            if any(n < 1200 for n in i.answered_n_by_wave.values())}
    print(f"\nitems with a wave under 1,200 answered n: {len(thin)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
