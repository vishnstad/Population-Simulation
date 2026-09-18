#!/usr/bin/env python
"""Discover the candidate item pool and write it to codebooks/candidate_pool.yaml.

Reproducible version of the filtering described in ``popsim/data/pool.py``:
Replicating Core -> scale 2-7 with a complete published topline -> wave
coverage -> attitudinal. Plus the items the checklist names by hand, which enter
regardless because five of them were dropped from GSS after 2021 and every
automatic filter loses them.

Usage:  python scripts/build_pool.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import pyreadstat
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from popsim.data.pool import filter_pool

DTA = REPO / ".." / "data" / "gss" / "data" / "GSS_stata" / "gss7224_r3a.dta"
CODEBOOK_PDF = REPO / ".." / "data" / "gss" / "codebooks" / "GSS_2022_Codebook.pdf"
TOPLINES = REPO / "codebooks" / "gss2022_published_toplines.yaml"
OUT = REPO / "codebooks" / "candidate_pool.yaml"

BED_WAVES = [2010, 2012, 2014, 2016, 2018, 2021, 2022]
CORE_SECTION = "Replicating Core"
PAGE_HDR = "Codebook and Unweighted Frequencies for the 2022 General Social Survey"

NAMED_BY_CHECKLIST = {"natroad", "finrela", "satfin", "spkcom", "colath", "natarms", "natsoc", "libcom", "pornlaw", "libhomo", "consci", "spkhomo", "colhomo", "spkmil", "conmedic", "conlegis", "suicide1", "natspac", "natenvir", "sexeduc", "spanking", "postlife", "colcom", "libath", "natmass", "libmil", "spkath", "conlabor", "spkrac", "nateduc", "fair", "life", "colrac", "librac", "colmil", "pray", "homosex", "polviews"}


def codebook_sections(pdf_path: Path) -> dict[str, str]:
    """Which questionnaire section each 2022 variable belongs to.

    The section is a running page header, printed on the line after the document
    header. Tracking it is how "Replicating Core" — the items repeated every
    round — is separated from the topical modules asked once.
    """
    import pdfplumber

    section: str | None = None
    out: dict[str, str] = {}
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            lines = (page.extract_text() or "").splitlines()
            for i, ln in enumerate(lines):
                if PAGE_HDR in ln:
                    for nxt in lines[i + 1:i + 4]:
                        cand = nxt.strip()
                        if cand and not cand.startswith(("Variable:", "Page", "Label:")) and len(cand) < 60:
                            section = cand
                            break
                    continue
                m = re.match(r"^Variable:\s+([A-Z0-9_]+)\s", ln)
                if m and section:
                    out.setdefault(m.group(1).lower(), section)
    return out


def answered_by_wave(items: list[str], chunk: int = 110) -> dict[str, dict[int, int]]:
    out: dict[str, dict[int, int]] = {}
    for k in range(0, len(items), chunk):
        sub = items[k:k + chunk]
        df, _ = pyreadstat.read_dta(str(DTA), usecols=["year", *sub], encoding="latin1")
        df = df[df["year"].isin(BED_WAVES)]
        for c in sub:
            n = pd.to_numeric(df[c], errors="coerce").notna().groupby(df["year"]).sum()
            out[c] = {int(w): int(n.get(w, 0)) for w in BED_WAVES}
        del df
    return out


def main() -> int:
    _, meta = pyreadstat.read_dta(str(DTA), metadataonly=True, encoding="latin1")
    known = set(meta.column_names)
    published = yaml.safe_load(TOPLINES.read_text())["items"]

    sections = codebook_sections(CODEBOOK_PDF)
    core = sorted(
        v for v, sec in sections.items()
        if sec == CORE_SECTION
        and v in known
        and published.get(v, {}).get("table_complete")
        and 2 <= published[v]["scale_length"] <= 7
    )
    print(f"Replicating Core, scale 2-7, complete topline: {len(core)}")

    candidates = answered_by_wave(core)
    for name in NAMED_BY_CHECKLIST:
        candidates.setdefault(name, {})
    extra = answered_by_wave([n for n in NAMED_BY_CHECKLIST if n in known and n not in core])
    candidates.update({k: v for k, v in extra.items()})

    pool, dropped = filter_pool(
        candidates,
        labels=meta.column_names_to_labels,
        always_include=NAMED_BY_CHECKLIST,
    )
    print(f"pool: {len(pool)}   dropped: {len(dropped)}")

    OUT.write_text(yaml.safe_dump({
        "source": "gss7224_r3a.dta + GSS_2022_Codebook.pdf",
        "bed_waves": BED_WAVES,
        "section_filter": CORE_SECTION,
        "n_pool": len(pool),
        "pool": pool,
        "answered_n_by_wave": {k: candidates[k] for k in pool},
        "dropped": dropped,
    }, sort_keys=False, allow_unicode=True, width=100))
    print(f"wrote {OUT}")

    missing_named = sorted(NAMED_BY_CHECKLIST - set(pool))
    if missing_named:
        print(f"WARNING: checklist-named items missing from the pool: {missing_named}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
