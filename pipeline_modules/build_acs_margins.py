"""
Build census raking margins from the ACS PUMS person file (module M6 input).

    python build_acs_margins.py --pums ../Dataset/csv_pus.zip --out config/acs_margins.yaml

Why this exists
---------------
Cluster weights come from the GSS sample. Survey samples are not the population:
they under-represent the young, the mobile, and the less educated even after the
survey's own weighting. Raking the cluster weights to external census margins
(iterative proportional fitting) makes the composition of a simulated population
match the real one, so a regional estimate is a statement about the region rather
than about who answered the phone.

The ACS PUMS person file is the standard US source. This script streams it in
chunks -- the 2024 file is ~2.4 GB across two CSVs -- reading only the six columns
needed and never holding the whole thing in memory.

Columns used
------------
``PWGTP``  person weight        ``AGEP``  age
``SEX``    sex                  ``SCHL``  educational attainment
``REGION`` census region        ``PINCP`` personal income

The recodes below map ACS categories onto the harmonised GSS demographic schema
from preprocessing. They are explicit and reviewable rather than inferred: a
silent mismatch here would rake toward the wrong population.
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, Optional

import numpy as np
import pandas as pd
import yaml

USECOLS = ["PWGTP", "AGEP", "SEX", "SCHL", "REGION", "PINCP"]

# ACS REGION -> harmonised GSS region
REGION_MAP = {1: "northeast", 2: "midwest", 3: "south", 4: "west"}

# ACS SEX -> harmonised sex
SEX_MAP = {1: "male", 2: "female"}


def age_band(age: float) -> Optional[str]:
    """GSS age bands. Under 18 is out of scope -- the GSS is an adult sample."""
    if age < 18:
        return None
    for hi, label in ((24, "18-24"), (34, "25-34"), (44, "35-44"),
                      (54, "45-54"), (64, "55-64")):
        if age <= hi:
            return label
    return "65+"


def education(schl: float) -> Optional[str]:
    """
    ACS SCHL -> the four-level harmonised education scale.

    SCHL 1-11  = no schooling through grade 11      -> primary
    SCHL 12-17 = grade 12 / high school diploma/GED -> secondary
    SCHL 18-20 = some college / associate degree    -> higher_secondary
    SCHL 21-24 = bachelor's and above               -> tertiary
    """
    if not np.isfinite(schl):
        return None
    s = int(schl)
    if s <= 11:
        return "primary"
    if s <= 17:
        return "secondary"
    if s <= 20:
        return "higher_secondary"
    return "tertiary"


def iter_pums_chunks(pums_path: Path, chunksize: int = 500_000) -> Iterator[pd.DataFrame]:
    """Stream every person CSV inside the PUMS zip (or a bare CSV) in chunks."""
    path = Path(pums_path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not members:
                raise SystemExit(f"no CSV inside {path.name}")
            for member in members:
                print(f"  reading {member} ...")
                with zf.open(member) as handle:
                    reader = pd.read_csv(
                        io.TextIOWrapper(handle, "utf-8"),
                        usecols=lambda c: c.strip('"') in USECOLS,
                        chunksize=chunksize,
                        low_memory=False,
                    )
                    for chunk in reader:
                        yield chunk
    else:
        for chunk in pd.read_csv(path, usecols=USECOLS, chunksize=chunksize, low_memory=False):
            yield chunk


def build_margins(pums_path: Path, income_quintiles: bool = True) -> Dict[str, Dict[str, float]]:
    """Accumulate weighted marginal totals for each raking dimension."""
    totals: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    income_sum = np.zeros(0)
    income_w = np.zeros(0)
    n_rows = 0

    for chunk in iter_pums_chunks(pums_path):
        chunk.columns = [c.strip('"') for c in chunk.columns]
        adults = chunk[chunk["AGEP"] >= 18]
        if adults.empty:
            continue
        n_rows += len(adults)
        w = adults["PWGTP"].to_numpy(dtype=float)

        bands = adults["AGEP"].map(age_band)
        for band, weight in zip(bands, w):
            if band:
                totals["age_band"][band] += weight

        for code, weight in zip(adults["SEX"], w):
            label = SEX_MAP.get(int(code))
            if label:
                totals["sex"][label] += weight

        for code, weight in zip(adults["REGION"], w):
            label = REGION_MAP.get(int(code)) if np.isfinite(code) else None
            if label:
                totals["region"][label] += weight

        for schl, weight in zip(adults["SCHL"], w):
            label = education(schl)
            if label:
                totals["education"][label] += weight

        if income_quintiles and "PINCP" in adults.columns:
            inc = adults["PINCP"].to_numpy(dtype=float)
            ok = np.isfinite(inc)
            income_sum = np.concatenate([income_sum, inc[ok]])
            income_w = np.concatenate([income_w, w[ok]])

        print(f"    {n_rows:,} adults processed", end="\r")

    print(f"\n  {n_rows:,} adult records total")

    margins = {
        dim: {k: v / sum(vals.values()) for k, v in sorted(vals.items())}
        for dim, vals in totals.items()
        if sum(vals.values()) > 0
    }

    # Income quintiles are shares by construction (0.2 each); what matters is
    # recording the cut points so preprocessing can be checked against them.
    if income_quintiles and len(income_sum):
        order = np.argsort(income_sum)
        sorted_inc = income_sum[order]
        cum = np.cumsum(income_w[order])
        cum /= cum[-1]
        cuts = [float(sorted_inc[np.searchsorted(cum, q)]) for q in (0.2, 0.4, 0.6, 0.8)]
        margins["income_band"] = {f"q{i + 1}": 0.2 for i in range(5)}
        margins["_income_quintile_cutpoints_usd"] = {
            f"p{int(q * 100)}": round(c, 0) for q, c in zip((0.2, 0.4, 0.6, 0.8), cuts)
        }

    return margins


def main() -> int:
    ap = argparse.ArgumentParser(description="Build ACS raking margins for M6")
    ap.add_argument("--pums", required=True, help="path to csv_pus.zip or an extracted CSV")
    ap.add_argument("--out", default="config/acs_margins.yaml")
    ap.add_argument("--no-income", action="store_true")
    args = ap.parse_args()

    pums = Path(args.pums)
    if not pums.exists():
        print(f"ERROR: {pums} not found.")
        print("The ACS PUMS person file is Dataset/csv_pus.zip (already downloaded).")
        return 1

    print(f"Building raking margins from {pums.name}")
    print("(streams the file in chunks; the 2024 PUMS takes a few minutes)")
    margins = build_margins(pums, income_quintiles=not args.no_income)

    out = Path(args.out)
    if not out.is_absolute():
        out = Path(__file__).resolve().parent / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "source": f"ACS PUMS person file ({pums.name})",
                "universe": "civilian adults aged 18+",
                "margins": margins,
            },
            f,
            sort_keys=False,
        )

    print(f"\nWrote {out}")
    for dim, dist in margins.items():
        if dim.startswith("_"):
            continue
        preview = ", ".join(f"{k} {v * 100:.1f}%" for k, v in list(dist.items())[:6])
        print(f"  {dim:<14} {preview}")
    print("\nEnable raking by setting in config/run_gss2024.yaml:")
    print(f"  aggregation.margins_file: Codes/{out.relative_to(Path(__file__).resolve().parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
