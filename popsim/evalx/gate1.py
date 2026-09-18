"""Gate 1 runner — Layer 0 on the GSS bed (checklist 1.5).

Assembles the three Layer 0 checks in :mod:`popsim.data.topline` over the full
published item pool and writes the result into a run directory.

Scope note. The gate runs on **GSS 2022**, not on the pooled bed, because the
published toplines on disk are the 2022 codebook's. That is the right scope: the
gate's job is to prove the *adapter* reproduces reality, and the adapter is the
same code for every wave. Items not fielded in 2022 (the whole ``spk/col/lib``
'homo' and 'mil' sub-batteries were dropped after 2021) are reported as skipped
with the reason, never silently passed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from ..data.adapters.gss import read_gss_columns
from ..data.harmonize import harmonize_demographics
from ..data.topline import (
    CheckReport,
    crosstab_margin_check,
    load_published,
    published_topline_check,
    weight_direction_check,
)

log = logging.getLogger("popsim.gate1")

__all__ = ["GATE_WAVE", "run_layer0"]

GATE_WAVE = 2022


def run_layer0(
    *,
    dta_path: str | Path,
    toplines_path: str | Path,
    margins_path: str | Path,
    item_pool: list[str] | None = None,
    wave: int = GATE_WAVE,
    tolerance_pp: float = 0.5,
    crosstab_axes: tuple[str, str] = ("degree", "sex"),
    out_dir: str | Path | None = None,
) -> CheckReport:
    published = load_published(toplines_path)
    wanted = sorted(set(item_pool) & set(published)) if item_pool else sorted(published)

    # Read only what the gate needs. The file is 598 MB; usecols is not optional.
    demo_cols = ["age", "sex", "degree", "region", "region_7222", "srcbelt", "mode"]
    cols = ["year", "id", "wtssps"] + demo_cols + wanted
    meta_cols_missing = []
    try:
        frame, _ = read_gss_columns(dta_path, cols)
    except KeyError:
        # Fall back to the intersection, and say which names were unknown, rather
        # than failing the whole gate over one renamed variable.
        from ..data.adapters.gss import gss_metadata
        known = set(gss_metadata(dta_path).column_names)
        meta_cols_missing = [c for c in cols if c not in known]
        frame, _ = read_gss_columns(dta_path, [c for c in cols if c in known])
        wanted = [w for w in wanted if w in known]

    frame = frame[frame["year"] == wave].copy()
    if frame.empty:
        raise ValueError(f"no rows for wave {wave}")

    checks = {k: v for k, v in published.items() if k in wanted and k in frame.columns}

    # --- A. unweighted toplines -------------------------------------------
    results = published_topline_check(
        frame, checks, wave=wave, tolerance_pp=tolerance_pp, weights=None
    )
    # Items the gate could not exercise, each with a reason.
    skipped: dict[str, str] = {c: "not a column of this release" for c in meta_cols_missing}
    for item_id in (item_pool or []):
        if item_id in checks:
            continue
        if item_id not in published:
            skipped[item_id] = (
                f"no entry in the {Path(toplines_path).name} pool — not fielded in "
                f"GSS {wave}, so there is no published figure to gate against"
            )
        else:
            skipped[item_id] = "absent from the frame"
    # Items whose published table did not parse cleanly are dropped by
    # load_published; name them so the count is honest.
    import yaml
    raw = yaml.safe_load(Path(toplines_path).read_text())["items"]
    for item_id in (item_pool or []):
        if item_id in raw and not raw[item_id].get("parse_consistent", False):
            skipped[item_id] = "published table did not parse consistently; check by hand"

    # --- B. one full cross-tab, both margins published ---------------------
    ct = crosstab_margin_check(frame, published, axes=crosstab_axes, tolerance_pp=tolerance_pp)

    # --- C. the weighted path, against ACS -------------------------------
    demo = harmonize_demographics(frame)
    margins = pd.read_parquet(margins_path)
    # Per-axis bounds. `sex` is tight because there is no two-year drift story
    # for the adult sex ratio; `degree` and `age_band` carry the lag the
    # checklist measured at 2.9 pp and 2.3 pp, so their bound is a little wider.
    axis_bounds = {"sex": 1.0, "degree": 3.5, "age_band": 3.5}
    weight_results = [
        weight_direction_check(demo, frame["wtssps"], margins,
                               axis=axis, max_gap_pp=bound)
        for axis, bound in axis_bounds.items()
        if axis in demo.columns and axis in margins.columns
    ]

    report = CheckReport(
        wave=wave, tolerance_pp=tolerance_pp,
        toplines=results, crosstabs=[ct], weights=weight_results, skipped=skipped,
    )

    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "layer0_report.json").write_text(json.dumps(report.to_dict(), indent=2))
        (out / "layer0_summary.txt").write_text(report.summary())
    return report
