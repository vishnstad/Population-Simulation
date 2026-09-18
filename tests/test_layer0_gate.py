"""Layer 0 — the gate (checklist 1.5, Part 2 Layer 0).

    "Match harmonized marginals against the published GSS toplines in the
     codebook PDFs on disk, within 0.5 pp, on at least 10 items plus one full
     cross-tab."

The project pool is gated here. The gate runs on GSS 2022, because the published
toplines on disk are the 2022 codebook's; the adapter code is the same for every
wave, so proving it on 2022 proves it.
"""

from __future__ import annotations

import pytest
import yaml

from popsim.data.topline import load_published
from popsim.evalx.gate1 import run_layer0

pytestmark = pytest.mark.layer0

POOL_PATH = "codebooks/gss_items.yaml"


@pytest.fixture(scope="module")
def pool(repo_root):
    return sorted(yaml.safe_load((repo_root / POOL_PATH).read_text())["items"])


@pytest.fixture(scope="module")
def report(cfg, repo_root, pool):
    return run_layer0(
        dta_path=cfg.bed_file,
        toplines_path=repo_root / "codebooks" / "gss2022_published_toplines.yaml",
        margins_path=cfg.data_path(cfg["aggregation.margins"]),
        item_pool=pool,
        tolerance_pp=cfg["evaluation.topline_tolerance_pp"],
    )


def test_gate_is_green(report):
    assert report.ok, "\n" + report.summary()


def test_at_least_ten_items_were_actually_checked(report, cfg):
    assert report.n_items_checked >= cfg["evaluation.topline_min_items"]


def test_every_checked_item_is_within_tolerance(report):
    bad = [(t.item_id, round(t.max_abs_diff_pp, 3)) for t in report.failures()]
    assert not bad, f"items outside {report.tolerance_pp} pp: {bad}"


def test_one_full_crosstab_reproduces_both_published_margins(report, cfg):
    assert len(report.crosstabs) >= cfg["evaluation.topline_min_crosstabs"]
    ct = report.crosstabs[0]
    assert ct.ok, ct.message
    assert set(ct.per_axis) == {"degree", "sex"}
    assert ct.n_cells == 10  # 5 degrees x 2 sexes


def test_weights_are_applied_and_land_near_the_population(report):
    assert report.weights, "the weighted path was never exercised"
    for w in report.weights:
        assert w.ok, w.message
    sex = next(w for w in report.weights if w.axis == "sex")
    # No two-year drift story exists for the adult sex ratio, so weighting has
    # to essentially nail it. This is the sharp end of the weight-misuse check.
    assert sex.weighted_l1_pp < 1.0, sex.message
    assert sex.weighted_l1_pp < sex.unweighted_l1_pp


def test_skipped_items_each_carry_a_reason(report):
    for item_id, reason in report.skipped.items():
        assert reason.strip(), f"{item_id} was skipped with no reason given"


def test_items_dropped_after_2021_are_skipped_not_silently_passed(report):
    """The whole spk/col/lib 'homo' and 'mil' sub-batteries left GSS after 2021."""
    dropped = {"spkhomo", "colhomo", "libhomo", "spkmil", "colmil", "libmil"}
    assert dropped <= set(report.skipped)
    assert not (dropped & {t.item_id for t in report.toplines})


def test_published_toplines_are_only_admitted_when_the_table_reconciles(repo_root):
    """A table that lost rows to a page break is not admissible evidence."""
    path = repo_root / "codebooks" / "gss2022_published_toplines.yaml"
    raw = yaml.safe_load(path.read_text())["items"]
    admitted = load_published(path)
    assert "natroad" in admitted
    # LIFENOW is an 0-10 ladder the codebook prints as three lines, with a
    # collapsed middle row. It reconciles to the printed total and is still not
    # a comparable table.
    assert raw["lifenow"]["parse_consistent"]
    assert not raw["lifenow"]["table_complete"]
    assert "lifenow" not in admitted


def test_a_reversed_scale_would_fail_the_gate(cfg, repo_root):
    """The bug Layer 0 exists to catch, injected on purpose.

    A reversed 5-point scale produces plausible, completely wrong numbers
    forever. If flipping one item's codes does not turn the gate red, the gate
    is not testing anything.
    """
    import pandas as pd

    from popsim.data.adapters.gss import read_gss_columns
    from popsim.data.topline import published_topline_check

    checks = load_published(repo_root / "codebooks" / "gss2022_published_toplines.yaml",
                            items=["finrela"])
    frame, _ = read_gss_columns(cfg.bed_file, ["year", "finrela"])
    frame = frame[frame["year"] == 2022].copy()

    clean = published_topline_check(frame, checks, tolerance_pp=0.5)
    assert clean[0].ok, clean[0].message

    reversed_frame = frame.copy()
    codes = checks["finrela"].codes
    flip = dict(zip(codes, reversed(codes), strict=True))
    reversed_frame["finrela"] = pd.to_numeric(
        reversed_frame["finrela"], errors="coerce"
    ).map(flip)
    dirty = published_topline_check(reversed_frame, checks, tolerance_pp=0.5)
    assert not dirty[0].ok, "a reversed scale slipped through the gate"
