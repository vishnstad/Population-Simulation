"""Phase 4/5 — the calibration layer, the store, the baselines, the harness.

The properties tested here are the ones that make the design's claims checkable:
``s = 0`` *is* B0a, oracle input round-trips at ``s = 1``, and the store never
double-counts a resumed cell. Each of those is an equality with an analytic
answer, so a regression shows up as a failed assertion rather than as a slightly
worse W1 that nobody notices.
"""

from __future__ import annotations

import numpy as np
import pytest

from popsim.agents.elicit import RawElicitation
from popsim.agents.runner import ElicitationStore, plan_cells
from popsim.calibration.fit import (
    CalibrationPair,
    Calibrator,
    apply_calibrator,
    deviation_reference,
    fit_calibrator,
)
from popsim.calibration.shape import (
    cdf,
    from_cdf,
    sd_pos,
    temper,
    temper_to_sd,
)
from popsim.evalx.baselines import (
    b0a_national_oracle,
    b4_uncalibrated,
    point_answers_to_histogram,
    resample_histogram,
)
from popsim.evalx.metrics import (
    between_cluster_sd,
    between_cluster_spearman,
    ece,
    interval_coverage,
    ordering_spearman,
    w1,
)

RNG = np.random.default_rng(17)


def _rand_hist(k: int, n: int = 1):
    x = RNG.random((n, k)) + 0.05
    return x / x.sum(axis=1, keepdims=True)


# ------------------------------------------------------------------- shape

@pytest.mark.parametrize("k", [2, 3, 4, 5, 6, 7])
def test_cdf_roundtrips(k):
    for h in _rand_hist(k, 20):
        assert np.allclose(from_cdf(cdf(h)), h, atol=1e-12)


def test_from_cdf_projects_a_non_monotone_input():
    """The only place a perturbed CDF is allowed to be clipped."""
    h = from_cdf(np.array([0.6, 0.3, 0.9]))
    assert np.all(h >= 0) and abs(h.sum() - 1) < 1e-12


def test_temper_sharpens_and_flattens():
    h = np.array([0.1, 0.4, 0.3, 0.2])
    assert sd_pos(temper(h, 4.0)) < sd_pos(h) < sd_pos(temper(h, 0.25))


def test_temper_to_sd_hits_reachable_targets():
    h = np.array([0.1, 0.4, 0.3, 0.2])
    for t in (0.05, 0.15, 0.25, sd_pos(h)):
        assert abs(sd_pos(temper_to_sd(h, t)) - t) < 1e-3


def test_temper_to_sd_is_identity_at_the_current_sd():
    """What Gate 4 leans on: alpha=0, beta=1 must not move the histogram."""
    for k in (2, 3, 4, 6):
        for h in _rand_hist(k, 10):
            assert np.allclose(temper_to_sd(h, sd_pos(h)), h, atol=1e-12)


def test_temper_to_sd_clamps_instead_of_raising_on_an_unreachable_target():
    """A symmetric bimodal histogram cannot be narrowed by power-tempering.

    A real limitation of the M5 step-2 mechanism, not a numerical failure — the
    variance-ratio metric reports the SD achieved, so it must come back with a
    histogram rather than an exception.
    """
    h = np.array([0.45, 0.05, 0.05, 0.45])
    out = temper_to_sd(h, 0.05)
    assert np.isfinite(out).all() and abs(out.sum() - 1) < 1e-12


# ------------------------------------------------------------- calibration

def _toy_pairs(n_items=6, n_clusters=8, k=4, noise=0.0, scale=1.0):
    """Anchor pairs where the model's deviation is ``scale`` x the truth's.

    So the scale the calibrator should recover is ``1 / scale``: the fitted term
    is what the raw deviation has to be multiplied by to reach the true one.
    """
    pairs, national = [], {}
    w = np.linspace(1.0, 2.0, n_clusters)
    for i in range(n_items):
        truths = _rand_hist(k, n_clusters)
        nat = (w / w.sum()) @ truths
        national[f"item{i}"] = nat
        cn = cdf(nat)
        for c in range(n_clusters):
            dev = cdf(truths[c]) - cn
            raw = from_cdf(cn + dev * scale
                           + noise * RNG.normal(0, 1, cn.size))
            pairs.append(CalibrationPair(
                cluster_id=f"c{c}", item_id=f"item{i}",
                raw=[float(x) for x in raw],
                truth=[float(x) for x in truths[c]],
                weight=float(w[c]), n_eff=200.0))
    return pairs, national


def test_oracle_pairs_fit_a_scale_of_one():
    pairs, national = _toy_pairs()
    cal = fit_calibrator(pairs, national, fit_per_cluster=False)
    assert abs(cal.scale_global - 1.0) < 0.02


def test_a_model_that_understates_deviations_fits_a_scale_above_one():
    """Deviations half the true size must be scaled back up by about two."""
    pairs, national = _toy_pairs(scale=0.5)
    cal = fit_calibrator(pairs, national, fit_per_cluster=False)
    assert cal.scale_global == pytest.approx(2.0, abs=0.15)


def test_a_model_that_overstates_deviations_fits_a_scale_below_one():
    pairs, national = _toy_pairs(scale=2.0)
    cal = fit_calibrator(pairs, national, fit_per_cluster=False)
    assert cal.scale_global == pytest.approx(0.5, abs=0.1)


def test_scale_zero_reproduces_b0a_exactly():
    """The ablation is the identity, not an approximation of it."""
    cal = Calibrator(variance_restoration=False)
    raw = {f"c{i}": h for i, h in enumerate(_rand_hist(4, 6))}
    nat = np.array([0.1, 0.2, 0.4, 0.3])
    wts = {c: 1.0 for c in raw}
    got = apply_calibrator(cal, raw, nat, wts, scheme="zero")
    b0a = b0a_national_oracle(nat, list(raw))
    for c in raw:
        assert np.allclose(got[c], b0a[c], atol=1e-12)
        assert w1(got[c], b0a[c]) == 0.0


def test_truth_in_returns_truth_out_at_scale_one():
    """Layer 3 in miniature: the deviation reference IS the weighted mixture."""
    cal = Calibrator(variance_restoration=False)
    truths = _rand_hist(5, 7)
    wts = {f"c{i}": float(w) for i, w in enumerate(np.linspace(1, 3, 7))}
    raw = {f"c{i}": truths[i] for i in range(7)}
    ws = np.asarray(list(wts.values())); ws = ws / ws.sum()
    mixture = ws @ truths
    got = apply_calibrator(cal, raw, mixture, wts, scheme="unit")
    for i in range(7):
        assert w1(got[f"c{i}"], truths[i]) < 1e-12


def test_deviation_reference_is_the_weighted_mixture_cdf():
    hs = list(_rand_hist(4, 5))
    w = np.array([1.0, 2.0, 3.0, 1.0, 1.0])
    ref = deviation_reference(hs, w)
    mixture = (w / w.sum()) @ np.vstack(hs)
    assert np.allclose(ref, cdf(mixture), atol=1e-12)


def test_calibrator_json_roundtrip(tmp_path):
    pairs, national = _toy_pairs()
    cal = fit_calibrator(pairs, national)
    back = Calibrator.from_json(cal.to_json(tmp_path / "cal.json"))
    assert back.scale_global == pytest.approx(cal.scale_global)
    assert back.scale_by_k == cal.scale_by_k


def test_per_cluster_scale_is_shrunk_toward_the_global_one():
    pairs, national = _toy_pairs(n_items=8, noise=0.02)
    cal = fit_calibrator(pairs, national, fit_per_cluster=True, tau=100.0)
    assert cal.scale_by_cluster
    spread = np.std(list(cal.scale_by_cluster.values()))
    assert spread < 1.0  # pooling, not free per-cluster fitting


def test_no_per_item_scale_is_ever_exposed():
    """Fitting s per item on targets is the invalid fix §8.5 forbids."""
    cal = fit_calibrator(*_toy_pairs())
    assert not hasattr(cal, "scale_by_item")


# ---------------------------------------------------------------- the store

def _rec(item, cluster, p, r):
    return RawElicitation(cluster_id=cluster, item_id=item, model="m", provider="p",
                          paraphrase_id=p, repeat_id=r, hist=[0.5, 0.5], ok=True)


def test_store_appends_and_resumes_without_double_counting(tmp_path):
    st = ElicitationStore(root=tmp_path, model="a/b:c", profile="dev")
    st.append([_rec("x", "c1", 0, 0), _rec("x", "c1", 0, 1)])
    assert st.done("x") == {("c1", 0, 0), ("c1", 0, 1)}
    st.append([_rec("x", "c2", 0, 0)])
    assert len(st.load(["x"])) == 3
    assert st.counts()["x"] == 3


def test_store_slugs_a_model_tag_into_one_directory(tmp_path):
    st = ElicitationStore(root=tmp_path, model="qwen2.5-ctx8k:7b-instruct-8192",
                          profile="permutation")
    st.append([_rec("i", "c", 0, 0)])
    assert st.path("i").exists()
    assert ":" not in st.dir.name and "/" not in st.dir.name


def test_plan_cells_is_item_major():
    cells = plan_cells(["a", "b"], ["c1", "c2", "c3"])
    assert [i for i, _ in cells] == ["a", "a", "a", "b", "b", "b"]


# ------------------------------------------------------------- baselines

def test_resample_histogram_preserves_mass():
    for k_in in (2, 3, 5):
        for k_out in (2, 4, 7):
            out = resample_histogram(_rand_hist(k_in)[0], k_out)
            assert out.size == k_out and abs(out.sum() - 1) < 1e-12


def test_b4_is_the_raw_ensemble_mean_renormalized():
    raw = {"c": np.array([2.0, 2.0])}
    assert np.allclose(b4_uncalibrated(raw)["c"], [0.5, 0.5])


def test_point_answers_become_a_histogram():
    h = point_answers_to_histogram([1, 1, 2, 3], [1, 2, 3])
    assert np.allclose(h, [0.5, 0.25, 0.25])


# --------------------------------------------------------------- metrics

def test_between_cluster_spearman_is_one_for_a_perfect_ordering():
    t = [np.array([p, 1 - p]) for p in (0.1, 0.3, 0.5, 0.7, 0.9)]
    p = [np.array([q, 1 - q]) for q in (0.15, 0.32, 0.48, 0.75, 0.85)]
    rho, _ = between_cluster_spearman(p, t)
    assert rho == pytest.approx(1.0)


def test_between_cluster_spearman_is_nan_for_a_hedged_model():
    """F2 in its pure form: one answer per item regardless of the card."""
    t = [np.array([p, 1 - p]) for p in (0.1, 0.4, 0.6, 0.9)]
    p = [np.array([0.5, 0.5])] * 4
    rho, _ = between_cluster_spearman(p, t)
    assert not np.isfinite(rho)


def test_between_cluster_sd_detects_collapse():
    t = [np.array([x, 1 - x]) for x in (0.1, 0.4, 0.6, 0.9)]
    flat = [np.array([0.5, 0.5])] * 4
    assert between_cluster_sd(flat) == pytest.approx(0.0)
    assert between_cluster_sd(t) > 0.2


def test_ece_is_zero_for_a_perfect_prediction():
    h = _rand_hist(4, 6)
    assert ece(h, h) == pytest.approx(0.0, abs=1e-12)


def test_interval_coverage_counts_what_is_inside():
    lo = np.array([0.0, 0.5]); hi = np.array([0.4, 1.0]); t = np.array([0.2, 0.2])
    assert interval_coverage(lo, hi, t) == pytest.approx(0.5)


def test_tolerance_battery_ordering_needs_three_items():
    a = {"spkath": np.array([0.3, 0.7]), "spkcom": np.array([0.5, 0.5])}
    assert not np.isfinite(ordering_spearman(a, a)[0])


def test_tolerance_battery_ordering_is_one_when_reproduced():
    truth = {"spkath": np.array([0.2, 0.8]), "spkcom": np.array([0.5, 0.5]),
             "spkrac": np.array([0.7, 0.3]), "spkhomo": np.array([0.3, 0.7])}
    pred = {k: np.array([v[0] * 0.9 + 0.05, 1 - (v[0] * 0.9 + 0.05)])
            for k, v in truth.items()}
    rho, _ = ordering_spearman(pred, truth)
    assert rho == pytest.approx(1.0)


# ------------------------------------------------- the gate, on the real bed

@pytest.mark.slow
def test_gate4_oracle_passthrough_is_green(cfg):
    from popsim.evalx.gate4 import run_oracle_passthrough
    from popsim.pipeline import build_bed

    bed = build_bed(cfg)
    res = run_oracle_passthrough(
        bed, anchor_items=sorted(bed.split["anchors"])[:10],
        target_items=bed.ranked_targets()[:6], cluster_ids=bed.clusters(20))
    assert res.ok, res.summary()
    assert res.max_w1_at_s1 < 1e-9
    assert res.max_w1_s0_vs_b0a == 0.0


@pytest.mark.slow
def test_the_level_is_the_mixture_of_the_scored_cells(cfg):
    from popsim.pipeline import build_bed

    bed = build_bed(cfg)
    item = bed.ranked_targets()[0]
    cids = bed.clusters(0)
    lvl = bed.level_hist(item, cids)
    hs, ws = [], []
    for c in cids:
        h = bed.stats.hist(c, item)
        w = bed.cell_weight(c, item)
        if h is not None and h.sum() > 0 and w > 0:
            hs.append(h); ws.append(w)
    ws = np.asarray(ws) / np.sum(ws)
    assert np.allclose(lvl, ws @ np.vstack(hs), atol=1e-12)


# ------------------------------------------- selection happens on anchors

def test_subspace_projection_is_idempotent_and_shrinks():
    """A projection removes something and then removes nothing more."""
    from popsim.calibration.fit import Calibrator

    rng = np.random.default_rng(3)
    U = np.linalg.svd(rng.normal(size=(8, 8)), full_matrices=False)[0][:, :2]
    cal = Calibrator(subspace_clusters=[f"c{i}" for i in range(8)],
                     subspace_u=U.tolist(), subspace_rank=2)
    dev = {f"c{i}": rng.normal(size=3) for i in range(8)}
    once = cal.project(dev)
    twice = cal.project(once)
    for c in dev:
        assert np.allclose(once[c], twice[c], atol=1e-12)
    assert (np.linalg.norm(np.vstack(list(once.values())))
            < np.linalg.norm(np.vstack(list(dev.values()))))


def test_subspace_projection_is_a_noop_at_rank_zero():
    from popsim.calibration.fit import Calibrator

    cal = Calibrator(subspace_rank=0)
    dev = {"a": np.array([0.1, 0.2])}
    assert cal.project(dev) is dev


def test_rank_and_scheme_are_chosen_out_of_fold(tmp_path):
    """In-sample selection would always pick the richest parameterisation."""
    pairs, national = _toy_pairs(n_items=9, n_clusters=10, noise=0.03)
    folds = {f"item{i}": i % 3 for i in range(9)}
    cal = fit_calibrator(pairs, national, folds=folds, fit_per_cluster=False)
    assert cal.notes["scheme_chosen"] in {"global", "by_k"}
    assert set(cal.notes["scheme_curve_oof"]) == {"global", "by_k"}


def test_variance_restoration_is_switched_on_only_if_it_helps_on_anchors():
    pairs, national = _toy_pairs()
    cal = fit_calibrator(pairs, national, fit_per_cluster=False)
    chosen = cal.notes["variance_restoration_anchor_w1"]["chosen"]
    assert chosen in {"on", "off"}
    assert cal.variance_restoration == (chosen == "on")


def test_forcing_variance_restoration_off_is_honoured():
    pairs, national = _toy_pairs()
    cal = fit_calibrator(pairs, national, variance_restoration=False,
                         fit_per_cluster=False)
    assert cal.variance_restoration is False


# ------------------------------------------------------ aggregation (M6)

def test_aggregate_is_the_weighted_mixture():
    from popsim.aggregate.mixture import aggregate

    d = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])}
    assert np.allclose(aggregate(d, {"a": 3.0, "b": 1.0}), [0.75, 0.25])


def test_key_normalises_an_extension_dtype_value():
    """`degree` is Int8, so `.to_numpy()` hands back 3.0, not 3."""
    from popsim.aggregate.mixture import _key

    assert _key(np.float64(3.0)) == "3" == _key(3) == _key(np.int8(3))
    assert _key(None) == "" == _key(float("nan"))
    assert _key("65+") == "65+"


@pytest.mark.slow
def test_raking_converges_and_moves_composition(cfg):
    import pandas as pd

    from popsim.aggregate.mixture import rake_weights
    from popsim.pipeline import build_bed

    bed = build_bed(cfg)
    margins = pd.read_parquet(cfg.data_path(cfg["aggregation.margins"]))
    res = rake_weights(bed, margins, axes=tuple(cfg["partition.axes"]),
                       max_iter=cfg["aggregation.rake_max_iter"],
                       tol=cfg["aggregation.rake_tol"])
    assert res.converged and not res.fallback, res.summary()
    assert res.max_margin_gap < 1e-5
    base = np.array([bed.cluster_weight[c] for c in bed.leaves_by_share])
    base = base / base.sum()
    new = np.array([res.weights[c] for c in bed.leaves_by_share])
    # A pooled 2010-2022 bed is not ACS 2024; if raking moved nothing it did not run.
    assert 0.01 < float(np.abs(new - base).sum()) < 1.0


# ------------------------------------------------------- the router (M7)

def _codebook():
    from pathlib import Path

    import yaml
    return yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "codebooks" / "gss_items.yaml").read_text()
    )["items"]


def test_verbatim_wording_routes_to_observed():
    """F3's defence: an item the data answers must never reach the LLM."""
    from popsim.scenarios.router import Router

    cb = _codebook()
    r = Router(cb, threshold=0.25)
    d = r.route(cb["natroad"]["text"])
    assert d.observed and d.matched_item == "natroad"


def test_a_novel_scenario_routes_to_simulate():
    from popsim.scenarios.router import Router

    d = Router(_codebook(), threshold=0.25).route(
        "Would gig workers in Bengaluru sign up for a UPI-based micro-pension "
        "that deducts 2% of each payout?")
    assert not d.observed and "honesty box" in d.explain()


def test_the_specs_085_threshold_has_zero_recall():
    """Recorded as a test so it cannot be quietly reverted to the spec's value."""
    from pathlib import Path

    import yaml

    from popsim.scenarios.router import Router, evaluate_threshold

    root = Path(__file__).resolve().parents[1]
    pairs = [(p["text"], p["item"], bool(p["duplicate"]))
             for p in yaml.safe_load(
                 (root / "codebooks" / "router_duplicate_pairs.yaml").read_text())["pairs"]]
    rows = {r["threshold"]: r for r in evaluate_threshold(Router(_codebook()), pairs)}
    assert rows[0.85]["recall"] == 0.0
    assert rows[0.25]["recall"] > 0.7 and rows[0.25]["precision"] > 0.9


def test_compiled_scenarios_carry_simulated_provenance():
    from popsim.scenarios.schema import Scenario, compile_scenario

    item = compile_scenario(Scenario(
        scenario_id="upi_pension", decision_question="Would you sign up?",
        labels=["definitely", "probably", "probably not", "definitely not"]))
    assert item["provenance"] == "simulated" and item["synthetic"] is True
    assert item["codes"] == [1, 2, 3, 4] and item["item_id"].startswith("scenario:")


def test_a_scenario_context_is_capped():
    from popsim.scenarios.schema import Scenario

    with pytest.raises(ValueError, match="500 words"):
        Scenario(scenario_id="x", decision_question="?", context="w " * 501,
                 labels=["a", "b"])


# ---------------------------------------------- the honesty box (6.5, M10)

def test_an_observed_answer_says_no_model_was_asked():
    from popsim.report.honesty import build_honesty_box

    b = build_honesty_box("q", None, None, provenance="observed")
    assert "observed" in b.sentence() and "no model was asked" in b.sentence()


def test_a_simulated_answer_with_no_close_item_refuses_to_quote_a_number():
    """Quoting an unrelated item's error is the demo-ware §F7 warns about."""
    from popsim.report.honesty import build_honesty_box
    from popsim.scenarios.router import Router

    rep = {"per_item": [{"item_id": "natroad", "w1": {"system": 0.05, "B0a": 0.09}}],
           "noise_floor": 0.0273, "model": "m"}
    b = build_honesty_box(
        "Would gig workers in Bengaluru adopt a UPI-based micro-pension?",
        Router(_codebook(), threshold=0.25), rep)
    assert not b.nearest
    assert "no validated question is close enough" in b.sentence()


def test_a_simulated_answer_near_a_scored_item_quotes_the_triple():
    from popsim.report.honesty import build_honesty_box
    from popsim.scenarios.router import Router

    cb = _codebook()
    rep = {"per_item": [{"item_id": "natroad", "w1": {"system": 0.05, "B0a": 0.09}}],
           "noise_floor": 0.0273, "model": "m"}
    b = build_honesty_box(cb["natroad"]["text"], Router(cb, threshold=0.25), rep)
    assert b.nearest and b.nearest[0]["item_id"] == "natroad"
    s = b.sentence()
    assert "0.0500" in s and "0.0900" in s and "0.0273" in s


def test_collect_cells_truncates_the_ensemble_in_store_order():
    import pandas as pd

    from popsim.evalx.harness import collect_cells

    df = pd.DataFrame([
        {"item_id": "i", "cluster_id": "c", "ok": True, "hist": [0.9, 0.1],
         "paraphrase_id": 0, "repeat_id": 1},
        {"item_id": "i", "cluster_id": "c", "ok": True, "hist": [0.1, 0.9],
         "paraphrase_id": 0, "repeat_id": 0},
    ])
    got = collect_cells(df, max_members=1)[("i", "c")]
    assert got.shape == (1, 2)
    assert np.allclose(got[0], [0.1, 0.9])   # repeat 0 first, not row order


def test_projection_on_a_subset_uses_the_restricted_basis():
    """A zero is not 'no information' — it asserts the cluster is average.

    Padding absent clusters with zeros and projecting in the full space spends
    the projection's budget honouring that assertion. Measured cost when this was
    wrong: 0.0769 -> 0.0944 macro W1 on the 14B at 16 of 56 clusters.
    """
    from popsim.calibration.fit import project_rows

    rng = np.random.default_rng(11)
    U = np.linalg.svd(rng.normal(size=(20, 20)), full_matrices=False)[0][:, :3]
    rows = np.arange(0, 20, 2)                     # half the clusters present
    D = rng.normal(size=(rows.size, 4))
    P = project_rows(U, rows, D)
    # idempotent, and a genuine contraction
    assert np.allclose(project_rows(U, rows, P), P, atol=1e-10)
    assert np.linalg.norm(P) < np.linalg.norm(D)
    # and NOT the same as zero-padding into the full space
    full = np.zeros((20, 4))
    full[rows] = D
    padded = (U @ (U.T @ full))[rows]
    assert not np.allclose(P, padded, atol=1e-6)


def test_projection_is_the_identity_when_rows_are_scarcer_than_the_rank():
    from popsim.calibration.fit import Calibrator

    rng = np.random.default_rng(5)
    U = np.linalg.svd(rng.normal(size=(10, 10)), full_matrices=False)[0][:, :4]
    cal = Calibrator(subspace_clusters=[f"c{i}" for i in range(10)],
                     subspace_u=U.tolist(), subspace_rank=4)
    dev = {f"c{i}": rng.normal(size=3) for i in range(3)}
    assert cal.project(dev) is dev


def test_residual_sd_is_estimated_per_scale_length():
    """The missing term in an honest interval, and it is fitted on anchors."""
    pairs, national = _toy_pairs(n_items=8, n_clusters=10, noise=0.05)
    cal = fit_calibrator(pairs, national, fit_per_cluster=False)
    assert cal.resid_sd_by_k
    sd = cal.residual_sd(4)
    assert sd is not None and sd.size == 3 and np.all(sd >= 0)
    assert cal.residual_sd(99) is None


def test_the_residual_term_widens_the_interval():
    from popsim.evalx.harness import _bootstrap_intervals

    rng = np.random.default_rng(2)
    stacks = {f"c{i}": _rand_hist(4, 3) for i in range(6)}
    level = np.array([0.25, 0.5, 0.75])
    ref = np.array([0.25, 0.5, 0.75])
    lo0, hi0 = _bootstrap_intervals(stacks, level, ref, 1.0, 120, 0.10, rng)
    lo1, hi1 = _bootstrap_intervals(stacks, level, ref, 1.0, 120, 0.10,
                                    np.random.default_rng(2),
                                    resid_sd=np.array([0.08, 0.08, 0.08]))
    w0 = float(np.mean([hi0[c] - lo0[c] for c in stacks]))
    w1_ = float(np.mean([hi1[c] - lo1[c] for c in stacks]))
    assert w1_ > w0 * 1.3


# ------------------------------------------- the pace floor must come back down

def test_the_throttle_backoff_decays_after_a_clean_streak():
    """A back-off that never decays is right for a minute and wrong for a day.

    Observed on the 14B arm: a run pacing at 28 calls/min met one throttled patch
    early, ratcheted to the 10 s/call ceiling, and spent the next eleven hours
    there. Only restarting the process cured it.
    """
    from popsim.llm.client import LLMClient, ProviderSpec

    spec = ProviderSpec(name="p", model="m", min_interval_s=2.0)
    c = LLMClient.__new__(LLMClient)
    c._pace = {"p": 8.0}
    c._ok_streak = {}
    for _ in range(LLMClient.PACE_RECOVERY_AFTER - 1):
        c._note_success(spec)
    assert c._pace["p"] == 8.0            # not yet
    c._note_success(spec)
    assert c._pace["p"] == pytest.approx(8.0 * LLMClient.PACE_RECOVERY_FACTOR)
    for _ in range(LLMClient.PACE_RECOVERY_AFTER * 20):
        c._note_success(spec)
    assert c._pace["p"] == pytest.approx(2.0)   # never below the configured floor


def test_a_provider_already_at_its_floor_does_not_drift_below_it():
    from popsim.llm.client import LLMClient, ProviderSpec

    spec = ProviderSpec(name="p", model="m", min_interval_s=2.1)
    c = LLMClient.__new__(LLMClient)
    c._pace = {"p": 2.1}
    c._ok_streak = {}
    for _ in range(LLMClient.PACE_RECOVERY_AFTER * 3):
        c._note_success(spec)
    assert c._pace["p"] == 2.1
