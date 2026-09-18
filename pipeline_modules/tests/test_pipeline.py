"""
Regression tests for the B17 pipeline.

These are contract tests, not smoke tests. Each one pins a property that was
either broken before or is load-bearing for the scientific claim, so a future
refactor that silently reintroduces the bug fails here.

    pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

CODES_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODES_DIR))

from shared.budget_guard import BudgetExceededError, BudgetGuard  # noqa: E402
from shared.cache import PromptCache  # noqa: E402
from shared.llm_client import LLMError, get_llm_client, is_mock, model_info  # noqa: E402
from shared.paths import load_config  # noqa: E402
from M3_statcards.builder import StatCardBuilder  # noqa: E402
from M4_elicitation.schemas import ElicitedDistributionOutput  # noqa: E402
from M5_calibration.crossfit import (  # noqa: E402
    CalibrationPair,
    CrossFittedCalibrator,
    make_crossfit_plan,
)
from M6_aggregation.scope import SegmentResolver, aggregate_segment  # noqa: E402
from M9_evaluation.metrics import (  # noqa: E402
    evaluate_cluster_predictions,
    wasserstein_1_distance,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def builder(cfg):
    if not cfg.tree_path.exists():
        pytest.skip("preprocessing outputs not present")
    return StatCardBuilder(
        tree_path=cfg.tree_path,
        stats_path=cfg.stats_path,
        codebook_path=cfg.codebook_path,
        split_path=cfg.split_path,
        pop_stats_path=cfg.pop_stats_path,
        individual_table_path=cfg.individual_table_path,
        granularity=cfg.granularity,
    )


# ======================================================================
# transport layer
# ======================================================================
def test_real_model_never_silently_becomes_a_mock(monkeypatch):
    """
    The original bug: get_llm_client() returned MockLLMClient for every model id,
    so a run that believed it was calling Claude was generating Dirichlet noise.
    Asking for a real model must either work or fail loudly.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(LLMError):
        get_llm_client("claude-haiku-4-5")

    assert not is_mock("claude-haiku-4-5")
    assert is_mock("mock-model")
    assert model_info("claude-haiku-4-5")["provider"] == "anthropic"


def test_unknown_model_is_rejected():
    with pytest.raises(KeyError):
        model_info("gpt-9-ultra")


def test_ensemble_members_are_distinct_draws(tmp_path):
    """
    The original bug: the cache key omitted the sample index, so repeats 2 and 3
    of each paraphrase were cache hits of repeat 1. A '3x3 ensemble' was really
    3 draws with zero within-paraphrase variance, and the bootstrap over ensemble
    members was resampling three identical points.
    """
    cache = PromptCache(db_path=tmp_path / "c.sqlite")
    client = get_llm_client(
        "mock-model", cache=cache, budget_guard=BudgetGuard(state_file=tmp_path / "b.json")
    )
    kwargs = dict(
        system_prompt="sys",
        user_prompt='Q?\nOptions: ["a", "b", "c"]',
        response_schema=ElicitedDistributionOutput,
        temperature=0.7,
    )
    draws = [tuple(client.generate_structured(**kwargs, sample_id=i).parsed["probabilities"])
             for i in range(9)]

    assert len(set(draws)) > 1, "ensemble collapsed to identical draws"
    assert np.asarray(draws).std(axis=0).mean() > 0.0

    # Same sample_id must still hit the cache -- resumability depends on it.
    first = client.generate_structured(**kwargs, sample_id=0)
    assert first.is_cached


def test_budget_ceiling_stops_before_spending(tmp_path):
    guard = BudgetGuard(max_usd=0.01, state_file=tmp_path / "b.json")
    guard.total_usd_spent = 0.0099
    with pytest.raises(BudgetExceededError):
        guard.preflight("claude-sonnet-5", est_prompt_tokens=100_000)


def test_pricing_comes_from_the_registry():
    guard = BudgetGuard()
    # Haiku 4.5 is $1/1M in, $5/1M out.
    assert guard.calculate_cost("claude-haiku-4-5", 1_000_000, 0) == pytest.approx(1.00)
    assert guard.calculate_cost("claude-haiku-4-5", 0, 1_000_000) == pytest.approx(5.00)


# ======================================================================
# leakage contract (M3)
# ======================================================================
def test_card_never_shows_its_own_target(builder):
    anchors = [i for i, r in builder.split_dict.items() if r.get("role_standard") == "anchor"]
    target = anchors[0]
    card = builder.build_stat_card(builder.leaf_ids[0], target_item_id=target)
    assert target not in {a.item_id for a in card.anchor_items}


def test_card_excludes_same_fold_anchors(builder):
    """
    Without fold exclusion the calibrator is trained on predictions made from cards
    that displayed the answer, learns 'the model is nearly perfect', and becomes a
    no-op that inflates the reported score.
    """
    anchors = sorted(
        i for i, r in builder.split_dict.items() if r.get("role_standard") == "anchor"
    )[:30]
    topics = {i: builder.codebook[i].get("topic", "general") for i in anchors}
    plan = make_crossfit_plan(anchors, topics, n_folds=3, seed=1)

    target = anchors[0]
    fold = plan[target]
    card = builder.build_stat_card(builder.leaf_ids[0], target_item_id=target, crossfit_plan=plan)
    shown = {a.item_id for a in card.anchor_items}

    assert not {i for i in shown if plan.get(i) == fold}, "same-fold anchor leaked onto the card"
    builder.verify_card_contract(card, target, plan)


def test_card_excludes_near_duplicates(builder):
    anchors = [i for i, r in builder.split_dict.items() if r.get("role_standard") == "anchor"]
    for target in anchors[:40]:
        dups = builder.near_duplicates(target)
        if not dups:
            continue
        card = builder.build_stat_card(builder.leaf_ids[0], target_item_id=target)
        assert not (dups & {a.item_id for a in card.anchor_items})
        return
    pytest.skip("no near-duplicate pairs in this codebook")


def test_crossfit_folds_are_balanced_and_cover_every_anchor():
    items = [f"item{i}" for i in range(60)]
    topics = {i: f"t{n % 7}" for n, i in enumerate(items)}
    plan = make_crossfit_plan(items, topics, n_folds=3, seed=7)

    assert set(plan) == set(items)
    sizes = np.bincount(list(plan.values()), minlength=3)
    assert sizes.max() - sizes.min() <= 3, f"folds badly unbalanced: {sizes}"


# ======================================================================
# calibration (M5)
# ======================================================================
def test_calibrator_restores_collapsed_variance():
    """
    The mechanism's whole purpose: raw histograms that are too peaked should come
    out with dispersion closer to the truth.
    """
    rng = np.random.RandomState(0)
    pairs = []
    for c in range(6):
        for k in range(12):
            true = rng.dirichlet([2.0, 2.0, 2.0, 2.0])
            # Simulate collapse: sharpen the true histogram hard.
            peaked = true**3
            peaked /= peaked.sum()
            pairs.append(
                CalibrationPair(f"c{c}", f"a{k}", k % 3, list(peaked), list(true), 60.0)
            )

    cal = CrossFittedCalibrator(tau=100.0)
    report = cal.fit(pairs)
    assert report["n_pairs"] == len(pairs)
    assert report["raw_variance_ratio_before_calibration"] < 1.0  # collapse present

    support = np.linspace(0, 1, 4)

    def sd(p):
        p = np.asarray(p, dtype=float)
        m = (p * support).sum()
        return float(np.sqrt((p * (support - m) ** 2).sum()))

    before, after, truth = [], [], []
    for p in pairs[:40]:
        before.append(sd(p.raw_probs))
        after.append(sd(cal.calibrate(p.cluster_id, p.raw_probs)))
        truth.append(sd(p.true_probs))

    err_before = abs(np.mean(before) - np.mean(truth))
    err_after = abs(np.mean(after) - np.mean(truth))
    assert err_after < err_before, "calibration did not move dispersion toward the truth"


def test_calibrate_before_fit_raises():
    with pytest.raises(RuntimeError):
        CrossFittedCalibrator().calibrate("c0", [0.25] * 4)


# ======================================================================
# metrics (M9)
# ======================================================================
def test_w1_is_zero_for_identical_and_max_for_opposite():
    assert wasserstein_1_distance([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)
    assert wasserstein_1_distance([1, 0, 0, 0], [0, 0, 0, 1]) == pytest.approx(1.0)


def test_between_cluster_sd_ratio_detects_collapse():
    """
    The F2 detector. A system that emits one histogram for every subgroup must
    score zero here even when its W1 looks respectable.
    """
    truth = [[0.7, 0.3], [0.5, 0.5], [0.3, 0.7], [0.4, 0.6], [0.6, 0.4]]
    collapsed = [[0.5, 0.5]] * 5
    m = evaluate_cluster_predictions(collapsed, truth)
    assert m["between_cluster_sd_ratio"] == pytest.approx(0.0, abs=1e-6)

    m2 = evaluate_cluster_predictions(truth, truth)
    assert m2["between_cluster_sd_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert m2["weighted_w1"] == pytest.approx(0.0, abs=1e-9)


def test_metrics_reject_mismatched_scale_lengths():
    with pytest.raises(ValueError):
        wasserstein_1_distance([0.5, 0.5], [0.3, 0.3, 0.4])


# ======================================================================
# segment scoping (M6)
# ======================================================================
def test_region_segment_covers_plausible_population_share(cfg):
    if not cfg.individual_table_path.exists():
        pytest.skip("individual table not present")
    resolver = SegmentResolver(cfg.individual_table_path, granularity=cfg.granularity)

    shares = {}
    for region in resolver.available_values("region"):
        seg = resolver.resolve({"region": region})
        shares[region] = seg.pop_share
        assert seg.weights.sum() == pytest.approx(1.0)
        assert len(seg) > 0

    assert sum(shares.values()) == pytest.approx(1.0, abs=0.02), shares


def test_partial_membership_is_fractional(cfg):
    """
    A coarse leaf can span both sexes. Treating it as wholly in or wholly out of
    a female-only segment biases the answer; membership must be fractional.
    """
    if not cfg.individual_table_path.exists():
        pytest.skip("individual table not present")
    resolver = SegmentResolver(cfg.individual_table_path, granularity=cfg.granularity)
    seg = resolver.resolve({"sex": "female"})
    assert any(0.0 < m < 0.999 for m in seg.membership.values()), (
        "no partial cluster found; membership is being treated as all-or-nothing"
    )
    assert 0.4 < seg.pop_share < 0.6


def test_aggregate_segment_returns_a_distribution(cfg):
    if not cfg.individual_table_path.exists():
        pytest.skip("individual table not present")
    resolver = SegmentResolver(cfg.individual_table_path, granularity=cfg.granularity)
    seg = resolver.resolve({"region": "south"})
    dists = {cid: [0.25, 0.25, 0.25, 0.25] for cid in seg.cluster_ids}
    out = aggregate_segment(dists, seg)
    assert out.sum() == pytest.approx(1.0)
    assert len(out) == 4
