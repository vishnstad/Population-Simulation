"""Phase 2 — partition, stats, noise floor, split (checklist 2.1-2.6)."""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from popsim.clustering.partition import build_cluster_tree
from popsim.clustering.pooling import shrinkage_weight
from popsim.clustering.stats import compute_cluster_stats, kish_n_eff, weighted_histogram
from popsim.evalx.metrics import js_divergence, w1
from popsim.evalx.split import SplitFrozenError, assert_frozen, make_split

# ------------------------------------------------------------------ metrics

def test_w1_is_zero_for_identical_histograms():
    h = np.array([0.2, 0.3, 0.5])
    assert w1(h, h) == pytest.approx(0.0)


def test_w1_on_the_normalized_scale_maxes_at_one():
    """Mass entirely at opposite ends of the scale is distance 1, whatever k."""
    for k in (2, 3, 5, 7):
        lo = np.zeros(k); lo[0] = 1.0
        hi = np.zeros(k); hi[-1] = 1.0
        assert w1(lo, hi) == pytest.approx(1.0), k


def test_w1_respects_ordinal_distance():
    """Unlike JS, W1 must know that option 1 is closer to 2 than to 5."""
    a = np.array([1.0, 0, 0, 0, 0])
    near = np.array([0, 1.0, 0, 0, 0])
    far = np.array([0, 0, 0, 0, 1.0])
    assert w1(a, near) < w1(a, far)
    # JS cannot tell them apart; that is exactly why W1 is the primary metric.
    assert js_divergence(a, near) == pytest.approx(js_divergence(a, far))


def test_w1_normalizes_unnormalized_input():
    assert w1(np.array([2.0, 2.0]), np.array([0.5, 0.5])) == pytest.approx(0.0)


# -------------------------------------------------------------------- stats

def test_kish_n_eff_equals_n_for_equal_weights():
    assert kish_n_eff(np.ones(100)) == pytest.approx(100.0)


def test_kish_n_eff_falls_when_weights_are_uneven():
    """The reason n_eff is recorded instead of raw n: pooled waves vary weights."""
    w = np.array([10.0] + [1.0] * 99)
    assert kish_n_eff(w) < 100.0


def test_weighted_histogram_uses_weights_not_counts():
    values = np.array([1, 1, 2])
    weights = np.array([1.0, 1.0, 8.0])
    h = weighted_histogram(values, weights, [1, 2])
    assert h == pytest.approx([0.2, 0.8])


def test_weighted_histogram_of_nothing_is_all_zero():
    h = weighted_histogram(np.array([]), np.array([]), [1, 2, 3])
    assert h.sum() == 0.0


# ---------------------------------------------------------------- partition

def _toy_frame(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    age = rng.choice(["18-24", "25-34", "35-44", "45-54", "55-64", "65+"], n)
    degree = rng.choice([0, 1, 2, 3, 4], n)
    sex = rng.choice(["male", "female"], n)
    # A response that genuinely depends on degree, so a supervised split has
    # something real to find.
    p = 0.15 + 0.15 * degree
    resp = np.where(rng.random(n) < p, 2, 1)
    return pd.DataFrame({
        "age_band": age, "degree": degree, "sex": sex,
        "weight": np.ones(n), "item_x": resp.astype("int16"),
    })


def test_the_tree_never_creates_a_cell_below_min_cell():
    """The whole point of going top-down (2.1).

    The spec merges cells after building the full cross, which needs a distance
    between histograms estimated from ~1.6 people — undefined, not noisy. A
    top-down tree checks the constraint before splitting, so the situation
    never arises.
    """
    frame = _toy_frame()
    tree = build_cluster_tree(frame, axes=["age_band", "degree", "sex"],
                              item_codes={"x": [1, 2]}, min_cell=100, k_target=20)
    assert tree.k >= 2
    for leaf in tree.leaves():
        assert leaf.n >= 100, f"{leaf.cluster_id} has {leaf.n} < min_cell"


def test_the_tree_stops_at_k_target():
    frame = _toy_frame()
    tree = build_cluster_tree(frame, axes=["age_band", "degree", "sex"],
                              item_codes={"x": [1, 2]}, min_cell=20, k_target=8)
    assert tree.k <= 8


def test_every_leaf_is_describable():
    """Spec §2.2 rejects embedding clustering because clusters must be renderable."""
    frame = _toy_frame()
    tree = build_cluster_tree(frame, axes=["age_band", "degree", "sex"],
                              item_codes={"x": [1, 2]}, min_cell=100, k_target=12)
    for leaf in tree.leaves():
        assert leaf.definition_text
        assert leaf.definition
        assert leaf.pop_share > 0


def test_assignment_is_a_partition():
    frame = _toy_frame()
    tree = build_cluster_tree(frame, axes=["age_band", "degree", "sex"],
                              item_codes={"x": [1, 2]}, min_cell=100, k_target=12)
    assigned = tree.assign(frame)
    counts = assigned.value_counts()
    assert counts.sum() == len(frame), "every respondent lands in exactly one leaf"
    assert set(counts.index) == {leaf.cluster_id for leaf in tree.leaves()}


def test_ordered_axes_split_contiguously():
    """A leaf must be a range of ages, not an arbitrary set of them."""
    from popsim.clustering.partition import ORDERED_AXES

    frame = _toy_frame()
    tree = build_cluster_tree(frame, axes=["age_band", "degree", "sex"],
                              item_codes={"x": [1, 2]}, min_cell=60, k_target=20)
    order = ORDERED_AXES["age_band"]
    for leaf in tree.leaves():
        vals = leaf.definition.get("age_band")
        if not vals or len(vals) < 2:
            continue
        idx = sorted(order.index(v) for v in vals)
        assert idx == list(range(idx[0], idx[-1] + 1)), f"{leaf.cluster_id} is non-contiguous"


def test_cluster_stats_histograms_sum_to_one():
    frame = _toy_frame()
    tree = build_cluster_tree(frame, axes=["degree"], item_codes={"x": [1, 2]},
                              min_cell=100, k_target=4)
    stats = compute_cluster_stats(frame, tree.assign(frame), ["x"], {"x": [1, 2]})
    assert len(stats.frame) > 0
    for h in stats.frame["hist"]:
        assert sum(h) == pytest.approx(1.0)


# ------------------------------------------------------------------ pooling

def test_shrinkage_moves_thin_cells_toward_the_parent():
    assert shrinkage_weight(0, tau=100) == 0.0
    assert shrinkage_weight(100, tau=100) == pytest.approx(0.5)
    assert shrinkage_weight(900, tau=100) == pytest.approx(0.9)
    assert shrinkage_weight(1e9, tau=100) > 0.999


# -------------------------------------------------------------------- split

def _split_kwargs(**over):
    items = [f"i{n}" for n in range(60)]
    base: dict = {
        "snr": {i: 3.0 - 0.02 * n for n, i in enumerate(items)},
        "topics": {i: f"t{n % 6}" for n, i in enumerate(items)},
        "roles": dict.fromkeys(items, "unassigned"),
        "wording_ok": dict.fromkeys(items, True),
        "leakage_resistant": set(items[:10]),
        "famous": set(items[50:]),
        "n_targets": 20,
        "n_anchors": 15,
        "n_anchor_only_low_snr": 5,
        "min_snr": 1.5,
    }
    base.update(over)
    return base


def test_targets_and_anchors_are_disjoint():
    s = make_split(**_split_kwargs())
    assert not set(s.targets) & set(s.anchors)
    assert not set(s.targets) & set(s.anchor_only_low_snr)


def test_the_leakage_resistant_set_is_seeded_into_targets():
    """§1.5 reports headline numbers on this subset, so it must sit inside targets."""
    kw = _split_kwargs()
    s = make_split(**kw)
    assert kw["leakage_resistant"] <= set(s.targets)


def test_sanity_and_excluded_items_never_become_targets():
    kw = _split_kwargs()
    kw["roles"]["i0"] = "sanity"
    kw["roles"]["i1"] = "excluded"
    s = make_split(**kw)
    assert "i0" not in s.targets and "i0" not in s.anchors
    assert "i1" not in s.targets and "i1" not in s.anchors
    assert "sanity" in s.ineligible["i0"]


def test_unverified_wording_blocks_the_anchor_role_too():
    """An anchor's question text is rendered into the stat card."""
    kw = _split_kwargs()
    kw["wording_ok"]["i5"] = False
    s = make_split(**kw)
    assert "i5" not in s.targets
    assert "i5" not in s.anchors, "a truncated anchor corrupts the card like a target would"
    assert "wording" in s.ineligible["i5"]


def test_targets_are_topic_stratified():
    s = make_split(**_split_kwargs())
    from collections import Counter
    spread = Counter(s.topics[i] for i in s.targets)
    assert len(spread) >= 4, f"targets collapsed onto {len(spread)} topics: {spread}"


def test_low_snr_items_can_still_be_anchor_context():
    kw = _split_kwargs()
    kw["snr"]["i59"] = 0.4
    s = make_split(**kw)
    assert "i59" not in s.targets
    assert "i59" in s.anchor_only_low_snr


def test_a_regenerated_split_that_differs_is_refused(tmp_path):
    """Re-deriving the split after seeing results is how held-out stops meaning held-out."""
    s = make_split(**_split_kwargs())
    path = tmp_path / "split.yaml"
    s.to_yaml(path)
    assert_frozen(path, s)

    shifted = make_split(**_split_kwargs(n_targets=22))
    with pytest.raises(SplitFrozenError, match="frozen"):
        assert_frozen(path, shifted)


def test_load_or_create_never_overwrites(tmp_path):
    from popsim.evalx.split import load_or_create

    path = tmp_path / "split.yaml"
    first = load_or_create(path, **_split_kwargs())
    second = load_or_create(path, **_split_kwargs(n_targets=25))
    assert second.targets == first.targets, "the frozen file must win"


# ------------------------------------------------- 2.4 split amendments
#
# "Persist the split — it must never be regenerated." It is not: an amendment is
# a separate file layered over the frozen one at load time, so the frozen
# artifact stays byte-identical. What the freeze protects is WHAT THE CLAIM IS
# MEASURED ON, so the invariants are: targets never change, amendments only add
# anchors, and only from the unassigned pool.


def test_the_shipped_amendment_adds_anchors_and_leaves_targets_alone():
    import yaml

    from popsim.evalx.split import load_effective_split

    root = pathlib.Path(__file__).resolve().parent.parent
    frozen_path = root / "codebooks" / "item_split.frozen.yaml"
    frozen = yaml.safe_load(frozen_path.read_text())
    eff, applied = load_effective_split(frozen_path)

    assert eff["targets"] == frozen["targets"], "an amendment must never touch targets"
    assert set(frozen["anchors"]) < set(eff["anchors"]), "amendment 1 must add anchors"
    assert applied, "the shipped amendment must be picked up automatically"

    cb = yaml.safe_load((root / "codebooks" / "gss_items.yaml").read_text())["items"]
    added = set(eff["anchors"]) - set(frozen["anchors"])
    assert all(cb[i]["topic"] == "spending_priorities" for i in added), (
        "amendment 1 is the spending battery only; the rule cannot serve the others"
    )
    spending = [a for a in eff["anchors"] if cb[a]["topic"] == "spending_priorities"]
    assert len(spending) >= 3, "the stated rule is >=3 anchors per topic with targets"


def test_an_amendment_may_never_promote_a_target(tmp_path):
    """The one thing the freeze exists to prevent."""
    import yaml

    from popsim.evalx.split import load_effective_split

    frozen = tmp_path / "item_split.frozen.yaml"
    frozen.write_text(yaml.safe_dump({"targets": ["t1", "t2"], "anchors": ["a1"]}))
    bad = tmp_path / "item_split.amendment_9.yaml"
    bad.write_text(yaml.safe_dump({"amendment": 9, "add_anchors": ["t1"]}))

    with pytest.raises(ValueError, match="TARGET"):
        load_effective_split(frozen, amendments=[bad])


def test_an_amendment_may_not_remove_anchors(tmp_path):
    """Removing one would change the cards every earlier run was scored on."""
    import yaml

    from popsim.evalx.split import load_effective_split

    frozen = tmp_path / "item_split.frozen.yaml"
    frozen.write_text(yaml.safe_dump({"targets": ["t1"], "anchors": ["a1", "a2"]}))
    bad = tmp_path / "item_split.amendment_9.yaml"
    bad.write_text(yaml.safe_dump({"amendment": 9, "remove_anchors": ["a1"]}))

    with pytest.raises(ValueError, match="additive only"):
        load_effective_split(frozen, amendments=[bad])
