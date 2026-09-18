"""Phase 3 — stat cards, elicitation, the permutation test (checklist 3.1-3.5)."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from popsim.agents.elicit import (
    ElicitationBatch,
    RawElicitation,
    elicit_distribution,
    load_paraphrases,
    parse_histogram,
)
from popsim.agents.statcard import (
    AnchorItem,
    CardContractError,
    StatCard,
    assert_card_contract,
    battery_near_duplicates,
)
from popsim.calibration.crossfit import make_crossfit_plan
from popsim.evalx.gate3 import derangement

# ------------------------------------------------------------- paraphrases

def test_three_paraphrases_are_checked_in():
    p = load_paraphrases()
    assert len(p) >= 3
    assert len({x.id for x in p}) == len(p)


def test_no_paraphrase_uses_roleplay_framing():
    """§M4: neutral statistical framing, never 'speak as'.

    Roleplay is what collapses within-cluster variance (F1) and what drives
    refusals on sensitive items — refusals concentrated on exactly the items
    that discriminate between groups.
    """
    banned = ("you are a ", "imagine you are", "pretend", "as a person", "in character")
    for p in load_paraphrases():
        blob = (p.preamble + " " + p.instruction).lower()
        for phrase in banned:
            assert phrase not in blob, f"{p.name} uses roleplay framing: {phrase!r}"


def test_paraphrases_do_not_point_the_wrong_way():
    """The template puts the question above the instruction."""
    for p in load_paraphrases():
        assert "question below" not in p.instruction.lower(), p.name


# --------------------------------------------------------- card contract

def _card(anchor_ids, cluster_id="c1"):
    return StatCard(
        cluster_id=cluster_id, definition_text="Men, aged 65+", pop_share=0.05,
        n_respondents=800, demo_marginals={},
        anchor_items=[
            AnchorItem(item_id=a, text=f"text {a}", labels=["yes", "no"],
                       hist=[0.6, 0.4], topic="t")
            for a in anchor_ids
        ],
        anchor_fold=0,
    )


def test_a_card_may_not_contain_its_own_target():
    with pytest.raises(CardContractError, match="target item itself"):
        assert_card_contract(_card(["spkath", "natroad"]), "natroad")


def test_a_card_may_not_contain_a_near_duplicate_of_its_target():
    """spkath/colath/libath are one proposition in three venues."""
    nd = battery_near_duplicates(["spkath", "colath", "libath", "natroad"])
    with pytest.raises(CardContractError, match="near-duplicate"):
        assert_card_contract(_card(["colath", "natroad"]), "spkath", nd)


def test_the_spending_battery_is_not_treated_as_duplicates():
    """Foreign aid and highway spending share a stem and are different questions."""
    nd = battery_near_duplicates(["natroad", "nataid", "natenvir"])
    assert nd.get("natroad", set()) == set()
    assert_card_contract(_card(["nataid", "natenvir"]), "natroad", nd)


def test_an_empty_card_is_refused():
    with pytest.raises(CardContractError, match="no anchors"):
        assert_card_contract(_card([]), "natroad")


def test_percentages_on_a_card_sum_to_exactly_100():
    """Largest-remainder rounding: three thirds must not print as 33/33/33."""
    a = AnchorItem(item_id="x", text="t", labels=["a", "b", "c"],
                   hist=[1 / 3, 1 / 3, 1 / 3], topic="t")
    assert sum(a.as_percentages()) == 100


# ------------------------------------------------------------- cross-fit

def test_an_anchor_never_sees_its_own_fold():
    anchors = [f"a{i}" for i in range(12)]
    targets = [f"t{i}" for i in range(6)]
    topics = {i: "x" for i in anchors + targets}
    plan = make_crossfit_plan(anchors, targets, topics, n_folds=3)
    for a in anchors:
        usable = plan.anchors_for(a)
        assert a not in usable
        assert all(plan.fold_of[u] != plan.fold_of[a] for u in usable)


def test_every_fold_is_used():
    anchors = [f"a{i}" for i in range(12)]
    plan = make_crossfit_plan(anchors, [], {i: "x" for i in anchors}, n_folds=3)
    assert set(plan.fold_of.values()) == {0, 1, 2}


# ------------------------------------------------------------- parsing
#
# The response contract is a JSON *object keyed by option label*, never an array.
# Gate 3's smoke run is why: `natroad`'s verbatim wording enumerates its options
# in a different order from its `labels`, and asked for an array the model
# answered in the wording's order (W1 0.349 against truth, 0.033 against truth
# reversed). 19 codebook items carry that conflict. These tests pin the contract.

SPEND = ["too little", "about right", "too much"]
YESNO = ["yes", "no"]


def _keyed(labels, nums):
    import json
    return json.dumps({"percentages": dict(zip(labels, nums))})


def test_a_clean_histogram_parses_and_normalizes():
    h, kind, total = parse_histogram(_keyed(SPEND, [50, 30, 20]), SPEND)
    assert kind is None
    assert h == pytest.approx([0.5, 0.3, 0.2])
    assert total == pytest.approx(100.0)


def test_the_histogram_comes_back_in_codes_order_whatever_order_the_keys_were_in():
    """The whole point of keying: key order must not reach the histogram."""
    shuffled = '{"percentages": {"too much": 20, "too little": 50, "about right": 30}}'
    h, kind, _ = parse_histogram(shuffled, SPEND)
    assert kind is None
    assert h == pytest.approx([0.5, 0.3, 0.2]), "must be in labels order, not reply order"


def test_a_positional_array_is_a_recorded_failure_not_a_histogram():
    """An array cannot say which option each number belongs to.

    Accepting it would reintroduce the natroad reordering on exactly the items
    where it does the most damage, so it is refused and counted instead.
    """
    h, kind, _ = parse_histogram('{"percentages": [50, 30, 20]}', SPEND)
    assert h is None and kind == "positional_response"


def test_label_matching_tolerates_case_and_whitespace_but_not_invention():
    h, kind, _ = parse_histogram(
        '{"percentages": {" Too Little ": 50, "About Right": 30, "too much.": 20}}', SPEND
    )
    assert kind is None and h == pytest.approx([0.5, 0.3, 0.2])

    h, kind, _ = parse_histogram(
        '{"percentages": {"too little": 50, "about right": 30, "loads": 20}}', SPEND
    )
    assert h is None and kind == "label_mismatch"


def test_a_histogram_that_does_not_sum_to_100_is_normalized_and_flagged():
    h, kind, total = parse_histogram(_keyed(SPEND, [50, 40, 30]), SPEND)
    assert kind is None
    assert sum(h) == pytest.approx(1.0)
    assert total == pytest.approx(120.0), "the raw sum must survive for the log"


def test_json_embedded_in_prose_is_recovered():
    text = 'Here is my estimate:\n' + _keyed(YESNO, [60, 40]) + '\nHope that helps.'
    h, kind, _ = parse_histogram(text, YESNO)
    assert kind is None and h == pytest.approx([0.6, 0.4])


def test_a_missing_option_is_never_padded():
    """Padding would assign zero probability to a real option, silently."""
    h, kind, _ = parse_histogram(
        '{"percentages": {"too little": 50, "about right": 50}}', SPEND
    )
    assert h is None and kind == "label_mismatch"


def test_a_refusal_is_not_counted_as_malformed():
    h, kind, _ = parse_histogram("I cannot provide estimates about this group.", SPEND)
    assert h is None and kind == "refusal"


def test_hedging_before_valid_json_is_not_a_refusal():
    text = 'I cannot be certain, but: ' + _keyed(YESNO, [70, 30])
    hist, kind, _ = parse_histogram(text, YESNO)
    assert kind is None, "a hedge that still answers is not a refusal"
    assert hist == pytest.approx([0.7, 0.3])


def test_negative_percentages_are_malformed():
    h, kind, _ = parse_histogram(_keyed(YESNO, [-10, 110]), YESNO)
    assert h is None and kind == "malformed_json"


# ------------------------------------------------------------- elicitation

class _FakeClient:
    """Returns scripted replies; records every prompt it was given."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []
        self.keys = []

    def complete(self, prompt, **kw):
        from popsim.llm.client import LLMResponse
        self.prompts.append(prompt)
        self.keys.append((kw.get("paraphrase_id"), kw.get("repeat_id")))
        text = self.replies[min(len(self.prompts) - 1, len(self.replies) - 1)]
        return LLMResponse(text=text, model="fake", provider="fake")


def _item():
    return {"item_id": "natroad", "text": "spending on highways?",
            "labels": ["too little", "about right", "too much"]}


def test_the_ensemble_makes_one_call_per_draw():
    client = _FakeClient([_keyed(SPEND, [50, 30, 20])])
    recs = elicit_distribution(
        card=_card(["a1"]), item=_item(), client=client,
        paraphrases=load_paraphrases(), n_paraphrase=3, n_repeat=3,
    )
    assert len(recs) == 9
    assert len(client.prompts) == 9
    assert all(r.ok for r in recs)
    assert len(set(client.keys)) == 9, "each draw must be its own cache key"


def test_malformed_output_is_retried_then_marked_failed():
    client = _FakeClient(["not json at all"])
    recs = elicit_distribution(
        card=_card(["a1"]), item=_item(), client=client,
        paraphrases=load_paraphrases(), n_paraphrase=1, n_repeat=1, max_retries=2,
    )
    assert len(client.prompts) == 3, "max_retries=2 means 3 attempts"
    assert recs[0].failure == "malformed_json" and not recs[0].ok


def test_a_retry_is_a_fresh_draw_not_a_cache_replay():
    """Otherwise a cache hit would replay the same malformed text forever."""
    client = _FakeClient(["bad", "bad", _keyed(SPEND, [40, 40, 20])])
    recs = elicit_distribution(
        card=_card(["a1"]), item=_item(), client=client,
        paraphrases=load_paraphrases(), n_paraphrase=1, n_repeat=1, max_retries=2,
    )
    assert recs[0].ok, "the third attempt succeeded"
    assert len({k[1] for k in client.keys}) == 3, "each attempt needs a distinct key"


def test_a_refusal_is_not_retried():
    client = _FakeClient(["I cannot provide that."])
    recs = elicit_distribution(
        card=_card(["a1"]), item=_item(), client=client,
        paraphrases=load_paraphrases(), n_paraphrase=1, n_repeat=1, max_retries=2,
    )
    assert len(client.prompts) == 1, "retrying a refusal only spends quota"
    assert recs[0].failure == "refusal"


def test_refusal_rate_per_item_drives_the_exclusion_rule():
    """Spec §M4: items above 10% refusal are excluded, and it is reported."""
    batch = ElicitationBatch()
    for i in range(10):
        batch.add(RawElicitation(
            cluster_id="c", item_id="touchy", model="m", provider="p",
            paraphrase_id=0, repeat_id=i, hist=None, ok=False,
            failure="refusal" if i < 3 else None,
        ))
    for i in range(10):
        batch.add(RawElicitation(
            cluster_id="c", item_id="calm", model="m", provider="p",
            paraphrase_id=0, repeat_id=i, hist=[0.5, 0.5], ok=True,
        ))
    assert batch.items_over_refusal_threshold(0.10) == ["touchy"]


# ------------------------------------------------------------ permutation

def test_a_derangement_leaves_no_cluster_with_its_own_card():
    """A plain shuffle leaves ~1/e of them holding their own card."""
    rng = np.random.default_rng(0)
    ids = [f"c{i}" for i in range(30)]
    for _ in range(20):
        swap = derangement(ids, rng)
        assert set(swap) == set(ids)
        assert sorted(swap.values()) == sorted(ids)
        assert all(swap[c] != c for c in ids)


# ------------------------------------------------ the gate's own verdict logic

def _result(real, permuted, baseline, noise=0.0273, **kw):
    from popsim.evalx.gate3 import PermutationResult
    return PermutationResult(
        model="m", provider="p", items=["i"], n_clusters=10, n_calls=100,
        real_w1=real, permuted_w1=permuted, baseline_w1=baseline, noise_floor=noise,
        failure_rates={"ok": 1.0}, **kw,
    )


def test_a_model_that_ignores_the_card_is_diagnosed_as_f2():
    r = _result(real=0.28, permuted=0.28, baseline=0.12)
    assert not r.ok
    assert "F2" in r.verdict()


def test_a_gap_smaller_than_the_truths_own_noise_is_not_a_measurement():
    r = _result(real=0.070, permuted=0.090, baseline=0.105, noise=0.0273)
    assert not r.ok, "a 0.02 gap under a 0.027 noise floor is not evidence"
    r2 = _result(real=0.070, permuted=0.110, baseline=0.105, noise=0.0273)
    assert r2.ok


def test_variance_collapse_does_not_fail_the_gate():
    """A flattened model can put permuted below B0a while still reading the card.

    Failing it here would be failing it for F1 — the bug the calibration layer
    exists to repair — on the gate whose job is to detect F2.
    """
    r = _result(real=0.017, permuted=0.094, baseline=0.125)
    assert r.ok, "the real-vs-permuted gap is large; that is the gate"
    assert not r.permuted_beats_baseline
    assert any("variance collapse" in d for d in r.diagnostics())


def test_no_data_is_reported_as_no_data_not_as_a_pass():
    r = _result(real=float("nan"), permuted=float("nan"), baseline=0.12)
    assert not r.ok
    assert "NO DATA" in r.verdict()


def test_high_refusal_items_surface_in_the_diagnostics():
    r = _result(real=0.05, permuted=0.12, baseline=0.10,
                refusal_over_threshold=["homosex", "pray"])
    assert r.ok
    assert any("refusal threshold" in d for d in r.diagnostics())


# ------------------------------------------------- 3.4 contract: option order
#
# The Gate 3 smoke run scored `natroad` at W1 0.349 against its truth and 0.033
# against its truth *reversed*. The cause was not the model: the item's verbatim
# wording — "(... are we spending too much, too little, or about the right
# amount on) Highways and bridges" — enumerates the options in the opposite
# order from `labels`, and a positional response format let the model answer in
# the wording's order while we scored it in the codes' order.
#
# Neither side of that conflict can be edited away. The wording is the
# instrument (§1.4a) and the codes carry the published toplines that Gate 0 is
# built on. So the contract has to be immune to it, and these two tests are what
# say so: one records how widespread the conflict is, the other pins the
# property that makes it harmless.


def test_the_wording_option_order_conflict_is_real_and_sized():
    import yaml

    from popsim.data.codebook import wording_option_order_conflicts

    root = pathlib.Path(__file__).resolve().parent.parent
    codebook = yaml.safe_load(
        (root / "codebooks" / "gss_items.yaml").read_text()
    )["items"]
    conflicts = wording_option_order_conflicts(codebook)

    assert "natroad" in conflicts, "the item that exposed this must still be caught"
    spending = {i for i in conflicts if i.startswith("nat")}
    assert len(spending) >= 15, (
        "the whole nat* spending battery shares one stem and therefore one "
        f"conflict; got {sorted(spending)}"
    )
    assert "polviews" in conflicts, (
        "polviews enumerates its 7-point scale out of order too, and it is the "
        "most heavily used attitudinal variable in the GSS"
    )


def test_a_conflicted_item_is_parsed_correctly_regardless_of_reply_order():
    """The property that makes the conflict harmless.

    A model that answers `natroad` in the *wording's* order — too much first —
    still lands in the right bins, because the keys say which bin is which.
    """
    reply = '{"percentages": {"too much": 10, "too little": 48, "about right": 42}}'
    hist, kind, _ = parse_histogram(reply, SPEND)
    assert kind is None
    assert hist == pytest.approx([0.48, 0.42, 0.10]), (
        "must be scored as 48% 'too little', not 48% in slot 2"
    )


# ---------------------------------------- 3.4 contract: no same-fold anchor
#
# The clause `assert_card_contract` claimed and never checked. `anchors_for`
# returned every anchor for a target, so `target_fold` — chosen in
# `make_crossfit_plan` as the fold densest in same-topic anchors, exactly so
# that excluding it removes likely near duplicates — was computed, written onto
# every card, and enforced nowhere.
#
# What it cost: `natroad`'s excluded fold is 0, `nataid` is in fold 0, and
# `_pick_anchors` breaks ties alphabetically, so `nataid` was the one spending
# anchor on every spending card. It is also the battery's extreme outlier —
# 57.7% "too much" where the battery median is near 10% — and the 7B tracked it
# at r = +0.85 on `natroad` and +0.90 on `natsoc` while correlating *negatively*
# with the truth.


def test_a_target_never_sees_anchors_from_its_own_fold():
    anchors = [f"a{i}" for i in range(12)]
    topics = {i: ("x" if int(i[1:]) % 2 else "y") for i in anchors}
    topics["t1"] = "x"
    plan = make_crossfit_plan(anchors, ["t1"], topics, n_folds=3)

    usable = plan.anchors_for("t1")
    excluded = plan.target_fold["t1"]
    assert usable, "a target must still have anchors left"
    assert all(plan.fold_of[a] != excluded for a in usable), (
        "anchors_for returned an anchor from the target's own excluded fold"
    )
    assert set(usable) < set(anchors), "the fold exclusion must actually remove something"


def test_the_card_contract_rejects_a_same_fold_anchor():
    from popsim.agents.statcard import AnchorItem, CardContractError, StatCard

    card = StatCard(
        cluster_id="c1", definition_text="d", pop_share=0.1, n_respondents=100,
        demo_marginals={}, anchor_fold=1,
        anchor_items=[AnchorItem(item_id="a1", text="q?", labels=["y", "n"],
                                 hist=[0.5, 0.5], topic="x")],
    )
    # a1 sits in the card's own excluded fold
    with pytest.raises(CardContractError, match="own excluded fold"):
        assert_card_contract(card, "t1", {}, fold_of={"a1": 1})
    # and is fine otherwise
    assert_card_contract(card, "t1", {}, fold_of={"a1": 2})


# ------------------------------------------- fail fast, not after 960 calls
#
# Three setup faults were each discovered only by completing a whole run and
# reading the failures afterwards: the positional response contract, nothing
# loading `.env` (401 on every call), and a per-second throttle read as the
# day's allowance. All three are visible in the FIRST response.
#
# The elicitation loop's tolerance is deliberate — a run of thousands of calls
# over days must survive a provider having a bad afternoon — but that same
# tolerance turns a broken setup into a full run of nothing.


def _tiny_setup():
    """Minimal (items, cards, codebook, codes) for driving the permutation test."""
    import pandas as pd

    from popsim.clustering.stats import ClusterStats

    codebook = {"natroad": {"text": "spending on highways?",
                            "labels": ["too little", "about right", "too much"],
                            "codes": [1, 2, 3], "topic": "spending_priorities"}}
    cluster_ids = ["c1", "c2"]
    cards = {("natroad", c): _card(["a1"], cluster_id=c) for c in cluster_ids}
    frame = pd.DataFrame({
        "item_natroad": [1, 2, 3, 1, 2, 3],
        "weight": [1.0] * 6,
    })
    sframe = pd.DataFrame([
        {"cluster_id": c, "item_id": "natroad", "n_eff": 50.0, "weight_sum": 100.0,
         "hist": [0.5, 0.3, 0.2]}
        for c in cluster_ids
    ])
    stats = ClusterStats(frame=sframe, codes={"natroad": [1, 2, 3]})
    return codebook, cluster_ids, cards, frame, stats


def test_a_broken_provider_path_stops_on_the_first_call():
    from popsim.evalx.gate3 import PreflightFailed, run_permutation_test

    codebook, cluster_ids, cards, frame, stats = _tiny_setup()
    client = _FakeClient(["nonsense"] * 500)

    with pytest.raises(PreflightFailed, match="first call failed"):
        run_permutation_test(
            items=["natroad"], cluster_ids=cluster_ids, cards=cards,
            codebook=codebook, stats=stats, frame=frame,
            cluster_of=None, codes={"natroad": [1, 2, 3]},
            client=client, paraphrases=load_paraphrases(),
            n_paraphrase=1, n_repeat=1, progress=False,
        )
    assert len(client.prompts) <= 3, (
        f"must stop after the preflight's retries, not grind on; "
        f"made {len(client.prompts)} calls"
    )


def test_the_preflight_can_be_turned_off_for_the_fake_model_dry_runs():
    """The `refuser` fake exists to prove the gate reports NO DATA, so it must
    still be able to produce a full run of failures on purpose."""
    from popsim.evalx.gate3 import run_permutation_test

    codebook, cluster_ids, cards, frame, stats = _tiny_setup()
    client = _FakeClient(["I cannot provide estimates about this group."] * 500)

    res = run_permutation_test(
        items=["natroad"], cluster_ids=cluster_ids, cards=cards,
        codebook=codebook, stats=stats, frame=frame,
        cluster_of=None, codes={"natroad": [1, 2, 3]},
        client=client, paraphrases=load_paraphrases(),
        n_paraphrase=1, n_repeat=1, progress=False,
        preflight=False, max_consecutive_failures=10_000,
    )
    assert not res.ok
    assert "NO DATA" in res.verdict()
