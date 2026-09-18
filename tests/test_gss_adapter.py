"""GSS adapter contract (checklist 1.1-1.3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from popsim.data.adapters.gss import (
    EXCLUDED_WAVES,
    EXCLUSION_REASONS,
    MISSING_SENTINEL,
    coerce_codes,
    load_gss,
)
from popsim.data.harmonize import (
    DEGREE_LABELS,
    DegreeScaleError,
    age_to_band,
    assert_degree_scale,
    harmonize_demographics,
)

BED_WAVES = [2010, 2012, 2014, 2016, 2018, 2021, 2022]
SMOKE_ITEMS = ["natroad", "spkath", "finrela", "homosex"]


# ------------------------------------------------------- no-microdata tests

def test_2024_is_excluded_and_the_reason_is_recorded():
    """Checklist 1.3: exclude GSS 2024 with a logged reason, asserted in a test."""
    assert 2024 in EXCLUDED_WAVES
    reason = EXCLUSION_REASONS[2024]
    assert "region_7222" in reason and "0%" in reason
    with pytest.raises(ValueError, match="2024"):
        load_gss("unused.dta", ["natroad"], waves=[2022, 2024])


def test_letter_missing_codes_never_become_responses():
    """GSS ships 'd'/'i'/'n'/'s' in the same column as integer responses."""
    s = pd.Series([1, 2, "d", 3, "i", None, "n", 2])
    coded, reserved = coerce_codes(s)
    assert list(coded) == [1, 2, -1, 3, -1, -1, -1, 2]
    assert reserved.tolist()[2] == "d"
    assert reserved.tolist()[4] == "i"
    # A refusal and an inapplicable are both missing, but they are distinguishable:
    # M4 reports refusal rate per item, and conflating the two inflates it.
    assert reserved.notna().sum() == 3


def test_negative_and_out_of_range_codes_are_missing():
    coded, _ = coerce_codes(pd.Series([-1, 0, 1, 99]))
    assert coded.tolist()[0] == MISSING_SENTINEL
    assert coded.tolist()[1] == 0  # 0 is a real code (degree: less than high school)


def test_age_bands_match_the_acs_margin_shape():
    assert age_to_band(17) is None       # not an adult
    assert age_to_band(18) == "18-24"
    assert age_to_band(24) == "18-24"
    assert age_to_band(25) == "25-34"
    assert age_to_band(89) == "65+"
    assert age_to_band(None) is None


def test_degree_scale_is_pinned_to_the_acs_margins():
    """The one coupling that fails silently: raking against the wrong scale."""
    assert list(DEGREE_LABELS) == [0, 1, 2, 3, 4]
    assert_degree_scale("gss_degree_0_4")
    with pytest.raises(DegreeScaleError, match="Regenerate"):
        assert_degree_scale("something_else")


def test_region_falls_back_and_records_which_variable_was_used():
    raw = pd.DataFrame({
        "age": [40, 40, 40],
        "sex": [1, 2, 1],
        "degree": [3, 1, 0],
        "region_7222": [5, np.nan, np.nan],   # division present / absent / absent
        "region": [3, 3, np.nan],             # 4-cat present / present / absent
        "srcbelt": [1, 6, np.nan],
        "mode": [1, 4, np.nan],
    })
    demo = harmonize_demographics(raw)
    assert demo["region_source"].tolist() == ["region_7222", "region4", "none"]
    assert demo["region"].tolist()[:2] == ["south atlantic", "south"]
    assert pd.isna(demo["region"].tolist()[2])
    assert demo["urban"].tolist()[:2] == [True, False]
    assert demo["mode"].tolist()[:2] == ["in-person", "web"]


# ---------------------------------------------------------- microdata tests

@pytest.mark.slow
def test_adapter_loads_the_bed(cfg):
    t = load_gss(cfg.bed_file, SMOKE_ITEMS, waves=BED_WAVES)
    assert t.waves == BED_WAVES
    assert 2024 not in t.waves
    assert t.weight_var == "wtssps"
    assert len(t) > 15_000
    assert (t.frame["weight"] > 0).all()
    assert t.frame["age_band"].notna().all()
    for i in SMOKE_ITEMS:
        assert f"item_{i}" in t.frame.columns


@pytest.mark.slow
def test_pooled_weights_give_every_wave_equal_mass(cfg):
    """Otherwise 2021 (n = 4,032, 87% web) drives the pooled truth."""
    t = load_gss(cfg.bed_file, ["natroad"], waves=BED_WAVES)
    mass = t.frame.groupby("wave")["weight"].sum()
    assert np.allclose(mass.values, mass.values[0], rtol=1e-9)


@pytest.mark.slow
def test_mode_is_carried_and_is_near_collinear_with_wave(cfg):
    """Recorded because NORC flags natroad, spkrac, conlegis, conmedic as mode-sensitive."""
    t = load_gss(cfg.bed_file, ["natroad"], waves=BED_WAVES)
    by_wave = t.frame.groupby("wave")["mode"].apply(lambda s: (s == "web").mean())
    assert by_wave.loc[2018] == 0.0
    assert by_wave.loc[2021] > 0.8
    assert 0.3 < by_wave.loc[2022] < 0.6


@pytest.mark.slow
def test_the_598mb_file_is_never_read_whole(cfg, monkeypatch):
    """usecols is mandatory: the read path must refuse an empty column list."""
    from popsim.data.adapters import gss as gss_mod
    with pytest.raises(ValueError, match="explicit column list"):
        gss_mod.read_gss_columns(cfg.bed_file, [])

    seen = {}
    real = gss_mod.pyreadstat.read_dta

    def spy(path, **kw):
        seen.update(kw)
        return real(path, **kw)

    monkeypatch.setattr(gss_mod.pyreadstat, "read_dta", spy)
    load_gss(cfg.bed_file, ["natroad"], waves=[2022])
    assert seen.get("encoding") == "latin1"
    assert seen.get("usecols"), "read_dta was called without usecols"


# ------------------------------------------------- wording guard (1.4a)

def test_truncated_wording_is_not_mistaken_for_the_instrument():
    """The codebook's Label is a SAS label, capped at 256 bytes.

    Long questions are cut off mid-sentence in the published PDF itself, so a
    label that *looks* like wording can be the question with its end missing.
    That is worse than a paraphrase: it reads as complete.
    """
    from popsim.data.codebook import classify_wording

    # A complete short question.
    assert classify_wording(
        "Which of these statements comes closest to your feelings about "
        "pornography laws?", from_pdf=True)[0] == "verified"

    # The spending/confidence battery shape: preamble in parentheses, then the
    # item stem. Complete despite no terminal punctuation.
    assert classify_wording(
        "(... are we spending too much, too little, or about the right amount "
        "on) Highways and bridges", from_pdf=True)[0] == "verified"

    # SPKATH as the PDF actually prints it — cut off at "rel". It contains a
    # ")" from "(city/town/community)", so a naive paren test would pass it.
    spkath = (
        "There are always some people whose ideas are considered bad or dangerous "
        "by other people. For instance, somebody who is against all churches and "
        "religion . . . If such a person wanted to make a speech in your "
        "(city/town/community) against churches and rel"
    )
    assert classify_wording(spkath, from_pdf=True)[0] == "truncated"

    # Not fielded in 2022 -> the .dta's short variable label, not a question.
    assert classify_wording("Allow homosexual to speak", from_pdf=False)[0] == "fallback_label"


def test_an_unverified_item_cannot_be_elicited_on():
    from popsim.data.codebook import Item, WordingNotVerified

    item = Item(
        item_id="spkhomo", dataset="gss", text="Allow homosexual to speak",
        text_source="dta label", scale_type="binary",
        labels=["yes, allowed to speak", "not allowed"], codes=[1, 2],
        topic="civil_liberties", wording_status="fallback_label",
        wording_note="not instrument wording",
    )
    with pytest.raises(WordingNotVerified, match="blocked"):
        item.assert_elicitable()

    item.wording_status = "verified"
    item.assert_elicitable()


def test_the_shipped_codebook_reports_its_blocked_items(repo_root):
    """The count changes as wording is filled in; the accounting may not."""
    from popsim.data.codebook import Codebook

    cb = Codebook.from_yaml(repo_root / "codebooks" / "gss_items.yaml")
    for item in cb.items.values():
        if item.wording_status != "verified":
            assert item.wording_note, f"{item.item_id} is blocked with no reason recorded"
        else:
            assert item.text_source, f"{item.item_id} is verified with no provenance"
    # Items may be blocked on wording — the pool is wider than the ballots cover.
    # What must hold is that nothing blocked reached the frozen split, in either
    # role: an anchor's question text is rendered into the stat card too.
    import yaml

    split_path = repo_root / "codebooks" / "item_split.frozen.yaml"
    if split_path.exists():
        split = yaml.safe_load(split_path.read_text())
        in_use = set(split["targets"]) | set(split["anchors"]) | set(split["anchor_only_low_snr"])
        blocked = {i.item_id for i in cb.needing_wording()}
        assert not (in_use & blocked), f"split uses items with unverified wording: {in_use & blocked}"


def test_items_not_fielded_in_2022_got_wording_from_the_questionnaire(repo_root):
    """The codebook cannot supply these; only the instrument can."""
    from popsim.data.codebook import Codebook

    cb = Codebook.from_yaml(repo_root / "codebooks" / "gss_items.yaml")
    for item_id in ("spkhomo", "colhomo", "libhomo", "spkmil", "libmil"):
        item = cb[item_id]
        assert item.wording_status == "verified", item.wording_note
        assert "ballot questionnaire" in item.text_source
        assert "?" in item.text, f"{item_id} wording carries no question"
        assert len(item.text) > 60, f"{item_id} wording looks like a label, not a question"


def test_the_tolerance_battery_polarity_is_recorded(repo_root):
    """The 3x5 grid is not coded consistently, and §5.2a's metric is an ordering.

    One cell and one whole row run the other way:
      spk*/col*  the permissive answer is the LOW code
      lib*       "remove"=1, "not remove"=2 -> permissive is the HIGH code
      colcom     "yes, fired"=4             -> permissive is the HIGH code
    An unaligned column inverts that group's ordering and turns a correct model
    into a visibly wrong one.
    """
    from popsim.data.codebook import Codebook

    cb = Codebook.from_yaml(repo_root / "codebooks" / "gss_items.yaml")

    for item_id in ("spkath", "spkcom", "spkhomo", "spkmil", "spkrac"):
        assert cb[item_id].tolerant_code == 1, cb[item_id].labels
        assert cb[item_id].reverse_coded is False

    for item_id in ("colath", "colhomo", "colrac", "colmil"):
        assert cb[item_id].tolerant_code == 4, cb[item_id].labels
        assert cb[item_id].reverse_coded is False

    for item_id in ("libath", "libcom", "libhomo", "libmil", "librac"):
        assert cb[item_id].tolerant_code == 2, cb[item_id].labels
        assert cb[item_id].reverse_coded is True, "the whole lib* row runs the other way"

    # The one cell that differs from its own row.
    assert cb["colcom"].tolerant_code == 5, cb["colcom"].labels
    assert cb["colcom"].reverse_coded is True


def test_polarity_is_only_inferred_inside_the_battery(repo_root):
    from popsim.data.codebook import Codebook, infer_tolerant_code

    cb = Codebook.from_yaml(repo_root / "codebooks" / "gss_items.yaml")
    for item_id in ("natroad", "finrela", "pray", "homosex", "polviews"):
        assert cb[item_id].tolerant_code is None
        assert cb[item_id].reverse_coded is False
    assert infer_tolerant_code("natroad", [1, 2, 3], ["too little", "about right", "too much"]) is None
