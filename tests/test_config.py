"""The config loader must refuse a config that has drifted back to the spec."""

from __future__ import annotations

import copy

import pytest
import yaml

from popsim.config import Config, ConfigError


def _mutated(main_config_path, **dotted):
    raw = yaml.safe_load(main_config_path.read_text())
    for k, v in dotted.items():
        parts = k.split(".")
        cur = raw
        for p in parts[:-1]:
            cur = cur[p]
        cur[parts[-1]] = v
    return Config(raw=raw, source_path=main_config_path,
                  repo_root=main_config_path.parent.parent)


def test_main_config_is_valid(cfg):
    assert cfg["partition.min_cell"] == 60
    assert cfg["partition.k_target"] == 56
    assert cfg["llm.budget_cap_usd"] == 0.0


@pytest.mark.parametrize(
    "override, needle",
    [
        ({"partition.min_cell": 40}, "min_cell"),          # the spec's value
        ({"partition.k_target": 150}, "k_target"),         # the spec's value
        ({"llm.budget_cap_usd": 1200}, "budget_cap_usd"),  # the spec's value
        ({"item_split.n_targets": 29}, "n_targets"),       # what 60/40 would leave
    ],
)
def test_spec_stale_parameters_are_rejected(main_config_path, override, needle):
    with pytest.raises(ConfigError, match=needle):
        _mutated(main_config_path, **override).validate()


def test_geography_in_the_partition_is_rejected(main_config_path):
    c = _mutated(main_config_path, **{"partition.axes": ["age_band", "degree", "sex", "urban"]})
    with pytest.raises(ConfigError, match="geography"):
        c.validate()


def test_gss_2024_cannot_enter_the_bed(main_config_path):
    c = _mutated(main_config_path, **{"bed.waves": [2016, 2018, 2021, 2022, 2024]})
    with pytest.raises(ConfigError, match="2024"):
        c.validate()


def test_pass_marks_must_be_a_real_20pc_improvement(main_config_path):
    raw = yaml.safe_load(main_config_path.read_text())
    raw["evaluation"]["pass_marks"]["all_targets"]["pass_w1"] = 0.0690  # a token 0.7% win
    with pytest.raises(ConfigError, match="20% improvement"):
        Config(raw=raw, source_path=main_config_path,
               repo_root=main_config_path.parent.parent).validate()


def test_pass_marks_absent_is_rejected(main_config_path):
    with pytest.raises(ConfigError, match="pre-register|pass_marks"):
        _mutated(main_config_path, **{"evaluation.pass_marks": {}}).validate()


def test_checkpointing_cannot_be_turned_off(main_config_path):
    with pytest.raises(ConfigError, match="checkpoint_every_call"):
        _mutated(main_config_path, **{"llm.checkpoint_every_call": False}).validate()


def test_sanity_items_are_pinned(main_config_path):
    with pytest.raises(ConfigError, match="finrela"):
        _mutated(main_config_path, **{"item_split.sanity_items": ["satfin"]}).validate()


def test_ensemble_profile_resolves(cfg):
    assert cfg.n_ensemble == 1  # dev profile
    raw = copy.deepcopy(cfg.raw)
    raw["elicitation"]["active_profile"] = "headline"
    c = Config(raw=raw, source_path=cfg.source_path, repo_root=cfg.repo_root)
    assert c.n_ensemble == 9  # 3 paraphrases x 3 repeats


# ------------------------------------------------- provider override guards
#
# Both of these were reachable footguns on the §7.2 rung-1 switch to the
# frontier arm, where a wasted run costs a day of a rate-capped free tier.


def test_a_string_provider_list_is_refused_with_the_right_syntax(main_config_path):
    """`--set llm.active_providers=[mistral]` is not JSON, so it arrives as a
    string and the router would iterate it one character at a time."""
    with pytest.raises(ConfigError) as exc:
        _mutated(main_config_path, **{"llm.active_providers": "[mistral]"}).validate()
    msg = str(exc.value)
    assert "not a list" in msg
    assert '["mistral"]' in msg, "the error must show the working syntax"


def test_an_unconfigured_provider_name_is_refused(main_config_path):
    with pytest.raises(ConfigError, match="not in llm.providers"):
        _mutated(main_config_path, **{"llm.active_providers": ["mistrl"]}).validate()


def test_the_shipped_mistral_tag_is_pinned(cfg):
    """Pinned 16 Sep 2026 to `mistral-small-2603`. If someone reaches for
    `-latest` again, the guard below is what catches it — but the shipped config
    should not need catching."""
    m = next(p for p in cfg["llm.providers"] if p["name"] == "mistral")
    assert not str(m["model"]).endswith(("-latest", ":latest")), (
        "the primary provider's model must be a dated version, not an alias"
    )


def test_a_floating_alias_is_refused_only_when_that_provider_is_active(
    cfg, main_config_path
):
    """cache.py keys on the model string, so an alias being repointed mid-run
    would mix two sets of weights into one ensemble and report the difference as
    elicitation variance. Only enforced for providers actually in use, so the
    parked arms keep their aliases as documentation.

    The floating tag is constructed here rather than read from the config: the
    shipped config is pinned, and a test that depended on it shipping an alias
    would start passing for the wrong reason the moment someone fixed it.
    """
    providers = copy.deepcopy(cfg["llm.providers"])
    for prov in providers:
        if prov["name"] == "mistral":
            prov["model"] = "mistral-small-latest"

    # inert while only ollama is active
    _mutated(main_config_path, **{"llm.providers": providers}).validate()

    with pytest.raises(ConfigError, match="floating alias"):
        _mutated(
            main_config_path,
            **{"llm.providers": providers, "llm.active_providers": ["mistral"]},
        ).validate()
