"""Config loader (checklist 0.2).

One YAML fully specifies a run. Loading a config for a real run snapshots the
*resolved* config into ``runs/<run_id>__<timestamp>/config.snapshot.yaml`` so a
result can always be traced back to the exact parameters that produced it.

Values here are authoritative over ``B17_architecture_spec.md``'s appendix. The
loader enforces the handful of invariants the checklist is explicit about, so a
config that has silently drifted back toward the spec's numbers fails loudly at
load time rather than quietly at analysis time.
"""

from __future__ import annotations

import copy
import datetime as _dt
import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = ["Config", "ConfigError", "load_config", "snapshot_run"]


class ConfigError(ValueError):
    """A config that would produce a misleading or unreproducible run."""


# Invariants the checklist fixes by measurement. Violating one of these is not a
# tuning choice; it is a regression toward the spec's stale numbers.
_INVARIANTS: list[tuple[str, Any, str]] = [
    ("partition.min_cell", 60, "checklist §1.2: at min_cell 40 the truth is noisier than the effect"),
    ("partition.k_target", 56, "checklist §1.2: GSS supports ~53 leaves at the median item, not 150"),
    ("partition.min_item_snr", 1.5, "checklist Layer 1: SNR gate"),
    ("item_split.n_targets", 40, "checklist §1.2: the §1.4 claim needs >= 40 targets"),
    ("llm.budget_cap_usd", 0.0, "checklist §0.7: the project runs on free tiers; the constraint is calls/day"),
]

#: The one invariant the checklist itself asks to be swept, in §7.3: "K in
#: {18, 36, 56, 100}". A sweep arm is a different run, not a regression toward
#: the spec's stale 150, so those values are admitted — and only those, so a
#: typo still fails loudly. Every other invariant holds regardless.
_K_SWEEP_VALUES = {18, 36, 56, 100}

_FORBIDDEN_PARTITION_AXES = {
    "region", "region_7222", "urban", "srcbelt", "xnorcsiz", "division",
}


def _dig(d: dict, dotted: str, default: Any = None) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


@dataclass
class Config:
    """A loaded, validated run config."""

    raw: dict[str, Any]
    source_path: Path
    repo_root: Path
    run_dir: Path | None = None
    _overrides: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- access
    def __getitem__(self, dotted: str) -> Any:
        sentinel = object()
        val = _dig(self.raw, dotted, sentinel)
        if val is sentinel:
            raise KeyError(f"config key not found: {dotted!r} (in {self.source_path})")
        return val

    def get(self, dotted: str, default: Any = None) -> Any:
        return _dig(self.raw, dotted, default)

    # ------------------------------------------------------------ resolution
    @property
    def data_root(self) -> Path:
        return (self.repo_root / self["paths.data_root"]).resolve()

    def data_path(self, relative: str) -> Path:
        """Resolve a path under ``data_root``, asserting it exists."""
        p = self.data_root / relative
        if not p.exists():
            raise ConfigError(
                f"data file missing: {p}\n"
                f"  (data_root={self.data_root}, relative={relative!r})"
            )
        return p

    @property
    def bed_file(self) -> Path:
        return self.data_path(self["bed.file"])

    @property
    def ensemble(self) -> dict[str, int]:
        profile = self["elicitation.active_profile"]
        profiles = self["elicitation.ensemble_profiles"]
        if profile not in profiles:
            raise ConfigError(
                f"unknown ensemble profile {profile!r}; have {sorted(profiles)}"
            )
        return dict(profiles[profile])

    @property
    def n_ensemble(self) -> int:
        e = self.ensemble
        return int(e["n_paraphrase"]) * int(e["n_repeat"])

    # ------------------------------------------------------------ validation
    def validate(self) -> None:
        errs: list[str] = []

        for dotted, expected, why in _INVARIANTS:
            sentinel = object()
            got = _dig(self.raw, dotted, sentinel)
            if got is sentinel:
                errs.append(f"{dotted}: missing (required; {why})")
            elif got != expected:
                if dotted == "partition.k_target" and got in _K_SWEEP_VALUES:
                    continue           # §7.3's own sweep
                errs.append(
                    f"{dotted}: got {got!r}, checklist fixes it at {expected!r} — {why}"
                )

        axes = set(self.get("partition.axes", []))
        bad = axes & _FORBIDDEN_PARTITION_AXES
        if bad:
            errs.append(
                f"partition.axes contains geography {sorted(bad)}. Measured: geography in "
                "the partition drops usable items from 74 to 35. It belongs in the stat "
                "card as a marginal (checklist §1.2)."
            )

        if 2024 in set(self.get("bed.waves", [])):
            errs.append(
                "bed.waves includes 2024. GSS 2024 has region_7222 and srcbelt 0% "
                "populated -> complete-demo n = 0 (checklist 1.3)."
            )
        if 2024 not in set(self.get("bed.exclude_waves", [])):
            errs.append("bed.exclude_waves must list 2024 with a logged reason (checklist 1.3).")
        if not str(self.get("bed.exclude_reason", "")).strip():
            errs.append("bed.exclude_reason is empty; the exclusion must be logged, not silent.")

        # Pass marks must exist before any elicitation run, and must actually be
        # a tightening of the baseline by >= 20%.
        marks = self.get("evaluation.pass_marks", {})
        if not marks:
            errs.append(
                "evaluation.pass_marks is empty. Pre-register the pass marks BEFORE the "
                "first elicitation run — a pass mark chosen after seeing results is not one."
            )
        for name, m in marks.items():
            try:
                base, pas, floor = m["baseline_w1"], m["pass_w1"], m["noise_floor"]
            except (TypeError, KeyError):
                errs.append(f"evaluation.pass_marks.{name}: needs baseline_w1, pass_w1, noise_floor")
                continue
            if not (floor < pas < base):
                errs.append(
                    f"evaluation.pass_marks.{name}: expected floor < pass < baseline, "
                    f"got floor={floor}, pass={pas}, baseline={base}"
                )
            # The checklist states pass marks rounded to 4 dp, so allow half a
            # unit in the last place rather than demanding exact 0.8 * baseline.
            elif pas > base * 0.8 + 5e-5:
                errs.append(
                    f"evaluation.pass_marks.{name}: pass_w1={pas} is not a 20% improvement "
                    f"on baseline_w1={base} (needs <= {base * 0.8:.4f})"
                )

        band = self.get("evaluation.variance_ratio_band")
        if not (isinstance(band, list) and len(band) == 2 and band[0] < 1.0 < band[1]):
            errs.append(f"evaluation.variance_ratio_band must bracket 1.0; got {band!r}")

        if int(self.get("llm.max_calls_per_run", 0)) <= 0:
            errs.append(
                "llm.max_calls_per_run must be a positive ceiling — it is the guard "
                "against a loop bug burning a day's quota in ten minutes (checklist 0.4d)."
            )
        # `--set llm.active_providers=[mistral]` is not JSON, so `_parse_override`
        # falls through to the raw string and the router then iterates it one
        # CHARACTER at a time. Unchecked, switching to the frontier arm for
        # §7.2 rung 1 either dies mid-run or quietly elicits nothing, and on a
        # rate-capped free tier a wasted run costs a day. Demand a list of names
        # that are actually configured.
        active = self.get("llm.active_providers")
        configured = {
            str(pr.get("name")) for pr in (self.get("llm.providers") or [])
            if isinstance(pr, dict)
        }
        if active is None:
            errs.append("llm.active_providers is missing; name at least one provider.")
        elif isinstance(active, str):
            errs.append(
                f"llm.active_providers is the string {active!r}, not a list. The router "
                f"would iterate it character by character. Use JSON on the command line: "
                f'--set \'llm.active_providers=["{active.strip("[]")}"]\''
            )
        elif not isinstance(active, list) or not active:
            errs.append(f"llm.active_providers must be a non-empty list; got {active!r}")
        else:
            unknown = sorted(str(x) for x in active if str(x) not in configured)
            if unknown:
                errs.append(
                    f"llm.active_providers names {unknown}, which are not in "
                    f"llm.providers ({sorted(configured)})."
                )

        if not self.get("llm.checkpoint_every_call", False):
            errs.append(
                "llm.checkpoint_every_call must be true. A 37k-call run at ~1k/day takes "
                "weeks; the runner must survive being stopped and resumed (checklist 0.4c)."
            )

        # Every provider model must carry an explicit version tag. `cache.py`
        # keys responses on the model STRING, so a floating tag ("qwen3", which
        # Ollama resolves to :latest) lets two different sets of weights share
        # one cache key. A resumed run would then mix responses from two models
        # into one ensemble and report the spread as elicitation variance.
        for prov in self.get("llm.providers", []):
            model = str(prov.get("model", ""))
            name = prov.get("name", "?")
            if prov.get("enabled") is False:
                continue  # a parked arm, not yet in use
            if str(name).startswith("ollama") and ":" not in model:
                errs.append(
                    f"llm.providers[{name}].model={model!r} has no version tag. Ollama "
                    f"resolves a bare name to :latest, which moves — and cache.py keys "
                    f"on the model string, so two different weights would collide under "
                    f"one cache key. Pin it (e.g. 'qwen2.5:7b-instruct')."
                )
            # The same argument applies to a hosted `-latest` alias, and it was
            # only ever checked for Ollama. `mistral-small-latest` is repointed
            # at new weights by the provider; cache.py keys on the model string,
            # so a run resumed across that change silently mixes two models into
            # one ensemble and reports the difference as elicitation variance.
            # Only enforced for providers actually in use, so the parked arms in
            # this file stay as documentation.
            if str(name) in set(self.get("llm.active_providers") or []) and (
                model.endswith(("-latest", ":latest"))
            ):
                errs.append(
                    f"llm.providers[{name}].model={model!r} is a floating alias and "
                    f"{name} is in llm.active_providers. The provider repoints it at new "
                    f"weights without notice, and cache.py keys responses on the model "
                    f"string, so a resumed run would mix two models under one key. Pin a "
                    f"dated version — `popsim doctor` lists the pinnable tags for every "
                    f"active provider (it reads the keys from .env the same way a run "
                    f"does, which a curl in your shell does not)."
                )
            if model.endswith(":latest"):
                errs.append(
                    f"llm.providers[{name}].model={model!r} pins ':latest', which moves. "
                    f"Pin a concrete tag so a resumed run reuses the same weights."
                )

        sanity = set(self.get("item_split.sanity_items", []))
        for required in ("finrela", "satfin"):
            if required not in sanity:
                errs.append(
                    f"item_split.sanity_items must contain {required!r} (checklist 1.4b): "
                    "it is income restated and must not reach a headline table."
                )

        if errs:
            raise ConfigError(
                f"{self.source_path} violates the checklist:\n  - " + "\n  - ".join(errs)
            )

    # -------------------------------------------------------------- snapshot
    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.raw)


def _git_provenance(repo_root: Path) -> dict[str, Any]:
    def _run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                args, cwd=repo_root, capture_output=True, text=True, timeout=10, check=False
            )
            return out.stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None

    return {
        "commit": _run("git", "rev-parse", "HEAD"),
        "branch": _run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_run("git", "status", "--porcelain")),
    }


def load_config(
    path: str | os.PathLike[str],
    *,
    overrides: dict[str, Any] | None = None,
    validate: bool = True,
) -> Config:
    """Load and validate a run config.

    ``overrides`` is a flat dict of dotted keys, applied after load — for
    ``--set llm.max_calls_per_run=10`` style CLI use. Overrides are recorded in
    the snapshot so a run is never described by a config it did not use.
    """
    src = Path(path).resolve()
    if not src.exists():
        raise ConfigError(f"config not found: {src}")
    raw = yaml.safe_load(src.read_text()) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"config must be a mapping at top level: {src}")

    applied: dict[str, Any] = {}
    for dotted, value in (overrides or {}).items():
        parts = dotted.split(".")
        cur = raw
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
            if not isinstance(cur, dict):
                raise ConfigError(f"override {dotted!r} traverses a non-mapping")
        cur[parts[-1]] = value
        applied[dotted] = value

    repo_root = src.parent.parent if src.parent.name == "configs" else src.parent
    cfg = Config(raw=raw, source_path=src, repo_root=repo_root, _overrides=applied)
    if validate:
        cfg.validate()
    return cfg


def snapshot_run(cfg: Config, *, run_id: str | None = None, note: str = "") -> Path:
    """Create ``runs/<run_id>__<utc>/`` and snapshot the resolved config into it.

    Returns the run directory. This is the first thing any run does — Gate 0 is
    exactly "a no-op run writes a config snapshot to runs/".
    """
    rid = run_id or cfg.get("run_id", "run")
    stamp = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (cfg.repo_root / cfg.get("paths.runs_root", "runs") / f"{rid}__{stamp}")
    run_dir.mkdir(parents=True, exist_ok=False)

    (run_dir / "config.snapshot.yaml").write_text(
        yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True)
    )
    # Keep the literal source file too: the snapshot is resolved, this is verbatim.
    shutil.copy2(cfg.source_path, run_dir / f"config.source.{cfg.source_path.name}")

    (run_dir / "provenance.json").write_text(
        json.dumps(
            {
                "run_id": rid,
                "started_utc": _dt.datetime.now(_dt.UTC).isoformat(),
                "config_source": str(cfg.source_path),
                "overrides": cfg._overrides,
                "ensemble": cfg.ensemble,
                "n_ensemble": cfg.n_ensemble,
                "git": _git_provenance(cfg.repo_root),
                "note": note,
            },
            indent=2,
        )
    )
    cfg.run_dir = run_dir
    return run_dir
