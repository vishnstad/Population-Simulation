"""Gate 2 runner — partition, stats, noise floor, split (checklist 2.6).

    **Gate 2:** K ~ 56, >= 40 scorable targets, split frozen to disk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..clustering.partition import build_cluster_tree, compare_to_full_cross
from ..clustering.pooling import pool_toward_parent
from ..clustering.stats import compute_cluster_stats
from ..data.adapters.gss import load_gss
from .noise import measure_noise_floor
from .split import ItemSplit, load_or_create

__all__ = ["Gate2Report", "run_gate2"]


@dataclass
class Gate2Report:
    k: int
    k_target: int
    n_targets: int
    n_anchors: int
    n_passing_snr: int
    split_path: str
    split_was_frozen: bool
    tree_check: dict[str, Any]
    noise_summary: str
    stats_notes: dict[str, Any]
    pooling_notes: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (
            abs(self.k - self.k_target) <= 3
            and self.n_targets >= 40
            and Path(self.split_path).exists()
        )

    def summary(self) -> str:
        frozen = ("  (loaded from disk — frozen)" if self.split_was_frozen
                  else "  (created and frozen now)")
        lines = [
            "Gate 2 — partition, noise floor, item split",
            f"  K                 {self.k} (target {self.k_target})",
            f"  items clearing    {self.n_passing_snr} at SNR >= 1.5",
            f"  targets / anchors {self.n_targets} / {self.n_anchors}",
            f"  split             {self.split_path}{frozen}",
            (
                f"  tree vs cross     adjusted Rand {self.tree_check['adjusted_rand']:.4f} "
                f"({self.tree_check['k_full_cross']} full-cross cells)"
            ),
            (
                f"  pooling           median lambda "
                f"{self.pooling_notes.get('median_shrinkage_lambda', 0):.3f}, "
                f"{self.pooling_notes.get('cells_mostly_parent', 0)} cells mostly parent"
            ),
        ]
        for w in self.warnings:
            lines.append(f"  ! {w}")
        lines.append(f"  => {'GREEN' if self.ok else 'RED'}")
        return "\n".join(lines)


def run_gate2(cfg, *, out_dir: str | Path | None = None, n_repeats: int = 30) -> Gate2Report:
    codebook = yaml.safe_load(
        (cfg.repo_root / cfg["paths.codebooks"] / "gss_items.yaml").read_text()
    )["items"]
    items = sorted(codebook)
    codes = {i: codebook[i]["codes"] for i in items}

    table = load_gss(cfg.bed_file, items, waves=cfg["bed.waves"],
                     weight_var=cfg["bed.weight_var"])

    tree = build_cluster_tree(
        table.frame, axes=cfg["partition.axes"], item_codes=codes,
        min_cell=cfg["partition.min_cell"], k_target=cfg["partition.k_target"],
    )
    assign = tree.assign(table.frame)
    tree_check = compare_to_full_cross(tree, table.frame)

    stats = compute_cluster_stats(table.frame, assign, items, codes)

    # Parent-level statistics, for pooling: each leaf's parent node computed from
    # all of its own respondents rather than an average of its children.
    parent_of_leaf = assign.map(
        lambda c: tree.nodes[c].parent if c in tree.nodes else None
    )
    parent_stats = compute_cluster_stats(table.frame, parent_of_leaf, items, codes)
    pooled = pool_toward_parent(stats, tree, parent_stats, tau=cfg["partition.tau_pooling"])

    noise = measure_noise_floor(
        table.frame, assign, items, codes,
        min_snr=cfg["partition.min_item_snr"], n_repeats=n_repeats,
        min_cell_answered=int(cfg["evaluation.truth_min_neff"]),
    )

    split_path = cfg.repo_root / cfg["item_split.freeze_path"]
    was_frozen = split_path.exists()
    split: ItemSplit = load_or_create(
        split_path,
        snr={k: v.snr for k, v in noise.items.items()},
        topics={i: codebook[i]["topic"] for i in items},
        roles={i: codebook[i].get("role", "unassigned") for i in items},
        wording_ok={i: codebook[i].get("wording_status") == "verified" for i in items},
        leakage_resistant={i for i in items if codebook[i].get("leakage_resistant")},
        famous={i for i in items if i.startswith(("spk", "col", "lib"))},
        n_targets=cfg["item_split.n_targets"],
        n_anchors=cfg["item_split.n_anchors"],
        n_anchor_only_low_snr=cfg["item_split.n_anchor_only_low_snr"],
        min_snr=cfg["partition.min_item_snr"],
    )

    warnings: list[str] = []
    if len(split.targets) < cfg["item_split.n_targets"]:
        warnings.append(
            f"only {len(split.targets)} targets, wanted {cfg['item_split.n_targets']}"
        )
    thin = stats.frame[stats.frame.n_eff < cfg["evaluation.truth_min_neff"]]
    if len(thin):
        warnings.append(
            f"{len(thin)} of {len(stats.frame)} (cluster, item) cells have n_eff below "
            f"{cfg['evaluation.truth_min_neff']} and are not scorable (spec §5.2)"
        )
    unverified = [t for t in split.targets
                  if codebook[t].get("wording_status") != "verified"]
    if unverified:
        warnings.append(f"targets with unverified wording: {unverified}")

    report = Gate2Report(
        k=tree.k, k_target=cfg["partition.k_target"],
        n_targets=len(split.targets), n_anchors=len(split.anchors),
        n_passing_snr=len(noise.passing()),
        split_path=str(split_path), split_was_frozen=was_frozen,
        tree_check=tree_check, noise_summary=noise.summary(),
        stats_notes=stats.notes, pooling_notes=pooled.notes, warnings=warnings,
    )

    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        tree.to_json(out / "cluster_tree.json")
        stats.to_parquet(out / "cluster_stats.parquet")
        pooled.to_parquet(out / "cluster_stats_pooled.parquet")
        noise.to_parquet(out / "noise_floor.parquet")
        (out / "gate2_summary.txt").write_text(report.summary() + "\n\n" + noise.summary())
        (out / "gate2_report.json").write_text(json.dumps({
            "k": report.k, "n_targets": report.n_targets, "n_anchors": report.n_anchors,
            "n_passing_snr": report.n_passing_snr, "tree_check": report.tree_check,
            "stats_notes": report.stats_notes, "pooling_notes": report.pooling_notes,
            "warnings": report.warnings, "ok": report.ok,
        }, indent=2, default=str))
    return report
