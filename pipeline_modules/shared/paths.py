"""
Central path + config resolution for the B17 pipeline.

Every script resolves its inputs and outputs through this module so that a run is
fully described by one YAML config file (architecture spec 3.2).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

CODES_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = CODES_DIR.parent
PREPROCESSING_DIR = PROJECT_ROOT / "preprocessing"
DEFAULT_CONFIG = CODES_DIR / "config" / "run_gss2024.yaml"


@dataclass
class RunConfig:
    """Everything a run needs. Loaded from one YAML file."""

    raw: Dict[str, Any] = field(default_factory=dict)
    config_path: Optional[Path] = None

    # ---- data inputs -------------------------------------------------
    @property
    def preprocessed_dir(self) -> Path:
        return (PROJECT_ROOT / self.raw["data"]["preprocessed_dir"]).resolve()

    @property
    def granularity(self) -> str:
        return self.raw["data"].get("granularity", "coarse")

    @property
    def tree_path(self) -> Path:
        name = "cluster_tree_coarse.json" if self.granularity == "coarse" else "cluster_tree.json"
        return self.preprocessed_dir / name

    @property
    def stats_path(self) -> Path:
        name = "cluster_stats_coarse.parquet" if self.granularity == "coarse" else "cluster_stats.parquet"
        return self.preprocessed_dir / name

    @property
    def codebook_path(self) -> Path:
        return self.preprocessed_dir / "item_codebook.yaml"

    @property
    def split_path(self) -> Path:
        return self.preprocessed_dir / "item_split.csv"

    @property
    def pop_stats_path(self) -> Path:
        return self.preprocessed_dir / "population_stats.parquet"

    @property
    def individual_table_path(self) -> Path:
        return self.preprocessed_dir / "individual_table.parquet"

    @property
    def margins_path(self) -> Optional[Path]:
        m = self.raw.get("aggregation", {}).get("margins_file")
        return (PROJECT_ROOT / m).resolve() if m else None

    # ---- run outputs -------------------------------------------------
    @property
    def run_name(self) -> str:
        return self.raw.get("run_name", "default")

    @property
    def out_dir(self) -> Path:
        d = (CODES_DIR / "runs" / self.run_name).resolve()
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def cache_path(self) -> Path:
        shared = self.raw.get("llm", {}).get("shared_cache", True)
        p = (CODES_DIR / "runs" / "llm_cache.sqlite") if shared else (self.out_dir / "llm_cache.sqlite")
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def budget_path(self) -> Path:
        return self.out_dir / "budget_state.json"

    # ---- sections ----------------------------------------------------
    @property
    def llm(self) -> Dict[str, Any]:
        return self.raw.get("llm", {})

    @property
    def elicitation(self) -> Dict[str, Any]:
        return self.raw.get("elicitation", {})

    @property
    def calibration(self) -> Dict[str, Any]:
        return self.raw.get("calibration", {})

    @property
    def evaluation(self) -> Dict[str, Any]:
        return self.raw.get("evaluation", {})

    @property
    def aggregation(self) -> Dict[str, Any]:
        return self.raw.get("aggregation", {})

    @property
    def split_regime(self) -> str:
        return self.evaluation.get("split_regime", "standard")

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node


def load_config(path: Optional[Path | str] = None) -> RunConfig:
    """Load a run config. Falls back to config/run_gss2024.yaml."""
    p = Path(path) if path else DEFAULT_CONFIG
    if not p.is_absolute():
        p = (CODES_DIR / p).resolve()
    if not p.exists():
        raise FileNotFoundError(
            f"Config not found: {p}\nExpected the default at {DEFAULT_CONFIG}"
        )
    with open(p, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return RunConfig(raw=raw, config_path=p)


def load_dotenv(path: Optional[Path] = None) -> Dict[str, str]:
    """
    Minimal .env loader so API keys never have to live in shell profiles.
    Values already present in os.environ always win.
    """
    p = Path(path) if path else (PROJECT_ROOT / ".env")
    loaded: Dict[str, str] = {}
    if not p.exists():
        return loaded
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        loaded[key] = value
        os.environ.setdefault(key, value)
    return loaded
