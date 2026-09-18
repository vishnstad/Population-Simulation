"""
Distributional elicitation (module M4).

For one (cluster, item) cell the elicitor asks the model, several times over,
what percentage of that subgroup would choose each response option. It never asks
the model to answer *as* a member of the group -- the framing is statistical
throughout, which is both the mitigation for persona variance collapse and the
reason sensitive items (religion, race) draw far fewer refusals.

Ensemble
--------
``n_paraphrases`` hand-written prompt framings x ``n_repeats`` samples at
temperature > 0. Each of the resulting draws carries a distinct ``sample_id``
that enters the cache key, so the members are genuinely independent draws. The
spread across them is retained as ``ensemble_sd`` and is what the bootstrap in M6
resamples.

Output
------
A long-format parquet with one row per draw (not per cell). Aggregation happens
downstream so that the ensemble spread is never lost on disk.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

try:  # package-relative when imported as Codes.M4_elicitation
    from ..shared.budget_guard import BudgetGuard, BudgetExceededError
    from ..shared.cache import PromptCache
    from ..shared.llm_client import BaseLLMClient, LLMError, get_llm_client
    from ..M3_statcards.builder import StatCard, StatCardBuilder
    from .prompts import SYSTEM_PROMPT, build_user_prompt
    from .schemas import ElicitedDistributionOutput, RawElicitationRecord
except (ImportError, ValueError):  # flat when Codes/ is on sys.path
    from shared.budget_guard import BudgetGuard, BudgetExceededError
    from shared.cache import PromptCache
    from shared.llm_client import BaseLLMClient, LLMError, get_llm_client
    from M3_statcards.builder import StatCard, StatCardBuilder
    from M4_elicitation.prompts import SYSTEM_PROMPT, build_user_prompt
    from M4_elicitation.schemas import ElicitedDistributionOutput, RawElicitationRecord

logger = logging.getLogger(__name__)

RAW_COLUMNS = [
    "cluster_id",
    "item_id",
    "model",
    "stage",
    "crossfit_fold",
    "paraphrase_id",
    "repeat_id",
    "sample_id",
    "probabilities",
    "reasoning_summary",
    "prompt_tokens",
    "completion_tokens",
    "cost_usd",
    "is_cached",
    "n_anchors_shown",
]


@dataclass
class CellResult:
    """All draws for one (cluster, item) cell, plus their summary."""

    cluster_id: str
    item_id: str
    model: str
    stage: str
    crossfit_fold: Optional[int]
    draws: List[List[float]] = field(default_factory=list)
    rows: List[Dict[str, Any]] = field(default_factory=list)
    n_failed: int = 0

    @property
    def mean_probabilities(self) -> List[float]:
        arr = np.asarray(self.draws, dtype=float)
        mean = arr.mean(axis=0)
        mean = mean / mean.sum()
        return [round(float(p), 6) for p in mean]

    @property
    def ensemble_sd(self) -> float:
        """Mean per-option SD across ensemble members: the elicitation uncertainty."""
        if len(self.draws) < 2:
            return 0.0
        return float(np.asarray(self.draws, dtype=float).std(axis=0, ddof=1).mean())


class DistributionElicitor:
    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        model: str = "claude-haiku-4-5",
        cache_path: Optional[Path] = None,
        budget_path: Optional[Path] = None,
        max_usd: float = 200.0,
        temperature: float = 0.7,
    ):
        cache = PromptCache(db_path=cache_path) if cache_path else PromptCache()
        budget = BudgetGuard(state_file=budget_path, max_usd=max_usd)
        self.client = llm_client or get_llm_client(model=model, cache=cache, budget_guard=budget)
        self.budget = self.client.budget_guard
        self.temperature = temperature

    # ------------------------------------------------------------------
    def elicit_cell(
        self,
        card: StatCard,
        item_id: str,
        topic: str,
        question_text: str,
        options: Sequence[str],
        stage: str = "target",
        crossfit_fold: Optional[int] = None,
        n_paraphrases: int = 3,
        n_repeats: int = 3,
    ) -> CellResult:
        """Run the full ensemble for one cell. Malformed draws are dropped and counted."""
        result = CellResult(
            cluster_id=card.cluster_id,
            item_id=item_id,
            model=self.client.model,
            stage=stage,
            crossfit_fold=crossfit_fold,
        )
        n_options = len(options)

        for p_id in range(1, n_paraphrases + 1):
            user_prompt = build_user_prompt(
                stat_card_text=card.rendered_prompt,
                topic=topic,
                question_text=question_text,
                options=list(options),
                paraphrase_id=p_id,
            )
            for r_id in range(1, n_repeats + 1):
                sample_id = (p_id - 1) * n_repeats + r_id
                try:
                    resp = self.client.generate_structured(
                        system_prompt=SYSTEM_PROMPT,
                        user_prompt=user_prompt,
                        response_schema=ElicitedDistributionOutput,
                        temperature=self.temperature,
                        sample_id=sample_id,
                    )
                except BudgetExceededError:
                    raise
                except LLMError as exc:
                    logger.warning("cell %s/%s sample %d failed: %s", card.cluster_id, item_id, sample_id, exc)
                    result.n_failed += 1
                    continue

                probs = list(resp.parsed["probabilities"])
                if len(probs) != n_options:
                    # The model returned the wrong number of bins. Do not guess --
                    # a silently reshaped histogram is a corrupt observation.
                    logger.warning(
                        "cell %s/%s sample %d: expected %d options, got %d - dropped",
                        card.cluster_id, item_id, sample_id, n_options, len(probs),
                    )
                    result.n_failed += 1
                    continue

                result.draws.append(probs)
                result.rows.append(
                    {
                        "cluster_id": card.cluster_id,
                        "item_id": item_id,
                        "model": self.client.model,
                        "stage": stage,
                        "crossfit_fold": -1 if crossfit_fold is None else int(crossfit_fold),
                        "paraphrase_id": p_id,
                        "repeat_id": r_id,
                        "sample_id": sample_id,
                        "probabilities": probs,
                        "reasoning_summary": (resp.parsed.get("reasoning_summary") or "")[:300],
                        "prompt_tokens": resp.prompt_tokens,
                        "completion_tokens": resp.completion_tokens,
                        "cost_usd": resp.cost_usd,
                        "is_cached": resp.is_cached,
                        "n_anchors_shown": len(card.anchor_items),
                    }
                )
        return result

    # ------------------------------------------------------------------
    def elicit_batch(
        self,
        card_builder: StatCardBuilder,
        cluster_ids: Sequence[str],
        item_ids: Sequence[str],
        stage: str = "target",
        split_regime: str = "standard",
        crossfit_plan: Optional[Dict[str, int]] = None,
        n_paraphrases: int = 3,
        n_repeats: int = 3,
        max_anchors: int = 12,
        output_parquet_path: Optional[Path] = None,
        resume: bool = True,
        progress_every: int = 25,
    ) -> pd.DataFrame:
        """
        Elicit every (cluster, item) cell in the cross product.

        Resumable: with ``resume=True`` any cell already present in the output
        parquet is skipped, so an interrupted overnight run continues where it
        stopped. Combined with the SQLite prompt cache, no completed call is ever
        paid for twice.

        For ``stage="anchor"`` the ``crossfit_plan`` is applied, so each anchor is
        predicted from a card built without its own fold. That is what makes the
        resulting pairs valid calibrator training data.
        """
        out_path = Path(output_parquet_path) if output_parquet_path else None
        existing = pd.DataFrame(columns=RAW_COLUMNS)
        done: Set[Tuple[str, str]] = set()
        if resume and out_path and out_path.exists():
            existing = pd.read_parquet(out_path)
            done = set(zip(existing["cluster_id"], existing["item_id"]))
            logger.info("resuming: %d cells already complete", len(done))

        todo = [(c, i) for c in cluster_ids for i in item_ids if (c, i) not in done]
        total = len(todo)
        print(
            f"  [{stage}] {len(cluster_ids)} clusters x {len(item_ids)} items "
            f"= {len(cluster_ids) * len(item_ids)} cells "
            f"({len(done)} done, {total} to run, {n_paraphrases * n_repeats} draws each)"
        )

        new_rows: List[Dict[str, Any]] = []
        n_failed_cells = 0
        try:
            for n, (cid, iid) in enumerate(todo, start=1):
                meta = card_builder.codebook[iid]
                options = list(meta.get("scale", {}).get("labels", []))
                if len(options) < 2:
                    continue

                fold = (crossfit_plan or {}).get(iid) if stage == "anchor" else None
                card = card_builder.build_stat_card(
                    cluster_id=cid,
                    target_item_id=iid,
                    split_regime=split_regime,
                    crossfit_plan=crossfit_plan if stage == "anchor" else None,
                    max_anchors=max_anchors,
                )
                # Hard stop rather than a silent leak.
                card_builder.verify_card_contract(
                    card, iid, crossfit_plan if stage == "anchor" else None
                )

                res = self.elicit_cell(
                    card=card,
                    item_id=iid,
                    topic=meta.get("topic", "general"),
                    question_text=meta.get("text", iid),
                    options=options,
                    stage=stage,
                    crossfit_fold=fold,
                    n_paraphrases=n_paraphrases,
                    n_repeats=n_repeats,
                )
                if not res.draws:
                    n_failed_cells += 1
                new_rows.extend(res.rows)

                if progress_every and n % progress_every == 0:
                    spent = self.budget.total_usd_spent
                    print(
                        f"    {n}/{total} cells | ${spent:.2f} spent | "
                        f"{self.budget.total_live_calls} live / {self.budget.total_cached_calls} cached"
                    )
                    if out_path:
                        self._flush(existing, new_rows, out_path)
        except (BudgetExceededError, KeyboardInterrupt) as exc:
            print(f"\n  [stopped] {type(exc).__name__}: {exc}")
            print("  Partial results are saved; re-run the same command to resume.")
        finally:
            if out_path:
                self._flush(existing, new_rows, out_path)

        df = self._combine(existing, new_rows)
        if n_failed_cells:
            logger.warning("%d cells produced no usable draw", n_failed_cells)
        return df

    @staticmethod
    def _combine(existing: pd.DataFrame, new_rows: List[Dict[str, Any]]) -> pd.DataFrame:
        """Concatenate resumed and fresh rows without tripping pandas' empty-frame path."""
        frames = [f for f in (existing, pd.DataFrame(new_rows)) if not f.empty]
        if not frames:
            return pd.DataFrame(columns=RAW_COLUMNS)
        return frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)

    @staticmethod
    def _flush(existing: pd.DataFrame, new_rows: List[Dict[str, Any]], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df = DistributionElicitor._combine(existing, new_rows)
        if not df.empty:
            df.to_parquet(path, index=False)


# ----------------------------------------------------------------------
# aggregation helper
# ----------------------------------------------------------------------
def aggregate_draws(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the long-format draw table to one row per (cluster, item) cell.

    Keeps the ensemble spread (``ensemble_sd``) and the draw count, both of which
    the uncertainty machinery in M6 needs.
    """
    if raw_df.empty:
        return pd.DataFrame(
            columns=["cluster_id", "item_id", "model", "stage", "crossfit_fold",
                     "mean_probabilities", "ensemble_sd", "n_draws"]
        )

    records = []
    for (cid, iid), grp in raw_df.groupby(["cluster_id", "item_id"], sort=False):
        arr = np.asarray([list(p) for p in grp["probabilities"]], dtype=float)
        mean = arr.mean(axis=0)
        mean = mean / mean.sum()
        records.append(
            {
                "cluster_id": cid,
                "item_id": iid,
                "model": grp["model"].iloc[0],
                "stage": grp["stage"].iloc[0],
                "crossfit_fold": int(grp["crossfit_fold"].iloc[0]),
                "mean_probabilities": [round(float(p), 6) for p in mean],
                "ensemble_sd": float(arr.std(axis=0, ddof=1).mean()) if len(arr) > 1 else 0.0,
                "n_draws": int(len(arr)),
            }
        )
    return pd.DataFrame(records)
