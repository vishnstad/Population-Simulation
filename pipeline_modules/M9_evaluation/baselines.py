"""
Benchmark baselines (module M9).

The point of this file is to make it hard for the project to fool itself. Each
baseline is a specific, cheap way of producing per-cluster distributions; if one
of them matches the calibrated cluster agent, a corresponding claim is dead.

=========  =======================================  ==========================================
ID         Baseline                                 What it kills if it wins
=========  =======================================  ==========================================
B0a        National marginal, oracle                The entire thesis. This is the operative
                                                    trivial baseline: "every subgroup answers
                                                    like the country". It is deliberately
                                                    unfair -- it gets the TRUE national
                                                    marginal, which for a held-out item we
                                                    would not have. Beating it requires real
                                                    between-cluster signal.
B0b        National marginal, LLM-predicted         Cluster conditioning. The fair version:
                                                    one population-level LLM call, copied to
                                                    every cluster.
B1         Nearest-anchor heuristic                 LLM reasoning. Copies each cluster's own
                                                    observed histogram from the anchor item
                                                    most textually similar to the target. No
                                                    LLM at prediction time.
B2         Per-individual persona sampling          The efficiency claim. Argyle-style silicon
                                                    sampling: one persona prompt per sampled
                                                    respondent, answers pooled into cluster
                                                    histograms. Costs O(N) instead of O(K).
B3         Supervised skyline (gradient boosting)   Nothing -- it trains on the target item's
                                                    real labels, which we assume absent. It
                                                    locates the ceiling.
B4         Uncalibrated cluster agent               The calibration layer itself (ablation).
=========  =======================================  ==========================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OrdinalEncoder

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# B0a / B0b -- no subgroup differentiation
# ----------------------------------------------------------------------
def baseline_b0a_oracle_marginal(
    population_marginal: Sequence[float], n_clusters: int
) -> List[List[float]]:
    """The true national histogram, copied to every cluster."""
    hist = [float(x) for x in population_marginal]
    return [list(hist) for _ in range(n_clusters)]


def baseline_b0b_predicted_marginal(
    predicted_population_marginal: Sequence[float], n_clusters: int
) -> List[List[float]]:
    """An LLM's single population-level histogram, copied to every cluster."""
    hist = [float(x) for x in predicted_population_marginal]
    return [list(hist) for _ in range(n_clusters)]


# ----------------------------------------------------------------------
# B1 -- nearest anchor by item wording
# ----------------------------------------------------------------------
class NearestAnchorBaseline:
    """
    For a target item, find the most textually similar anchor and copy each
    cluster's *observed* histogram on that anchor.

    This is a strong baseline and an important one: it uses real subgroup data and
    real item similarity but no reasoning. If it matches the LLM path, the LLM is
    adding nothing over "this question resembles that one".
    """

    def __init__(self, codebook: Dict[str, Dict[str, Any]], anchor_ids: Sequence[str]):
        self.codebook = codebook
        self.anchor_ids = [a for a in anchor_ids if a in codebook]
        texts = [codebook[a].get("text", a) or a for a in self.anchor_ids]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.matrix = self.vectorizer.fit_transform(texts) if texts else None

    def nearest_anchor(self, target_item_id: str, scale_length: Optional[int] = None) -> Optional[str]:
        """Most similar anchor whose scale length matches, so histograms are comparable."""
        if self.matrix is None or target_item_id not in self.codebook:
            return None
        text = self.codebook[target_item_id].get("text", target_item_id) or target_item_id
        sims = cosine_similarity(self.vectorizer.transform([text]), self.matrix)[0]
        for idx in np.argsort(sims)[::-1]:
            anchor = self.anchor_ids[idx]
            if anchor == target_item_id:
                continue
            if scale_length is not None:
                labels = self.codebook[anchor].get("scale", {}).get("labels", [])
                if len(labels) != scale_length:
                    continue
            return anchor
        return None

    def predict(
        self,
        target_item_id: str,
        cluster_ids: Sequence[str],
        stats_dict: Dict[tuple, Any],
        scale_length: int,
        fallback: Sequence[float],
    ) -> List[List[float]]:
        anchor = self.nearest_anchor(target_item_id, scale_length)
        out: List[List[float]] = []
        for cid in cluster_ids:
            row = stats_dict.get((cid, anchor)) if anchor else None
            if row is not None and len(row.hist) == scale_length:
                out.append([float(x) for x in row.hist])
            else:
                out.append([float(x) for x in fallback])
        return out


# ----------------------------------------------------------------------
# B2 -- per-individual persona (silicon) sampling
# ----------------------------------------------------------------------
PERSONA_SYSTEM = (
    "You are simulating a single survey respondent in the United States. "
    "Answer exactly as this person would, choosing one option. Output JSON only."
)

PERSONA_TEMPLATE = """You are a {age_band} year old {sex} living in {region} ({urban}).
Your education level is {education}. Your household income is in the {income_band} quintile
nationally. Your race is {race}. Your religion is {religion}. You are {marital}.

Survey question: "{question_text}"
Options: [{options_str}]

Answer with the single option you would choose.
Return JSON: {{"choice_index": <0-based index into the options list>}}
"""


@dataclass
class PersonaSamplingResult:
    cluster_dists: Dict[str, List[float]]
    n_calls: int
    cost_usd: float
    n_failed: int


def baseline_b2_persona_sampling(
    individuals: pd.DataFrame,
    cluster_ids: Sequence[str],
    cluster_col: str,
    item_id: str,
    question_text: str,
    options: Sequence[str],
    client: Any,
    n_per_cluster: int = 20,
    seed: int = 20260818,
    temperature: float = 1.0,
) -> PersonaSamplingResult:
    """
    Argyle-style silicon sampling, run at the cluster level so it is directly
    comparable on cost.

    Samples ``n_per_cluster`` real respondents from each cluster, prompts the model
    once per respondent with that person's demographics, and pools the sampled
    single answers into a histogram. This is the method the project claims to beat
    on both fidelity and cost, so it has to be implemented honestly rather than
    assumed to be bad.

    Cost is O(n_per_cluster x K) calls -- with the defaults this is roughly
    ``n_per_cluster / 9`` times the cost of one distributional cell.
    """
    from pydantic import BaseModel, Field

    class PersonaChoice(BaseModel):
        choice_index: int = Field(..., ge=0)

    rng = np.random.RandomState(seed)
    options_str = ", ".join(f'"{o}"' for o in options)
    out: Dict[str, List[float]] = {}
    n_calls = n_failed = 0
    cost = 0.0

    for cid in cluster_ids:
        members = individuals[individuals[cluster_col] == cid]
        if members.empty:
            continue
        take = min(n_per_cluster, len(members))
        idx = rng.choice(len(members), size=take, replace=False)
        counts = np.zeros(len(options), dtype=float)

        for row in members.iloc[idx].itertuples():
            prompt = PERSONA_TEMPLATE.format(
                age_band=getattr(row, "demo_age_band", "adult"),
                sex=getattr(row, "demo_sex", "person"),
                region=getattr(row, "demo_region", "the United States"),
                urban="urban" if str(getattr(row, "demo_urban", "")) == "True" else "rural",
                education=getattr(row, "demo_education", "unknown"),
                income_band=getattr(row, "demo_income_band", "unknown"),
                race=getattr(row, "demo_race", "unknown"),
                religion=getattr(row, "demo_religion", "unknown"),
                marital=getattr(row, "demo_marital", "unknown"),
                question_text=question_text,
                options_str=options_str,
            )
            try:
                resp = client.generate_structured(
                    system_prompt=PERSONA_SYSTEM,
                    user_prompt=prompt,
                    response_schema=PersonaChoice,
                    temperature=temperature,
                    sample_id=int(getattr(row, "Index", 0)),
                    max_tokens=64,
                )
                n_calls += 1
                cost += resp.cost_usd
                choice = int(resp.parsed["choice_index"])
                if 0 <= choice < len(options):
                    counts[choice] += 1.0
                else:
                    n_failed += 1
            except Exception as exc:  # noqa: BLE001
                logger.debug("persona call failed: %s", exc)
                n_failed += 1

        if counts.sum() > 0:
            out[cid] = [float(x) for x in (counts / counts.sum())]

    return PersonaSamplingResult(out, n_calls, cost, n_failed)


# ----------------------------------------------------------------------
# B3 -- supervised skyline
# ----------------------------------------------------------------------
DEMO_FEATURES = [
    "demo_age_band",
    "demo_sex",
    "demo_education",
    "demo_income_band",
    "demo_region",
    "demo_urban",
    "demo_race",
    "demo_religion",
    "demo_marital",
]


def baseline_b3_supervised_skyline(
    individuals: pd.DataFrame,
    cluster_ids: Sequence[str],
    cluster_col: str,
    item_id: str,
    scale_length: int,
    train_frac: float = 0.5,
    seed: int = 20260818,
) -> Optional[List[List[float]]]:
    """
    Gradient boosting on demographics -> response, trained on half the individuals'
    *actual* labels for the target item, then used to predict per-cluster
    histograms for everyone.

    This is not a competitor. It uses the labels the whole project assumes are
    unavailable, so it marks the ceiling: how much of this item is predictable from
    demographics at all. If the LLM path approaches it, there is little headroom
    left; if B3 itself is barely better than B0a, the item simply is not a
    demographic story and no method will look good on it.

    Returns ``None`` when the item has too little data to fit.
    """
    col = f"item_{item_id}"
    if col not in individuals.columns:
        return None

    data = individuals[individuals[col].notna() & (individuals[col] >= 0)].copy()
    if len(data) < 100:
        return None

    codes = data[col].astype(int)
    lo = int(codes.min())
    y = (codes - lo).to_numpy()
    if len(np.unique(y)) < 2:
        return None

    features = [f for f in DEMO_FEATURES if f in data.columns]
    x_raw = data[features].astype(str)
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    x = encoder.fit_transform(x_raw)

    rng = np.random.RandomState(seed)
    train_mask = rng.rand(len(data)) < train_frac
    if train_mask.sum() < 50 or len(np.unique(y[train_mask])) < 2:
        return None

    model = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.1, max_depth=6, random_state=seed
    )
    model.fit(x[train_mask], y[train_mask], sample_weight=data["weight"].to_numpy()[train_mask])

    proba = model.predict_proba(x)
    seen = list(model.classes_)

    out: List[List[float]] = []
    weights = data["weight"].to_numpy(dtype=float)
    cluster_series = data[cluster_col].astype(str).to_numpy()
    for cid in cluster_ids:
        sel = cluster_series == cid
        if not sel.any():
            out.append([1.0 / scale_length] * scale_length)
            continue
        w = weights[sel]
        avg = (proba[sel] * w[:, None]).sum(axis=0) / w.sum()
        full = np.zeros(scale_length, dtype=float)
        for pos, cls in enumerate(seen):
            if 0 <= int(cls) < scale_length:
                full[int(cls)] = avg[pos]
        full = full / full.sum() if full.sum() > 0 else np.full(scale_length, 1.0 / scale_length)
        out.append([float(v) for v in full])
    return out


# ----------------------------------------------------------------------
# B4 -- uncalibrated (ablation, needs no extra computation)
# ----------------------------------------------------------------------
def baseline_b4_uncalibrated(raw_dists: Sequence[Sequence[float]]) -> List[List[float]]:
    """Raw ensemble-mean M4 output with the M5 calibration layer switched off."""
    return [[float(x) for x in d] for d in raw_dists]
