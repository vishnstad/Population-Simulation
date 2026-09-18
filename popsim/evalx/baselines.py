"""Baselines (spec §5.4). What each one kills if it wins.

| id | baseline | kills |
|----|----------|-------|
| B0a | national marginal *oracle* — the TRUE population histogram copied to every cluster | the entire thesis |
| B0b | national marginal *predicted* — one population-level LLM call, copied to every cluster | cluster conditioning |
| B1  | nearest-anchor — copy each cluster's histogram from its most similar anchor item | LLM reasoning adds nothing over item similarity |
| B3  | supervised skyline — gradient boosting on demographics, trained on real target labels | nothing; it locates the ceiling |
| B4  | uncalibrated cluster agent (raw M4 output) | the calibration mechanism |
| B5  | point-valued cluster agent (forced single answer, x9 -> histogram) | distribution-valued elicitation |

B0a is the one §1.4 is written against, and it is deliberately unfair: it is
handed the true national marginal, which no LLM path has. The system's primary
mode is given that same marginal precisely so the comparison is about subgroup
structure and nothing else — **the deviation scale s = 0 reproduces B0a exactly**,
which makes B0a an ablation of the method rather than an outside competitor.

B3 uses labels the whole design assumes absent. It is reported as a ceiling and
never as something to beat; a system at or above B3 would mean the target item
carries no demographic signal beyond what the agent already found.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "b0a_national_oracle",
    "b0b_national_predicted",
    "b1_nearest_anchor",
    "b3_supervised_skyline",
    "b4_uncalibrated",
    "point_answers_to_histogram",
    "resample_histogram",
]


def b0a_national_oracle(national: np.ndarray, cluster_ids) -> dict[str, np.ndarray]:
    h = np.asarray(national, dtype=float)
    return {c: h.copy() for c in cluster_ids}


def b0b_national_predicted(pred_national: np.ndarray, cluster_ids) -> dict[str, np.ndarray]:
    h = np.asarray(pred_national, dtype=float)
    return {c: h.copy() for c in cluster_ids}


def resample_histogram(h: np.ndarray, k_out: int) -> np.ndarray:
    """Move a histogram onto a scale of a different length, through its CDF.

    Needed only by B1, where the most similar anchor can have a different number
    of options from the target. Interpolating the CDF on the normalized position
    axis is the operation W1 itself is defined on, so the baseline is not
    penalised by an arbitrary re-binning choice.
    """
    h = np.asarray(h, dtype=float)
    k_in = h.size
    if k_in == k_out:
        s = h.sum()
        return h / s if s > 0 else h
    c_in = np.cumsum(h / h.sum())
    x_in = np.linspace(0.0, 1.0, k_in)
    x_out = np.linspace(0.0, 1.0, k_out)
    c_out = np.interp(x_out, x_in, c_in)
    c_out[-1] = 1.0
    out = np.clip(np.diff(np.concatenate([[0.0], c_out])), 0.0, None)
    s = out.sum()
    return out / s if s > 0 else np.full(k_out, 1.0 / k_out)


def b1_nearest_anchor(
    target_id: str,
    anchor_pool,
    codebook: dict[str, dict],
    stats,
    cluster_ids,
) -> tuple[dict[str, np.ndarray], str]:
    """Copy each cluster's histogram from its most textually similar anchor.

    TF-IDF cosine over the verbatim instrument wording — the same method §6.1
    fixes for the router, and for the same reason: a neural embedder is not
    load-bearing here and TF-IDF is reproducible without a model download.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    pool = [a for a in anchor_pool if a in codebook and a != target_id]
    if not pool:
        return {}, ""
    texts = [codebook[target_id]["text"]] + [codebook[a]["text"] for a in pool]
    X = TfidfVectorizer(stop_words="english").fit_transform(texts)
    sims = (X[1:] @ X[0].T).toarray().ravel()
    best = pool[int(np.argmax(sims))]
    k_out = len(codebook[target_id]["codes"])
    out: dict[str, np.ndarray] = {}
    for c in cluster_ids:
        h = stats.hist(c, best)
        if h is None or h.sum() <= 0:
            continue
        out[c] = resample_histogram(np.asarray(h, dtype=float), k_out)
    return out, best


def b4_uncalibrated(raw_by_cluster: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {}
    for c, h in raw_by_cluster.items():
        if h is None:
            continue
        h = np.asarray(h, dtype=float)
        if h.sum() > 0:
            out[c] = h / h.sum()
    return out


def point_answers_to_histogram(codes_chosen, codes: list[int]) -> np.ndarray:
    """B5: a forced single answer, repeated, becomes a histogram by counting."""
    out = np.zeros(len(codes), dtype=float)
    for c in codes_chosen:
        if c in codes:
            out[codes.index(c)] += 1.0
    s = out.sum()
    return out / s if s > 0 else out


def b3_supervised_skyline(
    frame,
    assign,
    item_id: str,
    codes: list[int],
    cluster_ids,
    *,
    axes=("age_band", "degree", "sex", "region", "urban"),
    train_frac: float = 0.5,
    seed: int = 17,
) -> dict[str, np.ndarray]:
    """Gradient boosting on demographics -> response, on real target labels.

    Trained on half the respondents who answered the item, predicted-proba
    averaged within each cluster on the held-out half. This uses exactly the
    labels the rest of the design assumes do not exist, so it is a *ceiling*
    marker, not a competitor. Spec §5.4: "it locates the ceiling".
    """
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier

    col = f"item_{item_id}"
    if col not in frame.columns:
        return {}
    y = frame[col].to_numpy()
    mask = np.isin(y, codes)
    if mask.sum() < 100:
        return {}
    feats = [a for a in axes if a in frame.columns]
    X = frame.loc[mask, feats].copy()
    for c in feats:
        X[c] = pd.Categorical(X[c].astype(str)).codes
    yy = y[mask]
    rng = np.random.default_rng(seed)
    tr = rng.random(mask.sum()) < train_frac
    if tr.sum() < 50 or len(np.unique(yy[tr])) < 2:
        return {}
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
    clf.fit(X[tr], yy[tr])
    proba = clf.predict_proba(X)
    cls = list(clf.classes_)
    full = np.zeros((proba.shape[0], len(codes)))
    for j, c in enumerate(codes):
        if c in cls:
            full[:, j] = proba[:, cls.index(c)]
    w = pd.to_numeric(frame.loc[mask, "weight"], errors="coerce").fillna(0.0).to_numpy()
    # `assign` carries pd.NA for the respondents outside every leaf, and a plain
    # comparison on that dtype raises rather than returning False. Positional
    # masking too: `assign` is indexed like `frame`, so it has to be taken to a
    # numpy array before the row mask is applied.
    cl = (assign.astype("object").fillna("").to_numpy()
          if hasattr(assign, "to_numpy") else np.asarray(assign, dtype=object))[mask]
    out: dict[str, np.ndarray] = {}
    for c in cluster_ids:
        m = (cl == c) & (~tr)
        if m.sum() < 5:
            m = cl == c
        if m.sum() == 0:
            continue
        h = np.average(full[m], axis=0, weights=np.clip(w[m], 1e-9, None))
        s = h.sum()
        if s > 0:
            out[c] = h / s
    return out
