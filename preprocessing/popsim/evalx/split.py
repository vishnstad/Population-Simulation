"""
Item heterogeneity + anchor/target split -- B17 §5.1, and the `heterogeneity`
field of the Item codebook (§3.3 M1).

WHY THIS MODULE MATTERS MORE THAN IT LOOKS
------------------------------------------
§1.4's falsifiable claim (i) is scored on "the top-quartile heterogeneity
items (items where clusters genuinely differ)". If heterogeneity is estimated
naively, that quartile is selected on sampling noise and the headline metric
is measured on the wrong items.

With ~3.3k respondents split over ~150 clusters, a cluster holds ~22 people.
The raw between-cluster variance share of ANY item -- including one where the
true cluster means are all identical -- is then inflated by roughly 1/n per
cluster. So we compute BOTH:

  h_raw  : naive weighted between-cluster variance share (what a plain
           groupby gives you, and what will overstate heterogeneity)
  h_icc  : one-way random-effects ANOVA estimate that subtracts the
           within-cluster mean square before forming the ratio -- an
           approximately unbiased ICC, which can legitimately go negative
           when the truth is "no cluster structure at all"

`h_icc` is the one the split and the top-quartile selection use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Heterogeneity
# ---------------------------------------------------------------------------

def item_heterogeneity(
    table: pd.DataFrame,
    cluster_col: str,
    items: dict[str, list[int]],
    *,
    weight_col: str = "weight",
    min_cell_n: int = 5,
) -> pd.DataFrame:
    """Per-item between-cluster variance share, naive and debiased.

    Ordinal/binary items are scored on their numeric code scale. Nominal items
    are handled by averaging the decomposition over one-hot indicators, which
    reduces to the ordinal case for binary scales.
    """
    w_all = table[weight_col].to_numpy(float)
    cl = table[cluster_col].to_numpy()
    recs = []

    for item, scale in items.items():
        y = table[f"item_{item}"].to_numpy(dtype="float64", na_value=np.nan)
        ok = np.isfinite(y) & np.isfinite(w_all) & (w_all > 0)
        if ok.sum() < 30:
            recs.append({"item_id": item, "h_raw": np.nan, "h_icc": np.nan,
                         "n_resp": int(ok.sum()), "n_clusters_used": 0})
            continue

        yo, wo, co = y[ok], w_all[ok], cl[ok]
        # drop clusters too thin to contribute a mean
        uniq, inv = np.unique(co, return_inverse=True)
        counts = np.bincount(inv, minlength=len(uniq))
        keep_cl = counts >= min_cell_n
        keep = keep_cl[inv]
        yo, wo, inv = yo[keep], wo[keep], np.unique(co[keep], return_inverse=True)[1]
        K = inv.max() + 1 if len(inv) else 0
        if K < 2:
            recs.append({"item_id": item, "h_raw": np.nan, "h_icc": np.nan,
                         "n_resp": int(ok.sum()), "n_clusters_used": int(K)})
            continue

        h_raw, h_icc = _decompose(yo, wo, inv, K)
        recs.append({
            "item_id": item,
            "h_raw": h_raw,
            "h_icc": h_icc,
            "n_resp": int(len(yo)),
            "n_clusters_used": int(K),
        })

    return pd.DataFrame.from_records(recs)


def _decompose(y: np.ndarray, w: np.ndarray, inv: np.ndarray, K: int):
    """Weighted one-way variance decomposition."""
    W = np.bincount(inv, weights=w, minlength=K)
    S = np.bincount(inv, weights=w * y, minlength=K)
    good = W > 0
    means = np.full(K, np.nan)
    means[good] = S[good] / W[good]

    grand = float((w * y).sum() / w.sum())
    ssb = float((W[good] * (means[good] - grand) ** 2).sum())
    ssw = float((w * (y - means[inv]) ** 2).sum())
    sst = ssb + ssw
    h_raw = ssb / sst if sst > 0 else np.nan

    # random-effects ANOVA debias
    n = len(y)
    Kg = int(good.sum())
    if Kg < 2 or n - Kg < 1:
        return h_raw, np.nan
    msb = ssb / (Kg - 1)
    msw = ssw / (n - Kg)
    counts = np.bincount(inv, minlength=K)[good].astype(float)
    n0 = (counts.sum() - (counts ** 2).sum() / counts.sum()) / (Kg - 1)
    if n0 <= 0:
        return h_raw, np.nan
    var_b = (msb - msw) / n0
    denom = var_b + msw
    h_icc = var_b / denom if denom > 0 else np.nan
    return h_raw, h_icc


# ---------------------------------------------------------------------------
# Anchor / target split
# ---------------------------------------------------------------------------

def stratified_split(
    het: pd.DataFrame,
    topics: dict[str, str],
    *,
    anchor_frac: float = 0.60,
    seed: int = 20260818,
    het_col: str = "h_icc",
) -> pd.DataFrame:
    """STANDARD regime (§5.1): stratify by topic x heterogeneity quartile,
    assign 60% anchor / 40% target within each stratum."""
    rng = np.random.default_rng(seed)
    df = het.copy()
    df["topic"] = df["item_id"].map(topics)
    v = df[het_col]
    q = pd.qcut(v.rank(method="first"), 4, labels=["q1", "q2", "q3", "q4"])
    df["het_quartile"] = q.astype("object").where(v.notna(), "na")
    df["role_standard"] = "target"

    for _, grp in df.groupby(["topic", "het_quartile"], dropna=False, observed=True):
        ids = np.array(grp.index.to_numpy(), copy=True)
        rng.shuffle(ids)
        n_anchor = int(round(anchor_frac * len(ids)))
        # never let a stratum contribute zero anchors if it has >=2 items
        if len(ids) >= 2:
            n_anchor = min(max(n_anchor, 1), len(ids) - 1)
        df.loc[ids[:n_anchor], "role_standard"] = "anchor"
    return df


def adversarial_split(
    df: pd.DataFrame,
    *,
    target_topics: list[str] | None = None,
    anchor_frac: float = 0.60,
    seed: int = 20260818,
) -> pd.DataFrame:
    """ADVERSARIAL regime (§5.1): hold out ENTIRE topics. This is the regime
    that measures true out-of-domain transfer, and the one where the
    calibration layer's "anchors unrepresentative of targets" failure mode
    (§3.3 M5) actually bites.

    Topics are chosen greedily so the held-out side lands nearest to
    (1 - anchor_frac) of items, preferring a spread of topic sizes over one
    giant topic.
    """
    df = df.copy()
    sizes = df["topic"].value_counts()
    if target_topics is None:
        want = (1 - anchor_frac) * len(df)
        rng = np.random.default_rng(seed)
        order = list(sizes.index)
        rng.shuffle(order)
        chosen, tot = [], 0
        for t in sorted(order, key=lambda t: -sizes[t]):
            if tot >= want:
                break
            if sizes[t] > want:      # skip topics that would blow the budget alone
                continue
            chosen.append(t)
            tot += sizes[t]
        target_topics = chosen
    df["role_adversarial"] = np.where(df["topic"].isin(target_topics), "target", "anchor")
    df.attrs["adversarial_target_topics"] = sorted(target_topics)
    return df
