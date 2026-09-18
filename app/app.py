"""Stage A — the demo surface (checklist 6.4-6.6, spec §M10).

    "pick country -> author scenario -> see population + segment distributions
     with intervals, top divergent segments, and a mandatory **honesty box**.
     **No report renders without it.**"

The honesty box is not a footer here; it is rendered before the numbers, because
a reader who sees a distribution first has already believed it. Every simulated
answer carries what the same system scored on the most similar questions it was
actually measured on, and a question the survey already answers never reaches a
model at all — that is the router, and it is the architectural answer to F3.

    streamlit run app/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from popsim.agents.elicit import load_paraphrases
from popsim.agents.runner import ElicitationStore
from popsim.aggregate.mixture import aggregate
from popsim.calibration.fit import apply_calibrator
from popsim.config import load_config
from popsim.evalx.harness import collect_cells, fit_from_store
from popsim.evalx.metrics import w1
from popsim.pipeline import build_bed
from popsim.report.honesty import build_honesty_box
from popsim.scenarios.router import Router
from popsim.scenarios.schema import Scenario, compile_scenario

REPO = Path(__file__).resolve().parents[1]

st.set_page_config(page_title="B17 — population simulation", layout="wide")


@st.cache_resource(show_spinner="Building the bed…")
def _load(model: str, profile: str):
    cfg = load_config(REPO / "configs" / "gss_main.yaml")
    bed = build_bed(cfg)
    codebook = bed.codebook
    router = Router(codebook, threshold=float(cfg["router.threshold"]))
    store = ElicitationStore(root=cfg.repo_root / "runs" / "_elicit_store",
                             model=model, profile=profile)
    anchors = collect_cells(store.load(sorted(bed.split["anchors"])))
    cal = None
    if anchors:
        cal, _ = fit_from_store(bed, anchors, min_neff=cfg["evaluation.truth_min_neff"])
    reports = sorted((cfg.repo_root / "runs").glob("*/layer4_report.json"),
                     key=lambda p: p.stat().st_mtime)
    layer4 = None
    for p in reversed(reports):
        d = json.loads(p.read_text())
        if d.get("model") == model:
            layer4 = d
            break
    return cfg, bed, codebook, router, store, cal, layer4


def _available_models() -> list[str]:
    root = REPO / "runs" / "_elicit_store"
    return sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []


def _hist_table(labels, hist, lo=None, hi=None) -> pd.DataFrame:
    d = {"option": list(labels), "share": [f"{100 * x:.1f}%" for x in hist]}
    if lo is not None:
        d["90% interval"] = [f"{100 * a:.1f} – {100 * b:.1f}%" for a, b in zip(lo, hi, strict=True)]
    return pd.DataFrame(d)


st.title("B17 — what would this group say?")
st.caption(
    "Cluster-level, distribution-valued elicitation with cross-fitted anchor "
    "calibration. Built on the GSS, pooled 2010–2022, 56 demographic clusters."
)

models = _available_models()
if not models:
    st.error("No elicitation store on disk. Run `popsim elicit` first.")
    st.stop()

with st.sidebar:
    st.header("Run")
    model = st.selectbox("Elicitation arm", models,
                         index=len(models) - 1, key="model")
    profile = st.selectbox("Ensemble profile", ["permutation", "minieval", "headline", "dev"],
                           index=0, key="profile")
    st.caption("Arms differ only by model. Everything else — the cards, the split, "
               "the folds, the calibrator's selection rule — is identical.")

cfg, bed, codebook, router, store, cal, layer4 = _load(model, profile)

if cal is None:
    st.error(f"No anchor elicitations for `{model}` / `{profile}`, so the calibration "
             f"layer cannot be fitted. Run `popsim elicit --scope anchors`.")
    st.stop()

scored_items = sorted({r["item_id"] for r in (layer4 or {}).get("per_item", [])})
mode = st.radio("What do you want to ask?",
                ["A validated survey question", "A new question of my own"],
                horizontal=True, key="mode")

if mode == "A validated survey question":
    item_id = st.selectbox("Question", scored_items or sorted(bed.split["targets"]),
                           index=(scored_items.index(cfg["evaluation.demo_item"])
                                  if cfg["evaluation.demo_item"] in scored_items else 0),
                           key="item")
    question = codebook[item_id]["text"]
    labels = codebook[item_id]["labels"]
else:
    question = st.text_area(
        "The question, worded exactly as you would ask it",
        "Would you sign up for a pension scheme that automatically saves 2% of "
        "every payment you receive through a payments app?", key="q")
    raw = st.text_input("Answer options, comma separated",
                        "definitely would, probably would, probably would not, "
                        "definitely would not", key="labels")
    labels = [x.strip() for x in raw.split(",") if x.strip()]
    item_id = None

decision = router.route(question)
st.subheader("1 · Does the survey already answer this?")
(st.success if decision.observed else st.info)(decision.explain())

box = build_honesty_box(question, router, layer4,
                        provenance="observed" if decision.observed else "simulated")
st.subheader("2 · How much to trust what follows")
(st.success if decision.observed else st.warning)(box.sentence())
if box.note:
    st.caption(box.note)
if box.nearest:
    st.dataframe(pd.DataFrame(box.nearest)[["item_id", "similarity", "w1", "baseline"]]
                 .rename(columns={"w1": "system W1", "baseline": "B0a W1"}),
                 hide_index=True, width="stretch")

st.subheader("3 · The answer")
clusters = bed.clusters(0)

if decision.observed and decision.matched_item in codebook:
    served = decision.matched_item
    labels = codebook[served]["labels"]
    cids = [c for c in clusters if bed.stats.hist(c, served) is not None
            and bed.stats.hist(c, served).sum() > 0]
    dists = {c: bed.stats.hist(c, served) for c in cids}
    pop = aggregate(dists, bed.raked_weight)
    st.caption(f"Served from microdata as `{served}` — provenance **observed**. "
               f"Weights raked to ACS 2024.")
elif item_id and (cells := collect_cells(store.load([item_id]))):
    raw_by_cluster = {c: cells[(item_id, c)].mean(axis=0)
                      for c in clusters if (item_id, c) in cells}
    cids = list(raw_by_cluster)
    level = bed.level_hist(item_id, cids)
    wt = {c: bed.cell_weight(c, item_id) for c in cids}
    dists = apply_calibrator(cal, raw_by_cluster, level, wt,
                             topic=bed.topics.get(item_id))
    pop = aggregate(dists, bed.raked_weight)
    st.caption(f"Replayed from the stored elicitation for `{item_id}` — provenance "
               f"**simulated**. {len(cids)} clusters, calibrated, raked to ACS 2024.")
else:
    st.info("This question has no stored elicitation. Press the button to spend "
            "live API calls on it — one per cluster, per ensemble member.")
    n_clusters = st.slider("Clusters to elicit (largest first)", 4, len(clusters), 12,
                           key="ncl")
    if not st.button("Elicit", type="primary"):
        st.stop()
    from popsim.agents.elicit import elicit_distribution
    from popsim.llm.client import LLMClient

    sc = Scenario(scenario_id="adhoc", decision_question=question, labels=labels)
    item = compile_scenario(sc)
    client = LLMClient.from_config(cfg)
    paras = load_paraphrases()
    cids = clusters[:n_clusters]
    raw_by_cluster = {}
    bar = st.progress(0.0, "eliciting…")
    for i, cid in enumerate(cids):
        card = bed.card(bed.ranked_targets()[0], cid)     # anchors only; no target leaks
        recs = elicit_distribution(card=card, item=item, client=client,
                                   paraphrases=paras, n_paraphrase=1, n_repeat=3,
                                   temperature=cfg["elicitation.temperature"])
        good = [np.asarray(r.hist, float) for r in recs if r.ok and r.hist]
        if good:
            raw_by_cluster[cid] = np.mean(np.vstack(good), axis=0)
        bar.progress((i + 1) / len(cids), f"{i + 1}/{len(cids)} clusters")
    bar.empty()
    if not raw_by_cluster:
        st.error("Every call failed. Run `popsim doctor --live`.")
        st.stop()
    cids = list(raw_by_cluster)
    # No observed level exists for a question the survey never asked, so the
    # level is the model's own population-level answer: the `predicted_level`
    # mode, and the one the honesty box's number is about.
    pop_card = bed.population_card(bed.ranked_targets()[0])
    recs = elicit_distribution(card=pop_card, item=item, client=client,
                               paraphrases=paras, n_paraphrase=1, n_repeat=3,
                               temperature=cfg["elicitation.temperature"])
    good = [np.asarray(r.hist, float) for r in recs if r.ok and r.hist]
    level = np.mean(np.vstack(good), axis=0) if good else np.full(len(labels), 1 / len(labels))
    wt = {c: bed.cluster_weight.get(c, 1.0) for c in cids}
    dists = apply_calibrator(cal, raw_by_cluster, level, wt)
    pop = aggregate(dists, bed.raked_weight)
    st.caption(f"Elicited live over {len(cids)} clusters. The population level is the "
               f"model's own — no survey topline exists for this question.")

c1, c2 = st.columns([1, 1])
with c1:
    st.markdown("**Population**")
    st.dataframe(_hist_table(labels, pop), hide_index=True, width="stretch")
    st.bar_chart(pd.DataFrame({"share": pop}, index=list(labels)))

with c2:
    st.markdown("**The groups that differ most from the population**")
    rows = []
    for c, h in dists.items():
        rows.append({
            "group": bed.card(bed.ranked_targets()[0], c).definition_text,
            "distance from population (W1)": round(float(w1(np.asarray(h), pop)), 4),
            "population share": f"{100 * bed.raked_weight.get(c, 0.0):.2f}%",
            **{str(lab): f"{100 * v:.0f}%" for lab, v in zip(labels, h, strict=False)},
        })
    df = pd.DataFrame(rows).sort_values("distance from population (W1)", ascending=False)
    st.dataframe(df.head(8), hide_index=True, width="stretch")

with st.expander("What this does not answer"):
    st.markdown(
        "Product and policy scenarios have no ground truth and never will. The "
        "held-out-item score above is the honest proxy, and it is the only "
        "accuracy claim made anywhere in this app.\n\n"
        "This system does **not** simulate society, predict policy outcomes or "
        "replace surveys — it is built on them and dies without them. Outputs are "
        "aggregate only; no synthetic individuals are generated, and the "
        "elicitation framing is statistical throughout, never *speak as*."
    )
