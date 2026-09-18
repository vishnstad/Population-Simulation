"""
B17 decision-support app (module M10).

    streamlit run M10_app/app.py

Every number shown here is read from the artefacts of an actual run in
``Codes/runs/<run_name>/``. Nothing is hardcoded. If a run has not produced a
given artefact the app says so and shows nothing, rather than displaying a
plausible placeholder -- a demo that invents its own benchmark numbers is worse
than no demo, because it is indistinguishable from a working system.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shared.paths import load_config, load_dotenv  # noqa: E402
from shared.llm_client import is_mock  # noqa: E402
from M3_statcards.builder import StatCardBuilder  # noqa: E402
from M6_aggregation.scope import SegmentResolver, aggregate_segment  # noqa: E402
from M7_router.router import OracleRouter  # noqa: E402
from M9_evaluation.metrics import hellinger_distance  # noqa: E402

st.set_page_config(page_title="B17 Population Simulation", page_icon="[]", layout="wide")


# ----------------------------------------------------------------------
@st.cache_resource
def load_assets(config_path: Optional[str] = None):
    load_dotenv()
    cfg = load_config(config_path)
    builder = StatCardBuilder(
        tree_path=cfg.tree_path,
        stats_path=cfg.stats_path,
        codebook_path=cfg.codebook_path,
        split_path=cfg.split_path,
        pop_stats_path=cfg.pop_stats_path,
        individual_table_path=cfg.individual_table_path,
        granularity=cfg.granularity,
    )
    resolver = SegmentResolver(cfg.individual_table_path, granularity=cfg.granularity)
    router = OracleRouter(codebook=list(builder.codebook.values()), threshold=0.85)
    return cfg, builder, resolver, router


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# ----------------------------------------------------------------------
st.title("B17 - Hierarchical Population Simulation")
st.caption(
    "Cluster-level distributional elicitation with cross-fitted anchor calibration. "
    "GSS 2024, United States."
)

try:
    cfg, builder, resolver, router = load_assets()
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load run assets: {exc}")
    st.info("Run `python preprocessing/run_pipeline.py configs/gss_main.yaml` first.")
    st.stop()

out_dir = cfg.out_dir
manifest = read_json(out_dir / "elicitation_manifest.json")
summary = read_json(out_dir / f"benchmark_summary_{cfg.split_regime}.json")

# --- global run-state banner -------------------------------------------
if manifest is None:
    st.warning(
        "**No elicitation run found.** The demographic and observed-data views below "
        "work, but nothing can be simulated yet. Run `python run_elicitation.py --stage all`."
    )
elif manifest.get("is_mock") or is_mock(manifest.get("model", "")):
    st.error(
        f"**MOCK RUN** (`{manifest.get('model')}`). The elicitations are random draws. "
        "Every simulated number and every benchmark figure in this app is noise. "
        "Re-run with a real model before showing this to anyone as a result."
    )
else:
    st.success(
        f"Live run: **{manifest.get('model')}** | "
        f"{manifest.get('n_clusters')} clusters | "
        f"{len(manifest.get('target_items', []))} held-out target items | "
        f"${manifest.get('budget', {}).get('total_usd_spent', 0):.2f} spent"
    )

with st.sidebar:
    st.header("Run configuration")
    st.write(f"**Run:** `{cfg.run_name}`")
    st.write(f"**Tree:** {cfg.granularity} (K = {len(builder.leaf_ids)})")
    st.write(f"**Split:** {cfg.split_regime}")
    st.write(f"**Model:** `{cfg.llm.get('model')}`")
    st.caption(
        "The coarse tree (K=26) is used because it is the finest partition where a "
        "majority of (cluster x item) cells clear the n_eff >= 30 scoring gate on "
        "3,309 respondents. See the K decision note in `config/run_gss2024.yaml`."
    )

tab_sim, tab_seg, tab_bench = st.tabs(
    ["Ask a question", "Segment explorer", "Benchmark results"]
)

# ======================================================================
# TAB 1 -- ask a question
# ======================================================================
with tab_sim:
    st.subheader("Ask a question of a region or segment")

    col_q, col_s = st.columns([3, 2])
    with col_q:
        question = st.text_area(
            "Question",
            value="Do you favor or oppose federal subsidies for rooftop solar panel installation?",
            height=90,
        )
        options_text = st.text_input(
            "Response options (comma-separated; used only if the question is unseen)",
            value="Strongly favor, Favor, Oppose, Strongly oppose",
        )
        options = [o.strip() for o in options_text.split(",") if o.strip()]

    with col_s:
        region = st.selectbox("Region", ["(whole country)"] + resolver.available_values("region"))
        sex = st.selectbox("Sex", ["(any)"] + resolver.available_values("sex"))
        age = st.multiselect("Age band", resolver.available_values("age_band"))

    query: Dict[str, Any] = {}
    if region != "(whole country)":
        query["region"] = region
    if sex != "(any)":
        query["sex"] = sex
    if age:
        query["age_band"] = age

    if st.button("Route and answer", type="primary"):
        segment = resolver.resolve(query)
        route = router.route(question, options=options)

        c1, c2, c3 = st.columns(3)
        c1.metric("Population covered", f"{segment.pop_share * 100:.1f}%")
        c2.metric("Survey respondents", f"{segment.n_respondents:,}")
        c3.metric("Clusters in segment", len(segment))

        if route.decision == "OBSERVED":
            st.success(
                f"**Observed.** This question matches codebook item `{route.matched_item_id}` "
                f"(cosine {route.similarity_score:.3f}). Served as an exact weighted cross-tab "
                "from microdata - no LLM call, no cost, no estimation error."
            )
            meta = builder.codebook[route.matched_item_id]
            hist = resolver.observed_histogram(route.matched_item_id, query)
            labels = meta.get("scale", {}).get("labels", [])
            if hist:
                if len(labels) != len(hist):
                    labels = [f"code {i+1}" for i in range(len(hist))]
                st.bar_chart(pd.DataFrame({"share": hist}, index=labels))
                st.dataframe(
                    pd.DataFrame({"Option": labels, "Share": [f"{p*100:.1f}%" for p in hist]}),
                    hide_index=True, use_container_width=True,
                )
            else:
                st.warning("No responses to this item within the selected segment.")
        else:
            st.info(
                f"**Unseen.** Highest codebook similarity was {route.similarity_score:.3f}, "
                "below the 0.85 threshold, so this routes to the simulation path."
            )
            st.markdown(
                "Simulating from the app is intentionally not wired up - it would spend "
                "API budget on every button press. Run it from the command line, where the "
                "cost is explicit:"
            )
            seg_arg = ",".join(f"{k}={v if isinstance(v, str) else '/'.join(v)}"
                               for k, v in query.items()) or "region=south"
            st.code(
                f'python simulate_region.py --segment "{seg_arg}" \\\n'
                f'    --question "{question}" \\\n'
                f'    --options "{",".join(options)}"',
                language="bash",
            )

# ======================================================================
# TAB 2 -- segment explorer (pure observed data, always available)
# ======================================================================
with tab_seg:
    st.subheader("How do subgroups differ on questions the survey does ask?")
    st.caption(
        "This view is entirely observed microdata - the skyline the simulation is "
        "trying to reproduce for unseen questions."
    )

    items = sorted(builder.codebook.keys())
    default = items.index("natheal") if "natheal" in items else 0
    item_id = st.selectbox(
        "Survey item", items, index=default,
        format_func=lambda i: f"{i} - {(builder.codebook[i].get('text') or '')[:60]}",
    )

    meta = builder.codebook[item_id]
    labels = meta.get("scale", {}).get("labels", [])
    st.write(f"**Question:** {meta.get('text', item_id)}")

    national = resolver.observed_histogram(item_id, {})
    if national is None:
        st.warning("No data for this item.")
    else:
        rows = []
        for cid in builder.leaf_ids:
            stat = builder.stats_dict.get((cid, item_id))
            node = builder.nodes_by_id.get(cid, {})
            if stat is None or len(stat.hist) != len(national):
                continue
            hist = [float(x) for x in stat.hist]
            rows.append(
                {
                    "Subgroup": node.get("definition_text", cid),
                    "Divergence from national": round(hellinger_distance(hist, national), 3),
                    "Modal answer": labels[int(np.argmax(hist))] if labels else int(np.argmax(hist)),
                    "Modal share": f"{max(hist) * 100:.0f}%",
                    "n_eff": round(float(stat.n_eff), 1),
                    "Scoreable": "yes" if float(stat.n_eff) >= cfg.evaluation.get("truth_min_neff", 30) else "no",
                    "Pop share": f"{float(node.get('pop_share', 0)) * 100:.1f}%",
                }
            )
        df = pd.DataFrame(rows).sort_values("Divergence from national", ascending=False)
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.caption(
            f"'Scoreable = no' means this cell's effective sample size is below "
            f"{cfg.evaluation.get('truth_min_neff', 30)}, so its 'truth' is too noisy to "
            "benchmark against and it is excluded from scoring."
        )

# ======================================================================
# TAB 3 -- benchmark results
# ======================================================================
with tab_bench:
    st.subheader("Measured out-of-sample performance")

    if summary is None:
        st.warning(
            "**No benchmark has been run for this configuration.** There are no results "
            "to show. Run `python run_elicitation.py --stage all` then `python benchmark.py`."
        )
    else:
        if summary.get("is_mock_run"):
            st.error("These figures come from a MOCK run. They are noise, not results.")

        head = pd.DataFrame(summary["headline"])
        st.dataframe(head, hide_index=True, use_container_width=True)
        st.caption(
            "**W1** Wasserstein-1 to true cluster histograms (lower is better). "
            "**var_ratio** predicted/true within-cluster SD, target [0.8, 1.2]. "
            "**between_SD_ratio** predicted/true spread across clusters - near zero means "
            "the model gives every subgroup the same answer. **rank_rho** Spearman "
            "correlation of subgroup means."
        )

        verdict = summary.get("verdict", {})
        c1, c2, c3 = st.columns(3)
        if "w1_improvement_vs_b0a_pct" in verdict:
            gain = verdict["w1_improvement_vs_b0a_pct"]
            c1.metric("W1 vs national marginal", f"{gain:+.1f}%", delta=f"target +20%")
            c1.write("**PASS**" if verdict.get("claim_i_met") else "**FAIL**")
        vr = verdict.get("variance_ratio")
        if vr is not None:
            c2.metric("Variance ratio", f"{vr:.3f}", delta="target 0.8-1.2")
            c2.write("**PASS**" if verdict.get("claim_ii_met") else "**FAIL**")
        ours = next((h for h in summary["headline"] if "ours" in h["system"]), None)
        if ours:
            c3.metric("Between-cluster SD ratio", f"{ours['between_SD_ratio']:.3f}")
            c3.write("**collapse warning**" if ours["between_SD_ratio"] < 0.5 else "signal present")

        st.markdown(
            f"**Scored on** {summary['n_items_scored']} items "
            f"({summary['n_items_skipped']} skipped: ground truth too thin at "
            f"n_eff >= {summary['min_neff']}). Split: `{summary['split_regime']}`."
        )

        per_item_path = out_dir / f"benchmark_per_item_{cfg.split_regime}.csv"
        if per_item_path.exists():
            with st.expander("Per-item detail"):
                st.dataframe(pd.read_csv(per_item_path), hide_index=True, use_container_width=True)

        cal = summary.get("calibrator", {})
        if cal:
            with st.expander("Calibrator fit"):
                st.json(cal)
                st.caption(
                    "`raw_variance_ratio_before_calibration` is the dispersion of the raw "
                    "LLM output relative to truth. Below 1.0 is the variance collapse the "
                    "calibration layer exists to repair."
                )
