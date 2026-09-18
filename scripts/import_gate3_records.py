"""Fold a Gate 3 run's real-arm records into the elicitation store.

The four Gate 3 runs each spent 480-960 real calls on target items under exactly
the card contract Phase 4 uses — same split, same folds, same response contract,
same paraphrase and repeat indices. Re-eliciting them would spend quota to
reproduce records already on disk, so they are imported instead.

Only the **real** arm is imported: a permuted-arm record was produced from
another cluster's card and belongs to the permutation test, not to a prediction
for the cluster it is filed under.

Refuses a run whose config snapshot does not match the current one on the things
that change what a card contains — the split, the amendment state, the fold
exclusion and the anchors per card. Importing across those would put records
produced under one contract into a store the harness reads as another.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from popsim.agents.runner import ElicitationStore
from popsim.config import load_config

#: Runs whose snapshot predates a config key but whose contract is recorded
#: elsewhere. `enforce_fold_exclusion` and `anchor_pick` were added to the config
#: on 16 Sep AFTER run 3 had already been made with the fold exclusion enforced
#: in code; PREREGISTRATION.md §1 records that run as "3.4 fold exclusion
#: enforced" and §2 explains why it was retained. The snapshot cannot show a key
#: that did not exist, so the exemption is named here, per run, with its
#: evidence — rather than by loosening the check for everything.
TRUSTED = {
    "gss_main__20260916T081708Z": (
        "PREREGISTRATION.md §1 run 3: fold exclusion enforced, keyed response "
        "contract, un-amended split. The config keys postdate the run."
    ),
}

CONTRACT_KEYS = [
    ("elicitation", "anchors_per_card"),
    ("elicitation", "enforce_fold_exclusion"),
    ("elicitation", "anchor_pick"),
    ("elicitation", "temperature"),
    ("item_split", "freeze_path"),
    ("calibration", "crossfit_folds"),
    ("partition", "min_cell"),
    ("partition", "k_target"),
]


def check(snapshot: dict, cfg) -> list[str]:
    bad = []
    for sect, key in CONTRACT_KEYS:
        want = cfg[f"{sect}.{key}"]
        got = (snapshot.get(sect) or {}).get(key)
        if got is None:
            # A missing key is a refusal, not a pass. `enforce_fold_exclusion`
            # did not exist before 16 Sep, and the runs that predate it built
            # cards that included the target's own fold — a different contract
            # wearing the same name.
            bad.append(f"{sect}.{key}: absent from the run snapshot")
        elif got != want:
            bad.append(f"{sect}.{key}: run has {got!r}, config has {want!r}")
    return bad


def main(run_dirs: list[str], *, profile: str = "permutation", apply: bool = False) -> None:
    cfg = load_config("configs/gss_main.yaml")
    root = cfg.repo_root / cfg["paths.runs_root"] / "_elicit_store"
    for rd in run_dirs:
        d = Path(rd)
        raw_path = d / "permutation_raw.parquet"
        if not raw_path.exists():
            print(f"{d.name}: no permutation_raw.parquet — skipped")
            continue
        snap = yaml.safe_load((d / "config.snapshot.yaml").read_text()) \
            if (d / "config.snapshot.yaml").exists() else {}
        problems = check(snap, cfg)
        if problems and d.name in TRUSTED:
            print(f"   trusted despite {len(problems)} absent key(s): {TRUSTED[d.name]}")
            problems = [p for p in problems if "absent from the run snapshot" not in p]
        df = pd.read_parquet(raw_path)
        real = df[df["permuted_from"].isna()]
        n_amd = len(json.loads(json.dumps(snap.get("item_split", {}))).get("amendments", []) or [])
        model = str(real["model"].iloc[0]) if len(real) else "?"
        print(f"{d.name}: {len(real)} real records, model {model}")
        if problems:
            for p in problems:
                print(f"   REFUSED — {p}")
            continue
        if n_amd:
            print("   REFUSED — run used an amended split")
            continue
        if not apply:
            print("   (dry run; pass --apply to write)")
            continue
        store = ElicitationStore(root=root, model=model, profile=profile)
        added = 0
        for item_id, g in real.groupby("item_id"):
            have = store.done(item_id)
            recs = []
            for r in g.to_dict("records"):
                key = (r["cluster_id"], int(r["paraphrase_id"]), int(r["repeat_id"]))
                if key in have:
                    continue
                have.add(key)
                recs.append(type("R", (), {"item_id": item_id, "__dict__": r})())
            if recs:
                store.dir.mkdir(parents=True, exist_ok=True)
                with store.path(item_id).open("a") as fh:
                    for rec in recs:
                        row = dict(rec.__dict__)
                        row["hist"] = (list(map(float, row["hist"]))
                                       if row.get("hist") is not None else None)
                        for k, v in list(row.items()):
                            if hasattr(v, "item"):
                                row[k] = v.item()
                            elif v is not None and str(type(v)).startswith("<class 'numpy"):
                                row[k] = str(v)
                            elif isinstance(v, float) and math.isnan(v):
                                row[k] = None
                        fh.write(json.dumps(row, default=str) + "\n")
                added += len(recs)
        print(f"   imported {added} records into {store.dir}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args, apply="--apply" in sys.argv)
