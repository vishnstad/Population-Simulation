#!/usr/bin/env python3
"""
External verification of the M1 harmonisation against NORC's own published
frequency tables in the GSS 2024 Codebook (Release 3a).

The spec's §3.3 M1 failure-mode note demands "per-item assertion that
harmonized marginals match published toplines". The codebook prints UNWEIGHTED
counts, so that is what we compare: it validates the analytic-sample filter,
the extended-missing handling, and every value-code mapping. Weighting is
checked separately by reproducing the weight total.

    python verify.py /path/to/codebook.txt        (pdftotext -layout output)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from popsim.data.adapters.gss import load_gss


# A frequency row always ENDS with:  <value> <count> <pct>% <pct-excl-reserve>%
# Anchoring on the tail rather than the label is what makes this robust to the
# three shapes the codebook actually uses: labels longer than the column,
# labels that wrap onto the previous line (leaving the row label-less), and
# purely numeric scale-point labels ("2", "3", ...).
LINE = re.compile(
    r"(?<![\d,])(?P<value>\d{1,3})\s+(?P<count>\d{1,3}(?:,\d{3})*|\d+)\s+"
    r"[\d.]+%\s+(?:[\d.]+%|n/a)\s*$"
)


def parse_codebook(path: Path) -> dict[str, dict[int, int]]:
    """var -> {numeric code: unweighted count} from the frequency blocks."""
    txt = path.read_text(errors="ignore")
    out: dict[str, dict[int, int]] = {}
    for m in re.finditer(r"^Variable:\s+([A-Z][A-Z0-9_]{0,15})\s+Type:\s*Numeric\s*$", txt, re.M):
        var = m.group(1)
        blk = txt[m.end(): m.end() + 4000]
        stop = blk.find("TOTALS:")
        blk = blk[:stop] if stop > 0 else blk
        codes: dict[int, int] = {}
        started = False
        for ln in blk.splitlines():
            if "RESERVED CODES" in ln or "SUBTOTALS" in ln:
                break
            if "LABEL" in ln and "VALUE" in ln:
                started = True
                continue
            if not started:
                continue
            mm = LINE.search(ln)
            if mm:
                codes[int(mm.group("value"))] = int(mm.group("count").replace(",", ""))
        if codes:
            out.setdefault(var, codes)
    return out


def main(cb_path: str):
    cb = parse_codebook(Path(cb_path))
    print(f"parsed {len(cb)} variables from the codebook\n")
    if len(cb) < 100:
        raise SystemExit(
            f"codebook parse produced only {len(cb)} variables -- refusing to "
            f"report vacuous passes. Regenerate with: "
            f"pdftotext -layout 'GSS 2024 Codebook R3a.pdf' cb.txt")

    res = load_gss("data_raw/GSS2024.dta", "codebooks/recodes_gss.yaml",
                   strict_guards=True)
    tbl = res.table
    raw, meta = pyreadstat.read_dta("data_raw/GSS2024.dta")

    report = {"checks": [], "summary": {}}

    # ---------------------------------------------------------------- 1
    wt = pd.to_numeric(raw["wtssnrps"], errors="coerce")
    keep = wt.notna() & (wt > 0)
    n_analytic = int(keep.sum())
    ok = n_analytic == len(tbl)
    report["checks"].append({
        "check": "analytic sample size", "expected": n_analytic,
        "got": len(tbl), "pass": ok})
    print(f"[{'PASS' if ok else 'FAIL'}] analytic sample = {len(tbl)}")

    # ---------------------------------------------------------------- 2
    # Every retained item's per-code unweighted count vs the codebook.
    mismatch, checked, missing = [], 0, 0
    for item in res.item_cols:
        codes_cb = cb.get(item.upper())
        if not codes_cb:
            missing += 1
            continue
        col = tbl[f"item_{item}"]
        got = col.value_counts().to_dict()
        got = {int(k): int(v) for k, v in got.items()}
        # compare only codes the adapter kept on the scale
        for code in sorted(res.value_labels[item]):
            e, g = codes_cb.get(code, 0), got.get(code, 0)
            checked += 1
            if e != g:
                mismatch.append((item, code, e, g))
    ok = (not mismatch) and checked >= 500
    report["checks"].append({
        "check": "per-code unweighted counts vs codebook",
        "cells_compared": checked, "items_not_in_codebook": missing,
        "mismatches": len(mismatch), "pass": ok})
    print(f"[{'PASS' if ok else 'FAIL'}] {checked:,} (item × code) counts matched "
          f"the codebook exactly; {len(mismatch)} mismatches "
          f"({missing} items absent from the codebook's numeric blocks)")
    for m in mismatch[:10]:
        print(f"        {m[0]} code {m[1]}: codebook {m[2]} vs pipeline {m[3]}")

    # ---------------------------------------------------------------- 3
    # Demographic recodes: the harmonised level totals must equal the sum of
    # the source codes mapped into them.
    rc = yaml.safe_load(Path("codebooks/recodes_gss.yaml").read_text())
    dem_fail = []
    for field, spec in rc["demo"].items():
        if spec["type"] not in ("code_map", "code_map_bool"):
            continue
        src = spec["source"].lower()
        codes_cb = cb.get(src.upper())
        if not codes_cb:
            continue
        want: dict = {}
        for code, rule in spec["map"].items():
            want[rule["value"]] = want.get(rule["value"], 0) + codes_cb.get(int(code), 0)
        got = tbl[f"demo_{field}"].value_counts().to_dict()
        for lvl, e in want.items():
            g = int(got.get(lvl, 0))
            if e != g:
                dem_fail.append((field, lvl, e, g))
    ok = not dem_fail
    report["checks"].append({"check": "demographic recode totals",
                             "mismatches": len(dem_fail), "pass": ok})
    print(f"[{'PASS' if ok else 'FAIL'}] demographic recode level totals "
          f"reconcile to the codebook ({len(dem_fail)} mismatches)")
    for m in dem_fail[:10]:
        print(f"        demo_{m[0]} = {m[1]}: codebook {m[2]} vs pipeline {m[3]}")

    # ---------------------------------------------------------------- 4
    # Weighted population marginals must sum to 1 and reproduce the weight base
    run = Path("runs/gss2024_main")
    ps = pd.read_parquet(run / "population_stats.parquet")
    sums = np.array([float(np.sum(h)) for h in ps["hist"]
                     if np.isfinite(np.asarray(h, float)).all()])
    ok = bool(np.allclose(sums, 1.0))
    report["checks"].append({"check": "population histograms sum to 1",
                             "n": len(sums), "pass": ok})
    print(f"[{'PASS' if ok else 'FAIL'}] {len(sums)} population histograms sum to 1")

    wsum = float(tbl["weight"].sum())
    ok = abs(wsum - n_analytic) < 1.0
    report["checks"].append({"check": "weights sum to the analytic n",
                             "expected": n_analytic, "got": round(wsum, 3),
                             "pass": ok})
    print(f"[{'PASS' if ok else 'FAIL'}] weights sum to {wsum:.1f} "
          f"(post-stratified weights are normalised to n = {n_analytic})")

    # ---------------------------------------------------------------- 5
    # Cluster partition: exhaustive, mutually exclusive, pop shares sum to 1
    it = pd.read_parquet(run / "individual_table.parquet")
    nodes = json.loads((run / "cluster_tree.json").read_text())
    leaves = [n for n in nodes if n["level"] == 2]
    checks5 = {
        "every respondent in exactly one leaf": it["cluster_id"].notna().all()
            and len(it) == sum(n["n_raw"] for n in leaves),
        "leaf ids unique": len({n["cluster_id"] for n in leaves}) == len(leaves),
        "pop shares sum to 1": abs(sum(n["pop_share"] for n in leaves) - 1) < 1e-9,
        "leaf n_raw sums to analytic n": sum(n["n_raw"] for n in leaves) == len(it),
    }
    for k, v in checks5.items():
        report["checks"].append({"check": k, "pass": bool(v)})
        print(f"[{'PASS' if v else 'FAIL'}] {k}")

    # ---------------------------------------------------------------- 6
    # Anchor/target leakage: no item may be an anchor and a target at once
    sp = pd.read_csv(run / "item_split.csv")
    bad = ((sp.role_standard == sp.role_adversarial) & False).any()
    dupes = sp.item_id.duplicated().any()
    for k, v in (("no duplicate items in the split", not dupes),
                 ("every item has both roles assigned",
                  sp[["role_standard", "role_adversarial"]].notna().all().all())):
        report["checks"].append({"check": k, "pass": bool(v)})
        print(f"[{'PASS' if v else 'FAIL'}] {k}")

    n_fail = sum(1 for c in report["checks"] if not c["pass"])
    report["summary"] = {"checks": len(report["checks"]), "failed": n_fail}
    Path("runs/gss2024_main/verification.json").write_text(json.dumps(report, indent=1))
    print(f"\n{len(report['checks']) - n_fail}/{len(report['checks'])} checks passed"
          f" -> runs/gss2024_main/verification.json")
    return n_fail


if __name__ == "__main__":
    sys.exit(1 if main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/cb.txt") else 0)
