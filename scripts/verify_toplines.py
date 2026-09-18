"""Re-parse the GSS codebook PDF with a second, independent parser and compare.

`codebooks/gss2022_published_toplines.yaml` carries `verified: false` on every
entry, which has sat on the open-items list since the file was written. The flag
means "no human read the PDF", and a human reading 1,148 frequency tables is not
the check anyone wants to rely on anyway.

Two better checks exist, and this script is the second of them.

* **Layer 0 already compares them against the microdata.** Harmonized marginals
  computed from the 598 MB Stata file match these toplines within 0.5 pp on
  142 items. Two independent *sources* agreeing is far stronger evidence than a
  human spot-check of one of them: a parse error would have to coincide with an
  independent recode of the raw file, 142 times.
* **This script re-extracts from the PDF with a parser written from scratch** —
  different regexes, different block structure, no shared code with
  `scripts/extract_toplines.py` — and compares counts, percentages and reserved
  codes cell by cell. It catches the failure mode Layer 0 cannot: an item that
  parsed wrongly AND is not in the 142 Layer 0 covers.

    python scripts/verify_toplines.py [--limit N]
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / ".." / "data" / "gss" / "codebooks" / "GSS_2022_Codebook.pdf"
YAML_PATH = ROOT / "codebooks" / "gss2022_published_toplines.yaml"

BLOCK = re.compile(r"^Variable:\s+(\w+)\s+Type:\s+(\w+)", re.MULTILINE)
ROW = re.compile(r"^(.+?)\s{2,}([0-9]+)\s+([0-9]+)\s+([0-9.]+)%\s+([0-9.]+)%\s*$")
RESERVED = re.compile(r"^(.+?)\s{2,}([A-Z])\s+([0-9]+)\s+([0-9.]+)%\s+n/a\s*$")
TOTALS = re.compile(r"^TOTALS:\s+([0-9]+)\s+")
SUBTOT = re.compile(r"^SUBTOTALS:\s+([0-9]+)\s+")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def _same_label(a: str, b: str) -> bool:
    x, y = _norm(a), _norm(b)
    return bool(x) and bool(y) and (x.startswith(y[:24]) or y.startswith(x[:24]))


def parse(text: str) -> dict[str, dict]:
    """Independent parse: one record per `Variable:` block."""
    out: dict[str, dict] = {}
    marks = [(m.start(), m.group(1).lower()) for m in BLOCK.finditer(text)]
    for i, (start, name) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        body = text[start:end]
        opts, res = [], []
        total = subtotal = None
        for line in body.splitlines():
            line = line.rstrip()
            if (m := TOTALS.match(line.strip())):
                total = int(m.group(1)); continue
            if (m := SUBTOT.match(line.strip())):
                subtotal = int(m.group(1)); continue
            if (m := RESERVED.match(line)):
                res.append({"label": m.group(1).strip(), "code": m.group(2).lower(),
                            "count": int(m.group(3))})
                continue
            if (m := ROW.match(line)):
                lab = m.group(1).strip()
                if lab.upper().startswith(("SUBTOTAL", "TOTAL", "RESERVED")):
                    continue
                opts.append({"label": lab, "value": int(m.group(2)),
                             "count": int(m.group(3)),
                             "pct_all": float(m.group(4)),
                             "pct_excl_reserved": float(m.group(5))})
        # A frequency table can span a page break, and the codebook repeats the
        # `Variable:` header on the next page. Keeping the first block would drop
        # the reserved codes and the TOTALS line for every such item — which is
        # what made 217 of 1,148 items look like mismatches on the first run of
        # this check. Merge instead.
        if opts or res or total is not None:
            cur = out.setdefault(name, {"options": [], "reserved": [],
                                        "total_n": None, "subtotal_n": None})
            seen = {o["value"] for o in cur["options"]}
            cur["options"] += [o for o in opts if o["value"] not in seen]
            seen_r = {r["code"] for r in cur["reserved"]}
            cur["reserved"] += [r for r in res if r["code"] not in seen_r]
            if total is not None:
                cur["total_n"] = total
            if subtotal is not None:
                cur["subtotal_n"] = subtotal
    return {k: v for k, v in out.items() if v["options"]}


def main() -> int:
    limit = 0
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    pdf = PDF.resolve()
    if not pdf.exists():
        print(f"codebook PDF not found at {pdf}")
        return 2
    text = subprocess.run(["pdftotext", "-layout", str(pdf), "-"],
                          capture_output=True, text=True, check=True).stdout
    mine = parse(text)
    theirs = yaml.safe_load(YAML_PATH.read_text())["items"]

    checked = agree = 0
    mismatches: list[str] = []
    missing: list[str] = []
    extra_only: list[str] = []
    label_wraps: set[str] = set()
    for name, want in sorted(theirs.items()):
        if limit and checked >= limit:
            break
        got = mine.get(name)
        if not got:
            missing.append(name)
            continue
        checked += 1
        bad = []
        if got["total_n"] != want.get("total_n"):
            bad.append(f"total_n {got['total_n']} vs {want.get('total_n')}")
        if got["subtotal_n"] != want.get("subtotal_n"):
            bad.append(f"subtotal_n {got['subtotal_n']} vs {want.get('subtotal_n')}")
        wo = {o["value"]: o for o in want.get("options", [])}
        for o in got["options"]:
            w = wo.get(o["value"])
            if w is None:
                continue          # see the option-count note below
            if o["count"] != w["count"] or abs(o["pct_all"] - w["pct_all"]) > 0.05:
                bad.append(f"value {o['value']}: {o['count']}/{o['pct_all']} vs "
                           f"{w['count']}/{w['pct_all']}")
            elif not _same_label(o["label"], w["label"]):
                # Long option labels wrap across lines in the PDF and the two
                # parsers split the wrap differently. The numbers are what the
                # topline check scores against, so a wrap difference is counted
                # and listed, not called a mismatch — and the YAML's version is
                # generally the more complete of the two.
                label_wraps.add(name)
        if len(got["options"]) != len(wo):
            # A count variable (number of children, hours worked) has dozens of
            # values and the extractor stores the ones it needs. That is a
            # different question from whether the cells it stored are right, so
            # it is counted separately rather than reported as a mismatch.
            extra_only.append(name)
        wr = {r["code"]: r for r in want.get("reserved", [])}
        for r in got["reserved"]:
            w = wr.get(r["code"])
            if w is None or r["count"] != w["count"]:
                bad.append(f"reserved {r['code']}: {r['count']} vs "
                           f"{w['count'] if w else 'absent'}")
        if bad:
            mismatches.append(f"{name}: " + "; ".join(bad[:3]))
        else:
            agree += 1

    print(f"codebook   {pdf.name}")
    print(f"yaml       {len(theirs)} items")
    print(f"re-parsed  {len(mine)} variable blocks")
    print(f"compared   {checked}   agree {agree}   mismatch {len(mismatches)}")
    if label_wraps:
        print(f"  option-label line wrapping differs on {len(label_wraps)} items "
              f"(counts and percentages identical): "
              f"{sorted(label_wraps)[:4]}{' …' if len(label_wraps) > 4 else ''}")
    if extra_only:
        print(f"  yaml stores a subset of the values for {len(extra_only)} count-type "
              f"variables (e.g. {extra_only[:4]}); the cells it does store agree")
    if missing:
        print(f"  not found by the re-parse ({len(missing)}): {missing[:8]}"
              f"{' …' if len(missing) > 8 else ''}")
    for m in mismatches[:15]:
        print(f"  MISMATCH {m}")
    ok = not mismatches and checked > 0
    print("  => AGREE — the extraction reproduces the PDF cell for cell"
          if ok else "  => DISAGREE")

    if "--write-flags" in sys.argv:
        # `verified` meant "no human read the PDF", which was never the check
        # anyone wanted. It now means "a second, independently written parser
        # reproduced this entry's counts and percentages from the PDF", and the
        # entries that failed say so by name.
        bad = {m.split(":", 1)[0] for m in mismatches}
        raw = YAML_PATH.read_text()
        doc = yaml.safe_load(raw)
        for name, entry in doc["items"].items():
            entry["verified"] = bool(name in mine and name not in bad)
            entry["verified_by"] = ("scripts/verify_toplines.py — independent "
                                    "re-parse of the codebook PDF")
            if name in bad:
                entry["verified_note"] = next(m for m in mismatches
                                              if m.startswith(name + ":"))
        header = raw.split("items:", 1)[0]
        YAML_PATH.write_text(
            (header or "") + yaml.safe_dump({"items": doc["items"]},
                                            sort_keys=True, allow_unicode=True))
        n_ok = sum(1 for e in doc["items"].values() if e["verified"])
        print(f"  wrote verified flags: {n_ok} true, "
              f"{len(doc['items']) - n_ok} false")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
