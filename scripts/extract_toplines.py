#!/usr/bin/env python
"""Extract published toplines and question wording from the GSS 2022 codebook PDF.

Checklist 1.4a and the Layer 0 gate. The PDF is titled *"Codebook and Unweighted
Frequencies for the 2022 General Social Survey"*, and its variable pages carry,
for each item:

    Variable: NATROAD Type: Numeric
    Label:
    (... are we spending too much, too little,
    or about the right amount on) Highways and bridges
    ...
    LABEL VALUE COUNT PCT  Codes
    TOO LITTLE   1  1885  53.2%  54.0%
    ...
    SUBTOTALS:      3488  98.4%  100.0%
    RESERVED CODES:
    DON'T KNOW  D     54   1.5%  n/a
    TOTALS:         3544 100.0%  100.0%

Two columns of percentages: ``PCT`` over *all* respondents including reserved
codes, and ``PCT Excl. Reserve Codes`` over answering respondents only. The
second is the one a response histogram must match — it is the distribution over
people who actually answered the item.

The published figures are **unweighted**. That is what makes them usable as a
Layer 0 gate: they are a direct function of the recode, with no weighting step
in between, so a mismatch is a recode or scale bug and nothing else.

Output: ``codebooks/gss2022_published_toplines.yaml``. Every entry is flagged
``verified: false`` until a human has spot-checked it against the PDF.

Usage:
    python scripts/extract_toplines.py [--items natroad,spkath] [--all]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
DEFAULT_PDF = REPO / ".." / "data" / "gss" / "codebooks" / "GSS_2022_Codebook.pdf"
DEFAULT_OUT = REPO / "codebooks" / "gss2022_published_toplines.yaml"

VAR_RE = re.compile(r"^Variable:\s*([A-Z0-9_]+)\s+Type:\s*(\w+)", re.MULTILINE)
ROW_RE = re.compile(
    r"^(?P<label>.+?)\s+(?P<value>-?\d+)\s+(?P<count>[\d,]+)\s+"
    r"(?P<pct>[\d.]+)%\s+(?P<pct_excl>[\d.]+)%\s*$"
)
# Long option labels wrap, and the PDF then prints the numeric row on a line of
# its own with the label text above and/or below it — PORNLAW is the clearest
# example. So a row is matched in two shapes: label-then-numbers on one line,
# and numbers alone with the label accumulated from neighbouring lines.
BARE_ROW_RE = re.compile(
    r"^(?P<value>-?\d+)\s+(?P<count>[\d,]+)\s+(?P<pct>[\d.]+)%\s+(?P<pct_excl>[\d.]+)%\s*$"
)
RESERVED_RE = re.compile(
    r"^(?P<label>.+?)\s+(?P<code>[A-Za-z])\s+(?P<count>[\d,]+)\s+(?P<pct>[\d.]+)%\s+n/a\s*$"
)
SUBTOTAL_RE = re.compile(r"^SUBTOTALS:\s+(?P<count>[\d,]+)\s+(?P<pct>[\d.]+)%")
TOTAL_RE = re.compile(r"^TOTALS:\s+(?P<count>[\d,]+)\s+(?P<pct>[\d.]+)%")


def _int(s: str) -> int:
    return int(s.replace(",", ""))



def _values_contiguous(values: list[int]) -> bool:
    """Do the captured option values cover every code from min to max?

    For long scales the codebook does not print every row: LIFENOW is a 0-10
    ladder shown as three lines — the first, a collapsed middle carrying the
    count of everything between, and the last. The captured values are then
    {1, 2, 10} rather than {1..11}, the counts still reconcile to the printed
    total, and the table looks complete while being a summary. Requiring
    contiguity is what separates a real 3-option table like NATROAD from a
    3-line rendering of an 11-option one.
    """
    if not values:
        return False
    return sorted(values) == list(range(min(values), max(values) + 1))


def parse_variable_block(block: str) -> dict | None:
    """Parse one ``Variable: X ... TOTALS:`` block into a topline record."""
    m = VAR_RE.search(block)
    if not m:
        return None
    name, vtype = m.group(1), m.group(2)

    # Label text sits between "Label:" and the next "Notes:"/table header.
    # "Label:" may be alone on its line or carry the first line of wording after
    # it. Both shapes occur in this PDF; taking only the first would silently
    # yield an empty question text for items like FINRELA and POLVIEWS.
    label = ""
    lm = re.search(r"^Label:[ \t]*(.*?)^(?:Notes:|PCT[ \t]*$|LABEL\s+VALUE)", block, re.MULTILINE | re.DOTALL)
    if lm:
        label = " ".join(x.strip() for x in lm.group(1).strip().splitlines() if x.strip())

    options, reserved = [], []
    subtotal = total = None
    in_reserved = False
    in_table = False
    pending: list[str] = []   # wrapped label lines seen before a bare numeric row

    for line in block.splitlines():
        line = line.rstrip()
        if re.match(r"^LABEL\s+VALUE", line):
            in_table = True
            pending.clear()
            continue
        if line.startswith("RESERVED CODES:"):
            in_reserved = True
            pending.clear()
            continue
        sm = SUBTOTAL_RE.match(line)
        if sm:
            subtotal = _int(sm.group("count"))
            pending.clear()
            continue
        tm = TOTAL_RE.match(line)
        if tm:
            total = _int(tm.group("count"))
            continue
        if in_reserved:
            rm = RESERVED_RE.match(line)
            if rm:
                reserved.append({
                    "label": rm.group("label").strip(),
                    "code": rm.group("code").lower(),
                    "count": _int(rm.group("count")),
                })
            continue

        om = ROW_RE.match(line)
        if om and not om.group("label").strip().startswith(("SUBTOTALS", "TOTALS")):
            options.append({
                "label": om.group("label").strip(),
                "value": int(om.group("value")),
                "count": _int(om.group("count")),
                "pct_all": float(om.group("pct")),
                "pct_excl_reserved": float(om.group("pct_excl")),
            })
            pending.clear()
            continue

        bm = BARE_ROW_RE.match(line)
        if bm and in_table:
            options.append({
                "label": " ".join(pending).strip(),
                "value": int(bm.group("value")),
                "count": _int(bm.group("count")),
                "pct_all": float(bm.group("pct")),
                "pct_excl_reserved": float(bm.group("pct_excl")),
            })
            pending.clear()
            continue

        if in_table and line.strip():
            if options and not options[-1]["label"].endswith(tuple("0123456789")):
                # A continuation of the option label that wrapped past its row.
                # Only ever appended to the option immediately above, and only
                # while no new row has been seen.
                pending.append(line.strip())
            else:
                pending.append(line.strip())

    # Option label text is taken no further than this. A long label wraps around
    # its own numeric row, so the tail lands on the line below and there is no
    # reliable way to tell a tail from the next option's head. The authoritative
    # source for option labels is the .dta value-label map, which is exact and
    # machine-readable; the PDF's job here is the *question wording* and the
    # *published counts*, and both of those parse cleanly. Labels below are
    # marked advisory so nothing downstream mistakes them for the instrument.

    if not options or subtotal is None:
        return None

    counted = sum(o["count"] for o in options)
    return {
        "item_id": name.lower(),
        "type": vtype,
        "published_label": label,
        "options": options,
        "subtotal_n": subtotal,
        "total_n": total,
        "reserved": reserved,
        # Two independent consistency checks on the extraction itself, before
        # any comparison to our pipeline.
        #
        # `parse_consistent`: the captured option counts add to the printed
        # subtotal. Necessary, not sufficient.
        #
        # `table_complete`: subtotal + reserved codes = the printed total. This
        # is the one that catches a table spanning a page break, where the
        # options captured are only the fragment on the first page while the
        # SUBTOTALS/TOTALS lines come from further down. Such a fragment can
        # still satisfy `parse_consistent` by accident, and it is the source of
        # every occupation-code and count-variable false alarm. Only items that
        # satisfy both may be gated on.
        "parse_consistent": counted == subtotal,
        "table_complete": (
            total is not None
            and counted == subtotal
            and subtotal + sum(r["count"] for r in reserved) == total
            and _values_contiguous([o["value"] for o in options])
        ),
        "scale_length": len(options),
        "option_labels_are_advisory": True,  # authoritative source is the .dta value labels
        "source": "GSS_2022_Codebook.pdf (Codebook and Unweighted Frequencies, Release 4)",
        "weighted": False,
        "verified": False,   # flip to true once a human has eyeballed it
    }


def _completeness(rec: dict) -> tuple[int, int, int]:
    """Rank two parses of the same variable. Higher is better."""
    return (
        int(bool(rec.get("table_complete"))),
        int(rec.get("total_n") is not None),
        len(rec.get("options", [])),
    )


def extract(pdf_path: Path, wanted: set[str] | None = None) -> dict[str, dict]:
    import pdfplumber

    out: dict[str, dict] = {}
    raw_blocks: dict[str, list[str]] = {}
    buf: list[str] = []

    def flush():
        if not buf:
            return
        block = "\n".join(buf)
        rec = parse_variable_block(block)

        if rec is None:
            # The tail of a table that was cut at a page footer carries only the
            # remaining reserved codes and the TOTALS line under a repeated
            # header — no option rows, so it does not parse on its own. It is
            # still the missing half of the item above it. Stitch and re-parse.
            head = VAR_RE.search(block)
            if not head:
                return
            item_id = head.group(1).lower()
            prev = out.get(item_id)
            if prev is None or prev.get("table_complete"):
                return
            merged = parse_variable_block("\n".join(raw_blocks.get(item_id, []) + buf))
            if merged and _completeness(merged) > _completeness(prev):
                out[item_id] = merged
                raw_blocks[item_id] = raw_blocks.get(item_id, []) + list(buf)
            return

        if wanted is not None and rec["item_id"] not in wanted:
            return
        # A variable table that runs over a page break is printed twice: once
        # truncated at the page footer, then again in full at the top of the
        # next page. Taking the first parse would keep the truncated copy — for
        # CONLABOR, CONLEGIS, HOMOSEX and NATSPAC that means a record with no
        # TOTALS line, which then fails `table_complete` and is dropped from the
        # gate. So prefer the more complete of the two.
        item_id = rec["item_id"]
        prev = out.get(item_id)
        if prev is None:
            out[item_id] = rec
            raw_blocks[item_id] = list(buf)
            return
        if _completeness(rec) > _completeness(prev):
            out[item_id] = rec
            raw_blocks[item_id] = list(buf)
            return
        # Neither copy is complete on its own: the table was cut at the page
        # footer and resumes under a repeated header on the next page, so the
        # rows are split across two blocks. Stitch the buffers and re-parse. The
        # repeated header lines are inert — they carry no table rows — so the
        # merged parse simply sees the whole table.
        if not prev.get("table_complete"):
            merged = parse_variable_block("\n".join(raw_blocks.get(item_id, []) + buf))
            if merged and _completeness(merged) > _completeness(prev):
                out[item_id] = merged
                raw_blocks[item_id] = raw_blocks.get(item_id, []) + list(buf)

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for line in text.splitlines():
                if line.startswith("Variable:"):
                    flush()
                    buf = [line]
                elif buf:
                    buf.append(line)
        flush()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--items", type=str, default="", help="comma-separated item ids")
    ap.add_argument("--all", action="store_true", help="extract every variable in the PDF")
    args = ap.parse_args(argv)

    wanted = None
    if args.items:
        wanted = {i.strip().lower() for i in args.items.split(",") if i.strip()}
    elif not args.all:
        ap.error("pass --items or --all")

    recs = extract(args.pdf, wanted)
    if wanted:
        missing = sorted(wanted - set(recs))
        if missing:
            print(f"WARNING: not found in PDF: {missing}", file=sys.stderr)

    bad = [k for k, v in recs.items() if not v["parse_consistent"]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(
        {"source_pdf": args.pdf.name, "items": recs}, sort_keys=True, allow_unicode=True, width=100
    ))
    print(f"wrote {len(recs)} items -> {args.out}")
    if bad:
        print(f"WARNING: {len(bad)} items whose option counts do not sum to the "
              f"printed subtotal (check by hand): {sorted(bad)[:20]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
