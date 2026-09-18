#!/usr/bin/env python
"""Set verbatim instrument wording for an item (checklist 1.4a).

Needed because the GSS codebook PDF cannot supply it for every item. The Label
field is a SAS label capped at 256 bytes, so long questions are cut off
mid-sentence in the published PDF itself — SPKATH ends at "...against churches
and rel" — and six items were not fielded in 2022 at all, so they have no entry.

Until an item's wording is verified, ``Item.assert_elicitable()`` refuses to let
a prompt be built from it. This script is how the block is lifted, and it
records where the text came from so the provenance survives into the writeup.

    python scripts/set_wording.py --item spkhomo \
        --text "…verbatim question text…" \
        --source "GSS 2021 Codebook R1, p. 412"

    python scripts/set_wording.py --from-yaml wording_patch.yaml
    python scripts/set_wording.py --list

The GSS ballot questionnaires are the source of the missing text. See
``data/gss/quex/PUT_BALLOT_PDFS_HERE.md`` for the six download URLs — the NORC
host is not reachable from this environment, so they are fetched by hand.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

CODEBOOK = REPO / "codebooks" / "gss_items.yaml"


def _load() -> dict:
    return yaml.safe_load(CODEBOOK.read_text())


def _save(doc: dict) -> None:
    CODEBOOK.write_text(
        yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=100)
    )


def set_one(doc: dict, item_id: str, text: str, source: str) -> None:
    items = doc["items"]
    if item_id not in items:
        raise SystemExit(f"{item_id!r} is not in the codebook; have {len(items)} items")
    text = " ".join(text.split())
    if len(text) < 25:
        raise SystemExit(
            f"{item_id}: {text!r} is {len(text)} chars — that is a label, not a question. "
            f"Paste the full instrument text including any shared preamble."
        )
    if not source.strip():
        raise SystemExit(f"{item_id}: --source is required; wording with no provenance is not verified")
    items[item_id]["text"] = text
    items[item_id]["text_source"] = source.strip()
    items[item_id]["wording_status"] = "verified"
    items[item_id]["wording_note"] = ""
    print(f"  {item_id}: verified, {len(text)} chars, source={source!r}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--item")
    ap.add_argument("--text")
    ap.add_argument("--source", default="")
    ap.add_argument("--from-yaml", type=Path,
                    help="YAML mapping item_id -> {text, source}")
    ap.add_argument("--list", action="store_true",
                    help="show every item still blocked, with why")
    args = ap.parse_args(argv)

    doc = _load()

    if args.list:
        blocked = {k: v for k, v in doc["items"].items()
                   if v.get("wording_status") != "verified" and v.get("role") != "excluded"}
        print(f"{len(blocked)} of {len(doc['items'])} items blocked for elicitation:\n")
        for k, v in sorted(blocked.items()):
            print(f"  {k:10s} [{v.get('wording_status')}]")
            print(f"    have: {v['text'][:110]!r}")
            print(f"    why : {v.get('wording_note', '')[:150]}")
        return 0

    if args.from_yaml:
        patch = yaml.safe_load(args.from_yaml.read_text()) or {}
        print(f"applying {len(patch)} wordings from {args.from_yaml}:")
        for item_id, rec in patch.items():
            set_one(doc, item_id, rec["text"], rec.get("source", ""))
        _save(doc)
        return 0

    if not (args.item and args.text):
        ap.error("pass --item and --text, or --from-yaml, or --list")
    set_one(doc, args.item, args.text, args.source)
    _save(doc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
