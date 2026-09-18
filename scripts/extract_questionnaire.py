#!/usr/bin/env python
"""Extract verbatim question wording from the GSS ballot questionnaires (1.4a).

Why this exists
---------------
The checklist says to transcribe wording from the codebook PDF. That is not
possible: the codebook stores each question in a SAS label, SAS labels cap at
256 bytes, and long questions are therefore cut off mid-sentence *in the
published codebook itself* — SPKATH ends at "...against churches and rel". Five
more pool items were not fielded in 2022 and have no codebook entry at all.

The questionnaires are the instrument. They print each item as

    SPKATH: RadioButton

    "There are always some people whose ideas are considered bad or dangerous
    by other people. For instance, somebody who is against all churches and
    religion...

    If such a person wanted to make a speech in your community against churches
    and religion, should he be allowed to speak, or not?"

    Codes:
    [1] " [CATI UCase: Yes, allowed] "
    [2] [CATI UCase: Not allowed]

which is unambiguous, mnemonic-keyed, and carries the response options too.

Battery preambles
-----------------
Within a battery the preamble is printed once, on the first item, and the
following items carry only their own sentence: COLATH is just "Should such a
person be allowed to teach in a college or university, or not?". The codebook's
convention is to restate the preamble in parentheses, and this script
reconstructs exactly that, so wording taken from here is directly comparable to
wording taken from the codebook.

That reconstruction is then **checked against the codebook** for every item
whose codebook entry was not truncated. An item where the two disagree is
reported, not silently written.

Usage:
    python scripts/extract_questionnaire.py                # report
    python scripts/extract_questionnaire.py --write        # apply to the codebook
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

QUEX_DIR = REPO / ".." / "data" / "gss" / "quex"
CODEBOOK = REPO / "codebooks" / "gss_items.yaml"

#: The 2022 instrument is authoritative for items fielded in 2022; the 2021
#: instrument only for the items dropped afterwards. The wording is not always
#: identical between them — the 2021 web version says "in your community" where
#: the 2022 CATI version says "in your (city/town/community)" — so the order
#: matters and the later instrument wins.
BALLOT_PRIORITY = [
    ("2022", "GSS2022_Ballot*.pdf"),
    ("2021", "GSS2020_CrossSection_Ballot*.pdf"),
]

#: The two instrument generations are typeset differently and both are needed:
#: the 2022 ballots are authoritative for items still fielded in 2022, the 2021
#: ballots for the five dropped afterwards.
#:
#:   2021  ``SPKATH: RadioButton``          text in double quotes, ``Codes:``  ``[1] label``
#:   2022  ``SPKATH: Categorical (Single)`` bare text,              ``Categories:`` ``{tok} Label``
ITEM_RE = re.compile(
    r"^([A-Z][A-Z0-9_]{1,20}):\s+"
    r"(RadioButton|CheckBox|TextBox|Numeric|Slider|Categorical \([A-Za-z]+\)|Long|Text|Double)\s*$"
)
CODE_RE = re.compile(r'^\[(\d+)\]\s*"?\s*(?:\[CATI UCase:\s*)?([^"\]]*?)\s*\]?\s*"?\s*$')
CATEGORY_RE = re.compile(r"^\{([a-z0-9_]+)\}\s+(.+?)\s*$")
#: Structural keywords that end a question block. The script-syntax patterns are
#: deliberately tight: an earlier `^If\s+\w+` swallowed legitimate question text,
#: because LIBATH literally begins "If some people in your community suggested…"
#: and SPKMIL's second paragraph begins "If such a person wanted…". Requiring the
#: script shape — an assignment, or a trailing `Then` — keeps the instrument text.
STOP_RE = re.compile(
    r"^(MaxValue|MinValue|Codes:|Categories:|PRELOADACTION|DISPLAY IF|SOFT CHECK|HARD CHECK"
    r"|If\s+[A-Z][A-Za-z0-9_.]*\s*(=|<>|Then\b)"
    r"|End If\s*$|Else\s*$|[A-Za-z0-9_.]+\.Response\s*=)"
)
#: Page furniture injected into the middle of a 2022 block.
FOOTER_RE = re.compile(r"^\s*MDDtoDOC\s*-|^\s*GSS\d{4}_BALLOT", re.IGNORECASE)

#: Reserved codes / category tokens; not response options.
RESERVED_CODES = {77, 98, 99, 88, 89}
RESERVED_TOKENS = {"dontknow", "refused", "noanswer", "na", "skipped", "skippedonweb", "iap"}

def _pdf_text(path: Path) -> str:
    import subprocess
    out = subprocess.run(
        ["pdftotext", "-layout", str(path), "-"],
        capture_output=True, text=True, check=True,
    )
    return out.stdout


def parse_ballot(text: str) -> dict[str, dict]:
    """Pull every ``MNEMONIC: Type`` block with its question text and options.

    Handles both instrument generations. The 2021 form quotes its question text,
    so the quote marks delimit it; the 2022 form does not, so the block runs
    until a structural keyword. Page footers appear *inside* 2022 blocks and are
    dropped rather than treated as the end of the question.
    """
    lines = text.splitlines()
    out: dict[str, dict] = {}
    i = 0
    while i < len(lines):
        m = ITEM_RE.match(lines[i].strip())
        if not m:
            i += 1
            continue
        mnemonic, qtype = m.group(1), m.group(2)
        quoted_form = qtype in ("RadioButton", "CheckBox", "TextBox", "Numeric", "Slider")
        i += 1

        buf: list[str] = []
        started = not quoted_form   # 2022: text starts immediately
        while i < len(lines):
            raw = lines[i].rstrip()
            stripped = raw.strip()
            if FOOTER_RE.match(stripped):
                i += 1
                continue
            if ITEM_RE.match(stripped):
                break
            if STOP_RE.match(stripped):
                if started and buf:
                    break
                i += 1
                continue
            if quoted_form and not started and stripped.startswith('"'):
                started = True
            if started:
                buf.append(raw)
                if quoted_form and stripped.endswith('"') and len(stripped) > 1:
                    i += 1
                    break
            i += 1

        # Options
        options: list[tuple[int, str]] = []
        j, in_options = i, False
        while j < len(lines) and j < i + 80:
            stripped = lines[j].strip()
            if ITEM_RE.match(stripped):
                break
            if stripped.startswith(("Codes:", "Categories:")):
                in_options = True
                j += 1
                continue
            if in_options:
                cm = CODE_RE.match(stripped)
                if cm and int(cm.group(1)) not in RESERVED_CODES:
                    label = cm.group(2).strip().strip('"').strip()
                    if label:
                        options.append((int(cm.group(1)), label))
                else:
                    tm = CATEGORY_RE.match(stripped)
                    if tm and tm.group(1).replace("_", "") not in RESERVED_TOKENS:
                        options.append((len(options) + 1, tm.group(2).strip()))
            j += 1

        block = "\n".join(buf).strip()
        if quoted_form:
            block = block.removeprefix('"').removesuffix('"')
        paras = [" ".join(p.split()) for p in re.split(r"\n\s*\n", block) if p.strip()]
        if paras:
            out.setdefault(mnemonic.lower(), {
                "type": qtype, "paragraphs": paras, "codes": options,
            })
        i = max(i, 1)
    return out


#: The tolerance battery is five groups of three. The preamble is printed once,
#: on the SPK item, and COL/LIB carry only their own sentence.
TOLERANCE_GROUPS = ("ath", "rac", "com", "mil", "homo")


def clean_markup(text: str) -> str:
    """Resolve mode-conditional markup to the web wording and drop the rest.

    Bracket spans nest ("[CATI TEXT: [HANDCARD A14]]") and are sometimes left
    unclosed when the surrounding paragraph ends, so this walks the string with
    a depth counter rather than applying a regex. A regex leaves stray brackets
    behind — POLVIEWS came out starting with a bare "]" — and stray punctuation
    in an instrument text is exactly the kind of silent corruption §1.4a is
    about.

    ``[WEB TEXT: x]`` keeps ``x``; every other bracket span is dropped whole.
    """
    out: list[str] = []
    i, depth, keep_from_depth = 0, 0, None
    while i < len(text):
        ch = text[i]
        if ch == "[":
            tag = text[i + 1:i + 40]
            if depth == 0 and tag.upper().startswith("WEB TEXT:"):
                keep_from_depth = depth
                i += 1 + len("WEB TEXT:")
                depth += 1
                continue
            depth += 1
            i += 1
            continue
        if ch == "]":
            if depth > 0:
                depth -= 1
                if keep_from_depth is not None and depth == keep_from_depth:
                    keep_from_depth = None
            i += 1
            continue
        if depth == 0 or keep_from_depth is not None:
            out.append(ch)
        i += 1
    if depth > 0 and keep_from_depth is None:
        # An interviewer instruction whose closing bracket fell outside the
        # captured paragraph — PRAY's "[CATI TEXT:... USE CATEGORIES AS PROBES".
        # Everything from the unclosed bracket on is instruction, not question.
        pass
    cleaned = "".join(out)
    cleaned = " ".join(cleaned.split())
    for bad, good in ((" ,", ","), (" .", "."), (" ?", "?"), ("( ", "("), (" )", ")")):
        cleaned = cleaned.replace(bad, good)
    return cleaned.strip()


def tolerance_group(mnemonic: str) -> str | None:
    for prefix in ("spk", "col", "lib"):
        if mnemonic.startswith(prefix):
            rest = mnemonic[len(prefix):]
            if rest in TOLERANCE_GROUPS:
                return rest
    return None


def reconstruct(mnemonic: str, rec: dict, preambles: dict[str, str]) -> str:
    """Return the question in the codebook's convention: "(preamble…) question".

    A preamble is attached ONLY within the tolerance battery, and only from the
    SPK item of the same group. An earlier attempt carried the most recent
    preamble forward across item boundaries, which prepended a question about
    gun type to FINRELA and a question about how often you do things to
    POLVIEWS. Scoping it to the group is what makes it correct rather than
    merely usually right.
    """
    paras = [clean_markup(p) for p in rec["paragraphs"]]
    paras = [p for p in paras if p]
    if not paras:
        return ""

    group = tolerance_group(mnemonic)

    if len(paras) >= 2:
        pre, body = paras[0].rstrip(". "), " ".join(paras[1:])
        if group and mnemonic.startswith("spk"):
            preambles[group] = pre
        return f"{pre}... {body}"

    body = paras[0]
    if group and group in preambles:
        return f"({preambles[group]}…) {body}"
    return body


#: A closing paren followed by a short item stem at the very end — the shape of
#: the spending and confidence batteries: "…in them?) Organized labor".
_STEM_TAIL_RE = re.compile(r"\)[.\s]*[A-Z][A-Za-z0-9 ,'&./-]{1,70}$")


def looks_complete(text: str, preambles: set[str]) -> tuple[bool, str]:
    """Is this a whole question, or a fragment the layout split in half?

    An earlier version of this required a question mark. That was wrong and
    rejected nine complete items: the confidence battery ends on an institution
    name ("…in them?) The executive branch of the federal government"), and the
    agree/disagree items end on the statement being rated ("…the woman takes
    care of the home and family."). Neither ends in a question mark and both are
    the entire instrument text.

    What it was really trying to catch was a specific failure — an item coming
    out as nothing but its battery preamble, which is what SPKMIL did before the
    stop-pattern bug was fixed. So that is what is checked now, directly.
    """
    t = text.strip()
    if len(t) < 25:
        return False, f"only {len(t)} chars"
    if t.rstrip(".") in {p.rstrip(".") for p in preambles}:
        return False, "identical to its battery preamble — the question half was lost"
    ends_ok = t.endswith((".", "?", "!", '"', "…", "...")) or bool(_STEM_TAIL_RE.search(t))
    if not ends_ok:
        return False, f"ends {t[-30:]!r} — neither terminal punctuation nor an item stem"
    first = t.lstrip("(").lstrip()
    if first[:1].islower():
        return False, f"starts lowercase ({first[:40]!r}) — looks like the tail of a split sentence"
    return True, ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="apply to codebooks/gss_items.yaml")
    ap.add_argument("--quex-dir", type=Path, default=QUEX_DIR)
    args = ap.parse_args(argv)

    doc = yaml.safe_load(CODEBOOK.read_text())
    items = doc["items"]

    # mnemonic -> (year, record), later years overwritten only if absent
    found: dict[str, tuple[str, dict]] = {}
    for year, pattern in BALLOT_PRIORITY:
        for pdf in sorted(args.quex_dir.glob(pattern)):
            parsed = parse_ballot(_pdf_text(pdf))
            preambles: dict[str, str] = {}
            for mnemonic, rec in parsed.items():
                rec["reconstructed"] = reconstruct(mnemonic, rec, preambles)
                grp = tolerance_group(mnemonic)
                if grp and grp in preambles:
                    rec["preamble"] = preambles[grp]
                rec["ballot"] = pdf.name
                if mnemonic not in found:
                    found[mnemonic] = (year, rec)
            print(f"  parsed {pdf.name}: {len(parsed)} items")

    print(f"\n{len(found)} distinct mnemonics across the ballots\n")

    # ---- self-check against the codebook's own untruncated entries --------
    agree = disagree = 0
    for item_id, meta in sorted(items.items()):
        if meta.get("wording_status") != "verified" or item_id not in found:
            continue
        _, rec = found[item_id]
        a = re.sub(r"[^a-z0-9]", "", meta["text"].lower())
        b = re.sub(r"[^a-z0-9]", "", rec["reconstructed"].lower())
        if a and b and (a in b or b in a or a[:80] == b[:80]):
            agree += 1
        else:
            disagree += 1
            print(f"  DIFFERS {item_id}:")
            print(f"    codebook : {meta['text'][:110]}")
            print(f"    ballot   : {rec['reconstructed'][:110]}")
    print(f"\nself-check against untruncated codebook entries: {agree} agree, {disagree} differ")

    # ---- what this unblocks ---------------------------------------------
    blocked = [k for k, v in items.items()
               if v.get("wording_status") != "verified" and v.get("role") != "excluded"]
    fixable = [k for k in blocked if k in found]
    missing = [k for k in blocked if k not in found]

    all_preambles = {
        rec.get("preamble", "") for _, rec in found.values() if rec.get("preamble")
    }
    clean, rejected = [], []
    for k in fixable:
        ok, why = looks_complete(found[k][1]["reconstructed"], all_preambles)
        (clean if ok else rejected).append((k, why))

    print(f"\nblocked items: {len(blocked)}")
    print(f"  extracted cleanly ({len(clean)}):")
    for k, _ in sorted(clean):
        year, rec = found[k]
        print(f"    {k:10s} [{year}] {rec['reconstructed'][:130]}")
    if rejected:
        print(f"\n  extraction REJECTED ({len(rejected)}) — paste these by hand:")
        for k, why in sorted(rejected):
            year, rec = found[k]
            print(f"    {k:10s} {why}")
            print(f"      got: {rec['reconstructed'][:130]!r}")
    if missing:
        print(f"  NOT in any ballot: {sorted(missing)}")
    fixable = [k for k, _ in clean]

    if not args.write:
        print("\n(dry run — pass --write to apply)")
        return 0

    for k in fixable:
        year, rec = found[k]
        items[k]["text"] = rec["reconstructed"]
        items[k]["text_source"] = (
            f"GSS {year} ballot questionnaire, {rec['ballot']} — verbatim instrument text, "
            f"web wording where the instrument differs by mode"
        )
        items[k]["wording_status"] = "verified"
        items[k]["wording_note"] = ""
    CODEBOOK.write_text(
        yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=100)
    )
    print(f"\nwrote {len(fixable)} wordings -> {CODEBOOK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
