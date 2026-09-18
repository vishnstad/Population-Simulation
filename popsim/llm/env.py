"""Load API keys from `.env` into the process environment.

Checklist 0.7 says to export `MISTRAL_API_KEY`, `GROQ_API_KEY`,
`GOOGLE_API_KEY`, `CEREBRAS_API_KEY`, and `.env.example` in this repo tells you
to copy it to `.env`. Both are true and nothing connected them: `client.py`
read `os.environ` only, `.env` is gitignored and not sourced by an interactive
shell, so the keys existed on disk and were invisible to every run.

What that costs is not an error message. `LLMClient` treats a missing key as an
empty bearer token, sends the request anyway, and the provider returns 401 —
which the elicitation loop records as a `provider` failure per cell, exactly as
designed, because a run of thousands of calls over days must survive a provider
having a bad afternoon rather than dying. So the failure mode is a complete run
of recorded failures that looks like the provider was down. On a rate-capped
free tier that is a day.

Deliberately not `python-dotenv`: checklist 0.3 pins an exact dependency set
that is verified working, and this is fifteen lines.

**The environment always wins.** A key exported in the shell overrides `.env`,
so a one-off run with a different key needs no file edit and CI — which has no
`.env` — is unaffected.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["find_dotenv", "load_dotenv"]

#: What this process has loaded from a `.env`, name -> source file. Accumulated
#: so a second call reports the same thing as the first: `main()` loads keys
#: before dispatch, so `doctor` calling `load_dotenv` again would otherwise see
#: nothing left to do and report the keys as having come from the shell.
_LOADED: dict[str, str] = {}


def find_dotenv(start: Path | None = None) -> Path | None:
    """Nearest `.env` at or above ``start``, so the CLI works from a subdirectory."""
    here = (start or Path.cwd()).resolve()
    for d in (here, *here.parents):
        p = d / ".env"
        if p.is_file():
            return p
    return None


def load_dotenv(path: str | os.PathLike[str] | None = None) -> dict[str, str]:
    """Set any key in `.env` that is not already in the environment.

    Returns the names that were loaded, mapped to their source file, so a caller
    can report provenance without ever printing a value.
    """
    p = Path(path) if path else find_dotenv()
    if p is None or not p.is_file():
        return dict(_LOADED)

    loaded: dict[str, str] = {}
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        if not key or not key.replace("_", "").isalnum():
            continue
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if key in os.environ and os.environ[key] != "":
            continue          # the shell wins, always
        os.environ[key] = val
        loaded[key] = str(p)
    _LOADED.update(loaded)
    return dict(_LOADED)
