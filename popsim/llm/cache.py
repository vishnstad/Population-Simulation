"""Response cache (checklist 0.4 / 0.4c).

This module is **load-bearing, not a nicety**. A 37k-call run at ~1k calls/day
takes weeks, so the runner has to survive being stopped and resumed
indefinitely. The cache is what makes a resume free: every completed call is on
disk, keyed deterministically, and a resumed run replays it without touching a
provider's quota.

Key design
----------
The checklist says "keyed on model+prompt hash". That is necessary but not
sufficient here, because elicitation deliberately samples the *same* prompt
several times at temperature 0.7 (``n_repeat``). If the key were only
model+prompt, the 3 repeats would collapse to one cached response and the
ensemble spread — which ``elicit_sd`` and every variance number depend on —
would be identically zero.

So the key is ``sha256(model | prompt | temperature | paraphrase_id | repeat_id
| schema_version)``. Two calls share a cache entry exactly when they are the
same draw of the same prompt from the same model, which is the only case where
reusing a response is honest.

Storage is one JSON file per key under a two-level shard, written atomically so
a kill -9 mid-write cannot corrupt an entry.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

__all__ = ["CacheKey", "CachedResponse", "ResponseCache"]

SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class CacheKey:
    model: str
    prompt: str
    temperature: float
    paraphrase_id: int = 0
    repeat_id: int = 0
    schema_version: str = SCHEMA_VERSION

    def digest(self) -> str:
        h = hashlib.sha256()
        for part in (
            self.model,
            self.prompt,
            f"{self.temperature:.6f}",
            str(self.paraphrase_id),
            str(self.repeat_id),
            self.schema_version,
        ):
            h.update(part.encode("utf-8"))
            h.update(b"\x1f")  # unit separator: no field-boundary collisions
        return h.hexdigest()


@dataclass
class CachedResponse:
    text: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usd: float = 0.0
    finish_reason: str | None = None
    attempts: int = 1
    created_unix: float = 0.0
    key_digest: str = ""
    meta: dict[str, Any] | None = None


class ResponseCache:
    """Content-addressed, crash-safe, append-only response store."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def path_for(self, key: CacheKey) -> Path:
        d = key.digest()
        return self.root / d[:2] / d[2:4] / f"{d}.json"

    def get(self, key: CacheKey) -> CachedResponse | None:
        p = self.path_for(key)
        if not p.exists():
            self.misses += 1
            return None
        try:
            payload = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            # A truncated entry is a miss, not a crash. Atomic writes make this
            # nearly impossible, but "nearly" is not a guarantee worth betting a
            # three-week run on.
            self.misses += 1
            return None
        self.hits += 1
        return CachedResponse(**payload)

    def put(self, key: CacheKey, resp: CachedResponse) -> None:
        resp.created_unix = resp.created_unix or time.time()
        resp.key_digest = key.digest()
        p = self.path_for(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(asdict(resp), indent=2))
        tmp.replace(p)  # atomic on POSIX

    def __len__(self) -> int:
        return sum(1 for _ in self.root.rglob("*.json"))

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}
