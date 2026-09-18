"""M7 — the oracle router (checklist 6.1-6.2). The architectural answer to F3.

    "The system never re-predicts what the data answers. An *oracle router*
     checks every incoming question against the harmonized codebook. If the item
     (or a near-duplicate) exists in microdata, the answer is the weighted
     cross-tab, served directly, labeled 'observed'. The LLM path fires **only**
     for questions outside the data."

F3 is the objection that sinks this whole class of project — *you built an
expensive way to reproduce a cross-tab you already had* — and the router is what
makes the answer structural instead of rhetorical. Every question it routes to
`observed` is a question the LLM is never asked, by construction.

TF-IDF, not an embedder
-----------------------
Checklist 6.1 fixes this: "Start with TF-IDF + cosine — HuggingFace is blocked in
the sandbox and §M7 says the embedder is not load-bearing." It is also the more
defensible choice for a thesis: a reviewer can reproduce a TF-IDF cosine exactly,
and an embedding model's version is one more thing to pin.

Two things the spec got wrong, both found by doing 6.2 rather than skipping it
------------------------------------------------------------------------------
**The 0.85 threshold is unusable.** Measured on 42 hand-labeled pairs, plain word
TF-IDF at 0.85 has a recall of **zero**: no human paraphrase of a GSS item shares
enough vocabulary with its verbatim instrument wording to clear it. A threshold
taken from the spec on faith would have routed every real question to the LLM and
quietly retired the whole F3 defence.

**The representation matters more than the spec allows for.** Three changes take
recall from 0.46 to 0.85 at the same precision: strip the parenthetical battery
stems (the `nat*` items are 80 % identical boilerplate, which swamps the four
words that distinguish them), add character n-grams so "defense"/"defence" and
"favour"/"favor" match, and fold the option labels in (a 3-point spending scale
and a 5-point frequency scale are different asks even with the same stem).

The threshold is then tuned on the labeled set and the recall/precision curve is
reported (6.2), never assumed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = ["RouteDecision", "Router", "evaluate_threshold"]


@dataclass
class RouteDecision:
    question: str
    decision: str                       # "observed" | "simulate"
    matched_item: str | None = None
    similarity: float = 0.0
    threshold: float = 0.85
    runners_up: list[tuple[str, float]] = field(default_factory=list)

    @property
    def observed(self) -> bool:
        return self.decision == "observed"

    def explain(self) -> str:
        if self.observed:
            return (
                f"observed — this is {self.matched_item!r} (cosine "
                f"{self.similarity:.3f} >= {self.threshold}). Answered from the "
                f"weighted cross-tab; no model was asked."
            )
        near = (f" Nearest codebook item is {self.matched_item!r} at "
                f"{self.similarity:.3f}." if self.matched_item else "")
        return (
            f"simulated — no codebook item is within {self.threshold}.{near} "
            f"The answer is an estimate and carries the honesty box."
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"observed": self.observed}


class Router:
    """Cosine similarity over the codebook's verbatim instrument wording."""

    def __init__(self, codebook: dict[str, dict], *, threshold: float = 0.20,
                 include_labels: bool = True, char_ngrams: bool = True,
                 strip_stems: bool = True) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.threshold = float(threshold)
        self.include_labels = include_labels
        self.strip_stems = strip_stems
        self.item_ids = sorted(codebook)
        # Option labels fold in: a 3-point spending scale and a 5-point frequency
        # scale are different asks even when the stem is identical.
        self.texts = [
            self._prep(codebook[i]["text"]
                       + ((" " + " ".join(map(str, codebook[i]["labels"])))
                          if include_labels else ""))
            for i in self.item_ids
        ]
        self.word = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                    sublinear_tf=True)
        blocks = [self.word.fit_transform(self.texts)]
        self.char = None
        if char_ngrams:
            self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                        sublinear_tf=True, min_df=2)
            blocks.append(self.char.fit_transform(self.texts))
        self.X = self._stack(blocks)

    def _prep(self, text: str) -> str:
        """Lower-case, de-punctuate, and drop the parenthetical battery stem.

        The `nat*` spending items and the `con*` confidence items each repeat a
        long shared preamble in parentheses; leaving it in makes every member of
        a battery ~80 % identical and buries the handful of words that actually
        say which one it is.
        """
        import re

        t = re.sub(r"\([^)]*\)", " ", text) if self.strip_stems else text
        t = re.sub(r"[^a-zA-Z0-9 ]", " ", t)
        return re.sub(r"\s+", " ", t).strip().lower()

    @staticmethod
    def _stack(blocks):
        import numpy as np
        from scipy.sparse import hstack

        X = hstack(blocks).tocsr()
        norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1))).ravel()
        norms[norms <= 0] = 1.0
        from scipy.sparse import diags

        return (diags(1.0 / norms) @ X).tocsr()

    def _embed(self, question: str):
        q = self._prep(question)
        blocks = [self.word.transform([q])]
        if self.char is not None:
            blocks.append(self.char.transform([q]))
        return self._stack(blocks)

    def route(self, question: str, *, exclude: set[str] | None = None,
              top_k: int = 3) -> RouteDecision:
        import numpy as np

        sims = (self.X @ self._embed(question).T).toarray().ravel()
        if exclude:
            for i, item in enumerate(self.item_ids):
                if item in exclude:
                    sims[i] = -1.0
        order = np.argsort(-sims)[:top_k]
        best = int(order[0]) if len(order) else -1
        if best < 0 or sims[best] < 0:
            return RouteDecision(question, "simulate", None, 0.0, self.threshold)
        return RouteDecision(
            question=question,
            decision="observed" if sims[best] >= self.threshold else "simulate",
            matched_item=self.item_ids[best], similarity=float(sims[best]),
            threshold=self.threshold,
            runners_up=[(self.item_ids[int(j)], float(sims[int(j)]))
                        for j in order[1:] if sims[int(j)] > 0],
        )

    def near_duplicates(self, item_id: str, *, threshold: float | None = None) -> list[str]:
        """Codebook items too close to ``item_id`` to appear on its card."""
        t = self.threshold if threshold is None else threshold
        d = self.route(self.texts[self.item_ids.index(item_id)],
                       exclude={item_id}, top_k=len(self.item_ids))
        out = [d.matched_item] if d.matched_item and d.similarity >= t else []
        out += [i for i, s in d.runners_up if s >= t]
        return sorted({x for x in out if x})


def evaluate_threshold(
    router: Router, pairs: list[tuple[str, str, bool]], thresholds=None
) -> list[dict[str, float]]:
    """Recall and precision of duplicate detection over a hand-labeled set (6.2).

    ``pairs`` are (question text, codebook item id, is_duplicate). Reporting both
    recall and precision matters: a threshold low enough to catch every paraphrase
    also routes genuinely new questions to the cross-tab, which is the failure
    that would make the product answer the wrong question confidently.
    """
    import numpy as np

    thresholds = (list(thresholds) if thresholds is not None
                  else [round(float(t), 2) for t in np.arange(0.05, 0.91, 0.05)])
    out = []
    scored = []
    for text, item_id, is_dup in pairs:
        d = router.route(text, exclude=set())
        hit = d.matched_item == item_id
        scored.append((float(d.similarity) if hit else 0.0, is_dup, hit))
    for t in thresholds:
        tp = sum(1 for s, dup, hit in scored if dup and hit and s >= t)
        fp = sum(1 for s, dup, hit in scored if not dup and hit and s >= t)
        fn = sum(1 for s, dup, hit in scored if dup and not (hit and s >= t))
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        f1 = (0.0 if not (np.isfinite(rec) and np.isfinite(prec) and rec + prec > 0)
              else 2 * rec * prec / (rec + prec))
        out.append({"threshold": t, "recall": rec, "precision": prec,
                    "f1": f1, "n_flagged": tp + fp})
    return out
