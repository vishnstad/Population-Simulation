"""
Oracle Router & Scenario Compiler for Module M7.

Protects against spending LLM budget on questions already answered by microdata (F3 defense):
- Compares input question text against all items in the harmonized codebook using cosine similarity
- If similarity > 0.85 -> Routes to OBSERVED (direct empirical survey cross-tab)
- If similarity <= 0.85 -> Routes to SIMULATED (compiles into synthetic Item for M4/M5 simulation)
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class RouteDecision:
    query_text: str
    decision: str  # "OBSERVED" or "SIMULATED"
    matched_item_id: Optional[str]
    similarity_score: float
    compiled_item: Optional[Dict[str, Any]] = None


class OracleRouter:
    def __init__(self, codebook: List[Dict[str, Any]], threshold: float = 0.85):
        self.threshold = threshold
        self.codebook = codebook
        self.item_ids = [item["item_id"] for item in codebook]
        self.item_texts = [item.get("text", item["item_id"]) for item in codebook]

        # Initialize TF-IDF representation
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        if self.item_texts:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.item_texts)
        else:
            self.tfidf_matrix = None

    def route(
        self,
        query_text: str,
        options: Optional[List[str]] = None,
        topic: str = "custom_scenario",
    ) -> RouteDecision:
        """
        Routes a natural language query question to either OBSERVED or SIMULATED.
        """
        if self.tfidf_matrix is None or not self.item_texts:
            return RouteDecision(
                query_text=query_text,
                decision="SIMULATED",
                matched_item_id=None,
                similarity_score=0.0,
                compiled_item=self._compile_synthetic_item(query_text, options, topic),
            )

        query_vec = self.vectorizer.transform([query_text])
        sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        best_item_id = self.item_ids[best_idx]

        if best_score >= self.threshold:
            decision = "OBSERVED"
            compiled = None
            logger.info(
                f"Query matched observed item '{best_item_id}' (cosine={best_score:.4f} >= {self.threshold}). "
                "Serving from empirical microdata cross-tab."
            )
        else:
            decision = "SIMULATED"
            compiled = self._compile_synthetic_item(query_text, options, topic)
            logger.info(
                f"Query routed to simulation (highest cosine={best_score:.4f} < {self.threshold})."
            )

        return RouteDecision(
            query_text=query_text,
            decision=decision,
            matched_item_id=best_item_id if best_score > 0.3 else None,
            similarity_score=round(best_score, 4),
            compiled_item=compiled,
        )

    def _compile_synthetic_item(
        self, query_text: str, options: Optional[List[str]], topic: str
    ) -> Dict[str, Any]:
        opt_list = options or ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]
        return {
            "item_id": f"synthetic_{abs(hash(query_text)) % 100000}",
            "text": query_text,
            "topic": topic,
            "scale": {
                "type": "ordinal",
                "labels": opt_list,
                "codes": list(range(1, len(opt_list) + 1)),
            },
            "synthetic": True,
        }
