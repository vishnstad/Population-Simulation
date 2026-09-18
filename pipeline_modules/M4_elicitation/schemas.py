"""
Pydantic Schemas for Module M4 (Distributional Elicitation).
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class ElicitedDistributionOutput(BaseModel):
    """Structured JSON schema enforced on the LLM output."""

    probabilities: List[float] = Field(
        ...,
        description="Probability or percentage mass for each response option, ordered exactly as given in the prompt.",
    )
    reasoning_summary: Optional[str] = Field(
        default="",
        description="Concise sociological/statistical rationale for this distribution (max 50 words).",
    )

    @field_validator("probabilities")
    @classmethod
    def validate_probabilities(cls, v: List[float]) -> List[float]:
        if not v:
            raise ValueError("Probability list cannot be empty.")
        # Handle percentage scale (0-100) vs probability scale (0-1)
        total = sum(v)
        if total > 1.5:  # Model emitted percentages summing to ~100
            v = [p / 100.0 for p in v]
            total = sum(v)

        if total <= 0:
            raise ValueError("Sum of probabilities must be positive.")

        # Re-normalize to exact unit sum
        normalized = [round(p / total, 6) for p in v]
        # Fix small floating point residual on first element
        diff = 1.0 - sum(normalized)
        normalized[0] = round(normalized[0] + diff, 6)
        return normalized


class RawElicitationRecord(BaseModel):
    """Individual sample record stored in parquet/records."""

    cluster_id: str
    item_id: str
    model: str
    paraphrase_id: int
    repeat_id: int
    probabilities: List[float]
    reasoning_summary: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    is_cached: bool
    latency_seconds: float
