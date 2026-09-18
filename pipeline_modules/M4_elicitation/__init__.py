"""M4 -- distributional elicitation: the LLM call and its ensemble."""

from .elicit import CellResult, DistributionElicitor, aggregate_draws
from .prompts import PARAPHRASES, SYSTEM_PROMPT, build_user_prompt
from .schemas import ElicitedDistributionOutput, RawElicitationRecord

__all__ = [
    "DistributionElicitor",
    "CellResult",
    "aggregate_draws",
    "SYSTEM_PROMPT",
    "PARAPHRASES",
    "build_user_prompt",
    "ElicitedDistributionOutput",
    "RawElicitationRecord",
]
