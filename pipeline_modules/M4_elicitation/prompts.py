"""
Prompt Templates and Paraphrase Variations for Module M4 (Distributional Elicitation).

Enforces 3 distinct prompt framings (Paraphrase A, B, C) to ensure ensemble robustness (F8 mitigation).
"""

from typing import Dict, List

SYSTEM_PROMPT = """You are a senior quantitative survey methodologist and demographic statistician.
Your objective is to produce realistic, calibrated probability distributions over survey response options for specific population subgroups.

Critical Guidelines:
1. Output valid JSON adhering strictly to the schema.
2. The probabilities across the specified response options must reflect genuine intra-group dispersion, not uniform consensus.
3. Condition carefully on the demographic constraints and empirical reference anchor items provided in the Stat Card.
"""

PARAPHRASE_A = """{stat_card_text}

=== TARGET SURVEY QUESTION ===
Topic: {topic}
Question: "{question_text}"
Options: [{options_str}]

TASK (Analytical Survey Framing):
Based on the demographic profile and empirical reference points above, estimate the percentage distribution of adults in this specific demographic group who would select each option.

Return JSON format:
{{
  "probabilities": [p_1, p_2, ..., p_{num_options}],
  "reasoning_summary": "<max 50 words on key demographic drivers>"
}}
"""

PARAPHRASE_B = """{stat_card_text}

=== TARGET SURVEY QUESTION ===
Topic: {topic}
Question: "{question_text}"
Options: [{options_str}]

TASK (Sociological Demography Framing):
Consider the cultural, economic, and regional factors that influence this specific demographic segment. What proportion of this subgroup holds each opinion?

Return JSON format:
{{
  "probabilities": [p_1, p_2, ..., p_{num_options}],
  "reasoning_summary": "<max 50 words on sociological factors>"
}}
"""

PARAPHRASE_C = """{stat_card_text}

=== TARGET SURVEY QUESTION ===
Topic: {topic}
Question: "{question_text}"
Options: [{options_str}]

TASK (Behavioral Opinion Estimation):
Predict the expected empirical frequency distribution across a representative sample of 1,000 respondents strictly matching this cluster's demographic definition.

Return JSON format:
{{
  "probabilities": [p_1, p_2, ..., p_{num_options}],
  "reasoning_summary": "<max 50 words on expected behavioral distribution>"
}}
"""

PARAPHRASES: Dict[int, str] = {
    1: PARAPHRASE_A,
    2: PARAPHRASE_B,
    3: PARAPHRASE_C,
}


def build_user_prompt(
    stat_card_text: str,
    topic: str,
    question_text: str,
    options: List[str],
    paraphrase_id: int = 1,
) -> str:
    template = PARAPHRASES.get(paraphrase_id, PARAPHRASE_A)
    options_str = ", ".join([f'"{opt}"' for opt in options])
    return template.format(
        stat_card_text=stat_card_text,
        topic=topic.replace("_", " ").title(),
        question_text=question_text,
        options_str=options_str,
        num_options=len(options),
    )
