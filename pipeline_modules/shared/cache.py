"""
Persistent Prompt/Response Cache Module for B17 Population Simulation.

Uses SQLite to store LLM responses keyed on SHA-256 hash of:
(model, system_prompt, user_prompt, temperature, top_p, structured_schema_name).
Guarantees zero duplicate API spends and enables instant resumption of experiments.
"""

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


class PromptCache:
    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = Path(__file__).resolve().parent.parent / "data_cache" / "llm_cache.sqlite"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prompt_cache (
                    cache_key TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    system_prompt TEXT,
                    user_prompt TEXT NOT NULL,
                    temperature REAL,
                    response_json TEXT NOT NULL,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_model ON prompt_cache(model)"
            )
            conn.commit()

    @staticmethod
    def compute_key(
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.95,
        extra_key: str = "",
    ) -> str:
        payload = f"{model}:::{system_prompt}:::{user_prompt}:::{temperature:.3f}:::{top_p:.3f}:::{extra_key}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT response_json, prompt_tokens, completion_tokens, created_at 
                FROM prompt_cache 
                WHERE cache_key = ?
                """,
                (cache_key,),
            )
            row = cursor.fetchone()
            if row:
                return {
                    "response_json": json.loads(row[0]),
                    "prompt_tokens": row[1],
                    "completion_tokens": row[2],
                    "created_at": row[3],
                    "is_cached": True,
                }
        return None

    def set(
        self,
        cache_key: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        response_json: Dict[str, Any],
        prompt_tokens: int,
        completion_tokens: int,
    ):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO prompt_cache 
                (cache_key, model, system_prompt, user_prompt, temperature, response_json, prompt_tokens, completion_tokens)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cache_key,
                    model,
                    system_prompt,
                    user_prompt,
                    temperature,
                    json.dumps(response_json),
                    prompt_tokens,
                    completion_tokens,
                ),
            )
            conn.commit()

    def get_stats(self) -> Dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), SUM(prompt_tokens), SUM(completion_tokens) FROM prompt_cache")
            count, pt, ct = cursor.fetchone()
            return {
                "total_cached_entries": count or 0,
                "total_prompt_tokens_saved": pt or 0,
                "total_completion_tokens_saved": ct or 0,
            }
