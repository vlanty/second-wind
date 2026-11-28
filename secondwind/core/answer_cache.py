"""Answer cache (architecture 2.6): the same question gets the same answer until something it
depends on changes - new data, the profile or goal, the mode, the language, the day or the prompt.
Per user; follow-ups (which depend on the dialog) and detailed answers are not cached.
"""

from __future__ import annotations

import hashlib
import json
import re

KEEP = 20       # answers kept per user; older ones are deleted on write


def normalize(question: str) -> str:
    """Case, ё, punctuation and spacing do not make a different question."""
    q = question.lower().replace("ё", "е")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", q)).strip()


def fingerprint(user_id: str, question: str) -> str:
    """For the logs: shows repeats without the text."""
    return hashlib.sha256(f"{user_id}|{normalize(question)}".encode()).hexdigest()[:10]


def key(user_id: str, question: str, *, lang: str, mode: str, data_version: str, profile: dict,
        goal: dict | None, day: str, prompt_version: str) -> str:
    parts = [user_id, normalize(question), lang, mode, data_version, profile, goal, day, prompt_version]
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False).encode()).hexdigest()[:32]
