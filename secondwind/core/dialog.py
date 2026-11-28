"""Short dialog history (architecture 2.5): the last exchanges of the last day, for follow-ups
like "and yesterday?" or "why?". Only answered on-topic questions and emergencies are kept, so an
off-topic text or an attempt to change the instructions never becomes part of a later prompt.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta

KEEP = 3                        # exchanges (question + answer): about the agreed "last 5 messages"
MAX_AGE = timedelta(hours=24)
PREVIOUS_FOR_CLASSIFIER = 2     # the classifier sees only these last questions, never answers
ANSWER_CHARS = 1500             # an answer is cut to this length in the prompt


def entry(question: str, answer: str, now: datetime) -> dict:
    at = now.isoformat()
    return {"q": question, "a": answer, "at": at, "id": hashlib.sha256(f"{at}|{question}".encode()).hexdigest()[:10]}


def recent(entries: list[dict], now: datetime) -> list[dict]:
    """Oldest first, at most KEEP, none older than MAX_AGE."""
    fresh = [e for e in entries if now - datetime.fromisoformat(e["at"]) < MAX_AGE]
    return sorted(fresh, key=lambda e: e["at"])[-KEEP:]


def previous_questions(entries: list[dict]) -> list[str]:
    return [e["q"] for e in entries][-PREVIOUS_FOR_CLASSIFIER:]


def _safe(text: str) -> str:
    return text.replace("<", "‹").replace(">", "›")


def block(entries: list[dict]) -> str:
    """The history for the answering model, oldest first; content, not instructions."""
    lines = []
    for e in entries:
        a = e["a"] if len(e["a"]) <= ANSWER_CHARS else e["a"][:ANSWER_CHARS] + "…"
        lines.append(f"Athlete: {_safe(e['q'])}\nVita: {_safe(a)}")
    return "\n\n".join(lines)
