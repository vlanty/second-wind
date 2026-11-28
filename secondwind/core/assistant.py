"""Asking the AI model: prompts, the request, the answer.

Vendor-neutral: the request goes through the `LLM` port. Prompts are in English; the answer
language is set per request.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from secondwind.core.i18n import language_name
from secondwind.core.models import DEFAULT_MODE, effort_for, model_override, tier_for
from secondwind.core.ports import LLM, Block, LLMRequest

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(name: str) -> str:
    """A prompt file from core/prompts/ (the trailing newline of the file is not part of it)."""
    return (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8").removesuffix("\n")


# The templates are the system prompt; they use str.format fields {persona} {format} {language}.
# Content (athlete note, data, question) goes into tagged blocks of the user message.
# The answer is shown in Telegram: format.md asks for the Markdown subset we convert to Telegram HTML.
FORMAT_NOTE = load_prompt("format")
# The bot's persona. The project is "Second Wind"; the bot introduces itself as Vita.
PERSONA = load_prompt("persona")
GOAL_NOTE = load_prompt("no_goal")  # used when no athlete note is given
COACH_PROMPT = load_prompt("coach")
HEALTH_PROMPT = load_prompt("health")
NUTRITION_PROMPT = load_prompt("nutrition")
ASK_PROMPT = load_prompt("ask")


# Output ceiling per task: answers are asked to stay within 250-400 words; thinking counts too.
ANSWER_TOKENS = {"chat": 3000, "analysis": 4000}
DEFAULT_ANSWER_TOKENS = 3000
_SCOPE_LINE = re.compile(r"^\s*scope:\s*(yes|no)\b[^\n]*(\n|$)", re.IGNORECASE)


def split_scope(text: str) -> tuple[bool | None, str]:
    """The answer's first line `SCOPE: yes|no` (second lock, see ask.md) and the text without it.
    None when the line is missing. Nothing after a "no" is ever shown."""
    m = _SCOPE_LINE.match(text)
    if not m:
        return None, text.strip()
    if m.group(1).lower() == "no":
        return False, ""
    return True, text[m.end():].strip()


class Refused(Exception):
    """The model (and any fallback model) declined the request."""


@dataclass
class Answer:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


def _block(tag: str, text: str) -> str:
    """Content in a tagged block. Angle brackets are replaced so that the text (a question, an
    activity name) cannot close the block and pose as something else."""
    return f"<{tag}>\n{text.replace('<', '‹').replace('>', '›')}\n</{tag}>"


def build_request(prompt_template: str, data_summary: str, task: str, lang: str,
                  mode: str = DEFAULT_MODE, athlete: str | None = None, question: str | None = None,
                  dialog: str | None = None) -> LLMRequest:
    system = prompt_template.format(language=language_name(lang), format=FORMAT_NOTE, persona=PERSONA)
    content = "\n\n".join(_block(tag, text) for tag, text in
                          [("athlete", athlete or GOAL_NOTE), ("data", data_summary)]
                          + ([("dialog", dialog)] if dialog else [])
                          + ([("question", question)] if question else []))
    return LLMRequest(tier=tier_for(task, mode), model=model_override(task, mode), effort=effort_for(task),
                      system=[Block(system)], messages=[{"role": "user", "content": content}], task=task,
                      max_tokens=ANSWER_TOKENS.get(task, DEFAULT_ANSWER_TOKENS))


def analyze(llm: LLM, prompt_template: str, data_summary: str, task: str, lang: str,
            mode: str = DEFAULT_MODE, athlete: str | None = None, question: str | None = None,
            on_text: Callable[[str], None] | None = None, dialog: str | None = None) -> Answer:
    """Ask the model; with on_text the answer is streamed (adapters without streaming answer at once)."""
    req = build_request(prompt_template, data_summary, task, lang, mode, athlete, question, dialog)
    stream = getattr(llm, "stream", None) if on_text else None
    r = stream(req, on_text) if stream else llm.complete(req)
    if r.refused:
        raise Refused(r.refusal_details)
    if r.stop_reason == "max_tokens":
        logger.warning("model answer cut by max_tokens (task=%s, model=%s)", task, r.model)
    logger.info("llm task=%s model=%s in=%s out=%s", task, r.model, r.input_tokens, r.output_tokens)
    return Answer(r.text, r.model, r.input_tokens, r.output_tokens, r.cost_usd)
