"""Scope gate: only messages recognized as on topic reach the answering model.

The small model sees the message alone (no data, no profile) and returns one label from a fixed
list. Anything else (an error, a refusal, a cut or unknown answer) means "do not let it through".
Accuracy is measured by scripts/eval_scope.py on contracts/golden/scope.yaml.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from secondwind.core.assistant import load_prompt
from secondwind.core.models import effort_for, model_override, tier_for
from secondwind.core.ports import LLM, Block, LLMRequest, LLMResponse
from secondwind.core.router import FLAGS, MONTH, PERIODS, Route, asks_for_analysis, extract

logger = logging.getLogger(__name__)

TASK = "classify"
TIMEOUT_S = 8.0
ON_TOPIC = ("training", "health", "sleep", "nutrition")
INTENTS = ("fact.last_workout", "fact.records", "fact.run_distance", "fact.metric",
           "calc.zones", "calc.pace", "calc.race_predict")
BLOCKED = ("emergency", "crisis", "injection", "off_topic", "smalltalk", "unclear")
LABELS = ON_TOPIC + INTENTS + BLOCKED
ERROR, REFUSED = "classifier_error", "refused"   # not labels: what happened instead of one
NEEDS_PERIOD = ("fact.run_distance", "fact.metric")

CLASSIFY_PROMPT = load_prompt("classify")


class ScopeAnswer(BaseModel):
    """What the classifier must return; the JSON schema sent to the model is built from it."""
    model_config = ConfigDict(extra="forbid", strict=True)
    label: Literal[LABELS]  # type: ignore[valid-type]


class RewriteAnswer(BaseModel):
    """What the follow-up rewrite must return (prompts/rewrite.md)."""
    model_config = ConfigDict(extra="forbid", strict=True)
    follow_up: bool          # the message continues the previous questions
    question: str            # the message as one standalone question (or unchanged)


SCHEMA = ScopeAnswer.model_json_schema()
REWRITE_SCHEMA = RewriteAnswer.model_json_schema()
REWRITE_TASK = "rewrite"
REWRITE_PROMPT = load_prompt("rewrite")
MAX_QUESTION = 500           # a longer rewrite is ignored
# When the message and its rewrite get different labels, the first of these wins; otherwise the
# rewrite's label (it says what the follow-up asks for).
STRICTER = (ERROR, REFUSED, "emergency", "crisis", "injection", "off_topic")


@dataclass
class Verdict:
    label: str                           # one of LABELS, or ERROR / REFUSED
    responses: list[LLMResponse] = field(default_factory=list)   # for the usage counter
    follow_up: bool = False
    question: str = ""                   # standalone question; the message itself unless a usable rewrite

    @property
    def response(self) -> LLMResponse | None:
        return self.responses[0] if self.responses else None


def _request(task: str, prompt: str, schema: dict, content: dict) -> LLMRequest:
    return LLMRequest(tier=tier_for(task), model=model_override(task), effort=effort_for(task), task=task,
                      system=[Block(prompt)], max_tokens=1024, schema=schema, timeout=TIMEOUT_S,
                      messages=[{"role": "user", "content": json.dumps(content, ensure_ascii=False)}])


def _failed(r: LLMResponse) -> str | None:
    if r.refused:
        return REFUSED
    if r.stop_reason == "max_tokens":
        return ERROR
    return None


def classify(llm: LLM, text: str) -> Verdict:
    """One label for one message; the classifier sees nothing else."""
    try:
        r = llm.complete(_request(TASK, CLASSIFY_PROMPT, SCHEMA, {"message": text}))
    except Exception as e:  # noqa: BLE001 - a broken classifier must not open the gate
        logger.error("classifier failed: %s", type(e).__name__)
        return Verdict(ERROR, question=text)
    label = _failed(r)
    if label is None:
        try:
            label = ScopeAnswer.model_validate_json(r.text).label
        except ValidationError:
            label = ERROR
    logger.info("scope label=%s model=%s in=%s out=%s", label, r.model, r.input_tokens, r.output_tokens)
    return Verdict(label, [r], question=text)


def rewrite(llm: LLM, text: str, previous: list[str]) -> tuple[str | None, LLMResponse | None]:
    """The message as a standalone question when it continues the previous questions, else None.
    Only the questions are sent, never the answers or the data."""
    try:
        r = llm.complete(_request(REWRITE_TASK, REWRITE_PROMPT, REWRITE_SCHEMA,
                                  {"previous_questions": previous, "message": text}))
    except Exception as e:  # noqa: BLE001 - without a rewrite the message is classified as it is
        logger.error("rewrite failed: %s", type(e).__name__)
        return None, None
    if _failed(r):
        return None, r
    try:
        a = RewriteAnswer.model_validate_json(r.text)
    except ValidationError:
        return None, r
    q = a.question.strip()
    usable = a.follow_up and 0 < len(q) <= MAX_QUESTION and q != text
    logger.info("rewrite follow_up=%s used=%s", a.follow_up, usable)
    return (q if usable else None), r


def check(llm: LLM, text: str, previous: list[str] | None = None) -> Verdict:
    """The gate for a free-form message. With previous questions, a follow-up is rewritten and both
    the message and the rewrite are classified; the stricter label wins."""
    if not previous:
        return classify(llm, text)
    standalone, rw = rewrite(llm, text, previous)
    if standalone is None:
        v = classify(llm, text)
        v.responses = ([rw] if rw else []) + v.responses
        return v
    with ThreadPoolExecutor(2) as pool:
        original, rewritten = pool.map(lambda q: classify(llm, q), (text, standalone))
    label = next((lb for lb in STRICTER if lb in (original.label, rewritten.label)), rewritten.label)
    return Verdict(label, ([rw] if rw else []) + original.responses + rewritten.responses,
                   follow_up=True, question=standalone)


def scope_route(verdict: Verdict, text: str, lang: str, today: date | None = None) -> Route:
    """What to do with a classified message: a fixed text, an existing handler or the model."""
    label = verdict.label
    if label in ON_TOPIC:
        return Route("model", "model", lang)
    if label in INTENTS and asks_for_analysis(text):   # "why ...": never a bare number
        return Route("model", "model", lang)
    if label in INTENTS:
        named = any(re.search(p, text, FLAGS) for p, _ in PERIODS) or MONTH.search(text) is not None
        if label in NEEDS_PERIOD and not named:      # no default period here: the model reads the question
            return Route("model", "model", lang)
        params = extract(label, text, lang, today or date.today())
        return Route("model", "model", lang) if params is None else Route("intent", label, lang, params)
    return Route("filter", f"filter.{label}", lang)
