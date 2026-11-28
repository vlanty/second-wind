"""Scope gate: only messages recognized as on topic reach the answering model.

The small model sees the message alone (no data, no profile) and returns one label from a fixed
list. Anything else (an error, a refusal, a cut or unknown answer) means "do not let it through".
Accuracy is measured by scripts/eval_scope.py on contracts/golden/scope.yaml.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
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


SCHEMA = ScopeAnswer.model_json_schema()


@dataclass
class Verdict:
    label: str                           # one of LABELS, or ERROR / REFUSED
    response: LLMResponse | None = None  # for the usage counter


def _label(r: LLMResponse) -> str:
    if r.refused:
        return REFUSED
    if r.stop_reason == "max_tokens":
        return ERROR
    try:
        return ScopeAnswer.model_validate_json(r.text).label
    except ValidationError:
        return ERROR


def classify(llm: LLM, text: str) -> Verdict:
    req = LLMRequest(tier=tier_for(TASK), model=model_override(TASK), effort=effort_for(TASK), task=TASK,
                     system=[Block(CLASSIFY_PROMPT)], max_tokens=1024, schema=SCHEMA, timeout=TIMEOUT_S,
                     messages=[{"role": "user", "content": json.dumps({"message": text}, ensure_ascii=False)}])
    try:
        r = llm.complete(req)
    except Exception as e:  # noqa: BLE001 - a broken classifier must not open the gate
        logger.error("classifier failed: %s", type(e).__name__)
        return Verdict(ERROR)
    verdict = Verdict(_label(r), r)
    logger.info("scope label=%s model=%s in=%s out=%s", verdict.label, r.model, r.input_tokens, r.output_tokens)
    return verdict


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
