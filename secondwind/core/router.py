"""Filter and rule router: what can be answered without the AI model.

Rules are data (secondwind/contracts/rules/*.yaml, per language); this module only runs them and
extracts parameters (period, metric, numbers). Golden cases in contracts/golden/router.yaml are
the contract every implementation must pass.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import yaml

RULES_DIR = Path(__file__).resolve().parent.parent / "contracts" / "rules"
LANGS = ("ru", "en")
MAX_CHARS = 1000             # longer messages are not sent to the model (architecture 2.1)
FLAGS = re.IGNORECASE | re.UNICODE


@dataclass
class Route:
    kind: str                     # "filter" | "intent" | "model"
    key: str                      # filter.* / fact.* / calc.* / "model"
    lang: str
    params: dict = field(default_factory=dict)


def detect_lang(text: str) -> str | None:
    """ru if the message has Cyrillic, en if it has Latin letters, None for anything else."""
    if re.search(r"[А-Яа-яЁё]", text):
        return "ru"
    if re.search(r"[A-Za-z]", text):
        return "en"
    return None


@lru_cache(maxsize=1)
def _rules() -> tuple[list, dict, list]:
    f = yaml.safe_load((RULES_DIR / "filter.yaml").read_text(encoding="utf-8"))
    r = yaml.safe_load((RULES_DIR / "router.yaml").read_text(encoding="utf-8"))
    def compiled(d):
        return {lg: [re.compile(p, FLAGS) for p in (d or {}).get(lg, [])] for lg in LANGS}

    filters = [(x["key"], compiled(x), compiled(x.get("unless")), bool(x.get("any_language"))) for x in f["rules"]]
    to_model = {lg: [re.compile(p, FLAGS) for p in r["to_model"].get(lg, [])] for lg in LANGS}
    intents = [(x["intent"], {lg: [re.compile(p, FLAGS) for p in x.get(lg, [])] for lg in LANGS}) for x in r["intents"]]
    return filters, to_model, intents


def _any(patterns, text) -> bool:
    return any(p.search(text) for p in patterns)


RED_FLAG = "filter.red_flag"


def _filter_key(text: str, lang: str, only: str | None = None) -> str | None:
    """Key of the first filter rule that matches. Rules marked any_language (emergencies, attempts
    to change the instructions) are checked in every language: people mix them in one message."""
    for key, pats, unless, any_language in _rules()[0]:
        if only and key != only:
            continue
        langs = LANGS if any_language else (lang,)
        if any(_any(pats[lg], text) for lg in langs) and not any(_any(unless[lg], text) for lg in langs):
            return key
    return None


def is_red_flag(text: str) -> bool:
    """Possible emergency. Checked before anything else, whatever the length or the dialog state."""
    text = _normalize(text)
    lang = detect_lang(text)
    return lang is not None and _filter_key(text, lang, only=RED_FLAG) is not None


def _normalize(text: str | None) -> str:
    return (text or "").strip().replace("\u2019", "'").replace("ё", "е").replace("Ё", "Е")


def route(text: str, ui_lang: str = "en", today: date | None = None) -> Route:
    """Decide how to answer a free-form message."""
    text = _normalize(text)
    lang = detect_lang(text)
    if not text:
        return Route("filter", "filter.empty", ui_lang)
    if is_red_flag(text):
        return Route("filter", RED_FLAG, lang or ui_lang)
    if len(text) > MAX_CHARS:
        return Route("filter", "filter.too_long", lang or ui_lang)
    if lang is None:
        return Route("filter", "filter.unsupported_language", ui_lang)
    _, to_model, intents = _rules()
    key = _filter_key(text, lang)
    if key:
        return Route("filter", key, lang)
    if _any(to_model[lang], text):
        return Route("model", "model", lang)
    today = today or date.today()
    for intent, pats in intents:
        if _any(pats[lang], text):
            params = extract(intent, text, lang, today)
            if params is not None:
                return Route("intent", intent, lang, params)
    return Route("model", "model", lang)


# ── parameters ────────────────────────────────────────────────────────────────
PERIODS = [  # (regex, period code); first match wins
    (r"позавчера|day before yesterday", "day_before"),
    (r"вчера|yesterday|last night", "yesterday"),
    (r"сегодня|today|this morning|tonight", "today"),
    (r"прошл\w* недел|last week", "last_week"),
    (r"(эт\w*|текущ\w*|за) недел|this week|past week", "week"),
    (r"прошл\w* месяц|last month", "last_month"),
    (r"(эт\w*|текущ\w*|за) месяц|this month|past month", "month"),
    (r"(эт\w*|текущ\w*|за) год|this year|past year", "year"),
]

METRICS = [  # (regex, metric code)
    (r"пульс\w* (в )?поко|чсс поко|resting (heart rate|hr)|\brhr\b", "resting_hr"),
    (r"hrv|вариабельн", "hrv"),
    (r"body battery|батаре", "body_battery"),
    (r"стресс|stress", "stress"),
    (r"шаг|steps", "steps"),
    (r"vo2|мпк", "vo2max"),
    (r"\bсон\b|\bсна\b|\bспал\w*|sleep|slept", "sleep"),
]


def period_dates(code: str, today: date) -> tuple[date, date]:
    monday = today - timedelta(days=today.weekday())
    first = today.replace(day=1)
    prev_end = first - timedelta(days=1)
    return {
        "today": (today, today), "yesterday": (today - timedelta(days=1),) * 2,
        "day_before": (today - timedelta(days=2),) * 2,
        "week": (monday, today), "last_week": (monday - timedelta(days=7), monday - timedelta(days=1)),
        "month": (first, today), "last_month": (prev_end.replace(day=1), prev_end),
        "year": (today.replace(month=1, day=1), today),
    }[code]


def _period(text: str, default: str) -> str:
    for pattern, code in PERIODS:
        if re.search(pattern, text, FLAGS):
            return code
    return default


NAMED_DISTANCES = [(r"полумарафон|half[- ]?marathon", 21.0975), (r"марафон|marathon", 42.195)]
NUM = r"(\d+(?:[.,]\d+)?)"


def _seconds(token: str) -> int | None:
    parts = token.replace(".", ":").split(":")
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return None
    if len(nums) == 3:
        return nums[0] * 3600 + nums[1] * 60 + nums[2]
    if len(nums) == 2:
        return nums[0] * 60 + nums[1]
    return None


def _pace_params(text: str) -> dict | None:
    t = text.lower()
    dist = None
    for pattern, km in NAMED_DISTANCES:
        if re.search(pattern, t):
            dist = km
            break
    m = re.search(NUM + r"\s*(км|km|k)\b", t)
    if m:
        dist = float(m.group(1).replace(",", "."))
    pace = None
    m = re.search(r"(?:темп\w*|pace)\s*(?:of\s*)?(\d+[:.]\d\d)", t) or re.search(r"(\d+[:.]\d\d)\s*(?:на|/|per)\s*(?:км|km)", t)
    if m:
        pace = _seconds(m.group(1))
    total = None
    m = re.search(r"(?:\bза\b|\bin\b)\s*(\d+:\d\d(?::\d\d)?)", t)
    if m:
        total = _seconds(m.group(1))
    else:
        h = re.search(r"(?:\bза\b|\bin\b)\s*(\d+)\s*(?:ч|час\w*|h|hours?)\b(?:\s*(\d+)\s*(?:мин\w*|min\w*))?", t)
        mins = re.search(r"(?:\bза\b|\bin\b)\s*(\d+)\s*(?:мин\w*|min\w*)", t)
        if h:
            total = int(h.group(1)) * 3600 + int(h.group(2) or 0) * 60
        elif mins:
            total = int(mins.group(1)) * 60
    if not dist or dist <= 0 or dist > 1000:
        return None
    if total and total > 0:
        return {"distance_km": dist, "total_s": total}
    if pace and pace > 0:
        return {"distance_km": dist, "pace_s": pace}
    return None


def _race_params(text: str) -> dict | None:
    t = text.lower()
    for pattern, km in NAMED_DISTANCES:
        if re.search(pattern, t):
            return {"distance_km": km}
    m = re.search(NUM + r"\s*(км|km|k)\b", t)
    if m:
        km = float(m.group(1).replace(",", "."))
        return {"distance_km": km} if 1 <= km <= 100 else None
    return None


# A period these rules cannot read (a month, a weekday, a date, "since ..."): the model answers.
OTHER_PERIOD = re.compile(
    r"\b(январ|феврал|март[ае]?\b|апрел|ма[йея]\b|июн|июл|август|сентябр|октябр|ноябр|декабр)\w*"
    r"|\b(понедельник|вторник|сред[ау]\b|четверг|пятниц|суббот|воскресень)\w*|\b(лет[оа]|летом|осен\w*|зим\w*|весн\w*)\b"
    r"|\bс начала\b|\b\d+\W+(дн|день|недел|месяц)\w*|\b\d{1,2}\.\d{1,2}\b"
    r"|\b(january|february|march|april|may|june|july|august|september|october|november|december)\b"
    r"|\b(mon|tues|wednes|thurs|fri|satur|sun)days?\b|\b(summer|autumn|fall|winter|spring)\b|\bsince\b"
    r"|\b\d+\W+(days?|weeks?|months?)\b", FLAGS)


def extract(intent: str, text: str, lang: str, today: date) -> dict | None:
    """Parameters for an intent, or None when they cannot be read (then the model answers)."""
    if intent == "calc.pace":
        return _pace_params(text)
    if intent == "calc.race_predict":
        return _race_params(text)
    if intent in ("fact.run_distance", "fact.metric") and OTHER_PERIOD.search(text):
        return None
    if intent == "fact.run_distance":
        code = _period(text, "week")
        start, end = period_dates(code, today)
        return {"period": code, "start": start, "end": end}
    if intent == "fact.metric":
        metric = next((code for pattern, code in METRICS if re.search(pattern, text, FLAGS)), None)
        if not metric:
            return None
        code = _period(text, "today")
        if code not in ("today", "yesterday", "day_before"):
            return None                      # a value over a period is an analysis question
        return {"metric": metric, "period": code, "date": period_dates(code, today)[0]}
    return {}
