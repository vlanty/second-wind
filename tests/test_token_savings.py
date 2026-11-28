"""Spending fewer tokens: prompt cache, short answers with "more", answers from the data for
frequent questions, an answer cache. Written before the code."""

import asyncio
import json
import logging
import types
from datetime import date, datetime, timedelta, timezone

import pytest

from secondwind.adapters.llm.claude import ClaudeLLM, cost_usd
from secondwind.adapters.state.memory import MemoryStore
from secondwind.apps.bot import handlers
from secondwind.core import assistant, models
from secondwind.core.i18n import t
from secondwind.core.ports import LLMResponse
from secondwind.core.router import route

TODAY = date.today()


# ── 1. prompt cache ───────────────────────────────────────────────────────────
def test_the_stable_part_of_a_question_is_marked_for_the_cache():
    req = assistant.build_request(assistant.ASK_PROMPT, "DATA", "chat", "ru", athlete="ATHLETE",
                                  question="как я?", dialog="Athlete: q\nVita: a")
    content = req.messages[0]["content"]
    cached, rest = content[:req.cache_prefix], content[req.cache_prefix:]
    assert "ATHLETE" in cached and "DATA" in cached and cached.rstrip().endswith("</data>")
    assert "как я?" in rest and "<dialog>" in rest and "<dialog>" not in cached


def test_commands_are_not_cached():
    """A /coach prefix is not shared with anything within the hour: the write would cost double."""
    assert assistant.build_request(assistant.COACH_PROMPT, "DATA", "analysis", "ru").cache_prefix == 0


class FakeClient:
    def __init__(self, usage):
        self.calls = []
        resp = types.SimpleNamespace(stop_reason="end_turn", model="claude-sonnet-5-5", stop_details=None,
                                     usage=types.SimpleNamespace(**usage),
                                     content=[types.SimpleNamespace(type="text", text="ok")])
        create = lambda **kw: self.calls.append(kw) or resp  # noqa: E731
        self.messages = types.SimpleNamespace(create=create)
        self.beta = types.SimpleNamespace(messages=types.SimpleNamespace(create=create))


def test_the_adapter_sends_a_one_hour_cache_mark_and_prices_cached_tokens():
    usage = dict(input_tokens=100, output_tokens=50, cache_creation_input_tokens=0, cache_read_input_tokens=5000)
    client = FakeClient(usage)
    req = assistant.build_request(assistant.ASK_PROMPT, "DATA", "chat", "ru", question="как я?")
    r = ClaudeLLM(client=client).complete(req)
    (first, second) = client.calls[0]["messages"][0]["content"]
    assert first["cache_control"] == {"type": "ephemeral", "ttl": "1h"} and "cache_control" not in second
    assert first["text"] + second["text"] == req.messages[0]["content"]
    assert r.cache_read_tokens == 5000 and r.cache_write_tokens == 0
    # Sonnet 5.5: input $2/M, cache read 0.05x, output $10/M
    assert r.cost_usd == pytest.approx((100 * 2 + 5000 * 2 * 0.05 + 50 * 10) / 1e6)


def test_a_cache_write_costs_double_for_the_hour():
    assert cost_usd("claude-sonnet-5-5", 100, 50, cache_write=5000) == pytest.approx((100 * 2 + 5000 * 4 + 500) / 1e6)


# ── 2. short answers and "more" ───────────────────────────────────────────────
def test_answers_are_short_and_more_is_detailed_on_a_stronger_model():
    short = assistant.build_request(assistant.ASK_PROMPT, "D", "chat", "ru", question="q").system[0].text
    more = assistant.build_request(assistant.ASK_PROMPT, "D", "chat", "ru", question="q", detail=True)
    assert "120 words" in short and "400 words" in more.system[0].text
    assert (models.more_tier("normal"), models.more_tier("economy"), models.more_tier("max")) == ("large", "medium", "large")
    assert more.tier == "large" and more.model is None


# ── 3. frequent questions from the data ───────────────────────────────────────
@pytest.mark.parametrize("text,key", [
    ("готов ли я сегодня к тренировке?", "calc.readiness"), ("я готова тренироваться?", "calc.readiness"),
    ("какая у меня готовность", "calc.readiness"), ("am I ready to train today?", "calc.readiness"),
    ("как моя нагрузка?", "fact.load"), ("какая у меня нагрузка сейчас", "fact.load"), ("what's my training load", "fact.load"),
    ("как я спал на этой неделе", "fact.metric"), ("средний пульс покоя за месяц", "fact.metric"),
    ("сколько шагов за месяц", "fact.metric"), ("какой hrv был в октябре", "fact.metric"),
    ("average resting heart rate last week", "fact.metric"),
])
def test_frequent_questions_are_answered_from_the_data(text, key):
    assert route(text, "ru", date(2026, 11, 28)).key == key


@pytest.mark.parametrize("text", ["готов ли я к длинной в воскресенье?", "готов ли я к марафону?",
                                  "почему у меня низкая готовность", "нагрузка не слишком большая?",
                                  "почему я плохо спал на этой неделе", "расскажи про мой сон за неделю"])
def test_questions_with_conditions_or_reasons_still_go_to_the_model(text):
    assert route(text, "ru", date(2026, 11, 28)).kind == "model"


class Reader:
    def get_health_last_days(self, days):
        return [{"date": TODAY - timedelta(days=i), "resting_hr": 45 + i % 3, "hrv_last_night": 55,
                 "hrv_baseline_low": 50, "hrv_baseline_high": 60, "sleep_seconds": 27000, "sleep_score": 80,
                 "stress_avg": 20, "body_battery_high": 90} for i in range(days - 1, -1, -1)]

    def get_load_last_days(self, days):
        return [{"date": TODAY, "acute_load": 300, "chronic_load": 310, "load_ratio": 0.97, "load_ratio_zone": "optimal"}]

    def get_health_period(self, start, end):
        days = (end - start).days + 1
        return [{"date": start + timedelta(days=i), "resting_hr": 44 + i % 4, "hrv_last_night": 50 + i % 5,
                 "sleep_seconds": 25200 + 600 * (i % 3), "deep_sleep_seconds": 4000, "sleep_score": 78,
                 "steps": 9000 + 100 * i, "stress_avg": 22, "body_battery_high": 85} for i in range(days)]


class Upd:
    effective_chat = type("C", (), {"id": 42})()


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    monkeypatch.setenv("OWNER_USER_ID", "u-1")
    monkeypatch.setattr(handlers, "_bq", lambda update: Reader())


def answer(text, lang="ru"):
    return handlers._fast_answer(Upd(), route(text, lang, TODAY), {})[0]


def test_readiness_and_load_answers(env):
    ready = answer("готов ли я сегодня к тренировке?")
    assert t("ru", "est.readiness") in ready and t("ru", "est.label") in ready
    assert "0.97" in answer("как моя нагрузка?")


def test_period_stats_answer(env):
    text = answer("средний пульс покоя за месяц")
    assert "45" in text and "44" in text and "47" in text          # average, lowest and highest
    sleep = answer("как я спал на этой неделе")
    assert ":" in sleep and t("ru", "stat.name.sleep") in sleep


# ── 4. answer cache and the question fingerprint ──────────────────────────────
class LLM:
    def __init__(self):
        self.requests = []

    def complete(self, req):
        self.requests.append(req)
        if req.task == "classify":
            return LLMResponse(json.dumps({"label": "health"}), "h", 400, 10, 0.0002)
        if req.task == "rewrite":
            return LLMResponse(json.dumps({"follow_up": False, "question": "x"}), "h", 300, 10, 0.0001)
        return LLMResponse(f"SCOPE: yes\nОтвет {len(self.answered())}", "s", 5000, 300, 0.01)

    def answered(self):
        return [r for r in self.requests if r.task not in ("classify", "rewrite")]


class Msg:
    def __init__(self, text):
        self.text, self.sent, self.markups = text, [], []

    async def reply_text(self, text, **kw):
        self.sent.append(text)
        self.markups.append(kw.get("reply_markup"))


class TUpd:
    effective_user = None
    callback_query = None

    def __init__(self, text):
        self.message = self.effective_message = Msg(text)
        self.effective_chat = type("C", (), {"id": 42})()


@pytest.fixture
def chat(monkeypatch):
    store = MemoryStore()
    for k, v in {"TELEGRAM_CHAT_ID": "42", "OWNER_USER_ID": "u-1", "BOT_LANGUAGE": "ru"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(handlers, "_store", lambda: store)
    monkeypatch.setattr(handlers, "_context", lambda bq: "DATA")
    monkeypatch.setattr(handlers, "_bq", lambda update: Reader())
    monkeypatch.setattr(handlers, "_athlete_inputs", lambda update: {})
    monkeypatch.setattr(handlers, "_data_version", lambda update: "sync-1")
    llm = LLM()
    monkeypatch.setattr(handlers, "_llm", lambda: llm)

    def send(text):
        store.update_guard("u-1", {})            # a later message, not a repeat within 2 minutes
        u = TUpd(text)
        asyncio.run(handlers.on_text(u, None))
        return u.message
    send.store, send.llm, send.mp = store, llm, monkeypatch
    return send


def test_the_same_question_is_answered_from_the_cache(chat):
    first = chat("Почему у меня вырос пульс покоя?").sent[-1]
    second = chat("почему у меня  вырос пульс покоя").sent[-1]          # case, spaces and punctuation differ
    assert first == second and len(chat.llm.answered()) == 1


def test_new_data_or_profile_or_mode_means_a_new_answer(chat):
    chat("почему у меня вырос пульс покоя?")
    chat.mp.setattr(handlers, "_data_version", lambda update: "sync-2")
    chat("почему у меня вырос пульс покоя?")
    chat.store.update_profile("u-1", {"weight_kg": 70})
    chat("почему у меня вырос пульс покоя?")
    chat.store.update_settings("u-1", {"mode": "max"})
    chat("почему у меня вырос пульс покоя?")
    assert len(chat.llm.answered()) == 4


def test_an_off_topic_answer_is_never_cached(chat):
    chat.llm.complete = (lambda orig: (lambda req: LLMResponse("SCOPE: no", "s", 5000, 5, 0.01)
                                       if req.task == "chat" else orig(req)))(chat.llm.complete)
    chat("почему у меня вырос пульс покоя?")
    assert chat.store.get_cached_answer("u-1", handlers._answer_key(TUpd("x"), {}, "почему у меня вырос пульс покоя?", "ru")) is None


def test_the_cache_keeps_a_bounded_number_of_answers():
    store = MemoryStore()
    for i in range(handlers.ANSWER_CACHE_KEEP + 5):
        store.put_cached_answer("u-1", f"k{i}", f"a{i}")
    assert store.get_cached_answer("u-1", "k0") is None and store.get_cached_answer("u-1", f"k{handlers.ANSWER_CACHE_KEEP + 4}")


def test_the_question_fingerprint_is_logged_without_the_text(chat, caplog):
    caplog.set_level(logging.INFO)
    chat("почему у меня вырос пульс покоя?")
    chat("почему у меня вырос пульс покоя?")
    prints = [r.getMessage() for r in caplog.records if "question fp=" in r.getMessage()]
    assert len(prints) == 2 and prints[0].split("fp=")[1][:10] == prints[1].split("fp=")[1][:10]
    assert "пульс" not in caplog.text and "cache=hit" in prints[1]


# ── 5. the "more" button ──────────────────────────────────────────────────────
def test_an_answer_has_a_more_button_that_asks_a_stronger_model_once(chat):
    msg = chat("почему у меня вырос пульс покоя?")
    (button,) = [b for row in msg.markups[-1].inline_keyboard for b in row]
    assert button.callback_data.startswith("more:")

    edits = []

    class Query:
        data = button.callback_data
        message = msg

        async def answer(self):
            pass

        async def edit_message_reply_markup(self, reply_markup=None):
            edits.append(reply_markup)

    u = TUpd("")
    u.callback_query = Query()
    asyncio.run(handlers.on_button(u, None))
    more = chat.llm.answered()[-1]
    assert more.tier == "large" and "400 words" in more.system[0].text and "почему у меня вырос пульс покоя?" in more.messages[0]["content"]
    assert edits == [None]                                                  # the button is removed


def test_a_fact_answer_has_no_more_button(chat):
    msg = chat("какой пульс покоя сегодня")
    assert msg.markups[-1] is None
