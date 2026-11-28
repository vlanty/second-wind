"""Second lock in the answer and spending limits. Written before the code."""

import asyncio
import logging
from datetime import datetime, timedelta, timezone

import pytest

from secondwind.adapters.state.memory import MemoryStore
from secondwind.apps.bot import handlers
from secondwind.core import assistant, guard
from secondwind.core.i18n import t
from secondwind.core.ports import LLMResponse

NOW = datetime(2026, 11, 28, 12, 0, tzinfo=timezone.utc)
QUESTION = "почему у меня вырос пульс покоя?"


# ── second lock ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize("text,on_topic,body", [
    ("SCOPE: yes\nПульс в норме.", True, "Пульс в норме."),
    ("SCOPE: no", False, ""),
    ("  scope: NO \n\nрецепт: мука", False, ""),           # whatever follows a "no" is never shown
    ("Пульс в норме.", None, "Пульс в норме."),             # no marker: the classifier already let it in
])
def test_the_scope_line_is_read_and_removed(text, on_topic, body):
    assert assistant.split_scope(text) == (on_topic, body)


def test_the_ask_prompt_asks_for_the_scope_line():
    assert "SCOPE: yes" in assistant.ASK_PROMPT and "SCOPE: no" in assistant.ASK_PROMPT
    for name in ("COACH_PROMPT", "HEALTH_PROMPT", "NUTRITION_PROMPT"):   # commands have no question
        assert "SCOPE:" not in getattr(assistant, name)


@pytest.mark.parametrize("task,limit", [("chat", 3000), ("analysis", 4000)])
def test_answers_have_a_realistic_token_ceiling(task, limit):
    req = assistant.build_request(assistant.ASK_PROMPT, "DATA", task, "ru", question="как я?")
    assert req.max_tokens == limit


# ── daily budget ──────────────────────────────────────────────────────────────
def test_daily_spending_is_counted_per_day():
    g = guard.record_spend({}, NOW, 0.6)
    g = guard.record_spend(g, NOW + timedelta(hours=1), 0.5)
    assert not guard.budget_allowed(g, NOW + timedelta(hours=2), limit_usd=1.0)
    assert guard.budget_allowed(g, NOW + timedelta(days=1), limit_usd=1.0)            # a new day
    assert guard.record_spend(g, NOW + timedelta(days=1), 0.1)["spent_usd"] == pytest.approx(0.1)


class LLM:
    def __init__(self, answer):
        self.answer, self.tasks = answer, []

    def complete(self, req):
        self.tasks.append(req.task)
        if req.task == "classify":
            return LLMResponse('{"label": "health"}', "h", 400, 10, 0.0002)
        return LLMResponse(self.answer, "s", 5000, 600, 0.4)


class Msg:
    def __init__(self, text):
        self.text, self.sent = text, []

    async def reply_text(self, text, **kw):
        self.sent.append(text)
        return self


class Upd:
    effective_user = None
    callback_query = None

    def __init__(self, text):
        self.message = self.effective_message = Msg(text)
        self.effective_chat = type("C", (), {"id": 42})()


@pytest.fixture
def bot(monkeypatch):
    store = MemoryStore()
    for k, v in {"TELEGRAM_CHAT_ID": "42", "OWNER_USER_ID": "u-1", "BOT_LANGUAGE": "ru", "DAILY_BUDGET_USD": "1"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(handlers, "_store", lambda: store)
    monkeypatch.setattr(handlers, "_context", lambda bq: "DATA")
    monkeypatch.setattr(handlers, "_bq", lambda update: None)
    monkeypatch.setattr(handlers, "_athlete_inputs", lambda update: {})

    def send(llm, text):
        monkeypatch.setattr(handlers, "_llm", lambda: llm)
        u = Upd(text)
        asyncio.run(handlers.on_text(u, None))
        return u.message.sent
    return send


def test_an_answer_marked_off_topic_is_replaced_by_the_fixed_text(bot):
    sent = bot(LLM("SCOPE: no\nВот рецепт шарлотки: ..."), QUESTION)
    assert sent[-1] == t("ru", "filter.off_topic") and not any("шарлотк" in s for s in sent)


def test_the_scope_line_is_not_shown(bot):
    sent = bot(LLM("SCOPE: yes\n**Пульс** в норме."), QUESTION)
    assert "Пульс" in sent[-1] and not any("SCOPE" in s for s in sent)


def test_a_missing_scope_line_is_logged_and_the_answer_shown(bot, caplog):
    caplog.set_level(logging.WARNING)
    assert "Пульс в норме" in bot(LLM("Пульс в норме."), QUESTION)[-1]
    assert "scope line missing" in caplog.text


def test_questions_stop_when_the_daily_budget_is_spent(bot):
    llm = LLM("SCOPE: yes\nок")
    bot(llm, QUESTION)                                     # 0.4 + classifier
    bot(llm, "а почему упал HRV?")                         # 0.8
    bot(llm, "что делать со сном?")                        # 1.2: over the limit after this one
    tasks = len(llm.tasks)
    assert bot(llm, "а с весом что?") == [t("ru", "filter.budget_spent")] and len(llm.tasks) == tasks
    assert "112" in bot(llm, "болит в груди")[-1]          # emergencies are always answered


def test_commands_respect_the_budget_too(bot, monkeypatch):
    llm = LLM("ок")
    for q in ("почему вырос пульс?", "почему упал HRV?", "почему плохой сон?"):
        bot(llm, q)
    monkeypatch.setattr(handlers, "_llm", lambda: llm)
    u = Upd("/coach")
    asyncio.run(handlers.cmd_coach(u, None))
    assert u.message.sent == [t("ru", "filter.budget_spent")]
