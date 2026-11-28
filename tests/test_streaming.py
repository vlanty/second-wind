"""Answers appear while the model writes them. Written before the code."""

import asyncio
import time

import pytest
from telegram.error import BadRequest

from secondwind.adapters.state.memory import MemoryStore
from secondwind.apps.bot import handlers
from secondwind.core.i18n import t
from secondwind.core.ports import LLMResponse

QUESTION = "почему у меня вырос пульс покоя?"


class StreamLLM:
    """Classifies as on topic, then streams the answer in pieces with pauses."""

    def __init__(self, pieces, pause=0.05):
        self.pieces, self.pause, self.streamed = pieces, pause, False

    def complete(self, req):
        assert req.task == "classify", "answers must be streamed"
        return LLMResponse('{"label": "health"}', "h", 400, 10, 0.0002)

    def stream(self, req, on_text):
        self.streamed = True
        for p in self.pieces:
            time.sleep(self.pause)
            on_text(p)
        return LLMResponse("".join(self.pieces), "s", 5000, 600, 0.02)


class Sent:
    """A message the bot sent; records every edit."""

    def __init__(self, log, text, kw):
        self.log, self.edits = log, [(text, kw)]

    async def edit_text(self, text, **kw):
        if self.edits[-1][0] == text:
            raise BadRequest("Message is not modified")
        self.edits.append((text, kw))
        return self


class Chat:
    id = 42

    def __init__(self):
        self.messages = []

    async def reply_text(self, text, **kw):
        m = Sent(self.messages, text, kw)
        self.messages.append(m)
        return m


class Upd:
    effective_user = None
    callback_query = None

    def __init__(self, text):
        self.chat = Chat()
        self.message = self.effective_message = type("M", (), {"text": text, "reply_text": self.chat.reply_text})()
        self.effective_chat = type("C", (), {"id": 42})()


@pytest.fixture
def bot(monkeypatch):
    store = MemoryStore()
    for k, v in {"TELEGRAM_CHAT_ID": "42", "OWNER_USER_ID": "u-1", "BOT_LANGUAGE": "ru"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(handlers, "_store", lambda: store)
    monkeypatch.setattr(handlers, "_context", lambda bq: "DATA")
    monkeypatch.setattr(handlers, "_bq", lambda update: None)
    monkeypatch.setattr(handlers, "_athlete_inputs", lambda update: {})
    monkeypatch.setattr(handlers, "EDIT_EVERY_S", 0.1)

    def send(llm, text=QUESTION, command=None):
        monkeypatch.setattr(handlers, "_llm", lambda: llm)
        u = Upd(text)
        asyncio.run(command(u, None) if command else handlers.on_text(u, None))
        return u.chat.messages
    return send


PIECES = ["SCOPE: yes\n", "**Пульс** ", "держался 45–47, ", "сейчас 44. ", "Всё ", "в норме."]


def test_the_answer_grows_in_one_message_and_ends_formatted(bot):
    llm = StreamLLM(PIECES)
    (msg,) = bot(llm)
    texts = [e[0] for e in msg.edits]
    assert llm.streamed and texts[0] == t("ru", "ai.working", label=t("ru", "ai.label.ask"))
    partial = texts[1:-1]
    assert len(partial) >= 2 and all(len(a) <= len(b) for a, b in zip(partial, partial[1:]))   # it grows
    final, kw = msg.edits[-1]
    assert final == "<b>Пульс</b> держался 45–47, сейчас 44. Всё в норме." and kw.get("parse_mode") == "HTML"
    assert not any("SCOPE" in x for x in texts)


def test_nothing_of_an_off_topic_answer_is_ever_shown(bot):
    (msg,) = bot(StreamLLM(["SCOPE: no\n", "Вот рецепт ", "шарлотки: ", "мука, яйца."]))
    assert msg.edits[-1][0] == t("ru", "filter.off_topic")
    assert not any("рецепт" in e[0] or "SCOPE" in e[0] for e in msg.edits)


def test_a_long_answer_continues_in_new_messages(bot):
    long = ["SCOPE: yes\n"] + [f"Абзац {i}. " + "слово " * 120 + "\n\n" for i in range(10)]
    msgs = bot(StreamLLM(long, pause=0.01))
    assert len(msgs) >= 2 and all(len(m.edits[-1][0]) <= 4096 for m in msgs)
    shown = " ".join(m.edits[-1][0] for m in msgs)
    assert "Абзац 0." in shown and "Абзац 9." in shown


def test_commands_are_streamed_too(bot):
    llm = StreamLLM(["Отчёт ", "тренера."])
    (msg,) = bot(llm, command=handlers.cmd_coach)
    assert llm.streamed and msg.edits[-1][0] == "Отчёт тренера."


def test_a_failed_stream_shows_the_error_in_the_same_message(bot):
    class Broken(StreamLLM):
        def stream(self, req, on_text):
            on_text("SCOPE: yes\nНачало")
            raise RuntimeError("connection reset")
    (msg,) = bot(Broken([]))
    assert msg.edits[-1][0] == t("ru", "ai.error")
