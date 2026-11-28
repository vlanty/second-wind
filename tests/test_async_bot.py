"""The bot never blocks its event loop: everything that waits for the network runs in a thread.
Written before the code (architecture 10: blocking clients inside async handlers are not allowed)."""

import ast
import asyncio
import inspect
import json
import time

import pytest

from secondwind.adapters.state.memory import MemoryStore
from secondwind.apps.bot import handlers
from secondwind.core.ports import LLMResponse

SLOW = 0.4
# Calls that wait for BigQuery, Firestore, the AI model or render charts.
BLOCKING = {"_bq", "_store", "_llm", "_user", "_athlete_inputs", "_context", "_fast_answer", "_scope_check",
            "_guarded_route", "analyze", "classify", "handle_input", "apply_setting", "render", "_record_usage", "_fast_metric", "_race_prediction", "_athlete",
            "_today_data", "_remember", "_recent_dialog",
            "_answer_key", "_cached_answer", "_cache_answer", "_data_version"}


def _calls_in(node):
    """Calls made directly in this function body (not inside lambdas or nested functions)."""
    stack = list(ast.iter_child_nodes(node))
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if isinstance(n, ast.Call):
            yield n
        stack.extend(ast.iter_child_nodes(n))


def _name(call):
    f = call.func
    return f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""


def test_async_handlers_make_no_blocking_calls():
    tree = ast.parse(inspect.getsource(handlers))
    bad = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.AsyncFunctionDef):
            continue
        for call in _calls_in(fn):
            name = _name(call)
            if name in BLOCKING or (name == "_lang" and len(call.args) < 2):   # _lang(update) reads Firestore
                bad.append(f"{fn.name}: {name}() line {call.lineno}")
    assert bad == []


class SlowLLM:
    def complete(self, req):
        time.sleep(SLOW)
        if req.task == "classify":
            return LLMResponse(json.dumps({"label": "training"}), "h", 400, 10, 0.0001)
        return LLMResponse("ANSWER", "m", 10, 5, 0.01)


class Msg:
    def __init__(self, text):
        self.text, self.sent = text, []

    async def reply_text(self, text, **kw):
        self.sent.append(text)


class Upd:
    effective_user = None
    callback_query = None

    def __init__(self, text):
        self.message = self.effective_message = Msg(text)
        self.effective_chat = type("C", (), {"id": 42})()


class Reader:
    def get_source_status(self):
        time.sleep(SLOW)
        return []


@pytest.fixture
def bot(monkeypatch):
    store = MemoryStore()
    for k, v in {"TELEGRAM_CHAT_ID": "42", "OWNER_USER_ID": "u-1", "BOT_LANGUAGE": "ru", "DATA_SOURCES": ""}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(handlers, "_store", lambda: store)
    monkeypatch.setattr(handlers, "_llm", lambda: SlowLLM())
    monkeypatch.setattr(handlers, "_context", lambda bq: "DATA")
    monkeypatch.setattr(handlers, "_bq", lambda update: Reader())
    monkeypatch.setattr(handlers, "_athlete_inputs", lambda update: {})


def test_two_questions_are_answered_at_the_same_time(bot):
    a, b = Upd("почему вырос пульс покоя?"), Upd("как подвестись к старту?")

    async def both():
        await asyncio.gather(handlers.on_text(a, None), handlers.on_text(b, None))

    start = time.monotonic()
    asyncio.run(both())
    took = time.monotonic() - start
    assert "ANSWER" in a.message.sent[-1] and "ANSWER" in b.message.sent[-1]
    assert took < 3 * SLOW, took          # one question = classifier + answer = 2 * SLOW


def test_a_command_is_not_held_up_by_a_question(bot):
    q, cmd = Upd("почему вырос пульс покоя?"), Upd("/status")
    done = {}

    async def run():
        async def timed(name, coro):
            await coro
            done[name] = time.monotonic()
        start = time.monotonic()
        await asyncio.gather(timed("question", handlers.on_text(q, None)), timed("status", handlers.cmd_status(cmd, None)))
        return start

    start = asyncio.run(run())
    assert done["status"] - start < 1.5 * SLOW and done["status"] < done["question"]
