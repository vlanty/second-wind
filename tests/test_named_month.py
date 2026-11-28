"""Questions about a named month are answered from the data, and the model sees month names.
Written before the code (2026-11-28: "сколько я пробежал в октябре" went to the model and got a
hedged answer because the data only said "previous month")."""

from datetime import date

import pytest

from secondwind.apps.bot import handlers
from secondwind.core import analytics, assistant
from secondwind.core.router import route
from secondwind.core.scope import Verdict, scope_route

TODAY = date(2026, 11, 28)


@pytest.mark.parametrize("text,start,end", [
    ("сколько я пробежал в октябре", date(2026, 10, 1), date(2026, 10, 31)),
    ("сколько км набегал за сентябрь", date(2026, 9, 1), date(2026, 9, 30)),
    ("сколько я пробежал в ноябре", date(2026, 11, 1), TODAY),                 # this month: so far
    ("сколько я пробежал в декабре", date(2025, 12, 1), date(2025, 12, 31)),   # a later month: last year
    ("сколько я пробежал в октябре 2025", date(2025, 10, 1), date(2025, 10, 31)),
    ("how far did I run in october", date(2026, 10, 1), date(2026, 10, 31)),
    ("how many km did I run in February 2026", date(2026, 2, 1), date(2026, 2, 28)),
])
def test_a_named_month_is_answered_without_the_model(text, start, end):
    r = route(text, "ru", TODAY)
    assert (r.kind, r.key) == ("intent", "fact.run_distance"), r
    assert (r.params["start"], r.params["end"]) == (start, end)


@pytest.mark.parametrize("text", ["сколько я пробежал 12 ноября", "сколько я пробежал с октября",
                                  "сколько я пробежал в октябре и ноябре", "how far did I run since october",
                                  "какой пульс покоя был в октябре", "сколько я пробежал в октябре прошлого года"])
def test_other_month_wording_still_goes_to_the_model(text):
    assert route(text, "ru", TODAY).kind == "model"


def test_the_classifier_path_reads_a_named_month_too():
    r = scope_route(Verdict("fact.run_distance"), "сколько намотал в октябре", "ru", TODAY)
    assert r.kind == "intent" and r.params["start"] == date(2026, 10, 1)


def test_the_answer_names_the_month(monkeypatch):
    class Reader:
        def get_period(self, start, end):
            return {"runs": 14, "run_km": 88.55}
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    monkeypatch.setenv("OWNER_USER_ID", "u-1")
    monkeypatch.setattr(handlers, "_bq", lambda update: Reader())
    upd = type("U", (), {"effective_chat": type("C", (), {"id": 42})()})()
    text, _ = handlers._fast_answer(upd, route("сколько я пробежал в октябре", "ru", TODAY), {})
    assert text.startswith("🏃 в октябре (01.10–31.10)") and ("88.5" in text or "88.6" in text)
    en, _ = handlers._fast_answer(upd, route("how far did I run in october 2025", "en", TODAY), {})
    assert en.startswith("🏃 in October 2025 (01.10–31.10)")


def test_the_model_sees_month_names():
    note = analytics.month_note({"n": 1}, {"n": 2}, TODAY)
    assert "November 2026 (1–28, so far)" in note and "October 2026 (whole month)" in note
    assert "previous month" not in note


def test_russian_answers_address_the_athlete_informally():
    assert "ты" in assistant.PERSONA and "informal" in assistant.PERSONA


def test_a_question_about_one_fact_gets_a_short_answer():
    assert "one or two sentences" in assistant.ASK_PROMPT
