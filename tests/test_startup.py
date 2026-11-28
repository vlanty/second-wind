"""Cold start: the bot accepts messages as soon as possible. Written before the code."""

import os
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "123456:test-token-not-real")

from fastapi.testclient import TestClient  # noqa: E402

from secondwind.apps.bot import main  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def test_chart_libraries_are_not_loaded_at_start():
    code = "import sys, secondwind.apps.bot.main; print(sorted(m for m in ('matplotlib', 'numpy') if m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True,
                         env={**os.environ, "TELEGRAM_BOT_TOKEN": "123456:test-token-not-real"})
    assert out.stdout.strip() == "[]"


def test_charts_still_render():
    from secondwind.apps.bot import charts

    png = charts.volume("ru", [{"week": date(2026, 11, d), "km": d} for d in (2, 9, 16, 23)])
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_startup_does_not_wait_for_the_telegram_menu(monkeypatch):
    import asyncio
    called = []

    async def fake_init():
        called.append("init")

    async def slow_menu(bot):
        called.append("menu")
        await asyncio.sleep(3)
        called.append("menu done")

    monkeypatch.setenv("GCP_PROJECT", "p")
    monkeypatch.setenv("BQ_DATASET", "d")
    monkeypatch.setattr(main.tg, "initialize", fake_init)
    monkeypatch.setattr(main, "setup_menu", slow_menu)
    start = time.monotonic()
    with TestClient(main.app) as c:
        ready = time.monotonic() - start
        assert c.get("/health").status_code == 200
        for _ in range(50):
            if "menu" in called:
                break
            time.sleep(0.01)
    assert ready < 1 and called[:2] == ["init", "menu"]
