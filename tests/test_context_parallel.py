"""The data for the model is read with parallel queries. Written before the code (2026-11-28:
nine sequential BigQuery queries took 11 s of a 30 s answer)."""

import time
from datetime import date

from secondwind.apps.bot import handlers

PAUSE = 0.2


class SlowReader:
    def _w(self, v):
        time.sleep(PAUSE)
        return v

    def get_last_activity(self):
        return self._w({"name": "Run", "type": "running", "source": "garmin", "activity_date": date(2026, 11, 27),
                        "distance_meters": 8000, "duration_seconds": 2700})

    def get_week_activities(self, weeks_back=0):
        return self._w([])

    def get_health_last_days(self, days=14):
        return self._w([{"date": date(2026, 11, 28), "resting_hr": 44}])

    def get_load_last_days(self, days=1):
        return self._w([{"acute_load": 300, "chronic_load": 310, "load_ratio": 0.98, "load_ratio_zone": "optimal"}])

    def get_source_status(self):
        return self._w([])

    def get_period(self, start, end):
        return self._w({"n": 3, "runs": 2, "run_km": 16.0, "secs": 7200})

    def get_records(self, since):
        return self._w([])


class SequentialReader(SlowReader):
    pass


def test_the_context_is_read_in_parallel_and_is_unchanged(monkeypatch):
    monkeypatch.setenv("DATA_SOURCES", "")
    start = time.monotonic()
    parallel = handlers._context(SlowReader())
    took = time.monotonic() - start
    assert took < 4 * PAUSE, took                      # 9 queries one after another would take 9 * PAUSE
    monkeypatch.setattr(handlers, "CONTEXT_WORKERS", 1)
    assert handlers._context(SequentialReader()) == parallel
