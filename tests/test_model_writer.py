"""Re-reading recent days must never erase what an earlier run stored."""

from google.cloud import bigquery

from secondwind.adapters.warehouse.model_writer import BigQueryModelWriter


class Job:
    def result(self):
        return []


class Client:
    def __init__(self):
        self.loaded = []

    def get_table(self, name):
        cols = [("session_id", "STRING"), ("user_id", "STRING"), ("attrs", "JSON"), ("name", "STRING"), ("is_main", "BOOL")]
        return type("T", (), {"schema": [bigquery.SchemaField(n, t) for n, t in cols]})()

    def load_table_from_json(self, rows, stage, job_config=None):
        self.loaded.extend(rows)
        return Job()

    def query(self, sql, job_config=None):
        return Job()

    def delete_table(self, *a, **k):
        pass


def test_missing_values_are_left_out_of_the_load():
    """A None in a JSON column loads as JSON null, which COALESCE treats as a value: it would
    overwrite the stored attrs (fit_ref). An absent key loads as SQL NULL and is kept."""
    c = Client()
    BigQueryModelWriter("p", "d", "u-1", client=c).upsert(
        "sessions", [{"session_id": "s1", "attrs": None, "name": None, "is_main": False}])
    assert c.loaded == [{"session_id": "s1", "user_id": "u-1", "is_main": False}]


def test_real_values_including_falsy_ones_are_loaded():
    c = Client()
    BigQueryModelWriter("p", "d", "u-1", client=c).upsert(
        "sessions", [{"session_id": "s1", "attrs": {"fit_ref": "gs://b/1.zip"}, "name": "", "is_main": False}])
    assert c.loaded[0]["attrs"] == {"fit_ref": "gs://b/1.zip"}
    assert c.loaded[0]["name"] == "" and c.loaded[0]["is_main"] is False
