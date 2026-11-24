import os
import sqlite3

import pandas as pd
import pytest

from src import store


def tables():
    return {
        "logins": pd.DataFrame({
            "timestamp": pd.to_datetime(["2026-09-01 08:00", "2026-09-01 02:00", "2026-09-02 09:00"]),
            "username": ["a.a", "a.a", "b.b"], "dev": ["d1", "d1", "d2"], "src_ip": None, "city": None,
            "country": ["US", "US", "US"], "outcome": ["success", "failure", "success"], "failure_reason": None,
        }),
        "file_events": pd.DataFrame({
            "timestamp": pd.to_datetime(["2026-09-01 10:00", "2026-09-01 11:00", "2026-09-01 12:00"]),
            "username": ["a.a", "a.a", "a.a"], "dev": "d1", "obj": "/x", "dest": None, "sensitivity": None,
            "action": ["usb_copy", "download", "email_external"], "size_bytes": [100, 5000, 50],
        }),
        "rule_flags": pd.DataFrame({
            "employee_id": ["a.a", "a.a"], "date": ["2026-09-01", "2026-09-01"],
            "timestamp": pd.to_datetime(["2026-09-01 10:00", "2026-09-01 11:00"]),
            "rule_name": ["usb_exfiltration", "bulk_download"], "severity": ["Medium", "High"], "evidence": ["e1", "e2"],
        }),
        "ml_scores": pd.DataFrame({
            "employee_id": ["a.a", "b.b"], "timestamp": pd.to_datetime(["2026-09-01", "2026-09-02"]),
            "failed_logins": [1, 0], "anomaly_score": [0.8, 0.4], "anomaly_flag": [True, False],
        }),
        "incidents": pd.DataFrame({
            "incident_id": ["INC-1"], "employee_id": ["a.a"], "date": ["2026-09-01"],
            "risk_score": [90], "severity": ["High"], "triggered_rules": ["bulk_download"],
        }),
        "ai_investigations": pd.DataFrame({
            "incident_id": ["INC-1"], "summary": ["s"], "evidence": ["ev"], "recommended_steps": ["r"],
        }),
    }


def test_normalize_renames_username_and_derives_date():
    out = store.normalize(tables()["logins"], "logins")
    assert list(out.columns[:3]) == ["employee_id", "date", "timestamp"]
    assert out.loc[0, "date"] == "2026-09-01"
    assert out.loc[0, "timestamp"] == "2026-09-01 08:00:00"


def test_normalize_keeps_extras_after_required_and_stores_bools_as_ints():
    out = store.normalize(tables()["ml_scores"], "ml_scores")
    assert list(out.columns) == ["employee_id", "date", "anomaly_score", "anomaly_flag", "timestamp", "failed_logins"]
    assert out["anomaly_flag"].tolist() == [1, 0]


def test_normalize_empty_gives_schema_columns():
    assert list(store.normalize(None, "incidents").columns) == store.SCHEMA["incidents"]


def test_normalize_rejects_missing_required_column():
    with pytest.raises(ValueError, match="severity"):
        store.normalize(tables()["incidents"].drop(columns=["severity"]), "incidents")


def test_save_all_writes_tables_and_views(tmp_path):
    db = str(tmp_path / "s.db")
    counts = store.save_all(tables(), db)
    assert counts == {"logins": 3, "file_events": 3, "rule_flags": 2, "ml_scores": 2, "incidents": 1, "ai_investigations": 1}
    with sqlite3.connect(db) as con:
        names = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")}
    assert {"v_employee_day", "v_employee_summary", "v_incident_detail", "v_rule_flags_daily"} <= names
    assert not os.path.exists(db + ".tmp")


def test_employee_day_view_joins_every_stage(tmp_path):
    db = str(tmp_path / "s.db")
    store.save_all(tables(), db)
    day = store.read("v_employee_day", db).set_index(["employee_id", "date"])
    a = day.loc[("a.a", "2026-09-01")]
    assert (a["logins"], a["failed_logins"], a["file_events"]) == (2, 1, 3)
    assert a["bytes_sent_out"] == 150
    assert (a["rule_flags"], a["rule_severity"]) == (2, "High")
    assert (a["anomaly_flag"], a["ml_failed_logins"]) == (1, 1)
    assert (a["incidents"], a["risk_score"], a["incident_severity"]) == (1, 90, "High")
    b = day.loc[("b.b", "2026-09-02")]
    assert (b["rule_flags"], b["incidents"]) == (0, 0)
    assert pd.isna(b["rule_severity"])
    assert len(day) == 2


def test_summary_and_incident_detail(tmp_path):
    db = str(tmp_path / "s.db")
    store.save_all(tables(), db)
    summary = store.read("v_employee_summary", db).set_index("employee_id")
    assert summary.loc["a.a", "top_rule_severity"] == "High"
    assert summary.loc["a.a", "last_flagged_date"] == "2026-09-01"
    assert pd.isna(summary.loc["b.b", "last_flagged_date"])
    detail = store.read("v_incident_detail", db)
    assert detail.loc[0, "summary"] == "s"
    assert detail.loc[0, "ai_evidence"] == "ev"


def test_save_all_works_with_only_required_stages_and_replaces_old_db(tmp_path):
    db = str(tmp_path / "s.db")
    store.save_all(tables(), db)
    t = tables()
    store.save_all({"logins": t["logins"], "file_events": t["file_events"], "rule_flags": t["rule_flags"]}, db)
    assert store.read("incidents", db).empty
    assert store.read("v_incident_detail", db).empty
    assert len(store.read("v_employee_day", db)) == 2
