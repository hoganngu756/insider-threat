import pandas as pd

from src import placeholders
from src.store import SCHEMA


def flags(*rows):
    return pd.DataFrame(rows, columns=["employee_id", "date", "rule_name", "severity"])


def test_one_incident_per_employee_day_scored_from_worst_rule():
    rf = flags(("a.a", "2026-09-01", "bulk_download", "High"),
               ("a.a", "2026-09-01", "usb_exfiltration", "Medium"),
               ("b.b", "2026-09-02", "new_device_foreign_login", "Low"))
    out = placeholders.build_incidents(rf, None)
    assert list(out.columns) == SCHEMA["incidents"]
    out = out.set_index("employee_id")
    assert out.loc["a.a", "risk_score"] == 70
    assert out.loc["a.a", "severity"] == "High"
    assert out.loc["a.a", "triggered_rules"] == "bulk_download, usb_exfiltration"
    assert (out.loc["b.b", "risk_score"], out.loc["b.b", "severity"]) == (20, "Low")


def test_ml_anomalies_add_incidents_and_dates_come_from_timestamp():
    ml = pd.DataFrame({"employee_id": ["c.c", "d.d"], "timestamp": pd.to_datetime(["2026-09-03", "2026-09-03"]),
                       "anomaly_score": [0.8, 0.3], "anomaly_flag": [True, False]})
    out = placeholders.build_incidents(flags(), ml)
    assert out[["employee_id", "date", "triggered_rules", "risk_score"]].values.tolist() == [
        ["c.c", "2026-09-03", "ml_anomaly", 30]]
    assert out.loc[0, "incident_id"] == "INC-0001"


def test_no_inputs_gives_empty_schema():
    assert list(placeholders.build_incidents(None, None).columns) == SCHEMA["incidents"]
    assert list(placeholders.build_investigations(pd.DataFrame(columns=SCHEMA["incidents"])).columns) == \
        SCHEMA["ai_investigations"]


def test_investigations_are_marked_as_placeholders():
    inc = placeholders.build_incidents(flags(("a.a", "2026-09-01", "bulk_download", "High")), None)
    out = placeholders.build_investigations(inc)
    assert list(out.columns) == SCHEMA["ai_investigations"]
    assert all(out[c].str.startswith("PLACEHOLDER").all() for c in ["summary", "evidence", "recommended_steps"])
