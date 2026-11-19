import pandas as pd

from src.detection.new_device import detect_new_device_foreign

BASE = pd.Timestamp("2026-08-03 09:00:00")


def login(day, user="a.b", dev="dev-1", country="US", outcome="success"):
    return dict(timestamp=BASE + pd.Timedelta(days=day), username=user, dev=dev,
                country=country, outcome=outcome)


def run(rows, files=None):
    files = pd.DataFrame(files or [], columns=["timestamp", "username", "dev"])
    return detect_new_device_foreign(pd.DataFrame(rows), files)


def test_new_device_and_foreign_is_high():
    flags = run([login(0), login(5), login(40, dev="dev-new", country="NG")])
    assert len(flags) == 1
    row = flags.iloc[0]
    assert (row.rule_name, row.severity, row.date) == ("new_device_foreign_login", "High", "2026-09-12")
    assert "dev-new" in row.evidence and "NG" in row.evidence and "US" in row.evidence


def test_new_device_only_is_low():
    assert run([login(0), login(40, dev="dev-2")]).iloc[0].severity == "Low"


def test_foreign_only_is_medium():
    assert run([login(0), login(40, country="RO")]).iloc[0].severity == "Medium"


def test_second_device_during_baseline_not_flagged():
    assert run([login(0), login(10, dev="dev-2"), login(40)]).empty


def test_normal_logins_not_flagged():
    assert run([login(0), login(30), login(40), login(50)]).empty


def test_new_device_only_flagged_on_first_use():
    flags = run([login(0), login(40, dev="dev-2"), login(41, dev="dev-2")])
    assert len(flags) == 1


def test_failed_logins_are_ignored():
    assert run([login(0), login(40, dev="dev-x", country="CN", outcome="failure")]).empty


def test_device_seen_in_file_events_is_not_new():
    files = [dict(timestamp=BASE + pd.Timedelta(days=5), username="a.b", dev="dev-2")]
    assert run([login(0), login(40, dev="dev-2")], files).empty


def test_missing_country_and_device_not_flagged():
    assert run([login(0), login(40, dev=None, country=None)]).empty


def test_history_is_per_user():
    # b.c uses dev-2 from the start; a.b's first use of dev-2 is new
    rows = [login(0), login(0, user="b.c", dev="dev-2"), login(40, dev="dev-2"), login(41, user="b.c", dev="dev-2")]
    flags = run(rows)
    assert list(flags.employee_id) == ["a.b"]
