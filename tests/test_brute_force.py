import pandas as pd

from src.detection.brute_force import detect_brute_force


def make_logins(rows):
    """rows: (minute offset, user, outcome, ip)"""
    base = pd.Timestamp("2026-08-03 10:00:00")
    return pd.DataFrame([
        dict(timestamp=base + pd.Timedelta(minutes=m), username=u, outcome=o, src_ip=ip)
        for m, u, o, ip in rows
    ])


def test_burst_is_flagged_with_evidence():
    logins = make_logins([(i, "a.b", "failure", "1.2.3.4") for i in range(12)])
    flags = detect_brute_force(logins)
    assert len(flags) == 1
    row = flags.iloc[0]
    assert (row.employee_id, row.date, row.rule_name, row.severity) == ("a.b", "2026-08-03", "brute_force", "Medium")
    assert row.timestamp == pd.Timestamp("2026-08-03 10:00:00")
    assert "12 failed logins" in row.evidence and "1.2.3.4" in row.evidence


def test_few_failures_not_flagged():
    logins = make_logins([(i, "a.b", "failure", "1.2.3.4") for i in range(9)])
    assert detect_brute_force(logins).empty


def test_slow_failures_not_flagged():
    # 12 failures but 1 every 20 minutes: never 10 inside 30 minutes
    logins = make_logins([(i * 20, "a.b", "failure", "1.2.3.4") for i in range(12)])
    assert detect_brute_force(logins).empty


def test_failures_are_counted_per_user():
    # 6 failures each for two users: 12 in total but no single account reaches 10
    rows = [(i, "a.b", "failure", "1.2.3.4") for i in range(6)] + \
           [(i, "c.d", "failure", "1.2.3.4") for i in range(6)]
    assert detect_brute_force(make_logins(rows)).empty


def test_success_after_burst_is_high_severity():
    rows = [(i, "a.b", "failure", "1.2.3.4") for i in range(10)] + [(15, "a.b", "success", "1.2.3.4")]
    flags = detect_brute_force(make_logins(rows))
    assert flags.iloc[0].severity == "High" and "successful login followed" in flags.iloc[0].evidence


def test_one_burst_gives_one_flag():
    logins = make_logins([(i, "a.b", "failure", "1.2.3.4") for i in range(40)])
    assert len(detect_brute_force(logins)) == 1


def test_two_separate_bursts_give_two_flags():
    rows = [(i, "a.b", "failure", "1.2.3.4") for i in range(10)] + \
           [(600 + i, "a.b", "failure", "1.2.3.4") for i in range(10)]
    assert len(detect_brute_force(make_logins(rows))) == 2
