import pandas as pd

from src.detection.credential_stuffing import detect_credential_stuffing


def make_logins(rows):
    """rows: (minute offset, user, outcome, ip)"""
    base = pd.Timestamp("2026-08-03 10:00:00")
    return pd.DataFrame([
        dict(timestamp=base + pd.Timedelta(minutes=m), username=u, outcome=o, src_ip=ip)
        for m, u, o, ip in rows
    ])


def test_one_ip_many_accounts_flagged_once_per_account():
    logins = make_logins([(i, f"user{i}", "failure", "9.9.9.9") for i in range(6)])
    flags = detect_credential_stuffing(logins)
    assert len(flags) == 6 and set(flags.rule_name) == {"credential_stuffing"}
    assert set(flags.employee_id) == {f"user{i}" for i in range(6)}
    assert set(flags.severity) == {"Medium"}
    assert "9.9.9.9" in flags.evidence.iloc[0] and "6 accounts" in flags.evidence.iloc[0]


def test_few_accounts_not_flagged():
    logins = make_logins([(i, f"user{i}", "failure", "9.9.9.9") for i in range(4)])
    assert detect_credential_stuffing(logins).empty


def test_same_account_repeated_is_not_stuffing():
    # 20 failures but only one account: that is brute force, not stuffing
    logins = make_logins([(i, "a.b", "failure", "9.9.9.9") for i in range(20)])
    assert detect_credential_stuffing(logins).empty


def test_different_ips_do_not_add_up():
    logins = make_logins([(i, f"user{i}", "failure", f"9.9.9.{i}") for i in range(10)])
    assert detect_credential_stuffing(logins).empty


def test_accounts_spread_over_hours_not_flagged():
    logins = make_logins([(i * 60, f"user{i}", "failure", "9.9.9.9") for i in range(6)])
    assert detect_credential_stuffing(logins).empty


def test_success_from_same_ip_is_high():
    rows = [(i, f"user{i}", "failure", "9.9.9.9") for i in range(6)] + [(10, "user2", "success", "9.9.9.9")]
    flags = detect_credential_stuffing(make_logins(rows))
    assert set(flags.severity) == {"High"} and "later succeeded on 1" in flags.evidence.iloc[0]


def test_failures_without_ip_are_ignored():
    logins = make_logins([(i, f"user{i}", "failure", None) for i in range(8)])
    assert detect_credential_stuffing(logins).empty
