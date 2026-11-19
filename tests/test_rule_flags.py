import pandas as pd
import pytest

from src.detection.rule_flags import FLAG_COLUMNS, combine


def frame(user, rule, severity="High", ts="2026-08-03 10:00:00"):
    t = pd.Timestamp(ts)
    return pd.DataFrame([[user, t.strftime("%Y-%m-%d"), t, rule, severity, "x"]], columns=FLAG_COLUMNS)


def test_combine_has_exact_columns_and_is_sorted():
    out = combine([frame("b.b", "brute_force"), frame("a.a", "bulk_download"), pd.DataFrame(columns=FLAG_COLUMNS)])
    assert list(out.columns) == ["employee_id", "date", "timestamp", "rule_name", "severity", "evidence"]
    assert list(out["employee_id"]) == ["a.a", "b.b"]


def test_combine_rejects_unknown_rule_or_severity():
    with pytest.raises(AssertionError):
        combine([frame("a.a", "made_up_rule")])
    with pytest.raises(AssertionError):
        combine([frame("a.a", "brute_force", severity="Critical")])
