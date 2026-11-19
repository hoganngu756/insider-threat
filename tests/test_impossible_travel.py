import pandas as pd
import pytest

from src.detection.impossible_travel import detect_impossible_travel, distance_km, normalize_city

BASE = pd.Timestamp("2026-08-03 09:00:00")


def run(rows):
    """rows: (minute offset, city, country[, outcome])"""
    df = pd.DataFrame([
        dict(timestamp=BASE + pd.Timedelta(minutes=r[0]), username="a.b", city=r[1], country=r[2],
             outcome=r[3] if len(r) > 3 else "success")
        for r in rows
    ])
    return detect_impossible_travel(df)


def test_distance_dallas_houston():
    assert distance_km((32.78, -96.80), (29.76, -95.37)) == pytest.approx(360, abs=15)


@pytest.mark.parametrize("raw, expected", [("DALLAS, TX", "Dallas"), ("dallas", "Dallas"),
                                           ("Austin, TX", "Austin"), (None, None), (float("nan"), None)])
def test_normalize_city(raw, expected):
    assert normalize_city(raw) == expected


def test_international_jump_is_high():
    flags = run([(0, "Dallas, TX", "US"), (30, "Bucharest", "RO")])
    assert len(flags) == 1
    row = flags.iloc[0]
    assert (row.rule_name, row.severity, row.employee_id) == ("impossible_travel", "High", "a.b")
    assert row.timestamp == BASE + pd.Timedelta(minutes=30)
    assert "Dallas -> Bucharest" in row.evidence and "30 min" in row.evidence


def test_domestic_jump_is_medium():
    assert run([(0, "Denver", "US"), (10, "Chicago", "US")]).iloc[0].severity == "Medium"


def test_feasible_travel_not_flagged():
    # Dallas -> Houston (360 km) in 2 hours = 180 km/h
    assert run([(0, "Dallas", "US"), (120, "Houston", "US")]).empty


def test_long_flight_with_enough_time_not_flagged():
    # Dallas -> Bucharest (~9,300 km) after 20 hours = 465 km/h
    assert run([(0, "Dallas", "US"), (20 * 60, "Bucharest", "RO")]).empty


def test_same_city_different_spelling_not_flagged():
    assert run([(0, "Dallas, TX", "US"), (1, "DALLAS", "US")]).empty


def test_failed_logins_and_unknown_cities_ignored():
    assert run([(0, "Dallas", "US"), (5, "Bucharest", "RO", "failure")]).empty
    assert run([(0, "Dallas", "US"), (5, "Atlantis", "US")]).empty
    assert run([(0, "Dallas", "US"), (5, None, None)]).empty


def test_compares_consecutive_logins_only():
    # Dallas, Houston 2h later (fine), Houston -> Bucharest 10 min later (impossible)
    flags = run([(0, "Dallas", "US"), (120, "Houston", "US"), (130, "Bucharest", "RO")])
    assert len(flags) == 1 and "Houston -> Bucharest" in flags.iloc[0].evidence
