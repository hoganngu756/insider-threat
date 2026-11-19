import pandas as pd
import pytest

from src.detection.load_logs import clean_user, parse_ts

# Chicago is UTC-5 in summer (CDT) and UTC-6 in winter (CST).


@pytest.mark.parametrize("raw, expected", [
    ("1785000000", "2026-07-25 12:20:00"),                # epoch = 17:20 UTC -> 12:20 CDT
    ("2026-08-03T14:00:00Z", "2026-08-03 09:00:00"),      # ISO Z, UTC
    ("2026-08-03T14:00:00+00:00", "2026-08-03 09:00:00"), # ISO offset, UTC
    ("2026-08-03T10:00:00-05:00", "2026-08-03 10:00:00"), # ISO non-UTC offset
    ("2026-11-02T14:00:00Z", "2026-11-02 08:00:00"),      # after DST ends: UTC-6
    ("2026-08-03 09:15:30", "2026-08-03 09:15:30"),       # naive = local
    ("08/03/2026 09:15:30", "2026-08-03 09:15:30"),
    ("08/03/2026 09:15:30 PM", "2026-08-03 21:15:30"),
    ("03-Aug-2026 09:15:30", "2026-08-03 09:15:30"),
    ("03-Aug-2026 09:14:39 AM", "2026-08-03 09:14:39"),
    ("09/24/2026 09:50", "2026-09-24 09:50:00"),
    ("Aug 03, 2026 09:15 AM", "2026-08-03 09:15:00"),
    ("  2026-08-03 09:15:30  ", "2026-08-03 09:15:30"),   # whitespace
])
def test_parse_ts_formats(raw, expected):
    assert parse_ts(raw) == pd.Timestamp(expected)


@pytest.mark.parametrize("raw", ["", "N/A", "NULL", "-", "unknown", "garbage", None])
def test_parse_ts_bad_values_return_none(raw):
    assert parse_ts(raw) is None


@pytest.mark.parametrize("raw", [
    "john.smith", "John Smith", "john.smith@corp.example.com",
    "CORP\\john.smith", "JOHN.SMITH", "  john.smith  ", "CORP\\John Smith",
    "John.Smith@CORP.EXAMPLE.COM",
])
def test_clean_user_variants(raw):
    assert clean_user(raw) == "john.smith"


@pytest.mark.parametrize("raw", ["", "   ", "N/A", "NULL", "-", "unknown", None])
def test_clean_user_blank_becomes_none(raw):
    assert clean_user(raw) is None
