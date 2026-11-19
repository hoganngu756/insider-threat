import pandas as pd
import pytest

from src.detection.clean_columns import (clean_action, clean_country, clean_device, clean_ip,
                                         clean_outcome, clean_path, clean_sensitivity,
                                         clean_size, split_events)

NULLS = ["", "N/A", "n/a", "NULL", "null", "-", "unknown", "None", None, float("nan")]


@pytest.mark.parametrize("raw, expected", [
    ("us", "US"), ("US", "US"), ("U.S.", "US"), ("USA", "US"), ("United States", "US"),
    ("ROU", "RO"), ("Romania", "RO"), ("RO", "RO"),
    ("CHN", "CN"), ("China", "CN"), ("CN", "CN"),
    ("NG", "NG"), ("NGA", "NG"), ("Nigeria", "NG"),
])
def test_clean_country(raw, expected):
    assert clean_country(raw) == expected


@pytest.mark.parametrize("raw", ["Success", "true", "1", "OK", "success", "SUCCESS", "Y"])
def test_outcome_success(raw):
    assert clean_outcome(raw) == "success"


@pytest.mark.parametrize("raw", ["fail", "Failure", "N", "FAILED", "0", "denied", "failed", "false"])
def test_outcome_failure(raw):
    assert clean_outcome(raw) == "failure"


@pytest.mark.parametrize("raw, expected", [
    ("Read", "read"), ("READ", "read"), ("open", "open"), ("Edit", "modify"), ("modify", "modify"),
    ("dl", "download"), ("DOWNLOAD", "download"), ("Download ", "download"), ("dowload", "download"),
    ("Upload", "upload"), ("sent external", "email_external"), ("email_ext", "email_external"),
    ("USB copy", "usb_copy"), ("copy_to_usb", "usb_copy"), ("cloud upload", "cloud_upload"),
])
def test_clean_action(raw, expected):
    assert clean_action(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("int", "internal"), ("Internal", "internal"), ("CONF", "confidential"), ("conf.", "confidential"),
    ("Confidential", "confidential"), ("PUB", "public"), ("RESTRICTED", "restricted"), ("secret", "secret"),
])
def test_clean_sensitivity(raw, expected):
    assert clean_sensitivity(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("98 KB", 98 * 1024), ("1.20 MB", round(1.2 * 1024 * 1024)), ("1,234,567", 1234567),
    ("4096", 4096), ("-1", None), ("N/A", None), ("", None), ("garbage", None),
])
def test_clean_size(raw, expected):
    assert clean_size(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("73.33.19.35", "73.33.19.35"), ("203.0.113.5:443", "203.0.113.5"),
    ("999.1.1.1", None), ("unknown", None), ("-", None), ("", None), ("1.2.3", None),
])
def test_clean_ip(raw, expected):
    assert clean_ip(raw) == expected


def test_clean_device_and_path():
    assert clean_device("DEV-E105-L ") == "dev-e105-l"
    assert clean_path("\\share\\hr\\doc.pdf") == "/share/hr/doc.pdf"


@pytest.mark.parametrize("fn", [clean_country, clean_outcome, clean_action, clean_sensitivity,
                                clean_size, clean_ip, clean_device, clean_path])
def test_nulls_become_none(fn):
    assert all(fn(v) is None for v in NULLS)


def test_split_events_routes_and_keeps_every_row():
    base = dict(dev="DEV-1", src_ip="", city="", country="", outcome="", obj="", action="",
                size="", dest="", notes="")
    rows = [
        dict(base, evt="LOGON", outcome="Y", src_ip="1.2.3.4:22", country="USA", notes="bad password"),
        dict(base, evt="auth", outcome="denied"),
        dict(base, evt="file_access", obj="\\a\\b", action="dl", size="2 KB", notes="conf."),
        dict(base, evt="weird"),
    ]
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.Timestamp("2026-08-03")
    df["username"] = "a.b"
    logins, files, rep = split_events(df)
    assert (len(logins), len(files), rep["unknown_evt_rows"]) == (2, 1, 1)
    assert logins.loc[0, ["src_ip", "country", "outcome", "failure_reason"]].tolist() == \
        ["1.2.3.4", "US", "success", "bad_password"]
    assert files.loc[0, ["obj", "action", "size_bytes", "sensitivity"]].tolist() == \
        ["/a/b", "download", 2048, "confidential"]
