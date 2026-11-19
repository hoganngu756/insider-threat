import pandas as pd

from src.detection.exfiltration import detect_bulk_download, detect_usb_exfiltration

BASE = pd.Timestamp("2026-08-03 10:00:00")


def files(rows):
    """rows: (minute offset, action, size_bytes, sensitivity[, user])"""
    return pd.DataFrame([
        dict(timestamp=BASE + pd.Timedelta(minutes=r[0]), username=r[4] if len(r) > 4 else "a.b",
             action=r[1], size_bytes=r[2], sensitivity=r[3])
        for r in rows
    ])


def test_bulk_download_flagged_with_evidence():
    f = files([(i, "download", 1_000_000, "confidential") for i in range(25)])
    flags = detect_bulk_download(f)
    assert len(flags) == 1
    row = flags.iloc[0]
    assert (row.employee_id, row.date, row.rule_name, row.severity) == ("a.b", "2026-08-03", "bulk_download", "Medium")
    assert row.timestamp == BASE
    assert "25 downloads (25 MB, 25 sensitive)" in row.evidence


def test_bulk_download_high_when_very_large():
    f = files([(i, "download", 1000, "public") for i in range(100)])
    assert detect_bulk_download(f).iloc[0].severity == "High"


def test_normal_downloads_not_flagged():
    assert detect_bulk_download(files([(i, "download", 1000, "public") for i in range(7)])).empty


def test_bulk_download_counts_only_downloads_per_user_and_day():
    rows = [(i, "read", 1000, "public") for i in range(30)] + \
           [(i, "download", 1000, "public", "a.b") for i in range(10)] + \
           [(i, "download", 1000, "public", "c.d") for i in range(10)]
    assert detect_bulk_download(files(rows)).empty


def test_bulk_download_splits_by_day():
    rows = [(i, "download", 1000, "public") for i in range(12)] + \
           [(24 * 60 + i, "download", 1000, "public") for i in range(12)]
    assert detect_bulk_download(files(rows)).empty


def test_bulk_download_handles_missing_sizes():
    f = files([(i, "download", None, "public") for i in range(20)])
    assert "0 MB" in detect_bulk_download(f).iloc[0].evidence


def test_usb_large_copy_flagged():
    f = files([(0, "usb_copy", 300_000_000, "confidential"), (5, "usb_copy", 200_000_000, "internal")])
    flags = detect_usb_exfiltration(f)
    row = flags.iloc[0]
    assert (row.rule_name, row.severity) == ("usb_exfiltration", "Medium")
    assert "500 MB" in row.evidence and "2 usb_copy" in row.evidence


def test_usb_high_for_restricted_or_over_1gb():
    assert detect_usb_exfiltration(files([(0, "usb_copy", 150_000_000, "restricted")])).iloc[0].severity == "High"
    assert detect_usb_exfiltration(files([(0, "cloud_upload", 2_000_000_000, "internal")])).iloc[0].severity == "High"


def test_small_external_transfers_not_flagged():
    f = files([(0, "usb_copy", 3_000_000, "secret"), (1, "email_external", 2_000_000, "internal")])
    assert detect_usb_exfiltration(f).empty


def test_internal_actions_ignored_and_after_hours_counted():
    f = files([(0, "upload", 500_000_000, "internal"), (11 * 60, "usb_copy", 200_000_000, "internal")])
    flags = detect_usb_exfiltration(f)  # 21:00 copy is after hours
    assert len(flags) == 1 and "1 after hours" in flags.iloc[0].evidence
