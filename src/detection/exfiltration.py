"""Step 3e: data exfiltration = bulk downloads, and large copies out via USB / personal cloud / external email."""
import pandas as pd

from src.detection.brute_force import FLAG_COLUMNS

SENSITIVE = ["confidential", "restricted", "secret"]
EXTERNAL_ACTIONS = ["usb_copy", "cloud_upload", "email_external"]  # data leaving the company
MB = 1_000_000


def _per_user_day(df):
    """Yield ((user, 'YYYY-MM-DD'), rows) for every user-day in a file-events frame."""
    return df.groupby(["username", df["timestamp"].dt.strftime("%Y-%m-%d")])


def detect_bulk_download(files, min_downloads=20, high_downloads=100):
    """Flag a user-day with >= `min_downloads` downloads (normal users do 1-7 per day).
    High if >= `high_downloads`, otherwise Medium."""
    rows = []
    for (user, date), g in _per_user_day(files[files["action"] == "download"]):
        n = len(g)
        if n < min_downloads:
            continue
        mb = g["size_bytes"].sum() / MB            # unknown sizes (null) count as 0
        sensitive = int(g["sensitivity"].isin(SENSITIVE).sum())
        hours = (g["timestamp"].max() - g["timestamp"].min()).total_seconds() / 3600
        evidence = f"{n} downloads ({mb:,.0f} MB, {sensitive} sensitive) in {hours:.1f} h"
        rows.append([user, date, g["timestamp"].min(), "bulk_download",
                     "High" if n >= high_downloads else "Medium", evidence])
    return pd.DataFrame(rows, columns=FLAG_COLUMNS)


def detect_usb_exfiltration(files, min_mb=100, high_mb=1000):
    """Flag a user-day that sends >= `min_mb` MB out by USB, personal cloud or external email
    (normal users send under 4 MB a day). High if >= `high_mb` MB or any restricted/secret file."""
    rows = []
    for (user, date), g in _per_user_day(files[files["action"].isin(EXTERNAL_ACTIONS)]):
        mb = g["size_bytes"].sum() / MB
        if mb < min_mb:
            continue
        channels = ", ".join(f"{n} {a}" for a, n in g["action"].value_counts().items())
        hours = g["timestamp"].dt.hour
        after_hours = int(((hours < 7) | (hours >= 19)).sum())
        top_secret = int(g["sensitivity"].isin(["restricted", "secret"]).sum())
        evidence = f"{mb:,.0f} MB sent out ({channels}); {top_secret} restricted/secret; {after_hours} after hours"
        rows.append([user, date, g["timestamp"].min(), "usb_exfiltration",
                     "High" if (mb >= high_mb or top_secret) else "Medium", evidence])
    return pd.DataFrame(rows, columns=FLAG_COLUMNS)


if __name__ == "__main__":
    import sys

    from src.detection.clean_columns import split_events
    from src.detection.load_logs import load_raw

    events, _ = load_raw(sys.argv[1] if len(sys.argv) > 1 else "data/mock_data.csv")
    _, files, _ = split_events(events)
    for name, flags in [("bulk_download", detect_bulk_download(files)),
                        ("usb_exfiltration", detect_usb_exfiltration(files))]:
        print(f"\n{name}: {len(flags)} flags")
        print(flags.drop(columns="rule_name").to_string())
