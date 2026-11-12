"""Step 1: load the raw SIEM export and normalize timestamp + username."""
import re

import pandas as pd

LOCAL_TZ = "America/Chicago"
NULL_LIKES = {"", "n/a", "na", "null", "none", "-", "unknown", "nan"}

# Naive (local-time) formats we expect. Day/month order is assumed US style (month first).
NAIVE_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %I:%M:%S %p",
    "%m/%d/%Y %H:%M",
    "%d-%b-%Y %H:%M:%S",
    "%d-%b-%Y %I:%M:%S %p",
    "%b %d, %Y %I:%M %p",
    "%b %d %Y %H:%M:%S",
]


def is_null_like(s) -> bool:
    return s is None or (not isinstance(s, str) and pd.isna(s)) or str(s).strip().lower() in NULL_LIKES


def parse_ts(s):
    """Return a naive America/Chicago pandas Timestamp, or None if unparseable."""
    if is_null_like(s):
        return None
    s = str(s).strip()

    # Epoch seconds (e.g. "1785000000" or "1785000000.5") are UTC.
    if re.fullmatch(r"\d{9,11}(\.\d+)?", s):
        return pd.Timestamp(float(s), unit="s", tz="UTC").tz_convert(LOCAL_TZ).tz_localize(None)

    # ISO with "Z" or an offset (+00:00, -05:00, +0000) is UTC-aware -> convert to Chicago.
    if re.search(r"(Z|[+-]\d{2}:?\d{2})$", s) and re.match(r"\d{4}-\d{2}-\d{2}[T ]", s):
        try:
            return pd.Timestamp(s).tz_convert(LOCAL_TZ).tz_localize(None)
        except (ValueError, TypeError):
            return None

    # Everything else is already local time: try each known format.
    for fmt in NAIVE_FORMATS:
        try:
            return pd.to_datetime(s, format=fmt)
        except (ValueError, TypeError):
            continue
    return None  # unknown format: caller reports these instead of guessing


def clean_user(u):
    """Normalize to lowercase first.last; None for blank/null-like values."""
    if is_null_like(u):
        return None
    u = str(u).strip().lower()
    u = re.sub(r"^corp\\", "", u)                  # CORP\first.last
    u = re.sub(r"@corp\.example\.com$", "", u)     # first.last@corp.example.com
    u = re.sub(r"\s+", ".", u.strip())             # "first last" -> "first.last"
    return u or None


def load_raw(path):
    """Load, clean, and sort the raw file. Returns (events, report dict)."""
    report = {}
    df = pd.read_csv(path, comment="#", on_bad_lines="skip", dtype=str,
                     keep_default_na=False, engine="python")
    report["rows_loaded"] = len(df)

    df = df[df["ts"].str.strip().str.lower() != "ts"]  # repeated header rows
    report["after_header_rows"] = len(df)

    df = df.drop_duplicates()
    report["after_duplicates"] = len(df)

    df = df.copy()
    df["timestamp"] = df["ts"].map(parse_ts)
    df["username"] = df["user"].map(clean_user)
    report["bad_timestamps"] = int(df["timestamp"].isna().sum())
    report["bad_usernames"] = int(df["username"].isna().sum())
    report["bad_ts_samples"] = df.loc[df["timestamp"].isna(), "ts"].head(10).tolist()

    df = df.dropna(subset=["timestamp", "username"])
    report["after_missing_ts_or_user"] = len(df)

    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    return df, report


if __name__ == "__main__":
    import sys

    events, rep = load_raw(sys.argv[1] if len(sys.argv) > 1 else "data/mock_data.csv")
    for k, v in rep.items():
        print(f"{k}: {v}")
    print("min/max timestamp:", events["timestamp"].min(), events["timestamp"].max())
    print("distinct usernames:", events["username"].nunique())
    print(events.sample(12, random_state=1)[["ts", "timestamp", "user", "username"]].to_string())
