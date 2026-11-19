"""Step 3c: successful login from a device the user never used before and/or an unusual country."""
import pandas as pd

from src.detection.brute_force import FLAG_COLUMNS


def detect_new_device_foreign(logins, files, baseline_days=28):
    """Compare each successful login with that user's own history (nothing is hard-coded).

    - History starts at the user's first event. The first `baseline_days` days only BUILD the
      history (people legitimately add a second device early on), so nothing is flagged then.
    - New device: first time the user is seen on this device (via a successful login or file event).
    - Foreign: login country is not one of the user's countries from the baseline period.
    Severity: both = High, foreign only = Medium, new device only = Low.
    """
    ok = logins[logins["outcome"] == "success"]
    # Device history = successful logins + file events (failed logins may come from an attacker's device)
    seen = pd.concat([ok[["username", "dev", "timestamp"]], files[["username", "dev", "timestamp"]]]).dropna()
    first_seen = seen.groupby(["username", "dev"])["timestamp"].min()
    user_start = seen.groupby("username")["timestamp"].min()

    rows = []
    for r in ok.itertuples():
        start = user_start.get(r.username)
        if start is None:
            continue
        baseline_end = start + pd.Timedelta(days=baseline_days)
        if r.timestamp < baseline_end:
            continue  # still learning this user's normal

        new_dev = pd.notna(r.dev) and first_seen[(r.username, r.dev)] == r.timestamp

        usual = set(ok.loc[(ok["username"] == r.username) & (ok["timestamp"] < baseline_end), "country"].dropna())
        foreign = pd.notna(r.country) and bool(usual) and r.country not in usual
        if not (new_dev or foreign):
            continue

        parts = []
        if new_dev:
            parts.append(f"new device {r.dev}")
        if foreign:
            parts.append(f"login from {r.country} (usual: {', '.join(sorted(usual))})")
        evidence = "successful login: " + "; ".join(parts)
        severity = "High" if (new_dev and foreign) else "Medium" if foreign else "Low"
        rows.append([r.username, r.timestamp.strftime("%Y-%m-%d"), r.timestamp,
                     "new_device_foreign_login", severity, evidence])
    return pd.DataFrame(rows, columns=FLAG_COLUMNS)


if __name__ == "__main__":
    import sys

    from src.detection.clean_columns import split_events
    from src.detection.load_logs import load_raw

    events, _ = load_raw(sys.argv[1] if len(sys.argv) > 1 else "data/mock_data.csv")
    logins, files, _ = split_events(events)
    flags = detect_new_device_foreign(logins, files)
    print(len(flags), "flags")
    print(flags.to_string())
