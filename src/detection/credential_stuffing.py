"""Step 3b: credential stuffing = ONE source IP failing logins across MANY accounts."""
from collections import Counter

import pandas as pd

from src.detection.brute_force import FLAG_COLUMNS


def _episodes(g, threshold, window):
    """g = one IP's failures sorted by time. Returns [(start, end), ...] of bursts."""
    times, users = g["timestamp"].tolist(), g["username"].tolist()
    win = pd.Timedelta(window)
    in_window = Counter()      # failures per user inside the trailing window
    left = 0
    episodes = []
    for right, t in enumerate(times):
        in_window[users[right]] += 1
        while t - times[left] > win:           # slide the window start forward
            in_window[users[left]] -= 1
            if in_window[users[left]] == 0:
                del in_window[users[left]]
            left += 1
        if len(in_window) >= threshold:        # distinct accounts failing in this window
            if episodes and t - episodes[-1][1] <= win:
                episodes[-1][1] = t            # same burst: extend it
            else:
                episodes.append([times[left], t])
    return episodes


def detect_credential_stuffing(logins, threshold=5, window="30min"):
    """Flag an IP that fails against >= `threshold` different accounts within `window`.

    Output has one row per targeted account (so it joins on employee_id + date).
    Severity: High if the IP later logs in successfully to one of those accounts, else Medium.
    """
    fails = logins[(logins["outcome"] == "failure") & logins["src_ip"].notna()]
    successes = logins[logins["outcome"] == "success"]
    rows = []
    for ip, g in fails.sort_values("timestamp").groupby("src_ip"):
        for start, end in _episodes(g, threshold, window):
            burst = g[(g["timestamp"] >= start) & (g["timestamp"] <= end)]
            accounts = burst["username"].nunique()
            minutes = max(1, round((end - start).total_seconds() / 60))

            # Did this IP get into any of the targeted accounts afterwards?
            hit = successes[(successes["src_ip"] == ip) & (successes["timestamp"] >= start)
                            & successes["username"].isin(burst["username"])]
            evidence = f"{ip} failed logins against {accounts} accounts in {minutes} min"
            if len(hit):
                evidence += f"; later succeeded on {hit['username'].nunique()} of them"
            severity = "High" if len(hit) else "Medium"

            # one row per targeted account, stamped with that account's first failure
            for user, first in burst.groupby("username")["timestamp"].min().items():
                rows.append([user, first.strftime("%Y-%m-%d"), first, "credential_stuffing", severity, evidence])
    return pd.DataFrame(rows, columns=FLAG_COLUMNS)


if __name__ == "__main__":
    import sys

    from src.detection.clean_columns import split_events
    from src.detection.load_logs import load_raw

    events, _ = load_raw(sys.argv[1] if len(sys.argv) > 1 else "data/mock_data.csv")
    logins, _, _ = split_events(events)
    flags = detect_credential_stuffing(logins)
    print(len(flags), "flags;", flags["employee_id"].nunique(), "accounts;", flags["evidence"].nunique(), "distinct evidence")
    print(flags["evidence"].value_counts().to_string())
    print(flags["severity"].value_counts().to_dict(), flags["date"].value_counts().to_dict())
    print(flags.head(5).to_string())
