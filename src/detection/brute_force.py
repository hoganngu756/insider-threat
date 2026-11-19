"""Step 3a: brute force = many failed logins on ONE account in a short time."""
import pandas as pd

FLAG_COLUMNS = ["employee_id", "date", "timestamp", "rule_name", "severity", "evidence"]


def _episodes(times, threshold, window):
    """Group one user's failure times into bursts. Returns [(start, end, count), ...]."""
    s = pd.Series(1, index=pd.DatetimeIndex(times)).sort_index()
    in_window = s.rolling(window).sum()           # failures in the trailing window, per event
    hot = in_window[in_window >= threshold]       # events where the threshold is reached
    episodes = []
    for t, n in hot.items():
        start = t - pd.Timedelta(window)          # trailing window covers (start, t]
        if episodes and t - episodes[-1][1] <= pd.Timedelta(window):
            episodes[-1][1] = t                   # still the same burst: extend it
        else:
            episodes.append([start, t])
    # Count the real failures inside each burst and snap start to the first one.
    out = []
    for start, end in episodes:
        inside = s.index[(s.index > start) & (s.index <= end)]
        out.append((inside.min(), inside.max(), len(inside)))
    return out


def detect_brute_force(logins, threshold=10, window="30min", success_within="60min"):
    """Flag users with >= `threshold` failed logins inside `window`.

    Severity: High if the burst is large (>= 50 failures) or a successful login follows it
    (the attacker may have got in); otherwise Medium.
    """
    fails = logins[logins["outcome"] == "failure"]
    successes = logins[logins["outcome"] == "success"]
    rows = []
    for user, g in fails.groupby("username"):
        for start, end, n in _episodes(g["timestamp"], threshold, window):
            burst = g[(g["timestamp"] >= start) & (g["timestamp"] <= end)]
            ips = burst["src_ip"].dropna()
            top_ip = ips.value_counts().index[0] if len(ips) else "unknown IP"
            minutes = max(1, round((end - start).total_seconds() / 60))

            after = successes[(successes["username"] == user) & (successes["timestamp"] > end)
                              & (successes["timestamp"] <= end + pd.Timedelta(success_within))]
            evidence = f"{n} failed logins in {minutes} min from {top_ip}"
            if ips.nunique() > 1:
                evidence += f" (+{ips.nunique() - 1} other IPs)"
            if len(after):
                evidence += "; successful login followed"

            severity = "High" if (n >= 50 or len(after)) else "Medium"
            rows.append([user, start.strftime("%Y-%m-%d"), start, "brute_force", severity, evidence])
    return pd.DataFrame(rows, columns=FLAG_COLUMNS)


if __name__ == "__main__":
    import sys

    from src.detection.clean_columns import split_events
    from src.detection.load_logs import load_raw

    events, _ = load_raw(sys.argv[1] if len(sys.argv) > 1 else "data/mock_data.csv")
    logins, _, _ = split_events(events)
    flags = detect_brute_force(logins)
    print(len(flags), "flags")
    print(flags.to_string())
