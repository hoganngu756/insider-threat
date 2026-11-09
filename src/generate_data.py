import numpy as np
import pandas as pd


def generate_logs(n_employees=50, days=30, seed=0) -> pd.DataFrame:
    """Normal behaviour + injected scenarios. Returns one row per event."""
    rng = np.random.default_rng(seed)
    # TODO: normal logins (office hours, usual device/location, small file transfers)
    # TODO: inject brute force (many failed logins in minutes)
    # TODO: inject new device/location login at odd hour
    # TODO: inject large data download / exfiltration
    cols = ["timestamp", "user", "event", "success", "device", "location", "bytes"]
    return pd.DataFrame(columns=cols)
