"""Step 4: run every detector and write rule_flags.csv (the deliverable for teammates)."""
import pandas as pd

from src.detection.brute_force import FLAG_COLUMNS, detect_brute_force
from src.detection.clean_columns import split_events
from src.detection.credential_stuffing import detect_credential_stuffing
from src.detection.exfiltration import detect_bulk_download, detect_usb_exfiltration
from src.detection.impossible_travel import detect_impossible_travel
from src.detection.load_logs import load_raw
from src.detection.new_device import detect_new_device_foreign

VALID_RULES = {"brute_force", "credential_stuffing", "impossible_travel",
               "new_device_foreign_login", "bulk_download", "usb_exfiltration"}
VALID_SEVERITIES = {"Low", "Medium", "High"}


def combine(flag_frames):
    """Stack the detectors' outputs into one table sorted by employee, then time."""
    flags = pd.concat(flag_frames, ignore_index=True)[FLAG_COLUMNS]
    assert set(flags["rule_name"]) <= VALID_RULES, "unexpected rule_name"
    assert set(flags["severity"]) <= VALID_SEVERITIES, "unexpected severity"
    return flags.sort_values(["employee_id", "timestamp", "rule_name"]).reset_index(drop=True)


def build_rule_flags(path="data/mock_data.csv"):
    events, _ = load_raw(path)
    logins, files, _ = split_events(events)
    return combine([
        detect_brute_force(logins),
        detect_credential_stuffing(logins),
        detect_new_device_foreign(logins, files),
        detect_impossible_travel(logins),
        detect_bulk_download(files),
        detect_usb_exfiltration(files),
    ])


if __name__ == "__main__":
    import os

    flags = build_rule_flags()
    os.makedirs("data/processed", exist_ok=True)
    flags.to_csv("data/processed/rule_flags.csv", index=False)  # never touches data/mock_data.csv
    print(len(flags), "flags written to data/processed/rule_flags.csv")
    print(flags.groupby(["rule_name", "severity"]).size().to_string())
    print("\nemployees flagged:", flags["employee_id"].nunique())
    print(flags.groupby("employee_id").agg(flags=("rule_name", "size"), rules=("rule_name", lambda r: ", ".join(sorted(set(r)))),
                                           top=("severity", lambda s: "High" if "High" in set(s) else "Medium" if "Medium" in set(s) else "Low"))
          .sort_values("flags", ascending=False).head(12).to_string())
