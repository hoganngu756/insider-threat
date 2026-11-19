"""Step 2: split events into login / file events and clean the remaining columns."""
import ipaddress
import re

import pandas as pd

from src.detection.load_logs import is_null_like

LOGIN_EVTS = {"login", "logon", "auth"}
FILE_EVTS = {"file", "file_access"}

COUNTRY = {
    "us": "US", "u.s.": "US", "usa": "US", "united states": "US",
    "ro": "RO", "rou": "RO", "romania": "RO",
    "cn": "CN", "chn": "CN", "china": "CN",
    "ng": "NG", "nga": "NG", "nigeria": "NG",
}
OUTCOME = {
    "success": "success", "true": "success", "1": "success", "ok": "success", "y": "success",
    "fail": "failure", "failure": "failure", "failed": "failure", "false": "failure",
    "0": "failure", "n": "failure", "denied": "failure",
}
ACTION = {
    "read": "read", "open": "open",
    "edit": "modify", "modify": "modify",
    "download": "download", "dl": "download", "dowload": "download",  # "dowload" is a typo in the data
    "upload": "upload",
    "email_external": "email_external", "sent_external": "email_external", "email_ext": "email_external",
    "usb_copy": "usb_copy", "copy_to_usb": "usb_copy",
    "cloud_upload": "cloud_upload",
}
SENSITIVITY = {
    "public": "public", "pub": "public",
    "internal": "internal", "int": "internal",
    "confidential": "confidential", "conf": "confidential", "conf.": "confidential",
    "restricted": "restricted",
    "secret": "secret",  # kept as its own level; not merged into restricted
}
SIZE_UNITS = {"kb": 1024, "mb": 1024 ** 2}  # assumes binary units (1 KB = 1024 bytes)


def _key(v):
    """Lowercase, trimmed, spaces -> underscores. None if null-like."""
    if is_null_like(v):
        return None
    return re.sub(r"\s+", "_", str(v).strip().lower())


def _lookup(v, mapping):
    """Map a raw label to its canonical form; unknown labels pass through unchanged (lowercased)."""
    k = _key(v)
    return None if k is None else mapping.get(k, k)


def clean_country(v):
    # Country names have spaces ("United States"), so look up with spaces, not underscores.
    if is_null_like(v):
        return None
    k = str(v).strip().lower()
    return COUNTRY.get(k, k.upper())


def clean_outcome(v):
    return _lookup(v, OUTCOME)


def clean_action(v):
    return _lookup(v, ACTION)


def clean_sensitivity(v):
    return _lookup(v, SENSITIVITY)


def clean_device(v):
    k = _key(v)
    return k


def clean_text(v):
    """Lowercase/trim a free-text label (dest, failure reason); null-like -> None."""
    return _key(v)


def clean_ip(v):
    """Strip a ':port', validate; invalid or null-like -> None."""
    if is_null_like(v):
        return None
    s = str(v).strip().split(":")[0]
    try:
        return str(ipaddress.ip_address(s))
    except ValueError:
        return None


def clean_size(v):
    """Parse '98 KB', '1.20 MB', '1,234,567', '4096' to bytes; N/A, blank, negatives -> None."""
    if is_null_like(v):
        return None
    s = str(v).strip().lower().replace(",", "")
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*(kb|mb)?", s)
    if not m:
        return None
    n = float(m.group(1)) * SIZE_UNITS.get(m.group(2), 1)
    return int(round(n)) if n >= 0 else None  # "-1" is a placeholder, not a real size


def clean_path(v):
    """Use forward slashes so '\\share\\x' and '/share/x' match."""
    return None if is_null_like(v) else str(v).strip().replace("\\", "/")


def split_events(events):
    """Return (logins, files, report). Rows with an unknown evt go to neither and are counted."""
    evt = events["evt"].map(_key)
    report = {
        "total": len(events),
        "login_rows": int(evt.isin(LOGIN_EVTS).sum()),
        "file_rows": int(evt.isin(FILE_EVTS).sum()),
        "unknown_evt_rows": int((~evt.isin(LOGIN_EVTS | FILE_EVTS)).sum()),
    }

    lg = events[evt.isin(LOGIN_EVTS)]
    logins = pd.DataFrame({
        "timestamp": lg["timestamp"], "username": lg["username"],
        "dev": lg["dev"].map(clean_device),
        "src_ip": lg["src_ip"].map(clean_ip),
        "city": lg["city"].map(lambda v: None if is_null_like(v) else str(v).strip()),
        "country": lg["country"].map(clean_country),
        "outcome": lg["outcome"].map(clean_outcome),
        "failure_reason": lg["notes"].map(clean_text),  # for logins, notes = failure reason
    }).reset_index(drop=True)

    fl = events[evt.isin(FILE_EVTS)]
    files = pd.DataFrame({
        "timestamp": fl["timestamp"], "username": fl["username"],
        "dev": fl["dev"].map(clean_device),
        "obj": fl["obj"].map(clean_path),
        "action": fl["action"].map(clean_action),
        "size_bytes": pd.array(fl["size"].map(clean_size), dtype="Int64"),
        "dest": fl["dest"].map(clean_text),
        "sensitivity": fl["notes"].map(clean_sensitivity),  # for files, notes = sensitivity
    }).reset_index(drop=True)
    return logins, files, report


if __name__ == "__main__":
    import sys

    from src.detection.load_logs import load_raw

    events, _ = load_raw(sys.argv[1] if len(sys.argv) > 1 else "data/mock_data.csv")
    logins, files, rep = split_events(events)
    print(rep)
    for name, df in [("logins", logins), ("files", files)]:
        print(f"\n== {name}: {len(df)} rows; nulls per column ==")
        print(df.isna().sum().to_dict())
        for c in df.columns:
            if c in ("country", "outcome", "failure_reason", "action", "sensitivity", "dest"):
                print(c, df[c].value_counts(dropna=False).to_dict())
    print("\nsize_bytes describe:\n", files["size_bytes"].describe().to_string())
    print("\nraw vs cleaned sample (files):")
    print(events.loc[events["evt"].str.lower().str.contains("file", na=False), ["size", "action", "notes"]]
          .head(0).to_string())
    print(logins.sample(8, random_state=1).to_string())
    print(files.sample(8, random_state=1).to_string())
