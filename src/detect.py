import pandas as pd
from sklearn.ensemble import IsolationForest


def build_features(events: pd.DataFrame) -> pd.DataFrame:
    """Per user per day: failed logins, off-hours logins, new devices, bytes transferred."""
    # TODO
    return pd.DataFrame()


def score(features: pd.DataFrame) -> pd.DataFrame:
    """Isolation Forest anomaly score + rule-based indicators -> risk score and level."""
    if features.empty:
        return features
    num = features.select_dtypes("number")
    model = IsolationForest(contamination=0.02, random_state=0).fit(num)
    features["anomaly_score"] = -model.score_samples(num)
    # TODO: combine with failed_logins, off_hours, new_device, bytes into risk_score
    features["risk_score"] = features["anomaly_score"]
    return features
