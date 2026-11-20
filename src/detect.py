import pandas as pd
from sklearn.ensemble import IsolationForest
from src.detection.exfiltration import EXTERNAL_ACTIONS

def build_features(logins: pd.DataFrame, files:pd.DataFrame) -> pd.DataFrame:
    """
    Build one feature row per user per day.
    """
    logins = logins.copy()
    files = files.copy()

    # --------------------------------------------------
    # 1. Prepare timestamps and dates and remove unusable rows
    # --------------------------------------------------

    # "2026-08-03 14:30:00 -> 2026-08-..."
    logins["timestamp"] = pd.to_datetime(
        logins["timestamp"],
        errors="coerce"
    )

    files["timestamp"] = pd.to_datetime(
        files["timestamp"],
        errors="coerce" #replace error with NaT
    )

    #Remove rows with missing timestamps or usernames
    logins = logins.dropna(
        subset=["timestamp", "username"]
    )

    files = files.dropna(
        subset=["timestamp", "username"]
    )

    # Create a daily grouping column
    #extract date portion of each timestamp and store in a new column date
    logins["date"] = logins["timestamp"].dt.strftime("%Y-%m-%d")
    files["date"] = files["timestamp"].dt.strftime("%Y-%m-%d")

     # --------------------------------------------------
    # 2. LOGIN FEATURES
    # --------------------------------------------------
    #total logins
    total_logins = (
    logins.groupby(["username", "date"])
    .size()
    .rename("total_logins")
    .reset_index()
)

    # Total failed logins per user per day
    failed_logins = (
        logins[logins["outcome"] == "failure"]
        .groupby(["username", "date"])
        .size()
        .rename("failed_logins")
        .reset_index()
    )

    # Off-hours logins
    #
    # Simple definition:
    # before 7 AM or at/after 7 PM
    #
    logins["hour"] = logins["timestamp"].dt.hour

    off_hours_logins = (
        logins[
            (logins["hour"] < 7)
            | (logins["hour"] >= 19)
        ]
        .groupby(["username", "date"])
        .size()
        .rename("off_hours_logins")
        .reset_index()
    )

    # --------------------------------------------------
    # 3. NEW DEVICE FEATURE
    # --------------------------------------------------

    #Unique devices and countries in a day
    unique_devices = (
    logins.groupby(["username", "date"])["dev"]
    .nunique()
    .rename("unique_devices")
    .reset_index()
)

    unique_countries = (
        logins.groupby(["username", "date"])["country"]
        .nunique()
        .rename("unique_countries")
        .reset_index())

    # --------------------------------------------------
    # 4. FILE / TRANSFER FEATURES
    # --------------------------------------------------

    # Make sure size_bytes is numeric
    files["size_bytes"] = pd.to_numeric(
        files["size_bytes"], errors="coerce"
    ).fillna(0)

    
    file_upload_actions = ["upload"]
    file_download_actions = ["download"]

    file_uploads = (
        files[files["action"].isin(file_upload_actions)]
        .groupby(["username", "date"])
        .size()
        .rename("file_uploads")
        .reset_index()
    )

    file_downloads = (
        files[files["action"].isin(file_download_actions)]
        .groupby(["username", "date"])
        .size()
        .rename("file_downloads")
        .reset_index()
    )

    bytes_uploaded = (
        files[files["action"].isin(file_upload_actions)]
        .groupby(["username", "date"])["size_bytes"]
        .sum()
        .rename("bytes_uploaded")
        .reset_index()
    )

    bytes_downloaded = (
        files[files["action"].isin(file_download_actions)]
        .groupby(["username", "date"])["size_bytes"]
        .sum()
        .rename("bytes_downloaded")
        .reset_index()
    )

    external_transfers = (
    files[files["action"].isin(EXTERNAL_ACTIONS)]
    .groupby(["username", "date"])
    .size()
    .rename("external_transfers")
    .reset_index()
    )

    bytes_external = (
    files[files["action"].isin(EXTERNAL_ACTIONS)]
    .groupby(["username", "date"])["size_bytes"]
    .sum()
    .rename("bytes_external")
    .reset_index()
    )


    files["hour"] = files["timestamp"].dt.hour

    files["is_off_hours"] = (
        (files["hour"] < 7) | (files["hour"] >= 19)
    )

    off_hours_file_events = (
        files[files["is_off_hours"]]
        .groupby(["username", "date"])
        .size()
        .rename("off_hours_file_events")
        .reset_index()
    )

     # --------------------------------------------------
    # 5. CREATE THE BASE USER-DAY TABLE
    # --------------------------------------------------

    # Get every user/day combination appearing in either
    # login or file data
    login_days = logins[["username", "date"]].drop_duplicates()
    file_days = files[["username", "date"]].drop_duplicates()

    features = (
        pd.concat([login_days, file_days], ignore_index=True)
        .drop_duplicates()
    )

    # --------------------------------------------------
    # 6. MERGE ALL FEATURES
    # --------------------------------------------------

    feature_tables = [
        total_logins,
        failed_logins,
        off_hours_logins,
        unique_devices,
        unique_countries,
        file_uploads,
        file_downloads,
        bytes_uploaded,
        bytes_downloaded,
        external_transfers,
        bytes_external,
        off_hours_file_events
    ]

    for table in feature_tables:
        features = features.merge(
            table,
            on=["username", "date"],
            how="left"
        )

    # Missing feature values mean zero recorded activity
    feature_columns = [
        "total_logins",
        "failed_logins",
        "off_hours_logins",
        "unique_devices",
        "unique_countries",
        "file_uploads",
        "file_downloads",
        "bytes_uploaded",
        "bytes_downloaded",
        "external_transfers",
        "bytes_external",
        "off_hours_file_events"
    ]

    features[feature_columns] = features[feature_columns].fillna(0)

    # Rename username to employee_id for consistency
    features = features.rename(
        columns={"username": "employee_id"}
    )

    features["timestamp"] = pd.to_datetime(features["date"])

    return (
        features.drop(columns=["date"])
        .sort_values(["employee_id", "timestamp"])
        .reset_index(drop=True)
    )






def score(features: pd.DataFrame, contamination: float = 0.02) -> pd.DataFrame:
    """
    Calculate anomaly scores and flags using Isolation Forest.

    anomaly_score:
        Higher values indicate more unusual employee-day activity.

    anomaly_flag:
        True indicates that an observation was identified as unusual
        by Isolation Forest. It does not necessarily indicate malicious
        activity.

    contamination:
        Expected proportion of anomalies in the dataset.
        Defaults to 0.02 (2%).
    """
    
    if features.empty:
        features["anomaly_score"] = 0.0
        features["anomaly_flag"] = False
        return features

    features = features.copy()

    feature_columns = [
        "total_logins",
        "failed_logins",
        "off_hours_logins",
        "unique_devices",
        "unique_countries",
        "file_uploads",
        "file_downloads",
        "bytes_uploaded",
        "bytes_downloaded",
        "external_transfers",
        "bytes_external",
        "off_hours_file_events"
        ]

    if len(features) < 2:
        features["anomaly_score"] = 0.0
        features["anomaly_flag"] = False
        return features


    X = features[feature_columns].fillna(0)

    model = IsolationForest(
        contamination=contamination,
        random_state=0
    )

    model.fit(X)

    features["anomaly_score"] = (
        -model.score_samples(X)
    )

    features["anomaly_flag"] = (
        model.predict(X) == -1
    )

    return features


if __name__ == "__main__":
    from src.detection.load_logs import load_raw
    from src.detection.clean_columns import split_events
    import os

    # Load and clean the actual dataset
    events, report = load_raw("data/mock_data.csv")
    logins, files, report = split_events(events)

    # Run ML pipeline
    features = build_features(logins, files)
    ml_output = score(features)

    # Save results
    os.makedirs("data/processed", exist_ok=True)
    ml_output.to_csv(
        "data/processed/ml_output.csv",
        index=False
    )

    print("Employee-day observations:", len(features))
    print("Anomalies detected:", ml_output["anomaly_flag"].sum())
    print("ML output saved to data/processed/ml_output.csv")