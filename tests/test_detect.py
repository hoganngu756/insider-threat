import pandas as pd
from src.detect import build_features, score

#BASIC TEST

# Create sample login data
logins = pd.DataFrame({
    "username": [
        "alice.smith",
        "alice.smith",
        "bob.jones"
    ],
    "timestamp": [
        "2026-08-03 08:00:00",
        "2026-08-03 20:00:00",
        "2026-08-03 02:00:00"
    ],
    "outcome": [
        "success",
        "failure",
        "success"
    ],
    "dev": [
        "laptop_01",
        "laptop_01",
        "phone_01"
    ],
    "country": [
        "United States",
        "United States",
        "Canada"
    ]
})

# Create sample file data
files = pd.DataFrame({
    "username": [
        "alice.smith",
        "alice.smith",
        "bob.jones"
    ],
    "timestamp": [
        "2026-08-03 10:00:00",
        "2026-08-03 11:00:00",
        "2026-08-03 12:00:00"
    ],
    "action": [
        "upload",
        "download",
        "cloud_upload"
    ],
    "size_bytes": [
        100,
        50,
        200
    ]
})

# Run feature engineering
features = build_features(logins, files)

print(features.to_string(index=False))

ml_output = score(features)
print(ml_output.to_string(index=False))


#TEST WITH MOCK DATA
from src.detection.load_logs import load_raw
from src.detection.clean_columns import split_events

events, report = load_raw("data/mock_data.csv")

# Clean and separate login/file events
logins, files, report = split_events(events)

# Build features
features = build_features(logins, files)

# Run Isolation Forest
ml_output = score(features)

# Save results
ml_output.to_csv(
    "data/processed/ml_output.csv",
    index=False
)

# Print basic information
print("Number of login events:", len(logins))
print("Number of file events:", len(files))
print("Number of employee-day observations:", len(features))

print(
    "Number of anomalies:",
    ml_output["anomaly_flag"].sum()
)

print(ml_output.to_string(index=False))