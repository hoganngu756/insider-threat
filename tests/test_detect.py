import pandas as pd
from src.detect import build_features, score


def make_sample_data():
    """Create small login and file datasets for testing."""

    logins = pd.DataFrame({
        "username": [
            "alice.smith",
            "alice.smith",
            "bob.jones",
            "carol.lee"
        ],
        "timestamp": [
            "2026-08-03 08:00:00",
            "2026-08-03 20:00:00",
            "2026-08-03 02:00:00",
            "2026-08-03 10:00:00"
        ],
        "outcome": [
            "success",
            "failure",
            "success",
            "success"
        ],
        "dev": [
            "laptop_01",
            "laptop_01",
            "phone_01",
            "laptop_03"
        ],
        "country": [
            "United States",
            "United States",
            "Canada",
            "United States"
        ]
    })

    files = pd.DataFrame({
        "username": [
            "alice.smith",
            "bob.jones",
            "carol.lee"
        ],
        "timestamp": [
            "2026-08-03 10:00:00",
            "2026-08-03 12:00:00",
            "2026-08-03 13:00:00"
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

    return logins, files


def test_alice_login_features():
    """Check that daily features for alice are calculated correctly."""

    logins, files = make_sample_data()

    features = build_features(logins, files)

    # Check Alice's calculated features
    alice = features[
        features["employee_id"] == "alice.smith"
    ].iloc[0]

    #test alice login features
    assert alice["total_logins"] == 2
    assert alice["failed_logins"] == 1
    assert alice["off_hours_logins"] == 1
    assert alice["unique_devices"] == 1
    assert alice["unique_countries"] == 1

def test_alice_file_uploads():
    logins, files = make_sample_data()

    features = build_features(logins, files)

    # Check Alice's calculated features
    alice = features[
        features["employee_id"] == "alice.smith"
    ].iloc[0]

    assert alice["file_uploads"] == 1
    assert alice["file_downloads"] == 0
    assert alice["bytes_uploaded"] == 100

def test_external_transfers():
    logins, files = make_sample_data()
    features = build_features(logins, files)

    carol = features[
        features["employee_id"] == "carol.lee"
    ].iloc[0]

    assert carol["external_transfers"] == 1
    assert carol["bytes_external"] == 200

def test_one_row_per_employee_per_day():
    logins, files = make_sample_data()

    features = build_features(logins, files)

    assert not features.duplicated(
        subset=["employee_id", "timestamp"]
    ).any()

def test_score_output():
    logins, files = make_sample_data()

    features = build_features(logins, files)

    result = score(features)

    # Row count should remain unchanged
    assert len(result) == len(features)

    # Check that the two ML output columns exist
    assert "anomaly_score" in result.columns
    assert "anomaly_flag" in result.columns

    # Check that scores are valid
    assert result["anomaly_score"].notna().all()

    # Check that flags are Boolean
    assert result["anomaly_flag"].isin([True, False]).all()

def test_score_output():
    """Check that Isolation Forest generates scores and flags."""

    logins, files = make_sample_data()

    features = build_features(logins, files)
    result = score(features)

    # Check required output columns
    assert "anomaly_score" in result.columns
    assert "anomaly_flag" in result.columns

    # Check that scores were generated
    assert result["anomaly_score"].notna().all()

    # Check that anomaly flags are Boolean
    assert result["anomaly_flag"].isin([True, False]).all()

    # Check that the number of observations is unchanged
    assert len(result) == len(features)

def test_obvious_spike_has_highest_score():

    # Create 30 normal observations
    normal_data = pd.DataFrame({
        "employee_id": [f"user{i}" for i in range(30)],
        "timestamp": pd.date_range(
            start="2026-08-01",
            periods=30
        ),
        "total_logins": [2] * 30,
        "failed_logins": [0] * 30,
        "off_hours_logins": [0] * 30,
        "unique_devices": [1] * 30,
        "unique_countries": [1] * 30,
        "file_uploads": [0] * 30,
        "file_downloads": [1] * 30,
        "bytes_uploaded": [0] * 30,
        "bytes_downloaded": [100] * 30,
        "external_transfers": [0] * 30,
        "bytes_external": [0] * 30,
        "off_hours_file_events": [0] * 30
    })

    # Create one extremely unusual observation
    spike = pd.DataFrame({
        "employee_id": ["spike.user"],
        "timestamp": [pd.Timestamp("2026-08-31")],
        "total_logins": [1000],
        "failed_logins": [500],
        "off_hours_logins": [500],
        "unique_devices": [20],
        "unique_countries": [10],
        "file_uploads": [100],
        "file_downloads": [100],
        "bytes_uploaded": [1000000000],
        "bytes_downloaded": [1000000000],
        "external_transfers": [100],
        "bytes_external": [1000000000],
        "off_hours_file_events": [100]
    })

    # Combine normal observations and spike
    features = pd.concat(
        [normal_data, spike],
        ignore_index=True
    )

    result = score(features)

    # Find the employee with the highest anomaly score
    highest_score_employee = result.loc[
        result["anomaly_score"].idxmax(),
        "employee_id"
    ]

    assert highest_score_employee == "spike.user"

def test_empty_features():
    """Check behavior when there are no observations."""

    empty = pd.DataFrame()

    result = score(empty)

    assert result.empty


def test_single_observation():
    """Check behavior when there is only one employee-day."""

    features = pd.DataFrame({
        "employee_id": ["alice.smith"],
        "timestamp": [pd.Timestamp("2026-08-03")],
        "total_logins": [2],
        "failed_logins": [0],
        "off_hours_logins": [0],
        "unique_devices": [1],
        "unique_countries": [1],
        "file_uploads": [0],
        "file_downloads": [0],
        "bytes_uploaded": [0],
        "bytes_downloaded": [0],
        "external_transfers": [0],
        "bytes_external": [0],
        "off_hours_file_events": [0]
    })

    result = score(features)

    assert result["anomaly_score"].iloc[0] == 0.0
    assert result["anomaly_flag"].iloc[0] == False

def test_off_hours_file_events():
    logins, files = make_sample_data()

    # Add a file event at 2 AM
    files.loc[len(files)] = {
        "username": "alice.smith",
        "timestamp": "2026-08-03 02:00:00",
        "action": "usb_copy",
        "size_bytes": 1000
    }

    features = build_features(logins, files)

    alice = features[
        features["employee_id"] == "alice.smith"
    ].iloc[0]

    assert alice["off_hours_file_events"] == 1