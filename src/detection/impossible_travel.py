"""Step 3d: impossible travel = two successful logins too far apart for the time between them."""
from math import asin, cos, radians, sin, sqrt

import pandas as pd

from src.detection.brute_force import FLAG_COLUMNS

# (lat, lon) of every city that appears in the data. The log has no coordinates, so this
# small lookup is needed; a city that is not listed here is skipped (and reported).
CITY_COORDS = {
    "Dallas": (32.78, -96.80), "Austin": (30.27, -97.74), "Houston": (29.76, -95.37),
    "Atlanta": (33.75, -84.39), "Denver": (39.74, -104.99), "Seattle": (47.61, -122.33),
    "Phoenix": (33.45, -112.07), "Chicago": (41.88, -87.63),
    "Lagos": (6.52, 3.38), "Bucharest": (44.43, 26.10), "Shenzhen": (22.54, 114.06),
}
MAX_SPEED_KMH = 900  # about airliner cruise speed; faster than this is not physically possible


def normalize_city(c):
    """'DALLAS, TX' / 'dallas' -> 'Dallas'; null -> None."""
    if c is None or pd.isna(c):
        return None
    return str(c).split(",")[0].strip().title() or None


def distance_km(a, b):
    """Great-circle (haversine) distance between two (lat, lon) points."""
    lat1, lon1, lat2, lon2 = map(radians, (*a, *b))
    h = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371 * asin(sqrt(h))


def detect_impossible_travel(logins, max_speed=MAX_SPEED_KMH, min_km=100):
    """Compare each user's consecutive successful logins; flag the later one if the implied
    speed exceeds `max_speed` km/h. High if the countries differ, Medium if domestic."""
    ok = logins[logins["outcome"] == "success"].copy()
    ok["city_n"] = ok["city"].map(normalize_city)
    ok = ok[ok["city_n"].isin(CITY_COORDS)].sort_values(["username", "timestamp"])

    rows = []
    for user, g in ok.groupby("username"):
        prev = None
        for r in g.itertuples():
            if prev is not None and r.city_n != prev.city_n:
                km = distance_km(CITY_COORDS[prev.city_n], CITY_COORDS[r.city_n])
                hours = max((r.timestamp - prev.timestamp).total_seconds() / 3600, 1 / 60)  # floor at 1 min
                speed = km / hours
                if km >= min_km and speed > max_speed:
                    international = pd.notna(r.country) and pd.notna(prev.country) and r.country != prev.country
                    minutes = round(hours * 60)
                    evidence = (f"{prev.city_n} -> {r.city_n}: {km:,.0f} km in {minutes} min "
                                f"({speed:,.0f} km/h)")
                    rows.append([user, r.timestamp.strftime("%Y-%m-%d"), r.timestamp, "impossible_travel",
                                 "High" if international else "Medium", evidence])
            prev = r
    return pd.DataFrame(rows, columns=FLAG_COLUMNS)


if __name__ == "__main__":
    import sys

    from src.detection.clean_columns import split_events
    from src.detection.load_logs import load_raw

    events, _ = load_raw(sys.argv[1] if len(sys.argv) > 1 else "data/mock_data.csv")
    logins, _, _ = split_events(events)
    ok = logins[logins["outcome"] == "success"]
    unknown = sorted(set(ok["city"].map(normalize_city).dropna()) - set(CITY_COORDS))
    print("cities with no coordinates (skipped):", unknown)
    flags = detect_impossible_travel(logins)
    print(len(flags), "flags")
    print(flags.to_string())
