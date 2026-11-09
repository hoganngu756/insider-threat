from src import detect, explain, generate_data, store


def main():
    events = generate_data.generate_logs()
    store.save(events)
    features = detect.build_features(store.load())
    scored = detect.score(features)
    if not scored.empty:
        alerts = scored.nlargest(10, "risk_score").copy()
        alerts["explanation"] = [explain.explain_alert(r) for r in alerts.to_dict("records")]
        alerts.to_csv("data/alerts.csv", index=False)


if __name__ == "__main__":
    main()
