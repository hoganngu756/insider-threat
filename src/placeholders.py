"""Stand-in incidents and AI write-ups so the database and dashboard can be built before risk scoring lands.

Used only with `python main.py --placeholders`. Never written to data/processed/.
"""
import pandas as pd

from src.store import SCHEMA

RULE_POINTS = {"High": 60, "Medium": 40, "Low": 20}
ANOMALY_POINTS = 30


def _severity(score):
    return "High" if score >= 70 else "Medium" if score >= 40 else "Low"


def build_incidents(rule_flags, ml_scores):
    """One incident per employee-day with a rule flag or an ML anomaly flag."""
    parts = []
    if rule_flags is not None and not rule_flags.empty:
        rf = rule_flags.assign(points=rule_flags["severity"].map(RULE_POINTS).fillna(0))
        parts.append(rf[["employee_id", "date", "rule_name", "points"]])
    if ml_scores is not None and not ml_scores.empty and "anomaly_flag" in ml_scores:
        ml = ml_scores[ml_scores["anomaly_flag"].astype(bool)].copy()
        if "date" not in ml.columns:
            ml["date"] = pd.to_datetime(ml["timestamp"]).dt.strftime("%Y-%m-%d")
        ml = ml.assign(rule_name="ml_anomaly", points=ANOMALY_POINTS)
        parts.append(ml[["employee_id", "date", "rule_name", "points"]])
    if not parts:
        return pd.DataFrame(columns=SCHEMA["incidents"])

    hits = pd.concat(parts, ignore_index=True)
    out = (hits.groupby(["employee_id", "date"])
               .agg(top=("points", "max"), extra=("rule_name", "nunique"),
                    triggered_rules=("rule_name", lambda r: ", ".join(sorted(set(r)))))
               .reset_index()
               .sort_values(["date", "employee_id"])
               .reset_index(drop=True))
    out["risk_score"] = (out["top"] + 10 * (out["extra"] - 1)).clip(upper=100).astype(int)
    out["severity"] = out["risk_score"].map(_severity)
    out["incident_id"] = [f"INC-{i:04d}" for i in range(1, len(out) + 1)]
    return out[SCHEMA["incidents"]]


def build_investigations(incidents):
    """Placeholder text in the agreed columns, clearly marked so it is never mistaken for LLM output."""
    if incidents.empty:
        return pd.DataFrame(columns=SCHEMA["ai_investigations"])
    return pd.DataFrame({
        "incident_id": incidents["incident_id"],
        "summary": "PLACEHOLDER: " + incidents["severity"] + " risk for " + incidents["employee_id"]
                   + " on " + incidents["date"] + " (" + incidents["triggered_rules"] + ")",
        "evidence": "PLACEHOLDER: triggered " + incidents["triggered_rules"],
        "recommended_steps": "PLACEHOLDER: review this employee's logins and file transfers for the day",
    })
