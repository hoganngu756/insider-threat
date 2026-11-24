"""Export the dashboard tables from SQLite to CSVs that Power BI Desktop imports with Get Data > Text/CSV."""
import os

from src import store

EXPORTS = {
    "employee_day.csv": "v_employee_day",
    "employee_summary.csv": "v_employee_summary",
    "incident_detail.csv": "v_incident_detail",
    "rule_flags.csv": "rule_flags",
    "rule_flags_daily.csv": "v_rule_flags_daily",
}


def export(db=store.DB, out_dir="data/powerbi"):
    os.makedirs(out_dir, exist_ok=True)
    for filename, source in EXPORTS.items():
        store.read(source, db).to_csv(os.path.join(out_dir, filename), index=False)
    return list(EXPORTS)
