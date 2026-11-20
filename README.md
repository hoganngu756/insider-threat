# Insider Threat Detection

AI-powered security analytics on synthetic employee logs. Detects brute-force logins, compromised accounts, and data exfiltration with rules and Isolation Forest, scores risk, and uses an LLM to explain alerts.

## Pipeline

`python main.py` runs every stage in order:

1. **Load and clean** (`src/detection/load_logs.py`, `clean_columns.py`): reads the raw SIEM export `data/mock_data.csv` and splits it into login and file events.
2. **Rule detection** (`src/detection/`): six detectors write `data/processed/rule_flags.csv`.
3. **ML anomaly detection** (`src/detect.py`): daily features per employee plus Isolation Forest write `data/processed/ml_output.csv`.
4. **Risk scoring** (`src/risk.py`): writes `data/processed/incidents.csv`.
5. **AI investigations** (`src/explain.py`): writes `data/processed/ai_investigations.csv`.
6. **Storage** (`src/store.py`): loads everything into SQLite at `data/security.db` and builds the dashboard views.
7. **Power BI export** (`src/powerbi.py`): writes the dashboard CSVs to `data/powerbi/`.

Stages 1 and 2 must succeed. Stages 3 to 5 are optional: if a stage's code isn't in the repo yet it shows as `skipped`, and if it crashes it shows as `failed` while the rest still runs. The run ends with a status table and exits 1 if any stage failed.

## Run

`data/mock_data.csv` is gitignored. Get it from the team and put it in `data/`.

```
pip install -r requirements.txt
python main.py --skip-llm
```

Options:

1. `--skip-llm` runs the AI stage without calling the API.
2. `--placeholders` fills missing risk scoring and AI output with rows marked `PLACEHOLDER`, so the dashboard can be built early. Placeholders go into the database and Power BI files only, never `data/processed/`.
3. `--strict` stops on the first failing stage.
4. `--data`, `--db`, `--processed-dir`, `--powerbi-dir` change the paths.

Tests: `python -m pytest`

## Hand-off format

Every table joins on `employee_id` (`first.last`) and `date` (`YYYY-MM-DD`).

| Stage | Entry point | Required columns |
|---|---|---|
| Rules | `flag_events(logins, files)` | `employee_id, date, timestamp, rule_name, severity, evidence` |
| ML | `build_features(logins, files)` then `score(features)` | `employee_id, date` (or `timestamp`), `anomaly_score, anomaly_flag` |
| Risk | `src/risk.py: build_incidents(rule_flags, ml_scores)` | `incident_id, employee_id, date, risk_score, severity, triggered_rules` |
| AI | `src/explain.py: build_investigations(incidents, use_llm)` | `incident_id, summary, evidence, recommended_steps` |

Extra columns are kept. A stage that returns rows without a required column fails in the storage step with the missing column named.

## Database

`data/security.db` has one table per stage (`logins`, `file_events`, `rule_flags`, `ml_scores`, `incidents`, `ai_investigations`) and these views:

1. `v_employee_day`: one row per employee per day with login and file counts, bytes sent out (USB, personal cloud, external email), rule flags, ML score and features (`ml_*`), and incident risk.
2. `v_employee_summary`: one row per employee with totals, top severity and last flagged date.
3. `v_incident_detail`: each incident with its AI summary and recommended steps.
4. `v_rule_flags_daily`: flag counts per day, rule and severity.

## Power BI

1. Run the pipeline.
2. In Power BI Desktop, use Get Data > Text/CSV and import each file in `data/powerbi/`.
3. Relate `employee_summary` to `employee_day`, `incident_detail` and `rule_flags` on `employee_id`.

## Stack

Python, Pandas, NumPy, scikit-learn, SQLite, LLM API, Power BI
