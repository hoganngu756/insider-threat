"""SQLite storage for every pipeline output, plus the views the dashboard reads."""
import os
import sqlite3

import pandas as pd

DB = "data/security.db"

SCHEMA = {
    "logins": ["employee_id", "date", "timestamp", "dev", "src_ip", "city", "country", "outcome", "failure_reason"],
    "file_events": ["employee_id", "date", "timestamp", "dev", "obj", "action", "size_bytes", "dest", "sensitivity"],
    "rule_flags": ["employee_id", "date", "timestamp", "rule_name", "severity", "evidence"],
    "ml_scores": ["employee_id", "date", "anomaly_score", "anomaly_flag"],
    "incidents": ["incident_id", "employee_id", "date", "risk_score", "severity", "triggered_rules"],
    "ai_investigations": ["incident_id", "summary", "evidence", "recommended_steps"],
}

INDEXES = {
    "logins": ["employee_id", "date"],
    "file_events": ["employee_id", "date"],
    "rule_flags": ["employee_id", "date"],
    "ml_scores": ["employee_id", "date"],
    "incidents": ["employee_id", "date"],
    "ai_investigations": ["incident_id"],
}

EXTERNAL_ACTIONS = ("usb_copy", "cloud_upload", "email_external")


def _rank(col):
    return f"CASE {col} WHEN 'High' THEN 3 WHEN 'Medium' THEN 2 WHEN 'Low' THEN 1 END"


def _label(expr):
    return f"CASE {expr} WHEN 3 THEN 'High' WHEN 2 THEN 'Medium' WHEN 1 THEN 'Low' END"


def normalize(df, table):
    """Shape a stage's output for its table: employee_id, a YYYY-MM-DD date, text timestamps.

    Required columns come first and extra columns are kept. A non-empty frame missing a
    required column raises, so a changed hand-off format fails here instead of in the dashboard.
    """
    required = SCHEMA[table]
    if df is None or df.empty:
        return pd.DataFrame(columns=required)
    df = df.rename(columns={"username": "employee_id"}).copy()
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
        if "date" in required and "date" not in df.columns:
            df["date"] = ts.dt.strftime("%Y-%m-%d")
        df["timestamp"] = ts.dt.strftime("%Y-%m-%d %H:%M:%S")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{table} is missing required columns: {missing}")
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
    extras = [c for c in df.columns if c not in required]
    return df[required + extras].reset_index(drop=True)


def _views(ml_columns):
    base = {"employee_id", "date", "timestamp", "anomaly_score", "anomaly_flag"}
    ml_select = "".join(f', m."{c}" AS "ml_{c}"' for c in ml_columns if c not in base)
    external = ", ".join(f"'{a}'" for a in EXTERNAL_ACTIONS)
    return {
        "v_employee_day": f"""
            WITH keys AS (
                SELECT employee_id, date FROM logins
                UNION SELECT employee_id, date FROM file_events
                UNION SELECT employee_id, date FROM rule_flags
                UNION SELECT employee_id, date FROM ml_scores
                UNION SELECT employee_id, date FROM incidents
            ),
            lg AS (
                SELECT employee_id, date, COUNT(*) AS logins, SUM(outcome = 'failure') AS failed_logins
                FROM logins GROUP BY employee_id, date
            ),
            fe AS (
                SELECT employee_id, date, COUNT(*) AS file_events,
                       SUM(CASE WHEN action IN ({external}) THEN COALESCE(size_bytes, 0) ELSE 0 END) AS bytes_sent_out
                FROM file_events GROUP BY employee_id, date
            ),
            rf AS (
                SELECT employee_id, date, COUNT(*) AS rule_flags,
                       GROUP_CONCAT(DISTINCT rule_name) AS rules,
                       MAX({_rank("severity")}) AS rule_rank
                FROM rule_flags GROUP BY employee_id, date
            ),
            inc AS (
                SELECT employee_id, date, COUNT(*) AS incidents, MAX(risk_score) AS risk_score,
                       MAX({_rank("severity")}) AS incident_rank
                FROM incidents GROUP BY employee_id, date
            )
            SELECT k.employee_id, k.date,
                   COALESCE(lg.logins, 0) AS logins,
                   COALESCE(lg.failed_logins, 0) AS failed_logins,
                   COALESCE(fe.file_events, 0) AS file_events,
                   COALESCE(fe.bytes_sent_out, 0) AS bytes_sent_out,
                   COALESCE(rf.rule_flags, 0) AS rule_flags,
                   rf.rules,
                   {_label("rf.rule_rank")} AS rule_severity,
                   m.anomaly_score, m.anomaly_flag{ml_select},
                   COALESCE(inc.incidents, 0) AS incidents,
                   inc.risk_score,
                   {_label("inc.incident_rank")} AS incident_severity
            FROM keys k
            LEFT JOIN lg USING (employee_id, date)
            LEFT JOIN fe USING (employee_id, date)
            LEFT JOIN rf USING (employee_id, date)
            LEFT JOIN ml_scores m USING (employee_id, date)
            LEFT JOIN inc USING (employee_id, date)
        """,
        "v_employee_summary": f"""
            SELECT employee_id,
                   COUNT(*) AS active_days,
                   SUM(logins) AS logins,
                   SUM(failed_logins) AS failed_logins,
                   SUM(bytes_sent_out) AS bytes_sent_out,
                   SUM(rule_flags) AS rule_flags,
                   SUM(COALESCE(anomaly_flag, 0)) AS anomaly_days,
                   SUM(incidents) AS incidents,
                   MAX(risk_score) AS max_risk_score,
                   {_label(f"MAX({_rank('incident_severity')})")} AS top_incident_severity,
                   {_label(f"MAX({_rank('rule_severity')})")} AS top_rule_severity,
                   MAX(CASE WHEN rule_flags > 0 OR incidents > 0 THEN date END) AS last_flagged_date
            FROM v_employee_day GROUP BY employee_id
        """,
        "v_incident_detail": """
            SELECT i.*, a.summary, a.evidence AS ai_evidence, a.recommended_steps
            FROM incidents i LEFT JOIN ai_investigations a USING (incident_id)
        """,
        "v_rule_flags_daily": """
            SELECT date, rule_name, severity, COUNT(*) AS flags, COUNT(DISTINCT employee_id) AS employees
            FROM rule_flags GROUP BY date, rule_name, severity
        """,
    }


def save_all(tables, db=DB):
    """Write every table and rebuild the views. The old database is replaced only after a full write."""
    os.makedirs(os.path.dirname(db) or ".", exist_ok=True)
    tmp = db + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    frames = {name: normalize(tables.get(name), name) for name in SCHEMA}
    con = sqlite3.connect(tmp)
    try:
        for name, df in frames.items():
            df.to_sql(name, con, index=False)
            for col in INDEXES[name]:
                con.execute(f'CREATE INDEX "ix_{name}_{col}" ON "{name}" ("{col}")')
        for name, sql in _views(frames["ml_scores"].columns).items():
            con.execute(f'CREATE VIEW "{name}" AS {sql}')
        con.commit()
    finally:
        con.close()
    os.replace(tmp, db)
    return {name: len(df) for name, df in frames.items()}


def read(name, db=DB):
    """Read a table or view into a DataFrame."""
    with sqlite3.connect(db) as con:
        return pd.read_sql(f'SELECT * FROM "{name}"', con)
