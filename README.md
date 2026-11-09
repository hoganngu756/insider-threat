# Insider Threat Detection

AI-powered security analytics on synthetic employee logs. Detects brute-force logins, compromised accounts, and data exfiltration using Isolation Forest, scores risk, and uses an LLM to explain alerts.

## Pipeline

1. `src/generate_data.py` – create synthetic logs (logins, devices, file activity) with injected attacks
2. `src/store.py` – load logs into SQLite
3. `src/detect.py` – features, Isolation Forest, risk score
4. `src/explain.py` – LLM summary + next investigation steps per alert
5. `data/alerts.csv` – output for Power BI

## Run

```
pip install -r requirements.txt
python main.py
```

## Stack

Python, Pandas, NumPy, scikit-learn, SQLite, LLM API, Power BI
