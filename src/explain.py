import anthropic
import os
import time

import pandas as pd

from src.risk_genai import Risk_genAI as risk_ai


def build_investigations(incidents, use_llm=True):
    """
    Generate one investigation report for each identified incident.

    Uses Groq when use_llm=True and template reports otherwise.
    Returns a DataFrame containing the investigation reports.
    """

    if incidents is None or incidents.empty:
        return pd.DataFrame(columns=risk_ai.REPORT_COLUMNS)

    # Initialize Groq only if AI investigations are requested.
    client = None

    if use_llm:
        from dotenv import load_dotenv
        from openai import OpenAI

        load_dotenv(risk_ai.PROJECT_FOLDER / ".env")

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing GROQ_API_KEY. Check the .env file beside Risk_genAI.py."
            )

        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            max_retries=0,
            timeout=45.0,
        )

    reports = []
    new_requests = 0
    last_request_time = None
    quota_reached = False

    # Investigate the highest-risk incidents first.
    priority_incidents = incidents.sort_values(
        ["risk_score", "timestamp"],
        ascending=[False, True],
    ).reset_index(drop=True)

    for _, row in priority_incidents.iterrows():
        incident_id = str(row["incident_id"])

        if client is not None and not quota_reached:
            can_call_groq = (
                risk_ai.MAX_REAL_AI_REPORTS is None
                or new_requests < risk_ai.MAX_REAL_AI_REPORTS
            )

            if can_call_groq:
                # Pause between API requests to respect the configured rate limit.
                if last_request_time is not None:
                    elapsed = time.monotonic() - last_request_time
                    pause = max(
                        0,
                        risk_ai.AI_REQUEST_DELAY_SECONDS - elapsed,
                    )
                    if pause > 0:
                        time.sleep(pause)

                new_requests += 1
                last_request_time = time.monotonic()

                try:
                    report = risk_ai.groq_with_retry(row, client)

                except Exception as error:
                    print(
                        f"WARNING: Groq failed for {incident_id}: {error}",
                        flush=True,
                    )

                    report = risk_ai.template_report(row)
                    report["method"] = "template_fallback"

                    if getattr(error, "status_code", None) == 429:
                        quota_reached = True

            else:
                report = risk_ai.template_report(row)
                report["method"] = "template_limit"

        elif client is not None and quota_reached:
            report = risk_ai.template_report(row)
            report["method"] = "template_rate_limit"

        else:
            # Offline mode: no Groq API request.
            report = risk_ai.template_report(row)

        reports.append({
            "incident_id": incident_id,
            **report,
        })

    return pd.DataFrame(reports, columns=risk_ai.REPORT_COLUMNS)