from src.risk_genai.Risk_genAI import score_and_find_incidents

def build_incidents(rule_flags, ml_scores):
    scored, incidents = score_and_find_incidents(
        ml_scores,
        rule_flags
    )

    return incidents