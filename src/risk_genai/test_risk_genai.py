""" Run: py -m unittest -v test_risk_genai.py"""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from Risk_genAI import (
    PROJECT_FOLDER, allowed_scenarios, calculate_risk,
    evidence_text, format_bytes, groq_report, load_ml_data,
    score_and_find_incidents,
)


def fake_client(answers):
    """Simulate Groq replies for testing without a real API key."""
    if isinstance(answers, dict):
        answers = [answers]
    replies = iter(answers)
    def create(**kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=json.dumps(next(replies)))
        )])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


class RiskAndAITests(unittest.TestCase):
    def setUp(self):
        self.df = load_ml_data(PROJECT_FOLDER / "sample_ml_output.csv")
        self.scored, self.incidents = score_and_find_incidents(self.df)

    def test_scores(self):
        scores = dict(zip(self.scored.employee_id, self.scored.risk_score))
        self.assertEqual(scores["E002"], 80)
        self.assertEqual(scores["E003"], 55)
        self.assertEqual(scores["E005"], 25)
        self.assertEqual(scores["E007"], 15)

    def test_four_incidents(self):
        self.assertEqual(len(self.scored), 7)
        self.assertEqual(len(self.incidents), 4)
        self.assertEqual(set(self.incidents.employee_id), {"E002", "E003", "E005", "E007"})

    def test_mib_conversion(self):
        self.assertEqual(format_bytes(110_000_000), "110,000,000 bytes (104.90 MiB)")

    def test_mismatched_login_counts_rejected(self):
        bad = self.df.copy()
        bad.loc[bad.employee_id == "E005", "total_logins"] = 9
        with tempfile.TemporaryDirectory() as folder:
            fn = Path(folder) / "invalid.csv"
            bad.to_csv(fn, index=False)
            with self.assertRaisesRegex(ValueError, "failed_logins cannot exceed"):
                load_ml_data(fn)

    def test_no_specific_attack_for_ml_only(self):
        row = self.incidents.loc[self.incidents.employee_id == "E007"].iloc[0]
        self.assertNotIn("credential_guessing", allowed_scenarios(row))
        self.assertNotIn("file_transfer", allowed_scenarios(row))
        self.assertIn("ml_anomaly_only", allowed_scenarios(row))

    def test_valid_ai_report(self):
        row = self.incidents.loc[self.incidents.employee_id == "E003"].iloc[0]
        answer = {"scenario_code": "file_transfer", "recommended_actions": "Review transfer logs and confirm authorization."}
        report = groq_report(row, fake_client(answer))
        self.assertEqual(report["method"], "groq")
        self.assertIn("104.90 MiB", report["key_evidence"])
        self.assertIn("Possible suspicious external", report["possible_scenario"])

    def test_wrong_scenario_is_rejected_then_fixed(self):
        row = self.incidents.loc[self.incidents.employee_id == "E007"].iloc[0]
        answers = [
            {"scenario_code": "credential_guessing", "recommended_actions": "Check login attempts."},
            {"scenario_code": "ml_anomaly_only", "recommended_actions": "Review raw logs and verify activity."},
        ]
        report = groq_report(row, fake_client(answers))
        self.assertIn("do not establish a specific attack", report["possible_scenario"])
        self.assertNotIn("9 failed", report["summary"])
        self.assertNotIn("brute-force", report["possible_scenario"])

    def test_repeated_invalid_scenario_fails(self):
        row = self.incidents.loc[self.incidents.employee_id == "E007"].iloc[0]
        answer = {"scenario_code": "credential_guessing", "recommended_actions": "Review the logs."}
        with self.assertRaisesRegex(ValueError, "supported scenario"):
            groq_report(row, fake_client([answer, answer]))

    def test_report_cannot_rewrite_factual_summary(self):
        row = self.incidents.loc[self.incidents.employee_id == "E007"].iloc[0]
        answer = {"scenario_code": "ml_anomaly_only", "recommended_actions": "Review sign-in logs."}
        report = groq_report(row, fake_client(answer))
        self.assertIn("Employee E007", report["summary"])
        self.assertIn("Failed logins: 0/9", report["key_evidence"])
        self.assertNotIn("9 failed", report["summary"])


if __name__ == "__main__":
    unittest.main()
