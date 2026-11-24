"""Simple local tests. Run: py -m unittest -v test_person3.py"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from person3 import (
    PROJECT_FOLDER, calculate_risk, evidence_text, format_bytes,
    groq_report, load_ml_data, score_and_find_incidents,
)


class Person3Tests(unittest.TestCase):
    def setUp(self):
        self.data = load_ml_data(PROJECT_FOLDER / "sample_ml_output.csv")

    def test_four_incidents(self):
        scored, incidents = score_and_find_incidents(self.data)
        self.assertEqual(len(scored), 7)
        self.assertEqual(len(incidents), 4)
        self.assertEqual(set(incidents["employee_id"]), {"E002", "E003", "E005", "E007"})

    def test_known_risk_scores(self):
        scored, _ = score_and_find_incidents(self.data)
        actual = dict(zip(scored.employee_id, scored.risk_score))
        self.assertEqual(actual["E002"], 80)
        self.assertEqual(actual["E003"], 55)
        self.assertEqual(actual["E005"], 25)
        self.assertEqual(actual["E007"], 15)

    def test_brute_force_alert_not_missed(self):
        _, incidents = score_and_find_incidents(self.data)
        e005 = incidents.loc[incidents.employee_id == "E005"].iloc[0]
        self.assertEqual(e005["severity"], "Low")
        self.assertIn("failed logins", e005["triggered_rules"])

    def test_correct_binary_units(self):
        self.assertEqual(format_bytes(110_000_000), "110,000,000 bytes (104.90 MiB)")
        row = self.data.loc[self.data.employee_id == "E003"].iloc[0]
        self.assertIn("110,000,000 bytes (104.90 MiB)", evidence_text(row))

    def test_impossible_failed_logins_rejected(self):
        bad = self.data.copy()
        bad.loc[bad.employee_id == "E005", "total_logins"] = 9
        with tempfile.TemporaryDirectory() as folder:
            csv = Path(folder) / "bad.csv"
            bad.to_csv(csv, index=False)
            with self.assertRaisesRegex(ValueError, "failed_logins cannot exceed"):
                load_ml_data(csv)

    def test_bad_ml_flag_rejected(self):
        bad = self.data.copy()
        bad["anomaly_flag"] = bad["anomaly_flag"].astype(str)
        bad.loc[0, "anomaly_flag"] = "maybe"
        with tempfile.TemporaryDirectory() as folder:
            csv = Path(folder) / "bad.csv"
            bad.to_csv(csv, index=False)
            with self.assertRaisesRegex(ValueError, "anomaly_flag"):
                load_ml_data(csv)

    def test_duplicate_employee_day_rejected(self):
        bad = pd.concat([self.data, self.data.iloc[[0]]], ignore_index=True)
        with tempfile.TemporaryDirectory() as folder:
            csv = Path(folder) / "bad.csv"
            bad.to_csv(csv, index=False)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_ml_data(csv)

    def test_groq_report_format_with_fake_api(self):
        # Tests that Groq's response can be parsed, without an API key or network.
        answer = {"narrative_summary": "The observed pattern warrants further review.",
                  "possible_scenario": "Possible account misuse; not confirmed.",
                  "recommended_actions": "Review sign-in logs and verify with the user."}

        def respond(**kwargs):
            self.assertEqual(kwargs["response_format"], {"type": "json_object"})
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(answer)))])

        client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=respond)))
        scored, incidents = score_and_find_incidents(self.data)
        report = groq_report(incidents.iloc[1], client)
        self.assertEqual(report["method"], "groq")
        self.assertIn("104.90 MiB", report["key_evidence"])
        self.assertIn("Possible account misuse", report["possible_scenario"])
        self.assertIn("AI assessment", report["summary"])


if __name__ == "__main__":
    unittest.main()
