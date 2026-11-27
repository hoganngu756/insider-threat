"""Unit tests. Run: py -m unittest -v test_risk_genai"""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import pandas as pd

from Risk_genAI import (
    PROJECT_FOLDER, allowed_scenarios, evidence_text, format_bytes,
    groq_report, load_ml_data, load_rule_flags, score_and_find_incidents,
)


def fake_client(answer):
    def create(**kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=json.dumps(answer))
        )])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


class Person3Tests(unittest.TestCase):
    def setUp(self):
        self.df = load_ml_data(PROJECT_FOLDER / "sample_ml_output.csv")
        self.flags = load_rule_flags(PROJECT_FOLDER / "sample_rule_flags.csv")
        self.scored, self.incidents = score_and_find_incidents(self.df, self.flags)

    def test_scores_without_rule_flags_match_original(self):
        old, old_incidents = score_and_find_incidents(self.df)
        values = dict(zip(old.employee_id, old.risk_score))
        self.assertEqual(values["E002"], 80)
        self.assertEqual(values["E003"], 55)
        self.assertEqual(values["E005"], 25)
        self.assertEqual(values["E007"], 15)
        self.assertEqual(len(old_incidents), 4)

    def test_rule_flagged_days_are_incidents(self):
        self.assertEqual(len(self.incidents), 6)
        self.assertEqual(set(self.incidents.employee_id),
                         {"E001", "E002", "E003", "E005", "E006", "E007"})

    def test_high_rule_applies_floor(self):
        row = self.scored.set_index("employee_id").loc["E001"]
        self.assertEqual(row.heuristic_score, 0)
        self.assertEqual(row.risk_score, 70)
        self.assertEqual(row.severity, "High")
        self.assertIn("impossible_travel", row.triggered_rules)

    def test_rule_does_not_reduce_heuristic(self):
        row = self.scored.set_index("employee_id").loc["E003"]
        self.assertEqual(row.heuristic_score, 55)
        self.assertEqual(row.risk_score, 55)
        self.assertEqual(row.severity, "Medium")

    def test_no_duplicate_incidents_when_multiple_rules(self):
        with tempfile.TemporaryDirectory() as t:
            file = Path(t) / "rules.csv"
            pd.DataFrame([
                {"employee_id":"E001","timestamp":"2026-11-20 12:00:00","rule_name":"impossible_travel","severity":"High"},
                {"employee_id":"E001","timestamp":"2026-11-20 13:00:00","rule_name":"new_device_foreign_login","severity":"Medium"},
            ]).to_csv(file, index=False)
            flags = load_rule_flags(file)
            scored, inc = score_and_find_incidents(self.df, flags)
            self.assertEqual(len(inc[inc.employee_id == "E001"]), 1)
            row = scored.set_index("employee_id").loc["E001"]
            self.assertIn("new_device_foreign_login", row.rule_flags)
            self.assertIn("impossible_travel", row.rule_flags)
            self.assertEqual(row.rule_severity, "High")

    def test_medium_rule_floor_and_account_target_scenario(self):
        with tempfile.TemporaryDirectory() as t:
            file = Path(t) / "rules.csv"
            pd.DataFrame([{"employee_id":"E004","timestamp":"2026-11-20",
                "rule_name":"credential_stuffing", "severity":"Medium"}]).to_csv(file, index=False)
            flags = load_rule_flags(file)
            scored, inc = score_and_find_incidents(self.df, flags)
            row = inc.set_index("employee_id").loc["E004"]
            self.assertEqual(row.risk_score, 40)
            self.assertEqual(row.severity, "Medium")
            self.assertIn("authentication_attack_target", allowed_scenarios(row))

    def test_unmatched_rule_day_raises(self):
        bad_flags = self.flags.copy()
        bad_flags.loc[0, "employee_id"] = "UNKNOWN"
        with self.assertRaisesRegex(ValueError, "missing from ML output"):
            score_and_find_incidents(self.df, bad_flags)

    def test_invalid_rule_schema_raises(self):
        with tempfile.TemporaryDirectory() as t:
            file = Path(t) / "bad.csv"
            pd.DataFrame([{"employee_id":"E001","timestamp":"2026-11-20", "unknown":"x"}]).to_csv(file,index=False)
            with self.assertRaisesRegex(ValueError, "needs columns"):
                load_rule_flags(file)

    def test_invalid_severity_raises(self):
        with tempfile.TemporaryDirectory() as t:
            file = Path(t) / "bad.csv"
            pd.DataFrame([{"employee_id":"E001","timestamp":"2026-11-20", "rule_name":"x","severity":"Extreme"}]).to_csv(file,index=False)
            with self.assertRaisesRegex(ValueError, "Unknown rule severity"):
                load_rule_flags(file)

    def test_inactive_rules_do_not_trigger(self):
        with tempfile.TemporaryDirectory() as t:
            file = Path(t) / "flags.csv"
            pd.DataFrame([{"employee_id":"E004","timestamp":"2026-11-20", "rule_name":"impossible_travel","severity":"High","flagged":False}]).to_csv(file,index=False)
            flags=load_rule_flags(file)
            scored, inc = score_and_find_incidents(self.df, flags)
            self.assertNotIn("E004", inc.employee_id.tolist())

    def test_mib_conversion(self):
        self.assertEqual(format_bytes(110000000), "110,000,000 bytes (104.90 MiB)")

    def test_impossible_failed_logins_rejected(self):
        bad = self.df.copy()
        bad.loc[bad.employee_id=="E005", "total_logins"] = 9
        with tempfile.TemporaryDirectory() as t:
            file=Path(t)/"bad.csv"
            bad.to_csv(file,index=False)
            with self.assertRaisesRegex(ValueError, "failed_logins cannot exceed"):
                load_ml_data(file)

    def test_ai_evidence_includes_rule_name(self):
        row = self.incidents.set_index("employee_id").loc["E001"]
        self.assertIn("impossible_travel", evidence_text(row))
        self.assertIn("unusual_login", allowed_scenarios(row))

    def test_groq_fake_response_format(self):
        row=self.incidents.loc[self.incidents.employee_id == "E006"].iloc[0]
        answer={"scenario_code":"file_transfer", "recommended_actions":"Review transfer logs and verify authorization."}
        report=groq_report(row,fake_client(answer))
        self.assertEqual(report["method"],"groq")
        self.assertIn("Hogan",report["summary"])
        self.assertIn("usb_exfiltration",report["key_evidence"])
        self.assertIn("bulk_download",report["key_evidence"])


if __name__ == "__main__":
    unittest.main()
