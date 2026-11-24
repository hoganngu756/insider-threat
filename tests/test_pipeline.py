import os

import pandas as pd
import pytest

import main
from src import pipeline, store
from src.powerbi import EXPORTS

RAW_COLUMNS = ["ts", "user", "evt", "dev", "src_ip", "city", "country", "outcome", "notes", "obj", "action", "size", "dest"]


@pytest.fixture
def paths(tmp_path):
    raw = tmp_path / "raw.csv"
    rows = [["2026-09-01 08:00:00", "a.a", "login", "d1", "10.0.0.5", "Dallas", "US", "success", "", "", "", "", ""],
            ["2026-09-01 09:00:00", "a.a", "file", "d1", "", "", "", "", "internal", "/x", "download", "100", ""],
            ["2026-09-02 08:00:00", "b.b", "login", "d2", "10.0.0.6", "Dallas", "US", "failure", "bad_password",
             "", "", "", ""]]
    pd.DataFrame(rows, columns=RAW_COLUMNS).to_csv(raw, index=False)
    return {"data": str(raw), "db": str(tmp_path / "s.db"), "processed_dir": str(tmp_path / "processed"),
            "powerbi_dir": str(tmp_path / "powerbi"), "log": lambda _: None}


def ml_output(logins, files):
    return pd.DataFrame({"employee_id": ["a.a", "b.b"], "timestamp": pd.to_datetime(["2026-09-01", "2026-09-02"]),
                         "anomaly_score": [0.7, 0.4], "anomaly_flag": [True, False]})


def statuses(results):
    return {r.name: r.status for r in results}


def test_runs_end_to_end_with_ml_and_no_risk_module(paths, monkeypatch):
    monkeypatch.setattr(pipeline, "_run_ml", ml_output)
    results = pipeline.run(**paths)
    assert statuses(results) == {"load": "ok", "rules": "ok", "ml": "ok", "incidents": "skipped",
                                 "investigations": "skipped", "store": "ok", "powerbi": "ok"}
    assert sorted(os.listdir(paths["processed_dir"])) == ["ml_output.csv", "rule_flags.csv"]
    assert sorted(os.listdir(paths["powerbi_dir"])) == sorted(EXPORTS)
    assert len(store.read("logins", paths["db"])) == 2
    assert len(store.read("ml_scores", paths["db"])) == 2


def test_failing_optional_stage_is_reported_and_the_rest_still_runs(paths, monkeypatch):
    def broken(logins, files):
        raise KeyError("anomaly_score")

    monkeypatch.setattr(pipeline, "_run_ml", broken)
    results = pipeline.run(**paths)
    ml = next(r for r in results if r.name == "ml")
    assert (ml.status, ml.detail) == ("failed", "KeyError: 'anomaly_score'")
    assert statuses(results)["store"] == "ok"
    with pytest.raises(KeyError):
        pipeline.run(**paths, strict=True)


def test_real_risk_and_ai_stages_are_used_and_handed_off(paths, monkeypatch):
    seen = {}

    def incidents(rule_flags, ml_scores):
        return pd.DataFrame({"incident_id": ["I1"], "employee_id": ["a.a"], "date": ["2026-09-01"],
                             "risk_score": [80], "severity": ["High"], "triggered_rules": ["ml_anomaly"]})

    def investigations(inc, use_llm):
        seen["use_llm"] = use_llm
        return pd.DataFrame({"incident_id": ["I1"], "summary": ["s"], "evidence": ["e"], "recommended_steps": ["r"]})

    monkeypatch.setattr(pipeline, "_run_ml", ml_output)
    monkeypatch.setattr(pipeline, "_run_incidents", incidents)
    monkeypatch.setattr(pipeline, "_run_investigations", investigations)
    results = pipeline.run(**paths, use_llm=False, use_placeholders=True)
    assert statuses(results)["incidents"] == "ok"
    assert statuses(results)["investigations"] == "ok"
    assert seen["use_llm"] is False
    assert {"incidents.csv", "ai_investigations.csv"} <= set(os.listdir(paths["processed_dir"]))
    assert store.read("v_incident_detail", paths["db"]).loc[0, "summary"] == "s"


def test_placeholders_fill_the_database_but_not_the_handoff_files(paths, monkeypatch):
    monkeypatch.setattr(pipeline, "_run_ml", ml_output)
    results = pipeline.run(**paths, use_placeholders=True)
    assert statuses(results)["incidents"] == "placeholder"
    assert statuses(results)["investigations"] == "placeholder"
    assert "incidents.csv" not in os.listdir(paths["processed_dir"])
    detail = store.read("v_incident_detail", paths["db"])
    assert detail["summary"].str.startswith("PLACEHOLDER").all()
    assert len(detail) == 1


def test_contract_skips_until_the_code_exists():
    with pytest.raises(pipeline.Skip, match="src/not_a_module.py not found"):
        pipeline._contract("src.not_a_module", "run")
    with pytest.raises(pipeline.Skip, match=r"src.store.no_such_function\(\) not written yet"):
        pipeline._contract("src.store", "no_such_function")
    assert pipeline._contract("src.store", "save_all") is store.save_all


def test_main_exit_code_reflects_failed_stages(paths, monkeypatch):
    args = ["--data", paths["data"], "--db", paths["db"], "--processed-dir", paths["processed_dir"],
            "--powerbi-dir", paths["powerbi_dir"]]
    monkeypatch.setattr(pipeline, "_run_ml", ml_output)
    assert main.main(args) == 0
    monkeypatch.setattr(pipeline, "_run_ml", lambda logins, files: 1 / 0)
    assert main.main(args) == 1
