"""Run every stage in order, write the hand-off CSVs, load SQLite and export the Power BI files.

Load and rule detection must succeed. ML, risk scoring and AI write-ups are each optional: a stage
whose code isn't in the repo yet is skipped, and one that crashes is reported as failed while the
rest of the pipeline still runs.
"""
import importlib
import importlib.util
import os
from dataclasses import dataclass

from src import placeholders, powerbi, store
from src.detection.clean_columns import split_events
from src.detection.load_logs import load_raw
from src.detection.rule_flags import flag_events


@dataclass
class StageResult:
    name: str
    status: str
    rows: int = 0
    detail: str = ""


class Skip(Exception):
    pass


def _contract(module, func):
    """Return module.func, or raise Skip if that teammate's code isn't in the repo yet."""
    if importlib.util.find_spec(module) is None:
        raise Skip(f"{module.replace('.', '/')}.py not found")
    fn = getattr(importlib.import_module(module), func, None)
    if fn is None:
        raise Skip(f"{module}.{func}() not written yet")
    return fn


def _run_ml(logins, files):
    detect = importlib.import_module("src.detect")
    scored = detect.score(detect.build_features(logins, files))
    if scored is None or scored.empty:
        raise Skip("ML stage returned no rows")
    return scored


def _run_incidents(rule_flags, ml_scores):
    return _contract("src.risk", "build_incidents")(rule_flags, ml_scores)


def _run_investigations(incidents, use_llm):
    return _contract("src.explain", "build_investigations")(incidents, use_llm=use_llm)


def run(data="data/mock_data.csv", db=store.DB, processed_dir="data/processed", powerbi_dir="data/powerbi",
        use_llm=True, use_placeholders=False, strict=False, log=print):
    results = []

    def optional(name, fn):
        try:
            out = fn()
        except Skip as e:
            results.append(StageResult(name, "skipped", detail=str(e)))
            return None
        except Exception as e:
            if strict:
                raise
            results.append(StageResult(name, "failed", detail=f"{type(e).__name__}: {e}"))
            return None
        results.append(StageResult(name, "ok", len(out)))
        return out

    def stand_in(name, out):
        results[-1] = StageResult(name, "placeholder", len(out), results[-1].detail)
        return out

    log(f"loading {data}")
    events, report = load_raw(data)
    logins, files, _ = split_events(events)
    results.append(StageResult("load", "ok", len(events),
                               f"{len(logins)} logins, {len(files)} file events, "
                               f"{report['rows_loaded'] - len(events)} raw rows dropped"))

    log("running rule detectors")
    rule_flags = flag_events(logins, files)
    results.append(StageResult("rules", "ok", len(rule_flags)))

    log("running ML anomaly detection")
    ml_scores = optional("ml", lambda: _run_ml(logins, files))

    log("running risk scoring")
    incidents = optional("incidents", lambda: _run_incidents(rule_flags, ml_scores))
    real_incidents = incidents is not None
    if incidents is None and use_placeholders:
        incidents = stand_in("incidents", placeholders.build_incidents(rule_flags, ml_scores))

    log("running AI investigations")
    investigations = None
    if real_incidents:
        investigations = optional("investigations", lambda: _run_investigations(incidents, use_llm))
    else:
        results.append(StageResult("investigations", "skipped", detail="no real incidents to explain"))
    real_investigations = investigations is not None
    if investigations is None and use_placeholders and incidents is not None:
        investigations = stand_in("investigations", placeholders.build_investigations(incidents))

    os.makedirs(processed_dir, exist_ok=True)
    handoffs = {"rule_flags.csv": rule_flags, "ml_output.csv": ml_scores}
    if real_incidents:
        handoffs["incidents.csv"] = incidents
    if real_investigations:
        handoffs["ai_investigations.csv"] = investigations
    for filename, df in handoffs.items():
        if df is not None:
            df.to_csv(os.path.join(processed_dir, filename), index=False)

    log(f"writing {db}")
    counts = store.save_all({
        "logins": logins, "file_events": files, "rule_flags": rule_flags, "ml_scores": ml_scores,
        "incidents": incidents, "ai_investigations": investigations,
    }, db)
    results.append(StageResult("store", "ok", sum(counts.values()),
                               ", ".join(f"{k} {v}" for k, v in counts.items())))

    log(f"exporting Power BI files to {powerbi_dir}")
    written = powerbi.export(db, powerbi_dir)
    results.append(StageResult("powerbi", "ok", len(written), ", ".join(written)))
    return results


def format_results(results):
    rows = [(r.name, r.status, str(r.rows), r.detail) for r in results]
    widths = [max(len(row[i]) for row in rows + [("stage", "status", "rows", "")]) for i in range(3)]
    lines = [f"{'stage':<{widths[0]}}  {'status':<{widths[1]}}  {'rows':>{widths[2]}}  detail"]
    for name, status, rows_, detail in rows:
        lines.append(f"{name:<{widths[0]}}  {status:<{widths[1]}}  {rows_:>{widths[2]}}  {detail}")
    return "\n".join(lines)
