"""Read-only NeuroDIC command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Callable

from .agent.errors import ControlPlaneError, ErrorRecord, error_envelope
from .agent.inspect import inspect_artifact, inspect_case, inspect_config, inspect_pipeline, inspect_result
from .agent.evaluate import evaluate_result
from .agent.diagnose import diagnose_result, diagnose_quality_report, load_quality_report
from .agent.trials import plan_trial
from .agent.execution import execute_trial
from .agent.compare import compare_quality_reports
from .agent.best import load_best, update_best
from .agent.recommend import recommend_from_diagnosis
from .config import load_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neurodic", description="Read-only NeuroDIC inspection")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="Inspect existing configuration, cases, pipelines, or artifacts")
    inspect_sub = inspect.add_subparsers(dest="target", required=True)
    for name in ("case", "config", "pipeline", "result"):
        item = inspect_sub.add_parser(name, help=f"Read-only inspect {name}")
        item.add_argument("--config", required=True, help="Solver YAML path")
        item.add_argument("--case-key", help="case_paths.yaml mapping key")
        item.add_argument("--case-paths", default="config/case_paths.yaml")
        item.add_argument("--case-root", help="Explicit case root override")
        item.add_argument("--solver", help="Canonical solver ID")
        item.add_argument("--format", choices=("json", "text"), default="json")
    artifact = inspect_sub.add_parser("artifact", help="Read-only inspect one artifact")
    artifact.add_argument("--path", required=True); artifact.add_argument("--case-root", required=True)
    artifact.add_argument("--artifact-type", default="unknown"); artifact.add_argument("--artifact-schema", default="unknown/v1")
    artifact.add_argument("--producer-stage", default="unknown"); artifact.add_argument("--format", choices=("json", "text"), default="json")
    evaluate = sub.add_parser("evaluate", help="Read-only evidence and quality evaluation")
    evaluate.add_argument("--config", required=True); evaluate.add_argument("--case-key")
    evaluate.add_argument("--case-paths", default="config/case_paths.yaml"); evaluate.add_argument("--case-root")
    evaluate.add_argument("--solver"); evaluate.add_argument("--profile", default="config/quality_profiles/default.yaml")
    evaluate.add_argument("--format", choices=("json", "text"), default="json")
    diagnose = sub.add_parser("diagnose", help="Read-only deterministic failure-family diagnosis")
    source = diagnose.add_mutually_exclusive_group(required=True)
    source.add_argument("--quality", help="Existing QualityReport JSON path")
    source.add_argument("--config", help="Solver YAML path to inspect/evaluate/diagnose")
    diagnose.add_argument("--case-key"); diagnose.add_argument("--case-paths", default="config/case_paths.yaml")
    diagnose.add_argument("--case-root"); diagnose.add_argument("--solver"); diagnose.add_argument("--profile", default="config/quality_profiles/default.yaml")
    diagnose.add_argument("--format", choices=("json", "text"), default="json")
    recommend = sub.add_parser("recommend", help="Bounded diagnosis-driven dry-run recommendation; never executes")
    recommend.add_argument("--diagnosis", required=True); recommend.add_argument("--config", required=True); recommend.add_argument("--case-key")
    recommend.add_argument("--case-paths", default="config/case_paths.yaml"); recommend.add_argument("--trial-id")
    recommend.add_argument("--parameter-registry", default="config/agent/parameter_registry.yaml"); recommend.add_argument("--intervention-rules", default="config/agent/intervention_rules.yaml")
    recommend.add_argument("--format", choices=("json", "text"), default="json")
    trial = sub.add_parser("trial", help="Dry-run trial planning; never executes a solver")
    trial_sub = trial.add_subparsers(dest="trial_command", required=True)
    plan = trial_sub.add_parser("plan", help="Plan a sparse configuration override without writing files")
    plan.add_argument("--config", required=True, help="Baseline solver YAML path")
    plan.add_argument("--override", help="Sparse YAML mapping containing only changed fields")
    plan.add_argument("--case-key"); plan.add_argument("--case-paths", default="config/case_paths.yaml")
    plan.add_argument("--case-root"); plan.add_argument("--solver"); plan.add_argument("--trial-id")
    plan.add_argument("--restore-missing", action="store_true", help="Also plan restoration of missing/unverified producer outputs")
    plan.add_argument("--format", choices=("json", "text"), default="json")
    execute = trial_sub.add_parser("execute", help="Execute one previously approved TrialPlan")
    execute.add_argument("--plan", required=True, help="TrialPlan JSON emitted by 'trial plan'")
    execute.add_argument("--managed-root", required=True, help="Existing trusted managed execution root")
    execute.add_argument("--action", help="One action_id already present in the approved plan")
    execute.add_argument("--format", choices=("json", "text"), default="json")
    compare = sub.add_parser("compare", help="Read-only QualityReport comparison")
    compare.add_argument("--baseline", required=True); compare.add_argument("--candidate", required=True)
    compare.add_argument("--profile", default="config/comparison_profiles/default.yaml"); compare.add_argument("--format", choices=("json", "text"), default="json")
    best = sub.add_parser("best", help="Explicit managed best-reference operations")
    best_sub = best.add_subparsers(dest="best_command", required=True)
    show = best_sub.add_parser("show"); show.add_argument("--managed-root", required=True); show.add_argument("--format", choices=("json", "text"), default="json")
    evaluate_best = best_sub.add_parser("evaluate"); evaluate_best.add_argument("--candidate", required=True); evaluate_best.add_argument("--managed-root", required=True); evaluate_best.add_argument("--profile", default="config/comparison_profiles/default.yaml"); evaluate_best.add_argument("--format", choices=("json", "text"), default="json")
    promote = best_sub.add_parser("promote"); promote.add_argument("--comparison", required=True); promote.add_argument("--baseline", required=True); promote.add_argument("--candidate", required=True); promote.add_argument("--managed-root", required=True); promote.add_argument("--expected-current-best-identity"); promote.add_argument("--format", choices=("json", "text"), default="json")
    return parser


def _text(value: dict) -> str:
    if value["status"] == "error":
        return "\n".join(f"{item['code']}: {item['message']}" for item in value["errors"])
    data = value["data"]
    lines = [f"operation: {value['operation']}"]
    for key in ("solver", "mode", "case_root"):
        if key in data: lines.append(f"{key}: {data[key]}")
    if "readiness" in data: lines.append(f"ready: {data['readiness']['ready']}")
    if "artifacts" in data: lines.append(f"artifacts: {len(data['artifacts'])}")
    if "stages" in data: lines.append(f"stages: {len(data['stages'])}")
    if "diagnosis" in data:
        diagnosis=data["diagnosis"]; lines += [f"overall_status: {diagnosis['overall_status']}", f"primary_diagnosis: {diagnosis['primary_diagnosis']}"]
    if "trial_plan" in data:
        plan = data["trial_plan"]; lines += [f"plan_status: {plan['plan_status']}", f"changes: {len(plan['changes'])}", f"minimum_rerun_stages: {len(plan['minimum_rerun_stages'])}", "execution: dry-run only"]
    return "\n".join(lines)


def _report_payload(path: str) -> dict:
    value = json.loads(open(path, encoding="utf-8").read())
    return value.get("data", {}).get("quality", value.get("quality", value))


def _comparison_payload(path: str) -> dict:
    value = json.loads(open(path, encoding="utf-8").read())
    return value.get("data", {}).get("comparison", value.get("comparison", value))


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as error:
        return int(error.code)
    try:
        if args.command == "evaluate": report = evaluate_result(args.config, case_key=args.case_key, case_paths=args.case_paths, case_root=args.case_root, solver=args.solver, profile=args.profile)
        elif args.command == "recommend": report = recommend_from_diagnosis(json.loads(open(args.diagnosis, encoding="utf-8").read()), args.config, case_key=args.case_key, case_paths=args.case_paths, trial_id=args.trial_id, parameter_registry=args.parameter_registry, intervention_rules=args.intervention_rules)
        elif args.command == "compare": report = compare_quality_reports(_report_payload(args.baseline), _report_payload(args.candidate), profile=args.profile)
        elif args.command == "best":
            if args.best_command == "show": report = load_best(args.managed_root)
            elif args.best_command == "evaluate":
                from .agent.best import evaluate_best_candidate
                report = evaluate_best_candidate(args.candidate, managed_root=args.managed_root, profile=args.profile)
            else: report = update_best(_comparison_payload(args.comparison), candidate_quality=args.candidate, baseline_quality=args.baseline, managed_root=args.managed_root, expected_current_best_identity=args.expected_current_best_identity)
        elif args.command == "trial":
            if args.trial_command == "execute":
                payload = json.loads(open(args.plan, encoding="utf-8").read())
                plan = payload.get("data", {}).get("trial_plan", payload)
                report = execute_trial(plan, managed_root=args.managed_root, action_id=args.action)
            else:
                override = load_config(args.override) if args.override else {}
                report = plan_trial(args.config, override=override, case_key=args.case_key, case_paths=args.case_paths,
                                    case_root=args.case_root, solver=args.solver, trial_id=args.trial_id,
                                    restore_missing=args.restore_missing)
        elif args.command == "diagnose":
            if args.quality:
                from .agent.schemas import Envelope
                report = Envelope(status="ok", operation="diagnose.quality", data={"diagnosis": diagnose_quality_report(load_quality_report(args.quality)).to_dict()})
            else: report = diagnose_result(args.config, case_key=args.case_key, case_paths=args.case_paths, case_root=args.case_root, solver=args.solver, profile=args.profile)
        elif args.target == "artifact": report = inspect_artifact(args.path, case_root=args.case_root, artifact_type=args.artifact_type, artifact_schema=args.artifact_schema, producer_stage=args.producer_stage)
        else:
            func: Callable = {"case": inspect_case, "config": inspect_config, "pipeline": inspect_pipeline, "result": inspect_result}[args.target]
            report = func(args.config, case_key=args.case_key, case_paths=args.case_paths, case_root=args.case_root, solver=args.solver)
        payload = report.to_dict(); exit_code = 0
    except ControlPlaneError as error:
        payload = error_envelope(f"{args.command}.{getattr(args, 'target', 'result')}", error.record).to_dict()
        exit_code = {"SCHEMA.INVALID": 3, "FILESYSTEM.NOT_FOUND": 4, "ARTIFACT.INVALID": 4, "FILESYSTEM.OUTSIDE_ROOT": 5}.get(error.record.code, 8)
    except FileNotFoundError as error:
        payload = error_envelope(f"{args.command}.{getattr(args, 'target', 'result')}", ErrorRecord("FILESYSTEM.NOT_FOUND", "Path does not exist", True, path=str(error.filename or ""))).to_dict(); exit_code = 4
    except (OSError, ValueError, KeyError) as error:
        payload = error_envelope(f"{args.command}.{getattr(args, 'target', 'result')}", ErrorRecord("SCHEMA.INVALID", "Invalid request", True, details={"reason": str(error)})).to_dict(); exit_code = 3
    if getattr(args, "format", "json") == "json":
        sys.stdout.write(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n")
    else: sys.stdout.write(_text(payload) + "\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
