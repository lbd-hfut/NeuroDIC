"""Loop 2 structural inspection tests; no test runs a scientific solver."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from neurodic.agent.inspect import inspect_artifact, inspect_case, inspect_config, inspect_pipeline, inspect_result


ROOT = Path(__file__).resolve().parents[2]


def _snapshot(root: Path) -> list[tuple[str, int, int]]:
    return sorted((str(path.relative_to(root)), path.stat().st_size, path.stat().st_mtime_ns)
                  for path in root.rglob("*") if path.is_file())


@pytest.mark.parametrize(("config", "case_key", "solver"), [
    ("config/pin_2d.yaml", "pin_2d", "pin"),
    ("config/pin_stereo.yaml", "pin_stereo", "pin_stereo"),
    ("config/pin_multi.yaml", "pin_multi", "pin_multi"),
    ("config/ndef_multi.yaml", "ndef_multi", "ndef"),
])
def test_all_workflows_have_read_only_structural_inspection(config: str, case_key: str, solver: str) -> None:
    report = inspect_case(ROOT / config, case_key=case_key, case_paths=ROOT / "config/case_paths.yaml").to_dict()
    assert report["status"] == "ok"
    assert report["data"]["solver"] == solver
    assert report["data"]["config"]["effective_config"]["case"]
    assert report["data"]["stages"]
    assert report["data"]["capabilities"]["resume_supported"] is False


def test_effective_config_uses_current_case_mapping_not_readme() -> None:
    report = inspect_config(ROOT / "config/pin_multi.yaml", case_key="pin_multi", case_paths=ROOT / "config/case_paths.yaml").to_dict()
    config = report["data"]["effective_config"]
    assert config["fusion"]["enabled"] is True
    assert config["case"]["calibration"] == "result/calibration/calibration_result_scaled.json"


def test_missing_input_is_readiness_not_inspection_failure(tmp_path: Path) -> None:
    solver = tmp_path / "pin.yaml"; paths = tmp_path / "paths.yaml"; case = tmp_path / "case"
    solver.write_text("solver: pin\nmode: planar_2d\n", encoding="utf-8")
    paths.write_text(f"pin_2d:\n  case:\n    root: {case}\n    images_dir: .\n  output:\n    result: result/pin\n", encoding="utf-8")
    report = inspect_case(solver, case_key="pin_2d", case_paths=paths).to_dict()
    assert report["status"] == "ok"
    assert report["data"]["readiness"]["ready"] is False
    assert report["data"]["readiness"]["missing"][0]["artifact"] == "planar_image_series"


def test_artifact_containment_and_legacy_provenance() -> None:
    case = ROOT / "case" / "Multi" / "CylinderDIC"
    path = case / "result" / "pin_multi_slover" / "manifest.json"
    report = inspect_artifact(path, case_root=case, artifact_type="pin_multi_manifest", artifact_schema="json/v1", producer_stage="pin_multi.pair_solve").to_dict()
    artifact = report["data"]["artifact"]
    assert artifact["identity"]["strength"] == "metadata"
    assert artifact["provenance_status"] == "legacy_incomplete"
    assert not Path(artifact["location"]).is_absolute()
    with pytest.raises(Exception):
        inspect_artifact(ROOT / "README.md", case_root=case)


def test_pipeline_and_result_are_views_over_same_read_only_case_contract() -> None:
    arguments = (ROOT / "config/ndef_multi.yaml",)
    kwargs = {"case_key": "ndef_multi", "case_paths": ROOT / "config/case_paths.yaml"}
    pipeline = inspect_pipeline(*arguments, **kwargs).to_dict()
    result = inspect_result(*arguments, **kwargs).to_dict()
    assert pipeline["operation"] == "inspect.pipeline"
    assert result["operation"] == "inspect.result"
    assert pipeline["data"]["solver"] == result["data"]["solver"] == "ndef"


def test_inspection_does_not_mutate_case_tree() -> None:
    case = ROOT / "case" / "Multi" / "CylinderDIC"
    before = _snapshot(case)
    arguments = {"case_paths": ROOT / "config/case_paths.yaml"}
    inspect_case(ROOT / "config/pin_multi.yaml", case_key="pin_multi", **arguments)
    inspect_config(ROOT / "config/ndef_multi.yaml", case_key="ndef_multi", **arguments)
    inspect_pipeline(ROOT / "config/ndef_multi.yaml", case_key="ndef_multi", **arguments)
    inspect_result(ROOT / "config/pin_multi.yaml", case_key="pin_multi", **arguments)
    assert _snapshot(case) == before


def test_native_free_isolated_import_and_inspection(tmp_path: Path) -> None:
    code = "from neurodic.agent.inspect import inspect_case; print(inspect_case('config/pin_2d.yaml', case_key='pin_2d').to_dict()['data']['solver'])"
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "python")}
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=environment, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "pin"
    assert "Traceback" not in result.stderr


def test_cli_json_text_errors_and_help_are_machine_safe() -> None:
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "python")}
    base = [sys.executable, "-m", "neurodic.cli"]
    help_result = subprocess.run(base + ["--help"], cwd=ROOT, env=environment, capture_output=True, text=True)
    assert help_result.returncode == 0
    json_result = subprocess.run(base + ["inspect", "case", "--config", "config/ndef_multi.yaml", "--case-key", "ndef_multi", "--format", "json"], cwd=ROOT, env=environment, capture_output=True, text=True)
    assert json_result.returncode == 0 and json_result.stderr == ""
    payload = json.loads(json_result.stdout)
    assert payload["operation"] == "inspect.case" and payload["data"]["solver"] == "ndef"
    text_result = subprocess.run(base + ["inspect", "config", "--config", "config/pin_2d.yaml", "--case-key", "pin_2d", "--format", "text"], cwd=ROOT, env=environment, capture_output=True, text=True)
    assert text_result.returncode == 0 and "operation: inspect.config" in text_result.stdout
    bad = subprocess.run(base + ["inspect", "case", "--config", "missing.yaml"], cwd=ROOT, env=environment, capture_output=True, text=True)
    assert bad.returncode == 4
    assert json.loads(bad.stdout)["errors"][0]["code"] == "FILESYSTEM.NOT_FOUND"
