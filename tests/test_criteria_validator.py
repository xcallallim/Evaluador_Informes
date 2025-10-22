from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
import os

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import utils.criteria_validator as criteria_cli
from utils.criteria_validator import validate_file
from utils.validators import VALIDATORS


@pytest.fixture(scope="module")
def criteria_dir() -> Path:
    return REPO_ROOT / "data" / "criteria"


def test_validators_registered() -> None:
    """All expected report types should be present in the registry."""

    assert "institucional" in VALIDATORS
    assert "politica_nacional" in VALIDATORS


def test_institutional_schema_passes(criteria_dir: Path) -> None:
    """The institutional methodology JSON should validate without errors."""

    result = validate_file(criteria_dir / "metodologia_institucional.json")
    assert result.ok()
    assert not result.errors


def test_politica_nacional_schema_passes(criteria_dir: Path) -> None:
    """The national policy methodology JSON should validate without errors."""

    result = validate_file(criteria_dir / "metodologia_politica_nacional.json")
    assert result.ok()
    assert not result.errors


def test_unknown_tipo_informe_emits_warning(tmp_path: Path) -> None:
    """Unknown report types should produce a warning but no errors."""

    payload = {"tipo_informe": "desconocido"}
    file_path = tmp_path / "metodologia.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_file(file_path)
    assert result.ok()
    assert result.warnings


def test_cli_execution(criteria_dir: Path) -> None:
    """The CLI should exit successfully when validating known files."""

    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "utils.criteria_validator",
            str(criteria_dir / "metodologia_institucional.json"),
            str(criteria_dir / "metodologia_politica_nacional.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert process.returncode == 0, process.stdout + process.stderr


def test_cli_output_falls_back_to_ascii(monkeypatch) -> None:
    """Unicode symbols should degrade to ASCII when stdout cannot encode them."""

    result = criteria_cli.ValidationResult()
    result.errors.append("Error de prueba")
    result.warnings.append("Advertencia de prueba")

    monkeypatch.setattr(criteria_cli, "_stdout_supports", lambda symbol: False)

    lines = criteria_cli._format_output(Path("archivo.json"), result)

    assert "[ERROR]" in lines[1]
    assert "[WARN]" in lines[-2]


def test_stdout_supports_returns_false_when_encoding_missing(monkeypatch) -> None:
    """If stdout encoding is undefined, the helper should disable Unicode symbols."""

    class DummyStdout:
        encoding = None

    class DummySys:
        stdout = DummyStdout()

    monkeypatch.setattr(criteria_cli, "sys", DummySys())

    assert not criteria_cli._stdout_supports("✔")


def test_cli_accepts_repo_relative_paths_from_external_cwd(tmp_path: Path) -> None:
    """Relative paths should be resolved even when running from outside the repo."""

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    repo_pythonpath = str(REPO_ROOT)
    env["PYTHONPATH"] = (
        f"{repo_pythonpath}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else repo_pythonpath
    )

    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "utils.criteria_validator",
            "data/criteria/metodologia_institucional.json",
            "data/criteria/metodologia_politica_nacional.json",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
        env=env,
    )

    assert process.returncode == 0, process.stdout + process.stderr

if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))

# py tests/test_criteria_validator.py
# python -m tests.test_criteria_validator