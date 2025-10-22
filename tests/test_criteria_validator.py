"""Pruebas para la validación de criterios CLI y registro."""

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

from utils.criteria_validator import validate_file
from utils.validators import VALIDATORS


@pytest.fixture(scope="module")
def criteria_dir() -> Path:
    return REPO_ROOT / "data" / "criteria"


def test_validators_registered() -> None:
    """Todos los tipos de informes esperados deben estar presentes en el registro."""

    assert "institucional" in VALIDATORS
    assert "politica_nacional" in VALIDATORS


def test_institutional_schema_passes(criteria_dir: Path) -> None:
    """La metodología institucional JSON debe validarse sin errores."""

    result = validate_file(criteria_dir / "metodologia_institucional.json")
    assert result.ok()
    assert not result.errors


def test_politica_nacional_schema_passes(criteria_dir: Path) -> None:
    """La metodología de política nacional JSON debe validarse sin errores."""

    result = validate_file(criteria_dir / "metodologia_politica_nacional.json")
    assert result.ok()
    assert not result.errors


def test_unknown_tipo_informe_emits_warning(tmp_path: Path) -> None:
    """Los tipos de informes desconocidos deberían generar una advertencia, pero no errores."""

    payload = {"tipo_informe": "desconocido"}
    file_path = tmp_path / "metodologia.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_file(file_path)
    assert result.ok()
    assert result.warnings


def test_cli_execution(criteria_dir: Path) -> None:
    """La CLI debería salir exitosamente al validar archivos conocidos."""

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


def test_cli_accepts_repo_relative_paths_from_external_cwd(tmp_path: Path) -> None:
    """Las rutas relativas deben resolverse incluso cuando se ejecutan desde fuera del repositorio."""

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