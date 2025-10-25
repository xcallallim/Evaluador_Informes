"""Tests for the evaluation repository exports."""

from io import StringIO
import json
from pathlib import Path

import pytest

from data.models.evaluation import (
    DimensionResult,
    EvaluationResult,
    QuestionResult,
    SectionResult,
)
from reporting.repository import EvaluationRepository, flatten_evaluation


def _sample_evaluation() -> EvaluationResult:
    question = QuestionResult(
        question_id="q1",
        text="Pregunta",
        score=0.8,
        weight=1.0,
    )
    dimension = DimensionResult(
        name="Dimension",
        score=0.8,
        weight=1.0,
        questions=[question],
    )
    section = SectionResult(
        section_id="s1",
        title="Sección",
        score=0.8,
        weight=1.0,
        dimensions=[dimension],
    )
    evaluation = EvaluationResult(
        document_id="doc-1",
        score=0.8,
        sections=[section],
        metadata={
            "criteria_version": "v1",
            "run_id": "run-123",
            "model_name": "gpt-4o-mini",
            "tipo_informe": "institucional",
            "pipeline_version": "svc-1",
            "timestamp": "2024-05-01T10:00:00Z",
        },
    )
    return evaluation


def test_flatten_includes_criteria_version() -> None:
    evaluation = _sample_evaluation()
    rows = flatten_evaluation(evaluation)
    assert rows
    assert rows[0]["criteria_version"] == "v1"
    assert rows[0]["tipo_informe"] == "institucional"
    assert rows[0]["model_name"] == "gpt-4o-mini"
    assert rows[0]["pipeline_version"] == "svc-1"
    assert rows[0]["timestamp"] == "2024-05-01T10:00:00Z"


def test_export_rejects_unsupported_format(tmp_path: Path) -> None:
    evaluation = _sample_evaluation()
    repository = EvaluationRepository()
    destination = tmp_path / "salida.test"

    with pytest.raises(ValueError):
        repository.export(
            evaluation,
            metrics_summary={"global": {}, "sections": []},
            output_path=destination,
            output_format="parquet",
        )

    assert not destination.exists()


def test_export_creates_excel_csv_and_json(tmp_path: Path) -> None:
    pandas = pytest.importorskip("pandas")
    openpyxl = pytest.importorskip("openpyxl")

    evaluation = _sample_evaluation()
    repository = EvaluationRepository()
    output_path = tmp_path / "resultado.xlsx"
    metrics = {
        "global": {"score": 0.8},
        "sections": [{"section_id": "s1", "score": 0.8}],
    }

    excel_path = repository.export(
        evaluation,
        metrics_summary=metrics,
        output_path=output_path,
    )

    assert excel_path == output_path

    csv_path = output_path.with_suffix(".csv")
    json_path = output_path.with_suffix(".json")

    assert output_path.exists()
    assert csv_path.exists()
    assert json_path.exists()

    workbook = openpyxl.load_workbook(output_path)
    assert set(workbook.sheetnames) == {
        "preguntas",
        "resumen",
        "indice_global",
    }

    preguntas_df = pandas.read_excel(output_path, sheet_name="preguntas")
    assert not preguntas_df.empty
    assert preguntas_df.loc[0, "document_id"] == "doc-1"

    header_df = pandas.read_excel(output_path, sheet_name="indice_global")
    assert header_df.loc[0, "run_id"] == "run-123"
    assert header_df.loc[0, "model_name"] == "gpt-4o-mini"

    csv_df = pandas.read_csv(csv_path)
    assert csv_df.loc[0, "question_id"] == "q1"

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["evaluation"]["metadata"]["run_id"] == "run-123"
    assert payload["metrics"]["global"]["score"] == 0.8
    assert payload["rows"][0]["document_id"] == "doc-1"


def test_export_without_permissions(monkeypatch, tmp_path: Path) -> None:
    evaluation = _sample_evaluation()
    repository = EvaluationRepository()
    restricted_dir = tmp_path / "readonly"
    restricted_dir.mkdir()
    output_path = restricted_dir / "resultado.json"
    metrics = {
        "global": {"score": 0.8},
        "sections": [{"section_id": "s1", "score": 0.8}],
    }

    def fake_write_text(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise PermissionError("sin permisos")

    monkeypatch.setattr(Path, "write_text", fake_write_text)

    with pytest.raises(PermissionError):
        repository.export(
            evaluation,
            metrics_summary=metrics,
            output_path=output_path,
            output_format="json",
        )

# pytest tests/test_repository.py -v