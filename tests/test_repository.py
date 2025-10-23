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
        },
    )
    return evaluation


def test_flatten_includes_criteria_version() -> None:
    evaluation = _sample_evaluation()
    rows = flatten_evaluation(evaluation)
    assert rows
    assert rows[0]["criteria_version"] == "v1"


def test_export_rejects_unknown_format(tmp_path: Path) -> None:
    evaluation = _sample_evaluation()
    repository = EvaluationRepository()
    destination = tmp_path / "salida.test"

    with pytest.raises(ValueError):
        repository.export(
            evaluation,
            metrics_summary={"global": {}, "sections": []},
            output_path=destination,
            output_format="desconocido",
        )

    assert not destination.exists()


def test_export_json_uses_stringio(monkeypatch, tmp_path: Path) -> None:
    """Ensure JSON exports can be captured in-memory for validation."""

    evaluation = _sample_evaluation()
    repository = EvaluationRepository()
    output_path = tmp_path / "resultado.json"
    buffer = StringIO()

    def fake_write_text(self, data: str, encoding: str = "utf-8", errors: str | None = None):
        del encoding, errors
        buffer.seek(0)
        buffer.truncate(0)
        buffer.write(data)
        return len(data)

    monkeypatch.setattr(Path, "write_text", fake_write_text)

    result_path = repository.export(
        evaluation,
        metrics_summary={"global": {"score": 0.8}, "sections": []},
        output_path=output_path,
        output_format="json",
    )

    assert result_path == output_path
    assert not output_path.exists()

    payload = json.loads(buffer.getvalue())
    assert payload["evaluation"]["document_id"] == "doc-1"
    assert payload["metrics"]["global"]["score"] == 0.8


def test_export_json_success(tmp_path: Path) -> None:
    """Exercise the full JSON export pipeline writing to disk."""

    evaluation = _sample_evaluation()
    repository = EvaluationRepository()
    output_path = tmp_path / "resultado.json"
    metrics = {
        "global": {"score": 0.8},
        "sections": [{"section_id": "s1", "score": 0.8}],
    }

    written = repository.export(
        evaluation,
        metrics_summary=metrics,
        output_path=output_path,
        output_format="json",
    )

    assert written == output_path
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["evaluation"]["metadata"]["run_id"] == "run-123"
    assert payload["evaluation"]["metadata"]["model_name"] == "gpt-4o-mini"
    assert payload["metrics"]["global"]["score"] == 0.8


@pytest.mark.parametrize("output_format", ["csv", "parquet"])
def test_export_pandas_formats_in_memory(monkeypatch, tmp_path: Path, output_format: str) -> None:
    """Use StringIO to validate tabular formats without touching disk."""

    pandas = pytest.importorskip("pandas")

    evaluation = _sample_evaluation()
    repository = EvaluationRepository()
    output_path = tmp_path / f"resultado.{output_format}"
    metrics = {
        "global": {"score": 0.8},
        "sections": [{"section_id": "s1", "score": 0.8}],
    }

    buffer = StringIO()

    buffer_bytes: list[bytes] = []

    if output_format == "csv":
        original_to_csv = pandas.DataFrame.to_csv

        def fake_to_csv(self, path_or_buf=None, *args, **kwargs):  # type: ignore[no-untyped-def]
            # Redirect CSV serialization into memory for assertions.
            return original_to_csv(self, buffer, *args, **kwargs)

        monkeypatch.setattr(pandas.DataFrame, "to_csv", fake_to_csv)
    else:
        pytest.importorskip("pyarrow")
        original_to_parquet = pandas.DataFrame.to_parquet

        def fake_to_parquet(self, path, *args, **kwargs):  # type: ignore[no-untyped-def]
            from io import BytesIO

            bio = BytesIO()
            original_to_parquet(self, bio, *args, **kwargs)
            buffer_bytes.append(bio.getvalue())

        monkeypatch.setattr(pandas.DataFrame, "to_parquet", fake_to_parquet)

    repository.export(
        evaluation,
        metrics_summary=metrics,
        output_path=output_path,
        output_format=output_format,
    )

    if output_format == "csv":
        assert "document_id" in buffer.getvalue()
    else:
        # Ensure parquet serialization produced some bytes in memory.
        assert any(buffer_bytes)


def test_export_xlsx_creates_expected_sheets(tmp_path: Path) -> None:
    pandas = pytest.importorskip("pandas")
    openpyxl = pytest.importorskip("openpyxl")

    evaluation = _sample_evaluation()
    repository = EvaluationRepository()
    output_path = tmp_path / "resultado.xlsx"
    metrics = {
        "global": {"score": 0.8},
        "sections": [{"section_id": "s1", "score": 0.8}],
    }

    repository.export(
        evaluation,
        metrics_summary=metrics,
        output_path=output_path,
        output_format="xlsx",
    )

    workbook = openpyxl.load_workbook(output_path)
    assert set(workbook.sheetnames) == {
        "preguntas",
        "resumen",
        "indice_global",
    }
    header_df = pandas.read_excel(output_path, sheet_name="indice_global")
    assert header_df.loc[0, "run_id"] == "run-123"
    assert header_df.loc[0, "model_name"] == "gpt-4o-mini"


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