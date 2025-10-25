"""Tests for the training export helpers."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from data.models.evaluation import DimensionResult, EvaluationResult, QuestionResult, SectionResult
from reporting.repository import EvaluationRepository
from utils.training_export import (
    REQUIRED_COLUMNS,
    TrainingDataset,
    TrainingExportError,
    load_latest_training_dataset,
    load_training_dataset,
)


def _evaluation() -> EvaluationResult:
    question = QuestionResult(
        question_id="Q1",
        text="¿Cuenta con un plan?",
        score=3.5,
        weight=1.0,
        justification="El plan está actualizado",
        relevant_text="El documento presenta un plan estratégico 2024-2026.",
    )
    dimension = DimensionResult(
        name="Planificación",
        score=3.5,
        weight=1.0,
        questions=[question],
    )
    section = SectionResult(
        section_id="S1",
        title="Gobernanza",
        score=3.5,
        weight=1.0,
        dimensions=[dimension],
    )
    return EvaluationResult(
        document_id="doc-001",
        score=3.5,
        sections=[section],
        metadata={
            "tipo_informe": "institucional",
            "model_name": "mock-model",
            "pipeline_version": "svc-1",
            "criteria_version": "v1",
            "run_id": "test-run",
            "timestamp": "2024-05-01T12:00:00Z",
        },
    )


def test_load_training_dataset_projects_schema(tmp_path: Path) -> None:
    pandas = pytest.importorskip("pandas")
    openpyxl = pytest.importorskip("openpyxl")
    evaluation = _evaluation()
    metrics = {
        "global": {"score": evaluation.score},
        "sections": [{"section_id": "S1", "score": 3.5}],
    }
    output_path = tmp_path / "export.xlsx"

    repository = EvaluationRepository()
    repository.export(
        evaluation,
        metrics_summary=metrics,
        output_path=output_path,
        output_format="xlsx",
    )

    dataset = load_training_dataset(output_path)
    dataset.validate()

    assert dataset.path == output_path
    assert list(dataset.dataframe.columns) == list(REQUIRED_COLUMNS)
    row = dataset.dataframe.iloc[0]
    assert row["tipo_informe"] == "institucional"
    assert row["seccion"] == "Gobernanza"
    assert row["criterio"] == "Planificación"
    assert row["pregunta"].startswith("¿Cuenta")
    assert row["texto"].startswith("El documento")
    assert row["score"] == pytest.approx(3.5)
    assert row["justificacion"].startswith("El plan")
    assert row["model_name"] == "mock-model"
    assert row["pipeline_version"] == "svc-1"
    assert row["criteria_version"] == "v1"
    assert row["timestamp"] == "2024-05-01T12:00:00Z"


def test_validate_detects_out_of_range_scores() -> None:
    dataframe = pd.DataFrame(
        {
            "tipo_informe": ["institucional"],
            "seccion": ["Gobernanza"],
            "criterio": ["Planificación"],
            "pregunta": ["¿Cuenta con un plan?"],
            "texto": ["Texto"],
            "score": [5.5],
            "justificacion": ["Fuera de rango"],
            "model_name": ["mock"],
            "pipeline_version": ["svc-1"],
            "criteria_version": ["v1"],
            "timestamp": ["2024-05-01"],
        }
    )
    dataset = TrainingDataset(path=Path("dummy.xlsx"), dataframe=dataframe)
    with pytest.raises(TrainingExportError):
        dataset.validate()


def test_load_latest_training_dataset_prefers_newest(tmp_path: Path) -> None:
    pandas = pytest.importorskip("pandas")
    openpyxl = pytest.importorskip("openpyxl")
    evaluation = _evaluation()
    metrics = {
        "global": {"score": evaluation.score},
        "sections": [{"section_id": "S1", "score": 3.5}],
    }
    repository = EvaluationRepository()

    first_path = tmp_path / "export_a.xlsx"
    second_path = tmp_path / "export_b.xlsx"

    repository.export(
        evaluation,
        metrics_summary=metrics,
        output_path=first_path,
        output_format="xlsx",
    )
    older_mtime = first_path.stat().st_mtime - 60
    os.utime(first_path, (older_mtime, older_mtime))

    repository.export(
        evaluation,
        metrics_summary=metrics,
        output_path=second_path,
        output_format="xlsx",
    )

    dataset = load_latest_training_dataset("export_*.xlsx", directory=tmp_path)
    assert dataset.path == second_path
    dataset.validate()