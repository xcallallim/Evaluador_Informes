"""Tests for the evaluation repository exports."""

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