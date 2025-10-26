"""Pruebas de resiliencia que garantizan que el proceso de evaluación tolere entradas incompletas"""""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from services.evaluation_service import EvaluationService, ServiceConfig

from tests.test_pipeline_integration import (
    TRACKING_CLASSES,
    TrackingAIService,
    TrackingCleaner,
    TrackingLoader,
    TrackingPromptBuilder,
    TrackingPromptValidator,
    TrackingRepository,
    TrackingSplitter,
    _prepare_environment,
    sample_criteria,
)


@pytest.fixture
def resilience_env(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Configure a deterministic pipeline using tracking doubles."""

    _prepare_environment(monkeypatch)

    for cls in TRACKING_CLASSES:
        instances = getattr(cls, "instances", None)
        if isinstance(instances, list):
            for instance in list(instances):
                if hasattr(instance, "calls"):
                    getattr(instance, "calls").clear()  # type: ignore[attr-defined]
                if hasattr(instance, "exports"):
                    getattr(instance, "exports").clear()  # type: ignore[attr-defined]
            instances.clear()

    loader = TrackingLoader()
    cleaner = TrackingCleaner()
    splitter = TrackingSplitter()
    repository = TrackingRepository()
    prompt_builder = TrackingPromptBuilder()
    prompt_validator = TrackingPromptValidator()
    ai_service = TrackingAIService()

    service = EvaluationService(
        config=ServiceConfig(
            ai_provider="tracking",
            run_id="resilience-test",
            model_name="ceplan-mock",
            prompt_batch_size=1,
            retries=0,
            timeout_seconds=None,
            log_level="WARNING",
        ),
        loader=loader,
        cleaner=cleaner,
        splitter=splitter,
        repository=repository,
        ai_service_factory=lambda _: ai_service,
        prompt_builder=prompt_builder,
        prompt_validator=prompt_validator,
    )

    return SimpleNamespace(
        service=service,
        repository=repository,
        loader=loader,
        cleaner=cleaner,
        splitter=splitter,
        prompt_builder=prompt_builder,
        prompt_validator=prompt_validator,
        ai_service=ai_service,
    )


@pytest.fixture
def missing_section_document(tmp_path: Path) -> Path:
    path = tmp_path / "informe_incompleto.txt"
    path.write_text(
        "\n".join(
            [
                "[SECTION:resumen_ejecutivo]",
                "Contenido resumido disponible.",
            ]
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def empty_document(tmp_path: Path) -> Path:
    path = tmp_path / "informe_vacio.txt"
    path.write_text("Texto sin estructura relevante.", encoding="utf-8")
    return path


@pytest.fixture
def empty_section_document(tmp_path: Path) -> Path:
    path = tmp_path / "informe_seccion_vacia.txt"
    path.write_text(
        "\n".join(
            [
                "[SECTION:resumen_ejecutivo]",
                "",
                "[SECTION:gestion_ceplan]",
                "Contenido disponible en la segunda sección.",
            ]
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def policy_missing_blocks_document(tmp_path: Path) -> Path:
    path = tmp_path / "informe_politica_incompleto.txt"
    path.write_text(
        "\n".join(
            [
                "[SECTION:diagnostico]",
                "Diagnóstico disponible para evaluación.",
            ]
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def policy_blocks_criteria(tmp_path: Path) -> Path:
    path = tmp_path / "criterios_politica_bloques.json"
    payload = {
        "version": "ceplan-politica-resilience",
        "tipo_informe": "politica_nacional",
        "metodologia": "ponderada",
        "bloques": [
            {
                "id": "diagnostico",
                "titulo": "Diagnóstico",
                "ponderacion": 0.4,
                "preguntas": [
                    {
                        "id": "POL_DIAG",
                        "texto": "¿Cómo caracteriza el informe el diagnóstico del problema público?",
                        "ponderacion": 1.0,
                    }
                ],
            },
            {
                "id": "implementacion",
                "titulo": "Implementación",
                "ponderacion": 0.3,
                "preguntas": [
                    {
                        "id": "POL_IMPL",
                        "texto": "¿Cómo se implementan las intervenciones propuestas?",
                        "ponderacion": 1.0,
                    }
                ],
            },
            {
                "id": "seguimiento",
                "titulo": "Seguimiento",
                "ponderacion": 0.3,
                "preguntas": [
                    {
                        "id": "POL_SEG",
                        "texto": "¿Cómo se monitorean y evalúan los avances?",
                        "ponderacion": 1.0,
                    }
                ],
            },
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _load_exported_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["rows"]


def test_pipeline_handles_missing_sections(
    resilience_env: SimpleNamespace,
    missing_section_document: Path,
    sample_criteria: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    output_path = tmp_path / "resultado_incompleto.json"

    evaluation, metrics = resilience_env.service.run(
        input_path=missing_section_document,
        criteria_path=sample_criteria,
        output_path=output_path,
        output_format="json",
        extra_metadata={"scenario": "missing-section"},
    )

    assert len(resilience_env.ai_service.calls) == 1
    assert "gestión" in " ".join(record.message.lower() for record in caplog.records)

    summary = evaluation.metadata.get("segmenter_summary", {})
    assert summary["status_counts"]["missing"] == 1
    assert "gestion_ceplan" in summary["missing_sections"]

    missing_section = next(
        section for section in evaluation.sections if section.section_id == "gestion_ceplan"
    )
    skip_message = "Evaluación omitida por falta de sección"

    assert missing_section.score == pytest.approx(0.0)
    question = missing_section.dimensions[0].questions[0]
    assert missing_section.metadata.get("skipped") is True
    assert missing_section.metadata.get("skip_reason") == "missing_section"

    assert missing_section.dimensions, "La sección omitida debe conservar sus dimensiones"
    for dimension in missing_section.dimensions:
        assert dimension.score == pytest.approx(0.0)
        assert dimension.metadata.get("skipped") is True
        assert dimension.metadata.get("skip_reason") == "missing_section"
        assert dimension.questions, "Las dimensiones omitidas deben conservar sus preguntas"
        for question in dimension.questions:
            assert question.score == pytest.approx(0.0)
            assert question.justification == skip_message
            assert question.chunk_results == []
            assert question.metadata.get("skipped") is True
            assert question.metadata.get("skip_reason") == "missing_section"

    present_sections = [
        section for section in evaluation.sections if section.section_id != "gestion_ceplan"
    ]
    assert present_sections, "Debe existir al menos una sección evaluada normalmente"
    for section in present_sections:
        assert section.score is not None and section.score >= 0.0
        assert section.metadata.get("skipped") is not True
        for dimension in section.dimensions:
            assert dimension.score is not None and 0.0 <= dimension.score <= 4.0
            assert dimension.metadata.get("skipped") is not True
            for question in dimension.questions:
                assert question.score is not None and 0.0 <= question.score <= 4.0
                assert question.justification
                assert question.justification != skip_message
                assert question.metadata.get("skipped") is not True

    global_metrics = metrics["global"]
    normalized = global_metrics.get("normalized_score")
    assert normalized is not None
    assert math.isnan(float(normalized)) is False
    assert global_metrics["segmenter_flagged_sections"] == 1
    assert global_metrics["segmenter_flagged_breakdown"]["missing"] == 1

    assert output_path.exists()
    exported_rows = _load_exported_rows(output_path)
    exported_missing = [row for row in exported_rows if row["section_id"] == "gestion_ceplan"]
    assert exported_missing
    for row in exported_missing:
        assert row["section_score"] == 0
        assert row["dimension_score"] == 0
        assert row["question_score"] == 0
        assert row["justification"] == skip_message
        assert row.get("metadata.skipped") is True
        assert row.get("metadata.skip_reason") == "missing_section"

    exported_present = [row for row in exported_rows if row["section_id"] != "gestion_ceplan"]
    assert exported_present
    for row in exported_present:
        assert 0 <= row["section_score"] <= 4
        assert 0 <= row["dimension_score"] <= 4
        assert 0 <= row["question_score"] <= 4
        assert row["justification"] != skip_message 


def test_pipeline_handles_irrelevant_document(
    resilience_env: SimpleNamespace,
    empty_document: Path,
    sample_criteria: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    output_path = tmp_path / "resultado_vacio.json"

    evaluation, metrics = resilience_env.service.run(
        input_path=empty_document,
        criteria_path=sample_criteria,
        output_path=output_path,
        output_format="json",
        extra_metadata={"scenario": "empty-document"},
    )

    assert len(resilience_env.ai_service.calls) == 0
    messages = " ".join(record.message.lower() for record in caplog.records)
    assert "no se detectaron secciones válidas" in messages

    summary = evaluation.metadata.get("segmenter_summary", {})
    assert summary["status_counts"]["missing"] == 2
    assert summary["issues_detected"] is True

    global_metrics = metrics["global"]
    assert global_metrics["segmenter_flagged_sections"] == 2
    assert global_metrics["normalized_score"] == pytest.approx(0.0)

    exported_rows = _load_exported_rows(output_path)
    assert all(row["question_score"] == 0 for row in exported_rows)
    assert all(
        row["justification"] == "Evaluación omitida por falta de sección"
        for row in exported_rows
    )


def test_pipeline_handles_empty_sections(
    resilience_env: SimpleNamespace,
    empty_section_document: Path,
    sample_criteria: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    output_path = tmp_path / "resultado_seccion_vacia.json"

    evaluation, metrics = resilience_env.service.run(
        input_path=empty_section_document,
        criteria_path=sample_criteria,
        output_path=output_path,
        output_format="json",
        extra_metadata={"scenario": "empty-section"},
    )

    assert len(resilience_env.ai_service.calls) == 1
    messages = " ".join(record.message.lower() for record in caplog.records)
    assert "sin contenido" in messages

    summary = evaluation.metadata.get("segmenter_summary", {})
    assert summary["status_counts"]["empty"] == 1
    assert "resumen_ejecutivo" in summary["empty_sections"]

    empty_section = next(
        section for section in evaluation.sections if section.section_id == "resumen_ejecutivo"
    )
    question = empty_section.dimensions[0].questions[0]
    assert question.score == pytest.approx(0.0)
    assert question.justification == "Sin contenido para evaluar"
    assert question.metadata.get("skip_reason") == "empty_section"

    global_metrics = metrics["global"]
    assert global_metrics["segmenter_flagged_sections"] == 1
    breakdown = global_metrics["segmenter_flagged_breakdown"]
    assert breakdown["empty"] == 1

    exported_rows = _load_exported_rows(output_path)
    empty_rows = [row for row in exported_rows if row["section_id"] == "resumen_ejecutivo"]
    assert empty_rows
    assert empty_rows[0]["question_score"] == 0
    assert empty_rows[0]["justification"] == "Sin contenido para evaluar"


def test_policy_pipeline_handles_missing_blocks(
    resilience_env: SimpleNamespace,
    policy_missing_blocks_document: Path,
    policy_blocks_criteria: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    output_path = tmp_path / "resultado_politica_incompleto.json"

    evaluation, metrics = resilience_env.service.run(
        input_path=policy_missing_blocks_document,
        criteria_path=policy_blocks_criteria,
        output_path=output_path,
        output_format="json",
        extra_metadata={"scenario": "policy-missing-blocks"},
    )

    assert len(resilience_env.ai_service.calls) == 1
    messages = " ".join(record.message.lower() for record in caplog.records)
    assert "bloque" in messages
    assert "no encontrada" in messages

    summary = evaluation.metadata.get("segmenter_summary", {})
    assert summary["status_counts"]["missing"] == 2
    assert summary["issues_detected"] is True

    missing_ids = {section.section_id: section for section in evaluation.sections}
    assert missing_ids["implementacion"].score == pytest.approx(0.0)
    missing_question = missing_ids["implementacion"].dimensions[0].questions[0]
    assert missing_question.score == pytest.approx(0.0)
    assert missing_question.justification == "Evaluación omitida por falta de sección"
    assert missing_question.metadata.get("skip_reason") == "missing_section"

    global_metrics = metrics["global"]
    assert global_metrics["segmenter_flagged_sections"] == 2
    assert global_metrics["segmenter_flagged_breakdown"]["missing"] == 2
    normalized = global_metrics.get("normalized_score")
    assert normalized is not None
    assert math.isnan(float(normalized)) is False

    assert output_path.exists()
    exported_rows = _load_exported_rows(output_path)
    exported_missing = [
        row for row in exported_rows if row["section_id"] in {"implementacion", "seguimiento"}
    ]
    assert exported_missing
    assert all(row["question_score"] == 0 for row in exported_missing)
    assert all(
        row["justification"] == "Evaluación omitida por falta de sección"
        for row in exported_missing
    )