"""Integration test exercising the full evaluation pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from services.evaluation_service import EvaluationService, MockAIService, ServiceConfig


@pytest.fixture
def deterministic_ai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force MockAIService to emit deterministic scores during the test."""

    def _deterministic_evaluate(self: MockAIService, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        question = kwargs.get("question") or {}
        levels = question.get("niveles", [])
        max_score = 4.0
        for level in levels:
            if isinstance(level, dict) and "valor" in level:
                try:
                    candidate = float(level["valor"])
                except (TypeError, ValueError):
                    continue
                max_score = max(max_score, candidate)
        score = round(0.78 * max_score, 2)
        return {
            "score": score,
            "justification": "Respuesta generada por MockAIService determinista.",
            "relevant_text": "Fragmento seleccionado",
            "metadata": {
                "model": self.model_name,
                "mock": True,
                "deterministic": True,
            },
        }

    monkeypatch.setattr(MockAIService, "evaluate", _deterministic_evaluate)


def _write_sample_document(target: Path) -> None:
    content = "\n".join(
        [
            "=== PAGE 1 ===",
            "Resumen Ejecutivo",
            "El informe resume los principales logros y retos del periodo evaluado.",
            "Incluye hallazgos clave para la toma de decisiones.",
            "",
            "=== PAGE 2 ===",
            "Prioridades de la política institucional",
            "La entidad prioriza el fortalecimiento de capacidades y servicios.",
            "Detalla iniciativas estratégicas para el siguiente año.",
        ]
    )
    target.write_text(content, encoding="utf-8")


def _write_criteria(target: Path) -> Dict[str, Any]:
    criteria: Dict[str, Any] = {
        "version": "integracion-v1",
        "tipo_informe": "institucional",
        "metodologia": "ponderada",
        "metrica_global": {"escala_resultado": {"min": 0.0, "max": 4.0}},
        "secciones": [
            {
                "id": "resumen_ejecutivo",
                "titulo": "Resumen Ejecutivo",
                "ponderacion": 0.6,
                "dimensiones": [
                    {
                        "nombre": "Contexto General",
                        "ponderacion": 1.0,
                        "metodo_agregacion": "promedio_ponderado",
                        "preguntas": [
                            {
                                "id": "P1",
                                "texto": "¿Cuál es el panorama general del informe?",
                                "ponderacion": 0.5,
                                "niveles": [{"valor": 4.0}, {"valor": 2.0}],
                            },
                            {
                                "id": "P2",
                                "texto": "¿Qué elementos se destacan en el resumen ejecutivo?",
                                "ponderacion": 0.5,
                                "niveles": [{"valor": 4.0}],
                            },
                        ],
                    }
                ],
            },
            {
                "id": "prioridades_politica_institucional",
                "titulo": "Prioridades de la política institucional",
                "ponderacion": 0.4,
                "dimensiones": [
                    {
                        "nombre": "Planificación",
                        "ponderacion": 1.0,
                        "preguntas": [
                            {
                                "id": "P3",
                                "texto": "¿Cómo se gestionan las prioridades institucionales?",
                                "ponderacion": 1.0,
                                "niveles": [{"valor": 4.0}],
                            }
                        ],
                    }
                ],
            },
        ],
    }
    target.write_text(json.dumps(criteria, ensure_ascii=False, indent=2), encoding="utf-8")
    return criteria


@pytest.mark.usefixtures("deterministic_ai")
def test_full_pipeline_execution(tmp_path: Path) -> None:
    document_path = tmp_path / "documento_institucional.txt"
    criteria_path = tmp_path / "criterios.json"
    output_path = tmp_path / "resultado.json"
    second_output_path = tmp_path / "resultado_segundo.json"

    _write_sample_document(document_path)
    criteria = _write_criteria(criteria_path)

    config = ServiceConfig(
        ai_provider="mock",
        run_id="pipeline-integration",
        model_name="mock-integration",
        prompt_batch_size=1,
        retries=0,
        timeout_seconds=15.0,
        log_level="ERROR",
    )
    service = EvaluationService(config=config)

    evaluation, metrics = service.run(
        input_path=document_path,
        criteria_path=criteria_path,
        output_path=output_path,
        output_format="json",
        extra_metadata={"invocation": "integration-test"},
    )

    assert output_path.exists()
    exported = json.loads(output_path.read_text(encoding="utf-8"))

    assert evaluation.document_id == document_path.stem
    assert evaluation.document_type == "institucional"
    assert evaluation.metadata["criteria_version"] == criteria["version"]
    assert evaluation.metadata["model_name"] == "mock-integration"
    assert evaluation.metadata["run_id"] == "pipeline-integration"
    assert evaluation.metadata["mode"] == "completo"
    assert evaluation.metadata["prompt_batch_size"] == 1
    assert evaluation.metadata["retries"] == 0
    assert evaluation.metadata["timeout_seconds"] == 15.0
    assert evaluation.metadata["runs"][-1]["model"] == "mock-integration"
    assert evaluation.metadata["source_path"] == str(document_path)

    assert len(evaluation.sections) == 2
    section_ids = [section.section_id for section in evaluation.sections]
    assert section_ids == ["resumen_ejecutivo", "prioridades_politica_institucional"]
    first_section = evaluation.sections[0]
    second_section = evaluation.sections[1]
    assert pytest.approx(first_section.weight, rel=1e-6) == 0.6
    assert pytest.approx(second_section.weight, rel=1e-6) == 0.4

    for section in evaluation.sections:
        for dimension in section.dimensions:
            assert dimension.questions, "Cada dimensión debe contener preguntas evaluadas"
            for question in dimension.questions:
                assert question.score is not None
                assert 0.0 <= question.score <= 4.0
                assert question.chunk_results, "Cada pregunta debe incluir resultados por chunk"
                chunk_sources: List[str] = []
                for chunk in question.chunk_results:
                    assert chunk.score is not None and 0.0 <= chunk.score <= 4.0
                    metadata = chunk.metadata
                    assert metadata.get("mock") is True
                    assert metadata.get("model") == "mock-integration"
                    # chunk_metadata preserves segmentation + cleaning report
                    chunk_meta = metadata.get("chunk_metadata") or {}
                    document_meta = chunk_meta.get("document_metadata") or {}
                    assert "cleaning_report" in document_meta
                    if "source_id" in chunk_meta:
                        chunk_sources.append(str(chunk_meta["source_id"]))
                # At least one chunk should come from the section that defines the question
                assert section.section_id in chunk_sources

    assert evaluation.score is not None
    expected_normalised = (evaluation.score / 4.0) * 100.0
    assert metrics["methodology"] == "institucional"
    assert metrics["global"]["raw_score"] == pytest.approx(evaluation.score)
    assert metrics["global"]["normalized_score"] == pytest.approx(expected_normalised)
    assert metrics["totals"]["sections_total"] == 2
    assert metrics["sections"][0]["section_id"] == "resumen_ejecutivo"
    assert metrics["sections"][1]["section_id"] == "prioridades_politica_institucional"

    assert exported["evaluation"]["metadata"]["run_id"] == "pipeline-integration"
    assert exported["evaluation"]["metadata"]["criteria_version"] == criteria["version"]
    assert exported["metrics"] == metrics
    assert exported["extra"]["config"]["run_id"] == "pipeline-integration"
    assert exported["extra"]["tipo_informe"] == "institucional"
    assert exported["extra"]["criteria_version"] == criteria["version"]
    assert exported["extra"]["invocation"] == "integration-test"

    evaluation_repeat, metrics_repeat = service.run(
        input_path=document_path,
        criteria_path=criteria_path,
        output_path=second_output_path,
        output_format="json",
        extra_metadata={"invocation": "integration-test"},
    )

    assert second_output_path.exists()
    assert evaluation_repeat.score == pytest.approx(evaluation.score)
    repeat_section_scores = [section.score for section in evaluation_repeat.sections]
    original_section_scores = [section.score for section in evaluation.sections]
    assert repeat_section_scores == pytest.approx(original_section_scores)  # type: ignore[arg-type]
    assert metrics_repeat == metrics