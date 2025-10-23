"""Casos de prueba del evaluador para informes de política nacional."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence

import pytest

from data.models.document import Document
from data.models.evaluation import EvaluationResult
from data.models.evaluator import Evaluator, ModelResponse


@dataclass
class _FakeChunk:
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class FakeAIService:
    def __init__(self, responses: Iterable[Any]) -> None:
        self._responses = list(responses)
        self.prompts: List[str] = []
        self.arguments: List[Dict[str, Any]] = []

    def evaluate(self, prompt: str, **kwargs: Any) -> Any:
        if not self._responses:
            raise AssertionError(
                "No hay suficientes respuestas simuladas para el AIService."
            )
        self.prompts.append(prompt)
        self.arguments.append(kwargs)
        return self._responses.pop(0)


def _document_with_chunks(chunks: Sequence[tuple[str, Dict[str, Any] | None]]) -> Document:
    fake_chunks = [
        _FakeChunk(text, metadata or {})
        for text, metadata in chunks
    ]
    return Document(
        content="Documento base para política nacional",
        metadata={"id": "doc-politica", "title": "Política Nacional"},
        chunks=fake_chunks,
    )


def _base_politica_criteria() -> Dict[str, Any]:
    return {
        "tipo_informe": "politica_nacional",
        "version": "2024-test",
        "metodologia": "normalizada",
        "escala": {"min": 0, "max": 2},
        "bloques": [
            {
                "id": "B1",
                "titulo": "Bloque Estratégico",
                "ponderacion": 0.6,
                "preguntas": [
                    {"id": "P1", "texto": "Pregunta estratégica 1", "ponderacion": 0.4},
                    {"id": "P2", "texto": "Pregunta estratégica 2", "ponderacion": 0.6},
                ],
            },
            {
                "id": "B2",
                "titulo": "Bloque Operativo",
                "ponderacion": 0.4,
                "preguntas": [
                    {"id": "P3", "texto": "Pregunta operativa", "ponderacion": 1.0},
                ],
            },
        ],
    }


def _flatten_scores(criteria: Dict[str, Any], score_map: Dict[str, List[float]]) -> List[ModelResponse]:
    responses: List[ModelResponse] = []
    for bloque in criteria["bloques"]:
        for pregunta in bloque.get("preguntas", []):
            for idx, value in enumerate(score_map[pregunta["id"]], start=1):
                responses.append(
                    ModelResponse(
                        score=value,
                        justification=f"Justificación {pregunta['id']}-{idx}",
                    )
                )
    return responses


def test_evaluator_for_politica_nacional_multiple_chunks() -> None:
    document = _document_with_chunks(
        [
            ("Contenido del bloque principal", {"page": 1}),
            ("Contenido complementario", {"page": 2}),
        ]
    )
    criteria = _base_politica_criteria()

    chunk_scores = {
        "P1": [2.0, 1.0],
        "P2": [1.5, 0.5],
        "P3": [0.5, 1.5],
    }
    responses = _flatten_scores(criteria, chunk_scores)
    service = FakeAIService(responses)
    evaluator = Evaluator(service)

    result = evaluator.evaluate(document, criteria)

    expected_prompt_calls = sum(len(values) for values in chunk_scores.values())
    assert len(service.prompts) == expected_prompt_calls
    assert all("Sección: Bloque" in prompt for prompt in service.prompts)
    assert all("Pregunta:" in prompt for prompt in service.prompts)
    assert all("Fragmento:" in prompt for prompt in service.prompts)

    section_scores: Dict[str, float] = {}
    for section in result.sections:
        assert section.section_id in {"B1", "B2"}
        assert len(section.dimensions) == 1
        dimension = section.dimensions[0]
        assert dimension.questions
        weight_sum = 0.0
        weighted_sum = 0.0
        for question in dimension.questions:
            expected_scores = chunk_scores[question.question_id]
            assert [chunk.score for chunk in question.chunk_results] == expected_scores
            assert question.score == pytest.approx(sum(expected_scores) / len(expected_scores))
            for chunk in question.chunk_results:
                assert "ai_latency_ms" in chunk.metadata
                assert chunk.metadata.get("chunk_metadata", {}).get("page") in {1, 2}
            question_weight = question.weight or 1.0
            weight_sum += question_weight
            weighted_sum += question_weight * (question.score or 0.0)
        assert weight_sum > 0
        section_scores[section.section_id] = weighted_sum / weight_sum

    total_weight = sum(
        section.weight if section.weight is not None else 1.0 for section in result.sections
    )
    expected_total = sum(
        (section.weight if section.weight is not None else 1.0) * section_scores[section.section_id]
        for section in result.sections
    ) / total_weight
    assert result.score == pytest.approx(expected_total)


def test_politica_nacional_ignores_empty_blocks() -> None:
    document = _document_with_chunks([("Fragmento único", {"page": 1})])
    criteria = _base_politica_criteria()
    empty_block = {
        "id": "B_VACIO",
        "titulo": "Bloque sin preguntas",
        "ponderacion": 0.9,
        "preguntas": [],
    }
    criteria["bloques"].append(empty_block)

    responses = [
        ModelResponse(score=1.5, justification="J-P1"),
        ModelResponse(score=0.5, justification="J-P2"),
        ModelResponse(score=2.0, justification="J-P3"),
    ]
    service = FakeAIService(responses)
    evaluator = Evaluator(service)

    result = evaluator.evaluate(document, criteria)

    empty_section = next(section for section in result.sections if section.section_id == "B_VACIO")
    assert not empty_section.dimensions or not empty_section.dimensions[0].questions
    assert empty_section.score is None

    # Solo se consideran los bloques con puntajes válidos.
    scored_sections = [section for section in result.sections if section.score is not None]
    assert len(scored_sections) == 2
    total_weight = sum(section.weight or 1.0 for section in scored_sections)
    expected = sum((section.weight or 1.0) * section.score for section in scored_sections) / total_weight
    assert result.score == pytest.approx(expected)


def test_politica_nacional_normalises_irregular_weights() -> None:
    document = _document_with_chunks([("Fragmento base", {"page": 1})])
    criteria = _base_politica_criteria()
    criteria["bloques"][0]["ponderacion"] = 1.5
    criteria["bloques"][1]["ponderacion"] = 3.0
    criteria["bloques"][0]["preguntas"][0]["ponderacion"] = 3.0
    criteria["bloques"][0]["preguntas"][1]["ponderacion"] = 1.0
    criteria["bloques"][1]["preguntas"][0]["ponderacion"] = 2.5

    responses = [
        ModelResponse(score=2.0, justification="P1"),
        ModelResponse(score=0.0, justification="P2"),
        ModelResponse(score=1.0, justification="P3"),
    ]
    service = FakeAIService(responses)
    evaluator = Evaluator(service)

    result = evaluator.evaluate(document, criteria)

    section_scores: Dict[str, float] = {}
    for section in result.sections:
        dimension = section.dimensions[0]
        weighted = sum(
            (question.weight or 1.0) * (question.score or 0.0)
            for question in dimension.questions
        )
        weight_total = sum(question.weight or 1.0 for question in dimension.questions)
        section_scores[section.section_id] = weighted / weight_total

    expected_total = sum(
        (section.weight or 1.0) * section_scores[section.section_id]
        for section in result.sections
    ) / sum(section.weight or 1.0 for section in result.sections)
    assert result.score == pytest.approx(expected_total)


def test_politica_nacional_serialisation_roundtrip() -> None:
    document = _document_with_chunks([("Fragmento serializable", {"page": 5})])
    criteria = _base_politica_criteria()
    responses = [
        ModelResponse(score=2.0, justification="P1"),
        ModelResponse(score=1.0, justification="P2"),
        ModelResponse(score=1.5, justification="P3"),
    ]
    service = FakeAIService(responses)
    evaluator = Evaluator(service)

    result = evaluator.evaluate(document, criteria)
    payload = result.to_dict()
    clone = EvaluationResult.from_dict(payload)

    assert clone.to_dict() == payload
    assert clone.criteria_source == criteria["version"]
    assert clone.methodology == criteria["metodologia"]
    assert clone.document_type == "politica_nacional"
    assert clone.sections[0].dimensions[0].questions[0].chunk_results[0].metadata["chunk_metadata"]["page"] == 5
    assert result.metadata.get("methodology") == criteria["metodologia"]


def test_politica_nacional_handles_incomplete_ai_responses() -> None:
    document = _document_with_chunks([("Fragmento para respuestas incompletas", {"page": 9})])
    criteria = _base_politica_criteria()

    responses: List[Any] = [
        "Respuesta sin score",  # P1 -> score None, se conserva el texto como justificación.
        {"score": 1.0, "justification": "Respuesta parcial"},  # P2
        ModelResponse(score=2.0, justification="Respuesta completa"),  # P3
    ]
    service = FakeAIService(responses)
    evaluator = Evaluator(service)

    result = evaluator.evaluate(document, criteria)

    question_map = {
        question.question_id: question
        for section in result.sections
        for question in section.dimensions[0].questions
    }

    assert question_map["P1"].score is None
    assert "Respuesta sin score" in (question_map["P1"].justification or "")
    assert question_map["P2"].score == pytest.approx(1.0)
    assert question_map["P3"].score == pytest.approx(2.0)

    # Sólo las preguntas con puntajes válidos contribuyen al índice final.
    contributing_sections = [section for section in result.sections if section.score is not None]
    assert contributing_sections
    assert result.score is not None
    assert result.score <= 2.0