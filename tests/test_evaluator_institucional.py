"""Casos de prueba del evaluador para informes institucionales."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List

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
            raise AssertionError("No hay suficientes respuestas simuladas para el AIService.")
        self.prompts.append(prompt)
        self.arguments.append(kwargs)
        return self._responses.pop(0)


def _round(value: float) -> float:
    return round(value, 6)


def test_evaluator_generates_weighted_scores() -> None:
    document = Document(
        content="Resumen general",
        metadata={"id": "doc-123", "title": "Informe de prueba"},
        chunks=[
            _FakeChunk("Texto del chunk 1", {"page": 1}),
            _FakeChunk("Texto del chunk 2", {"page": 2}),
        ],
    )

    criteria: Dict[str, Any] = {
        "tipo_informe": "institucional",
        "version": "v1",
        "secciones": [
            {
                "titulo": "Sección 1",
                "ponderacion": 0.6,
                "dimensiones": [
                    {
                        "nombre": "Dimensión A",
                        "ponderacion": 0.7,
                        "metodo_agregacion": "promedio_ponderado",
                        "preguntas": [
                            {"id": "Q1", "texto": "Pregunta uno", "ponderacion": 0.4},
                            {"id": "Q2", "texto": "Pregunta dos", "ponderacion": 0.6},
                        ],
                    },
                    {
                        "nombre": "Dimensión B",
                        "ponderacion": 0.3,
                        "preguntas": [
                            {"id": "Q3", "texto": "Pregunta tres"},
                        ],
                    },
                ],
            },
            {
                "titulo": "Sección 2",
                "ponderacion": 0.4,
                "dimensiones": [
                    {
                        "nombre": "Dimensión C",
                        "preguntas": [
                            {"id": "Q4", "texto": "Pregunta cuatro"},
                        ],
                    }
                ],
            },
        ],
    }

    responses = [
        ModelResponse(score=1.0, justification="J1-1", relevant_text="R1-1"),
        ModelResponse(score=2.0, justification="J1-2", relevant_text="R1-2"),
        ModelResponse(score=3.0, justification="J2-1", relevant_text="R2-1"),
        ModelResponse(score=1.0, justification="J2-2", relevant_text="R2-2"),
        ModelResponse(score=2.0, justification="J3-1", relevant_text="R3-1"),
        ModelResponse(score=2.0, justification="J3-2", relevant_text="R3-2"),
        ModelResponse(score=3.0, justification="J4-1", relevant_text="R4-1"),
        ModelResponse(score=3.0, justification="J4-2", relevant_text="R4-2"),
    ]
    service = FakeAIService(responses)
    evaluator = Evaluator(service)

    result = evaluator.evaluate(document, criteria)

    # Se evalúa cada pregunta en ambos chunks.
    assert len(service.prompts) == 8
    assert all("Pregunta" in prompt for prompt in service.prompts)
    assert all("Sección:" in prompt for prompt in service.prompts)
    assert all("Dimensión:" in prompt for prompt in service.prompts)
    assert all("Fragmento:" in prompt for prompt in service.prompts)
    assert any("Metadatos del fragmento" in prompt for prompt in service.prompts)

    # Validar promedios ponderados.
    section1 = result.sections[0]
    section2 = result.sections[1]
    assert _round(section1.dimensions[0].questions[0].score) == pytest.approx(1.5)
    assert section1.dimensions[0].questions[0].justification == "J1-2"
    assert section1.dimensions[0].score == pytest.approx(1.8)
    assert section1.score == pytest.approx(1.86)
    assert section2.score == pytest.approx(3.0)
    assert result.score == pytest.approx(2.316)

    # Aseguramos que los metadatos básicos estén presentes.
    assert result.document_id == "doc-123"
    assert result.document_type == "institucional"
    assert result.criteria_source == "v1"
    assert result.methodology is None
    assert result.metadata["criteria_version"] == "v1"
    first_chunk_metadata = section1.dimensions[0].questions[0].chunk_results[0].metadata
    assert "ai_latency_ms" in first_chunk_metadata
    assert first_chunk_metadata["chunk_metadata"] == {"page": 1}
    assert section1.dimensions[0].questions[0].chunk_results[1].justification == "J1-2"

    # Serialización básica.
    serialised = result.to_dict()
    assert serialised["document_id"] == "doc-123"
    assert len(serialised["sections"]) == 2


def test_evaluator_handles_missing_scores() -> None:
    document = Document(content="Contenido sin chunks", metadata={"id": "doc-void"})

    criteria = {
        "tipo_informe": "institucional",
        "secciones": [
            {
                "titulo": "Única sección",
                "dimensiones": [
                    {
                        "nombre": "Única dimensión",
                        "preguntas": [
                            {"id": "Q1", "texto": "Pregunta sin respuesta"},
                            {"id": "Q2", "texto": "Pregunta con respuesta"},
                        ],
                    }
                ],
            }
        ],
    }

    responses = [
        ModelResponse(score=None, justification=None, relevant_text=None),
        ModelResponse(score=2.0, justification="J2", relevant_text="R2"),
    ]
    service = FakeAIService(responses)
    evaluator = Evaluator(service)

    result = evaluator.evaluate(document, criteria)

    dimension = result.sections[0].dimensions[0]
    assert dimension.questions[0].score is None
    assert dimension.questions[1].score == pytest.approx(2.0)
    assert dimension.score == pytest.approx(2.0)
    assert result.score == pytest.approx(2.0)
    # En ausencia de puntuaciones válidas se preservan las justificaciones nulas.
    assert dimension.questions[0].justification is None
    # Se usó un chunk sintético ya que el documento no tenía cortes predefinidos.
    assert len(service.prompts) == 2
    assert "Contenido sin chunks" in service.prompts[0]


def test_evaluator_uses_chunk_weights_when_available() -> None:
    document = Document(
        content="",
        metadata={"id": "doc-weighted"},
        chunks=[
            _FakeChunk("C1", {"relevance": 1}),
            _FakeChunk("C2", {"relevance": 3}),
        ],
    )

    criteria = {
        "tipo_informe": "institucional",
        "secciones": [
            {
                "titulo": "Única",
                "dimensiones": [
                    {
                        "nombre": "Única",
                        "preguntas": [
                            {"id": "Q1", "texto": "Peso"},
                        ],
                    }
                ],
            }
        ],
    }

    responses = [
        ModelResponse(score=1.0, justification="J-1"),
        ModelResponse(score=3.0, justification="J-2"),
    ]

    service = FakeAIService(responses)
    evaluator = Evaluator(service)

    result = evaluator.evaluate(document, criteria)

    question = result.sections[0].dimensions[0].questions[0]
    assert question.score == pytest.approx(2.5)
    weights = [chunk.metadata.get("weight") for chunk in question.chunk_results]
    assert weights == [1.0, 3.0]


def test_evaluator_handles_ai_failures_gracefully() -> None:
    class ExplodingAIService:
        def evaluate(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("boom")

    document = Document(content="Texto", metadata={"id": "doc-error"})
    criteria = {
        "tipo_informe": "institucional",
        "secciones": [
            {
                "titulo": "Sección",
                "dimensiones": [
                    {
                        "nombre": "Dimensión",
                        "preguntas": [
                            {"id": "Q1", "texto": "Pregunta"},
                        ],
                    }
                ],
            }
        ],
    }

    evaluator = Evaluator(ExplodingAIService())
    result = evaluator.evaluate(document, criteria)

    question = result.sections[0].dimensions[0].questions[0]
    assert question.score is None
    assert "error" in question.chunk_results[0].metadata
    assert "boom" in question.chunk_results[0].justification


def test_evaluator_normalises_unexpected_ai_payload() -> None:
    class OddAIService:
        def __init__(self) -> None:
            self.prompts: List[str] = []

        def evaluate(self, prompt: str, **_: Any) -> Any:
            self.prompts.append(prompt)
            return "respuesta libre sin formato"

    document = Document(content="Texto", metadata={"id": "doc-odd"})
    criteria = {
        "tipo_informe": "institucional",
        "secciones": [
            {
                "titulo": "Sección",
                "dimensiones": [
                    {
                        "nombre": "Dimensión",
                        "preguntas": [
                            {"id": "Q1", "texto": "Pregunta"},
                        ],
                    }
                ],
            }
        ],
    }

    evaluator = Evaluator(OddAIService())
    result = evaluator.evaluate(document, criteria)

    question = result.sections[0].dimensions[0].questions[0]
    assert question.score is None
    assert question.justification == "respuesta libre sin formato"


def test_evaluator_supports_global_blocks() -> None:
    document = Document(content="", metadata={"id": "doc-global"})
    criteria = {
        "tipo_informe": "politica_nacional",
        "metodologia": "global",
        "bloques": [
            {
                "id": "B1",
                "nombre": "Bloque 1",
                "ponderacion": 1,
                "preguntas": [
                    {"id": "Q1", "texto": "Pregunta global"},
                ],
            }
        ],
    }

    responses = [ModelResponse(score=4.0, justification="Justificación")]
    service = FakeAIService(responses)
    evaluator = Evaluator(service)

    result = evaluator.evaluate(document, criteria)

    assert result.methodology == "global"
    assert len(result.sections) == 1
    section = result.sections[0]
    assert section.section_id == "B1"
    assert section.dimensions[0].questions[0].score == pytest.approx(4.0)


def test_dimension_weighting_with_sparse_questions() -> None:
    document = Document(
        content="", metadata={"id": "doc-dim"}, chunks=[_FakeChunk("C1"), _FakeChunk("C2")]
    )

    criteria = {
        "tipo_informe": "institucional",
        "secciones": [
            {
                "titulo": "S1",
                "ponderacion": 1.0,
                "dimensiones": [
                    {
                        "nombre": "D1",
                        "ponderacion": 2,
                        "preguntas": [
                            {"id": "Q1", "texto": "P1", "ponderacion": 1},
                        ],
                    },
                    {
                        "nombre": "D2",
                        "ponderacion": 1,
                        "preguntas": [
                            {"id": "Q2", "texto": "P2"},
                        ],
                    },
                    {
                        "nombre": "D3",
                        "ponderacion": 3,
                        "preguntas": [],
                    },
                ],
            }
        ],
    }

    responses = [
        ModelResponse(score=1.0, justification="J1-c1"),
        ModelResponse(score=1.0, justification="J1-c2"),
        ModelResponse(score=5.0, justification="J2-c1"),
        ModelResponse(score=5.0, justification="J2-c2"),
    ]
    evaluator = Evaluator(FakeAIService(responses))

    result = evaluator.evaluate(document, criteria)

    section = result.sections[0]
    section.recompute_score()
    assert len(section.dimensions) == 3
    assert section.dimensions[2].score is None
    # Weighted average: ((2*1.0) + (1*5.0)) / (2+1) = 7/3
    assert section.score == pytest.approx(7 / 3)


def test_evaluation_result_generated_at_traces_timestamp() -> None:
    document = Document(content="", metadata={"id": "doc-time"})
    criteria = {"tipo_informe": "institucional", "secciones": []}

    evaluator = Evaluator(FakeAIService([]))
    before = datetime.utcnow() - timedelta(seconds=1)
    result = evaluator.evaluate(document, criteria)
    after = datetime.utcnow() + timedelta(seconds=1)

    assert before <= result.generated_at <= after
    payload = result.to_dict()
    assert isinstance(payload["generated_at"], str)
    reconstructed = EvaluationResult.from_dict(payload)
    assert isinstance(reconstructed.generated_at, datetime)
    assert abs((reconstructed.generated_at - result.generated_at).total_seconds()) < 1

# pytest tests/test_evaluator_institucional.py -v
