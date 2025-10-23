from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

import pytest

from data.models.document import Document
from data.models.evaluator import Evaluator, ModelResponse


@dataclass
class _FakeChunk:
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class FakeAIService:
    def __init__(self, responses: Iterable[ModelResponse]) -> None:
        self._responses = list(responses)
        self.prompts: List[str] = []
        self.arguments: List[Dict[str, Any]] = []

    def evaluate(self, prompt: str, **kwargs: Any) -> ModelResponse:
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
    assert result.metadata["criteria_version"] == "v1"
    assert section1.dimensions[0].questions[0].chunk_results[0].metadata == {}
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