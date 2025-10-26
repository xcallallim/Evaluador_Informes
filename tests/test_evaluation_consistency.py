from __future__ import annotations

import csv
import hashlib
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pytest

from data.models.document import Document
import data.models.evaluation as evaluation_models
from services.evaluation_service import (
    EvaluationFilters,
    EvaluationService,
    ServiceConfig,
)
import services.evaluation_service as evaluation_module
from services.ai_service import MockAIService
import services.ai_service as ai_module


@dataclass(slots=True)
class _FakeChunk:
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class SeededMockAIService(MockAIService):
    """MockAIService variant with deterministic metadata and seeded scores."""

    def __init__(self, *args: Any, seed: int = 0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._seed = seed

    def evaluate(
        self, prompt: Optional[str] = None, **kwargs: Any
    ) -> Mapping[str, Any]:
        result = super().evaluate(prompt, **kwargs)
        payload = dict(result)
        metadata = dict(payload.get("metadata", {}))
        digest = hashlib.sha1((prompt or "").encode("utf-8")).hexdigest()
        metadata["request_id"] = f"seed-{self._seed}-{digest[:12]}"
        metadata["timestamp"] = "2024-01-01T00:00:00+00:00"
        payload["metadata"] = metadata
        return payload

    def _deterministic_score(
        self, prompt: str, levels: Iterable[float]
    ) -> float:  # pragma: no cover - exercised via evaluate
        digest = hashlib.sha256((prompt + str(self._seed)).encode("utf-8")).digest()
        base = int.from_bytes(digest[:4], "big")
        candidates = list(levels)
        if candidates:
            index = base % len(candidates)
            return float(candidates[index])
        fallback = [0.0, 1.0, 2.0, 3.0, 4.0]
        return float(fallback[base % len(fallback)])


def _build_document(*, version: str = "base") -> Document:
    intro_text = (
        "La introducción detalla los objetivos estratégicos y evidencia avances tangibles."
    )
    analysis_text = (
        "El análisis describe la implementación con métricas, riesgos mitigados y resultados sostenidos."
    )
    if version == "modificado":
        analysis_text = (
            "El análisis actualizado enfatiza brechas críticas, dificultades pendientes y ausencia de resultados."
        )

    document = Document(
        content=f"{intro_text}\n{analysis_text}",
        metadata={"id": "DOC-CONSISTENCIA", "title": "Informe de Consistencia"},
        sections={"introduccion": intro_text, "analisis": analysis_text},
    )
    document.chunks = [
        _FakeChunk(intro_text, {"section_id": "introduccion", "page": 1}),
        _FakeChunk(analysis_text, {"section_id": "analisis", "page": 2}),
    ]
    return document


def _levels() -> List[Dict[str, float]]:
    return [{"valor": float(value)} for value in (0, 1, 2, 3, 4)]


def _build_criteria() -> Dict[str, Any]:
    return {
        "version": "consistencia-v1",
        "tipo_informe": "institucional",
        "metodologia": "ponderada",
        "secciones": [
            {
                "id": "introduccion",
                "titulo": "Introducción",
                "ponderacion": 0.5,
                "dimensiones": [
                    {
                        "nombre": "Contexto",
                        "ponderacion": 1.0,
                        "preguntas": [
                            {
                                "id": "Q1",
                                "texto": "¿Se explican los objetivos estratégicos?",
                                "ponderacion": 1.0,
                                "niveles": _levels(),
                            },
                            {
                                "id": "Q2",
                                "texto": "¿Se detallan los avances logrados?",
                                "ponderacion": 1.0,
                                "niveles": _levels(),
                            },
                            {
                                "id": "Q3",
                                "texto": "¿Se identifican desafíos relevantes?",
                                "ponderacion": 1.0,
                                "niveles": _levels(),
                            },
                        ],
                    }
                ],
            },
            {
                "id": "analisis",
                "titulo": "Análisis",
                "ponderacion": 0.5,
                "dimensiones": [
                    {
                        "nombre": "Implementación",
                        "ponderacion": 1.0,
                        "preguntas": [
                            {
                                "id": "Q4",
                                "texto": "¿Se describen los procesos implementados?",
                                "ponderacion": 1.0,
                                "niveles": _levels(),
                            },
                            {
                                "id": "Q5",
                                "texto": "¿Se documentan los riesgos y mitigaciones?",
                                "ponderacion": 1.0,
                                "niveles": _levels(),
                            },
                            {
                                "id": "Q6",
                                "texto": "¿Se presentan resultados sostenidos?",
                                "ponderacion": 1.0,
                                "niveles": _levels(),
                            },
                        ],
                    }
                ],
            },
        ],
    }


@pytest.fixture(autouse=True)
def _freeze_time(monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import datetime as real_datetime

    class _FrozenDateTime(real_datetime):
        @classmethod
        def utcnow(cls) -> "_FrozenDateTime":
            return cls(2024, 1, 1, 0, 0, 0)

        @classmethod
        def now(cls, tz=None) -> "_FrozenDateTime":  # pragma: no cover - compatibility
            base = cls(2024, 1, 1, 0, 0, 0)
            if tz is not None:
                return base.replace(tzinfo=tz)
            return base

    monkeypatch.setattr(evaluation_models, "datetime", _FrozenDateTime)
    monkeypatch.setattr(evaluation_module, "datetime", _FrozenDateTime)
    monkeypatch.setattr(ai_module, "datetime", _FrozenDateTime)

    class _FakeTime:
        def __init__(self) -> None:
            self._counter = 1000.0

        def perf_counter(self) -> float:
            self._counter += 1.0
            return self._counter

        def sleep(self, seconds: float) -> None:  # pragma: no cover - compatibility
            self._counter += float(seconds)

    fake_time = _FakeTime()
    monkeypatch.setattr(evaluation_module.time, "perf_counter", fake_time.perf_counter)
    monkeypatch.setattr(evaluation_module.time, "sleep", fake_time.sleep)


@pytest.fixture
def evaluation_service_factory() -> Any:
    def _factory(*, run_id: str, seed: int = 0) -> EvaluationService:
        config = ServiceConfig(run_id=run_id, model_name="mock-seeded")

        def _factory_inner(_: ServiceConfig) -> SeededMockAIService:
            return SeededMockAIService(model_name=config.model_name, seed=seed)

        return EvaluationService(config=config, ai_service_factory=_factory_inner)

    return _factory


def _flatten_question_scores(entity: Any) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    if hasattr(entity, "sections"):
        sections = entity.sections  # EvaluationResult
    elif hasattr(entity, "dimensions"):
        sections = [entity]  # SectionResult
    else:  # pragma: no cover - defensive guard
        raise TypeError("Entidad desconocida para extracción de puntajes.")

    for section in sections:
        for dimension in section.dimensions:
            for question in dimension.questions:
                if question.score is not None:
                    scores[question.question_id] = float(question.score)
    return scores


def _pearson_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or not x:
        raise ValueError("Se requieren secuencias del mismo tamaño para correlación.")
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    den_x = sum((a - mean_x) ** 2 for a in x)
    den_y = sum((b - mean_y) ** 2 for b in y)
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x ** 0.5 * den_y ** 0.5)


def _normalise_evaluation_dict(payload: Mapping[str, Any]) -> Dict[str, Any]:
    data = copy.deepcopy(dict(payload))
    data["generated_at"] = "2024-01-01T00:00:00"
    metadata = data.get("metadata")
    if isinstance(metadata, dict):
        metadata["timestamp"] = "2024-01-01T00:00:00"
        runs = metadata.get("runs")
        if isinstance(runs, list):
            for entry in runs:
                if isinstance(entry, dict) and "executed_at" in entry:
                    entry["executed_at"] = "2024-01-01T00:00:00"
        performance = metadata.get("performance")
        if isinstance(performance, dict):
            performance["execution_time_seconds"] = 0.0
            if "memory_current_mib" in performance:
                performance["memory_current_mib"] = 0.0
            if "memory_peak_mib" in performance:
                performance["memory_peak_mib"] = 0.0
            if "memory_delta_mib" in performance:
                performance["memory_delta_mib"] = 0.0
    return data


def test_internal_reproducibility_three_runs(
    evaluation_service_factory: Any, tmp_path: Path
) -> None:
    service = evaluation_service_factory(run_id="run-repro", seed=11)
    results: List[Dict[str, Any]] = []
    for index in range(3):
        evaluation, _ = service.run(
            document=_build_document(),
            criteria_data=_build_criteria(),
            mode="global",
            output_path=tmp_path / f"repro_{index}.json",
            output_format="json",
        )
        results.append(_normalise_evaluation_dict(evaluation.to_dict()))

    assert results[0] == results[1] == results[2]


def test_section_sensitivity_partial_updates(
    evaluation_service_factory: Any, tmp_path: Path
) -> None:
    baseline_service = evaluation_service_factory(run_id="run-base", seed=7)
    baseline_evaluation, _ = baseline_service.run(
        document=_build_document(),
        criteria_data=_build_criteria(),
        mode="global",
        output_path=tmp_path / "baseline.json",
        output_format="json",
    )

    updated_service = evaluation_service_factory(run_id="run-partial", seed=7)
    partial_filters = EvaluationFilters(section_ids=["analisis"])
    partial_evaluation, _ = updated_service.run(
        document=_build_document(version="modificado"),
        criteria_data=_build_criteria(),
        mode="parcial",
        filters=partial_filters,
        previous_result=baseline_evaluation,
        output_path=tmp_path / "partial.json",
        output_format="json",
    )

    merged = updated_service._merge_results(baseline_evaluation, partial_evaluation)

    def _section_by_id(result: Any, section_id: str) -> Any:
        for section in result.sections:
            if section.section_id == section_id:
                return section
        raise AssertionError(f"Sección {section_id} no encontrada")

    baseline_intro = _section_by_id(baseline_evaluation, "introduccion")
    merged_intro = _section_by_id(merged, "introduccion")
    assert merged_intro.score == baseline_intro.score

    baseline_analysis = _section_by_id(baseline_evaluation, "analisis")
    merged_analysis = _section_by_id(merged, "analisis")

    assert merged_analysis.score != baseline_analysis.score

    baseline_other_scores = _flatten_question_scores(baseline_intro)
    merged_other_scores = _flatten_question_scores(merged_intro)
    assert merged_other_scores == baseline_other_scores


def test_dimension_average_matches_question_scores(
    evaluation_service_factory: Any, tmp_path: Path
) -> None:
    service = evaluation_service_factory(run_id="run-coherencia", seed=5)
    evaluation, _ = service.run(
        document=_build_document(),
        criteria_data=_build_criteria(),
        mode="global",
        output_path=tmp_path / "coherencia.json",
        output_format="json",
    )

    for section in evaluation.sections:
        for dimension in section.dimensions:
            if len(dimension.questions) >= 3:
                question_scores = [q.score for q in dimension.questions if q.score is not None]
                assert len(question_scores) == len(dimension.questions)
                average = sum(question_scores) / len(question_scores)
                assert dimension.score == pytest.approx(average)


def test_model_stability_across_run_ids(
    evaluation_service_factory: Any, tmp_path: Path
) -> None:
    document = _build_document()
    criteria = _build_criteria()

    service_a = evaluation_service_factory(run_id="run-stability-a", seed=13)
    evaluation_a, _ = service_a.run(
        document=_build_document(),
        criteria_data=_build_criteria(),
        mode="global",
        output_path=tmp_path / "stability_a.json",
        output_format="json",
    )

    service_b = evaluation_service_factory(run_id="run-stability-b", seed=13)
    evaluation_b, _ = service_b.run(
        document=_build_document(),
        criteria_data=_build_criteria(),
        mode="global",
        output_path=tmp_path / "stability_b.json",
        output_format="json",
    )

    scores_a = _flatten_question_scores(evaluation_a)
    scores_b = _flatten_question_scores(evaluation_b)

    assert scores_a == scores_b


def test_human_reference_correlation(
    evaluation_service_factory: Any, tmp_path: Path
) -> None:
    service = evaluation_service_factory(run_id="run-correlacion", seed=19)
    evaluation, _ = service.run(
        document=_build_document(),
        criteria_data=_build_criteria(),
        mode="global",
        output_path=tmp_path / "correlacion.json",
        output_format="json",
    )

    actual_scores = _flatten_question_scores(evaluation)

    expected_path = Path(__file__).parent / "data" / "expected_score.csv"
    expected: Dict[str, float] = {}
    with expected_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            expected[row["question_id"]] = float(row["expected_score"])

    common_ids = sorted(set(actual_scores) & set(expected))
    assert common_ids, "No hay preguntas coincidentes entre puntajes esperados y reales."

    expected_vector = [expected[qid] for qid in common_ids]
    actual_vector = [actual_scores[qid] for qid in common_ids]
    correlation = _pearson_correlation(expected_vector, actual_vector)

    assert correlation >= 0.9

# pytest tests/test_evaluation_consistency.py -v