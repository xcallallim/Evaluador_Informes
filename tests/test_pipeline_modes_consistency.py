from __future__ import annotations

import copy
from collections import deque
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

import pytest

from data.models.document import Document
from data.models.evaluation import (
    DimensionResult,
    EvaluationResult,
    QuestionResult,
    SectionResult,
)
from reporting.repository import flatten_evaluation
from services import evaluation_service as evaluation_module
from services.evaluation_service import EvaluationFilters, EvaluationService, ServiceConfig


class _DummySummary:
    def __init__(self, payload: Mapping[str, Any]) -> None:
        self._payload = dict(payload)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._payload)


def _build_section(
    section_id: str,
    *,
    title: str,
    weight: float,
    question_id: str,
    question_score: float,
) -> SectionResult:
    question = QuestionResult(
        question_id=question_id,
        text=f"Pregunta {question_id}",
        weight=1.0,
        score=question_score,
    )
    dimension = DimensionResult(
        name=f"Dimension {section_id}",
        weight=1.0,
        questions=[question],
    )
    dimension.recompute_score()
    section = SectionResult(
        section_id=section_id,
        title=title,
        weight=weight,
        dimensions=[dimension],
    )
    section.recompute_score()
    return section


def _build_evaluation(
    sections: List[SectionResult], *, metadata: Mapping[str, Any] | None = None
) -> EvaluationResult:
    base_metadata: Dict[str, Any] = {
        "tipo_informe": "institucional",
        "criteria_version": "test-v1",
    }
    if metadata:
        base_metadata.update(dict(metadata))
    evaluation = EvaluationResult(
        document_id="DOC-001",
        document_type="institucional",
        methodology="institucional",
        sections=sections,
        metadata=base_metadata,
    )
    evaluation.recompute_score()
    return evaluation


@pytest.mark.parametrize("output_format", ["json"])
def test_pipeline_modes_consistency(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, output_format: str) -> None:
    criteria_data: Dict[str, Any] = {
        "version": "test-v1",
        "tipo_informe": "institucional",
        "secciones": [
            {
                "id": "resumen",
                "titulo": "Resumen Ejecutivo",
                "dimensiones": [
                    {
                        "nombre": "General",
                        "preguntas": [
                            {
                                "id": "P1",
                                "texto": "Pregunta 1",
                            }
                        ],
                        "niveles": [
                            {"valor": 0},
                            {"valor": 1},
                            {"valor": 2},
                            {"valor": 3},
                        ],
                    }
                ],
            },
            {
                "id": "planificacion",
                "titulo": "Planificación",
                "dimensiones": [
                    {
                        "nombre": "Detalle",
                        "preguntas": [
                            {
                                "id": "P2",
                                "texto": "Pregunta 2",
                            }
                        ],
                        "niveles": [
                            {"valor": 0},
                            {"valor": 1},
                            {"valor": 2},
                            {"valor": 3},
                        ],
                    }
                ],
            },
        ],
    }

    criteria_data_v2 = copy.deepcopy(criteria_data)
    criteria_data_v2["version"] = "test-v2"
    criteria_data_v2["secciones"][1]["dimensiones"][0]["niveles"].append({"valor": 4})
    criteria_data_v2["secciones"][1]["dimensiones"][0]["niveles"].append({"valor": 5})

    global_section = _build_section(
        "resumen", title="Resumen Ejecutivo", weight=0.6, question_id="P1", question_score=3.0
    )
    partial_section = _build_section(
        "planificacion", title="Planificación", weight=0.4, question_id="P2", question_score=2.0
    )
    updated_section = _build_section(
        "planificacion", title="Planificación", weight=0.4, question_id="P2", question_score=3.5
    )

    global_result = _build_evaluation([global_section, partial_section])
    partial_result = _build_evaluation([partial_section])
    incremental_update = _build_evaluation(
        [updated_section], metadata={"criteria_version": "test-v2"}
    )

    results_queue = deque([global_result, partial_result, incremental_update])
    captured_criteria: List[Dict[str, Any]] = []
    captured_metrics_inputs: List[Dict[str, Any]] = []

    class _FakeEvaluator:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - trivial
            pass

        def evaluate(
            self,
            document: Document,
            criteria: Mapping[str, Any],
            *,
            document_id: str,
        ) -> EvaluationResult:
            captured_criteria.append(copy.deepcopy(dict(criteria)))
            result = copy.deepcopy(results_queue.popleft())
            result.metadata.setdefault("document_id", document_id)
            result.recompute_score()
            return result

    def _fake_calculate_metrics(
        evaluation: EvaluationResult, criteria: Mapping[str, Any]
    ) -> _DummySummary:
        captured_metrics_inputs.append(
            {
                "evaluation": copy.deepcopy(evaluation),
                "criteria": copy.deepcopy(dict(criteria)),
            }
        )
        payload = {
            "methodology": evaluation.methodology,
            "global": {"raw_score": evaluation.score},
            "sections": [
                {
                    "section_id": section.section_id,
                    "title": section.title,
                    "score": section.score,
                }
                for section in evaluation.sections
            ],
        }
        return _DummySummary(payload)

    export_calls: List[Dict[str, Any]] = []

    class _FakeRepository:
        def export(
            self,
            evaluation: EvaluationResult,
            metrics_summary: Mapping[str, Any],
            *,
            output_path: Path,
            output_format: str,
            extra_metadata: Mapping[str, Any] | None = None,
        ) -> Path:  # pragma: no cover - exercised indirectly
            if isinstance(metrics_summary, Mapping):
                metrics_payload = copy.deepcopy(dict(metrics_summary))
            elif hasattr(metrics_summary, "to_dict"):
                metrics_payload = copy.deepcopy(dict(metrics_summary.to_dict()))
            else:
                metrics_payload = copy.deepcopy(metrics_summary)

            export_calls.append(
                {
                    "evaluation": copy.deepcopy(evaluation),
                    "metrics": metrics_payload,
                    "output_path": output_path,
                    "output_format": output_format,
                    "extra_metadata": copy.deepcopy(dict(extra_metadata or {})),
                }
            )
            return output_path

    monkeypatch.setattr(evaluation_module, "ValidatingEvaluator", _FakeEvaluator)
    monkeypatch.setattr(evaluation_module, "calculate_metrics", _fake_calculate_metrics)

    service = EvaluationService(config=ServiceConfig())
    service.repository = _FakeRepository()

    document = Document(
        content="Documento de prueba",
        metadata={"id": "DOC-001", "document_type": "institucional"},
        pages=["Documento de prueba"],
    )

    evaluation_global, metrics_global = service.run(
        document=document,
        criteria_data=criteria_data,
        tipo_informe="institucional",
        mode="global",
        config_overrides={"run_id": "run-global"},
    )

    assert {section.section_id for section in evaluation_global.sections} == {
        "resumen",
        "planificacion",
    }
    assert len(metrics_global["sections"]) == 2
    assert len(captured_criteria[0]["secciones"]) == 2

    filters = EvaluationFilters(section_ids=["planificacion"])
    evaluation_partial, metrics_partial = service.run(
        document=document,
        criteria_data=criteria_data,
        tipo_informe="institucional",
        mode="parcial",
        filters=filters,
        config_overrides={"run_id": "run-partial"},
    )

    assert [section.section_id for section in evaluation_partial.sections] == [
        "planificacion"
    ]
    assert len(metrics_partial["sections"]) == 1
    assert len(captured_criteria[1]["secciones"]) == 1
    assert captured_criteria[1]["secciones"][0]["id"] == "planificacion"

    incremental_output = tmp_path / f"incremental_result.{output_format}"
    evaluation_incremental, metrics_incremental = service.run(
        document=document,
        criteria_data=criteria_data_v2,
        tipo_informe="institucional",
        mode="incremental",
        filters=filters,
        previous_result=evaluation_global,
        output_path=incremental_output,
        output_format=output_format,
        config_overrides={"run_id": "run-incremental"},
    )

    assert len(results_queue) == 0
    assert [section.section_id for section in evaluation_incremental.sections] == [
        "resumen",
        "planificacion",
    ]
    updated_scores = {
        section.section_id: section.score for section in evaluation_incremental.sections
    }
    assert updated_scores["resumen"] == pytest.approx(3.0)
    assert updated_scores["planificacion"] == pytest.approx(3.5)
    assert metrics_incremental["global"]["raw_score"] == pytest.approx(3.2)
    assert len(captured_criteria[2]["secciones"]) == 1
    assert len(captured_metrics_inputs[2]["evaluation"].sections) == 2

    assert len(export_calls) == 3, "Se esperan tres exportaciones (global, parcial, incremental)"
    exported = export_calls[-1]
    exported_sections = [
        section.section_id for section in exported["evaluation"].sections
    ]
    assert exported_sections == ["planificacion"]
    assert len(exported["metrics"]["sections"]) == 1
    assert exported["metrics"]["sections"][0]["section_id"] == "planificacion"
    assert exported["output_path"] == incremental_output
    assert exported["output_format"] == output_format
    assert exported["extra_metadata"]["filters"]["section_ids"] == [
        "planificacion"
    ]

    assert evaluation_incremental.metadata["pipeline_version"] == evaluation_global.metadata["pipeline_version"]
    assert evaluation_incremental.metadata["criteria_version"] == "test-v2"

    assert evaluation_incremental.metadata["criteria_version_history"] == [
        "test-v1",
        "test-v2",
    ]
    assert evaluation_incremental.metadata["run_id_history"] == [
        "run-global",
        "run-incremental",
    ]
    assert evaluation_incremental.metadata["pipeline_version_history"] == [
        evaluation_module.SERVICE_VERSION
    ]
    assert evaluation_incremental.metadata["parent_run_id"] == "run-global"
    assert evaluation_incremental.metadata["runs"][-1]["pipeline_version"] == evaluation_module.SERVICE_VERSION
    assert evaluation_incremental.metadata["runs"][-1]["criteria_version"] == "test-v2"

    planificacion_section = next(
        section
        for section in evaluation_incremental.sections
        if section.section_id == "planificacion"
    )
    assert planificacion_section.metadata["run_id_history"] == [
        "run-global",
        "run-incremental",
    ]
    assert planificacion_section.metadata["pipeline_version_history"] == [
        evaluation_module.SERVICE_VERSION
    ]
    dimension_metadata = planificacion_section.dimensions[0].metadata
    assert dimension_metadata["run_id_history"] == [
        "run-global",
        "run-incremental",
    ]
    assert dimension_metadata["pipeline_version_history"] == [
        evaluation_module.SERVICE_VERSION
    ]
    question_metadata = planificacion_section.dimensions[0].questions[0].metadata
    assert question_metadata["run_id_history"] == [
        "run-global",
        "run-incremental",
    ]
    assert question_metadata["pipeline_version_history"] == [
        evaluation_module.SERVICE_VERSION
    ]

    for section in evaluation_incremental.sections:
        dimension_names = [dimension.name for dimension in section.dimensions]
        assert len(dimension_names) == len(set(dimension_names))
        for dimension in section.dimensions:
            question_ids = [question.question_id for question in dimension.questions]
            assert len(question_ids) == len(set(question_ids))

    assert len(exported_sections) == len(set(exported_sections))
    for section in exported["evaluation"].sections:
        dim_names = [dimension.name for dimension in section.dimensions]
        assert len(dim_names) == len(set(dim_names))
        for dimension in section.dimensions:
            qids = [question.question_id for question in dimension.questions]
            assert len(qids) == len(set(qids))

    exported_section = exported["evaluation"].sections[0]
    assert exported_section.metadata["run_id_history"] == [
        "run-global",
        "run-incremental",
    ]
    assert exported_section.metadata["pipeline_version_history"] == [
        evaluation_module.SERVICE_VERSION
    ]
    exported_dimension = exported_section.dimensions[0]
    assert exported_dimension.metadata["run_id_history"] == [
        "run-global",
        "run-incremental",
    ]
    assert exported_dimension.metadata["pipeline_version_history"] == [
        evaluation_module.SERVICE_VERSION
    ]
    exported_question = exported_dimension.questions[0]
    assert exported_question.metadata["run_id_history"] == [
        "run-global",
        "run-incremental",
    ]
    assert exported_question.metadata["pipeline_version_history"] == [
        evaluation_module.SERVICE_VERSION
    ]

    metrics_sections = exported["metrics"]["sections"]
    assert len(metrics_sections) == len(
        {entry["section_id"] for entry in metrics_sections}
    )
    section_metrics = metrics_sections[0]
    assert section_metrics["raw_score"] is not None
    assert section_metrics["normalized_score"] is not None
    assert section_metrics["scale_min"] is not None
    assert section_metrics["scale_max"] == pytest.approx(5.0)
    assert section_metrics["criteria_version"] == "test-v2"

    global_metrics = exported["metrics"]["global"]
    assert global_metrics["raw_score"] is not None
    assert global_metrics["normalized_score"] is not None
    assert global_metrics["scale_min"] == pytest.approx(0.0)
    assert global_metrics["scale_max"] == pytest.approx(20.0)
    assert global_metrics["criteria_version"] == "test-v2"

    global_flattened = flatten_evaluation(evaluation_global)
    incremental_flattened = flatten_evaluation(evaluation_incremental)
    partial_flattened = flatten_evaluation(exported["evaluation"])

    incremental_subset = [
        row for row in incremental_flattened if row["section_id"] == "planificacion"
    ]
    assert partial_flattened == incremental_subset
    common_questions = {row["question_id"] for row in incremental_flattened}
    assert common_questions == {row["question_id"] for row in global_flattened}

    def _hash_payload(payload: list[dict[str, Any]]) -> str:
        serialised = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.md5(serialised.encode("utf-8")).hexdigest()

    assert _hash_payload(global_flattened) != _hash_payload(partial_flattened)
    assert _hash_payload(incremental_subset) == _hash_payload(partial_flattened)


def test_partial_mode_missing_section_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    criteria_data: Dict[str, Any] = {
        "version": "test-v1",
        "tipo_informe": "institucional",
        "secciones": [
            {
                "id": "resumen",
                "titulo": "Resumen Ejecutivo",
                "dimensiones": [
                    {
                        "nombre": "General",
                        "preguntas": [
                            {
                                "id": "P1",
                                "texto": "Pregunta 1",
                            }
                        ],
                    }
                ],
            },
            {
                "id": "planificacion",
                "titulo": "Planificación",
                "dimensiones": [
                    {
                        "nombre": "Detalle",
                        "preguntas": [
                            {
                                "id": "P2",
                                "texto": "Pregunta 2",
                            }
                        ],
                    }
                ],
            },
        ],
    }

    available_section = _build_section(
        "resumen", title="Resumen Ejecutivo", weight=1.0, question_id="P1", question_score=2.0
    )
    evaluation_missing = _build_evaluation([available_section])

    class _FakeEvaluator:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - trivial
            pass

        def evaluate(
            self,
            document: Document,
            criteria: Mapping[str, Any],
            *,
            document_id: str,
        ) -> EvaluationResult:
            return copy.deepcopy(evaluation_missing)

    monkeypatch.setattr(evaluation_module, "ValidatingEvaluator", _FakeEvaluator)

    service = EvaluationService(config=ServiceConfig())
    document = Document(
        content="Documento de prueba",
        metadata={"id": "DOC-002", "document_type": "institucional"},
        pages=["Documento de prueba"],
    )

    filters = EvaluationFilters(section_ids=["planificacion"])
    with pytest.raises(ValueError, match="secciones faltantes"):
        service.run(
            document=document,
            criteria_data=criteria_data,
            tipo_informe="institucional",
            mode="parcial",
            filters=filters,
            config_overrides={"run_id": "run-partial-missing"},
        )


def test_duplicate_run_id_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    criteria_data: Dict[str, Any] = {
        "version": "test-v1",
        "tipo_informe": "institucional",
        "secciones": [],
    }

    previous_section = _build_section(
        "resumen", title="Resumen Ejecutivo", weight=1.0, question_id="P1", question_score=2.0
    )
    previous_result = _build_evaluation([previous_section], metadata={"run_id": "run-prev"})
    previous_result.metadata.setdefault("runs", []).append({"run_id": "run-prev"})
    previous_result.metadata.setdefault("run_id_history", ["run-prev"])

    class _NeverEvaluator:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - trivial
            raise AssertionError("No se debe inicializar el evaluador cuando el run_id es duplicado")

    monkeypatch.setattr(evaluation_module, "ValidatingEvaluator", _NeverEvaluator)

    service = EvaluationService(config=ServiceConfig())
    document = Document(
        content="Documento de prueba",
        metadata={"id": "DOC-003", "document_type": "institucional"},
        pages=["Documento de prueba"],
    )

    filters = EvaluationFilters(section_ids=["resumen"])

    with pytest.raises(ValueError, match="run_id proporcionado"):
        service.run(
            document=document,
            criteria_data=criteria_data,
            tipo_informe="institucional",
            mode="incremental",
            filters=filters,
            previous_result=previous_result,
            config_overrides={"run_id": "run-prev"},
        )


def test_incremental_interruption_marks_incomplete(monkeypatch: pytest.MonkeyPatch) -> None:
    criteria_data: Dict[str, Any] = {
        "version": "test-v1",
        "tipo_informe": "institucional",
        "secciones": [
            {
                "id": "planificacion",
                "titulo": "Planificación",
                "dimensiones": [
                    {
                        "nombre": "Detalle",
                        "preguntas": [
                            {
                                "id": "P2",
                                "texto": "Pregunta 2",
                            }
                        ],
                    }
                ],
            }
        ],
    }

    previous_section = _build_section(
        "planificacion", title="Planificación", weight=1.0, question_id="P2", question_score=2.0
    )
    previous_result = _build_evaluation([previous_section], metadata={"run_id": "run-prev"})
    snapshot = copy.deepcopy(previous_result.metadata)

    class _ExplodingEvaluator:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - trivial
            pass

        def evaluate(
            self,
            document: Document,
            criteria: Mapping[str, Any],
            *,
            document_id: str,
        ) -> EvaluationResult:
            raise RuntimeError("boom")

    monkeypatch.setattr(evaluation_module, "ValidatingEvaluator", _ExplodingEvaluator)

    service = EvaluationService(config=ServiceConfig())
    document = Document(
        content="Documento de prueba",
        metadata={"id": "DOC-004", "document_type": "institucional"},
        pages=["Documento de prueba"],
    )

    filters = EvaluationFilters(section_ids=["planificacion"])

    with pytest.raises(RuntimeError, match="Ejecución incremental incompleta"):
        service.run(
            document=document,
            criteria_data=criteria_data,
            tipo_informe="institucional",
            mode="incremental",
            filters=filters,
            previous_result=previous_result,
            config_overrides={"run_id": "run-new"},
        )

    assert previous_result.metadata == snapshot


def test_load_previous_result_invalid_json(tmp_path: Path) -> None:
    corrupted = tmp_path / "prev.json"
    corrupted.write_text("{invalid", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupto o incompleto"):
        evaluation_module._load_previous_result(corrupted)


def test_load_previous_result_without_sections(tmp_path: Path) -> None:
    from data.models.evaluation import EvaluationResult

    empty = EvaluationResult(
        document_id="DOC-005",
        document_type="institucional",
        methodology="institucional",
        sections=[],
        metadata={},
    )
    payload = {"evaluation": empty.to_dict()}
    path = tmp_path / "empty.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="no contiene secciones evaluadas"):
        evaluation_module._load_previous_result(path)

# pytest tests/test_pipeline_modes_consistency.py -v