"""Performance resilience tests to ensure the evaluator handles production load."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping
import uuid

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
)


def _reset_instances() -> None:
    for cls in TRACKING_CLASSES:
        instances = getattr(cls, "instances", None)
        if isinstance(instances, list):
            for instance in list(instances):
                if hasattr(instance, "calls"):
                    getattr(instance, "calls").clear()  # type: ignore[attr-defined]
                if hasattr(instance, "exports"):
                    getattr(instance, "exports").clear()  # type: ignore[attr-defined]
            instances.clear()


@pytest.fixture(autouse=True)
def _tracking_state_guard() -> Iterable[None]:
    _reset_instances()
    yield
    _reset_instances()


def _build_section_payload(section: Mapping[str, Any], question_index: int) -> Dict[str, Any]:
    question_id = f"{section['id'].upper()}_Q{question_index}"
    return {
        "id": question_id,
        "texto": f"Pregunta {question_index} de {section['id']}",
        "ponderacion": 1.0,
        "niveles": [{"valor": 0.0}, {"valor": 2.0}, {"valor": 4.0}],
    }


def _build_criteria(tmp_path: Path, name: str, sections: List[Dict[str, Any]]) -> Path:
    payload = {
        "version": f"perf-{name}",
        "tipo_informe": "institucional",
        "metodologia": "ponderada",
        "secciones": [],
    }
    total_sections = len(sections)
    for section in sections:
        section_payload = {
            "id": section["id"],
            "titulo": section["id"].replace("_", " ").title(),
            "ponderacion": round(1.0 / max(1, total_sections), 4),
            "dimensiones": [
                {
                    "nombre": f"dimension_{section['id']}",
                    "ponderacion": 1.0,
                    "preguntas": [
                        _build_section_payload(section, question)
                        for question in range(section["questions"])
                    ],
                }
            ],
        }
        payload["secciones"].append(section_payload)
    criteria_path = tmp_path / f"criteria_{name}.json"
    criteria_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return criteria_path


def _build_document(tmp_path: Path, name: str, sections: List[Dict[str, Any]]) -> Path:
    lines: List[str] = []
    for section in sections:
        lines.append(f"[SECTION:{section['id']}]")
        paragraph = (
            f"Contenido de {section['id']} "
            f"{' '.join([section['id']] * 5)}"
        )
        for index in range(section["paragraphs"]):
            lines.append(f"{paragraph} párrafo {index}")
    document_path = tmp_path / f"document_{name}.txt"
    document_path.write_text("\n".join(lines), encoding="utf-8")
    return document_path


@pytest.fixture
def service_factory(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("pandas")
    _prepare_environment(monkeypatch)

    def _factory(run_id: str | None = None) -> EvaluationService:
        loader = TrackingLoader()
        cleaner = TrackingCleaner()
        splitter = TrackingSplitter()
        repository = TrackingRepository()
        prompt_builder = TrackingPromptBuilder()
        prompt_validator = TrackingPromptValidator()
        ai_service = TrackingAIService()

        config = ServiceConfig(
            ai_provider="tracking",
            run_id=run_id or f"performance-{uuid.uuid4().hex}",
            model_name="tracking-model",
            prompt_batch_size=1,
            retries=0,
            timeout_seconds=None,
            log_level="ERROR",
        )

        return EvaluationService(
            config=config,
            loader=loader,
            cleaner=cleaner,
            splitter=splitter,
            repository=repository,
            ai_service_factory=lambda _: ai_service,
            prompt_builder=prompt_builder,
            prompt_validator=prompt_validator,
        )

    return _factory


def _count_questions(evaluation) -> int:
    total = 0
    for section in evaluation.sections:
        for dimension in section.dimensions:
            total += len(dimension.questions)
    return total


def _hash_json(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))

    rows: List[Dict[str, Any]] = []
    for row in payload.get("rows", []):
        cleaned_row: Dict[str, Any] = {}
        for key, value in row.items():
            if key in {"timestamp", "run_id", "metadata.run_id", "metadata.run_id_history", "document_id"}:
                continue
            if key == "chunk_results":
                simplified_chunks = []
                for chunk in value:
                    simplified_chunks.append(
                        {
                            "index": chunk.get("index"),
                            "score": chunk.get("score"),
                            "justification": chunk.get("justification"),
                            "relevant_text": chunk.get("relevant_text"),
                        }
                    )
                cleaned_row[key] = simplified_chunks
                continue
            cleaned_row[key] = value
        rows.append(cleaned_row)

    evaluation_payload = payload.get("evaluation", {})
    evaluation = {
        "document_id": "normalized-document",
        "document_type": evaluation_payload.get("document_type"),
        "score": evaluation_payload.get("score"),
    }

    metrics = dict(payload.get("metrics", {}))
    metrics.pop("system", None)

    normalised = json.dumps(
        {
            "rows": rows,
            "evaluation": evaluation,
            "metrics": metrics,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def test_execution_time_and_memory_metrics_scale(service_factory, tmp_path: Path) -> None:
    service = service_factory(run_id="performance-scale")

    size_profiles = {
        "small": [
            {"id": "seccion_resumen", "paragraphs": 2, "questions": 2},
            {"id": "seccion_gestion", "paragraphs": 2, "questions": 1},
        ],
        "medium": [
            {"id": "seccion_resumen", "paragraphs": 4, "questions": 3},
            {"id": "seccion_gestion", "paragraphs": 4, "questions": 3},
            {"id": "seccion_proyectos", "paragraphs": 4, "questions": 2},
        ],
        "large": [
            {"id": "seccion_resumen", "paragraphs": 6, "questions": 4},
            {"id": "seccion_gestion", "paragraphs": 6, "questions": 4},
            {"id": "seccion_proyectos", "paragraphs": 6, "questions": 4},
            {"id": "seccion_resultados", "paragraphs": 6, "questions": 3},
        ],
    }

    durations: Dict[str, float] = {}
    totals: Dict[str, int] = {}
    per_question: Dict[str, float] = {}

    for name, profile in size_profiles.items():
        document = _build_document(tmp_path, name, profile)
        criteria = _build_criteria(tmp_path, name, profile)
        output_path = tmp_path / f"result_{name}.json"
        evaluation, metrics = service.run(
            input_path=document,
            criteria_path=criteria,
            output_path=output_path,
            output_format="json",
        )
        system_metrics = metrics.get("system", {})
        assert system_metrics["execution_time_seconds"] > 0
        assert system_metrics["memory_peak_mib"] >= system_metrics["memory_current_mib"]
        assert system_metrics["questions_total"] == _count_questions(evaluation)
        assert system_metrics["questions_evaluated"] == _count_questions(evaluation)
        durations[name] = system_metrics["execution_time_seconds"]
        totals[name] = system_metrics["questions_total"]
        per_question[name] = durations[name] / max(1, totals[name])

    assert totals["small"] < totals["medium"] < totals["large"]
    assert per_question["medium"] <= per_question["small"] + 0.05
    assert per_question["large"] <= per_question["small"] + 0.05


def test_memory_usage_remains_stable_across_runs(service_factory, tmp_path: Path) -> None:
    service = service_factory(run_id="performance-memory")
    profile = [
        {"id": "seccion_memoria", "paragraphs": 8, "questions": 5},
        {"id": "seccion_control", "paragraphs": 8, "questions": 4},
    ]
    document = _build_document(tmp_path, "memory", profile)
    criteria = _build_criteria(tmp_path, "memory", profile)

    peaks: List[float] = []
    for iteration in range(5):
        output_path = tmp_path / f"memory_{iteration}.json"
        _, metrics = service.run(
            input_path=document,
            criteria_path=criteria,
            output_path=output_path,
            output_format="json",
        )
        peaks.append(metrics["system"]["memory_peak_mib"])

    assert max(peaks) - min(peaks) < 5


def test_parallel_runs_produce_consistent_results(service_factory, tmp_path: Path) -> None:
    profile = [
        {"id": "seccion_parallel", "paragraphs": 3, "questions": 3},
        {"id": "seccion_batch", "paragraphs": 3, "questions": 3},
    ]
    criteria = _build_criteria(tmp_path, "parallel", profile)
    documents = [
        _build_document(tmp_path, f"parallel_{index}", profile)
        for index in range(4)
    ]

    def _worker(index: int) -> tuple[str, Dict[str, Any]]:
        service = service_factory(run_id=f"parallel-{index}")
        output_path = tmp_path / f"parallel_result_{index}.json"
        _, metrics = service.run(
            input_path=documents[index],
            criteria_path=criteria,
            output_path=output_path,
            output_format="json",
        )
        return _hash_json(output_path), metrics["system"]

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(_worker, range(len(documents))))

    hashes = {result[0] for result in results}
    assert len(hashes) == 1
    for _, system_metrics in results:
        assert system_metrics["execution_time_seconds"] > 0


def test_repeated_runs_remain_stable(service_factory, tmp_path: Path) -> None:
    profile = [
        {"id": "seccion_stable", "paragraphs": 5, "questions": 4},
        {"id": "seccion_resultados", "paragraphs": 5, "questions": 4},
    ]
    document = _build_document(tmp_path, "stable", profile)
    criteria = _build_criteria(tmp_path, "stable", profile)

    reference_hash: str | None = None
    for iteration in range(25):
        service = service_factory(run_id="performance-stability")
        output_path = tmp_path / f"stable_result_{iteration}.json"
        _, metrics = service.run(
            input_path=document,
            criteria_path=criteria,
            output_path=output_path,
            output_format="json",
        )
        payload_hash = _hash_json(output_path)
        if reference_hash is None:
            reference_hash = payload_hash
        else:
            assert payload_hash == reference_hash
        assert metrics["system"]["execution_time_seconds"] > 0

# pytest tests/resilience/test_performance_characteristics.py -v