"""Pruebas de integración extremo a extremo para el pipeline CEPLAN."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Mapping, Tuple

import pytest

from data.models.document import Document
from data.models.evaluation import EvaluationResult
from metrics import calculate_metrics as real_calculate_metrics
from reporting.repository import EvaluationRepository
from services import evaluation_service as evaluation_module
from services.evaluation_service import EvaluationService, ServiceConfig
from utils.prompt_validator import PromptValidationResult, PromptValidator


pytestmark = pytest.mark.integration


class TrackingLoader:
    """Loader simplificado que conserva el contenido original."""

    instances: List["TrackingLoader"] = []

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        type(self).instances.append(self)

    def load(
        self,
        filepath: str,
        *,
        extract_tables: bool = True,
        extract_images: bool = True,
    ) -> Document:
        path = Path(filepath)
        content = path.read_text(encoding="utf-8")
        metadata = {
            "source": str(path),
            "filename": path.name,
            "extension": path.suffix or ".txt",
            "processed_with": "tracking-loader",
            "pages": [content],
            "raw_text": content,
        }
        document = Document(content=content, metadata=metadata, pages=[content])
        self.calls.append(
            {
                "path": str(path),
                "extract_tables": extract_tables,
                "extract_images": extract_images,
            }
        )
        return document


class TrackingCleaner:
    """Cleaner que registra la operación y mantiene encabezados."""

    instances: List["TrackingCleaner"] = []

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        type(self).instances.append(self)

    def clean_document(
        self, document: Document, return_report: bool = False
    ) -> Document | tuple[Document, Dict[str, Any]]:
        header = ""
        for line in document.content.splitlines():
            if line.startswith("[SECTION:"):
                continue
            if line.strip():
                header = line.strip()
                break
        report = {
            "removed": [],
            "preserved_headers": [header] if header else [],
            "cleaned_at": "integration-test",
        }
        cleaned = Document(
            content=document.content,
            metadata=dict(document.metadata),
            pages=list(document.pages),
            tables=list(document.tables),
            images=list(document.images),
            sections=dict(document.sections),
            chunks=list(document.chunks),
        )
        self.calls.append({"source": document.metadata.get("source"), "header": header})
        if return_report:
            return cleaned, report
        return cleaned


class TrackingSegmenter:
    """Segmenter controlado por la prueba para asegurar trazabilidad."""

    instances: List["TrackingSegmenter"] = []

    def __init__(self, tipo: str) -> None:
        self.tipo = tipo
        self.calls: List[Dict[str, Any]] = []
        TrackingSegmenter.instances.append(self)

    def segment_document(self, document: Document) -> Document:
        sections: Dict[str, str] = {}
        current_id: str | None = None
        buffer: List[str] = []
        for line in document.content.splitlines():
            if line.startswith("[SECTION:") and line.endswith("]"):
                if current_id is not None:
                    sections[current_id] = "\n".join(buffer).strip()
                current_id = line[len("[SECTION:") : -1]
                buffer = []
                continue
            buffer.append(line)
        if current_id is not None:
            sections[current_id] = "\n".join(buffer).strip()
        document.sections = sections
        self.calls.append(
            {
                "document": document.metadata.get("source"),
                "sections": list(sections.keys()),
                "sections_content": dict(sections),
                "tipo": self.tipo,
            }
        )
        return document


class TrackingSplitter:
    """Crea chunks sintéticos manteniendo metadatos estructurados."""

    instances: List["TrackingSplitter"] = []

    def __init__(self, truncation_limit: int = 800) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.truncation_limit = truncation_limit
        type(self).instances.append(self)

    def _build_chunk_metadata(
        self, document: Document, section_id: str, index: int, text: str
    ) -> Dict[str, Any]:
        was_truncated = len(text) > self.truncation_limit
        metadata: Dict[str, Any] = {
            "section_id": section_id,
            "source_id": f"{section_id}-chunk-{index}",
            "document_metadata": {
                "id": document.metadata.get("id"),
                "cleaning_report": document.metadata.get("cleaning_report"),
            },
            "was_truncated": was_truncated,
        }
        if was_truncated:
            head_size = max(40, self.truncation_limit // 4)
            tail_size = max(40, self.truncation_limit // 4)
            metadata.update(
                {
                    "truncation_marker": "… [texto truncado]",
                    "head_preview": text[:head_size],
                    "tail_preview": text[-tail_size:],
                    "original_length": len(text),
                }
            )
        return metadata

    def split_document(self, document: Document) -> Document:
        chunks: List[SimpleNamespace] = []
        for index, (section_id, text) in enumerate(document.sections.items()):
            chunk_metadata = self._build_chunk_metadata(
                document, section_id, index, text
            )
            chunks.append(SimpleNamespace(page_content=text, metadata=chunk_metadata))
        if not chunks:
            chunk_metadata = self._build_chunk_metadata(
                document,
                "general",
                0,
                document.content,
            )
            chunks.append(SimpleNamespace(page_content=document.content, metadata=chunk_metadata))
        document.chunks = chunks
        self.calls.append(
            {
                "document": document.metadata.get("source"),
                "chunk_count": len(chunks),
                "truncation_limit": self.truncation_limit,
                "chunk_sections": [
                    chunk.metadata.get("section_id") for chunk in chunks
                ],
                "chunk_texts": [chunk.page_content for chunk in chunks],
            }
        )
        return document


class TrackingPromptBuilder:
    """Builder determinista que conserva el contexto recibido."""

    instances: List["TrackingPromptBuilder"] = []

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        instances: List["TrackingPromptBuilder"] = []

    def __call__(
        self,
        *,
        document: Document,
        criteria: Mapping[str, Any],
        section: Mapping[str, Any],
        dimension: Mapping[str, Any],
        question: Mapping[str, Any],
        chunk_text: str,
        chunk_metadata: Mapping[str, Any],
        extra_instructions: str | None = None,
    ) -> str:
        record = {
            "document_id": document.metadata.get("id"),
            "section_id": section.get("id") or section.get("id_segmenter"),
            "question_id": question.get("id"),
            "chunk_section": chunk_metadata.get("section_id"),
            "extra": extra_instructions,
        }
        self.calls.append(record)
        return (
            f"CEPLAN::{record['section_id']}::{record['question_id']}::"
            f"{chunk_metadata.get('source_id')}"
        )


class TrackingPromptValidator:
    """Validator que aprueba prompts con una puntuación fija."""

    VERSION = "tracking-validator-1.0"

    instances: List["TrackingPromptValidator"] = []

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        type(self).instances.append(self)

    def validate(self, prompt: str, context: Mapping[str, Any]) -> PromptValidationResult:
        self.calls.append({"prompt": prompt, "context": dict(context)})
        return PromptValidationResult(
            is_valid=True,
            quality_score=0.93,
            alerts=["ok"],
            metadata={"validator": "tracking"},
        )


class TrackingAIService:
    """Servicio de IA determinista para la prueba de integración."""

    instances: List["TrackingAIService"] = []

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.model_name = "tracking-model"
        type(self).instances.append(self)

    def evaluate(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        call = {
            "prompt": prompt,
            "question": kwargs.get("question", {}).get("id"),
            "section": kwargs.get("section", {}).get("id"),
            "chunk_index": kwargs.get("chunk_index"),
        }
        self.calls.append(call)
        chunk_index = kwargs.get("chunk_index") or 0
        section_payload = kwargs.get("section") or {}
        criteria_payload = kwargs.get("criteria") or {}
        chunk_metadata = kwargs.get("chunk_metadata") or {}
        section_id = section_payload.get("id") or section_payload.get("id_segmenter")
        chunk_section = chunk_metadata.get("section_id")
        if section_id and chunk_section and section_id != chunk_section:
            metadata = {
                "model": self.model_name,
                "provider": "tracking",
                "response_tokens": 64 + len(self.calls),
                "missing_section": True,
            }
            return {
                "score": 0.0,
                "justification": "sin evidencia disponible en el fragmento",
                "relevant_text": None,
                "metadata": metadata,
            }

        report_type = str(criteria_payload.get("tipo_informe", "")).lower()
        if report_type in {"politica_nacional", "politica"}:
            base_score = 1.0 + 0.5 * (int(chunk_index) % 3)
            score = min(2.0, max(0.0, base_score))
        else:
            score = 3.25 + 0.05 * float(chunk_index)
        return {
            "score": round(score, 2),
            "justification": f"evaluacion-{call['question']}",
            "relevant_text": kwargs.get("chunk_metadata", {}).get("section_id"),
            "metadata": {
                "model": self.model_name,
                "provider": "tracking",
                "response_tokens": 64 + len(self.calls),
            },
        }


class TrackingRepository(EvaluationRepository):
    """Repositorio que conserva los parámetros de exportación."""

    instances: List["TrackingRepository"] = []

    def __init__(self) -> None:
        super().__init__()
        self.exports: List[Dict[str, Any]] = []
        type(self).instances.append(self)

    def export(
        self,
        evaluation: EvaluationResult,
        metrics_summary: Mapping[str, Any],
        *,
        output_path: Path,
        output_format: str = "json",
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        record = {
            "output_path": Path(output_path),
            "format": output_format,
            "extra_metadata": dict(extra_metadata or {}),
            "metrics": dict(metrics_summary),
        }
        result = super().export(
            evaluation,
            metrics_summary,
            output_path=output_path,
            output_format=output_format,
            extra_metadata=extra_metadata,
        )
        self.exports.append(record)
        return result


TRACKING_CLASSES = (
    TrackingLoader,
    TrackingCleaner,
    TrackingSegmenter,
    TrackingSplitter,
    TrackingPromptBuilder,
    TrackingPromptValidator,
    TrackingAIService,
    TrackingRepository,
)


@pytest.fixture(autouse=True)
def _reset_tracking_state() -> Iterator[None]:
    for cls in TRACKING_CLASSES:
        instances = getattr(cls, "instances", None)
        if isinstance(instances, list):
            for instance in list(instances):
                if hasattr(instance, "calls"):
                    getattr(instance, "calls").clear()  # type: ignore[attr-defined]
                if hasattr(instance, "exports"):
                    getattr(instance, "exports").clear()  # type: ignore[attr-defined]
            instances.clear()
    yield
    for cls in TRACKING_CLASSES:
        instances = getattr(cls, "instances", None)
        if isinstance(instances, list):
            for instance in list(instances):
                if hasattr(instance, "calls"):
                    getattr(instance, "calls").clear()  # type: ignore[attr-defined]
                if hasattr(instance, "exports"):
                    getattr(instance, "exports").clear()  # type: ignore[attr-defined]
            instances.clear()


def _install_metrics_spy(
    monkeypatch: pytest.MonkeyPatch,
) -> Tuple[List[Dict[str, Any]], Any]:
    """Intercepta ``calculate_metrics`` para inspeccionar las llamadas."""

    metrics_calls: List[Dict[str, Any]] = []

    def tracking_calculate_metrics(
        evaluation: EvaluationResult,
        criteria: Mapping[str, Any],
        *,
        normalized_range: tuple[float, float] | None = None,
        weights: Mapping[str, float] | None = None,
    ):
        metrics_calls.append(
            {
                "score": evaluation.score,
                "sections": [section.section_id for section in evaluation.sections],
                "normalized_range": normalized_range,
            }
        )
        return real_calculate_metrics(
            evaluation,
            criteria,
            normalized_range=normalized_range,
            weights=weights,
        )

    monkeypatch.setattr(evaluation_module, "calculate_metrics", tracking_calculate_metrics)
    return metrics_calls, tracking_calculate_metrics


def _prepare_environment(monkeypatch: pytest.MonkeyPatch) -> List[Dict[str, Any]]:
    """Configura los parches comunes para las pruebas de integración."""

    TrackingSegmenter.instances.clear()
    monkeypatch.setattr(evaluation_module, "Segmenter", TrackingSegmenter)
    metrics_calls, _ = _install_metrics_spy(monkeypatch)
    return metrics_calls


class SlightlyDriftingAIService(TrackingAIService):
    """Servicio que introduce pequeñas variaciones controladas en el puntaje."""

    def __init__(self, drift: float) -> None:
        super().__init__()
        self._drift = drift

    def evaluate(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        base_response = super().evaluate(prompt, **kwargs)
        adjusted = dict(base_response)
        metadata = dict(base_response.get("metadata", {}))
        score = adjusted.get("score")
        if score is not None:
            adjusted["score"] = round(float(score) + self._drift, 2)
        metadata["response_tokens"] = metadata.get("response_tokens", 0) + 1
        adjusted["metadata"] = metadata
        return adjusted


POLICY_MISSING_SECTION_ID = "seguimiento_no_encontrado"


@pytest.fixture
def sample_document(tmp_path: Path) -> Path:
    document_path = tmp_path / "informe_ceplan.txt"
    document_path.write_text(
        "\n".join(
            [
                "[SECTION:resumen_ejecutivo]",
                "Resumen Ejecutivo - CEPLAN 2025",
                "Conclusiones",
                "1. Introducción El informe sintetiza hallazgos clave.",
                "El informe describe hitos estratégicos y logros institucionales.",
                "",
                "[SECTION:gestion_ceplan]",
                "Gestión CEPLAN alineada a la metodología institucional.",
                "Se evidencian acciones y resultados medibles.",
            ]
        ),
        encoding="utf-8",
    )
    return document_path


@pytest.fixture
def policy_criteria(tmp_path: Path) -> Path:
    criteria_path = tmp_path / "criterios_politica.json"
    payload = {
        "version": "ceplan-pn-e2e-v1",
        "tipo_informe": "politica_nacional",
        "metodologia": "indice",
        "escala": {
            "min": 0.0,
            "max": 2.0,
            "niveles": [
                {"valor": 0.0},
                {"valor": 0.5},
                {"valor": 1.0},
                {"valor": 1.5},
                {"valor": 2.0},
            ],
        },
        "escala_normalizada": {"min": 0.0, "max": 20.0},
        "secciones": [
            {
                "id": "resumen_ejecutivo",
                "titulo": "Resumen Ejecutivo",
                "ponderacion": 0.5,
                "dimensiones": [
                    {
                        "nombre": "Enfoque",
                        "ponderacion": 1.0,
                        "preguntas": [
                            {
                                "id": "PN_Q1",
                                "texto": "¿Cómo se articula la política?",
                                "ponderacion": 1.0,
                                "niveles": [
                                    {"valor": 0.0},
                                    {"valor": 0.5},
                                    {"valor": 1.0},
                                    {"valor": 1.5},
                                    {"valor": 2.0},
                                ],
                            }
                        ],
                    }
                ],
            },
            {
                "id": "gestion_ceplan",
                "titulo": "Gestión CEPLAN",
                "ponderacion": 0.3,
                "dimensiones": [
                    {
                        "nombre": "Monitoreo",
                        "ponderacion": 1.0,
                        "preguntas": [
                            {
                                "id": "PN_Q2",
                                "texto": "¿Qué seguimiento se realiza?",
                                "ponderacion": 1.0,
                                "niveles": [
                                    {"valor": 0.0},
                                    {"valor": 0.5},
                                    {"valor": 1.0},
                                    {"valor": 1.5},
                                    {"valor": 2.0},
                                ],
                            }
                        ],
                    }
                ],
            },
            {
                "id": POLICY_MISSING_SECTION_ID,
                "titulo": "Seguimiento complementario",
                "ponderacion": 0.2,
                "dimensiones": [
                    {
                        "nombre": "Sin evidencia",
                        "ponderacion": 1.0,
                        "preguntas": [
                            {
                                "id": "PN_Q3",
                                "texto": "¿Existe evidencia complementaria?",
                                "ponderacion": 1.0,
                                "niveles": [
                                    {"valor": 0.0},
                                    {"valor": 0.5},
                                    {"valor": 1.0},
                                    {"valor": 1.5},
                                    {"valor": 2.0},
                                ],
                            }
                        ],
                    }
                ],
            },
        ],
    }
    criteria_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return criteria_path


@pytest.fixture
def truncated_document(tmp_path: Path) -> Path:
    document_path = tmp_path / "informe_truncado.txt"
    long_body = " ".join(["inicio"] + ["contenido"] * 1500 + ["cierre"])
    document_path.write_text(
        "\n".join(
            [
                "[SECTION:resumen_ejecutivo]",
                long_body,
                "",
                "[SECTION:gestion_ceplan]",
                "Sección secundaria sin truncamiento.",
            ]
        ),
        encoding="utf-8",
    )
    return document_path


@pytest.fixture
def sample_criteria(tmp_path: Path) -> Path:
    criteria_path = tmp_path / "criterios.json"
    payload = {
        "version": "ceplan-e2e-v1",
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
                        "nombre": "Contexto",
                        "ponderacion": 1.0,
                        "preguntas": [
                            {
                                "id": "CEPLAN_Q1",
                                "texto": "¿Cómo se resume la estrategia?",
                                "ponderacion": 1.0,
                                "niveles": [{"valor": 4.0}, {"valor": 2.0}],
                            }
                        ],
                    }
                ],
            },
            {
                "id": "gestion_ceplan",
                "titulo": "Gestión CEPLAN",
                "ponderacion": 0.4,
                "dimensiones": [
                    {
                        "nombre": "Implementación",
                        "ponderacion": 1.0,
                        "preguntas": [
                            {
                                "id": "CEPLAN_Q2",
                                "texto": "¿Cómo se gestiona CEPLAN?",
                                "ponderacion": 1.0,
                                "niveles": [{"valor": 4.0}],
                            }
                        ],
                    }
                ],
            },
        ],
    }
    criteria_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return criteria_path


def test_pipeline_integration_e2e(
    tmp_path: Path,
    sample_document: Path,
    sample_criteria: Path,
    policy_criteria: Path,
    truncated_document: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = TrackingLoader()
    cleaner = TrackingCleaner()
    splitter = TrackingSplitter()
    prompt_builder = TrackingPromptBuilder()
    prompt_validator = TrackingPromptValidator()
    repository = TrackingRepository()
    ai_service = TrackingAIService()

    pd = pytest.importorskip("pandas")

    metrics_calls = _prepare_environment(monkeypatch)

    config = ServiceConfig(
        ai_provider="tracking",
        run_id="ceplan-e2e",
        model_name="ceplan-mock",
        prompt_batch_size=2,
        retries=0,
        timeout_seconds=10.0,
        log_level="ERROR",
    )

    service = EvaluationService(
        config=config,
        loader=loader,
        cleaner=cleaner,
        splitter=splitter,
        repository=repository,
        ai_service_factory=lambda _: ai_service,
        prompt_builder=prompt_builder,
        prompt_validator=prompt_validator,
    )
    original_threshold = evaluation_module.PROMPT_QUALITY_THRESHOLD
    export_formats = ["json", "csv", "xlsx"]
    output_paths = {
        "json": tmp_path / "resultado.json",
        "csv": tmp_path / "resultado.csv",
        "xlsx": tmp_path / "resultado.xlsx",
    }

    evaluations: Dict[str, EvaluationResult] = {}
    metrics_by_format: Dict[str, Dict[str, Any]] = {}

    for fmt in export_formats:
        evaluation, metrics = service.run(
            input_path=sample_document,
            criteria_path=sample_criteria,
            output_path=output_paths[fmt],
            output_format=fmt,
            extra_metadata={"invocation": "integration"},
        )
        evaluations[fmt] = evaluation
        metrics_by_format[fmt] = metrics

    criteria_payload = json.loads(sample_criteria.read_text(encoding="utf-8"))

    def _build_criteria_index(
        payload: Mapping[str, Any]
    ) -> tuple[
        Dict[str, Dict[str, Any]],
        Dict[str, int],
        int,
        Dict[tuple[str, str], Dict[str, Any]],
        List[str],
    ]:
        section_map: Dict[str, Dict[str, Any]] = {}
        section_question_counts: Dict[str, int] = {}
        question_metadata: Dict[tuple[str, str], Dict[str, Any]] = {}
        section_order: List[str] = []
        total = 0
        
        def _extract_scale_values(question_def: Mapping[str, Any]) -> List[Any]:
            direct_scale = question_def.get("escala") or question_def.get("valores_escala")
            if isinstance(direct_scale, Mapping):
                direct_scale = direct_scale.get("niveles")
            if isinstance(direct_scale, (list, tuple, set)) and direct_scale:
                extracted_direct: List[Any] = []
                for entry in direct_scale:
                    if isinstance(entry, Mapping) and "valor" in entry:
                        extracted_direct.append(entry.get("valor"))
                    else:
                        extracted_direct.append(entry)
                return extracted_direct
            niveles = question_def.get("niveles")
            if isinstance(niveles, list) and niveles:
                extracted: List[Any] = []
                for level in niveles:
                    if isinstance(level, Mapping) and "valor" in level:
                        extracted.append(level.get("valor"))
                    else:
                        extracted.append(level)
                return extracted
            return []

        for section_def in payload.get("secciones", []):
            section_id = section_def.get("id") or section_def.get("id_segmenter")
            if not section_id:
                continue
            section_order.append(section_id)
            section_weight = float(section_def.get("ponderacion", 1.0) or 0.0)
            dimension_map: Dict[str, Dict[str, Any]] = {}
            questions_in_section = 0
            for dimension_def in section_def.get("dimensiones", []):
                dimension_name = (
                    dimension_def.get("nombre")
                    or dimension_def.get("id")
                    or "dimension"
                )
                dimension_weight = float(dimension_def.get("ponderacion", 1.0) or 0.0)
                question_weights: Dict[str, float] = {}
                for question_def in dimension_def.get("preguntas", []):
                    question_id = question_def.get("id") or question_def.get("texto")
                    if not question_id:
                        continue
                    question_weight = float(question_def.get("ponderacion", 1.0) or 0.0)
                    question_weights[question_id] = question_weight
                    questions_in_section += 1
                    total += 1
                    question_metadata[(section_id, question_id)] = {
                        "section_id": section_id,
                        "dimension_name": dimension_name,
                        "dimension_id": dimension_def.get("id"),
                        "scale": _extract_scale_values(question_def),
                    }
                dimension_map[dimension_name] = {
                    "weight": dimension_weight,
                    "questions": question_weights,
                }
            section_map[section_id] = {
                "weight": section_weight,
                "dimensions": dimension_map,
            }
            section_question_counts[section_id] = questions_in_section
        return section_map, section_question_counts, total, question_metadata, section_order

    (
        section_index,
        questions_per_section,
        total_questions,
        question_metadata,
        criteria_section_order,
    ) = _build_criteria_index(
        criteria_payload
    )

    evaluation_one = evaluations["json"]
    evaluation_two = evaluations["csv"]
    metrics_one = metrics_by_format["json"]
    metrics_two = metrics_by_format["csv"]

    assert len(loader.calls) == len(export_formats)
    assert loader.calls[0]["path"] == str(sample_document)
    assert len(cleaner.calls) == len(export_formats)
    assert cleaner.calls[0]["header"] == "Resumen Ejecutivo - CEPLAN 2025"

    assert len(TrackingSegmenter.instances) == len(export_formats)
    first_segmenter_sections = TrackingSegmenter.instances[0].calls[0]["sections"]
    assert first_segmenter_sections == ["resumen_ejecutivo", "gestion_ceplan"]
    section_payloads = TrackingSegmenter.instances[0].calls[0]["sections_content"]
    assert "Conclusiones" in section_payloads["resumen_ejecutivo"]
    assert "1. Introducción El informe" in section_payloads["resumen_ejecutivo"]
    assert "Conclusiones" not in section_payloads["gestion_ceplan"]

    assert len(splitter.calls) == len(export_formats)
    assert all(call["chunk_count"] == 2 for call in splitter.calls)
    first_splitter_call = splitter.calls[0]
    assert first_splitter_call["chunk_sections"] == [
        "resumen_ejecutivo",
        "gestion_ceplan",
    ]
    assert "Conclusiones" in first_splitter_call["chunk_texts"][0]
    assert "Conclusiones" not in first_splitter_call["chunk_texts"][1]

    evaluation_section_order = [section.section_id for section in evaluation_one.sections]
    assert evaluation_section_order == criteria_section_order

    def _resolve_chunk_weight(metadata: Mapping[str, Any]) -> float | None:
        for key in ("weight", "ponderacion", "relevance", "peso"):
            value = metadata.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        nested = metadata.get("chunk_metadata")
        if isinstance(nested, Mapping):
            return _resolve_chunk_weight(nested)
        return None

    def _weighted_average(pairs: List[tuple[float, float | None]]) -> float | None:
        weighted_items: List[tuple[float, float]] = []
        fallback: List[float] = []
        for weight, value in pairs:
            if value is None:
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            fallback.append(numeric_value)
            try:
                numeric_weight = float(weight)
            except (TypeError, ValueError):
                continue
            if numeric_weight > 0:
                weighted_items.append((numeric_weight, numeric_value))
        if weighted_items:
            total_weight = sum(weight for weight, _ in weighted_items)
            if total_weight > 0:
                return sum(weight * value for weight, value in weighted_items) / total_weight
        if fallback:
            return sum(fallback) / len(fallback)
        return None

    def _aggregate_chunks(question: Any) -> float | None:
        weighted_items: List[tuple[float, float]] = []
        fallback: List[float] = []
        for chunk in question.chunk_results:
            score = chunk.score
            if score is None:
                continue
            try:
                numeric_score = float(score)
            except (TypeError, ValueError):
                continue
            fallback.append(numeric_score)
            metadata = chunk.metadata if isinstance(chunk.metadata, Mapping) else {}
            chunk_weight = _resolve_chunk_weight(metadata) or 0.0
            if chunk_weight > 0:
                weighted_items.append((chunk_weight, numeric_score))
        if weighted_items:
            total_weight = sum(weight for weight, _ in weighted_items)
            if total_weight > 0:
                return sum(weight * value for weight, value in weighted_items) / total_weight
        if fallback:
            return sum(fallback) / len(fallback)
        return None

    chunk_counts_per_section: Dict[str, int] = {}
    expected_question_scores: Dict[str, float | None] = {}
    expected_dimension_scores: Dict[tuple[str, str], float | None] = {}
    expected_section_scores: Dict[str, float | None] = {}
    section_pairs: List[tuple[float, float | None]] = []

    for section in evaluation_one.sections:
        section_id = section.section_id or ""
        section_definition = section_index.get(section_id)
        if not section_definition:
            continue
        dimension_pairs: List[tuple[float, float | None]] = []
        per_section_chunk_counts: List[int] = []
        for dimension in section.dimensions:
            dimension_definition = section_definition["dimensions"].get(dimension.name)
            if not dimension_definition:
                continue
            question_pairs: List[tuple[float, float | None]] = []
            for question in dimension.questions:
                expected_score = _aggregate_chunks(question)
                expected_question_scores[question.question_id] = expected_score
                assert question.score == pytest.approx(expected_score)
                per_section_chunk_counts.append(len(question.chunk_results))
                question_weight = dimension_definition["questions"].get(question.question_id, 1.0)
                question_pairs.append((question_weight, expected_score))
                for chunk in question.chunk_results:
                    metadata = chunk.metadata
                    assert metadata["prompt_was_valid"] is True
                    assert metadata["prompt_quality_score"] == pytest.approx(0.93)
                    assert metadata["provider"] == "tracking"
                    chunk_meta = metadata["chunk_metadata"]
                    assert chunk_meta["document_metadata"]["cleaning_report"][
                        "preserved_headers"
                    ][0] == "Resumen Ejecutivo - CEPLAN 2025"
            expected_dimension_score = _weighted_average(question_pairs)
            expected_dimension_scores[(section_id, dimension.name)] = expected_dimension_score
            assert dimension.score == pytest.approx(expected_dimension_score)
            dimension_weight = dimension_definition.get("weight", 1.0)
            dimension_pairs.append((dimension_weight, expected_dimension_score))
        expected_section_score = _weighted_average(dimension_pairs)
        expected_section_scores[section_id] = expected_section_score
        assert section.score == pytest.approx(expected_section_score)
        section_weight = section_definition.get("weight", 1.0)
        section_pairs.append((section_weight, expected_section_score))
        chunk_counts_per_section[section_id] = max(per_section_chunk_counts or [0])

    expected_global_score = _weighted_average(section_pairs)
    assert evaluation_one.score == pytest.approx(expected_global_score)

    scale = criteria_payload.get("metrica_global", {}).get("escala_resultado", {})
    scale_min = float(scale.get("min", 0.0))
    scale_max = float(scale.get("max", 4.0))
    expected_normalized = None
    if expected_global_score is not None and scale_max > scale_min:
        expected_normalized = (
            (expected_global_score - scale_min) / (scale_max - scale_min)
        ) * 100.0

    assert metrics_one["global"]["raw_score"] == pytest.approx(expected_global_score)
    if expected_normalized is not None:
        assert metrics_one["global"]["normalized_score"] == pytest.approx(expected_normalized)

    metrics_section_order = [entry["section_id"] for entry in metrics_one["sections"]]
    assert metrics_section_order == criteria_section_order
    metrics_sections = {entry["section_id"]: entry for entry in metrics_one["sections"]}
    for section_id, expected_score in expected_section_scores.items():
        section_metrics = metrics_sections[section_id]
        assert section_metrics["score"] == pytest.approx(expected_score)
        if expected_score is not None and scale_max > scale_min:
            expected_section_normalized = (
                (expected_score - scale_min) / (scale_max - scale_min)
            ) * 100.0
            assert section_metrics["normalized_score"] == pytest.approx(
                expected_section_normalized
            )

    assert len(metrics_calls) == len(export_formats)
    assert metrics_calls[0]["sections"] == first_segmenter_sections

    detected_sections = first_segmenter_sections
    expected_prompts_per_run = sum(
        chunk_counts_per_section.get(section_id, 0)
        * questions_per_section.get(section_id, 0)
        for section_id in detected_sections
    )
    total_expected_prompt_calls = expected_prompts_per_run * len(export_formats)


    assert len(prompt_builder.calls) == total_expected_prompt_calls
    assert len(prompt_validator.calls) == total_expected_prompt_calls
    assert len(ai_service.calls) == total_expected_prompt_calls

    for record in prompt_builder.calls[:expected_prompts_per_run]:
        assert record["section_id"] in {"resumen_ejecutivo", "gestion_ceplan"}
        assert record["chunk_section"] in {"resumen_ejecutivo", "gestion_ceplan"}

    observed_chunk_sections = {entry["chunk_section"] for entry in prompt_builder.calls}
    assert observed_chunk_sections == {"resumen_ejecutivo", "gestion_ceplan"}

    assert len(prompt_builder.calls) == len(prompt_validator.calls)

    first_run_builder_calls = prompt_builder.calls[:expected_prompts_per_run]
    first_run_validations = prompt_validator.calls[:expected_prompts_per_run]
    first_run_ai_calls = ai_service.calls[:expected_prompts_per_run]

    for builder_record, validation in zip(first_run_builder_calls, first_run_validations):
        context = validation["context"]
        section_id = builder_record["section_id"]
        question_id = builder_record["question_id"]
        assert (section_id, question_id) in question_metadata
        metadata = question_metadata[(section_id, question_id)]

        assert context["report_type"] == "institucional"
        assert context["chunk_index"] in {0, 1}
        assert context["section_id"] == section_id
        assert context["question_id"] == question_id
        assert context.get("dimension_name") == metadata["dimension_name"]

        dimension_id = metadata.get("dimension_id")
        if dimension_id:
            assert context.get("dimension_id") == dimension_id
        else:
            assert "dimension_id" not in context or context.get("dimension_id") in {None, ""}

        expected_scale_values = metadata.get("scale") or []
        if expected_scale_values:
            assert context["expected_scale_values"] == expected_scale_values
        else:
            assert "expected_scale_values" not in context or context.get("expected_scale_values") in (None, [], ())

        assert context.get("was_truncated") in (None, False)
        assert "truncation_marker" not in context

    for builder_record, validation, ai_call in zip(
        first_run_builder_calls, first_run_validations, first_run_ai_calls
    ):
        assert ai_call["question"] == builder_record["question_id"]
        assert ai_call["section"] == builder_record["section_id"]
        assert ai_call["chunk_index"] == validation["context"]["chunk_index"]
        assert ai_call["prompt"].startswith("CEPLAN::")

    for call in first_run_ai_calls:
        assert call["question"] in {"CEPLAN_Q1", "CEPLAN_Q2"}
        assert call["chunk_index"] in {0, 1}

    assert evaluation_one.document_id == evaluation_two.document_id
    assert len({evaluation.document_id for evaluation in evaluations.values()}) == 1
    assert evaluation_one.document_type == "institucional"
    assert evaluation_one.methodology == "ponderada"
    assert evaluation_one.metadata["model_name"] == "ceplan-mock"
    assert evaluation_one.metadata["prompt_batch_size"] == 2
    assert evaluation_one.metadata["validator_version"] == getattr(
        PromptValidator, "VERSION", "unknown"
    )
    assert evaluation_one.metadata["runs"][-1]["run_id"] == "ceplan-e2e"
    summary = evaluation_one.metadata.get("prompt_validation", {})
    assert summary.get("total_prompts") == expected_prompts_per_run
    assert summary.get("rejected_prompts") == 0
    assert summary.get(
        "sections"
    ), "Se espera resumen de secciones en la validación de prompts"
    assert summary.get("threshold") == pytest.approx(original_threshold)

    assert evaluation_one.metadata["builder_version"] == type(prompt_builder).__name__
    assert evaluation_one.metadata["criteria_version"] == criteria_payload["version"]
    assert evaluation_one.metadata["pipeline_version"] == evaluation_module.SERVICE_VERSION
    assert evaluation_one.metadata["model_name"] == "ceplan-mock"

    chunk_latencies: List[float] = []
    token_counts: List[int] = []
    for section in evaluation_one.sections:
        for dimension in section.dimensions:
            for question in dimension.questions:
                for chunk in question.chunk_results:
                    metadata = chunk.metadata
                    latency = metadata.get("ai_latency_ms")
                    if latency is not None:
                        chunk_latencies.append(float(latency))
                    tokens = metadata.get("response_tokens")
                    if tokens is not None:
                        token_counts.append(int(tokens))

    assert chunk_latencies, "Se esperaban latencias registradas en los metadatos"
    average_latency = sum(chunk_latencies) / len(chunk_latencies)
    assert average_latency > 0
    if summary.get("average_latency_ms") is not None:
        assert summary["average_latency_ms"] == pytest.approx(average_latency)

    assert token_counts, "Se esperaban tokens de respuesta en los metadatos"
    assert token_counts == sorted(token_counts)
    assert token_counts[0] > 0

    aggregated_token_counts: List[int] = []
    for evaluation in evaluations.values():
        for section in evaluation.sections:
            for dimension in section.dimensions:
                for question in dimension.questions:
                    for chunk in question.chunk_results:
                        tokens = chunk.metadata.get("response_tokens")
                        if tokens is not None:
                            aggregated_token_counts.append(int(tokens))

    assert aggregated_token_counts == sorted(aggregated_token_counts)
    assert aggregated_token_counts[0] >= token_counts[0]

    assert evaluation_one.score == pytest.approx(evaluation_two.score)
    assert metrics_one["global"]["raw_score"] == pytest.approx(evaluation_one.score)
    assert metrics_two["global"]["raw_score"] == pytest.approx(evaluation_two.score)

    assert len(repository.exports) == len(export_formats)
    assert repository.exports[0]["output_path"] == output_paths["json"]
    assert (
        repository.exports[0]["extra_metadata"]["config"]["model_name"]
        == "ceplan-mock"
    )

    exports_before_error = len(repository.exports)
    unsupported_output = tmp_path / "resultado_unsupported.foo"
    with pytest.raises(ValueError) as exc_info:
        service.run(
            input_path=sample_document,
            criteria_path=sample_criteria,
            output_path=unsupported_output,
            output_format="foo",
            extra_metadata={"invocation": "unsupported"},
        )

    assert "Formato de salida no soportado" in str(exc_info.value)
    assert not unsupported_output.exists()
    assert len(repository.exports) == exports_before_error

    unicode_document = tmp_path / "informe_unicode.txt"
    unicode_document.write_text(
        "\n".join(
            [
                "[SECTION:resumen_ejecutivo]",
                "   Rësúmen\u00a0General ☂️ 2025  ",
                "Contenido relevante con acentos y símbolos.",
                "",
                "[SECTION:gestion_ceplan]",
                "Seguimiento complementario sin modificaciones.",
            ]
        ),
        encoding="utf-8",
    )
    unicode_output = tmp_path / "resultado_unicode.json"
    service.run(
        input_path=unicode_document,
        criteria_path=sample_criteria,
        output_path=unicode_output,
        output_format="json",
        extra_metadata={"invocation": "unicode"},
    )
    assert cleaner.calls[-1]["header"] == "Rësúmen\u00a0General ☂️ 2025"
    assert "☂️" in cleaner.calls[-1]["header"]

    policy_payload = json.loads(policy_criteria.read_text(encoding="utf-8"))
    (
        policy_section_index,
        _,
        _,
        _policy_question_metadata,
        policy_section_order,
    ) = _build_criteria_index(policy_payload)
    policy_output = tmp_path / "resultado_politica.json"
    evaluation_policy, metrics_policy = service.run(
        input_path=sample_document,
        criteria_path=policy_criteria,
        output_path=policy_output,
        output_format="json",
        extra_metadata={"invocation": "policy"},
    )

    assert evaluation_policy.document_type == "politica_nacional"
    assert metrics_policy["methodology"] == "politica_nacional"
    policy_summary = evaluation_policy.metadata.get("prompt_validation", {})
    assert policy_summary.get("total_prompts", 0) > 0
    assert policy_summary.get("rejected_prompts", 0) == 0

    evaluation_policy_order = [section.section_id for section in evaluation_policy.sections]
    assert evaluation_policy_order == policy_section_order
    policy_metrics_order = [entry["section_id"] for entry in metrics_policy["sections"]]
    assert policy_metrics_order == policy_section_order

    policy_section_pairs: List[tuple[float, float | None]] = []
    half_point_scores: List[float] = []
    for section in evaluation_policy.sections:
        section_id = section.section_id or ""
        definition = policy_section_index.get(section_id)
        if not definition:
            continue
        dimension_pairs: List[tuple[float, float | None]] = []
        for dimension in section.dimensions:
            dimension_definition = definition["dimensions"].get(dimension.name)
            if not dimension_definition:
                continue
            dimension_weight = dimension_definition.get("weight", 1.0)
            question_pairs: List[tuple[float, float | None]] = []
            for question in dimension.questions:
                expected_score = _aggregate_chunks(question)
                assert question.score == pytest.approx(expected_score)
                question_weight = dimension_definition["questions"].get(
                    question.question_id, 1.0
                )
                question_pairs.append((question_weight, expected_score))
                if section_id == POLICY_MISSING_SECTION_ID:
                    assert question.justification
                    assert "sin evidencia" in question.justification.lower()
                for chunk in question.chunk_results:
                    if chunk.score is not None:
                        half_point_scores.append(float(chunk.score))
                    if section_id == POLICY_MISSING_SECTION_ID:
                        assert chunk.score == pytest.approx(0.0)
                        assert chunk.justification
                        assert "sin evidencia" in chunk.justification.lower()
                        assert chunk.metadata.get("missing_section") is True
            expected_dimension_score = _weighted_average(question_pairs)
            assert dimension.score == pytest.approx(expected_dimension_score)
            dimension_pairs.append((dimension_weight, expected_dimension_score))
        expected_section_score = _weighted_average(dimension_pairs)
        assert section.score == pytest.approx(expected_section_score)
        policy_section_pairs.append((definition.get("weight", 1.0), expected_section_score))
        if section_id == POLICY_MISSING_SECTION_ID:
            assert expected_section_score == pytest.approx(0.0)

    expected_policy_global = _weighted_average(policy_section_pairs)
    policy_scale = policy_payload.get("escala", {})
    policy_min = float(policy_scale.get("min", 0.0))
    policy_max = float(policy_scale.get("max", 2.0))
    assert metrics_policy["global"]["raw_score"] == pytest.approx(expected_policy_global)
    assert metrics_policy["global"]["normalized_min"] == pytest.approx(0.0)
    assert metrics_policy["global"]["normalized_max"] == pytest.approx(20.0)
    if expected_policy_global is not None and policy_max > policy_min:
        expected_policy_normalized = (
            (expected_policy_global - policy_min) / (policy_max - policy_min)
        ) * 20.0
        assert metrics_policy["global"]["normalized_score"] == pytest.approx(
            expected_policy_normalized
        )

    policy_metrics_sections = {
        entry["section_id"]: entry for entry in metrics_policy["sections"]
    }
    missing_metrics = policy_metrics_sections[POLICY_MISSING_SECTION_ID]
    assert missing_metrics["score"] == pytest.approx(0.0)
    assert missing_metrics["normalized_score"] == pytest.approx(0.0)

    assert any(
        not math.isclose(score % 1.0, 0.0, abs_tol=1e-6)
        and math.isclose((score * 2.0) % 1.0, 0.0, abs_tol=1e-6)
        for score in half_point_scores
    )

    truncating_validator = TrackingPromptValidator()
    truncating_splitter = TrackingSplitter(truncation_limit=300)
    truncating_service = EvaluationService(
        config=ServiceConfig(
            ai_provider="tracking",
            run_id="ceplan-e2e-trunc",
            model_name="ceplan-mock",
            prompt_batch_size=1,
            retries=0,
            timeout_seconds=10.0,
            log_level="ERROR",
        ),
        loader=TrackingLoader(),
        cleaner=TrackingCleaner(),
        splitter=truncating_splitter,
        repository=TrackingRepository(),
        ai_service_factory=lambda _: TrackingAIService(),
        prompt_builder=TrackingPromptBuilder(),
        prompt_validator=truncating_validator,
    )

    trunc_output = tmp_path / "resultado_truncado.json"
    evaluation_truncated, _ = truncating_service.run(
        input_path=truncated_document,
        criteria_path=sample_criteria,
        output_path=trunc_output,
        output_format="json",
        extra_metadata={"invocation": "truncation"},
    )

    trunc_summary = evaluation_truncated.metadata.get("prompt_validation", {})
    assert trunc_summary.get("rejected_prompts", 0) == 0
    assert any(
        call["context"].get("was_truncated") is True
        for call in truncating_validator.calls
    )

    truncated_chunk_detected = False
    truncated_expectations: List[Dict[str, Any]] = []
    for section in evaluation_truncated.sections:
        section_id = section.section_id or ""
        for dimension in section.dimensions:
            for question in dimension.questions:
                question_id = question.question_id or ""
                assert (section_id, question_id) in question_metadata
                metadata = question_metadata[(section_id, question_id)]
                for chunk in question.chunk_results:
                    chunk_meta = chunk.metadata.get("chunk_metadata", {})
                    is_truncated = bool(chunk_meta.get("was_truncated"))
                    if is_truncated:
                        truncated_chunk_detected = True
                        assert chunk.metadata.get("prompt_rejected") is not True
                        assert chunk.metadata.get("prompt_was_valid") is True
                        assert chunk_meta.get("truncation_marker") == "… [texto truncado]"
                        assert chunk_meta.get("head_preview")
                        assert chunk_meta.get("tail_preview")
                        assert (
                            chunk_meta.get("original_length", 0)
                            > truncating_splitter.truncation_limit
                        )
                    truncated_expectations.append(
                        {
                            "section_id": section_id,
                            "question_id": question_id,
                            "scale": metadata.get("scale") or [],
                            "is_truncated": is_truncated,
                            "marker": chunk_meta.get("truncation_marker") if is_truncated else None,
                        }
                    )
    assert truncated_chunk_detected

    assert len(truncating_validator.calls) == len(truncated_expectations)
    for expected, validation in zip(truncated_expectations, truncating_validator.calls):
        context = validation["context"]
        assert context["section_id"] == expected["section_id"]
        assert context["question_id"] == expected["question_id"]
        scale_values = expected["scale"]
        if scale_values:
            assert context["expected_scale_values"] == scale_values
        else:
            assert "expected_scale_values" not in context or context.get("expected_scale_values") in (None, [], ())
        if expected["is_truncated"]:
            assert context.get("was_truncated") is True
            assert context.get("truncation_marker") == expected["marker"]
        else:
            assert context.get("was_truncated") in (None, False)
            assert "truncation_marker" not in context or not context.get("truncation_marker")

    flaky_ai_service = TrackingAIService()
    retry_attempts = {"count": 0}
    original_retry_evaluate = flaky_ai_service.evaluate

    def flaky_evaluate(prompt: str, **kwargs: Any) -> Dict[str, Any]:
        retry_attempts["count"] += 1
        if retry_attempts["count"] == 1:
            raise RuntimeError("transient failure")
        return original_retry_evaluate(prompt, **kwargs)

    monkeypatch.setattr(flaky_ai_service, "evaluate", flaky_evaluate)

    retry_service = EvaluationService(
        config=ServiceConfig(
            ai_provider="tracking",
            run_id="ceplan-e2e-retry",
            model_name="ceplan-mock",
            prompt_batch_size=1,
            retries=1,
            timeout_seconds=10.0,
            log_level="ERROR",
        ),
        loader=TrackingLoader(),
        cleaner=TrackingCleaner(),
        splitter=TrackingSplitter(),
        repository=TrackingRepository(),
        ai_service_factory=lambda _: flaky_ai_service,
        prompt_builder=TrackingPromptBuilder(),
        prompt_validator=TrackingPromptValidator(),
    )

    retry_output = tmp_path / "resultado_retry.json"
    evaluation_retry, metrics_retry = retry_service.run(
        input_path=sample_document,
        criteria_path=sample_criteria,
        output_path=retry_output,
        output_format="json",
        extra_metadata={"invocation": "retry"},
    )

    assert retry_attempts["count"] == len(flaky_ai_service.calls) + 1
    assert len(flaky_ai_service.calls) == expected_prompts_per_run
    assert evaluation_retry.score is not None
    assert evaluation_retry.metadata["retries"] == 1
    retry_summary = evaluation_retry.metadata.get("prompt_validation", {})
    assert retry_summary.get("average_latency_ms")
    assert retry_summary.get("average_latency_ms") > 0
    for section in evaluation_retry.sections:
        for dimension in section.dimensions:
            for question in dimension.questions:
                for chunk in question.chunk_results:
                    assert chunk.metadata.get("ai_latency_ms", 0) >= 0

    assert metrics_retry["global"]["raw_score"] == pytest.approx(
        evaluation_retry.score
    )

    exported_json = json.loads(output_paths["json"].read_text(encoding="utf-8"))
    assert exported_json["evaluation"]["metadata"]["model_name"] == "ceplan-mock"
    assert exported_json["metrics"]["methodology"] == "institucional"
    assert exported_json["extra"]["invocation"] == "integration"
    assert exported_json["extra"]["tipo_informe"] == "institucional"
    assert exported_json["extra"]["pipeline_version"] == evaluation_module.SERVICE_VERSION
    exported_sections_order = [
        section["section_id"] for section in exported_json["evaluation"]["sections"]
    ]
    assert exported_sections_order == criteria_section_order

    csv_df = pd.read_csv(output_paths["csv"])
    assert len(csv_df) >= total_questions
    csv_columns = [column.lower() for column in csv_df.columns]
    assert any("section" in column or "seccion" in column for column in csv_columns)
    assert any("criter" in column for column in csv_columns)
    assert any("question" in column or "pregunta" in column for column in csv_columns)
    assert any("score" in column or "puntaje" in column for column in csv_columns)
    assert any("justification" in column for column in csv_columns)
    assert any("prompt" in column for column in csv_columns)

    xlsx_preguntas = pd.read_excel(output_paths["xlsx"], sheet_name="preguntas")
    assert len(xlsx_preguntas) >= total_questions
    xlsx_columns = [column.lower() for column in xlsx_preguntas.columns]
    assert any("section" in column or "seccion" in column for column in xlsx_columns)
    assert any("question" in column or "pregunta" in column for column in xlsx_columns)
    assert any("score" in column or "puntaje" in column for column in xlsx_columns)
    assert any("prompt" in column for column in xlsx_columns)

    xlsx_header = pd.read_excel(output_paths["xlsx"], sheet_name="indice_global")
    header_columns = [column.lower() for column in xlsx_header.columns]
    assert any("model" in column or "modelo" in column for column in header_columns)
    assert any("tipo" in column and "informe" in column for column in header_columns)
    assert xlsx_header.iloc[0]["model_name"] == "ceplan-mock"

    assert evaluation_one.to_dict()["metadata"]["model_name"] == "ceplan-mock"
    assert isinstance(evaluation_one, EvaluationResult)

    high_threshold = 0.95
    monkeypatch.setattr(evaluation_module, "PROMPT_QUALITY_THRESHOLD", high_threshold)
    threshold_service = EvaluationService(
        config=ServiceConfig(
            ai_provider="tracking",
            run_id="ceplan-e2e-threshold",
            model_name="ceplan-mock",
            prompt_batch_size=1,
            retries=0,
            timeout_seconds=10.0,
            log_level="ERROR",
        ),
        loader=TrackingLoader(),
        cleaner=TrackingCleaner(),
        splitter=TrackingSplitter(),
        repository=TrackingRepository(),
        ai_service_factory=lambda _: TrackingAIService(),
        prompt_builder=TrackingPromptBuilder(),
        prompt_validator=TrackingPromptValidator(),
    )

    high_threshold_output = tmp_path / "resultado_threshold_alto.json"
    evaluation_threshold, _ = threshold_service.run(
        input_path=sample_document,
        criteria_path=sample_criteria,
        output_path=high_threshold_output,
        output_format="json",
        extra_metadata={"invocation": "threshold"},
    )

    threshold_summary = evaluation_threshold.metadata.get("prompt_validation", {})
    assert threshold_summary.get("threshold") == pytest.approx(high_threshold)
    assert threshold_summary.get("total_prompts") == expected_prompts_per_run
    assert threshold_summary.get("rejected_prompts") == expected_prompts_per_run

    for section in evaluation_threshold.sections:
        for dimension in section.dimensions:
            for question in dimension.questions:
                for chunk in question.chunk_results:
                    assert chunk.metadata.get("prompt_rejected") is True
                    assert chunk.metadata.get("prompt_was_valid") is True
                    assert chunk.score is None

    monkeypatch.setattr(
        evaluation_module, "PROMPT_QUALITY_THRESHOLD", original_threshold
    )

    summary_before_rejection = summary.get("rejected_prompts", 0)
    ai_calls_before_rejection = len(ai_service.calls)
    prompt_calls_before_rejection = len(prompt_builder.calls)

    def _rejecting_validate(
        self, prompt: str, context: Mapping[str, Any]
    ) -> PromptValidationResult:
        self.calls.append({"prompt": prompt, "context": dict(context)})
        return PromptValidationResult(
            is_valid=False,
            quality_score=0.2,
            alerts=["prompt rechazado"],
            metadata={"reason": "integration-test"},
        )

    monkeypatch.setattr(TrackingPromptValidator, "validate", _rejecting_validate)

    rejected_output = tmp_path / "resultado_rechazado.json"
    evaluation_rejected, metrics_rejected = service.run(
        input_path=sample_document,
        criteria_path=sample_criteria,
        output_path=rejected_output,
        output_format="json",
        extra_metadata={"invocation": "integration"},
    )

    assert len(prompt_builder.calls) == prompt_calls_before_rejection + expected_prompts_per_run
    assert len(prompt_validator.calls) == prompt_calls_before_rejection + expected_prompts_per_run
    assert len(ai_service.calls) == ai_calls_before_rejection

    rejection_summary = evaluation_rejected.metadata.get("prompt_validation", {})
    assert rejection_summary.get("rejected_prompts", 0) > summary_before_rejection
    assert rejection_summary.get("total_prompts") == expected_prompts_per_run

    for section in evaluation_rejected.sections:
        for dimension in section.dimensions:
            for question in dimension.questions:
                for chunk in question.chunk_results:
                    assert chunk.metadata.get("prompt_rejected") is True
                    assert chunk.metadata.get("prompt_was_valid") is False
                    assert chunk.score is None

    rejected_score = evaluation_rejected.score
    if rejected_score is not None:
        assert rejected_score == pytest.approx(0.0)
    else:
        assert metrics_rejected["global"]["raw_score"] is None

    exported_rejected = json.loads(rejected_output.read_text(encoding="utf-8"))
    assert exported_rejected["evaluation"]["metadata"]["prompt_validation"][
        "rejected_prompts"
    ] == rejection_summary.get("rejected_prompts")


@pytest.mark.slow
def test_pipeline_integration_real_ai_tolerance(
    tmp_path: Path,
    sample_document: Path,
    sample_criteria: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_environment(monkeypatch)

    def _execute(run_id: str, ai_service: TrackingAIService) -> EvaluationResult:
        service = EvaluationService(
            config=ServiceConfig(
                ai_provider="tracking",
                run_id=run_id,
                model_name="ceplan-mock",
                prompt_batch_size=1,
                retries=0,
                timeout_seconds=10.0,
                log_level="ERROR",
            ),
            loader=TrackingLoader(),
            cleaner=TrackingCleaner(),
            splitter=TrackingSplitter(),
            repository=TrackingRepository(),
            ai_service_factory=lambda _: ai_service,
            prompt_builder=TrackingPromptBuilder(),
            prompt_validator=TrackingPromptValidator(),
        )
        evaluation, _ = service.run(
            input_path=sample_document,
            criteria_path=sample_criteria,
            output_path=tmp_path / f"resultado_{run_id}.json",
            output_format="json",
            extra_metadata={"invocation": run_id},
        )
        return evaluation

    evaluation_a = _execute("ceplan-e2e-real-a", SlightlyDriftingAIService(0.05))
    evaluation_b = _execute("ceplan-e2e-real-b", SlightlyDriftingAIService(-0.03))

    assert evaluation_a.score is not None and evaluation_b.score is not None
    difference = abs(evaluation_a.score - evaluation_b.score)
    assert difference < 0.2
    assert difference > 0

# pytest tests/test_pipeline_integration.py -v