"""Pruebas de integración extremo a extremo para el pipeline CEPLAN."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping

import pytest

from data.models.document import Document
from data.models.evaluation import EvaluationResult
from data.models.evaluator import Evaluator
from metrics import calculate_metrics as real_calculate_metrics
from reporting.repository import EvaluationRepository
from services import evaluation_service as evaluation_module
from services.evaluation_service import EvaluationService, ServiceConfig
from utils.prompt_validator import PromptValidationResult, PromptValidator


class TrackingLoader:
    """Loader simplificado que conserva el contenido original."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

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

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

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
                "tipo": self.tipo,
            }
        )
        return document


class TrackingSplitter:
    """Crea chunks sintéticos manteniendo metadatos estructurados."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def split_document(self, document: Document) -> Document:
        chunks: List[SimpleNamespace] = []
        for index, (section_id, text) in enumerate(document.sections.items()):
            chunk_metadata = {
                "section_id": section_id,
                "source_id": f"{section_id}-chunk-{index}",
                "document_metadata": {
                    "id": document.metadata.get("id"),
                    "cleaning_report": document.metadata.get("cleaning_report"),
                },
            }
            chunks.append(SimpleNamespace(page_content=text, metadata=chunk_metadata))
        if not chunks:
            chunk_metadata = {
                "section_id": "general",
                "source_id": "general-0",
                "document_metadata": {
                    "id": document.metadata.get("id"),
                    "cleaning_report": document.metadata.get("cleaning_report"),
                },
            }
            chunks.append(SimpleNamespace(page_content=document.content, metadata=chunk_metadata))
        document.chunks = chunks
        self.calls.append(
            {
                "document": document.metadata.get("source"),
                "chunk_count": len(chunks),
            }
        )
        return document


class TrackingPromptBuilder:
    """Builder determinista que conserva el contexto recibido."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

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

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

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

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.model_name = "tracking-model"

    def evaluate(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        call = {
            "prompt": prompt,
            "question": kwargs.get("question", {}).get("id"),
            "section": kwargs.get("section", {}).get("id"),
            "chunk_index": kwargs.get("chunk_index"),
        }
        self.calls.append(call)
        chunk_index = kwargs.get("chunk_index") or 0
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

    def __init__(self) -> None:
        super().__init__()
        self.exports: List[Dict[str, Any]] = []

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
        self.exports.append(record)
        return super().export(
            evaluation,
            metrics_summary,
            output_path=output_path,
            output_format=output_format,
            extra_metadata=extra_metadata,
        )


@pytest.fixture
def sample_document(tmp_path: Path) -> Path:
    document_path = tmp_path / "informe_ceplan.txt"
    document_path.write_text(
        "\n".join(
            [
                "[SECTION:resumen_ejecutivo]",
                "Resumen Ejecutivo - CEPLAN 2025",
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = TrackingLoader()
    cleaner = TrackingCleaner()
    splitter = TrackingSplitter()
    prompt_builder = TrackingPromptBuilder()
    prompt_validator = TrackingPromptValidator()
    repository = TrackingRepository()
    ai_service = TrackingAIService()

    TrackingSegmenter.instances = []
    monkeypatch.setattr(evaluation_module, "Segmenter", TrackingSegmenter)

    original_evaluator = evaluation_module.ValidatingEvaluator

    class FixedValidatingEvaluator(original_evaluator):
        def __init__(
            self,
            *args: Any,
            prompt_validator: PromptValidator,
            prompt_quality_threshold: float,
            prompt_batch_size: int = 1,
            **kwargs: Any,
        ) -> None:
            prompt_builder = kwargs.get("prompt_builder")
            extra_instructions = kwargs.get("extra_instructions")
            ai_client = getattr(evaluation_module, "_integration_ai_service", None)
            if ai_client is None:
                raise RuntimeError("AI service must be resolved before instantiating the evaluator")
            Evaluator.__init__(
                self,
                ai_client,
                prompt_builder=prompt_builder,
                extra_instructions=extra_instructions,
            )
            self.prompt_validator = prompt_validator
            self.prompt_quality_threshold = max(0.0, min(1.0, prompt_quality_threshold))
            self.prompt_batch_size = max(1, int(prompt_batch_size))

    monkeypatch.setattr(evaluation_module, "ValidatingEvaluator", FixedValidatingEvaluator)

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

    original_resolve_ai_service = EvaluationService._resolve_ai_service

    def tracking_resolve_ai_service(self, config: ServiceConfig):
        client = original_resolve_ai_service(self, config)
        setattr(evaluation_module, "_integration_ai_service", client)
        return client

    monkeypatch.setattr(EvaluationService, "_resolve_ai_service", tracking_resolve_ai_service)

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

    output_path = tmp_path / "resultado.json"
    second_output_path = tmp_path / "resultado_segundo.json"

    evaluation_one, metrics_one = service.run(
        input_path=sample_document,
        criteria_path=sample_criteria,
        output_path=output_path,
        output_format="json",
        extra_metadata={"invocation": "integration"},
    )

    evaluation_two, metrics_two = service.run(
        input_path=sample_document,
        criteria_path=sample_criteria,
        output_path=second_output_path,
        output_format="json",
        extra_metadata={"invocation": "integration"},
    )

    assert len(loader.calls) == 2
    assert loader.calls[0]["path"] == str(sample_document)
    assert len(cleaner.calls) == 2
    assert cleaner.calls[0]["header"] == "Resumen Ejecutivo - CEPLAN 2025"

    assert len(TrackingSegmenter.instances) == 2
    assert TrackingSegmenter.instances[0].calls[0]["sections"] == [
        "resumen_ejecutivo",
        "gestion_ceplan",
    ]

    assert len(splitter.calls) == 2
    assert all(call["chunk_count"] == 2 for call in splitter.calls)

    expected_prompt_calls = 2 * 2 * 2
    expected_prompts_per_run = expected_prompt_calls // 2
    assert len(prompt_builder.calls) == expected_prompt_calls
    assert len(prompt_validator.calls) == expected_prompt_calls
    assert len(ai_service.calls) == expected_prompt_calls

    for record in prompt_builder.calls[:2]:
        assert record["section_id"] in {"resumen_ejecutivo", "gestion_ceplan"}
        assert record["chunk_section"] in {"resumen_ejecutivo", "gestion_ceplan"}

    observed_chunk_sections = {entry["chunk_section"] for entry in prompt_builder.calls}
    assert observed_chunk_sections == {"resumen_ejecutivo", "gestion_ceplan"}

    for validation in prompt_validator.calls[:2]:
        context = validation["context"]
        assert context["report_type"] == "institucional"
        assert context["chunk_index"] in {0, 1}

    for call in ai_service.calls[:2]:
        assert call["question"] in {"CEPLAN_Q1", "CEPLAN_Q2"}
        assert call["chunk_index"] in {0, 1}

    assert len(metrics_calls) == 2
    assert metrics_calls[0]["sections"] == ["resumen_ejecutivo", "gestion_ceplan"]

    assert evaluation_one.document_id == evaluation_two.document_id
    assert evaluation_one.document_type == "institucional"
    assert evaluation_one.methodology == "ponderada"
    assert evaluation_one.metadata["model_name"] == "ceplan-mock"
    assert evaluation_one.metadata["prompt_batch_size"] == 2
    assert evaluation_one.metadata["validator_version"] == getattr(PromptValidator, "VERSION", "unknown")
    assert evaluation_one.metadata["runs"][-1]["run_id"] == "ceplan-e2e"
    summary = evaluation_one.metadata.get("prompt_validation", {})
    assert summary.get("total_prompts") == expected_prompts_per_run
    assert summary.get("rejected_prompts") == 0
    assert summary.get("sections"), "Se espera resumen de secciones en la validación de prompts"

    assert evaluation_one.score is not None
    assert evaluation_two.score is not None
    assert evaluation_one.score == pytest.approx(evaluation_two.score)
    assert metrics_one["global"]["raw_score"] == pytest.approx(evaluation_one.score)
    assert metrics_two["global"]["raw_score"] == pytest.approx(evaluation_two.score)

    assert repository.exports[0]["output_path"] == output_path
    assert repository.exports[0]["extra_metadata"]["config"]["model_name"] == "ceplan-mock"
    assert len(repository.exports) == 2

    exported = json.loads(output_path.read_text(encoding="utf-8"))
    assert exported["evaluation"]["metadata"]["model_name"] == "ceplan-mock"
    assert exported["metrics"]["methodology"] == "institucional"
    assert exported["extra"]["invocation"] == "integration"

    for section in evaluation_one.sections:
        assert section.section_id in {"resumen_ejecutivo", "gestion_ceplan"}
        for dimension in section.dimensions:
            for question in dimension.questions:
                assert question.score is not None
                for chunk in question.chunk_results:
                    metadata = chunk.metadata
                    assert metadata["prompt_was_valid"] is True
                    assert metadata["prompt_quality_score"] == pytest.approx(0.93)
                    assert metadata["provider"] == "tracking"
                    chunk_meta = metadata["chunk_metadata"]
                    assert chunk_meta["document_metadata"]["cleaning_report"][
                        "preserved_headers"
                    ][0] == "Resumen Ejecutivo - CEPLAN 2025"

    assert evaluation_one.to_dict()["metadata"]["model_name"] == "ceplan-mock"
    assert isinstance(evaluation_one, EvaluationResult)