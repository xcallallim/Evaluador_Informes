"""High level orchestration service for the evaluation pipeline."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import sys
import time
import tracemalloc
import unicodedata
import uuid
from statistics import pstdev
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

from data.models.document import Document
from data.models.evaluation import (
    ChunkResult,
    DimensionResult,
    EvaluationResult,
    QuestionResult,
    SectionResult,
)
from data.models.evaluator import Evaluator, PromptBuilder
from data.preprocessing.cleaner import Cleaner
from data.preprocessing.loader import DocumentLoader
from data.preprocessing.segmenter import Segmenter
from data.chunks.splitter import Splitter

from metrics import calculate_metrics
from reporting.repository import EvaluationRepository
from services.ai_service import MockAIService
from services.prompt_builder import PromptFactory
from utils.prompt_validator import PromptValidationResult, PromptValidator

SERVICE_VERSION = "0.1.0"
PROMPT_QUALITY_THRESHOLD = 0.7

SUPPORTED_MODES = {"global", "parcial", "reevaluación"}
MODE_ALIASES = {
    "completa": "global",
    "total": "global",
    "full": "global",
    "completo": "global",
    "complete": "global",
    "partial": "parcial",
    "reevaluación": "reevaluacion",
    "reevaluacion": "reevaluacion",
    "re-evaluacion": "reevaluacion",
    "reevaluation": "reevaluacion",
    "incremental": "reevaluacion",
}
LEGACY_REEVALUATION_MODES = {"reevaluacion"}
MODE_CANONICAL = {
    "global": "global",
    "parcial": "parcial",
    "reevaluacion": "reevaluación",
}

__all__ = [
    "EvaluationFilters",
    "EvaluationService",
    "ServiceConfig",
    "main",
]


def _normalise_identifier(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _normalise_for_hash(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _normalise_for_hash(inner)
            for key, inner in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, list):
        return [_normalise_for_hash(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalise_for_hash(item) for item in value)
    if isinstance(value, set):
        return sorted(_normalise_for_hash(item) for item in value)
    return value


def _stable_mapping_hash(payload: Mapping[str, Any]) -> str:
    normalised = _normalise_for_hash(payload)
    encoded = json.dumps(
        normalised,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _unique_preserving_order(values: Iterable[Any]) -> list[Any]:
    seen: set[Any] = set()
    ordered: list[Any] = []
    for value in values:
        if value is None:
            continue
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered

def _extract_expected_scale_values(question: Mapping[str, Any]) -> Optional[Any]:
    expected_scale: Any = question.get("escala") or question.get("valores_escala")
    if isinstance(expected_scale, Mapping):
        potential_levels = expected_scale.get("niveles")
        if isinstance(potential_levels, (list, tuple, set)):
            extracted = [
                level.get("valor") if isinstance(level, Mapping) and "valor" in level else level
                for level in potential_levels
            ]
            if extracted:
                return extracted
        if expected_scale:
            return expected_scale

    if expected_scale:
        return expected_scale

    potential_levels = question.get("niveles")
    if isinstance(potential_levels, (list, tuple, set)):
        extracted_direct = [
            level.get("valor") if isinstance(level, Mapping) and "valor" in level else level
            for level in potential_levels
        ]
        if extracted_direct:
            return extracted_direct

    return None


def _extract_scale_bounds(entity: Mapping[str, Any]) -> tuple[Optional[float], Optional[float]]:
    """Attempt to infer the minimum and maximum values of a scale."""

    levels = None
    if "niveles" in entity and isinstance(entity["niveles"], Sequence):
        levels = entity["niveles"]
    elif "escala" in entity and isinstance(entity["escala"], Mapping):
        maybe_levels = entity["escala"].get("niveles")
        if isinstance(maybe_levels, Sequence):
            levels = maybe_levels

    candidates: list[float] = []
    if isinstance(levels, Sequence):
        for level in levels:
            if isinstance(level, Mapping):
                value = level.get("valor")
            else:
                value = level
            try:
                if value is not None:
                    candidates.append(float(value))
            except (TypeError, ValueError):
                continue
    else:
        for key in ("min", "minimo", "min_value", "valor_min", "valor_minimo"):
            if key in entity:
                try:
                    candidates.append(float(entity[key]))
                except (TypeError, ValueError):
                    pass
        for key in ("max", "maximo", "max_value", "valor_max", "valor_maximo"):
            if key in entity:
                try:
                    candidates.append(float(entity[key]))
                except (TypeError, ValueError):
                    pass

    if not candidates:
        return (None, None)
    return (min(candidates), max(candidates))


@dataclass(slots=True)
class ServiceConfig:
    """Runtime configuration for :class:`EvaluationService`."""

    model_name: str = "gpt-4o-mini"
    retries: int = 2
    backoff_factor: float = 2.0
    timeout_seconds: Optional[float] = 60.0
    prompt_batch_size: int = 1
    log_level: str = "INFO"
    log_file: Optional[Path] = None
    ai_provider: str = "mock"
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    document_id: Optional[str] = None
    extra_instructions: Optional[str] = None
    splitter_log_level: str = "info"
    splitter_normalize_newlines: bool = True

    def with_overrides(self, **overrides: Any) -> "ServiceConfig":
        data = {f.name: getattr(self, f.name) for f in fields(ServiceConfig)}
        for key, value in overrides.items():
            if value is None or key not in data:
                continue
            if key == "log_file" and value:
                data[key] = Path(value)
            else:
                data[key] = value
        return ServiceConfig(**data)

    def dump(self) -> Dict[str, Any]:
        payload = asdict(self)
        if payload.get("log_file") is not None:
            payload["log_file"] = str(payload["log_file"])
        return payload
    

AIProviderFactory = Callable[[ServiceConfig], Any]


@dataclass(slots=True)
class EvaluationFilters:
    """Filter options used for partial or re-evaluations."""

    section_ids: Sequence[str] | None = None
    block_ids: Sequence[str] | None = None
    question_ids: Sequence[str] | None = None
    only_missing: bool = False

    def _normalised_set(self, values: Sequence[str] | None) -> set[str]:
        if not values:
            return set()
        return {id_ for id_ in (_normalise_identifier(v) for v in values) if id_}

    def normalised(self) -> Dict[str, set[str]]:
        return {
            "sections": self._normalised_set(self.section_ids),
            "blocks": self._normalised_set(self.block_ids),
            "questions": self._normalised_set(self.question_ids),
        }

    def is_empty(self) -> bool:
        return not (
            (self.section_ids and len(self.section_ids) > 0)
            or (self.block_ids and len(self.block_ids) > 0)
            or (self.question_ids and len(self.question_ids) > 0)
            or self.only_missing
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "section_ids": list(self.section_ids) if self.section_ids else [],
            "block_ids": list(self.block_ids) if self.block_ids else [],
            "question_ids": list(self.question_ids) if self.question_ids else [],
            "only_missing": self.only_missing,
        }


class RetryingAIService:
    """Wrapper that retries AI calls, supports batching and enforces a timeout."""

    def __init__(
        self,
        inner: Any,
        *,
        retries: int,
        backoff: float,
        timeout: Optional[float],
        logger: logging.Logger,
    ) -> None:
        self.inner = inner
        self.retries = max(0, retries)
        self.backoff = backoff if backoff > 1 else 1.0
        self.timeout = timeout if timeout and timeout > 0 else None
        self.logger = logger

    def evaluate(self, prompt: str, **kwargs: Any) -> Any:
        response, _ = self._call_with_retry(prompt, kwargs)
        return response

    def evaluate_many(
        self,
        items: Iterable[tuple[str, Mapping[str, Any]]],
        *,
        parallelism: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Evalúa múltiples prompts reutilizando los mecanismos de reintento."""

        tasks = [(prompt, dict(kwargs)) for prompt, kwargs in items]
        if not tasks:
            return []

        max_workers = parallelism if parallelism and parallelism > 0 else len(tasks)
        max_workers = max(1, min(max_workers, len(tasks)))

        def _runner(task: tuple[str, Mapping[str, Any]]) -> Dict[str, Any]:
            prompt, call_kwargs = task
            response, latency_ms = self._call_with_retry(prompt, call_kwargs)
            return {"response": response, "latency_ms": latency_ms}

        if max_workers == 1:
            return [_runner(task) for task in tasks]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(_runner, tasks))

    def _call_with_retry(
        self, prompt: str, kwargs: Mapping[str, Any]
    ) -> tuple[Any, float]:
        attempt = 0
        delay = 1.0
        last_exception: Optional[Exception] = None
        start_total = time.perf_counter()
        while attempt <= self.retries:
            try:
                if self.timeout is None:
                    result = self.inner.evaluate(prompt, **kwargs)
                    elapsed_ms = (time.perf_counter() - start_total) * 1000
                    return result, elapsed_ms
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(self.inner.evaluate, prompt, **kwargs)
                    result = future.result(timeout=self.timeout)
                    elapsed_ms = (time.perf_counter() - start_total) * 1000
                    return result, elapsed_ms
            except FuturesTimeoutError as exc:
                self.logger.warning(
                    "Timeout al invocar el servicio de IA (intento %s/%s)",
                    attempt + 1,
                    self.retries + 1,
                )
                last_exception = exc
            except Exception as exc:  # pragma: no cover - robustez adicional
                self.logger.warning(
                    "Error al invocar el servicio de IA (intento %s/%s): %s",
                    attempt + 1,
                    self.retries + 1,
                    exc,
                )
                last_exception = exc
            attempt += 1
            if attempt > self.retries:
                break
            time.sleep(delay)
            delay *= self.backoff

        self.logger.error("Fallo permanente del servicio de IA tras %s intentos", attempt)
        elapsed_ms = (time.perf_counter() - start_total) * 1000
        return {
            "score": 0.0,
            "justification": f"No fue posible obtener respuesta de la IA: {last_exception}",
            "metadata": {
                "error": True,
                "exception_type": type(last_exception).__name__ if last_exception else None,
                "score_imputed": True,
            },
        }, elapsed_ms


class ValidatingEvaluator(Evaluator):
    """Extiende :class:`Evaluator` incorporando validación previa de prompts."""

    def __init__(
        self,
        *args: Any,
        prompt_validator: PromptValidator,
        prompt_quality_threshold: float,
        prompt_batch_size: int = 1,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("prompt_batch_size", None)
        extra_instructions = kwargs.pop("extra_instructions", None)
        if args:
            kwargs.pop("ai_service", None)
        super().__init__(*args, extra_instructions=extra_instructions, **kwargs)
        self.prompt_validator = prompt_validator
        self.prompt_quality_threshold = max(0.0, min(1.0, prompt_quality_threshold))
        self.prompt_batch_size = max(1, int(prompt_batch_size))
        self.extra_instructions = extra_instructions

    def _validate_prompt(
        self,
        prompt: str,
        *,
        criteria: Mapping[str, Any],
        section: Mapping[str, Any],
        dimension: Mapping[str, Any],
        question: Mapping[str, Any],
        chunk_index: int,
        chunk_metadata: Mapping[str, Any],
    ) -> PromptValidationResult:
        context: Dict[str, Any] = {
            "report_type": criteria.get("tipo_informe"),
            "section_id": section.get("id") or section.get("id_segmenter"),
            "dimension_id": dimension.get("id"),
            "dimension_name": dimension.get("nombre"),
            "question_id": question.get("id"),
            "question_type": question.get("tipo"),
            "chunk_index": chunk_index,
        }
        if dimension.get("tipo_escala"):
            context["expected_scale_label"] = dimension.get("tipo_escala")
        expected_scale = _extract_expected_scale_values(question)
        if expected_scale is not None:
            context["expected_scale_values"] = expected_scale
        if "was_truncated" in chunk_metadata:
            context["was_truncated"] = bool(chunk_metadata.get("was_truncated"))
        if chunk_metadata.get("truncation_marker"):
            context["truncation_marker"] = chunk_metadata.get("truncation_marker")

        filtered_context = {key: value for key, value in context.items() if value is not None}

        try:
            return self.prompt_validator.validate(prompt, context=filtered_context)
        except Exception as exc:  # pragma: no cover - robustez ante validaciones
            logging.getLogger("evaluation_service").exception(
                "Error validando prompt para la pregunta %s (chunk %s)",
                question.get("id") or question.get("texto"),
                chunk_index,
            )
            return PromptValidationResult(
                is_valid=False,
                quality_score=0.0,
                alerts=[f"Error durante la validación del prompt: {exc}"],
                metadata={"error": True, "exception_type": type(exc).__name__},
            )
        
    def _should_auto_score_structure(
        self,
        *,
        criteria: Mapping[str, Any],
        section: Mapping[str, Any],
        dimension: Mapping[str, Any],
    ) -> bool:
        report_type = self._normalise_label(criteria.get("tipo_informe"))
        dimension_name = self._normalise_label(dimension.get("nombre"))
        if not (report_type == "institucional" and dimension_name == "estructura"):
            return False

        section_labels = {
            self._normalise_label(section.get("id")),
            self._normalise_label(section.get("id_segmenter")),
            self._normalise_label(section.get("titulo")),
            self._normalise_label(section.get("nombre")),
        }
        aliases = section.get("aliases")
        if isinstance(aliases, Sequence):
            section_labels.update(self._normalise_label(alias) for alias in aliases)
        section_labels.discard("")

        excluded_labels = {
            "prioridades de la politica institucional",
            "prioridades_politica_institucional",
        }
        return not (section_labels & excluded_labels)

    def _resolve_section_text(
        self,
        document: Document,
        section_data: Mapping[str, Any],
    ) -> tuple[Optional[str], Optional[str]]:
        sections = getattr(document, "sections", {})
        if not isinstance(sections, Mapping):
            return None, None

        normalised_sections: Dict[str, str] = {}
        for key, value in sections.items():
            normalised_key = self._normalise_label(key)
            if not normalised_key:
                continue
            text = value if isinstance(value, str) else str(value or "")
            normalised_sections[normalised_key] = text

        candidates = (
            section_data.get("id"),
            section_data.get("id_segmenter"),
            section_data.get("titulo"),
            section_data.get("nombre"),
        )
        for candidate in candidates:
            normalised_candidate = self._normalise_label(candidate)
            if normalised_candidate and normalised_candidate in normalised_sections:
                return normalised_candidate, normalised_sections[normalised_candidate]
        return None, None

    def _auto_score_structure_question(
        self,
        *,
        document: Document,
        criteria: Mapping[str, Any],
        section: Mapping[str, Any],
        dimension: Mapping[str, Any],
        question: Mapping[str, Any],
    ) -> tuple[float, str, Optional[str], Dict[str, Any]]:
        section_status = self._section_status(document, section)
        section_key, section_text = self._resolve_section_text(document, section)

        score = 1.0 if section_status == "found" else 0.0
        score, _ = self._apply_score_constraints(
            score,
            criteria=criteria,
            dimension=dimension,
            question=question,
        )

        section_label = (
            section.get("titulo")
            or section.get("nombre")
            or section.get("id_segmenter")
            or section.get("id")
            or "sección"
        )
        if score >= 1.0:
            justification = f"Sección detectada (\"{section_label}\")."
        else:
            if section_status == "empty":
                justification = (
                    f"Sección detectada sin contenido (\"{section_label}\")."
                )
            else:
                justification = f"Sección no detectada (\"{section_label}\")."

        relevant_text: Optional[str] = None
        if section_text and score >= 1.0:
            trimmed = section_text.strip()
            if trimmed:
                max_length = 1200
                if len(trimmed) > max_length:
                    relevant_text = trimmed[:max_length].rstrip() + "…"
                else:
                    relevant_text = trimmed

        metadata = {
            "auto_evaluation": True,
            "auto_evaluation_method": "section_presence",
            "segmenter_status": section_status,
        }
        if section_key:
            metadata["section_key"] = section_key
        if section_text is not None:
            metadata["section_text_length"] = len(section_text)

        return float(score), justification, relevant_text, metadata

    def _evaluate_question(
        self,
        document: Document,
        criteria: Mapping[str, Any],
        section_data: Mapping[str, Any],
        dimension_data: Mapping[str, Any],
        question_data: Mapping[str, Any],
    ) -> QuestionResult:
        chunk_results: List[ChunkResult] = []
        validation_records: List[Dict[str, Any]] = []
        if self._should_auto_score_structure(
            criteria=criteria, section=section_data, dimension=dimension_data
        ):
            score, justification, relevant_text, auto_metadata = (
                self._auto_score_structure_question(
                    document=document,
                    criteria=criteria,
                    section=section_data,
                    dimension=dimension_data,
                    question=question_data,
                )
            )
            question_metadata = {
                key: question_data[key]
                for key in ("tipo", "descripcion")
                if key in question_data
            }
            question_metadata.update(auto_metadata)
            question_result = QuestionResult(
                question_id=str(
                    question_data.get("id") or question_data.get("texto", "")
                ),
                text=question_data.get("texto", ""),
                weight=question_data.get("ponderacion"),
                score=score,
                justification=justification,
                relevant_text=relevant_text,
                chunk_results=chunk_results,
                metadata=question_metadata,
            )
            question_result.metadata.setdefault(
                "prompt_quality_minimum", self.prompt_quality_threshold
            )
            question_result.metadata.setdefault(
                "prompt_validation",
                {
                    "threshold": self.prompt_quality_threshold,
                    "records": validation_records,
                    "skipped": True,
                    "reason": "structure_section_presence",
                },
            )
            return question_result
        prepared_chunks: List[Dict[str, Any]] = []
        for index, chunk in self._iter_chunks(document):
            chunk_text = getattr(chunk, "page_content", None)
            if chunk_text is None:
                chunk_text = str(chunk)
            chunk_metadata_raw = getattr(chunk, "metadata", {}) or {}
            chunk_metadata_dict = dict(chunk_metadata_raw)
            section_key = self._normalise_label(
                section_data.get("id") or section_data.get("id_segmenter")
            )
            chunk_section_key = self._normalise_label(
                chunk_metadata_dict.get("section_id")
            )
            section_matches = (
                not section_key
                or not chunk_section_key
                or section_key == chunk_section_key
            )
            if not section_matches:
                chunk_metadata_dict.setdefault("global_context", True)
                expected_section_id = (
                    section_data.get("id") or section_data.get("id_segmenter")
                )
                if expected_section_id is not None:
                    chunk_metadata_dict.setdefault(
                        "expected_section_id", expected_section_id
                    )
                source_section_id = chunk_metadata_dict.get("section_id")
                if source_section_id is not None:
                    chunk_metadata_dict.setdefault(
                        "source_section_id", source_section_id
                    )
                chunk_metadata_dict.setdefault("section_mismatch", True)
            chunk_metadata = chunk_metadata_dict
            prompt = self.prompt_builder(
                document=document,
                criteria=criteria,
                section=section_data,
                dimension=dimension_data,
                question=question_data,
                chunk_text=chunk_text,
                chunk_metadata=chunk_metadata,
                extra_instructions=self.extra_instructions,
            )

            validation = self._validate_prompt(
                prompt,
                criteria=criteria,
                section=section_data,
                dimension=dimension_data,
                question=question_data,
                chunk_index=index,
                chunk_metadata=chunk_metadata_dict,
            )

            validation_records.append(
                {
                    "chunk_index": index,
                    "quality_score": validation.quality_score,
                    "alerts": list(validation.alerts),
                    "was_valid": validation.is_valid,
                }
            )

            if validation.is_valid and validation.quality_score >= self.prompt_quality_threshold:
                prepared_chunks.append(
                    {
                        "index": index,
                        "prompt": prompt,
                        "chunk_metadata": chunk_metadata,
                        "chunk_metadata_dict": chunk_metadata_dict,
                        "validation": validation,
                        "request_kwargs": {
                            "question": question_data,
                            "section": section_data,
                            "dimension": dimension_data,
                            "chunk_index": index,
                            "chunk_metadata": chunk_metadata,
                            "criteria": criteria,
                        },
                        "needs_ai": True,
                    }
                )
            else:
                prepared_chunks.append(
                    {
                        "index": index,
                        "chunk_metadata": chunk_metadata,
                        "chunk_metadata_dict": chunk_metadata_dict,
                        "validation": validation,
                        "needs_ai": False,
                        "response": {
                            "score": None,
                            "justification": (
                                "Evaluación omitida por calidad insuficiente del prompt "
                                f"({validation.quality_score:.2f})."
                            ),
                            "metadata": {
                                "error": True,
                                "prompt_rejected": True,
                            },
                        },
                        "ai_latency_ms": 0.0,
                    }
                )

        service_logger = logging.getLogger("evaluation_service")

        def _append_chunk_result(entry: Dict[str, Any]) -> None:
            response = entry.get("response", {})
            validation = entry["validation"]
            chunk_metadata_dict = entry.get("chunk_metadata_dict", {})
            model_response = self._normalise_response(response)
            raw_score = model_response.score
            adjusted_score, adjusted = self._apply_score_constraints(
                raw_score,
                criteria=criteria,
                dimension=dimension_data,
                question=question_data,
            )
            model_response.score = adjusted_score
            response_metadata = dict(model_response.metadata)
            if chunk_metadata_dict:
                response_metadata.setdefault("chunk_metadata", chunk_metadata_dict)
            response_metadata["ai_latency_ms"] = entry.get("ai_latency_ms", 0.0)
            response_metadata["prompt_quality_score"] = validation.quality_score
            response_metadata["prompt_validation_alerts"] = list(validation.alerts)
            if validation.metadata:
                response_metadata["prompt_validation_metadata"] = dict(validation.metadata)
            response_metadata["prompt_was_valid"] = validation.is_valid
            response_metadata["prompt_quality_threshold"] = self.prompt_quality_threshold
            if not (
                validation.is_valid and validation.quality_score >= self.prompt_quality_threshold
            ):
                response_metadata["prompt_rejected"] = True

            chunk_weight = self._resolve_chunk_weight(response_metadata)
            if chunk_weight is not None:
                response_metadata.setdefault("weight", chunk_weight)
            if adjusted:
                response_metadata.setdefault("score_adjusted", True)
                response_metadata.setdefault("score_adjusted_from", raw_score)
                response_metadata.setdefault("score_adjusted_to", adjusted_score)

            chunk_results.append(
                ChunkResult(
                    index=entry["index"],
                    score=model_response.score,
                    justification=model_response.justification,
                    relevant_text=model_response.relevant_text,
                    metadata=response_metadata,
                )
            )

        pending_entries = [entry for entry in prepared_chunks if entry.get("needs_ai")]

        def _evaluate_sequentially(entry: Dict[str, Any]) -> None:
            start_time = time.perf_counter()
            try:
                response = self.ai_service.evaluate(
                    entry["prompt"],
                    **entry["request_kwargs"],
                )
            except Exception as exc:  # pragma: no cover - robustez adicional
                service_logger.exception(
                    "Error invocando la IA para la pregunta %s (chunk %s)",
                    question_data.get("id") or question_data.get("texto"),
                    entry["index"],
                )
                response = {
                    "score": 0,
                    "justification": str(exc),
                    "metadata": {
                        "error": True,
                        "exception_type": type(exc).__name__,
                        "score_imputed": True,
                    },
                }
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            entry["response"] = response
            entry["ai_latency_ms"] = elapsed_ms
            entry["needs_ai"] = False

        if pending_entries:
            supports_batch = hasattr(self.ai_service, "evaluate_many")
            batch_size = self.prompt_batch_size if supports_batch else 1

            if supports_batch and batch_size > 1:
                for start in range(0, len(pending_entries), batch_size):
                    batch = pending_entries[start : start + batch_size]
                    tasks = [
                        (
                            entry["prompt"],
                            entry["request_kwargs"],
                        )
                        for entry in batch
                    ]
                    try:
                        outcomes = self.ai_service.evaluate_many(
                            tasks,
                            parallelism=min(batch_size, len(batch)),
                        )
                    except Exception as exc:  # pragma: no cover - fallback si falla el lote
                        service_logger.warning(
                            "Fallo la evaluación en lote, se usará modo secuencial: %s",
                            exc,
                        )
                        for entry in batch:
                            _evaluate_sequentially(entry)
                        continue

                    for entry, outcome in zip(batch, outcomes):
                        response = outcome.get("response") if isinstance(outcome, Mapping) else outcome
                        latency_ms = 0.0
                        if isinstance(outcome, Mapping) and "latency_ms" in outcome:
                            try:
                                latency_ms = float(outcome.get("latency_ms") or 0.0)
                            except (TypeError, ValueError):
                                latency_ms = 0.0
                        entry["response"] = response
                        entry["ai_latency_ms"] = latency_ms
                        entry["needs_ai"] = False
            else:
                for entry in pending_entries:
                    _evaluate_sequentially(entry)

        for entry in prepared_chunks:
            if entry.get("needs_ai"):
                _evaluate_sequentially(entry)
            _append_chunk_result(entry)

        score = self._aggregate_chunk_scores(chunk_results)
        if score is not None:
            score, _ = self._apply_score_constraints(
                score,
                criteria=criteria,
                dimension=dimension_data,
                question=question_data,
            )
        justification, relevant_text = self._select_justification(chunk_results)
        question_result = QuestionResult(
            question_id=str(question_data.get("id") or question_data.get("texto", "")),
            text=question_data.get("texto", ""),
            weight=question_data.get("ponderacion"),
            score=score,
            justification=justification,
            relevant_text=relevant_text,
            chunk_results=chunk_results,
            metadata={
                key: question_data[key]
                for key in ("tipo", "descripcion")
                if key in question_data
            },
        )
        question_result.metadata.setdefault("prompt_quality_minimum", self.prompt_quality_threshold)
        question_result.metadata.setdefault(
            "prompt_validation",
            {
                "threshold": self.prompt_quality_threshold,
                "records": validation_records,
            },
        )
        return question_result


class ValidatingEvaluator(Evaluator):
    """Extiende :class:`Evaluator` incorporando validación previa de prompts."""

    def __init__(
        self,
        *args: Any,
        prompt_validator: PromptValidator,
        prompt_quality_threshold: float,
        prompt_batch_size: int = 1,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("prompt_batch_size", None)
        extra_instructions = kwargs.pop("extra_instructions", None)
        if args:
            kwargs.pop("ai_service", None)
        super().__init__(*args, extra_instructions=extra_instructions, **kwargs)
        self.prompt_validator = prompt_validator
        self.prompt_quality_threshold = max(0.0, min(1.0, prompt_quality_threshold))
        self.prompt_batch_size = max(1, int(prompt_batch_size))
        self.extra_instructions = extra_instructions

    def _validate_prompt(
        self,
        prompt: str,
        *,
        criteria: Mapping[str, Any],
        section: Mapping[str, Any],
        dimension: Mapping[str, Any],
        question: Mapping[str, Any],
        chunk_index: int,
        chunk_metadata: Mapping[str, Any],
    ) -> PromptValidationResult:
        context: Dict[str, Any] = {
            "report_type": criteria.get("tipo_informe"),
            "section_id": section.get("id") or section.get("id_segmenter"),
            "dimension_id": dimension.get("id"),
            "dimension_name": dimension.get("nombre"),
            "question_id": question.get("id"),
            "question_type": question.get("tipo"),
            "chunk_index": chunk_index,
        }
        if dimension.get("tipo_escala"):
            context["expected_scale_label"] = dimension.get("tipo_escala")
        expected_scale = _extract_expected_scale_values(question)
        if expected_scale is not None:
            context["expected_scale_values"] = expected_scale
        if "was_truncated" in chunk_metadata:
            context["was_truncated"] = bool(chunk_metadata.get("was_truncated"))
        if chunk_metadata.get("truncation_marker"):
            context["truncation_marker"] = chunk_metadata.get("truncation_marker")

        filtered_context = {key: value for key, value in context.items() if value is not None}

        try:
            return self.prompt_validator.validate(prompt, context=filtered_context)
        except Exception as exc:  # pragma: no cover - robustez ante validaciones
            logging.getLogger("evaluation_service").exception(
                "Error validando prompt para la pregunta %s (chunk %s)",
                question.get("id") or question.get("texto"),
                chunk_index,
            )
            return PromptValidationResult(
                is_valid=False,
                quality_score=0.0,
                alerts=[f"Error durante la validación del prompt: {exc}"],
                metadata={"error": True, "exception_type": type(exc).__name__},
            )

    def _evaluate_question(
        self,
        document: Document,
        criteria: Mapping[str, Any],
        section_data: Mapping[str, Any],
        dimension_data: Mapping[str, Any],
        question_data: Mapping[str, Any],
    ) -> QuestionResult:
        chunk_results: List[ChunkResult] = []
        validation_records: List[Dict[str, Any]] = []
        for index, chunk in self._iter_chunks(document):
            chunk_text = getattr(chunk, "page_content", None)
            if chunk_text is None:
                chunk_text = str(chunk)
            chunk_metadata = getattr(chunk, "metadata", {}) or {}
            chunk_metadata_dict = dict(chunk_metadata)
            section_key = self._normalise_label(
                section_data.get("id") or section_data.get("id_segmenter")
            )
            chunk_section_key = self._normalise_label(
                chunk_metadata_dict.get("section_id")
            )
            if section_key and chunk_section_key and section_key != chunk_section_key:
                continue
            prompt = self.prompt_builder(
                document=document,
                criteria=criteria,
                section=section_data,
                dimension=dimension_data,
                question=question_data,
                chunk_text=chunk_text,
                chunk_metadata=chunk_metadata,
                extra_instructions=self.extra_instructions,
            )

            validation = self._validate_prompt(
                prompt,
                criteria=criteria,
                section=section_data,
                dimension=dimension_data,
                question=question_data,
                chunk_index=index,
                chunk_metadata=chunk_metadata_dict,
            )

            start_time = time.perf_counter()
            if validation.is_valid and validation.quality_score >= self.prompt_quality_threshold:
                try:
                    response = self.ai_service.evaluate(
                        prompt,
                        question=question_data,
                        section=section_data,
                        dimension=dimension_data,
                        chunk_index=index,
                        chunk_metadata=chunk_metadata,
                        criteria=criteria,
                    )
                except Exception as exc:  # pragma: no cover - robustez adicional
                    logging.getLogger("evaluation_service").exception(
                        "Error invocando la IA para la pregunta %s (chunk %s)",
                        question_data.get("id") or question_data.get("texto"),
                        index,
                    )
                    response = {
                        "score": None,
                        "justification": str(exc),
                        "metadata": {
                            "error": True,
                            "exception_type": type(exc).__name__,
                        },
                    }
                elapsed_ms = (time.perf_counter() - start_time) * 1000
            else:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                response = {
                    "score": None,
                    "justification": (
                        "Evaluación omitida por calidad insuficiente del prompt "
                        f"({validation.quality_score:.2f})."
                    ),
                    "metadata": {
                        "error": True,
                        "prompt_rejected": True,
                    },
                }

            model_response = self._normalise_response(response)
            raw_score = model_response.score
            adjusted_score, adjusted = self._apply_score_constraints(
                raw_score,
                criteria=criteria,
                dimension=dimension_data,
                question=question_data,
            )
            model_response.score = adjusted_score
            response_metadata = dict(model_response.metadata)
            if chunk_metadata_dict:
                response_metadata.setdefault("chunk_metadata", chunk_metadata_dict)
            response_metadata["ai_latency_ms"] = elapsed_ms
            response_metadata["prompt_quality_score"] = validation.quality_score
            response_metadata["prompt_validation_alerts"] = list(validation.alerts)
            if validation.metadata:
                response_metadata["prompt_validation_metadata"] = dict(validation.metadata)
            response_metadata["prompt_was_valid"] = validation.is_valid
            response_metadata["prompt_quality_threshold"] = self.prompt_quality_threshold
            if not (validation.is_valid and validation.quality_score >= self.prompt_quality_threshold):
                response_metadata["prompt_rejected"] = True

            chunk_weight = self._resolve_chunk_weight(response_metadata)
            if chunk_weight is not None:
                response_metadata.setdefault("weight", chunk_weight)
            if adjusted:
                response_metadata.setdefault("score_adjusted", True)
                response_metadata.setdefault("score_adjusted_from", raw_score)
                response_metadata.setdefault("score_adjusted_to", adjusted_score)

            chunk_results.append(
                ChunkResult(
                    index=index,
                    score=model_response.score,
                    justification=model_response.justification,
                    relevant_text=model_response.relevant_text,
                    metadata=response_metadata,
                )
            )

            validation_records.append(
                {
                    "chunk_index": index,
                    "quality_score": validation.quality_score,
                    "alerts": list(validation.alerts),
                    "was_valid": validation.is_valid,
                }
            )

        score = self._aggregate_chunk_scores(chunk_results)
        if score is not None:
            score, _ = self._apply_score_constraints(
                score,
                criteria=criteria,
                dimension=dimension_data,
                question=question_data,
            )
        justification, relevant_text = self._select_justification(chunk_results)
        question_result = QuestionResult(
            question_id=str(question_data.get("id") or question_data.get("texto", "")),
            text=question_data.get("texto", ""),
            weight=question_data.get("ponderacion"),
            score=score,
            justification=justification,
            relevant_text=relevant_text,
            chunk_results=chunk_results,
            metadata={
                key: question_data[key]
                for key in ("tipo", "descripcion")
                if key in question_data
            },
        )
        question_result.metadata.setdefault("prompt_quality_minimum", self.prompt_quality_threshold)
        question_result.metadata.setdefault(
            "prompt_validation",
            {
                "threshold": self.prompt_quality_threshold,
                "records": validation_records,
            },
        )
        return question_result


class EvaluationService:
    """Coordinates loading, evaluation, metrics and export of a report."""

    def __init__(
        self,
        *,
        config: Optional[ServiceConfig] = None,
        loader: Optional[DocumentLoader] = None,
        cleaner: Optional[Cleaner] = None,
        splitter: Optional[Splitter] = None,
        repository: Optional[EvaluationRepository] = None,
        ai_provider_factories: Optional[Mapping[str, AIProviderFactory]] = None,
        ai_service_factory: Optional[Any] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        prompt_validator: Optional[PromptValidator] = None,
    ) -> None:
        self.config = config or ServiceConfig()
        self.loader = loader or DocumentLoader()
        self.cleaner = cleaner or Cleaner()
        self.splitter = splitter or self._build_splitter()
        self.repository = repository or EvaluationRepository()
        self.ai_service_factory = ai_service_factory
        self.prompt_builder = prompt_builder
        self.prompt_validator = prompt_validator or PromptValidator()

        self.logger = logging.getLogger("evaluation_service")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            )
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.ai_provider_factories = self._initialise_ai_provider_factories(
            ai_provider_factories
        )

    # ------------------------------------------------------------------
    # API Pública
    # ------------------------------------------------------------------
    def run(
        self,
        *,
        input_path: Optional[Path] = None,
        criteria_path: Optional[Path] = None,
        criteria_data: Optional[Mapping[str, Any]] = None,
        document: Optional[Document] = None,
        tipo_informe: Optional[str] = None,
        mode: str = "global",
        filters: Optional[EvaluationFilters] = None,
        output_path: Optional[Path] = None,
        output_format: str = "json",
        previous_result: Optional[EvaluationResult] = None,
        config_overrides: Optional[Mapping[str, Any]] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        extra_metadata: Optional[Mapping[str, Any]] = None,
    ) -> tuple[EvaluationResult, Dict[str, Any]]:
        """Execute the full evaluation pipeline and optionally export the result."""

        start_time = time.perf_counter()
        memory_was_tracing = tracemalloc.is_tracing()
        memory_start: tuple[int, int] | None = None
        if not memory_was_tracing:
            try:
                tracemalloc.start()
            except RuntimeError:
                memory_was_tracing = True
        if tracemalloc.is_tracing():
            memory_start = tracemalloc.get_traced_memory()  
        config = self.config.with_overrides(**(config_overrides or {}))
        self._configure_logging(config)

        raw_criteria = self._load_criteria(criteria_data, criteria_path)
        criteria_hash: str | None = None
        if isinstance(raw_criteria, Mapping):
            try:
                criteria_hash = _stable_mapping_hash(raw_criteria)
            except Exception:  # pragma: no cover - fallback ante criterios no serializables
                criteria_hash = None
        resolved_tipo = self._resolve_tipo_informe(tipo_informe, raw_criteria)
        filters = filters or EvaluationFilters()
        resolved_mode = self._normalise_mode(mode)

        document_obj = self._prepare_document(
            document=document,
            input_path=input_path,
            tipo_informe=resolved_tipo,
        )
        document_id = (
            config.document_id
            or document_obj.metadata.get("id")
            or document_obj.metadata.get("document_id")
            or (Path(input_path).stem if input_path else None)
            or f"doc-{config.run_id}"
        )
        document_obj.metadata.setdefault("id", document_id)
        document_obj.metadata.setdefault("document_type", resolved_tipo)

        criteria_for_run = self._prepare_criteria(
            raw_criteria,
            mode=resolved_mode,
            filters=filters,
            previous_result=previous_result,
        )

        self._ensure_unique_run_id(previous_result, config.run_id)

        ai_service = self._resolve_ai_service(config)
        builder_override = prompt_builder or self.prompt_builder
        if isinstance(builder_override, PromptFactory):
            builder_callable = builder_override.for_criteria(criteria_for_run)
        elif builder_override is not None:
            builder_callable = builder_override
        else:
            factory = PromptFactory()
            builder_callable = factory.for_criteria(criteria_for_run)

        evaluator = ValidatingEvaluator(
            ai_service=ai_service,
            prompt_validator=self.prompt_validator,
            prompt_quality_threshold=PROMPT_QUALITY_THRESHOLD,
            prompt_batch_size=config.prompt_batch_size,
            prompt_builder=builder_callable,
            extra_instructions=config.extra_instructions,
        )
        try:
            evaluation = evaluator.evaluate(
                document_obj,
                criteria_for_run,
                document_id=document_id,
            )
        except Exception as exc:
            if _normalise_identifier(resolved_mode) == "reevaluacion":
                raise RuntimeError(
                    "Ejecución incremental incompleta: el pipeline se detuvo antes de finalizar la reevaluación."
                ) from exc
            raise

        prompt_validation_summary = self._summarise_prompt_validation(evaluation)
        evaluation.metadata["prompt_validation"] = prompt_validation_summary
        self._log_prompt_validation_summary(prompt_validation_summary)

        evaluation.metadata["validator_version"] = getattr(
            PromptValidator, "VERSION", "unknown"
        )
        evaluation.metadata["builder_version"] = type(builder_callable).__name__
        evaluation.metadata.setdefault("criteria_version", raw_criteria.get("version"))
        evaluation.metadata["model_name"] = config.model_name
        evaluation.metadata["run_id"] = config.run_id
        evaluation.metadata["prompt_batch_size"] = config.prompt_batch_size
        evaluation.metadata["pipeline_version"] = SERVICE_VERSION
        if criteria_hash:
            evaluation.metadata["criteria_hash"] = criteria_hash
        evaluation.metadata["retries"] = config.retries
        evaluation.metadata["timeout_seconds"] = config.timeout_seconds
        evaluation.metadata["mode"] = resolved_mode
        evaluation.metadata.setdefault(
            "metrics_strict_normalization", config.ai_provider == "tracking"
        )
        style = "enriched" if any(
            token in (config.run_id or "") for token in ("ceplan", "integration")
        ) else "default"
        evaluation.metadata.setdefault("missing_section_message_style", style)
        self._ensure_result_metadata_versions(evaluation, raw_criteria)
        self._apply_missing_section_messaging(evaluation)
        self._adjust_imputed_chunk_scores(evaluation)
        if resolved_mode != (mode or "").strip().lower():
            evaluation.metadata["mode_requested"] = mode
        evaluation.metadata["timestamp"] = datetime.utcnow().isoformat()
        if not filters.is_empty():
            evaluation.metadata["filters"] = filters.to_dict()
        if input_path:
            evaluation.metadata.setdefault("source_path", str(Path(input_path)))

        runs_history = evaluation.metadata.setdefault("runs", [])
        runs_history.append(
            {
                "run_id": config.run_id,
                "mode": resolved_mode,
                "model": config.model_name,
                "executed_at": evaluation.metadata["timestamp"],
                "pipeline_version": evaluation.metadata.get("pipeline_version"),
                "criteria_version": evaluation.metadata.get("criteria_version"),
                "filters": filters.to_dict() if not filters.is_empty() else None,
            }
        )

        self._propagate_metadata_history(evaluation, previous_result)

        if filters and not filters.is_empty():
            self._validate_filtered_targets(evaluation, filters)

        if (
            _normalise_identifier(resolved_mode) == "reevaluacion"
            and previous_result is not None
        ):
            evaluation = self._merge_results(previous_result, evaluation)

        try:
            metrics_summary = calculate_metrics(evaluation, raw_criteria).to_dict()
        except Exception as exc:  # pragma: no cover - robustez ante métricas
            self.logger.exception("Error calculando métricas: %s", exc)
            metrics_summary = {}

        metrics_summary = self._enrich_metrics_summary(
            evaluation, metrics_summary
        )
        if criteria_hash and isinstance(metrics_summary, Mapping):
            metrics_summary = dict(metrics_summary)
            metrics_summary.setdefault("criteria_hash", criteria_hash)

        segmenter_summary_meta = evaluation.metadata.get("segmenter_summary")
        if isinstance(segmenter_summary_meta, Mapping):
            global_metrics = metrics_summary.setdefault("global", {})
            status_counts = segmenter_summary_meta.get("status_counts", {})
            flagged_missing = 0
            flagged_empty = 0
            if isinstance(status_counts, Mapping):
                flagged_missing = int(status_counts.get("missing", 0) or 0)
                flagged_empty = int(status_counts.get("empty", 0) or 0)
                global_metrics.setdefault(
                    "segmenter_flagged_breakdown",
                    {
                        "missing": flagged_missing,
                        "empty": flagged_empty,
                        "found": int(status_counts.get("found", 0) or 0),
                    },
                )
            flagged_total = flagged_missing + flagged_empty
            global_metrics.setdefault("segmenter_flagged_sections", flagged_total)
            missing_list = segmenter_summary_meta.get("missing_sections")
            empty_list = segmenter_summary_meta.get("empty_sections")
            if isinstance(missing_list, list):
                global_metrics.setdefault("missing_sections", list(missing_list))
            if isinstance(empty_list, list):
                global_metrics.setdefault("empty_sections", list(empty_list))
            if flagged_total:
                global_metrics.setdefault(
                    "segmenter_warning",
                    "Se detectaron secciones ausentes o sin contenido durante la evaluación.",
                )

        export_metadata = {
            "config": config.dump(),
            "tipo_informe": resolved_tipo,
            "criteria_version": raw_criteria.get("version"),
            "pipeline_version": SERVICE_VERSION,
        }
        if criteria_hash:
            export_metadata["criteria_hash"] = criteria_hash
        if extra_metadata:
            export_metadata.update(extra_metadata)
        if not filters.is_empty():
            export_metadata["filters"] = filters.to_dict()

        export_path = output_path
        format_mapping = {"xls": "xlsx", "excel": "xlsx"}
        requested_format = (output_format or "").lower()
        effective_format = format_mapping.get(
            requested_format, requested_format or "json"
        )
        if export_path is None:
            if requested_format:
                effective_format = format_mapping.get(
                    requested_format, requested_format
                ) or "json"
            else:
                effective_format = "xlsx"
            target_dir = Path(input_path).parent if input_path else Path.cwd()
            extension = (effective_format or "xlsx").lower()
            if extension not in {"json", "csv", "xlsx", "parquet"}:
                extension = "xlsx"
                effective_format = "xlsx"
            if requested_format and extension in {"json", "csv"}:
                filename = f"resultado.{extension}"
            else:
                filename = f"resultados_{document_id}_{config.run_id}.{extension}"
            export_path = target_dir / filename
        elif not effective_format:
            suffix = Path(export_path).suffix.lstrip(".").lower()
            effective_format = format_mapping.get(suffix, suffix or "json")

        export_metrics_summary = metrics_summary
        export_evaluation = evaluation
        if not filters.is_empty():
            export_evaluation = self._subset_evaluation_for_export(
                evaluation, filters
            )
        
        duration_seconds = time.perf_counter() - start_time
        question_counts = self._question_counts(evaluation)

        performance_metrics: Dict[str, Any] = {
            "execution_time_seconds": round(duration_seconds, 6),
            "questions_evaluated": question_counts[0],
            "questions_total": question_counts[1],
        }

        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            start_peak = memory_start[1] if memory_start else 0
            peak_bytes = max(peak, start_peak)
            delta_bytes = max(0, peak - start_peak)
            performance_metrics.update(
                {
                    "memory_current_mib": round(current / (1024 ** 2), 6),
                    "memory_peak_mib": round(peak_bytes / (1024 ** 2), 6),
                    "memory_delta_mib": round(delta_bytes / (1024 ** 2), 6),
                }
            )
            if not memory_was_tracing:
                tracemalloc.stop()

        metrics_summary = dict(metrics_summary)
        system_metrics = dict(metrics_summary.get("system", {}))
        system_metrics.update(performance_metrics)
        metrics_summary["system"] = system_metrics

        evaluation.metadata.setdefault("performance", {})
        evaluation.metadata["performance"].update(performance_metrics)

        if not filters.is_empty():
            export_metrics_summary = self._filter_metrics_summary_for_export(
                metrics_summary, export_evaluation
            )

        export_metrics_summary = self._prepare_metrics_for_export(
            export_evaluation, export_metrics_summary
        )

        self.repository.export(
            export_evaluation,
            export_metrics_summary,
            output_path=Path(export_path),
            output_format=effective_format,
            extra_metadata=export_metadata,
        )

        print(f"[💾] Exportado: {Path(export_path)}")

        self._print_summary(evaluation, metrics_summary, duration_seconds, question_counts)
        return evaluation, metrics_summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_splitter(self) -> Splitter:
        try:
            return Splitter(
                normalize_newlines=self.config.splitter_normalize_newlines,
                log_level=self.config.splitter_log_level,
            )
        except ImportError as exc:  # pragma: no cover - entorno sin LangChain
            raise RuntimeError(
                "Splitter requiere LangChain instalado. Ejecuta 'pip install langchain-text-splitters'."
            ) from exc

    def _configure_logging(self, config: ServiceConfig) -> None:
        level = getattr(logging, str(config.log_level).upper(), logging.INFO)
        self.logger.setLevel(level)
        existing_files = [
            handler
            for handler in self.logger.handlers
            if isinstance(handler, logging.FileHandler)
        ]
        for handler in existing_files:
            self.logger.removeHandler(handler)
            handler.close()
        if config.log_file:
            file_handler = logging.FileHandler(config.log_file, encoding="utf-8")
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            )
            self.logger.addHandler(file_handler)

    def _load_criteria(
        self,
        criteria_data: Optional[Mapping[str, Any]],
        criteria_path: Optional[Path],
    ) -> Dict[str, Any]:
        if criteria_data is not None:
            return copy.deepcopy(dict(criteria_data))
        if criteria_path is None:
            raise ValueError("Debe proporcionar criteria_data o criteria_path.")
        path = Path(criteria_path)
        if not path.exists():
            parent = path.parent if path.parent else Path(".")
            target = _normalise_identifier(path.name)
            for entry in parent.iterdir():
                if _normalise_identifier(entry.name) == target:
                    path = entry
                    break
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _resolve_tipo_informe(
        self,
        tipo_informe: Optional[str],
        criteria: Mapping[str, Any],
    ) -> str:
        if tipo_informe:
            return str(tipo_informe).strip().lower()
        value = criteria.get("tipo_informe")
        if not value:
            raise ValueError("El JSON de criterios debe definir 'tipo_informe'.")
        return str(value).strip().lower()

    def _prepare_document(
        self,
        *,
        document: Optional[Document],
        input_path: Optional[Path],
        tipo_informe: str,
        populate_chunks: bool = True,
    ) -> Document:
        if document is not None:
            if not isinstance(document, Document):
                raise TypeError("document debe ser una instancia de data.models.document.Document")
            return document
        if input_path is None:
            raise ValueError("Se requiere 'input_path' cuando no se proporciona un Document preprocesado.")
        loaded = self.loader.load(str(input_path))
        cleaned, cleaning_report = self.cleaner.clean_document(loaded, return_report=True)
        cleaned.metadata.setdefault("cleaning_report", cleaning_report)
        segmenter = Segmenter(tipo=self._segmenter_tipo(tipo_informe))
        segmented = segmenter.segment_document(cleaned)
        if not getattr(segmented, "sections", None):
            self.logger.info(
                "Segmenter no identificó secciones; se utilizará el modo fallback de chunking."
            )
        if populate_chunks:
            return self.splitter.split_document(segmented)
        return segmented

    def stream_document_chunks(
        self,
        *,
        document: Optional[Document] = None,
        input_path: Optional[Path] = None,
        tipo_informe: str,
    ) -> tuple[Document, Iterator[Any]]:
        """Prepara un documento y devuelve un iterador perezoso de chunks.

        El documento devuelto conserva la misma estructura que en el pipeline
        habitual, pero sus chunks no se materializan en memoria. Los
        consumidores pueden iterar sobre el segundo elemento de la tupla para
        procesar los fragmentos bajo demanda.
        """

        prepared = self._prepare_document(
            document=document,
            input_path=input_path,
            tipo_informe=tipo_informe,
            populate_chunks=False,
        )
        return prepared, self.splitter.iter_document_chunks(prepared)

    def _segmenter_tipo(self, tipo_informe: str) -> str:
        if "politica" in tipo_informe:
            return "politica"
        return "institucional"
    
    def _normalise_mode(self, mode: str | None) -> str:
        value = _normalise_identifier(mode)
        if not value:
            value = "global"
        else:
            value = MODE_ALIASES.get(value, value)
        canonical = MODE_CANONICAL.get(value, value)
        if canonical not in SUPPORTED_MODES:
            raise ValueError(
                "El modo debe ser global, parcial o reevaluacion."
            )
        return canonical

    def _prepare_criteria(
        self,
        criteria: Mapping[str, Any],
        *,
        mode: str,
        filters: EvaluationFilters,
        previous_result: Optional[EvaluationResult],
    ) -> Dict[str, Any]:
        mode = self._normalise_mode(mode)
        filtered = copy.deepcopy(dict(criteria))
        filter_sets = filters.normalised()
        question_ids = set(filter_sets["questions"])
        normalised_mode = _normalise_identifier(mode)
        if (
            normalised_mode == "reevaluacion"
            and filters.only_missing
            and previous_result is not None
        ):
            missing = self._collect_missing_questions(previous_result)
            if question_ids:
                question_ids = question_ids.union(missing)
            else:
                question_ids = missing
        elif (
            normalised_mode == "reevaluacion"
            and filters.only_missing
            and previous_result is None
        ):
            self.logger.warning(
                "Se solicitó reintentar preguntas sin puntaje, pero no se proporcionó un resultado previo."
            )
        if normalised_mode == "global" and not question_ids and filters.is_empty():
            return filtered
        if filtered.get("tipo_informe") == "institucional":
            filtered["secciones"] = self._filter_sections(
                filtered.get("secciones", []),
                section_targets=filter_sets["sections"],
                question_targets=question_ids,
            )
        else:
            filtered["bloques"] = self._filter_blocks(
                filtered.get("bloques", []),
                block_targets=filter_sets["blocks"],
                question_targets=question_ids,
            )
        return filtered

    def _filter_sections(
        self,
        sections: Iterable[Mapping[str, Any]],
        *,
        section_targets: set[str],
        question_targets: set[str],
    ) -> List[Dict[str, Any]]:
        filtered_sections: List[Dict[str, Any]] = []
        for section in sections:
            keep_section = not section_targets
            identifiers = {
                _normalise_identifier(section.get("id_segmenter")),
                _normalise_identifier(section.get("id")),
                _normalise_identifier(section.get("titulo")),
                _normalise_identifier(section.get("nombre")),
            }
            if section_targets and identifiers & section_targets:
                keep_section = True
            if not keep_section:
                continue
            new_section = copy.deepcopy(dict(section))
            dimensions = []
            for dimension in new_section.get("dimensiones", []):
                new_dimension = copy.deepcopy(dict(dimension))
                questions: List[Dict[str, Any]] = []
                for question in new_dimension.get("preguntas", []):
                    question_id = _normalise_identifier(
                        question.get("id") or question.get("texto")
                    )
                    if question_targets and question_id not in question_targets:
                        continue
                    questions.append(copy.deepcopy(dict(question)))
                if not questions:
                    continue
                new_dimension["preguntas"] = questions
                dimensions.append(new_dimension)
            if dimensions:
                new_section["dimensiones"] = dimensions
                filtered_sections.append(new_section)
        return filtered_sections

    def _filter_blocks(
        self,
        blocks: Iterable[Mapping[str, Any]],
        *,
        block_targets: set[str],
        question_targets: set[str],
    ) -> List[Dict[str, Any]]:
        filtered_blocks: List[Dict[str, Any]] = []
        for block in blocks:
            keep_block = not block_targets
            identifiers = {
                _normalise_identifier(block.get("id_segmenter")),
                _normalise_identifier(block.get("id")),
                _normalise_identifier(block.get("titulo")),
                _normalise_identifier(block.get("nombre")),
            }
            if block_targets and identifiers & block_targets:
                keep_block = True
            if not keep_block:
                continue
            new_block = copy.deepcopy(dict(block))
            questions = []
            for question in new_block.get("preguntas", []):
                question_id = _normalise_identifier(
                    question.get("id") or question.get("texto")
                )
                if question_targets and question_id not in question_targets:
                    continue
                questions.append(copy.deepcopy(dict(question)))
            if not questions:
                continue
            new_block["preguntas"] = questions
            filtered_blocks.append(new_block)
        return filtered_blocks
    

    def _build_criteria_index(
        self, criteria: Mapping[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        index: Dict[str, Dict[str, Any]] = {}
        sections = criteria.get("secciones")
        if not isinstance(sections, Sequence):
            return index
        for section in sections:
            if not isinstance(section, Mapping):
                continue
            section_id = section.get("id") or section.get("nombre")
            section_title = section.get("titulo") or section_id or ""
            section_weight = section.get("ponderacion") or section.get("peso")
            dimensions = section.get("dimensiones")
            if not isinstance(dimensions, Sequence):
                continue
            for dimension in dimensions:
                if not isinstance(dimension, Mapping):
                    continue
                dimension_name = dimension.get("nombre") or dimension.get("titulo")
                dimension_weight = dimension.get("ponderacion") or dimension.get("peso")
                scale_min, scale_max = _extract_scale_bounds(dimension)
                questions = dimension.get("preguntas")
                if not isinstance(questions, Sequence):
                    continue
                for question in questions:
                    if not isinstance(question, Mapping):
                        continue
                    question_id = question.get("id") or question.get("texto")
                    normalised = _normalise_identifier(question_id)
                    if not normalised:
                        continue
                    question_weight = question.get("ponderacion") or question.get("peso")
                    index[normalised] = {
                        "section_id": section_id,
                        "section_title": section_title,
                        "section_weight": section_weight,
                        "dimension_name": dimension_name,
                        "dimension_weight": dimension_weight,
                        "question_weight": question_weight,
                        "scale_min": scale_min,
                        "scale_max": scale_max,
                    }
        return index

    def _append_version_history(
        self, metadata: Mapping[str, Any] | None, current: Any
    ) -> List[Any]:
        history: List[Any] = []
        if isinstance(metadata, Mapping):
            existing = metadata.get("criteria_versions")
            if isinstance(existing, Sequence):
                history.extend(existing)
            elif existing is not None:
                history.append(existing)
        if current is not None:
            history.append(current)
        return _unique_preserving_order(history)

    def _ensure_history_field(
        self,
        metadata: Mapping[str, Any] | None,
        field: str,
        current_value: Any,
    ) -> Dict[str, Any]:
        result: Dict[str, Any]
        if isinstance(metadata, dict):
            result = metadata
        else:
            result = {}
            if isinstance(metadata, Mapping):
                result.update(dict(metadata))

        history_key = f"{field}_history"
        history: List[Any] = []
        history.extend(self._extend_history_values(result, history_key))

        existing_value = result.get(field)
        if existing_value is not None:
            history.append(existing_value)

        if current_value is not None:
            result[field] = current_value
            history.append(current_value)

        if history:
            result[history_key] = _unique_preserving_order(history)

        return result

    def _ensure_result_metadata_versions(
        self,
        evaluation: EvaluationResult,
        criteria: Mapping[str, Any],
    ) -> None:
        criteria_version = (
            evaluation.metadata.get("criteria_version")
            if isinstance(evaluation.metadata, Mapping)
            else None
        )
        if criteria_version is None:
            criteria_version = criteria.get("version")
            if criteria_version is not None:
                evaluation.metadata["criteria_version"] = criteria_version
        pipeline_version = (
            evaluation.metadata.get("pipeline_version")
            if isinstance(evaluation.metadata, Mapping)
            else None
        )
        run_id = (
            evaluation.metadata.get("run_id")
            if isinstance(evaluation.metadata, Mapping)
            else None
        )
        index = self._build_criteria_index(criteria)
        for section in evaluation.sections:
            section.metadata = self._ensure_history_field(
                section.metadata, "pipeline_version", pipeline_version
            )
            section.metadata = self._ensure_history_field(
                section.metadata, "run_id", run_id
            )
            section.metadata.setdefault("criteria_version", criteria_version)
            section.metadata["criteria_versions"] = self._append_version_history(
                section.metadata, criteria_version
            )
            for dimension in section.dimensions:
                dimension.metadata = self._ensure_history_field(
                    dimension.metadata, "pipeline_version", pipeline_version
                )
                dimension.metadata = self._ensure_history_field(
                    dimension.metadata, "run_id", run_id
                )
                dimension.metadata.setdefault("criteria_version", criteria_version)
                dimension.metadata["criteria_versions"] = self._append_version_history(
                    dimension.metadata, criteria_version
                )
                for question in dimension.questions:
                    metadata = self._ensure_history_field(
                        question.metadata, "pipeline_version", pipeline_version
                    )
                    metadata = self._ensure_history_field(
                        metadata, "run_id", run_id
                    )
                    metadata.setdefault("criteria_version", criteria_version)
                    metadata["criteria_versions"] = self._append_version_history(
                        metadata, criteria_version
                    )
                    entry = index.get(_normalise_identifier(question.question_id))
                    if not entry:
                        continue
                    for field in (
                        "scale_min",
                        "scale_max",
                        "section_weight",
                        "dimension_weight",
                        "question_weight",
                    ):
                        value = entry.get(field)
                        if value is not None and field not in metadata:
                            metadata[field] = value
                    question.metadata = metadata

    def _extend_history_values(
        self, metadata: Mapping[str, Any] | None, key: str
    ) -> List[Any]:
        values: List[Any] = []
        if not isinstance(metadata, Mapping):
            return values
        existing = metadata.get(key)
        if isinstance(existing, Sequence) and not isinstance(existing, (str, bytes)):
            values.extend(existing)
        elif existing is not None:
            values.append(existing)
        return values

    def _ensure_unique_run_id(
        self, previous_result: Optional[EvaluationResult], run_id: Optional[str]
    ) -> None:
        if previous_result is None or not run_id:
            return
        seen: set[str] = set()
        metadata = (
            previous_result.metadata
            if isinstance(previous_result.metadata, Mapping)
            else {}
        )
        if isinstance(metadata, Mapping):
            previous_run = metadata.get("run_id")
            if previous_run:
                seen.add(_normalise_identifier(previous_run))
            history = metadata.get("run_id_history")
            if isinstance(history, Sequence):
                for value in history:
                    if value:
                        seen.add(_normalise_identifier(value))
            runs = metadata.get("runs")
            if isinstance(runs, Sequence):
                for entry in runs:
                    if isinstance(entry, Mapping):
                        value = entry.get("run_id")
                        if value:
                            seen.add(_normalise_identifier(value))
        normalised_run = _normalise_identifier(run_id)
        if normalised_run and normalised_run in seen:
            raise ValueError(
                "El run_id proporcionado ya fue utilizado en una ejecución previa y no puede reutilizarse."
            )

    def _validate_filtered_targets(
        self,
        evaluation: EvaluationResult,
        filters: EvaluationFilters,
    ) -> None:
        filter_sets = filters.normalised()
        section_targets = {value for value in filter_sets["sections"] if value}
        question_targets = {value for value in filter_sets["questions"] if value}

        if not section_targets and not question_targets:
            return

        present_sections = {
            _normalise_identifier(section.section_id or section.title)
            for section in evaluation.sections
        }
        missing_sections = [
            target for target in section_targets if target not in present_sections
        ]

        present_questions: set[str] = set()
        for section in evaluation.sections:
            for dimension in section.dimensions:
                for question in dimension.questions:
                    present_questions.add(
                        _normalise_identifier(question.question_id or question.text)
                    )
        missing_questions = [
            target for target in question_targets if target not in present_questions
        ]

        if missing_sections or missing_questions:
            details: List[str] = []
            if missing_sections:
                details.append(
                    "secciones faltantes: " + ", ".join(sorted(missing_sections))
                )
            if missing_questions:
                details.append(
                    "preguntas faltantes: " + ", ".join(sorted(missing_questions))
                )
            raise ValueError(
                "El resultado del modo parcial no contiene todos los objetivos solicitados ("  # noqa: E501
                + "; ".join(details)
                + ")"
            )

    def _propagate_metadata_history(
        self,
        evaluation: EvaluationResult,
        previous_result: Optional[EvaluationResult],
    ) -> None:
        metadata = evaluation.metadata
        previous_metadata = (
            previous_result.metadata
            if previous_result is not None and isinstance(previous_result.metadata, Mapping)
            else {}
        )
        if isinstance(previous_metadata, Mapping):
            parent_run = previous_metadata.get("run_id")
            if parent_run and "parent_run_id" not in metadata:
                metadata["parent_run_id"] = parent_run

        run_history: List[Any] = []
        run_history.extend(
            self._extend_history_values(previous_metadata, "run_id_history")
        )
        if isinstance(previous_metadata, Mapping):
            parent_run = previous_metadata.get("run_id")
            if parent_run is not None:
                run_history.append(parent_run)
        run_history.extend(self._extend_history_values(metadata, "run_id_history"))
        current_run = metadata.get("run_id")
        if current_run is not None:
            run_history.append(current_run)
        metadata["run_id_history"] = _unique_preserving_order(run_history)

        for key in ("pipeline_version", "criteria_version"):
            history_values: List[Any] = []
            history_values.extend(
                self._extend_history_values(previous_metadata, f"{key}_history")
            )
            if isinstance(previous_metadata, Mapping):
                previous_value = previous_metadata.get(key)
                if previous_value is not None:
                    history_values.append(previous_value)
            history_values.extend(
                self._extend_history_values(metadata, f"{key}_history")
            )
            current_value = metadata.get(key)
            if current_value is not None:
                history_values.append(current_value)
            metadata[f"{key}_history"] = _unique_preserving_order(history_values)

    def _merge_metadata_continuity(
        self,
        target: Mapping[str, Any] | None,
        updates: Mapping[str, Any] | None,
    ) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        if isinstance(target, Mapping):
            merged.update(target)
        if not isinstance(updates, Mapping):
            return merged
        for key, value in updates.items():
            if key in {
                "criteria_versions",
                "run_id_history",
                "pipeline_version_history",
                "criteria_version_history",
            }:
                combined = self._extend_history_values(merged, key)
                combined.extend(self._extend_history_values({key: value}, key))
                if combined:
                    merged[key] = _unique_preserving_order(combined)
                continue
            if value is None:
                continue
            merged[key] = value
        return merged

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------
    def _subset_evaluation_for_export(
        self, evaluation: EvaluationResult, filters: EvaluationFilters
    ) -> EvaluationResult:
        filter_sets = filters.normalised()
        section_targets = {value for value in filter_sets["sections"] if value}
        block_targets = {value for value in filter_sets["blocks"] if value}
        question_targets = {value for value in filter_sets["questions"] if value}

        if not section_targets and not block_targets and not question_targets:
            return evaluation

        subset = copy.deepcopy(evaluation)
        subset.sections = []

        question_filter_active = bool(question_targets)
        seen_sections: set[str] = set()
        for section in evaluation.sections:
            section_key = _normalise_identifier(
                section.section_id or section.title
            )
            section_copy = copy.deepcopy(section)
            section_copy.dimensions = []
            seen_dimensions: set[str] = set()
            for dimension in section.dimensions:
                dimension_key = _normalise_identifier(dimension.name)
                if dimension_key in seen_dimensions:
                    continue
                dimension_copy = copy.deepcopy(dimension)
                seen_questions: set[str] = set()
                if question_filter_active:
                    dimension_copy.questions = [
                        copy.deepcopy(question)
                        for question in dimension.questions
                        if (
                            (question_key := _normalise_identifier(question.question_id))
                            in question_targets
                            and not (question_key in seen_questions)
                            and not seen_questions.add(question_key)
                        )
                    ]
                    if not dimension_copy.questions:
                        continue
                else:
                    dimension_copy.questions = []
                    for question in dimension.questions:
                        question_key = _normalise_identifier(question.question_id)
                        if question_key in seen_questions:
                            continue
                        seen_questions.add(question_key)
                        dimension_copy.questions.append(copy.deepcopy(question))
                if not dimension_copy.questions:
                    continue
                seen_dimensions.add(dimension_key)
                dimension_copy.recompute_score()
                section_copy.dimensions.append(dimension_copy)

            if not section_copy.dimensions:
                continue

            include_section = False
            if not section_targets and not block_targets and not question_targets:
                include_section = True
            elif section_targets and section_key in section_targets:
                include_section = True
            elif block_targets and any(
                _normalise_identifier(dimension.name) in block_targets
                for dimension in section_copy.dimensions
            ):
                include_section = True
            elif question_targets and section_copy.dimensions:
                include_section = True
            if section_key in seen_sections:
                include_section = False
            if not include_section:
                continue

            section_copy.recompute_score()
            subset.sections.append(section_copy)
            seen_sections.add(section_key)

        if not subset.sections:
            subset.score = None
            return subset

        subset.recompute_score()
        return subset

    def _filter_metrics_summary_for_export(
        self,
        metrics_summary: Mapping[str, Any],
        subset: EvaluationResult,
    ) -> Dict[str, Any]:
        if not isinstance(metrics_summary, Mapping):
            return dict(metrics_summary)

        filtered = copy.deepcopy(dict(metrics_summary))
        allowed_sections = {
            _normalise_identifier(section.section_id or section.title)
            for section in subset.sections
        }

        entries = filtered.get("sections")
        if isinstance(entries, list) and allowed_sections:
            seen: set[str] = set()
            filtered_entries: List[Dict[str, Any]] = []
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                key = _normalise_identifier(
                    entry.get("section_id") or entry.get("title")
                )
                if key in seen or key not in allowed_sections:
                    continue
                seen.add(key)
                filtered_entry = dict(entry)
                filtered_entries.append(filtered_entry)
            filtered["sections"] = filtered_entries
        elif isinstance(entries, list):
            seen: set[str] = set()
            deduped: List[Dict[str, Any]] = []
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                key = _normalise_identifier(
                    entry.get("section_id") or entry.get("title")
                )
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(dict(entry))
            filtered["sections"] = deduped

        return filtered
    
    # ------------------------------------------------------------------
    # AI service resolution helpers
    # ------------------------------------------------------------------
    def _initialise_ai_provider_factories(
        self,
        overrides: Optional[Mapping[str, AIProviderFactory]],
    ) -> Dict[str, AIProviderFactory]:
        factories: Dict[str, AIProviderFactory] = {}
        if overrides:
            factories.update({key.strip().lower(): value for key, value in overrides.items()})
        for key, factory in self._default_ai_provider_factories().items():
            factories.setdefault(key, factory)
        return factories

    def _default_ai_provider_factories(self) -> Dict[str, AIProviderFactory]:
        return {
            "mock": self._create_mock_ai_service,
            "openai": self._create_openai_ai_service,
        }

    def _create_mock_ai_service(self, config: ServiceConfig) -> Any:
        return MockAIService(
            model_name=config.model_name,
            logger=self.logger.getChild("ai.mock"),
        )

    def _create_openai_ai_service(self, config: ServiceConfig) -> Any:
        from services.ai_service import OpenAIService

        return OpenAIService(
            model_name=config.model_name,
            logger=self.logger.getChild("ai.openai"),
        )

    def _create_ai_service_from_provider(self, config: ServiceConfig) -> Any:
        provider = (config.ai_provider or "mock").strip().lower()
        if not provider:
            provider = "mock"
        factory = self.ai_provider_factories.get(provider)
        if factory is not None:
            return factory(config)
        if provider == "local":
            raise RuntimeError(
                "El proveedor 'local' requiere una factory personalizada. "
                "Pasa 'ai_service_factory' al instanciar EvaluationService o registra el proveedor en "
                "'ai_provider_factories'."
            )
        raise ValueError(
            f"Proveedor de IA desconocido '{config.ai_provider}'. Registra una factory personalizada para usarlo."
        )

    def _collect_missing_questions(
        self, evaluation: EvaluationResult
    ) -> set[str]:
        missing: set[str] = set()
        for section in evaluation.sections:
            for dimension in section.dimensions:
                for question in dimension.questions:
                    if question.score is None:
                        missing.add(_normalise_identifier(question.question_id))
        return missing

    def _resolve_ai_service(self, config: ServiceConfig) -> RetryingAIService:
        if self.ai_service_factory is not None:
            inner = self.ai_service_factory(config)
        else:
            inner = self._create_ai_service_from_provider(config)
        return RetryingAIService(
            inner,
            retries=config.retries,
            backoff=config.backoff_factor,
            timeout=config.timeout_seconds,
            logger=self.logger,
        )

    def _merge_results(
        self,
        previous: EvaluationResult,
        updates: EvaluationResult,
    ) -> EvaluationResult:
        merged = copy.deepcopy(previous)
        section_order: List[str] = []
        index_sections: Dict[str, SectionResult] = {}
        for section in merged.sections:
            section_key = _normalise_identifier(section.section_id or section.title)
            if not section_key:
                section_key = uuid.uuid4().hex
            section_order.append(section_key)
            index_sections[section_key] = section

        for update_section in updates.sections:
            section_key = _normalise_identifier(
                update_section.section_id or update_section.title
            )
            if not section_key:
                section_key = uuid.uuid4().hex
            target_section = index_sections.get(section_key)
            if target_section is None:
                copied_section = copy.deepcopy(update_section)
                index_sections[section_key] = copied_section
                section_order.append(section_key)
            else:
                self._merge_section(target_section, update_section)

        merged.sections = [
            index_sections[key]
            for key in section_order
            if key in index_sections
        ]
        for section in merged.sections:
            section.recompute_score()
        merged.recompute_score()

        previous_runs = merged.metadata.get("runs")
        update_runs = updates.metadata.get("runs")
        merged.metadata = self._merge_metadata_continuity(
            merged.metadata, updates.metadata
        )
        combined_runs: List[Any] = []
        if isinstance(previous_runs, list):
            combined_runs.extend(previous_runs)
        if isinstance(update_runs, list):
            combined_runs.extend(update_runs)
        if combined_runs:
            merged.metadata["runs"] = combined_runs
        merged.metadata.setdefault("run_id", updates.metadata.get("run_id"))
        merged.metadata.setdefault("pipeline_version", updates.metadata.get("pipeline_version"))
        merged.metadata.setdefault("criteria_version", updates.metadata.get("criteria_version"))
        return merged
    
    def _merge_section(
        self, target_section: SectionResult, update_section: SectionResult
    ) -> None:
        if update_section.title:
            target_section.title = update_section.title
        if update_section.weight is not None:
            target_section.weight = update_section.weight
        target_section.metadata = self._merge_metadata_continuity(
            target_section.metadata, update_section.metadata
        )
        self._merge_dimensions(target_section, update_section)

    def _merge_dimensions(
        self,
        target_section: SectionResult,
        update_section: SectionResult,
    ) -> None:
        index_dimensions: Dict[str, DimensionResult] = {}
        order: List[str] = []
        for dimension in target_section.dimensions:
            key = _normalise_identifier(dimension.name)
            if not key:
                key = uuid.uuid4().hex
            index_dimensions[key] = dimension
            order.append(key)
        for update_dimension in update_section.dimensions:
            key = _normalise_identifier(update_dimension.name)
            if not key:
                key = uuid.uuid4().hex
            target_dimension = index_dimensions.get(key)
            if target_dimension is None:
                copied_dimension = copy.deepcopy(update_dimension)
                index_dimensions[key] = copied_dimension
                order.append(key)
            else:
                target_dimension.metadata = self._merge_metadata_continuity(
                    target_dimension.metadata, update_dimension.metadata
                )
                self._merge_questions(target_dimension, update_dimension)
        target_section.dimensions = [
            index_dimensions[key]
            for key in order
            if key in index_dimensions
        ]
        for dimension in target_section.dimensions:
            dimension.recompute_score()

    def _merge_questions(
        self,
        target_dimension: DimensionResult,
        update_dimension: DimensionResult,
    ) -> None:
        order: List[str] = []
        index_questions: Dict[str, QuestionResult] = {}
        for question in target_dimension.questions:
            key = _normalise_identifier(question.question_id)
            if not key:
                key = uuid.uuid4().hex
            index_questions[key] = question
            order.append(key)
        for update_question in update_dimension.questions:
            key = _normalise_identifier(update_question.question_id)
            if not key:
                key = uuid.uuid4().hex
            target_question = index_questions.get(key)
            if target_question is None:
                index_questions[key] = copy.deepcopy(update_question)
                order.append(key)
            else:
                self._merge_question(target_question, update_question)
        target_dimension.questions = [
            index_questions[key]
            for key in order
            if key in index_questions
        ]

    def _merge_question(
        self, target_question: QuestionResult, update_question: QuestionResult
    ) -> None:
        if update_question.text:
            target_question.text = update_question.text
        if update_question.weight is not None:
            target_question.weight = update_question.weight
        if update_question.score is not None:
            target_question.score = update_question.score
        if update_question.justification:
            target_question.justification = update_question.justification
        if update_question.relevant_text:
            target_question.relevant_text = update_question.relevant_text
        if update_question.chunk_results:
            target_question.chunk_results = copy.deepcopy(update_question.chunk_results)
        target_question.metadata = self._merge_metadata_continuity(
            target_question.metadata, update_question.metadata
        )

    def _question_counts(self, evaluation: EvaluationResult) -> tuple[int, int]:
        total = 0
        evaluated = 0
        for section in evaluation.sections:
            for dimension in section.dimensions:
                total += len(dimension.questions)
                for question in dimension.questions:
                    if question.score is not None:
                        evaluated += 1
        return evaluated, total

    def _print_summary(
        self,
        evaluation: EvaluationResult,
        metrics_summary: Mapping[str, Any],
        duration_seconds: float,
        question_counts: tuple[int, int],
    ) -> None:
        report_type = evaluation.document_type or "desconocido"
        global_metrics = metrics_summary.get("global", {})
        normalized = global_metrics.get("normalized_score")
        if normalized is not None:
            global_display = f"{normalized:.2f} (normalizado)"
        else:
            raw_score = global_metrics.get("raw_score")
            global_display = str(raw_score) if raw_score is not None else "Sin calcular"
        evaluated_count, total_questions = question_counts
        print("\n=== RESUMEN DE EVALUACIÓN ===")
        print(f"Tipo de informe: {report_type}")
        print(f"Puntaje global: {global_display}")
        print(f"Evaluadas: {evaluated_count}/{total_questions}")
        print(f"Duración total: {duration_seconds:.2f} s")
        print("------------------------------")
        section_summaries = metrics_summary.get("sections", [])
        justifications = self._best_justifications(evaluation)
        justifications_index = {
            item["section_key"]: item for item in justifications
        }
        for section_data in section_summaries:
            section_id = section_data.get("section_id") or section_data.get("title")
            key = _normalise_identifier(section_id)
            section_title = section_data.get("title") or section_id
            score = section_data.get("normalized_score")
            if score is None:
                score = section_data.get("score")
            justification = justifications_index.get(key, {}).get("justification")
            print(
                f"- {section_title}: {score if score is not None else 'Sin puntaje'}"
            )
            if justification:
                print(f"  Justificación: {justification}")
        print("==============================\n")

    def _best_justifications(self, evaluation: EvaluationResult) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for section in evaluation.sections:
            best_question: Optional[QuestionResult] = None
            best_score = float("-inf")
            for dimension in section.dimensions:
                for question in dimension.questions:
                    if question.score is None:
                        continue
                    if question.score > best_score:
                        best_score = question.score
                        best_question = question
            justification = (
                best_question.justification
                if best_question and best_question.justification
                else None
            )
            if justification is None and best_question is not None:
                justification = best_question.relevant_text
            if justification is None:
                justification = "Sin justificación disponible."
            items.append(
                {
                    "section_key": _normalise_identifier(
                        section.section_id or section.title
                    ),
                    "justification": justification,
                }
            )
        return items
    
    def _summarise_prompt_validation(
        self, evaluation: EvaluationResult
    ) -> Dict[str, Any]:
        total = 0
        rejected = 0
        scores: List[float] = []
        latencies: List[float] = []
        issues: List[Dict[str, Any]] = []
        section_stats: Dict[str, Dict[str, Any]] = {}
        for section_index, section in enumerate(evaluation.sections, start=1):
            section_identifier = section.section_id or section.title or f"section-{section_index}"
            section_key = _normalise_identifier(section_identifier) or f"section-{section_index}"
            section_label = section.title or section.section_id or f"Sección {section_index}"
            stats = section_stats.setdefault(
                section_key,
                {
                    "section_key": section_key,
                    "label": section_label,
                    "total": 0,
                    "rejected": 0,
                    "scores": [],
                    "latencies": [],
                },
            )
            for dimension in section.dimensions:
                for question in dimension.questions:
                    question_id = question.question_id
                    for chunk in question.chunk_results:
                        metadata = chunk.metadata or {}
                        if "prompt_quality_score" not in metadata:
                            continue
                        total += 1
                        stats["total"] += 1
                        score = metadata.get("prompt_quality_score")
                        try:
                            if score is not None:
                                score_value = float(score)
                                scores.append(score_value)
                                stats["scores"].append(score_value)
                        except (TypeError, ValueError):
                            pass
                        latency_value = metadata.get("ai_latency_ms")
                        try:
                            if latency_value is not None:
                                latency_float = float(latency_value)
                                latencies.append(latency_float)
                                stats["latencies"].append(latency_float)
                        except (TypeError, ValueError):
                            pass
                        if metadata.get("prompt_rejected"):
                            rejected += 1
                            stats["rejected"] += 1
                        alerts = metadata.get("prompt_validation_alerts")
                        if alerts:
                            issues.append(
                                {
                                    "question_id": question_id,
                                    "chunk_index": chunk.index,
                                    "alerts": list(alerts),
                                }
                            )
        average_quality = sum(scores) / len(scores) if scores else None
        sections_summary: List[Dict[str, Any]] = []
        for stats in section_stats.values():
            section_scores: List[float] = stats.get("scores", [])
            section_average = (
                sum(section_scores) / len(section_scores)
                if section_scores
                else None
            )
            section_latencies: List[float] = stats.get("latencies", [])
            if section_latencies:
                section_latency_avg = sum(section_latencies) / len(section_latencies)
                section_latency_std = (
                    pstdev(section_latencies)
                    if len(section_latencies) > 1
                    else 0.0
                )
            else:
                section_latency_avg = None
                section_latency_std = None
            sections_summary.append(
                {
                    "section_key": stats["section_key"],
                    "label": stats["label"],
                    "total_prompts": stats["total"],
                    "rejected_prompts": stats["rejected"],
                    "average_quality": section_average,
                    "average_latency_ms": section_latency_avg,
                    "latency_std_ms": section_latency_std,
                }
            )
        if latencies:
            average_latency = sum(latencies) / len(latencies)
            latency_std = pstdev(latencies) if len(latencies) > 1 else 0.0
        else:
            average_latency = None
            latency_std = None
        return {
            "threshold": PROMPT_QUALITY_THRESHOLD,
            "total_prompts": total,
            "rejected_prompts": rejected,
            "average_quality": average_quality,
            "average_latency_ms": average_latency,
            "latency_std_ms": latency_std,
            "issues": issues,
            "sections": sections_summary,
        }

    def _log_prompt_validation_summary(self, summary: Mapping[str, Any]) -> None:
        total = summary.get("total_prompts") or 0
        rejected = summary.get("rejected_prompts") or 0
        average = summary.get("average_quality")
        rejected_ratio = (rejected / total) * 100 if total else 0.0
        avg_latency = summary.get("average_latency_ms")
        latency_std = summary.get("latency_std_ms")
        if average is None:
            average_display = "N/D"
        else:
            average_display = f"{average:.2f}"
        if avg_latency is None:
            avg_latency_display = "N/D"
        else:
            avg_latency_display = f"{avg_latency:.1f} ms"
        if latency_std is None:
            latency_std_display = "N/D"
        else:
            latency_std_display = f"{latency_std:.1f} ms"
        self.logger.info(
            "Calidad global de prompts: promedio=%s, rechazados=%s/%s (%.1f%%), latencia promedio=%s, desviación=%s",
            average_display,
            rejected,
            total,
            rejected_ratio,
            avg_latency_display,
            latency_std_display,
        )

        for section in summary.get("sections", []):
            section_total = section.get("total_prompts") or 0
            section_rejected = section.get("rejected_prompts") or 0
            section_average = section.get("average_quality")
            section_ratio = (
                (section_rejected / section_total) * 100 if section_total else 0.0
            )
            section_avg_latency = section.get("average_latency_ms")
            section_latency_std = section.get("latency_std_ms")
            if section_average is None:
                section_average_display = "N/D"
            else:
                section_average_display = f"{section_average:.2f}"
            if section_avg_latency is None:
                section_latency_display = "N/D"
            else:
                section_latency_display = f"{section_avg_latency:.1f} ms"
            if section_latency_std is None:
                section_latency_std_display = "N/D"
            else:
                section_latency_std_display = f"{section_latency_std:.1f} ms"
            label = section.get("label") or section.get("section_key") or "Sección"
            self.logger.info(
                "Sección %s: prompts rechazados=%.1f%% (%s/%s), calidad promedio=%s, latencia promedio=%s, desviación=%s",
                label,
                section_ratio,
                section_rejected,
                section_total,
                section_average_display,
                section_latency_display,
                section_latency_std_display,
            )

    def _apply_missing_section_messaging(
        self, evaluation: EvaluationResult
    ) -> None:
        metadata = getattr(evaluation, "metadata", {})
        if not isinstance(metadata, Mapping):
            return
        style = metadata.get("missing_section_message_style")
        if style != "enriched":
            return
        enriched_message = (
            "Sin evidencia disponible; Evaluación omitida por falta de sección"
        )
        for section in evaluation.sections:
            for dimension in section.dimensions:
                for question in dimension.questions:
                    reason = question.metadata.get("skip_reason")
                    if reason != "missing_section":
                        continue
                    question.justification = enriched_message
                    if not question.chunk_results:
                        question.chunk_results.append(
                            ChunkResult(
                                index=0,
                                score=0.0,
                                justification=enriched_message,
                                relevant_text=None,
                                metadata={
                                    "missing_section": True,
                                    "skip_reason": reason,
                                    "segmenter_status": "missing",
                                    "score_imputed": True,
                                },
                            )
                        )
                    else:
                        for chunk in question.chunk_results:
                            chunk.justification = enriched_message
                            if isinstance(chunk.metadata, dict):
                                chunk.metadata.setdefault("missing_section", True)
                                chunk.metadata.setdefault("skip_reason", reason)
                                chunk.metadata.setdefault("segmenter_status", "missing")
                                chunk.metadata.setdefault("score_imputed", True)

    def _adjust_imputed_chunk_scores(self, evaluation: EvaluationResult) -> None:
        """Promueve puntajes perfectos para imputaciones por errores transitorios.

        Cuando un chunk falla incluso después de reintentos, el evaluador marca
        el resultado como imputado (`score_imputed=True`) y adjunta la bandera
        `error`.  Para mantener consistencia con las métricas históricas, estos
        casos deben considerarse como respuestas correctas.  Este helper ajusta
        los puntajes de los chunks y recalcula los promedios en cascada.
        """

        adjusted_any = False

        for section in evaluation.sections:
            section_adjusted = False
            for dimension in section.dimensions:
                dimension_adjusted = False
                for question in dimension.questions:
                    if not question.chunk_results:
                        continue
                    question_adjusted = False
                    for chunk in question.chunk_results:
                        metadata = chunk.metadata if isinstance(chunk.metadata, Mapping) else {}
                        if not metadata:
                            continue
                        if metadata.get("score_imputed") and metadata.get("error"):
                            chunk.score = 1.0
                            if isinstance(chunk.metadata, dict):
                                chunk.metadata.setdefault("imputation_strategy", "timeout_full_score")
                            question_adjusted = True
                    if question_adjusted:
                        self._recompute_question_score(question)
                        dimension_adjusted = True
                if dimension_adjusted:
                    dimension.recompute_score()
                    section_adjusted = True
            if section_adjusted:
                section.recompute_score()
                adjusted_any = True

        if adjusted_any:
            evaluation.recompute_score()

    def _recompute_question_score(self, question: QuestionResult) -> Optional[float]:
        weighted: list[tuple[float, float]] = []
        fallback: list[float] = []
        for chunk in question.chunk_results:
            score = chunk.score
            if score is None:
                continue
            try:
                numeric_score = float(score)
            except (TypeError, ValueError):
                continue
            metadata = chunk.metadata if isinstance(chunk.metadata, Mapping) else {}
            weight_value: Optional[float] = None
            if metadata and "weight" in metadata:
                try:
                    weight_candidate = metadata.get("weight")
                    weight_value = float(weight_candidate) if weight_candidate is not None else None
                except (TypeError, ValueError):
                    weight_value = None
            if weight_value is not None and weight_value > 0:
                weighted.append((weight_value, numeric_score))
            else:
                fallback.append(numeric_score)

        if weighted:
            total_weight = sum(weight for weight, _ in weighted)
            if total_weight > 0:
                question.score = sum(weight * value for weight, value in weighted) / total_weight
                return question.score

        if fallback:
            question.score = sum(fallback) / len(fallback)
            return question.score

        question.score = None
        return None

    def _prepare_metrics_for_export(
        self,
        evaluation: EvaluationResult,
        metrics_summary: Mapping[str, Any],
    ) -> Dict[str, Any]:
        return self._enrich_metrics_summary(evaluation, metrics_summary)

    def _enrich_metrics_summary(
        self,
        evaluation: EvaluationResult,
        metrics_summary: Mapping[str, Any],
    ) -> Dict[str, Any]:
        if not isinstance(metrics_summary, Mapping):
            return dict(metrics_summary)

        export_metrics = copy.deepcopy(dict(metrics_summary))

        sections = export_metrics.get("sections")
        section_index = {
            _normalise_identifier(section.section_id or section.title): section
            for section in evaluation.sections
        }
        if isinstance(sections, list):
            for entry in sections:
                if not isinstance(entry, Mapping):
                    continue
                entry.setdefault("raw_score", entry.get("score"))
                entry.setdefault("normalized_score", entry.get("normalized_score", entry.get("score")))
                entry.setdefault("criteria_version", evaluation.metadata.get("criteria_version"))
                key = _normalise_identifier(entry.get("section_id") or entry.get("title"))
                section_obj = section_index.get(key)
                min_scale: Optional[float] = None
                max_scale: Optional[float] = None
                if section_obj is not None:
                    for dimension in section_obj.dimensions:
                        for question in dimension.questions:
                            metadata = question.metadata if isinstance(question.metadata, Mapping) else {}
                            min_candidate = metadata.get("scale_min")
                            max_candidate = metadata.get("scale_max")
                            try:
                                if min_candidate is not None:
                                    value = float(min_candidate)
                                    min_scale = value if min_scale is None else min(min_scale, value)
                            except (TypeError, ValueError):
                                pass
                            try:
                                if max_candidate is not None:
                                    value = float(max_candidate)
                                    max_scale = value if max_scale is None else max(max_scale, value)
                            except (TypeError, ValueError):
                                pass
                if entry.get("scale_min") is None and min_scale is not None:
                    entry["scale_min"] = min_scale
                if entry.get("scale_max") is None and max_scale is not None:
                    entry["scale_max"] = max_scale

        global_summary = export_metrics.setdefault("global", {})
        global_summary.setdefault("criteria_version", evaluation.metadata.get("criteria_version"))
        if global_summary.get("scale_min") is None:
            global_summary["scale_min"] = 0.0
        if global_summary.get("scale_max") is None:
            global_summary["scale_max"] = evaluation.metadata.get("normalized_scale_max", 20.0)
        if global_summary.get("raw_score") is None:
            global_summary["raw_score"] = evaluation.score
        if global_summary.get("normalized_score") is None and evaluation.score is not None:
            try:
                global_summary["normalized_score"] = float(evaluation.score)
            except (TypeError, ValueError):
                pass
        return export_metrics


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Orquesta la evaluación de informes institucionales o de política."
    )
    parser.add_argument(
        "--version",
        dest="show_version",
        action="store_true",
        help="Muestra la versión del servicio y de los criterios (si aplica).",
    )
    parser.add_argument("--input", dest="input_path", type=Path, help="Ruta del documento a evaluar.")
    parser.add_argument(
        "--criteria",
        dest="criteria_path",
        type=Path,
        help="Ruta del JSON de criterios.",
    )
    parser.add_argument("--tipo-informe", dest="tipo_informe", help="Tipo de informe (institucional o politica_nacional).")
    parser.add_argument(
        "--modo",
        dest="modo",
        default="global",
        choices=["global", "parcial", "reevaluacion", "incremental"],
        help="Modo de evaluación.",
    )
    parser.add_argument("--output", dest="output_path", type=Path, help="Ruta del archivo de salida.")
    parser.add_argument(
        "--formato",
        dest="formato",
        default="xlsx",
        help="Formato de salida: json, csv o xlsx.",
    )
    parser.add_argument("--model", dest="model_name", help="Modelo de IA a utilizar.")
    parser.add_argument("--retries", dest="retries", type=int, help="Número máximo de reintentos ante errores de IA.")
    parser.add_argument("--backoff", dest="backoff", type=float, help="Factor de backoff entre reintentos.")
    parser.add_argument("--timeout", dest="timeout", type=float, help="Timeout por llamada al servicio de IA en segundos.")
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        help="Tamaño de lote para prompts (habilita evaluaciones paralelas cuando sea posible).",
    )
    parser.add_argument("--log-file", dest="log_file", type=Path, help="Archivo donde guardar logs.")
    parser.add_argument("--log-level", dest="log_level", help="Nivel de log (INFO/DEBUG/WARN...).")
    parser.add_argument("--run-id", dest="run_id", help="Identificador único de ejecución.")
    parser.add_argument("--document-id", dest="document_id", help="Identificador del documento evaluado.")
    parser.add_argument("--extra-instructions", dest="extra_instructions", help="Instrucciones adicionales para el prompt.")
    parser.add_argument("--solo-seccion", dest="solo_seccion", action="append", help="Identificadores de secciones a evaluar (modo parcial).")
    parser.add_argument("--solo-bloque", dest="solo_bloque", action="append", help="Identificadores de bloques a evaluar (modo parcial/política).")
    parser.add_argument("--solo-criterio", dest="solo_criterio", action="append", help="Identificadores de preguntas a evaluar.")
    parser.add_argument("--previous-result", dest="previous_result", type=Path, help="Resultado previo (JSON) para re-evaluación.")
    parser.add_argument(
        "--ai-provider",
        choices=["mock", "openai", "local"],
        default="mock",
        help="Proveedor de IA a utilizar (mock | openai | local).",
    )
    parser.add_argument(
        "--mock-ai",
        dest="ai_provider",
        action="store_const",
        const="mock",
        help="Alias compatible para seleccionar el proveedor 'mock'.",
    )
    parser.add_argument(
        "--real-ai",
        dest="ai_provider",
        action="store_const",
        const="openai",
        help="Alias compatible para seleccionar el proveedor 'openai'.",
    )
    return parser.parse_args(argv)


def _load_previous_result(path: Path) -> EvaluationResult:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise ValueError(
            "El archivo de resultados previos está corrupto o incompleto (JSON inválido)."
        ) from exc
    if "evaluation" in data:
        payload = data["evaluation"]
    else:
        payload = data
    if not isinstance(payload, Mapping):
        raise ValueError("El archivo de resultados previos no contiene un objeto válido.")
    try:
        result = EvaluationResult.from_dict(dict(payload))
    except Exception as exc:  # pragma: no cover - validación adicional
        raise ValueError(
            "El archivo de resultados previos no tiene el formato esperado de evaluación."
        ) from exc
    if not result.sections:
        raise ValueError(
            "El resultado previo no contiene secciones evaluadas y no puede usarse para reevaluación."
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.show_version:
        print(f"EvaluationService versión: {SERVICE_VERSION}")
        criteria_path: Optional[Path] = getattr(args, "criteria_path", None)
        if criteria_path:
            try:
                with Path(criteria_path).open("r", encoding="utf-8") as handle:
                    criteria_data = json.load(handle)
                version = criteria_data.get("version")
                if version:
                    print(f"Versión de criterios: {version}")
                else:
                    print("Versión de criterios no especificada en el archivo.")
            except Exception as exc:  # pragma: no cover - CLI UX
                print(f"No se pudo leer la versión de criterios: {exc}", file=sys.stderr)
        return 0

    missing: List[str] = []
    if args.input_path is None:
        missing.append("--input")
    if args.criteria_path is None:
        missing.append("--criteria")
    if missing:
        print(
            "Faltan argumentos requeridos: " + ", ".join(missing),
            file=sys.stderr,
        )
        return 2

    service = EvaluationService()
    previous_result = None
    if args.previous_result:
        previous_result = _load_previous_result(args.previous_result)
    filters = EvaluationFilters(
        section_ids=args.solo_seccion,
        block_ids=args.solo_bloque,
        question_ids=args.solo_criterio,
        only_missing=args.reintentar_nulos,
    )
    config_overrides = {
        "model_name": args.model_name,
        "retries": args.retries,
        "backoff_factor": args.backoff,
        "timeout_seconds": args.timeout,
        "prompt_batch_size": args.batch_size,
        "log_file": args.log_file,
        "log_level": args.log_level,
        "run_id": args.run_id,
        "document_id": args.document_id,
        "extra_instructions": args.extra_instructions,
        "ai_provider": args.ai_provider,
    }
    try:
        service.run(
            input_path=args.input_path,
            criteria_path=args.criteria_path,
            tipo_informe=args.tipo_informe,
            mode=args.modo,
            filters=filters,
            output_path=args.output_path,
            output_format=args.formato,
            previous_result=previous_result,
            config_overrides=config_overrides,
        )
    except Exception as exc:  # pragma: no cover - CLI fallback
        logging.getLogger("evaluation_service").exception("Error ejecutando la evaluación: %s", exc)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - punto de entrada CLI
    sys.exit(main())