"""High level orchestration service for the evaluation pipeline."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
import unicodedata
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

from data.models.document import Document
from data.models.evaluation import (
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

SERVICE_VERSION = "0.1.0"

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
    use_mock_ai: bool = True
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


class MockAIService:
    """Simple heuristic-based AI service used for local development."""

    def __init__(self, model_name: str = "mock-model") -> None:
        self.model_name = model_name

    def evaluate(self, prompt: str, **kwargs: Any) -> Mapping[str, Any]:
        question = kwargs.get("question") or {}
        levels = question.get("niveles", [])
        max_score = 4.0
        if isinstance(levels, Sequence):
            numeric_levels: List[float] = []
            for level in levels:
                if isinstance(level, Mapping) and "valor" in level:
                    try:
                        numeric_levels.append(float(level["valor"]))
                    except (TypeError, ValueError):
                        continue
            if numeric_levels:
                max_score = max(numeric_levels)
        base_value = abs(hash(prompt)) % 1000 / 1000
        score = round(base_value * max_score, 2)
        return {
            "score": score,
            "justification": (
                "Respuesta generada por MockAIService a partir del contenido del fragmento."
            ),
            "relevant_text": None,
            "metadata": {
                "model": self.model_name,
                "mock": True,
            },
        }


class RetryingAIService:
    """Wrapper that retries AI calls and enforces a timeout."""

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
        attempt = 0
        delay = 1.0
        last_exception: Optional[Exception] = None
        while attempt <= self.retries:
            try:
                if self.timeout is None:
                    return self.inner.evaluate(prompt, **kwargs)
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(self.inner.evaluate, prompt, **kwargs)
                    return future.result(timeout=self.timeout)
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
        return {
            "score": None,
            "justification": f"No fue posible obtener respuesta de la IA: {last_exception}",
            "metadata": {
                "error": True,
                "exception_type": type(last_exception).__name__ if last_exception else None,
            },
        }


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
        ai_service_factory: Optional[Any] = None,
        prompt_builder: Optional[PromptBuilder] = None,
    ) -> None:
        self.config = config or ServiceConfig()
        self.loader = loader or DocumentLoader()
        self.cleaner = cleaner or Cleaner()
        self.splitter = splitter or self._build_splitter()
        self.repository = repository or EvaluationRepository()
        self.ai_service_factory = ai_service_factory
        self.prompt_builder = prompt_builder

        self.logger = logging.getLogger("evaluation_service")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            )
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

    # ------------------------------------------------------------------
    # Public API
    def run(
        self,
        *,
        input_path: Optional[Path] = None,
        criteria_path: Optional[Path] = None,
        criteria_data: Optional[Mapping[str, Any]] = None,
        document: Optional[Document] = None,
        tipo_informe: Optional[str] = None,
        mode: str = "completo",
        filters: Optional[EvaluationFilters] = None,
        output_path: Optional[Path] = None,
        output_format: str = "json",
        previous_result: Optional[EvaluationResult] = None,
        config_overrides: Optional[Mapping[str, Any]] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        extra_metadata: Optional[Mapping[str, Any]] = None,
    ) -> tuple[EvaluationResult, Dict[str, Any]]:
        """Execute the full evaluation pipeline and optionally export the result."""

        start_time = time.time()
        config = self.config.with_overrides(**(config_overrides or {}))
        self._configure_logging(config)

        raw_criteria = self._load_criteria(criteria_data, criteria_path)
        resolved_tipo = self._resolve_tipo_informe(tipo_informe, raw_criteria)
        filters = filters or EvaluationFilters()

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
            mode=mode,
            filters=filters,
            previous_result=previous_result,
        )

        ai_service = self._resolve_ai_service(config)
        evaluator = Evaluator(
            ai_service,
            prompt_builder=prompt_builder or self.prompt_builder,
            extra_instructions=config.extra_instructions,
        )
        evaluation = evaluator.evaluate(
            document_obj,
            criteria_for_run,
            document_id=document_id,
        )

        evaluation.metadata.setdefault("criteria_version", raw_criteria.get("version"))
        evaluation.metadata["model_name"] = config.model_name
        evaluation.metadata["run_id"] = config.run_id
        evaluation.metadata["prompt_batch_size"] = config.prompt_batch_size
        evaluation.metadata["retries"] = config.retries
        evaluation.metadata["timeout_seconds"] = config.timeout_seconds
        evaluation.metadata["mode"] = mode
        evaluation.metadata["timestamp"] = datetime.utcnow().isoformat()
        if not filters.is_empty():
            evaluation.metadata["filters"] = filters.to_dict()
        if input_path:
            evaluation.metadata.setdefault("source_path", str(Path(input_path)))

        runs_history = evaluation.metadata.setdefault("runs", [])
        runs_history.append(
            {
                "run_id": config.run_id,
                "mode": mode,
                "model": config.model_name,
                "executed_at": evaluation.metadata["timestamp"],
            }
        )

        if mode == "reevaluacion" and previous_result is not None:
            evaluation = self._merge_results(previous_result, evaluation)

        try:
            metrics_summary = calculate_metrics(evaluation, raw_criteria).to_dict()
        except Exception as exc:  # pragma: no cover - robustez ante métricas
            self.logger.exception("Error calculando métricas: %s", exc)
            metrics_summary = {}

        export_metadata = {
            "config": config.dump(),
            "tipo_informe": resolved_tipo,
            "criteria_version": raw_criteria.get("version"),
        }
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
            if not requested_format or requested_format == "json":
                effective_format = "xlsx"
            target_dir = Path(input_path).parent if input_path else Path.cwd()
            extension = effective_format.lower()
            if extension not in {"json", "csv", "xlsx", "parquet"}:
                extension = "xlsx"
                effective_format = "xlsx"
            export_path = target_dir / f"resultados_{document_id}_{config.run_id}.{extension}"
        elif not effective_format:
            suffix = Path(export_path).suffix.lstrip(".").lower()
            effective_format = format_mapping.get(suffix, suffix or "json")

        self.repository.export(
            evaluation,
            metrics_summary,
            output_path=Path(export_path),
            output_format=effective_format,
            extra_metadata=export_metadata,
        )

        print(f"[💾] Exportado: {Path(export_path)}")

        duration_seconds = time.time() - start_time
        question_counts = self._question_counts(evaluation)
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

    def _prepare_criteria(
        self,
        criteria: Mapping[str, Any],
        *,
        mode: str,
        filters: EvaluationFilters,
        previous_result: Optional[EvaluationResult],
    ) -> Dict[str, Any]:
        mode = mode.lower()
        if mode not in {"completo", "parcial", "reevaluacion"}:
            raise ValueError("El modo debe ser completo, parcial o reevaluacion.")
        filtered = copy.deepcopy(dict(criteria))
        filter_sets = filters.normalised()
        question_ids = set(filter_sets["questions"])
        if mode == "reevaluacion" and filters.only_missing and previous_result is not None:
            missing = self._collect_missing_questions(previous_result)
            if question_ids:
                question_ids = question_ids.union(missing)
            else:
                question_ids = missing
        elif mode == "reevaluacion" and filters.only_missing and previous_result is None:
            self.logger.warning(
                "Se solicitó reintentar preguntas sin puntaje, pero no se proporcionó un resultado previo."
            )
        if mode == "completo" and not question_ids and filters.is_empty():
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
                questions = []
                for question in new_dimension.get("preguntas", []):
                    question_id = _normalise_identifier(
                        question.get("id") or question.get("texto")
                    )
                    if question_targets and question_id not in question_targets:
                        continue
                    questions.append(copy.deepcopy(dict(question)))
                if questions or not question_targets:
                    new_dimension["preguntas"] = (
                        questions if questions else new_dimension.get("preguntas", [])
                    )
                    dimensions.append(new_dimension)
            if dimensions or not question_targets:
                new_section["dimensiones"] = dimensions if dimensions else new_section.get("dimensiones", [])
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
            if questions or not question_targets:
                new_block["preguntas"] = (
                    questions if questions else new_block.get("preguntas", [])
                )
                filtered_blocks.append(new_block)
        return filtered_blocks

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
        elif config.use_mock_ai:
            inner = MockAIService(model_name=config.model_name)
        else:
            raise RuntimeError(
                "No se proporcionó un AIService real. Define 'ai_service_factory' al instanciar "
                "EvaluationService (por ejemplo, EvaluationService(ai_service_factory=mi_factory)) "
                "o ajusta tu config.py para registrar la factory correspondiente."
            )
        # TODO: Integrar aquí el envío en lote/paralelo cuando se implemente batching de prompts.
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
        index_sections: Dict[str, SectionResult] = {}
        for section in merged.sections:
            section_key = _normalise_identifier(section.section_id or section.title)
            index_sections[section_key] = section
        for update_section in updates.sections:
            section_key = _normalise_identifier(
                update_section.section_id or update_section.title
            )
            target_section = index_sections.get(section_key)
            if target_section is None:
                merged.sections.append(update_section)
                index_sections[section_key] = update_section
                target_section = update_section
            self._merge_dimensions(target_section, update_section)
            target_section.recompute_score()
        merged.recompute_score()
        previous_runs = merged.metadata.get("runs")
        update_runs = updates.metadata.get("runs")
        merged.metadata.update(updates.metadata)
        combined_runs: List[Any] = []
        if isinstance(previous_runs, list):
            combined_runs.extend(previous_runs)
        if isinstance(update_runs, list):
            combined_runs.extend(update_runs)
        if combined_runs:
            merged.metadata["runs"] = combined_runs
        return merged

    def _merge_dimensions(
        self,
        target_section: SectionResult,
        update_section: SectionResult,
    ) -> None:
        index_dimensions: Dict[str, DimensionResult] = {}
        for dimension in target_section.dimensions:
            index_dimensions[_normalise_identifier(dimension.name)] = dimension
        for update_dimension in update_section.dimensions:
            key = _normalise_identifier(update_dimension.name)
            target_dimension = index_dimensions.get(key)
            if target_dimension is None:
                target_section.dimensions.append(update_dimension)
                index_dimensions[key] = update_dimension
                target_dimension = update_dimension
            self._merge_questions(target_dimension, update_dimension)
            target_dimension.recompute_score()

    def _merge_questions(
        self,
        target_dimension: DimensionResult,
        update_dimension: DimensionResult,
    ) -> None:
        index_questions: Dict[str, QuestionResult] = {}
        for question in target_dimension.questions:
            index_questions[_normalise_identifier(question.question_id)] = question
        for update_question in update_dimension.questions:
            key = _normalise_identifier(update_question.question_id)
            index_questions[key] = update_question
        target_dimension.questions = list(index_questions.values())

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
    parser.add_argument("--modo", dest="modo", default="completo", choices=["completo", "parcial", "reevaluacion"], help="Modo de evaluación.")
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
    parser.add_argument("--batch-size", dest="batch_size", type=int, help="Tamaño de lote para prompts (reservado para batching futuro).")
    parser.add_argument("--log-file", dest="log_file", type=Path, help="Archivo donde guardar logs.")
    parser.add_argument("--log-level", dest="log_level", help="Nivel de log (INFO/DEBUG/WARN...).")
    parser.add_argument("--run-id", dest="run_id", help="Identificador único de ejecución.")
    parser.add_argument("--document-id", dest="document_id", help="Identificador del documento evaluado.")
    parser.add_argument("--extra-instructions", dest="extra_instructions", help="Instrucciones adicionales para el prompt.")
    parser.add_argument("--solo-seccion", dest="solo_seccion", action="append", help="Identificadores de secciones a evaluar (modo parcial).")
    parser.add_argument("--solo-bloque", dest="solo_bloque", action="append", help="Identificadores de bloques a evaluar (modo parcial/política).")
    parser.add_argument("--solo-criterio", dest="solo_criterio", action="append", help="Identificadores de preguntas a evaluar.")
    parser.add_argument("--previous-result", dest="previous_result", type=Path, help="Resultado previo (JSON) para re-evaluación.")
    parser.add_argument("--reintentar-nulos", dest="reintentar_nulos", action="store_true", help="En reevaluación, volver a ejecutar preguntas sin puntaje.")
    parser.add_argument("--mock-ai", dest="mock_ai", action="store_true", help="Usar servicio de IA simulado (por defecto).")
    parser.add_argument("--real-ai", dest="mock_ai", action="store_false", help="Usar servicio de IA real configurado mediante factory.")
    parser.set_defaults(mock_ai=True)
    return parser.parse_args(argv)


def _load_previous_result(path: Path) -> EvaluationResult:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if "evaluation" in data:
        payload = data["evaluation"]
    else:
        payload = data
    if not isinstance(payload, Mapping):
        raise ValueError("El archivo de resultados previos no contiene un objeto válido.")
    return EvaluationResult.from_dict(dict(payload))


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
        "use_mock_ai": args.mock_ai,
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