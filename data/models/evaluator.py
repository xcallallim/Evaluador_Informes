"""Evaluator module responsible for orchestrating AI powered scoring."""

from __future__ import annotations

import logging
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Set, Tuple

from data.models.document import Document
from data.models.evaluation import (
    ChunkResult,
    DimensionResult,
    EvaluationResult,
    QuestionResult,
    SectionResult,
)


logger = logging.getLogger(__name__)


class PromptBuilder(Protocol):
    """Callable used to craft prompts for the AI service."""

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
        extra_instructions: Optional[str] = None,
    ) -> str:
        ...


@dataclass(slots=True)
class ModelResponse:
    """Normalised response returned by the AI service."""

    score: Optional[float]
    justification: Optional[str] = None
    relevant_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def default_prompt_builder(
    *,
    document: Document,
    criteria: Mapping[str, Any],
    section: Mapping[str, Any],
    dimension: Mapping[str, Any],
    question: Mapping[str, Any],
    chunk_text: str,
    chunk_metadata: Mapping[str, Any],
    extra_instructions: Optional[str] = None,
) -> str:
    """Generate a verbose prompt to guide the language model."""

    document_title = document.metadata.get("title") or document.metadata.get("nombre")
    document_title = document_title or criteria.get("titulo") or "Documento evaluado"
    info_lines = [
        f"Documento: {document_title}",
    ]
    if criteria.get("tipo_informe"):
        info_lines.append(f"Tipo de informe: {criteria['tipo_informe']}")
    section_title = section.get("titulo") or section.get("nombre") or "Sección"
    info_lines.append(f"Sección: {section_title}")
    info_lines.append(f"Dimensión: {dimension.get('nombre', 'Dimensión')}" )
    info_lines.append(f"Pregunta: {question.get('texto', question.get('id', ''))}")

    chunk_lines = ["Analiza el siguiente fragmento y asigna un puntaje objetivo."]
    if chunk_metadata:
        chunk_lines.append(f"Metadatos del fragmento: {dict(chunk_metadata)}")
    chunk_lines.append("Fragmento:")
    chunk_lines.append(chunk_text or "<sin contenido>")

    prompt_parts = ["\n".join(info_lines), "\n".join(chunk_lines)]
    if extra_instructions:
        prompt_parts.append(extra_instructions)
    return "\n\n".join(part for part in prompt_parts if part)


class Evaluator:
    """Evaluates a document using a set of criteria and an AI service."""

    def __init__(
        self,
        ai_service: Any,
        *,
        prompt_builder: PromptBuilder | None = None,
        extra_instructions: Optional[str] = None,
    ) -> None:
        self.ai_service = ai_service
        self.prompt_builder: PromptBuilder = prompt_builder or default_prompt_builder
        self.extra_instructions = extra_instructions

    def evaluate(
        self,
        document: Document,
        criteria: Mapping[str, Any],
        *,
        document_id: Optional[str] = None,
    ) -> EvaluationResult:
        """Evaluate ``document`` according to ``criteria``."""

        evaluation = EvaluationResult(
            document_id=document_id or document.metadata.get("id"),
            document_type=criteria.get("tipo_informe"),
            criteria_source=criteria.get("version") or criteria.get("criteria_source"),
            methodology=criteria.get("metodologia") or criteria.get("tipo_metodologia"),
            metadata={
                "criteria_version": criteria.get("version"),
                "criteria_description": criteria.get("descripcion"),
                "methodology": criteria.get("metodologia")
                or criteria.get("tipo_metodologia"),
            },
        )

        sections = list(criteria.get("secciones", []) or [])
        if sections:
            for section_data in sections:
                section_result = self._evaluate_section(document, criteria, section_data)
                evaluation.sections.append(section_result)
                section_result.recompute_score()
        else:
            for section_result in self._evaluate_global(document, criteria):
                evaluation.sections.append(section_result)
                section_result.recompute_score()
        
        status_counts = {"found": 0, "missing": 0, "empty": 0}
        missing_sections: List[str] = []
        empty_sections: List[str] = []
        for section in evaluation.sections:
            segmenter_metadata = section.metadata.get("segmenter")
            status = None
            if isinstance(segmenter_metadata, Mapping):
                status = segmenter_metadata.get("status")
            status = str(status or "found")
            if status not in status_counts:
                status = "found"
            status_counts[status] += 1
            identifier = section.section_id or section.title or ""
            if status == "missing" and identifier:
                missing_sections.append(identifier)
            elif status == "empty" and identifier:
                empty_sections.append(identifier)

        if evaluation.sections and status_counts["found"] == 0:
            logger.warning(
                "⚠️ No se detectaron secciones válidas para evaluar; se devolverán puntajes cero."
            )

        evaluation.metadata.setdefault("segmenter_summary", {})
        segmenter_summary = evaluation.metadata["segmenter_summary"]
        if isinstance(segmenter_summary, dict):
            segmenter_summary["status_counts"] = dict(status_counts)
            segmenter_summary["missing_sections"] = list(missing_sections)
            segmenter_summary["empty_sections"] = list(empty_sections)
            segmenter_summary["total_sections"] = len(evaluation.sections)
            segmenter_summary["issues_detected"] = bool(
                missing_sections or empty_sections
            )

        evaluation.recompute_score()
        return evaluation

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _evaluate_section(
        self,
        document: Document,
        criteria: Mapping[str, Any],
        section_data: Mapping[str, Any],
    ) -> SectionResult:
        section_result = SectionResult(
            title=section_data.get("titulo") or section_data.get("nombre", "Sección"),
            section_id=section_data.get("id") or section_data.get("id_segmenter"),
            weight=section_data.get("ponderacion"),
            metadata={key: section_data[key] for key in ("id", "id_segmenter", "tipo") if key in section_data},
        )

        section_status = self._section_status(document, section_data)
        skip_reason, skip_justification = self._handle_segmenter_status(
            section_result,
            section_status,
            entity_label="Sección",
        )

        if skip_reason is not None and section_status == "empty":
            self._populate_skipped_section(
                section_result,
                [
                    (
                        dimension_data,
                        list(dimension_data.get("preguntas", [])),
                    )
                    for dimension_data in section_data.get("dimensiones", []) or []
                ],
                skip_reason=skip_reason,
                skip_justification=skip_justification or "",
                segmenter_status=section_status,
            )
            return section_result

        for dimension_data in section_data.get("dimensiones", []):
            dimension_result = self._evaluate_dimension(
                document,
                criteria,
                section_data,
                dimension_data,
            )
            section_result.dimensions.append(dimension_result)
            dimension_result.recompute_score()

        if skip_reason is not None:
            self._override_section_as_skipped(
                section_result,
                skip_reason=skip_reason,
                skip_justification=skip_justification or "",
                segmenter_status=section_status,
            )
        return section_result

    def _evaluate_dimension(
        self,
        document: Document,
        criteria: Mapping[str, Any],
        section_data: Mapping[str, Any],
        dimension_data: Mapping[str, Any],
    ) -> DimensionResult:
        dimension_result = DimensionResult(
            name=dimension_data.get("nombre", "Dimensión"),
            weight=dimension_data.get("ponderacion"),
            method=dimension_data.get("metodo_agregacion"),
            metadata={
                key: dimension_data[key]
                for key in ("tipo_escala", "descripcion", "id")
                if key in dimension_data
            },
        )

        for question_data in dimension_data.get("preguntas", []):
            question_result = self._evaluate_question(
                document,
                criteria,
                section_data,
                dimension_data,
                question_data,
            )
            dimension_result.questions.append(question_result)
        return dimension_result

    def _evaluate_question(
        self,
        document: Document,
        criteria: Mapping[str, Any],
        section_data: Mapping[str, Any],
        dimension_data: Mapping[str, Any],
        question_data: Mapping[str, Any],
    ) -> QuestionResult:
        chunk_results: List[ChunkResult] = []
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
                if not self._should_use_chunk_for_missing_section(
                    document, section_key, chunk_section_key
                ):
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
            start_time = time.perf_counter()
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
            except Exception as exc:  # pragma: no cover - exercised via tests
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                logger.exception(
                    "Error invoking AI service for question %s (chunk %s)",
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
            else:
                elapsed_ms = (time.perf_counter() - start_time) * 1000

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
            metadata={key: question_data[key] for key in ("tipo", "descripcion") if key in question_data},
        )
        return question_result

    def _aggregate_chunk_scores(self, chunk_results: Iterable[ChunkResult]) -> Optional[float]:
        weighted: List[tuple[float, float]] = []
        fallback: List[float] = []
        for chunk in chunk_results:
            if chunk.score is None:
                continue
            weight = self._get_chunk_weight(chunk)
            try:
                score_value = float(chunk.score)
            except (TypeError, ValueError):
                continue
            if weight is not None and weight > 0:
                try:
                    weighted.append((float(weight), score_value))
                    continue
                except (TypeError, ValueError):
                    pass
            fallback.append(score_value)

        if weighted:
            total_weight = sum(weight for weight, _ in weighted)
            if total_weight > 0:
                return sum(weight * score for weight, score in weighted) / total_weight
        if fallback:
            return sum(fallback) / len(fallback)
        return None

    def _select_justification(
        self, chunk_results: Iterable[ChunkResult]
    ) -> tuple[Optional[str], Optional[str]]:
        best_chunk = None
        best_score = float("-inf")
        for chunk in chunk_results:
            if chunk.score is None:
                continue
            if chunk.score > best_score:
                best_score = chunk.score
                best_chunk = chunk
        if best_chunk is None:
            for chunk in chunk_results:
                best_chunk = chunk
                break
        if best_chunk is None:
            return None, None
        return best_chunk.justification, best_chunk.relevant_text

    def _iter_chunks(self, document: Document) -> Iterable[tuple[int, Any]]:
        if document.chunks:
            for index, chunk in enumerate(document.chunks):
                yield index, chunk
            return

        class _PseudoChunk:
            def __init__(self, text: str) -> None:
                self.page_content = text
                self.metadata: Dict[str, Any] = {}

        yield 0, _PseudoChunk(document.content or "")

    @staticmethod
    def _normalise_response(response: Any) -> ModelResponse:
        if isinstance(response, ModelResponse):
            return response
        if isinstance(response, Mapping):
            metadata = response.get("metadata")
            if not isinstance(metadata, Mapping):
                metadata = {}
            return ModelResponse(
                score=response.get("score"),
                justification=response.get("justification"),
                relevant_text=response.get("relevant_text"),
                metadata=dict(metadata),
            )
        try:
            text_response = str(response)
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to serialise AI response of type %s", type(response))
            text_response = "<respuesta no disponible>"
        logger.warning(
            "Unexpected AI response type %s; treating as justification-only payload.",
            type(response),
        )
        return ModelResponse(score=None, justification=text_response)
    
    @staticmethod
    def _normalise_label(value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip().lower()
        if not text:
            return ""
        normalized = unicodedata.normalize("NFKD", text)
        return "".join(ch for ch in normalized if not unicodedata.combining(ch))
    
    def _section_status(
        self, document: Document, section_data: Mapping[str, Any]
    ) -> str:
        metadata = getattr(document, "metadata", {})
        missing_flag = False
        if isinstance(metadata, Mapping):
            missing_flag = bool(metadata.get("segmenter_missing_sections"))

        sections = getattr(document, "sections", None)
        if not isinstance(sections, Mapping):
            if missing_flag or not getattr(document, "chunks", None):
                return "missing"
            return "found"

        if not sections:
            if missing_flag:
                return "missing"
            if getattr(document, "chunks", None):
                return "found"
            return "missing"

        normalised_sections: Dict[str, str] = {}
        for key, value in sections.items():
            normalised_key = self._normalise_label(key)
            if not normalised_key:
                continue
            text = value if isinstance(value, str) else str(value or "")
            normalised_sections[normalised_key] = text

        candidate_keys = {
            self._normalise_label(section_data.get("id")),
            self._normalise_label(section_data.get("id_segmenter")),
            self._normalise_label(section_data.get("titulo")),
            self._normalise_label(section_data.get("nombre")),
        }
        candidate_keys = {key for key in candidate_keys if key}
        if not candidate_keys:
            return "found"

        for key in candidate_keys:
            if key not in normalised_sections:
                continue
            text = normalised_sections[key]
            if text.strip():
                return "found"
            return "missing" if missing_flag else "empty"

        if missing_flag or not normalised_sections:
            return "missing"

        return "missing"

    def _apply_score_constraints(
        self,
        score: Any,
        *,
        criteria: Mapping[str, Any],
        dimension: Mapping[str, Any],
        question: Mapping[str, Any],
    ) -> Tuple[Optional[float], bool]:
        if score is None:
            return None, False
        try:
            numeric = float(score)
        except (TypeError, ValueError):
            logger.warning("Discarding non-numeric score returned by AI: %s", score)
            return None, True

        report_type = self._normalise_label(criteria.get("tipo_informe"))
        dimension_name = self._normalise_label(dimension.get("nombre"))

        adjusted = numeric
        if report_type == "institucional":
            if dimension_name == "estructura":
                allowed = (0.0, 1.0)
            else:
                allowed = tuple(float(level) for level in range(0, 5))
            adjusted = min(allowed, key=lambda candidate: (abs(candidate - numeric), candidate))
        elif report_type in {
            "pn",
            "politica_nacional",
            "politica nacional",
            "politica-nacional",
        }:
            adjusted = max(0.0, min(2.0, numeric))
            adjusted = round(adjusted * 2.0) / 2.0

        changed = abs(adjusted - numeric) > 1e-9
        return float(adjusted), changed

    def _resolve_chunk_weight(self, metadata: Mapping[str, Any]) -> Optional[float]:
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
            return self._resolve_chunk_weight(nested)
        return None

    def _get_chunk_weight(self, chunk: ChunkResult) -> Optional[float]:
        return self._resolve_chunk_weight(chunk.metadata)

    def _evaluate_global(
        self, document: Document, criteria: Mapping[str, Any]
    ) -> Iterable[SectionResult]:
        bloques = criteria.get("bloques") or []
        for bloque in bloques:
            section_result = SectionResult(
                title=bloque.get("titulo") or bloque.get("nombre", "Bloque"),
                section_id=bloque.get("id") or bloque.get("id_segmenter"),
                weight=bloque.get("ponderacion"),
                metadata={
                    key: bloque[key]
                    for key in ("tipo", "descripcion", "codigo")
                    if key in bloque
                },
            )
            section_status = self._section_status(document, bloque)
            skip_reason, skip_justification = self._handle_segmenter_status(
                section_result,
                section_status,
                entity_label="Bloque",
            )
            if skip_reason is not None:
                self._populate_skipped_section(
                    section_result,
                    [
                        (
                            bloque,
                            list(bloque.get("preguntas", [])),
                        )
                    ],
                    skip_reason=skip_reason,
                    skip_justification=skip_justification or "",
                    segmenter_status=section_status,
                )
                yield section_result
                continue
            dimension_result = DimensionResult(
                name=bloque.get("nombre", section_result.title),
                weight=bloque.get("ponderacion"),
                method=bloque.get("metodo_agregacion"),
                metadata={
                    key: bloque[key]
                    for key in ("tipo_escala", "descripcion", "id")
                    if key in bloque
                },
            )
            for question_data in bloque.get("preguntas", []):
                question_result = self._evaluate_question(
                    document,
                    criteria,
                    bloque,
                    bloque,
                    question_data,
                )
                dimension_result.questions.append(question_result)
            dimension_result.recompute_score()
            section_result.dimensions.append(dimension_result)
            yield section_result

    def _handle_segmenter_status(
        self,
        section_result: SectionResult,
        status: str,
        *,
        entity_label: str,
    ) -> tuple[Optional[str], Optional[str]]:
        segmenter_metadata = section_result.metadata.setdefault("segmenter", {})
        if not isinstance(segmenter_metadata, dict):
            return None, None

        segmenter_metadata.setdefault("status", status)
        segmenter_metadata.setdefault("detected", status != "missing")
        if status == "missing":
            segmenter_metadata.setdefault("missing", True)
            logger.warning(
                "⚠️ %s '%s' no encontrada en el documento; se imputará puntaje 0.",
                entity_label,
                section_result.title,
            )
            return "missing_section", "Evaluación omitida por falta de sección"
        if status == "empty":
            segmenter_metadata.setdefault("empty", True)
            logger.warning(
                "⚠️ %s '%s' sin contenido para evaluar; se imputará puntaje 0.",
                entity_label,
                section_result.title,
            )
            return "empty_section", "Sin contenido para evaluar"
        return None, None

    def _populate_skipped_section(
        self,
        section_result: SectionResult,
        dimension_entries: Iterable[tuple[Mapping[str, Any], Iterable[Mapping[str, Any]]]],
        *,
        skip_reason: str,
        skip_justification: str,
        segmenter_status: str,
    ) -> None:
        segmenter_metadata = section_result.metadata.setdefault("segmenter", {})
        if isinstance(segmenter_metadata, dict):
            segmenter_metadata.setdefault("skip_reason", skip_reason)
            segmenter_metadata.setdefault("skipped", True)
            segmenter_metadata.setdefault("segmenter_status", segmenter_status)

        populated = False
        for dimension_data, questions in dimension_entries:
            dimension_result = DimensionResult(
                name=(
                    dimension_data.get("nombre")
                    or dimension_data.get("titulo")
                    or dimension_data.get("id")
                    or "Dimensión"
                ),
                weight=dimension_data.get("ponderacion"),
                method=dimension_data.get("metodo_agregacion"),
                metadata={
                    key: dimension_data[key]
                    for key in ("tipo_escala", "descripcion", "id", "codigo")
                    if key in dimension_data
                },
            )
            dimension_result.metadata.setdefault("skipped", True)
            dimension_result.metadata.setdefault("skip_reason", skip_reason)
            dimension_result.metadata.setdefault("segmenter_status", segmenter_status)

            for question_data in questions or []:
                dimension_result.questions.append(
                    self._build_skipped_question_result(
                        question_data,
                        skip_reason=skip_reason,
                        skip_justification=skip_justification,
                        segmenter_status=segmenter_status,
                    )
                )

            dimension_result.recompute_score()
            if dimension_result.score is None:
                dimension_result.score = 0.0
            section_result.dimensions.append(dimension_result)
            populated = True

        if not populated:
            # Garantizar una estructura consistente aunque no haya preguntas.
            placeholder_dimension = DimensionResult(
                name="Dimensión",
                weight=1.0,
                method=None,
                metadata={
                    "skipped": True,
                    "skip_reason": skip_reason,
                    "segmenter_status": segmenter_status,
                },
            )
            placeholder_dimension.score = 0.0
            section_result.dimensions.append(placeholder_dimension)

        section_result.recompute_score()
        if section_result.score is None:
            section_result.score = 0.0
        section_result.metadata.setdefault("skipped", True)
        section_result.metadata.setdefault("skip_reason", skip_reason)
        section_result.metadata.setdefault("segmenter_status", segmenter_status)

    def _build_skipped_question_result(
        self,
        question_data: Mapping[str, Any],
        *,
        skip_reason: str,
        skip_justification: str,
        segmenter_status: str,
    ) -> QuestionResult:
        question_metadata = {
            key: question_data[key]
            for key in ("tipo", "descripcion")
            if key in question_data
        }
        question_metadata.setdefault("skipped", True)
        question_metadata.setdefault("skip_reason", skip_reason)
        question_metadata.setdefault("segmenter_status", segmenter_status)
        return QuestionResult(
            question_id=str(question_data.get("id") or question_data.get("texto", "")),
            text=question_data.get("texto", ""),
            weight=question_data.get("ponderacion"),
            score=0.0,
            justification=skip_justification,
            relevant_text=None,
            chunk_results=[],
            metadata=question_metadata,
        )

    def _override_section_as_skipped(
        self,
        section_result: SectionResult,
        *,
        skip_reason: str,
        skip_justification: str,
        segmenter_status: str,
    ) -> None:
        segmenter_metadata = section_result.metadata.setdefault("segmenter", {})
        if isinstance(segmenter_metadata, dict):
            segmenter_metadata.setdefault("status", segmenter_status)
            segmenter_metadata.setdefault("skip_reason", skip_reason)
            segmenter_metadata.setdefault("skipped", True)
            segmenter_metadata.setdefault("segmenter_status", segmenter_status)

        for dimension_result in section_result.dimensions:
            dimension_result.metadata.setdefault("skipped", True)
            dimension_result.metadata.setdefault("skip_reason", skip_reason)
            dimension_result.metadata.setdefault("segmenter_status", segmenter_status)
            for question_result in dimension_result.questions:
                question_result.metadata.setdefault("skipped", True)
                question_result.metadata.setdefault("skip_reason", skip_reason)
                question_result.metadata.setdefault("segmenter_status", segmenter_status)
                question_result.score = 0.0
                question_result.justification = skip_justification
            dimension_result.recompute_score()
            if dimension_result.score is None:
                dimension_result.score = 0.0

        if not section_result.dimensions:
            placeholder = DimensionResult(
                name="Dimensión",
                weight=1.0,
                method=None,
                metadata={
                    "skipped": True,
                    "skip_reason": skip_reason,
                    "segmenter_status": segmenter_status,
                },
            )
            placeholder.score = 0.0
            section_result.dimensions.append(placeholder)

        section_result.recompute_score()
        if section_result.score is None:
            section_result.score = 0.0
        section_result.metadata.setdefault("skipped", True)
        section_result.metadata.setdefault("skip_reason", skip_reason)
        section_result.metadata.setdefault("segmenter_status", segmenter_status)

    def _should_use_chunk_for_missing_section(
        self,
        document: Document,
        section_key: str,
        chunk_section_key: str,
    ) -> bool:
        metadata = getattr(document, "metadata", {})
        if isinstance(metadata, Mapping) and metadata.get("segmenter_missing_sections"):
            return True
        return False


__all__ = ["Evaluator", "ModelResponse", "PromptBuilder", "default_prompt_builder"]