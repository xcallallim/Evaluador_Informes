"""Prompt composition utilities aligned with the CEPLAN evaluation methodology."""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Protocol, Sequence

from data.models.document import Document


class PromptBuilder(Protocol):
    """Protocol describing callables able to craft prompts for the AI."""

    def __call__(
        self,
        *,
        document: Document,
        criteria: Mapping[str, Any],
        section: Mapping[str, Any],
        dimension: Optional[Mapping[str, Any]],
        question: Optional[Mapping[str, Any]],
        chunk_text: str,
        chunk_metadata: Optional[Mapping[str, Any]],
        extra_instructions: Optional[str] = None,
    ) -> str:
        ...


@dataclass(slots=True)
class PromptContext:
    """Container with the ingredients required to build a prompt.

    The evaluator supports two primary modes when requesting an evaluation:

    * **Global (sección completa)** – requires ``document``, ``criteria``,
      ``section`` and ``chunk_text``. ``dimension`` and ``question`` may be
      omitted (``None``) to signal that the AI should assess the whole section.
    * **Parcial (criterio/pregunta)** – uses the same core fields as the global
      mode and additionally expects ``dimension`` and ``question`` describing the
      specific criterion or guiding question under review.

    ``chunk_metadata`` is optional in both modes and allows injecting
    traceability or pagination hints for the evaluated fragment.
    """

    document: Document
    criteria: Mapping[str, Any]
    section: Mapping[str, Any]
    chunk_text: str
    dimension: Optional[Mapping[str, Any]] = None
    question: Optional[Mapping[str, Any]] = None
    chunk_metadata: Optional[Mapping[str, Any]] = None
    extra_instructions: Optional[str] = None

    def __post_init__(self) -> None:  # pragma: no cover - simple default guard
        if self.chunk_metadata is None:
            self.chunk_metadata = {}


@dataclass(frozen=True)
class ScaleLevel:
    """Definition of a single level in an evaluation scale."""

    value: float
    description: str
    label: Optional[str] = None


@dataclass(slots=True)
class _PromptData:
    """Normalised representation of the evaluation context."""

    document_title: str
    report_type: str
    methodology: Optional[str]
    methodology_version: Optional[str]
    criteria_name: Optional[str]
    section_title: str
    dimension_label: Optional[str]
    question_label: Optional[str]
    chunk_text: str
    chunk_metadata: Mapping[str, Any]
    extra_instructions: Optional[str]
    was_truncated: bool


class BasePromptBuilder(ABC):
    """Base class that validates context data and renders the final prompt."""

    fragment_start_marker = "<<<INICIO_FRAGMENTO>>>"
    fragment_end_marker = "<<<FIN_FRAGMENTO>>>"
    truncation_suffix = "… [texto truncado]"
    default_max_chunk_chars = 12_000
    _spanish_markers = {
        " el ",
        " la ",
        " los ",
        " las ",
        " que ",
        " de ",
        " del ",
        " en ",
        " para ",
        " con ",
    }

    def __init__(
        self,
        *,
        max_chunk_chars: int = default_max_chunk_chars,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if max_chunk_chars <= len(self.truncation_suffix):
            raise ValueError("max_chunk_chars debe ser mayor que la longitud del sufijo de truncamiento.")
        self.max_chunk_chars = max_chunk_chars
        self._last_quality_metadata: Optional[Dict[str, Any]] = None
        logger_name = f"{__name__}.{self.__class__.__name__}"
        self.logger = logger or logging.getLogger(logger_name)
        self.logger.debug("logger_initialized", extra={"event": "logger_initialized", "logger_name": logger_name})

    def __call__(
        self,
        *,
        document: Document,
        criteria: Mapping[str, Any],
        section: Mapping[str, Any],
        dimension: Optional[Mapping[str, Any]],
        question: Optional[Mapping[str, Any]],
        chunk_text: str,
        chunk_metadata: Optional[Mapping[str, Any]],
        extra_instructions: Optional[str] = None,
    ) -> str:
        context = PromptContext(
            document=document,
            criteria=criteria,
            section=section,
            dimension=dimension,
            question=question,
            chunk_text=chunk_text,
            chunk_metadata=chunk_metadata,
            extra_instructions=extra_instructions,
        )
        return self.build_from_context(context)

    def build_from_context(self, context: PromptContext) -> str:
        """Generate a prompt using a validated :class:`PromptContext`."""

        self.logger.info(
            "build_prompt_start",
            extra={
                "event": "build_prompt_start",
                "report_type": self._clean_str(context.criteria.get("tipo_informe")),
            },
        )
        self.validate_context(context)
        data = self._normalise_context(context)
        prompt = self._compose_prompt(data)
        prompt = self._compact_prompt(prompt)
        self._last_quality_metadata = self._build_quality_metadata(data, prompt)
        self.logger.info(
            "build_prompt_complete",
            extra={
                "event": "build_prompt_complete",
                "prompt_length": len(prompt),
                "was_truncated": data.was_truncated,
            },
        )
        return prompt

    def validate_context(self, context: PromptContext) -> None:
        """Ensure the incoming context includes the required evaluation data."""

        if not isinstance(context.document, Document):
            raise TypeError("'document' debe ser una instancia de data.models.document.Document.")
        if not isinstance(context.criteria, MappingABC):
            raise TypeError("'criteria' debe ser un mapeo.")
        if not isinstance(context.section, MappingABC):
            raise TypeError("'section' debe ser un mapeo.")
        if context.dimension is not None and not isinstance(context.dimension, MappingABC):
            raise TypeError("'dimension' debe ser un mapeo cuando se proporciona.")
        if context.question is not None and not isinstance(context.question, MappingABC):
            raise TypeError("'question' debe ser un mapeo cuando se proporciona.")
        if context.chunk_metadata is not None and not isinstance(context.chunk_metadata, MappingABC):
            raise TypeError("'chunk_metadata' debe ser un mapeo cuando se proporciona.")

        cleaned_chunk = self._clean_str(context.chunk_text)
        if not cleaned_chunk:
            raise ValueError("'chunk_text' debe contener información para evaluar.")
        self._validate_language(cleaned_chunk)

        if context.question is not None and context.dimension is None:
            raise ValueError("Debe proporcionar 'dimension' cuando se especifica una 'question'.")

        methodology = self._clean_str(
            context.criteria.get("metodologia") or context.criteria.get("tipo_metodologia")
        )
        methodology_version = self._clean_str(
            context.criteria.get("version") or context.criteria.get("criteria_source")
        )
        if methodology_version and not methodology:
            raise ValueError(
                "Los criterios indican una versión metodológica pero no especifican la metodología aplicada."
            )

    # ------------------------------------------------------------------
    # Template methods implemented by subclasses
    @property
    @abstractmethod
    def scale_label(self) -> str:
        """Human readable name for the scale, including range information."""

    @property
    @abstractmethod
    def scale_levels(self) -> Sequence[ScaleLevel]:
        """Ordered collection describing each level of the scale."""

    def _build_objective_lines(self, data: _PromptData) -> list[str]:
        """Return bullet points describing the evaluation focus."""

        lines = [
            "Analiza el fragmento siguiendo la metodología oficial del CEPLAN.",
            "Determina el grado de cumplimiento del criterio indicado y susténtalo con evidencia textual.",
        ]
        if data.question_label:
            lines.append("Responde explícitamente a la pregunta orientadora al justificar el puntaje.")
        return lines

    def _response_guidance(self) -> str:
        """Clarify how the model must select the score values."""

        return "Utiliza únicamente los valores definidos en la escala y mantén el tono técnico-profesional."

    # ------------------------------------------------------------------
    # Helpers for subclasses
    def _normalise_context(self, context: PromptContext) -> _PromptData:
        chunk_text, truncated, extra_instructions = self._prepare_chunk_payload(context)
        metadata = self._prepare_chunk_metadata(context)
        return self._assemble_prompt_data(context, chunk_text, truncated, metadata, extra_instructions)
    
    def _prepare_chunk_payload(self, context: PromptContext) -> tuple[str, bool, Optional[str]]:
        chunk_text, truncated = self._clean_chunk_text(context.chunk_text)
        extra = self._clean_str(context.extra_instructions)
        return chunk_text, truncated, extra or None

    def _prepare_chunk_metadata(self, context: PromptContext) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        if context.chunk_metadata:
            for key, value in context.chunk_metadata.items():
                metadata[str(key)] = value
        return metadata

    def _assemble_prompt_data(
        self,
        context: PromptContext,
        chunk_text: str,
        truncated: bool,
        metadata: Mapping[str, Any],
        extra_instructions: Optional[str],
    ) -> _PromptData:

        criteria = context.criteria
        section = context.section
        dimension = context.dimension or {}
        question = context.question or {}
        document_title = self._first_non_empty(
            context.document.metadata.get("title"),
            context.document.metadata.get("nombre"),
            context.document.metadata.get("titulo"),
            criteria.get("titulo"),
            criteria.get("nombre"),
            "Documento evaluado",
        )
        report_type = self._clean_str(criteria.get("tipo_informe")) or "No especificado"
        methodology = self._clean_str(criteria.get("metodologia") or criteria.get("tipo_metodologia"))
        methodology_version = self._clean_str(criteria.get("version") or criteria.get("criteria_source"))
        criteria_name = self._clean_str(criteria.get("descripcion") or criteria.get("objetivo_general"))
        section_title = self._first_non_empty(
            section.get("titulo"),
            section.get("nombre"),
            section.get("descripcion"),
            "Sección",
        )
        dimension_label = None
        if dimension:
            dimension_label = self._first_non_empty(
                dimension.get("nombre"),
                dimension.get("titulo"),
                dimension.get("descripcion"),
                "Dimensión",
            )
        question_label = None
        if question:
            question_label = self._first_non_empty(
                question.get("texto"),
                question.get("pregunta"),
                question.get("descripcion"),
                question.get("id"),
            )

        return _PromptData(
            document_title=document_title,
            report_type=report_type,
            methodology=methodology or None,
            methodology_version=methodology_version or None,
            criteria_name=criteria_name or None,
            section_title=section_title,
            dimension_label=dimension_label,
            question_label=question_label,
            chunk_text=chunk_text,
            chunk_metadata=metadata,
            extra_instructions=extra_instructions,
            was_truncated=truncated,
        )

    def _compose_prompt(self, data: _PromptData) -> str:
        header_lines = [
            self._header_intro(data),
            f"Documento evaluado: {data.document_title}",
            f"Tipo de informe: {data.report_type}",
        ]
        if data.methodology:
            header_lines.append(f"Metodología aplicada: {data.methodology}")
        if data.methodology_version:
            header_lines.append(f"Versión de criterios: {data.methodology_version}")
        if data.criteria_name:
            header_lines.append(f"Síntesis del criterio metodológico: {data.criteria_name}")
        header_lines.append(f"Sección evaluada: «{data.section_title}».")
        if data.dimension_label:
            header_lines.append(f"Criterio/dimensión: «{data.dimension_label}».")
        if data.question_label:
            header_lines.append(f"Pregunta orientadora: {data.question_label}")
        if data.chunk_metadata:
            metadata_repr = json.dumps(data.chunk_metadata, ensure_ascii=False, sort_keys=True)
            header_lines.append(f"Metadatos del fragmento: {metadata_repr}")
        if data.was_truncated:
            header_lines.append(
                f"Nota: el fragmento se truncó a {self.max_chunk_chars} caracteres para optimizar el rendimiento del modelo."
            )

        objective_lines = self._build_objective_lines(data)
        objective_block = "Objetivo de la evaluación:\n" + "\n".join(f"- {line}" for line in objective_lines)

        scale_block = self._render_scale_block()
        fragment_block = self._render_fragment_block(data)
        response_block = self._render_response_block()

        parts = [
            "\n".join(header_lines),
            objective_block,
            scale_block,
            fragment_block,
            response_block,
        ]
        if data.extra_instructions:
            parts.append(data.extra_instructions)
        return "\n\n".join(part for part in parts if part)
    
    def _header_intro(self, data: _PromptData) -> str:
        return "Rol asignado: Especialista en seguimiento y evaluación del CEPLAN."

    def _build_quality_metadata(self, data: _PromptData, prompt: str) -> Dict[str, Any]:
        return {
            "prompt_length": len(prompt),
            "chunk_length": len(data.chunk_text),
            "was_truncated": data.was_truncated,
            "report_type": data.report_type,
            "scale_label": self.scale_label,
            "has_extra_instructions": bool(data.extra_instructions),
        }

    @property
    def last_quality_metadata(self) -> Optional[Mapping[str, Any]]:
        """Expose the metrics computed for the most recent prompt build."""

        return self._last_quality_metadata

    def _render_scale_block(self) -> str:
        lines = [f"Escala de valoración ({self.scale_label}):"]
        for level in self.scale_levels:
            value_text = self._format_scale_value(level.value)
            if level.label:
                lines.append(f"- {value_text}: {level.label}. {level.description}")
            else:
                lines.append(f"- {value_text}: {level.description}")
        lines.append("Selecciona el puntaje que represente con mayor fidelidad la evidencia encontrada.")
        return "\n".join(lines)

    def _render_fragment_block(self, data: _PromptData) -> str:
        lines = [
            "Texto a analizar (delimitado por los marcadores inferiores):",
            self.fragment_start_marker,
            data.chunk_text,
            self.fragment_end_marker,
        ]
        return "\n".join(lines)
    
    def _response_json_format(self) -> str:
        return (
            '{"score": <numero>, "justification": "<justificación técnica concisa>", '
            '"relevant_text": "<cita opcional>"}'
        )

    def _render_response_block(self) -> str:
        lines = [
            "Instrucciones de respuesta:",
            "1. Fundamenta el puntaje con argumentos técnicos y evidencia específica del fragmento.",
            "2. Mantén objetividad; si la evidencia es insuficiente, indícalo claramente.",
            "3. No inventes información que no esté presente en el texto analizado.",
            self._response_guidance(),
            "Formato de salida (JSON válido, sin texto adicional):",
            self._response_json_format(),
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Utility helpers
    def _clean_chunk_text(self, text: Any) -> tuple[str, bool]:
        cleaned = self._clean_str(text)
        if not cleaned:
            raise ValueError("'chunk_text' debe contener información para evaluar.")
        truncated = False
        if len(cleaned) > self.max_chunk_chars:
            limit = self.max_chunk_chars - len(self.truncation_suffix)
            cleaned = cleaned[:limit].rstrip() + self.truncation_suffix
            truncated = True
        return cleaned, truncated

    @staticmethod
    def _clean_str(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        return str(value).strip()

    @staticmethod
    def _first_non_empty(*values: Any) -> str:
        for value in values:
            text = BasePromptBuilder._clean_str(value)
            if text:
                return text
        return ""

    @staticmethod
    def _format_scale_value(value: float) -> str:
        if float(value).is_integer():
            return str(int(value))
        return ("{:.1f}".format(value)).rstrip("0").rstrip(".")
    
    def _compact_prompt(self, prompt: str) -> str:
        """Collapse repeated whitespace and trim surrounding blank lines."""

        collapsed = "\n".join(line.rstrip() for line in prompt.splitlines())
        collapsed = re.sub(r"\n{3,}", "\n\n", collapsed)
        collapsed = re.sub(r"[ \t]{2,}", " ", collapsed)
        return collapsed.strip()

    def _validate_language(self, text: str) -> None:
        sample = f" {text.lower()} "
        if any(marker in sample for marker in self._spanish_markers):
            return
        self.logger.warning("language_validation_failed", extra={"event": "language_validation_failed"})
        raise ValueError("El fragmento debe estar redactado en español para su evaluación.")

    def to_langchain_prompt(self, context: PromptContext) -> Dict[str, Any]:
        """Return a LangChain-compatible prompt payload."""

        prompt = self.build_from_context(context)
        return {
            "type": "prompt",
            "template": prompt,
            "metadata": dict(self.last_quality_metadata or {}),
        }

    @staticmethod
    def _build_scale_from_config(scale_config: Iterable[Mapping[str, Any]]) -> tuple[ScaleLevel, ...]:
        levels: list[ScaleLevel] = []
        for entry in scale_config:
            value = float(entry["value"])
            description = BasePromptBuilder._clean_str(entry.get("description"))
            label = BasePromptBuilder._clean_str(entry.get("label")) or None
            levels.append(
                ScaleLevel(
                    value=value,
                    description=description,
                    label=label,
                )
            )
        return tuple(levels)

    def _resolve_scale_config(
        self,
        scale_config: Optional[Sequence[Mapping[str, Any]] | str],
        *,
        default: Sequence[ScaleLevel],
    ) -> tuple[ScaleLevel, ...]:
        if not scale_config:
            return tuple(default)
        parsed_config: Iterable[Mapping[str, Any]]
        if isinstance(scale_config, str):
            parsed_config = json.loads(scale_config)
        else:
            parsed_config = scale_config
        return self._build_scale_from_config(parsed_config)


class InstitutionalPromptBuilder(BasePromptBuilder):
    """Prompt builder for institutional reports using the 0–4 CEPLAN scale."""

    _default_scale_levels = (
        ScaleLevel(0, "La sección no aborda el criterio o presenta información contradictoria.", "No cumple"),
        ScaleLevel(1, "Menciones muy generales; evidencia insuficiente o poco pertinente.", "Cumple muy poco"),
        ScaleLevel(2, "Cubre parcialmente el criterio con vacíos relevantes o poca claridad.", "Cumple parcialmente"),
        ScaleLevel(3, "Desarrolla la mayoría de aspectos con claridad aceptable y evidencia pertinente.", "Cumple bastante"),
        ScaleLevel(4, "Responde integralmente al criterio con evidencia sólida, clara y pertinente.", "Cumple totalmente"),
    )

    def __init__(
        self,
        *,
        max_chunk_chars: int = BasePromptBuilder.default_max_chunk_chars,
        scale_config: Optional[Sequence[Mapping[str, Any]] | str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(max_chunk_chars=max_chunk_chars, logger=logger)
        self._scale_levels = self._resolve_scale_config(scale_config, default=self._default_scale_levels)

    @property
    def scale_label(self) -> str:  # pragma: no cover - simple property
        return "0 a 4 (valores enteros)"

    @property
    def scale_levels(self) -> Sequence[ScaleLevel]:  # pragma: no cover - simple property
        return self._scale_levels

    def _build_objective_lines(self, data: _PromptData) -> list[str]:
        lines = super()._build_objective_lines(data)
        lines.append(
            "Evalúa la sección considerando los criterios de estructura, claridad y pertinencia establecidos por CEPLAN."
        )
        return lines
    
    def _header_intro(self, data: _PromptData) -> str:
        return (
            "Evaluación institucional CEPLAN: especialista analiza el fragmento y califica con la escala 0-4."
        )

class PolicyPromptBuilder(BasePromptBuilder):
    """Prompt builder for national policies using the 0–2 scale with half points."""

    _default_scale_levels = (
        ScaleLevel(0.0, "La respuesta es inexistente o contradice el criterio evaluado.", "No evidencia"),
        ScaleLevel(0.5, "Existe referencia mínima sin desarrollo o sin relación directa con el criterio.", "Evidencia incipiente"),
        ScaleLevel(1.0, "Aborda parcialmente el criterio con evidencias limitadas o poco articuladas.", "Cumplimiento parcial"),
        ScaleLevel(1.5, "Cubre la mayor parte del criterio con argumentación razonable pero perfectible.", "Cumplimiento alto"),
        ScaleLevel(2.0, "Responde completamente el criterio con evidencia clara, articulada y pertinente.", "Cumplimiento pleno"),
    )

    def __init__(
        self,
        *,
        max_chunk_chars: int = BasePromptBuilder.default_max_chunk_chars,
        scale_config: Optional[Sequence[Mapping[str, Any]] | str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(max_chunk_chars=max_chunk_chars, logger=logger)
        self._scale_levels = self._resolve_scale_config(scale_config, default=self._default_scale_levels)

    @property
    def scale_label(self) -> str:  # pragma: no cover - simple property
        return "0 a 2 (pasos de 0.5)"

    @property
    def scale_levels(self) -> Sequence[ScaleLevel]:  # pragma: no cover - simple property
        return self._scale_levels

    def _build_objective_lines(self, data: _PromptData) -> list[str]:
        lines = super()._build_objective_lines(data)
        lines.append(
            "Asegura la coherencia con la Política Nacional correspondiente y examina la consistencia del argumento."
        )
        return lines

    def _response_guidance(self) -> str:
        return "Selecciona puntajes en incrementos de 0.5 dentro del rango 0 a 2, según la evidencia disponible."
    
    def __init__(
        self,
        *,
        max_chunk_chars: int = BasePromptBuilder.default_max_chunk_chars,
        scale_config: Optional[Sequence[Mapping[str, Any]] | str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(max_chunk_chars=max_chunk_chars, logger=logger)
        self._scale_levels = self._resolve_scale_config(scale_config, default=self._default_scale_levels)

class PromptFactory:
    """Factory that selects the appropriate builder based on report type."""

    def __init__(self, *, default_builder: Optional[BasePromptBuilder] = None) -> None:
        self._default_builder = default_builder or InstitutionalPromptBuilder()
        self._builders: Dict[str, BasePromptBuilder] = {}

    def register(self, key: str, builder: BasePromptBuilder) -> None:
        if not key:
            raise ValueError("La clave de registro no puede estar vacía.")
        self._builders[key.strip().lower()] = builder

    def for_type(self, report_type: Optional[str]) -> BasePromptBuilder:
        if not report_type:
            return self._default_builder
        normalized = report_type.strip().lower()
        if not normalized:
            return self._default_builder
        for key, builder in self._builders.items():
            if key in normalized:
                return builder
        return self._builders.get(normalized, self._default_builder)

    def for_criteria(self, criteria: Mapping[str, Any]) -> BasePromptBuilder:
        report_type = None
        if isinstance(criteria, MappingABC):
            raw_type = criteria.get("tipo_informe")
            if raw_type is not None:
                report_type = str(raw_type)
        return self.for_type(report_type)

    def default(self) -> BasePromptBuilder:
        """Return the default builder used when no match is found."""

        return self._default_builder


_DEFAULT_FACTORY = PromptFactory()
_DEFAULT_FACTORY.register("institucional", InstitutionalPromptBuilder())
_DEFAULT_FACTORY.register("politica", PolicyPromptBuilder())
_DEFAULT_FACTORY.register("politica_nacional", PolicyPromptBuilder())
_DEFAULT_FACTORY.register("pn", PolicyPromptBuilder())


def build_prompt(
    *,
    document: Document,
    criteria: Mapping[str, Any],
    section: Mapping[str, Any],
    dimension: Optional[Mapping[str, Any]],
    question: Optional[Mapping[str, Any]],
    chunk_text: str,
    chunk_metadata: Optional[Mapping[str, Any]],
    extra_instructions: Optional[str] = None,
) -> str:
    """Create a detailed prompt guiding the language model evaluation."""

    builder = _DEFAULT_FACTORY.for_criteria(criteria)
    prompt = builder(
        document=document,
        criteria=criteria,
        section=section,
        dimension=dimension,
        question=question,
        chunk_text=chunk_text,
        chunk_metadata=chunk_metadata,
        extra_instructions=extra_instructions,
    )
    metadata = getattr(builder, "last_quality_metadata", None)
    if metadata:
        quality_block = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
        prompt = f"{prompt}\n\n<!-- prompt_quality: {quality_block} -->"
    return prompt


def build_prompt_from_context(context: PromptContext) -> str:
    """Adapter that generates a prompt starting from a :class:`PromptContext`."""

    builder = _DEFAULT_FACTORY.for_criteria(context.criteria)
    return builder.build_from_context(context)


__all__ = [
    "PromptBuilder",
    "PromptContext",
    "ScaleLevel",
    "BasePromptBuilder",
    "InstitutionalPromptBuilder",
    "PolicyPromptBuilder",
    "PromptFactory",
    "build_prompt",
    "build_prompt_from_context",
]