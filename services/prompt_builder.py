"""Utility helpers to compose prompts for the AI services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol

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

    document_title = (
        document.metadata.get("title")
        or document.metadata.get("nombre")
        or criteria.get("titulo")
        or "Documento evaluado"
    )
    section_title = section.get("titulo") or section.get("nombre") or "Sección"
    dimension_name = ""
    if dimension:
        dimension_name = (
            dimension.get("nombre")
            or dimension.get("titulo")
            or dimension.get("descripcion")
            or "Dimensión"
        )
    question_label = ""
    if question:
        question_label = question.get("texto") or question.get("id") or ""

    info_lines = [
        f"Documento: {document_title}",
        f"Tipo de informe: {criteria.get('tipo_informe', 'No especificado')}",
        f"Sección: {section_title}",
    ]
    if dimension_name:
        info_lines.append(f"Dimensión: {dimension_name}")
    if question_label:
        info_lines.append(f"Pregunta: {question_label}")

    chunk_lines = ["Analiza el siguiente fragmento y asigna un puntaje objetivo."]
    metadata_dict = dict(chunk_metadata or {})
    if metadata_dict:
        chunk_lines.append(f"Metadatos del fragmento: {metadata_dict}")
    chunk_lines.append("Fragmento:")
    chunk_lines.append(chunk_text or "<sin contenido>")

    prompt_parts = ["\n".join(info_lines), "\n".join(chunk_lines)]
    if extra_instructions:
        prompt_parts.append(extra_instructions)
    return "\n\n".join(part for part in prompt_parts if part)


def build_prompt_from_context(context: PromptContext) -> str:
    """Adapter that generates a prompt starting from a :class:`PromptContext`."""

    return build_prompt(
        document=context.document,
        criteria=context.criteria,
        section=context.section,
        dimension=context.dimension,
        question=context.question,
        chunk_text=context.chunk_text,
        chunk_metadata=context.chunk_metadata,
        extra_instructions=context.extra_instructions,
    )


__all__ = ["PromptBuilder", "PromptContext", "build_prompt", "build_prompt_from_context"]