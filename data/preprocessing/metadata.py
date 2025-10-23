"""Utilidades compartidas para componer metadatos y logging de documentos."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

from core.logger import log_info, log_warn
from data.models.document import Document


@dataclass
class LoaderContext:
    """Información parcial producida por los loaders especializados."""

    tables_meta: Dict[str, Any] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    images_meta: List[Any] = field(default_factory=list)
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tables_meta": dict(self.tables_meta),
            "issues": list(dict.fromkeys(self.issues)),
            "images_meta": list(self.images_meta),
            "extra_metadata": dict(self.extra_metadata),
        "extra_metadata": dict(self.extra_metadata),
        }


@dataclass(frozen=True)
class DocumentSummary:
    """Resumen de la carga empleado por capas superiores (logging, métricas, etc.)."""

    pages: List[str] = field(default_factory=list)
    tables: Dict[str, Any] = field(default_factory=dict)
    images: Optional[List[Any]] = None
    issues: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class DocumentComposition:
    """Empaqueta el documento final junto al resumen generado durante la carga."""

    document: Document
    summary: DocumentSummary


def extract_loader_context(metadata: MutableMapping[str, Any]) -> LoaderContext:
    """Extrae y elimina el contexto parcial almacenado en metadata."""

    raw_context: Any = {}
    if isinstance(metadata, Mapping):
        raw_context = metadata.pop("loader_context", {})

    if not isinstance(raw_context, Mapping):
        return LoaderContext()

    tables_meta = raw_context.get("tables_meta")
    if not isinstance(tables_meta, Mapping):
        tables_meta = {}

    issues = raw_context.get("issues")
    if not isinstance(issues, Iterable) or isinstance(issues, (str, bytes)):
        issues_list: List[str] = []
    else:
        issues_list = [str(item) for item in issues if item]

    images_meta = raw_context.get("images_meta")
    if not isinstance(images_meta, list):
        images_meta_list: List[Any] = []
    else:
        images_meta_list = images_meta

    extra_metadata = raw_context.get("extra_metadata")
    if not isinstance(extra_metadata, Mapping):
        extra_metadata = {}

    return LoaderContext(
        tables_meta=dict(tables_meta),
        issues=issues_list,
        images_meta=images_meta_list,
        extra_metadata=dict(extra_metadata),
    )


def prepare_metadata(
    filepath: str,
    extension: str,
    pages: Iterable[str],
    tables_meta: Mapping[str, Any],
    *,
    processed_with: str = "DocumentLoader",
    extra_metadata: Optional[Mapping[str, Any]] = None,
    images_meta: Optional[List[Any]] = None,
    issues: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Genera el diccionario de metadatos completo para el ``Document`` final."""

    filename = os.path.basename(filepath)
    metadata: Dict[str, Any] = {
        "filename": filename,
        "extension": extension,
        "source": filepath,
        "pages": list(pages) if pages else [],
        "tables": dict(tables_meta) if tables_meta else {},
        "processed_with": processed_with,
    }

    if extra_metadata:
        metadata.update(dict(extra_metadata))

    if images_meta is not None:
        metadata["images"] = list(images_meta)

    if issues:
        unique_issues: List[str] = []
        for issue in issues:
            if not issue:
                continue
            if issue not in unique_issues:
                unique_issues.append(str(issue))
        if unique_issues:
            metadata["issues"] = unique_issues

    return metadata


def log_document_summary(summary: DocumentSummary) -> None:
    """Centraliza el logging resumen tras la carga de un documento."""

    pages_list = list(summary.pages) if summary.pages else []
    tables_total = 0
    for _, table_group in (summary.tables or {}).items():
        try:
            tables_total += len(table_group)
        except TypeError:
            continue

    images_meta = summary.images or []
    images_count = len(images_meta)

    log_info("📊 Resumen de carga del documento:")
    log_info(f"   • Páginas detectadas: {len(pages_list)}")
    log_info(f"   • Tablas extraídas: {tables_total}")
    log_info(f"   • Imágenes extraídas: {images_count}")

    issues_list = [issue for issue in summary.issues if issue]
    if issues_list:
        log_warn("⚠ Documento cargado con advertencias. Revisar detalles:")
        for issue in issues_list:
            log_warn(f"   • {issue}")
    else:
        log_info("✅ Documento cargado correctamente ✅")


def compose_document(
    partial: Document,
    *,
    filepath: str,
    extension: str,
    include_images: bool,
) -> DocumentComposition:
    """Genera el documento final y su resumen reutilizable."""

    raw_metadata = dict(partial.metadata)
    context = extract_loader_context(raw_metadata)

    final_metadata = prepare_metadata(
        filepath=filepath,
        extension=extension,
        pages=partial.pages,
        tables_meta=context.tables_meta,
        extra_metadata=context.extra_metadata,
        images_meta=context.images_meta if include_images else None,
        issues=context.issues,
    )

    for key, value in raw_metadata.items():
        final_metadata.setdefault(key, value)

    images = context.images_meta if include_images else partial.images

    document = Document(
        content=partial.content,
        metadata=final_metadata,
        pages=partial.pages,
        tables=partial.tables,
        images=images,
        sections=partial.sections,
        chunks=partial.chunks,
    )

    summary = DocumentSummary(
        pages=list(partial.pages),
        tables=dict(final_metadata.get("tables", {})),
        images=list(context.images_meta) if include_images else None,
        issues=list(context.issues),
    )

    return DocumentComposition(document=document, summary=summary)