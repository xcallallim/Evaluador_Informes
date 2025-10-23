"""Orquesta la carga de documentos delegando en loaders especializados."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Protocol, Tuple, TYPE_CHECKING

from core.logger import log_error, log_info
from data.models.document import Document
from data.preprocessing import docx_loader, pdf_loader, txt_loader
from data.preprocessing.metadata import (
    extract_loader_context,
    log_document_summary,
    prepare_metadata,
)

if TYPE_CHECKING:
    from data.preprocessing.pdf_loader import PDFResourceExporter

class LoaderStrategy(Protocol):
    """Contrato mínimo para los loaders especializados."""

    def __call__(
        self,
        filepath: str,
        *,
        extract_tables: bool,
        extract_images: bool,
    ) -> Document:
        ...

class DocumentLoader:
    """Fachada thread-safe que compone resultados de loaders especializados."""

    def __init__(
        self,
        resource_exporter: Optional["PDFResourceExporter"] = None,
    ) -> None:
        pdf_loader.configure_ocr()
        self._ghostscript_cmd: Optional[str] = pdf_loader.resolve_ghostscript()
        self._resource_exporter: "PDFResourceExporter" = (
            resource_exporter or pdf_loader.FileSystemPDFResourceExporter()
        )
        self._strategies: Dict[str, LoaderStrategy] = {}
        self._register_default_strategies()

    # ------------------------------------------------------------------
    # Registro de estrategias
    # ------------------------------------------------------------------
    def _register_default_strategies(self) -> None:
        self.register_strategy(".txt", self._load_txt)
        self.register_strategy(".docx", self._load_docx)
        self.register_strategy(".pdf", self._load_pdf)

    def register_strategy(self, extension: str, strategy: LoaderStrategy) -> None:
        """Permite registrar loaders adicionales por extensión."""

        if not extension:
            raise ValueError("La extensión no puede estar vacía")
        
        ext = extension.lower()
        if not ext.startswith("."):
            ext = f".{ext}"

        self._strategies[ext] = strategy

    @property
    def supported_extensions(self) -> List[str]:
        """Lista de extensiones soportadas actualmente."""

        return sorted(self._strategies.keys())

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------
    def load(
        self,
        filepath: str,
        extract_tables: bool = True,
        extract_images: bool = True,
    ) -> Document:
        filepath = os.fspath(filepath)

        if not os.path.isfile(filepath):
            log_error(f"Archivo no encontrado: {filepath}")
            raise FileNotFoundError(f"Archivo no encontrado: {filepath}")

        _, ext = os.path.splitext(filepath)
        ext = ext.lower()

        strategy = self._strategies.get(ext)
        if strategy is None:
            log_error(f"Formato no soportado: {ext}")
            raise ValueError(f"Formato no soportado: {ext}")

        log_info(f"Cargando archivo: {filepath}")

        partial = strategy(
            filepath,
            extract_tables=extract_tables,
            extract_images=extract_images,
        )

        return self._finalize_document(
            partial,
            filepath=filepath,
            extension=ext,
            include_images=extract_images,
        )
    
    # ------------------------------------------------------------------
    # Estrategias internas
    # ------------------------------------------------------------------
    def _load_txt(
        self,
        filepath: str,
        *,
        extract_tables: bool,
        extract_images: bool,
    ) -> Document:
        return txt_loader.load(filepath)

    def _load_docx(
        self,
        filepath: str,
        *,
        extract_tables: bool,
        extract_images: bool,
    ) -> Document:
        return docx_loader.load(filepath, extract_tables=extract_tables)

    def _load_pdf(
        self,
        filepath: str,
        *,
        extract_tables: bool,
        extract_images: bool,
    ) -> Document:
        issues: List[str] = []

        def _ocr_loader(path: str) -> Tuple[str, List[str]]:
            try:
                return self._load_pdf_ocr(path, issues)
            except TypeError:
                return self._load_pdf_ocr(path)  # type: ignore[misc]

        return pdf_loader.load(
            filepath,
            extract_tables=extract_tables,
            extract_images=extract_images,
            ghostscript_cmd=self._ghostscript_cmd,
            issues=issues,
            detector=self._try_load_pdf_with_pymupdf,
            ocr_loader=_ocr_loader,
            resource_exporter=self._resource_exporter,
        )

    def _load_pdf_ocr(
        self, filepath: str, issues: Optional[List[str]] = None
    ) -> Tuple[str, List[str]]:
        """Permite personalizar la estrategia OCR manteniendo compatibilidad."""

        issues_list = issues if issues is not None else []
        return pdf_loader._load_pdf_ocr(filepath, issues_list)
    
    def _try_load_pdf_with_pymupdf(
        self, filepath: str
    ) -> Optional[Tuple[bool, List[str]]]:
        """Proxy para facilitar pruebas unitarias y personalización."""

        return pdf_loader._try_load_pdf_with_pymupdf(filepath)

    # ------------------------------------------------------------------
    # Composición final del documento
    # ------------------------------------------------------------------
    def _finalize_document(
        self,
        partial: Document,
        *,
        filepath: str,
        extension: str,
        include_images: bool,
    ) -> Document:
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

        log_document_summary(
            partial.pages,
            final_metadata.get("tables", {}),
            context.images_meta if include_images else None,
            context.issues,
        )

        images = context.images_meta if include_images else partial.images

        return Document(
            content=partial.content,
            metadata=final_metadata,
            pages=partial.pages,
            tables=partial.tables,
            images=images,
            sections=partial.sections,
            chunks=partial.chunks,
        )