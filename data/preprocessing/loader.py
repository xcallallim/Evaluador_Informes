"""Orquesta la carga de documentos delegando en loaders especializados."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Protocol, Tuple, TYPE_CHECKING

from core.logger import log_error, log_info
from data.models.document import Document
from data.preprocessing import docx_loader, pdf_loader, txt_loader
from data.preprocessing.metadata import (
    DocumentComposition,
    DocumentSummary,
    compose_document,
    log_document_summary,
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

class LoaderEventHandler(Protocol):
    """Eventos de alto nivel disparados durante ``load``."""

    def on_load_started(self, filepath: str) -> None:
        ...

    def on_load_succeeded(
        self,
        filepath: str,
        document: Document,
        summary: DocumentSummary,
    ) -> None:
        ...

    def on_load_failed(self, filepath: str, error: Exception) -> None:
        ...


class NullLoaderEventHandler:
    """Implementación por defecto silenciosa para facilitar pruebas."""

    def on_load_started(self, filepath: str) -> None:  # pragma: no cover - trivial
        return

    def on_load_succeeded(
        self,
        filepath: str,
        document: Document,
        summary: DocumentSummary,
    ) -> None:  # pragma: no cover - trivial
        return

    def on_load_failed(self, filepath: str, error: Exception) -> None:  # pragma: no cover - trivial
        return


class LoggingLoaderEventHandler(NullLoaderEventHandler):
    """Event handler that proxies the legacy logging behaviour."""

    def on_load_started(self, filepath: str) -> None:
        log_info(f"Cargando archivo: {filepath}")

    def on_load_succeeded(
        self,
        filepath: str,
        document: Document,
        summary: DocumentSummary,
    ) -> None:
        log_document_summary(summary)

    def on_load_failed(self, filepath: str, error: Exception) -> None:
        log_error(str(error))


class DocumentLoader:
    """Fachada segura para subprocesos que compone los resultados de los cargadores especializados.
    
    El cargador normaliza los objetos parciales :class:`~data.models.document.Document`
    devueltos por las estrategias especializadas (TXT, DOCX, PDF) para que todos
    cumplan con el contrato de datos descrito en :mod:`data.models.document`.
    Quienes llaman siempre deben interactuar con esta fachada en lugar de instanciar los 
    cargadores directamente para garantizar que los campos de metadatos obligatorios 
    (``source``, ``processed_with``, ``is_ocr``, etc.) estén presentes y sean consistentes."""

    def __init__(
        self,
        resource_exporter: Optional["PDFResourceExporter"] = None,
        events: Optional[LoaderEventHandler] = None,
    ) -> None:
        pdf_loader.configure_ocr()
        self._ghostscript_cmd: Optional[str] = pdf_loader.resolve_ghostscript()
        self._resource_exporter: "PDFResourceExporter" = (
            resource_exporter or pdf_loader.FileSystemPDFResourceExporter()
        )
        self._strategies: Dict[str, LoaderStrategy] = {}
        self._register_default_strategies()
        self._events: LoaderEventHandler = events or LoggingLoaderEventHandler()
        self.failures: int = 0
        self._failure_traces: List[Dict[str, str]] = []

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
    
    @property
    def failure_traces(self) -> List[Dict[str, str]]:
        """Historial inmutable de fallos reportados durante ``load``."""

        return list(self._failure_traces)

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

        self._events.on_load_started(filepath)

        try:
            if not os.path.isfile(filepath):
                raise FileNotFoundError(f"Archivo no encontrado: {filepath}")

            _, ext = os.path.splitext(filepath)
            ext = ext.lower()

            strategy = self._strategies.get(ext)
            if strategy is None:
                raise ValueError(f"Formato no soportado: {ext}")
            
            partial = strategy(
                filepath,
                extract_tables=extract_tables,
                extract_images=extract_images,
            )
            composition: DocumentComposition = compose_document(
                partial,
                filepath=filepath,
                extension=ext,
                include_images=extract_images,
            )
        except Exception as error:
            self._record_failure(filepath, error)
            raise

        self._events.on_load_succeeded(
            filepath,
            composition.document,
            composition.summary,
        )
        return composition.document

    def _record_failure(self, filepath: str, error: Exception) -> None:
        """Incrementa contadores y conserva trazabilidad para el pipeline."""

        self.failures += 1
        self._failure_traces.append(
            {
                "filepath": filepath,
                "error_type": type(error).__name__,
                "message": str(error),
            }
        )
        self._events.on_load_failed(filepath, error)
    
    
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
