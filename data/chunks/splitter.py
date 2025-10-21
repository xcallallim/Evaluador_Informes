# data/chunks/splitter.py

import importlib
import importlib.util
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type
from core.logger import log_info, log_warn

if TYPE_CHECKING:
    from langchain_core.documents import Document as LCDocumentType
else:
    LCDocumentType = Any  # type: ignore[assignment]


def _load_lc_document() -> Tuple[Optional[Type[Any]], Optional[str]]:
    """Resuelve la clase Document de LangChain sin romper herramientas estáticas."""

    candidates = (
        "langchain_core.documents",
        "langchain.schema",
    )
    for module_name in candidates:
        if importlib.util.find_spec(module_name) is None:
            continue

        module = importlib.import_module(module_name)
        document_cls = getattr(module, "Document", None)
        if document_cls is not None:
            return document_cls, module_name

    return None, None


def _suppress_schema_warnings() -> None:
    """Silencia los avisos deprecados cuando se usa langchain.schema."""

    try:
        deprecation_module = importlib.import_module(
            "langchain_core._api.deprecation"
        )
        LangChainDeprecationWarning = getattr(  # type: ignore[assignment]
            deprecation_module,
            "LangChainDeprecationWarning",
        )
    except (ModuleNotFoundError, AttributeError, ImportError):
        LangChainDeprecationWarning = DeprecationWarning  # type: ignore[assignment]

    warnings.filterwarnings(
        "ignore",
        category=LangChainDeprecationWarning,
        module=r"langchain\.schema",
    )


def _load_text_splitter() -> Type[Any]:
    """Obtiene una implementación de splitter compatible con distintas versiones."""

    splitter_candidates = (
        ("langchain_text_splitters", "CharacterTextSplitter"),
        ("langchain.text_splitter", "CharacterTextSplitter"),
        ("langchain.text_splitter", "RecursiveCharacterTextSplitter"),
    )

    for module_name, attribute in splitter_candidates:
        if importlib.util.find_spec(module_name) is None:
            continue

        module = importlib.import_module(module_name)
        splitter_cls = getattr(module, attribute, None)
        if splitter_cls is not None:
            return splitter_cls

    raise ImportError(
        "❌ No se encontró un text splitter compatible. "
        "Instala LangChain con: pip install langchain-text-splitters"
    )


_LCDocumentClass, _LC_IMPORT_SOURCE = _load_lc_document()
if _LC_IMPORT_SOURCE == "langchain.schema":
    _suppress_schema_warnings()

CharacterTextSplitter = _load_text_splitter()


__all__ = ["Splitter"]

if TYPE_CHECKING:
    from data.models.document import Document as InternalDocument


_HAS_LOGGED_SCHEMA_WARNING = False


class Splitter:
    """Divide cada sección de un documento en fragmentos con solapamiento."""

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 150):
        if _LCDocumentClass is None:
            raise ImportError(
                "Splitter requiere LangChain instalado."
                " Instala 'langchain-core>=0.1.0' o 'langchain>=0.1.0'."
            )

        global _HAS_LOGGED_SCHEMA_WARNING
        if _LC_IMPORT_SOURCE == "langchain.schema" and not _HAS_LOGGED_SCHEMA_WARNING:
            log_warn(
                "⚠️ LangChain en modo compatibilidad (langchain.schema). "
                "Actualiza a langchain-core>=0.1.0 para evitar avisos deprecados."
            )
            _HAS_LOGGED_SCHEMA_WARNING = True

        if chunk_size <= 0:
            raise ValueError("chunk_size debe ser mayor a cero")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap no puede ser negativo")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap debe ser menor que chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )

    def _split_content_map(
        self,
        content_map: Dict[str, str],
        *,
        origin: str,
        base_metadata: Optional[Dict[str, Any]] = None,
    ) -> List["LCDocumentType"]:
        """Divide cada entrada de ``content_map`` en fragmentos LangChain."""
        log_info(
            f"✂️ Iniciando división de {origin}s en chunks con LangChain..."
        )
        all_chunks: List["LCDocumentType"] = []
        shared_metadata = dict(base_metadata or {})

        for content_id, raw_value in content_map.items():
            if raw_value is None:
                continue

            text = raw_value if isinstance(raw_value, str) else str(raw_value)
            text = text.replace("\n\n", "\n").strip()
            if not text:
                continue

            documents = self.splitter.create_documents(
                texts=[text],
                metadatas=[{"source_id": content_id, "source_type": origin}],
            )
            section_total = len(documents)
            log_info(
                f"{origin.title()} '{content_id}' → {section_total} chunks creados."
            )

            for idx, chunk in enumerate(documents, start=1):
                metadata: Dict[str, Any] = {}
                metadata.update(shared_metadata)
                metadata.update(chunk.metadata or {})
                metadata.update(
                    {
                        "id": f"{content_id}_{idx}",
                        "chunk_index": idx,
                        "chunks_in_source": section_total,
                        "length": len(chunk.page_content),
                    }
                )
                chunk.metadata = metadata
                all_chunks.append(chunk)

        log_info(f"✅ División completada. Total chunks: {len(all_chunks)}.")
        return all_chunks

    def split_document(self, document: "InternalDocument") -> "InternalDocument":
        """Genera ``document.chunks`` a partir de ``document.sections``."""

        sections = getattr(document, "sections", None) or {}
        chunks: List["LCDocumentType"] = []
        document_metadata = getattr(document, "metadata", {}) or {}
        base_chunk_metadata: Dict[str, Any] = {
            "document_metadata": dict(document_metadata),
        }
        if document_metadata.get("id"):
            base_chunk_metadata["document_id"] = document_metadata["id"]

        if sections:
            chunks = self._split_content_map(
                sections,
                origin="section",
                base_metadata=base_chunk_metadata,
            )
        else:
            log_warn(
                "⚠️ Documento no contiene secciones. Intentando generar chunks con fallback."
            )

        if not chunks:
            pages = getattr(document, "pages", None) or []
            page_sections: Dict[str, str] = {}
            for idx, page in enumerate(pages, start=1):
                if page is None:
                    continue

                page_text = page if isinstance(page, str) else str(page)
                page_text = page_text.strip()
                if not page_text:
                    continue

                page_sections[f"page_{idx}"] = page_text
            if page_sections:
                log_warn(
                    "⚠️ No se generaron chunks por secciones. Usando páginas como fallback."
                )
                chunks = self._split_content_map(
                    page_sections,
                    origin="page",
                    base_metadata=base_chunk_metadata,
                )

        if not chunks:
            content = getattr(document, "content", "")
            if content is not None:
                content_text = content if isinstance(content, str) else str(content)
                content_text = content_text.strip()
            else:
                content_text = ""

            if content_text:
                log_warn(
                    "⚠️ No se generaron chunks por secciones ni páginas. Dividiendo el contenido completo."
                )
                chunks = self._split_content_map(
                    {"document": content_text},
                    origin="document",
                    base_metadata=base_chunk_metadata,
                )

        if not chunks:
            log_warn("⚠️ No fue posible generar chunks para el documento.")

        document.chunks = chunks
        return document