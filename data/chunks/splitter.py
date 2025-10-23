"""Divisor de secciones en fragmentos compatibles con LangChain."""

import importlib
import importlib.util
import warnings
from dataclasses import dataclass, field
from threading import Lock
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)
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


__all__ = ["Splitter"]

if TYPE_CHECKING:
    from data.models.document import Document as InternalDocument


@dataclass
class _SplitterState:
    """Thread-safe container for dynamic LangChain resources used by Splitter."""

    _lock: Lock = field(default_factory=Lock)
    _document_class: Optional[Type[Any]] = None
    _import_source: Optional[str] = None
    _splitter_cls: Optional[Type[Any]] = None
    _has_logged_schema_warning: bool = False
    _document_class_loaded: bool = False

    def ensure_document_class(self) -> Tuple[Optional[Type[Any]], Optional[str]]:
        """Resolve and cache the LangChain Document class in a thread-safe way."""

        with self._lock:
            if not self._document_class_loaded:
                document_class, import_source = _load_lc_document()
                self._document_class = document_class
                self._import_source = import_source
                self._document_class_loaded = True
                if import_source == "langchain.schema":
                    _suppress_schema_warnings()
            return self._document_class, self._import_source

    def should_log_schema_warning(self) -> bool:
        """Ensure the schema warning is logged at most once."""

        with self._lock:
            if self._has_logged_schema_warning:
                return False

            self._has_logged_schema_warning = True
            return True

    def get_splitter_cls(self) -> Type[Any]:
        """Resolve and cache the CharacterTextSplitter implementation safely."""

        with self._lock:
            if self._splitter_cls is None:
                self._splitter_cls = _load_text_splitter()
            return self._splitter_cls

    def reset_for_testing(self) -> None:
        """Reset cached state to facilitate deterministic testing."""

        with self._lock:
            self._document_class = None
            self._import_source = None
            self._splitter_cls = None
            self._has_logged_schema_warning = False
            self._document_class_loaded = False


_SPLITTER_STATE = _SplitterState()


def _reset_state_for_tests() -> None:
    """Expose internal reset helper for unit tests."""

    _SPLITTER_STATE.reset_for_testing()


class Splitter:
    """Divide cada sección de un documento en fragmentos con solapamiento.

    Además del comportamiento clásico (materializar ``document.chunks``) la
    clase ofrece :meth:`iter_document_chunks` y el flag ``stream`` en
    :meth:`_split_content_map` para obtener los fragmentos de forma perezosa,
    útil en pipelines con restricciones de memoria.
    """

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 150):
        document_class, import_source = _SPLITTER_STATE.ensure_document_class()
        if document_class is None:
            raise ImportError(
                "Splitter requiere LangChain instalado."
                " Instala 'langchain-core>=0.1.0' o 'langchain>=0.1.0'."
            )

        if (
            import_source == "langchain.schema"
            and _SPLITTER_STATE.should_log_schema_warning()
        ):
            log_warn(
                "⚠️ LangChain en modo compatibilidad (langchain.schema). "
                "Actualiza a langchain-core>=0.1.0 para evitar avisos deprecados."
            )

        if chunk_size <= 0:
            raise ValueError("chunk_size debe ser mayor a cero")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap no puede ser negativo")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap debe ser menor que chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        splitter_cls = _SPLITTER_STATE.get_splitter_cls()
        self.splitter = splitter_cls(
            separator="\n",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            keep_separator=True,
        )

    def _split_content_map(
        self,
        content_map: Dict[str, str],
        *,
        origin: str,
        base_metadata: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> Union[List["LCDocumentType"], Iterator["LCDocumentType"]]:
        """Divide cada entrada de ``content_map`` en fragmentos LangChain.

        Cuando ``stream`` es ``True`` la función devuelve un iterador que genera los
        chunks bajo demanda sin materializar una lista completa. El iterador incluye
        los mismos metadatos enriquecidos que la versión previa, por lo que los
        consumidores pueden alternar entre los dos modos sin cambios adicionales.
        """

        def _chunk_iterator() -> Iterator["LCDocumentType"]:
            log_info(
                f"✂️ Iniciando división de {origin}s en chunks con LangChain..."
            )
            shared_metadata = dict(base_metadata or {})
            total_chunks = 0

            for content_id, raw_value in content_map.items():
                if raw_value is None:
                    continue

                text = raw_value if isinstance(raw_value, str) else str(raw_value)
                text = text.strip()
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
                            "chunk_overlap": self.chunk_overlap,
                        }
                    )
                    chunk.metadata = metadata
                    total_chunks += 1
                    yield chunk

            log_info(f"✅ División completada. Total chunks: {total_chunks}.")

        iterator = _chunk_iterator()
        if stream:
            return iterator
        return list(iterator)

    def iter_document_chunks(
        self,
        document: "InternalDocument",
        *,
        base_metadata: Optional[Dict[str, Any]] = None,
    ) -> Iterator["LCDocumentType"]:
        """Genera los chunks de ``document`` de forma perezosa.

        El iterador aplica la misma lógica de *fallbacks* que
        :meth:`split_document`, intentando primero las secciones, luego las
        páginas y finalmente el contenido completo. Se detiene tan pronto como
        alguno de los niveles produce chunks, replicando el comportamiento
        previo pero sin materializar listas intermedias.
        """

        if base_metadata is None:
            document_metadata = getattr(document, "metadata", {}) or {}
            base_metadata = {
                "document_metadata": dict(document_metadata),
            }
            if document_metadata.get("id"):
                base_metadata["document_id"] = document_metadata["id"]
        
        shared_metadata = dict(base_metadata or {})
        produced_any = False

        sections = getattr(document, "sections", None) or {}
        if sections:
            for chunk in self._split_content_map(
                sections,
                origin="section",
                base_metadata=shared_metadata,
                stream=True,
            ):
                produced_any = True
                yield chunk
        else:
            log_warn(
                "⚠️ Documento no contiene secciones. Intentando generar chunks con fallback."
            )

        if produced_any:
            return

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
            for chunk in self._split_content_map(
                page_sections,
                origin="page",
                base_metadata=shared_metadata,
                stream=True,
            ):
                produced_any = True
                yield chunk

        if produced_any:
            return

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
            for chunk in self._split_content_map(
                {"document": content_text},
                origin="document",
                base_metadata=shared_metadata,
                stream=True,
            ):
                produced_any = True
                yield chunk

        if not produced_any:
            log_warn("⚠️ No fue posible generar chunks para el documento.")

    def split_document(self, document: "InternalDocument") -> "InternalDocument":
        """Genera ``document.chunks`` a partir de ``document.sections``."""

        chunk_iterator = self.iter_document_chunks(document)
        document.chunks = list(chunk_iterator)
        return document