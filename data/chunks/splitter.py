# data/chunks/splitter.py

from typing import Any, Dict, List, Optional, TYPE_CHECKING
from core.logger import log_info, log_warn

# ✅ Compatibilidad con distintas versiones de LangChain
try:
    from langchain_core.documents import Document as LCDocument  # langchain 0.2.x
except ImportError:
    try:
        from langchain.schema import Document as LCDocument  # langchain 0.1.x
    except ImportError:
        LCDocument = None  # LangChain no instalado todavía

# ✅ Splitter compatible con versiones nuevas y antiguas
try:
    from langchain.text_splitter import CharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        raise ImportError(
            "❌ No se encontró RecursiveCharacterTextSplitter. "
            "Instala LangChain con: pip install langchain-text-splitters"
        )


__all__ = ["Splitter"]

if TYPE_CHECKING:
    from data.models.document import Document as InternalDocument


class Splitter:
    """Divide cada sección de un documento en fragmentos con solapamiento."""

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 150):
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
    ) -> List[LCDocument]:
        """Divide cada entrada de ``content_map`` en fragmentos LangChain."""
        text = text.replace("\n\n", "\n").strip()
        log_info(
            f"✂️ Iniciando división de {origin}s en chunks con LangChain..."
        )
        all_chunks: List[LCDocument] = []
        shared_metadata = dict(base_metadata or {})

        for content_id, text in content_map.items():
            if not text or not text.strip():
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
        chunks: List[LCDocument] = []
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
            page_sections = {
                f"page_{idx}": page
                for idx, page in enumerate(pages, start=1)
                if isinstance(page, str) and page.strip()
            }
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
            if isinstance(content, str) and content.strip():
                log_warn(
                    "⚠️ No se generaron chunks por secciones ni páginas. Dividiendo el contenido completo."
                )
                chunks = self._split_content_map(
                    {"document": content},
                    origin="document",
                    base_metadata=base_chunk_metadata,
                )

        if not chunks:
            log_warn("⚠️ No fue posible generar chunks para el documento.")

        document.chunks = chunks
        return document