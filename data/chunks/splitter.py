# data/chunks/splitter.py

from typing import Any, Dict, List, TYPE_CHECKING
from core.logger import log_info, log_warn
from langchain.text_splitter import CharacterTextSplitter
from utils.chunking import save_chunks, split_text

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

    def split_sections(self, sections: Dict[str, str]) -> List[Dict[str, Any]]:
        """Divide cada sección y devuelve diccionarios ``{text, metadata}``."""

        log_info("✂️ Iniciando división de secciones en chunks con LangChain...")
        all_chunks: List[Dict[str, Any]] = []

        for section_id, text in sections.items():
            if not text or not text.strip():
                continue

            documents = self.splitter.create_documents(
                texts=[text],
                metadatas=[{"section": section_id}],
            )
            log_info(
                f"Sección '{section_id}' → {len(documents)} chunks creados."
            )

            for idx, chunk in enumerate(documents, start=1):
                metadata = dict(chunk.metadata) if chunk.metadata else {}
                metadata.update(
                    {
                        "id": f"{section_id}_{idx}",
                        "section": section_id,
                        "chunk_index": idx,
                        "length": len(chunk.page_content),
                    }
                )
                all_chunks.append(
                    {
                        "text": chunk.page_content,
                        "metadata": metadata,
                    }
                )

        log_info(f"✅ División completada. Total chunks: {len(all_chunks)}.")
        return all_chunks

    def split_document(self, document: "InternalDocument") -> "InternalDocument":
        """Genera ``document.chunks`` a partir de ``document.sections``."""

        sections = getattr(document, "sections", None) or {}
        chunks: List[Dict[str, Any]] = []

        if sections:
            chunks = self.split_sections(sections)
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
                chunks = self.split_sections(page_sections)

        if not chunks:
            content = getattr(document, "content", "")
            if isinstance(content, str) and content.strip():
                log_warn(
                    "⚠️ No se generaron chunks por secciones ni páginas. Dividiendo el contenido completo."
                )
                chunks = self.split_sections({"document": content})

        if not chunks:
            log_warn("⚠️ No fue posible generar chunks para el documento.")

        document.chunks = chunks
        return document