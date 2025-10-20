# data/chunks/splitter.py

import os
import math
import json
from typing import Dict, List, TYPE_CHECKING
from core.logger import log_info, log_warn
from langchain.schema import Document as LCDocument
from langchain.text_splitter import CharacterTextSplitter

if TYPE_CHECKING:
    from data.models.document import Document as InternalDocument


def split_text(
    cleaned_text: str, chunk_size: int = 1000, chunk_overlap: int = 150
) -> List[LCDocument]:
    """Divide texto plano en fragmentos utilizando ``CharacterTextSplitter``."""
    if not cleaned_text.strip():
        print("⚠️ No hay texto para dividir.")
        return []

    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunks = splitter.create_documents([cleaned_text])

    print(f"✅ Texto dividido en {len(chunks)} fragmentos.")
    return chunks

def save_chunks(
    chunks: List[LCDocument], output_folder: str, base_name: str = "chunks"
) -> None:
    """Guarda cada fragmento en un archivo ``.txt`` numerado secuencialmente."""
    os.makedirs(output_folder, exist_ok=True)
    for i, doc in enumerate(chunks, start=1):
        file_path = os.path.join(output_folder, f"{base_name}_{i:03}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(doc.page_content.strip())

    print(f"💾 Se guardaron {len(chunks)} fragmentos en '{output_folder}'.")

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

    def split_sections(self, sections: Dict[str, str]) -> List[LCDocument]:
        """Divide cada sección en ``LCDocument`` con metadatos enriquecidos."""

        log_info("✂️ Iniciando división de secciones en chunks con LangChain...")
        all_chunks: List[LCDocument] = []

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
                chunk.metadata.update(
                    {
                        "id": f"{section_id}_{idx}",
                        "section": section_id,
                        "chunk_index": idx,
                        "length": len(chunk.page_content),
                    }
                )
                all_chunks.append(chunk)

        log_info(f"✅ División completada. Total chunks: {len(all_chunks)}.")
        return all_chunks

    def split_document(self, document: "InternalDocument") -> "InternalDocument":
        """Genera ``document.chunks`` a partir de ``document.sections``."""

        sections = getattr(document, "sections", None)
        if not sections:
            log_warn("⚠️ Documento no contiene secciones. No se puede dividir.")
            document.chunks = []
            return document

        document.chunks = self.split_sections(sections)
        return document