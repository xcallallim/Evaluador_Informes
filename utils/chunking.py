"""Funciones auxiliares para manipular chunks generados con LangChain."""

from __future__ import annotations

import os
from typing import List

from langchain.schema import Document as LCDocument
from langchain.text_splitter import CharacterTextSplitter

__all__ = ["split_text", "save_chunks"]


def split_text(
    cleaned_text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
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
    chunks: List[LCDocument],
    output_folder: str,
    base_name: str = "chunks",
) -> None:
    """Guarda cada fragmento en un archivo ``.txt`` numerado secuencialmente."""
    os.makedirs(output_folder, exist_ok=True)
    for i, doc in enumerate(chunks, start=1):
        file_path = os.path.join(output_folder, f"{base_name}_{i:03}.txt")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(doc.page_content.strip())

    print(f"💾 Se guardaron {len(chunks)} fragmentos en '{output_folder}'.")
