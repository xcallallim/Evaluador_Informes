"""Funciones auxiliares para persistir chunks generados con LangChain."""


from __future__ import annotations

import os
from typing import List

from langchain.schema import Document as LCDocument

__all__ = ["save_chunks"]

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
