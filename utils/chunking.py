"""Funciones auxiliares para persistir chunks generados con LangChain.
Opcional, por lo que se mantiene fuera de utils.py"""

from __future__ import annotations

import os
from typing import List

try:  # pragma: no cover - compatibilidad en tiempo de ejecución
    from langchain_core.documents import Document as LCDocument
except ImportError:  # pragma: no cover
    try:
        from langchain.schema import Document as LCDocument  # type: ignore[no-redef]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Las utilidades de chunking requieren LangChain instalado. "
            "Ejecuta 'pip install langchain-core>=0.2.10'."
        ) from exc

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
