"""Carga especializada para documentos de texto plano."""

from __future__ import annotations

import re
from typing import List

from core.logger import log_error, log_info
from data.models.document import Document
from data.preprocessing.metadata import LoaderContext


def split_text_into_pages(text: str, max_chars: int = 4000) -> List[str]:
    """Divide un texto largo en páginas aproximadas cuando no hay delimitadores."""

    if not text:
        return []
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def load(filepath: str) -> Document:
    """Carga archivos ``.txt`` generando un ``Document`` parcial."""

    log_info("📄 Leyendo archivo TXT...")

    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as handle:
            content = handle.read()

        pages = re.split(r"\f|\n=== PAGE \d+ ===", content)
        if len(pages) <= 1:
            pages = split_text_into_pages(content, max_chars=4000)

        pages_with_tags = [f"=== PAGE {index + 1} ===\n{page.strip()}" for index, page in enumerate(pages)]
        full_text_with_pages = "\n\n".join(pages_with_tags)

        context = LoaderContext(
            tables_meta={},
            extra_metadata={"is_ocr": False, "extraction_method": "text"},
        )

        metadata = {"loader_context": context.as_dict()}

        return Document(
            content=full_text_with_pages,
            pages=pages,
            tables=[],
            metadata=metadata,
        )
    except Exception as exc:  # pragma: no cover - log y re-raise
        log_error(f"❌ Error leyendo archivo TXT: {exc}")
        raise