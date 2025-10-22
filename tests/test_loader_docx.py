# py tests/test_loader_docx.py
# python -m tests.test_loader_docx

"""Pruebas para cargar documentos DOCX con :class:`DocumentLoader`."""""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.preprocessing.loader import DocumentLoader

@pytest.fixture()
def loader() -> DocumentLoader:
    """Return a fresh ``DocumentLoader`` for each test."""

    return DocumentLoader()


def test_load_docx_extracts_text_tables_and_metadata(loader: DocumentLoader) -> None:
    """The loader should parse DOCX text, tables, and metadata correctly."""

    doc_path = Path("data/inputs/test_docx.docx")

    document = loader.load(str(doc_path))

    assert document.content.startswith("=== PAGE 1 ===")
    assert "Este es un documento Word de prueba." in document.content

    metadata = document.metadata
    assert metadata["filename"] == doc_path.name
    assert metadata["extension"] == ".docx"
    assert metadata["source"] == str(doc_path)
    assert metadata["processed_with"] == "DocumentLoader"

    pages = metadata["pages"]
    assert isinstance(pages, list)
    assert pages and "Tabla:" in pages[0]

    tables = metadata.get("tables", {})
    assert "docx" in tables
    assert len(tables["docx"]) == 1
    assert tables["docx"][0][0] == ["COLUMNA A", "COLUMNA B"]


def test_load_docx_handles_empty_document(loader: DocumentLoader, tmp_path: Path) -> None:
    """Loading an empty DOCX should succeed and return empty content."""

    empty_docx = tmp_path / "empty.docx"

    # Create an empty Word document on the fly.
    from docx import Document as DocxDocument

    DocxDocument().save(empty_docx)

    document = loader.load(str(empty_docx))

    assert document.content == ""
    assert document.metadata["filename"] == "empty.docx"
    assert document.metadata["tables"] == {}
    assert document.metadata["pages"] == []


def test_load_docx_missing_file_raises(loader: DocumentLoader, tmp_path: Path) -> None:
    """Trying to load a non-existent DOCX should raise ``FileNotFoundError``."""

    missing_path = tmp_path / "missing.docx"

    with pytest.raises(FileNotFoundError):
        loader.load(str(missing_path))

if __name__ == "__main__":  # pragma: no cover - manual invocation helper
    raise SystemExit(pytest.main([__file__]))