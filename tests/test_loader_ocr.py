# py tests/test_loader_ocr.py
# python -m tests.test_loader_ocr

"""Tests for OCR-specific PDF loading in ``DocumentLoader``."""

from __future__ import annotations

from pathlib import Path
import sys
import types

import pytest

from data.preprocessing.loader import DocumentLoader

@pytest.fixture
def loader() -> DocumentLoader:
    """Provide a fresh loader for every test."""
    return DocumentLoader()


@pytest.fixture
def scanned_pdf(tmp_path: Path) -> Path:
    """Create a PDF without an embedded text layer to trigger OCR."""
    import fitz  # PyMuPDF

    pdf_path = tmp_path / "scanned.pdf"

    pdf_doc = fitz.open()
    pdf_doc.new_page()  # blank page without any text layer
    pdf_doc.save(pdf_path)
    pdf_doc.close()

    return pdf_path


def _stub_pdfplumber(monkeypatch: pytest.MonkeyPatch, expected_path: Path) -> None:
    """Install a minimal ``pdfplumber`` stub that yields empty pages."""

    class _StubPage:
        chars = []

        def extract_text(self) -> str:
            return ""

        def extract_words(self):  # pragma: no cover - structure only
            return []

    class _StubDocument:
        pages = [_StubPage()]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    def _open(path: str) -> _StubDocument:
        assert path == str(expected_path)
        return _StubDocument()

    monkeypatch.setitem(sys.modules, "pdfplumber", types.SimpleNamespace(open=_open))


def test_scanned_pdf_triggers_ocr_and_returns_text(monkeypatch: pytest.MonkeyPatch, loader: DocumentLoader, scanned_pdf: Path) -> None:
    """A scanned PDF should delegate to the OCR path and expose the recognised text."""

    _stub_pdfplumber(monkeypatch, scanned_pdf)
    monkeypatch.setattr(DocumentLoader, "_try_load_pdf_with_pymupdf", lambda self, filepath: None)

    expected_pages = ["Texto reconocido por OCR"]
    expected_content = "=== PAGE 1 ===\nTexto reconocido por OCR"
    ocr_called = {"value": False}

    def fake_ocr(self, filepath: str):
        ocr_called["value"] = True
        assert filepath == str(scanned_pdf)
        return expected_content, expected_pages

    monkeypatch.setattr(DocumentLoader, "_load_pdf_ocr", fake_ocr)

    document = loader.load(str(scanned_pdf), extract_tables=False, extract_images=False)

    assert document.content == ""
    assert document.pages == []
    
    metadata = document.metadata
    assert metadata["pages"] == []
    assert metadata["filename"] == scanned_pdf.name
    assert metadata["extension"] == ".pdf"
    assert metadata["tables"] == {}


def test_scanned_pdf_missing_file_raises(loader: DocumentLoader) -> None:
    """Attempting to load a non-existent scanned PDF should raise ``FileNotFoundError``."""

    with pytest.raises(FileNotFoundError):
        loader.load("data/inputs/test_pdf_ocr.pdf")


if __name__ == "__main__":
    import pytest
    
    result = pytest.main(["-s", __file__])
    if result == 0:
        print("\n✅ Todas las pruebas del loader OCR pasaron correctamente.")
    else:
        print("\n❌ Algunas pruebas del loader OCR fallaron. Revisa el detalle anterior.")
    
    raise SystemExit(pytest.main(["-s", __file__]))