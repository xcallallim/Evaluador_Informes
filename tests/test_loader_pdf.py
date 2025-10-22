# py tests/test_loader_pdf.py
# python -m tests.test_loader_pdf

import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from data.preprocessing.loader import DocumentLoader

@pytest.fixture
def loader():
    """Return a fresh instance of the DocumentLoader for each test."""
    return DocumentLoader()


def test_loads_native_pdf_without_errors(loader, capsys):
    """Loading a known digital PDF should succeed and populate metadata."""
    pdf_path = Path("data/inputs/test_pdf.pdf")

    document = loader.load(str(pdf_path), extract_tables=False, extract_images=False)

    assert document.content, "Expected the loader to extract text from the PDF"
    assert "Resumen Ejecutivo" in document.content

    assert document.metadata["extension"] == ".pdf"
    assert document.metadata["filename"] == pdf_path.name
    assert document.metadata["source"] == str(pdf_path)
    assert document.metadata["processed_with"] == "DocumentLoader"

    assert document.metadata["pages"], "Metadata should include page level text"
    assert document.metadata["pages"] == document.pages
    assert len(document.metadata["pages"]) >= 1

    assert document.metadata["tables"] == {}

    captured = capsys.readouterr()
    assert "✅ PDF digital detectado" in captured.out
    assert "✅ Documento cargado correctamente ✅" in captured.out


def test_load_pdf_handles_missing_file(loader, capsys):
    """A missing PDF should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        loader.load("data/inputs/this_file_does_not_exist.pdf")

    captured = capsys.readouterr()
    assert "Archivo no encontrado" in captured.out


def test_load_pdf_handles_empty_document(monkeypatch, tmp_path, loader, capsys):
    """An empty or unusual PDF should not crash the loader."""
    empty_pdf_path = tmp_path / "empty.pdf"

    import fitz  # PyMuPDF

    pdf_doc = fitz.open()
    pdf_doc.new_page()  # create a blank page
    pdf_doc.save(empty_pdf_path)
    pdf_doc.close()

    def fake_ocr(self, filepath):
        assert filepath == str(empty_pdf_path)
        return "", []

    monkeypatch.setattr(DocumentLoader, "_load_pdf_ocr", fake_ocr)

    document = loader.load(str(empty_pdf_path), extract_tables=False, extract_images=False)

    assert document.content == ""
    assert document.pages == []
    assert document.metadata["pages"] == []
    assert document.metadata["filename"] == empty_pdf_path.name

    captured = capsys.readouterr()
    assert "PDF escaneado detectado" in captured.out


if __name__ == "__main__":
    import pytest
    
    result = pytest.main(["-s", __file__])
    if result == 0:
        print("\n✅ Todas las pruebas del loader PDF pasaron correctamente.")
    else:
        print("\n❌ Algunas pruebas del loader PDF fallaron. Revisa el detalle anterior.")
    
    raise SystemExit(pytest.main(["-s", __file__]))