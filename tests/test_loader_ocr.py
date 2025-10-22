# py tests/test_loader_ocr.py
# python -m tests.test_loader_ocr

import pytest

from data.preprocessing.loader import DocumentLoader

loader = DocumentLoader()

def test_digital_pdf_is_loaded_without_ocr(monkeypatch: pytest.MonkeyPatch) -> None:
    """Digital PDFs should bypass OCR and return the embedded text."""

    loader = DocumentLoader()

    def fail_if_called(_path: str):  # pragma: no cover - defensive guard
        raise AssertionError("_load_pdf_ocr should not run for digital PDFs")
    
    monkeypatch.setattr(loader, "_load_pdf_ocr", fail_if_called)

    doc = loader.load("data/inputs/test_pdf_ocr.pdf", extract_tables=False, extract_images=False)

    assert "Resumen Ejecutivo" in doc.content
    assert doc.metadata["extension"] == ".pdf"
    assert doc.metadata["filename"] == "test_pdf_ocr.pdf"
    assert len(doc.metadata["pages"]) >= 1

if __name__ == "__main__":
    import pytest
    
    result = pytest.main(["-s", __file__])
    if result == 0:
        print("\n✅ Todas las pruebas del loader OCR pasaron correctamente.")
    else:
        print("\n❌ Algunas pruebas del loader OCR fallaron. Revisa el detalle anterior.")
    
    raise SystemExit(pytest.main(["-s", __file__]))