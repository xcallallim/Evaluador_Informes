"""Tests ensuring the loader handles scanned PDFs via OCR."""

from __future__ import annotations

from pathlib import Path

import pytest

from data.preprocessing.loader import DocumentLoader


@pytest.fixture
def loader() -> DocumentLoader:
    """Return a fresh loader instance for every test."""
    return DocumentLoader()


def test_scanned_pdf_without_text_uses_ocr(monkeypatch: pytest.MonkeyPatch, loader: DocumentLoader) -> None:
    """When a PDF lacks an embedded text layer the loader must delegate to OCR."""

    pdf_path = Path("data/inputs/test_pdf_ocr.pdf")
    assert pdf_path.exists(), "The OCR sample PDF must be present for this test."

    # Force the detection path to mark the file as scanned so OCR is executed.
    monkeypatch.setattr(DocumentLoader, "_try_load_pdf_with_pymupdf", lambda self, filepath: (False, []))

    expected_pages = [
        "Texto reconocido por OCR en la página 1",
        "Texto reconocido por OCR en la página 2",
        "Texto reconocido por OCR en la página 3",
        "Texto reconocido por OCR en la página 4",
    ]
    expected_full_text = "\n\n".join(
        f"=== PAGE {index + 1} ===\n{page}" for index, page in enumerate(expected_pages)
    )
    ocr_called = {"value": False}

    def fake_ocr(self, filepath: str):
        ocr_called["value"] = True
        assert filepath == str(pdf_path)
        return expected_full_text, expected_pages

    monkeypatch.setattr(DocumentLoader, "_load_pdf_ocr", fake_ocr)

    document = loader.load(str(pdf_path), extract_tables=False, extract_images=False)

    assert ocr_called["value"], "OCR should be triggered for scanned PDFs"

    # The loader should not populate ``content`` with OCR pages to keep the pipeline clean.
    assert document.content == ""

    # OCR results must instead be exposed through pages and metadata for later stages.
    assert document.pages == expected_pages
    assert document.metadata["pages"] == expected_pages
    assert document.metadata["is_ocr"] is True
    assert document.metadata["raw_text"] == expected_full_text
    assert document.metadata["filename"] == pdf_path.name
    assert document.metadata["extension"] == ".pdf"
    assert document.metadata["tables"] == {}


def test_loader_raises_file_not_found_for_missing_pdf(loader: DocumentLoader) -> None:
    """If the target PDF does not exist the loader must abort with FileNotFoundError."""

    missing_path = Path("data/inputs/test_pdf_ocr.pdf").with_suffix(".missing")
    assert not missing_path.exists(), "The missing-path sentinel must not accidentally exist."

    with pytest.raises(FileNotFoundError):
        loader.load(str(missing_path))


if __name__ == "__main__":  # pragma: no cover - ayuda interactiva
    import pytest as _pytest

    result = _pytest.main(["-s", __file__])
    if result == 0:
        print("\n✅ Todas las pruebas del loader OCR pasaron correctamente.")
    else:
        print("\n❌ Algunas pruebas del loader OCR fallaron. Revisa el detalle anterior.")

# py tests/test_loader_ocr.py
# python -m tests.test_loader_ocr