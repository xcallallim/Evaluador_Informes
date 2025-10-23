"""Tests for the Cleaner stage using synthetic documents.
These tests replace the manual smoke script that relied on a specific PDF.
They assert the behaviour using a minimal in-memory Document instance.
"""

from __future__ import annotations

import pytest

from data.models.document import Document
from data.preprocessing.cleaner import Cleaner


@pytest.fixture
def noisy_document() -> Document:
    """Return a two-page document with repeated headers and extra noise."""

    content = "\n".join(
        [
            "=== PAGE 1 ===",
            "Gobierno del Perú",
            "Página 1 de 2",
            "1. Resumen Ejecutivo",
            "Fuente: Datos internos",
            "El informe resume avances.",
            "",
            "=== PAGE 2 ===",
            "Gobierno del Perú",
            "Página 2 de 2",
            "2. Conclusiones",
            "Documento generado automáticamente",
            "Las conclusiones finales.",
        ]
    )

    pages = [
        "\n".join(
            [
                "Gobierno del Perú",
                "Página 1 de 2",
                "1. Resumen Ejecutivo",
                "Fuente: Datos internos",
                "El informe resume avances.",
            ]
        ),
        "\n".join(
            [
                "Gobierno del Perú",
                "Página 2 de 2",
                "2. Conclusiones",
                "Documento generado automáticamente",
                "Las conclusiones finales.",
            ]
        ),
    ]

    return Document(
        content=content,
        metadata={"is_ocr": False, "filename": "sample.pdf"},
        pages=pages,
        tables=[],
        images=[],
    )

@pytest.fixture
def cleaner() -> Cleaner:
    """Provide a cleaner configured like the legacy script."""

    return Cleaner(remove_headers=True, remove_page_numbers=True, use_custom_headers=True)


def test_cleaner_removes_repeated_noise(noisy_document: Document, cleaner: Cleaner) -> None:
    """The cleaner removes headers, page numbers, and digital signatures."""

    cleaned_doc, report = cleaner.clean_document(noisy_document, return_report=True)

    expected_content = "\n".join(
        [
            "=== PAGE 1 ===",
            "1. Resumen Ejecutivo",
            "El informe resume avances.",
            "",
            "=== PAGE 2 ===",
            "2. Conclusiones",
            "Las conclusiones finales.",
        ]
    )

    assert cleaned_doc.content == expected_content
    assert report == {
        "headers_removed": 2,
        "footers_removed": 0,
        "page_numbers_removed": 2,
        "source_lines_removed": 1,
        "digital_sign_removed": 1,
        "other_noise_removed": 0,
    }

def test_cleaner_preserves_metadata(noisy_document: Document, cleaner: Cleaner) -> None:
    """Cleaning should not alter metadata, pages, tables, or images."""

    cleaned_doc = cleaner.clean_document(noisy_document, return_report=False)

    assert cleaned_doc.metadata == noisy_document.metadata
    assert cleaned_doc.pages == noisy_document.pages
    assert cleaned_doc.tables == noisy_document.tables
    assert cleaned_doc.images == noisy_document.images