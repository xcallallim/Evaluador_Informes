"""Unit tests for the Segmenter using synthetic cleaned documents."""

from __future__ import annotations

import pytest

from data.models.document import Document
from data.preprocessing.segmenter import Segmenter


@pytest.fixture
def segmenter() -> Segmenter:
    """Use the institutional segmentation rules without external files."""

    return Segmenter(tipo="institucional", fuzzy=False)


@pytest.fixture
def cleaned_document() -> Document:
    """Produce a minimal document that mimics the cleaner output."""

    content = "\n".join(
        [
            "=== PAGE 1 ===",
            "1. Resumen Ejecutivo",
            "El informe resume avances.",
            "",
            "2. Prioridades de la política institucional",
            "Las prioridades se enfocan en mejorar la gestión.",
            "",
            "Sección 4: Conclusiones",
            "Se concluye que el plan fue efectivo.",
        ]
    )
    return Document(content=content)


def test_segmenter_populates_known_sections(segmenter: Segmenter, cleaned_document: Document) -> None:
    """Segmenter assigns content to detected sections and leaves the rest empty."""

    segmented = segmenter.segment_document(cleaned_document)

    assert segmented.sections["resumen_ejecutivo"] == "El informe resume avances."
    assert (
        segmented.sections["prioridades_politica_institucional"]
        == "Las prioridades se enfocan en mejorar la gestión."
    )
    assert segmented.sections["conclusiones"] == "Se concluye que el plan fue efectivo."
    assert segmented.sections["anexos"] == ""