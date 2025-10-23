"""Tests for the section loader identification logic."""

from __future__ import annotations

import pytest

from data.criteria.section_loader import SectionLoader

pytest.importorskip("rapidfuzz", reason="Los encabezados con prefijos numéricos requieren fuzzy matching.")


@pytest.fixture
def politica_loader() -> SectionLoader:
    """Load the policy schema used by the original smoke test."""

    return SectionLoader(tipo="politica", fuzzy=True)


@pytest.mark.parametrize(
    ("line", "expected_id"),
    [
        ("1. Resumen Ejecutivo", "resumen_ejecutivo"),
        (
            "Capítulo II - Análisis de los resultados de la política nacional",
            "analisis_resultados",
        ),
        ("SECCIÓN 4: CONCLUSIONES", "conclusiones"),
    ],
)
def test_identify_section_matches_expected(politica_loader: SectionLoader, line: str, expected_id: str) -> None:
    """Lines resembling report headings should map to the configured section ids."""

    result = politica_loader.identify_section(line)
    assert result is not None
    section_id, *_ = result
    assert section_id == expected_id


def test_unknown_heading_returns_none(politica_loader: SectionLoader) -> None:
    """Random lines are ignored by the loader."""

    assert politica_loader.identify_section("Contenido libre sin título") is None