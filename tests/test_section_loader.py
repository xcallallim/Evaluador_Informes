"""Unit tests for :mod:`data.criteria.section_loader`."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional, Tuple

import pytest

from data.criteria import section_loader


@pytest.fixture
def loader_factory(
    tmp_path, monkeypatch
) -> Callable[[Dict[str, Dict[str, Any]], Dict[str, Any]], section_loader.SectionLoader]:
    """Build a :class:`SectionLoader` instance backed by a temporary schema file."""

    def _factory(
        schema: Dict[str, Dict[str, Any]],
        options: Dict[str, Any],
    ) -> section_loader.SectionLoader:
        # Rendimiento esperado: escribir JSONs mínimos en disco mantiene el costo por
        # prueba bien por debajo del milisegundo, incluso en CI, porque los esquemas
        # tienen pocas claves y se reutiliza el mismo directorio temporal.
        file_path = tmp_path / f"secciones_{options.get('tipo', 'demo')}.json"
        file_path.write_text(json.dumps(schema), encoding="utf-8")

        monkeypatch.setattr(section_loader, "CRITERIA_DIR", str(tmp_path))

        enable_fuzz: Optional[bool] = options.pop("_force_fuzz_available", None)
        if enable_fuzz is not None:
            monkeypatch.setattr(section_loader, "_HAS_FUZZ", enable_fuzz)
        fuzz_stub: Optional[Callable[[str, str], int]] = options.pop("_fuzz_stub", None)
        if fuzz_stub is not None:
            monkeypatch.setattr(
                section_loader,
                "fuzz",
                SimpleNamespace(token_set_ratio=fuzz_stub),
            )

        return section_loader.SectionLoader(**options)

    return _factory


@pytest.mark.parametrize(
    (
        "description",
        "schema",
        "line",
        "options",
        "expected",
    ),
    (
        (
            "duplicate headers prefer the first defined entry",
            {
                "introduccion": {
                    "title": "Introducción",
                    "aliases": ["Resumen inicial"],
                    "keywords": [],
                },
                "introduccion_bis": {
                    "title": "Introducción",
                    "aliases": [],
                    "keywords": [],
                },
            },
            "1. Introducción",
            {"tipo": "demo", "fuzzy": False},
            ("introduccion", "regex"),
        ),
        (
            "empty schema yields no matches (negative path)",
            {},
            "Metodología",
            {"tipo": "demo", "fuzzy": False},
            None,
        ),
        (
            "noisy content should be ignored",
            {
                "metodologia": {
                    "title": "Metodología",
                    "aliases": [],
                    "keywords": ["procedimiento"],
                }
            },
            "//// sin encabezado válido ////",
            {"tipo": "demo", "fuzzy": False},
            None,
        ),
        (
            "fuzzy mode tolerates accent and suffix noise",
            {
                "marco_teorico": {
                    "title": "Marco Teorico",
                    "aliases": [],
                    "keywords": [],
                }
            },
            "Marco teórico y antecedentes",
            {
                "tipo": "demo",
                "fuzzy": True,
                "_force_fuzz_available": True,
                "_fuzz_stub": lambda a, b: 95 if b in a else 0,
            },
            ("marco_teorico", "fuzzy"),
        ),
    ),
)
def test_identify_section_variations(
    loader_factory,
    description: str,
    schema: Dict[str, Dict[str, Any]],
    line: str,
    options: Dict[str, Any],
    expected: Optional[Tuple[str, str]],
) -> None:
    """Comprehensive identification scenarios spanning success and failure paths."""

    loader = loader_factory(schema, dict(options))
    result = loader.identify_section(line)

    if expected is None:
        assert result is None, description
        return

    assert result is not None, description
    section_id, match_type, *_ = result
    assert (section_id, match_type) == expected


def test_fuzzy_toggle_reflects_availability(loader_factory) -> None:
    """SectionLoader must disable fuzzy mode when the backend is absent."""

    schema = {
        "resumen": {"title": "Resumen", "aliases": [], "keywords": []}
    }

    loader_without_fuzz = loader_factory(
        schema,
        {"tipo": "demo", "fuzzy": True, "_force_fuzz_available": False},
    )
    assert loader_without_fuzz._fuzzy_enabled is False

    loader_with_fuzz = loader_factory(
        schema,
        {
            "tipo": "demo",
            "fuzzy": True,
            "_force_fuzz_available": True,
            "_fuzz_stub": lambda a, b: 100,
        },
    )
    assert loader_with_fuzz._fuzzy_enabled is True