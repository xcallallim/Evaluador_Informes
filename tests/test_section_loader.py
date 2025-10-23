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
            "roman chapter prefixes remain detectable via regex",
            {
                "introduccion": {
                    "title": "Introducción",
                    "aliases": ["Introduccion"],
                    "keywords": [],
                }
            },
            "Capitulo II - Introducción",
            {"tipo": "demo", "fuzzy": False},
            ("introduccion", "regex"),
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


def test_section_level_fuzzy_threshold(loader_factory) -> None:
    """Secciones pueden personalizar su umbral fuzzy para títulos cortos."""

    base_schema = {
        "recomendaciones": {
            "title": "Recomendaciones",
            "aliases": [],
            "keywords": [],
        }
    }
    stub = lambda _a, b: 82 if b == "recomendaciones" else 0

    base_loader = loader_factory(
        base_schema,
        {
            "tipo": "base",
            "fuzzy": True,
            "_force_fuzz_available": True,
            "_fuzz_stub": stub,
        },
    )
    assert (
        base_loader.identify_section("Aplicaciones recomendadas") is None
    ), "Default threshold should reject low score"

    tuned_schema = {
        "recomendaciones": {
            "title": "Recomendaciones",
            "aliases": [],
            "keywords": [],
            "fuzzy_threshold": 80,
        }
    }

    tuned_loader = loader_factory(
        tuned_schema,
        {
            "tipo": "tuned",
            "fuzzy": True,
            "_force_fuzz_available": True,
            "_fuzz_stub": stub,
        },
    )

    result = tuned_loader.identify_section("Aplicaciones recomendadas")
    assert result is not None
    assert result[0] == "recomendaciones"
    assert result[1] == "fuzzy"


def test_schema_default_fuzzy_threshold(loader_factory) -> None:
    """Configuración global aplica a todas las secciones sin umbral específico."""

    schema = {
        "__config__": {"default_fuzzy_threshold": 80},
        "recomendaciones": {
            "title": "Recomendaciones",
            "aliases": [],
            "keywords": [],
        },
    }
    stub = lambda _a, b: 82 if b == "recomendaciones" else 0

    loader = loader_factory(
        schema,
        {
            "tipo": "config",
            "fuzzy": True,
            "_force_fuzz_available": True,
            "_fuzz_stub": stub,
        },
    )

    result = loader.identify_section("Aplicaciones recomendadas")
    assert result is not None and result[1] == "fuzzy"

    assert (
        loader.identify_section("Aplicaciones recomendadas", fuzzy_threshold=90)
        is None
    ), "Call override should still be respected"


def test_keyword_ratio_threshold(loader_factory) -> None:
    """El refuerzo por keywords utiliza proporción de aciertos configurable."""

    schema = {
        "seguridad_datos": {
            "title": "Seguridad de Datos",
            "aliases": [],
            "keywords": ["acceso", "seguridad", "datos"],
        }
    }

    loader = loader_factory(schema, {"tipo": "kw", "fuzzy": False})

    match = loader.identify_section(
        "Plan de seguridad integral y acceso a los datos"
    )
    assert match is not None and match[0] == "seguridad_datos"

    no_match = loader.identify_section("Monitoreo de acceso continuo")
    assert no_match is None

    tuned_schema = {
        "seguridad_datos": {
            "title": "Seguridad de Datos",
            "aliases": [],
            "keywords": ["acceso", "seguridad", "datos"],
            "keyword_min_ratio": 0.3,
        }
    }

    tuned_loader = loader_factory(tuned_schema, {"tipo": "kw_tuned", "fuzzy": False})
    tuned_match = tuned_loader.identify_section("Monitoreo de acceso continuo")
    assert tuned_match is not None and tuned_match[0] == "seguridad_datos"


def test_keyword_fallback_without_regex_or_fuzzy(loader_factory) -> None:
    """Las keywords deben identificar secciones cuando regex y fuzzy no aplican."""

    schema = {
        "analisis_riesgos": {
            "title": "Análisis de Riesgos",
            "aliases": [],
            "keywords": ["riesgo", "impacto"],
        }
    }

    loader = loader_factory(schema, {"tipo": "kw_only", "fuzzy": False})

    line = "Evaluación del impacto y riesgo operativo"
    match = loader.identify_section(line)

    assert match is not None
    assert match[0] == "analisis_riesgos"
    assert match[1] == "keywords"


def test_normalize_handles_accents_and_case() -> None:
    """La normalización quita acentos y homogeneiza mayúsculas."""

    accented = section_loader._normalize("MARCO TEÓRICO", strip_accents=True)
    plain = section_loader._normalize("marco teorico", strip_accents=True)

    assert accented == plain == "marco teorico"


def test_invalid_schema_file_raises_runtime_error(tmp_path, monkeypatch) -> None:
    """Un JSON malformado debe generar un RuntimeError controlado."""

    bad_file = tmp_path / "secciones_err.json"
    bad_file.write_text("{ invalid json", encoding="utf-8")

    monkeypatch.setattr(section_loader, "CRITERIA_DIR", str(tmp_path))

    with pytest.raises(RuntimeError):
        section_loader.SectionLoader(tipo="err", fuzzy=False)


def test_fuzzy_match_returns_score_above_threshold(loader_factory) -> None:
    """El score fuzzy devuelto debe respetar el umbral mínimo requerido."""

    schema = {
        "resultados": {
            "title": "Resultados",
            "aliases": [],
            "keywords": [],
        }
    }

    def stub(a: str, b: str) -> int:
        return 91 if b == "resultados" else 10

    loader = loader_factory(
        schema,
        {
            "tipo": "fuzzy_score",
            "fuzzy": True,
            "_force_fuzz_available": True,
            "_fuzz_stub": stub,
        },
    )

    match = loader.identify_section("Se presentan los principales resultados del estudio")

    assert match is not None
    assert match[1] == "fuzzy"
    assert match[2] >= 86.0
