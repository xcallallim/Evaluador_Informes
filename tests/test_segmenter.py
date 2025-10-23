"""Unit tests for the Segmenter using synthetic cleaned documents."""

from __future__ import annotations

import re
from typing import Callable

import pytest

from data.models.document import Document
from data.preprocessing import segmenter as segmenter_module
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


@pytest.fixture
def make_segmenter(monkeypatch: pytest.MonkeyPatch) -> Callable[[bool], Segmenter]:
    """Factory that injects a controlled SectionLoader implementation."""

    def _factory(fuzzy: bool = True) -> Segmenter:
        class DummySectionLoader:
            def __init__(self, tipo: str, fuzzy_flag: bool):
                self.tipo = tipo
                self.sections = [
                    "resumen_ejecutivo",
                    "aplicacion_recomendaciones",
                    "conclusiones",
                    "anexos",
                    "introduccion",
                    "resultados",
                ]
                self.patterns = {
                    "resumen_ejecutivo": [
                        re.compile(r"^\s*resumen ejecutivo\b", re.IGNORECASE)
                    ],
                    "aplicacion_recomendaciones": [
                        re.compile(
                            r"^\s*aplicacion de recomendaciones\b", re.IGNORECASE
                        )
                    ],
                    "conclusiones": [
                        re.compile(r"^\s*conclusiones\b", re.IGNORECASE)
                    ],
                    "anexos": [re.compile(r"^\s*anexos\b", re.IGNORECASE)],
                    "introduccion": [
                        re.compile(
                            r"^\s*(?:\d+(?:\.\d+)*\s*[:\-\.]\s*)?introducci[oó]n\b",
                            re.IGNORECASE,
                        )
                    ],
                    "resultados": [
                        re.compile(
                            r"^\s*(?:\d+(?:\.\d+)*\s*[:\-\.]\s*)?resultados\b",
                            re.IGNORECASE,
                        )
                    ],
                }
                self._fuzzy_enabled = fuzzy_flag
                self.match_log = []

            def identify_section(self, line: str, **_: object):
                stripped = (line or "").strip().lower()
                normalized = (
                    stripped.replace("á", "a")
                    .replace("é", "e")
                    .replace("í", "i")
                    .replace("ó", "o")
                    .replace("ú", "u")
                )
                normalized = re.sub(
                    r"^(?:seccion\s*\d+\s*[:\-\.]\s*|\d+(?:\.\d+)*\s*[:\-\.]\s*)",
                    "",
                    normalized,
                )
                if normalized.startswith("resumen ejecutivo"):
                    result = ("resumen_ejecutivo", "regex", 100.0, line)
                    self.match_log.append(result)
                    return result
                if normalized.startswith("resumen general"):
                    result = ("resumen_ejecutivo", "alias", 95.0, line)
                    self.match_log.append(result)
                    return result
                if normalized.startswith("aplicacion de recomendaciones"):
                    result = ("aplicacion_recomendaciones", "regex", 100.0, line)
                    self.match_log.append(result)
                    return result
                if (
                    normalized.startswith("aplicaciones de las recomendaciones")
                    and self._fuzzy_enabled
                ):
                    result = ("aplicacion_recomendaciones", "fuzzy", 92.0, line)
                    self.match_log.append(result)
                    return result
                if (
                    normalized.startswith("aplicacion de las recomendaciones")
                    and self._fuzzy_enabled
                ):
                    result = ("aplicacion_recomendaciones", "fuzzy", 91.0, line)
                    self.match_log.append(result)
                    return result
                if (
                    normalized.startswith("recomendaciones aplicadas")
                    and self._fuzzy_enabled
                ):
                    result = ("aplicacion_recomendaciones", "fuzzy", 90.0, line)
                    self.match_log.append(result)
                    return result
                if normalized.startswith("conclusiones"):
                    result = ("conclusiones", "regex", 100.0, line)
                    self.match_log.append(result)
                    return result
                if normalized.startswith("anexos"):
                    result = ("anexos", "regex", 100.0, line)
                    self.match_log.append(result)
                    return result
                if normalized.startswith("introduccion"):
                    result = ("introduccion", "regex", 100.0, line)
                    self.match_log.append(result)
                    return result
                if normalized.startswith("resultados"):
                    result = ("resultados", "regex", 100.0, line)
                    self.match_log.append(result)
                    return result
                return None

        monkeypatch.setattr(
            segmenter_module,
            "SectionLoader",
            lambda tipo, fuzzy: DummySectionLoader(tipo, fuzzy),
        )
        return segmenter_module.Segmenter(tipo="institucional", fuzzy=fuzzy)

    return _factory


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


def test_segmenter_preserves_internal_blank_lines(segmenter: Segmenter) -> None:
    """Blank lines that separare párrafos se mantienen en las secciones."""

    content = "\n".join(
        [
            "1. Resumen Ejecutivo",
            "Primer párrafo del resumen.",
            "",
            "",
            "Segundo párrafo del resumen.",
            "Sección 4: Conclusiones",
            "Conclusiones finales.",
        ]
    )
    document = Document(content=content)

    segmented = segmenter.segment_document(document)

    expected = "Primer párrafo del resumen.\n\n\nSegundo párrafo del resumen."
    assert segmented.sections["resumen_ejecutivo"] == expected


def test_segmenter_supports_inline_heading_content(segmenter: Segmenter) -> None:
    """Content placed on the same line as the heading is preserved after the title."""

    content = "\n".join(
        [
            "1. Resumen Ejecutivo El informe resume avances inmediatos.",
            "",
            "Sección 4: Conclusiones Se concluye que el plan fue efectivo.",
        ]
    )
    document = Document(content=content)

    segmented = segmenter.segment_document(document)

    assert (
        segmented.sections["resumen_ejecutivo"]
        == "El informe resume avances inmediatos."
    )
    assert (
        segmented.sections["conclusiones"]
        == "Se concluye que el plan fue efectivo."
    )


def test_segmenter_uses_last_occurrence_for_repeated_headings(segmenter: Segmenter) -> None:
    """If a heading appears múltiples veces se conserva el contenido de la última aparición."""

    content = "\n".join(
        [
            "Resumen Ejecutivo",
            "Versión preliminar.",
            "",
            "Resumen Ejecutivo",
            "Versión definitiva.",
        ]
    )
    document = Document(content=content)

    segmented = segmenter.segment_document(document)

    assert segmented.sections["resumen_ejecutivo"] == "Versión definitiva."


def test_segmenter_handles_large_document(segmenter: Segmenter) -> None:
    """Procesa documentos grandes sin duplicar memoria innecesariamente."""

    body = "\n".join(f"Contenido línea {i}" for i in range(1, 5001))
    content = "\n".join(
        [
            "Resumen Ejecutivo",
            body,
            "Sección 4: Conclusiones",
            "Conclusión final.",
        ]
    )
    document = Document(content=content)

    segmented = segmenter.segment_document(document)

    assert segmented.sections["resumen_ejecutivo"].startswith("Contenido línea 1")
    assert segmented.sections["resumen_ejecutivo"].endswith("Contenido línea 5000")
    assert "Sección 4: Conclusiones" not in segmented.sections["resumen_ejecutivo"]
    assert segmented.sections["conclusiones"] == "Conclusión final."


def test_segmenter_detects_aliases_and_single_word_headings(
    make_segmenter: Callable[[bool], Segmenter]
) -> None:
    """Alias y encabezados de una palabra se asignan correctamente."""

    segmenter = make_segmenter(fuzzy=True)
    content = "\n".join(
        [
            "Resumen General",
            "Síntesis ejecutiva del informe.",
            "Conclusiones",
            "Conclusiones finales del documento.",
            "Anexos",
            "Listado de anexos relevantes.",
        ]
    )
    document = Document(content=content)

    segmented = segmenter.segment_document(document)

    assert segmented.sections["resumen_ejecutivo"] == "Síntesis ejecutiva del informe."
    assert segmented.sections["conclusiones"] == "Conclusiones finales del documento."
    assert segmented.sections["anexos"] == "Listado de anexos relevantes."


@pytest.mark.parametrize(
    "heading",
    [
        "Aplicaciones de las recomendaciones",
        "Aplicación de las recomendaciones",
        "Recomendaciones aplicadas",
    ],
)
def test_segmenter_fuzzy_detects_lexical_variations(
    make_segmenter: Callable[[bool], Segmenter], heading: str
) -> None:
    """Variaciones léxicas se reconocen cuando fuzzy está habilitado."""

    segmenter = make_segmenter(fuzzy=True)
    content = "\n".join(
        [
            "Resumen General",
            "Síntesis ejecutiva del informe.",
            heading,
            "Detalle de aplicación y seguimiento.",
            "Conclusiones",
            "Cierre del informe.",
        ]
    )
    document = Document(content=content)

    segmented = segmenter.segment_document(document)

    assert (
        segmented.sections["aplicacion_recomendaciones"]
        == "Detalle de aplicación y seguimiento."
    )


def test_segmenter_without_fuzzy_skips_variations(
    make_segmenter: Callable[[bool], Segmenter]
) -> None:
    """Sin fuzzy, las variaciones léxicas no deben etiquetarse como encabezados."""

    segmenter = make_segmenter(fuzzy=False)
    content = "\n".join(
        [
            "Resumen Ejecutivo",
            "Primer bloque de contexto.",
            "Aplicaciones de las recomendaciones",
            "Detalle de aplicación y seguimiento.",
            "Conclusiones",
            "Cierre del informe.",
        ]
    )
    document = Document(content=content)

    segmented = segmenter.segment_document(document)

    assert segmented.sections["aplicacion_recomendaciones"] == ""
    assert "Detalle de aplicación" in segmented.sections["resumen_ejecutivo"]


def test_segmenter_handles_consecutive_inline_headings(
    make_segmenter: Callable[[bool], Segmenter]
) -> None:
    """Encabezados inline consecutivos se separan respetando el contenido."""

    segmenter = make_segmenter(fuzzy=True)
    content = "\n".join(
        [
            "1. Introducción Este informe describe resultados preliminares.",
            "2. Resultados Se observa una mejora sostenida.",
            "Anexos Documentación adicional.",
        ]
    )
    document = Document(content=content)

    segmented = segmenter.segment_document(document)

    assert (
        segmented.sections["introduccion"]
        == "Este informe describe resultados preliminares."
    )
    assert (
        segmented.sections["resultados"]
        == "Se observa una mejora sostenida."
    )
    assert segmented.sections["anexos"] == "Documentación adicional."


def test_segmenter_provides_uncategorized_fallback(segmenter: Segmenter) -> None:
    """Cuando no hay encabezados, se crea la sección sin_clasificar."""

    document = Document(content="Texto suelto sin encabezados identificables.")

    segmented = segmenter.segment_document(document)

    assert segmented.sections == {
        "sin_clasificar": "Texto suelto sin encabezados identificables."
    }


def test_segmenter_handles_empty_content(segmenter: Segmenter) -> None:
    """Documentos vacíos no deben causar errores."""

    document = Document(content="")

    segmented = segmenter.segment_document(document)

    assert segmented.sections == {}


def test_segmenter_handles_objects_without_content(segmenter: Segmenter) -> None:
    """Objetos sin atributo ``content`` generan secciones vacías."""

    class Minimal:
        pass

    minimal = Minimal()

    segmented = segmenter.segment_document(minimal)  # type: ignore[arg-type]

    assert hasattr(segmented, "sections")
    assert segmented.sections == {}


def test_segmenter_assigns_text_outside_headings(
    make_segmenter: Callable[[bool], Segmenter]
) -> None:
    """Texto intercalado sin encabezado se anexa al bloque previo o fallback."""

    segmenter = make_segmenter(fuzzy=True)
    content = "\n".join(
        [
            "Resumen Ejecutivo",
            "El informe describe avances prioritarios.",
            "Texto suelto que amplía sin encabezado.",
            "Conclusiones",
            "Se observa cumplimiento total.",
            "Recomendaciones abiertas sin encabezado final.",
        ]
    )
    document = Document(content=content)

    segmented = segmenter.segment_document(document)

    assert "Texto suelto" in segmented.sections["resumen_ejecutivo"]
    assert segmented.sections["conclusiones"].endswith(
        "Recomendaciones abiertas sin encabezado final."
    )


def test_segmenter_preserves_section_order(
    make_segmenter: Callable[[bool], Segmenter]
) -> None:
    """Las secciones detectadas mantienen el orden esperado y tipo dict."""

    segmenter = make_segmenter(fuzzy=True)
    content = "\n".join(
        [
            "Resumen Ejecutivo",
            "Introducción inicial.",
            "Aplicacion de recomendaciones",
            "Detalle de aplicación y seguimiento.",
            "Conclusiones",
            "Cierre del informe.",
        ]
    )
    document = Document(content=content)

    segmented = segmenter.segment_document(document)

    assert isinstance(segmented.sections, dict)
    ordered_keys = list(segmented.sections.keys())
    expected_sequence = [
        "resumen_ejecutivo",
        "aplicacion_recomendaciones",
        "conclusiones",
    ]
    filtered = [key for key in ordered_keys if key in expected_sequence]
    assert filtered[:3] == expected_sequence
    assert "anexos" in ordered_keys


def test_segmenter_loader_reports_match_types(
    make_segmenter: Callable[[bool], Segmenter]
) -> None:
    """El mock del loader expone el tipo de coincidencia para trazabilidad."""

    segmenter = make_segmenter(fuzzy=True)
    loader = segmenter.loader

    regex_match = loader.identify_section("Resumen Ejecutivo")
    alias_match = loader.identify_section("Resumen General")
    fuzzy_match = loader.identify_section("Aplicaciones de las recomendaciones")

    assert regex_match is not None and regex_match[1] == "regex"
    assert alias_match is not None and alias_match[1] == "alias"
    assert fuzzy_match is not None and fuzzy_match[1] == "fuzzy"

# pytest tests/test_segmenter.py -v
