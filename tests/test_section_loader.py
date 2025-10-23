"""Unit tests for :mod:`data.criteria.section_loader`"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Dict, Any

import pytest

from data.criteria import section_loader


def _write_schema(tmp_path, monkeypatch, schema: Dict[str, Dict[str, Any]], tipo: str) -> None:
    """Helper that writes a schema JSON and points the loader towards it."""

    file_path = tmp_path / f"secciones_{tipo}.json"
    file_path.write_text(json.dumps(schema), encoding="utf-8")
    monkeypatch.setattr(section_loader, "CRITERIA_DIR", str(tmp_path))


def test_identify_section_without_fuzzy(tmp_path, monkeypatch) -> None:
    """When fuzzy mode is disabled, regex patterns should still identify headings."""

    schema = {
        "introduccion": {
            "title": "Introducción",
            "aliases": ["Resumen inicial"],
            "keywords": [],
        }
    }
    _write_schema(tmp_path, monkeypatch, schema, tipo="demo")

    loader = section_loader.SectionLoader(tipo="demo", fuzzy=False)
    
    result = loader.identify_section("1. Introducción")
    assert result is not None
    section_id, match_type, *_ = result
    assert section_id == "introduccion"
    assert match_type == "regex"
    assert loader._fuzzy_enabled is False


def test_identify_section_with_fuzzy(tmp_path, monkeypatch) -> None:
    """Fuzzy matching is used when requested and available."""

    schema = {
        "marco_teorico": {
            "title": "Marco Teorico",
            "aliases": [],
            "keywords": [],
        }
    }
    _write_schema(tmp_path, monkeypatch, schema, tipo="demo")

    # Force fuzzy availability with a deterministic stub so the test is hermetic.
    monkeypatch.setattr(section_loader, "_HAS_FUZZ", True)
    monkeypatch.setattr(
        section_loader,
        "fuzz",
        SimpleNamespace(token_set_ratio=lambda a, b: 100 if a == b else 0),
    )

    loader = section_loader.SectionLoader(tipo="demo", fuzzy=True)

    result = loader.identify_section("Marco teórico")
    assert result is not None
    section_id, match_type, score, _ = result
    assert section_id == "marco_teorico"
    assert match_type == "fuzzy"
    assert score == 100
    assert loader._fuzzy_enabled is True