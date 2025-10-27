import pytest

from utils.text_preprocessing import (
    clean_html,
    remove_page_breaks,
    collapse_repeated_characters,
    normalize_text,
)


def test_clean_html_removes_tags_and_entities():
    raw = "<p>Hola &amp; bienvenidos<br>al <strong>portal</strong></p>"
    assert clean_html(raw) == "Hola & bienvenidos al portal"


def test_remove_page_breaks_strips_markers():
    raw = "Primera pagina\f=== PAGE 2 ===\nSegunda"
    assert remove_page_breaks(raw) == "Primera pagina Segunda"


def test_collapse_repeated_characters_limits_sequences():
    raw = "Nooooo!!! Esto es increiiiible???"
    cleaned = collapse_repeated_characters(raw, max_repeats=2)
    assert cleaned == "Noo! Esto es increiible?"


def test_normalize_text_lowercases_and_strips_accents():
    raw = "¡Árboles y Cañones!"
    assert normalize_text(raw) == "arboles y canones"


def test_collapse_repeated_characters_invalid_max():
    with pytest.raises(ValueError):
        collapse_repeated_characters("texto", max_repeats=0)

# pytest tests/utils/test_text_preprocessing.py -v