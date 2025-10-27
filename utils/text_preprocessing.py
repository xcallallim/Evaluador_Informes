"""Utilidades para limpiar y normalizar contenido textual de datasets."""

from __future__ import annotations

import re
import unicodedata
from html import unescape

__all__ = [
    "clean_html",
    "remove_page_breaks",
    "collapse_repeated_characters",
    "normalize_text",
]


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_PAGE_BREAK_RE = re.compile(r"\f+")
_PAGE_MARKER_RE = re.compile(r"^=+\s*page\s*\d+\s*=+$", re.IGNORECASE | re.MULTILINE)
_WHITESPACE_RE = re.compile(r"\s+")
_REPEATED_CHAR_TEMPLATE = r"(.)\1{{{min_repeats},}}"
_PUNCTUATION_SINGLE = {".", ",", ";", ":", "!", "?", "¡", "¿"}
_PUNCT_TRANSLATION = str.maketrans({
    "¡": " ",
    "¿": " ",
    "–": "-",
    "—": "-",
})


def clean_html(text: str) -> str:
    """Remueve etiquetas HTML y decodificar entidades a partir de "texto"."""

    if not isinstance(text, str):
        return ""

    unescaped = unescape(text)
    without_tags = _HTML_TAG_RE.sub(" ", unescaped)
    collapsed = _WHITESPACE_RE.sub(" ", without_tags)
    return collapsed.strip()


def remove_page_breaks(text: str) -> str:
    """Quitar los caracteres de avance de página y los marcadores de página comunes del "texto"."""

    if not isinstance(text, str):
        return ""

    without_form_feed = _PAGE_BREAK_RE.sub("\n", text)
    without_markers = _PAGE_MARKER_RE.sub("", without_form_feed)
    return _WHITESPACE_RE.sub(" ", without_markers).strip()


def collapse_repeated_characters(text: str, max_repeats: int = 2) -> str:
    """Contraer caracteres repetidos a un máximo razonable."""

    if not isinstance(text, str) or not text:
        return ""

    if max_repeats < 1:
        raise ValueError("max_repeats debe ser al menos 1")

    pattern = re.compile(_REPEATED_CHAR_TEMPLATE.format(min_repeats=max_repeats))

    def _replacer(match: re.Match[str]) -> str:
        char = match.group(1)
        limit = 1 if char in _PUNCTUATION_SINGLE else max_repeats
        return char * limit

    collapsed = pattern.sub(_replacer, text)
    return collapsed


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_text(text: str) -> str:
    """Normalizar ``texto``: minúsculas, eliminar acentos y puntuación."""

    if not isinstance(text, str):
        return ""

    text = text.translate(_PUNCT_TRANSLATION)
    text = text.lower()
    text = _strip_accents(text)
    text = text.translate(str.maketrans("", "", "\"'`´^~.,;:!?¡¿()[]{}<>|\\/+-=*_#@&$%"))
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text