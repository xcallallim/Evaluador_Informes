"""Helpers for dealing with text normalisation, filesystem utilities and JSON.

The functions in this module are intentionally lightweight so they can be used
from both notebooks and production code.  They avoid raising unexpected
exceptions and provide better diagnostics when optional dependencies such as
``langchain_text_splitters`` are not available.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, List, Union

from core.logger import log_debug, log_error

try:  # pragma: no cover - import guard covered indirectly via fallback logic.
    from langchain_text_splitters import CharacterTextSplitter  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    CharacterTextSplitter = None

PathLike = Union[str, os.PathLike[str]]

__all__ = [
    "clean_spaces",
    "ensure_dir",
    "load_json",
    "normalize_text",
    "normalize_whitespace",
    "print_header",
    "save_json",
    "split_text",
    "strip_accents",
]

# ====================================
# FUNCIONES DE NORMALIZACIÓN
# ====================================

def strip_accents(text: str) -> str:
    """Return ``text`` without diacritics using Unicode normalisation."""
    if not isinstance(text, str):
        return text
    
    normalised = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalised if unicodedata.category(ch) != "Mn")


def normalize_text(text: str) -> str:
    """Return a simplified version of ``text`` for fuzzy comparisons."""

    if not isinstance(text, str):
        return ""
    
    simplified = strip_accents(text.lower())
    simplified = re.sub(r"[^a-z0-9\s]", " ", simplified)
    simplified = re.sub(r"\s+", " ", simplified).strip()
    return simplified


def clean_spaces(text: str) -> str:
    """Collapse duplicated whitespace while keeping content intact."""

    if not isinstance(text, str):
        return ""
    
    return re.sub(r"\s+", " ", text).strip()


def normalize_whitespace(text: str) -> str:
    """Return ``text`` with tabs/extra spaces removed but original casing kept."""

    if not isinstance(text, str):
        return ""

    cleaned = unicodedata.normalize("NFKC", text)
    cleaned = cleaned.replace("\uf0b7", "").replace("\u2022", "")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"[ ]*\n[ ]*", "\n", cleaned)
    return cleaned.strip()


# ====================================
# FUNCIONES PARA ARCHIVOS JSON
# ====================================
def load_json(path: PathLike, *, encoding: str = "utf-8") -> Any | None:
    """Load JSON content from *path* returning ``None`` on failures."""

    json_path = Path(path)

    try:
        with json_path.open("r", encoding=encoding) as file:
            return json.load(file)
    except FileNotFoundError:
        log_error(f"No se encontró el archivo JSON: {json_path}")
    except json.JSONDecodeError as exc:
        log_error(f"El archivo JSON está mal formado ({json_path}): {exc}")
    except OSError as exc:
        log_error(f"No se pudo abrir el JSON {json_path}: {exc}")
    return None


def save_json(data, path: PathLike, *, encoding: str = "utf-8") -> None:
    """Persist ``data`` as JSON in ``path``, creating directories as needed."""

    json_path = Path(path)
    ensure_dir(json_path.parent)

    try:
        with json_path.open("w", encoding=encoding) as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
    except TypeError as exc:
        log_error(f"No se pudo serializar JSON en {json_path}: {exc}")
    except OSError as exc:
        log_error(f"No se pudo guardar JSON en {json_path}: {exc}")


# ====================================
# UTILERÍA VISUAL
# ====================================

def print_header(title: str) -> None:
    """Print a simple header to separate stages when running in a terminal."""

    print(f"\n{'=' * 10} {title.upper()} {'=' * 10}")


# ====================================
# MANEJO DE CARPETAS
# ====================================

def ensure_dir(path: PathLike) -> Path:
    """Create *path* (and its parents) if it does not already exist."""

    directory = Path(path)
    if not directory.exists():
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # pragma: no cover - depends on permissions.
            log_error(f"No se pudo crear el directorio {directory}: {exc}")
            raise
        else:
            log_debug(f"Directorio creado: {directory}")
    return directory

# ====================================
# DIVISION POR CHUNKS
# ====================================

def _split_text_fallback(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Simple chunking fallback used when LangChain is not installed."""

    if not text:
        return []

    step = chunk_size - chunk_overlap
    chunks: List[str] = []
    for start in range(0, len(text), step):
        end = start + chunk_size
        chunks.append(text[start:end])
    return chunks


def split_text(
    text: str,
    chunk_size: int = 1500,
    chunk_overlap: int = 150,
) -> List[str]:
    """Split ``text`` into LangChain-compatible chunks.

    When ``langchain_text_splitters`` is not installed a lightweight fallback is
    used so the rest of the pipeline can continue to operate (for instance in
    unit tests or constrained environments).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size debe ser mayor que cero")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap no puede ser negativo")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap debe ser menor que chunk_size")
    if not isinstance(text, str):
        raise TypeError("text debe ser una cadena")

    if CharacterTextSplitter is None:
        log_debug(
            "langchain_text_splitters no está disponible; usando división simple"
        )
        return _split_text_fallback(text, chunk_size, chunk_overlap)
    
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_text(text)