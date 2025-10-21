# data/models/document.py

from __future__ import annotations

from collections.abc import Iterable as IterableABC
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, TYPE_CHECKING, cast

if TYPE_CHECKING:
    try:  # pragma: no cover - sólo para herramientas estáticas
        from langchain_core.documents import Document as LCDocumentType
    except ImportError:  # pragma: no cover
        from langchain.schema import Document as LCDocumentType  # type: ignore[no-redef]
else:  # pragma: no cover - en tiempo de ejecución preferimos evitar dependencias opcionales
    LCDocumentType = Any  # type: ignore[assignment]


def _coerce_text(value: Any) -> str:
    """Convierte cualquier valor en texto seguro para el pipeline."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _coerce_text_list(values: Optional[Iterable[Any]]) -> List[str]:
    """Normaliza colecciones de texto evitando nulos y espacios."""

    if not values:
        return []

    if isinstance(values, str):
        text = values.strip()
        return [text] if text else []

    normalized: List[str] = []
    for item in values:
        if item is None:
            continue
        text = _coerce_text(item).strip()
        if text:
            normalized.append(text)
    return normalized


def _coerce_sections(value: Optional[Mapping[Any, Any]]) -> Dict[str, str]:
    """Garantiza un diccionario {str: str} limpio para las secciones."""

    if not value:
        return {}

    if not isinstance(value, Mapping):
        return {}

    sections: Dict[str, str] = {}
    for key, raw_text in value.items():
        section_id = _coerce_text(key).strip()
        text = _coerce_text(raw_text).strip()
        if section_id and text:
            sections[section_id] = text
    return sections


def _coerce_list(value: Optional[Iterable[Any]]) -> List[Any]:
    """Devuelve siempre una lista sin nulos, aceptando tuplas o sets."""

    if not value:
        return []

    if isinstance(value, list):
        return [item for item in value if item is not None]

    if isinstance(value, (tuple, set)):
        return [item for item in value if item is not None]

    if isinstance(value, (str, bytes, bytearray)):
        return [value]

    if isinstance(value, IterableABC):
        return [item for item in value if item is not None]

    return [value]


@dataclass

class Document:
    """Representa un documento enriquecido con metadatos y chunks LangChain."""

    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    pages: List[str] = field(default_factory=list)
    tables: List[Any] = field(default_factory=list)
    images: List[Any] = field(default_factory=list)
    sections: Dict[str, str] = field(default_factory=dict)
    chunks: List["LCDocumentType"] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.content = _coerce_text(self.content).strip()

        if isinstance(self.metadata, MutableMapping):
            self.metadata = dict(self.metadata)
        else:
            self.metadata = {}

        self.pages = _coerce_text_list(self.pages)
        self.tables = _coerce_list(self.tables)
        self.images = _coerce_list(self.images)
        self.sections = _coerce_sections(self.sections)
        self.chunks = cast(List["LCDocumentType"], _coerce_list(self.chunks))

    def __repr__(self) -> str:
        return (
            "Document("
            f"pages={len(self.pages)}, "
            f"tables={len(self.tables)}, "
            f"sections={len(self.sections)}, "
            f"chunks={len(self.chunks)}"
            ")"
        )
    
    def as_langchain_documents(self) -> List["LCDocumentType"]:
        """Devuelve los chunks como ``langchain`` Documents listos para embeddings."""

        return list(self.chunks)

