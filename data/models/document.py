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
    """Entidad central que fluye a lo largo del pipeline de evaluación.

    La clase de datos "Documento" es generada por los cargadores especializados
    y se enriquece progresivamente en etapas posteriores (Limpiador, Segmentador, 
    Divisor y Evaluador). Además del atributo "contenido", el objeto transporta 
    datos estructurados que otros componentes utilizan para decidir cómo procesar 
    el documento.

    Metadata contract
    -----------------
    Se garantiza que la asignación de metadatos contenga las siguientes claves 
    una vez que :class:`~data.preprocessing.loader.DocumentLoader` devuelva la 
    instancia:

    ``source`` (``str``)
        Ruta absoluta al archivo ingerido. Utilizada por registradores y 
        exportadores para rastrear el artefacto original.

    ``filename`` (``str``)
       Parte del nombre base de ``source``. Expuesto en informes e interfaces de 
       usuario.

    ``extension`` (``str``)
        Extensión de archivo en minúsculas (incluido el punto inicial). Permite 
        la gestión posterior de formatos específicos.

    ``processed_with`` (``str``)
        Identifica el cargador responsable de ensamblar el documento. El valor 
        actual es «DocumentLoader» y actúa como protección de procedencia cuando 
        coexisten varios sistemas.

    ``pages`` (``List[str]``)
        Representación textual de cada página (sin los marcadores ``=== PAGE``) 
        tal como la extrajo el cargador. Los consumidores reutilizan estos datos 
        al limpiar encabezados/pies de página repetidos o al reconstruir la 
        paginación en las exportaciones.

    ``tables`` (``Dict[str, Any]``)
        Asignación del identificador del cargador a la colección de tablas 
        detectadas.Cada cargador define su propia carga útil: los cargadores 
        PDF devuelven diccionarios con pares página/ruta, mientras que los 
        cargadores DOCX envían matrices de celdas sin procesar. El contrato 
        solo garantiza el diccionario externo.

    ``is_ocr`` (``bool``)
        Indica si el texto se originó mediante un proceso de OCR. Se establece 
        en "Verdadero" para archivos PDF escaneados y en "Falso" para fuentes 
        digitales. El limpiador activa y desactiva la heurística difusa según 
        este valor.

    ``extraction_method`` (``str``)
        Describe cómo se obtuvo el contenido textual. Los valores actuales son 
        ``"ocr"``, ``"embedded_text"``, ``"docx"`` y ``"text"``. Al añadir un 
        nuevo cargador, esta enumeración se ampliará según corresponda.

    ``raw_text`` (``str``)
        Carga textual sin procesar proporcionada por el cargador antes de 
        cualquier limpieza. En el caso de los PDF con OCR, contiene el texto 
        reconocido, mientras que en otros formatos puede reflejar el contenido 
        o omitirse si no está disponible.

    ``images`` (``List[Any]`` | missing)
        Presente cuando el emisor solicitó la extracción de la imagen. Contiene 
        referencias del exportador (normalmente rutas del sistema de archivos) 
        generadas por el cargador de PDF.

    ``issues`` (``List[str]`` | missing)
        Recopilación desduplicada de advertencias reportadas durante la fase de 
        carga.Las tuberías envían estas advertencias al usuario, pero continúan 
        procesando el documento.

    Las claves de metadatos adicionales añadidas por cargadores especializados 
    (p. ej., ``language``, ``detected_encoding`` o ``source_hint``) se conservan 
    textualmente para que los usuarios finales puedan confiar en ellas. Las 
    entradas de contabilidad interna como ``loader_context`` **no deben** estar 
    presentes una vez que la API pública genere el documento.

    Atributos
    ----------
    content:
        Representación en texto plano de todo el documento con marcadores de 
        página cuando estén disponibles.
    metadata:
        Diccionario que cumple el contrato descrito anteriormente.
    pages:
        Lista de cadenas de páginas sin decoración, listas para ser 
        posprocesadas con limpiadores o divisores.
    tables:
        Cargas útiles de tablas específicas del cargador. Vacío cuando no hay 
        tablas disponibles.
    images:
        Referencias a imágenes extraídas. A menudo, rutas del sistema de archivos 
        generadas por :class:`~data.preprocessing.pdf_loader.PDFResourceExporter`.
    sections:
        Mapeo opcional utilizado por etapas de nivel superior para anotar secciones 
        lógicas (p. ej., resumen ejecutivo, anexos).
    chunks:
        Colección de instancias de ``langchain`` :class:`~langchain_core.documents.Document` 
        generadas por la etapa divisora.
    """

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

