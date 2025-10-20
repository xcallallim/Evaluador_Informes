# data/models/document.py

from typing import Optional, List, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from langchain.schema import Document as LCDocument

class Document:
    """
    Representa un documento cargado desde PDF, Word o texto plano.
    Contiene texto completo, páginas, tablas, imágenes y ahora chunks.
    """
    def __init__(
        self,
        content: str,
        metadata: dict,
        pages: list = None,
        tables: list = None,
        images: list = None,
        sections: Optional[Dict[str, str]] = None,       
        chunks: list = None          
    ):
        self.content = content
        self.metadata = metadata
        self.pages = pages or []
        self.tables = tables or []
        self.images = images or []
        self.sections = sections or {}         # Secciones detectadas {id: texto}
        self.chunks: List["LCDocument"] = []   # Resultado del Splitter
    def __repr__(self):
        return (
            "Document("
            f"pages={len(self.pages)}, "
            f"tables={len(self.tables)}, "
            f"sections={len(self.sections)}, "
            f"chunks={len(self.chunks)}"
            ")"
        )

