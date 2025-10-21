# data/models/document.py

from typing import List, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from langchain.schema import Document as LCDocument

class Document:
    """
    Representa un documento cargado desde PDF, Word o texto plano.
    Contiene texto completo, páginas y tablas, además de una lista de chunks
    compatibles con LangChain lista para generar embeddings.
    """
    def __init__(
        self,
        content: str,
        metadata: dict,
        pages: list = None,
        tables: list = None,
        images: list = None,
        sections: Optional[Dict[str, str]] = None
    ):
        self.content = content                  # Texto completo
        self.metadata = metadata                # Info del archivo
        self.pages = pages or []                # Texto por páginas
        self.tables = tables or []              # Tablas detectadas
        self.images = images or []              # Imágenes extraídas
        self.sections = sections or {}          # Secciones detectadas {id: texto}
        self.chunks: List["LCDocument"] = []   # Resultado del Splitter en LCDocument

    def __repr__(self):
        return (
            "Document("
            f"pages={len(self.pages)}, "
            f"tables={len(self.tables)}, "
            f"sections={len(self.sections)}, "
            f"chunks={len(self.chunks)}"
            ")"
        )
    
    def as_langchain_documents(self) -> List["LCDocument"]:
        """Devuelve los chunks como ``langchain.schema.Document`` listos para embeddings."""

        return list(self.chunks)

