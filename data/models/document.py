# data/models/document.py

from typing import Optional, List, Dict

class Document:
    """
    Representa un documento cargado desde PDF, Word o texto plano.
    Contiene texto completo, páginas y tablas.
    """
    def __init__(
        self,
        content: str,
        metadata: dict,
        pages: list = None,
        tables: list = None,
        images: list = None
    ):
        self.content = content                  # Texto completo
        self.metadata = metadata                # Info del archivo
        self.pages = pages or []                # Texto por páginas
        self.tables = tables or []              # Tablas detectadas
        self.images = images or []              # Imágenes extraídas

    def __repr__(self):
        return f"Document(pages={len(self.pages)}, tables={len(self.tables)})"