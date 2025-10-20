# data/preprocessing/segmenter.py

import re
from typing import Dict, List
from core.logger import log_info, log_warn
from data.models.document import Document
from data.criteria.section_loader import SectionLoader

class Segmenter:
    """
    Segmenta el texto limpio de un informe en secciones según patrones del SectionLoader.
    Puede trabajar con informes institucionales o de política nacional.
    """

    def __init__(self, tipo: str = "institucional", fuzzy: bool = True):
        self.tipo = tipo
        self.loader = SectionLoader(tipo=tipo, fuzzy=fuzzy)
        log_info(f"Segmenter inicializado (tipo={tipo}, fuzzy={fuzzy})")

    # ----------------------------------------------------------
    # FUNCIÓN PRINCIPAL
    # ----------------------------------------------------------
    def segment_document(self, document: Document) -> Document:
        """
        Divide el contenido limpio del documento en secciones.
        Devuelve un nuevo Document con atributo `sections` (dict ordenado según JSON).
        """
        if not document or not document.content:
            log_warn("Documento vacío recibido para segmentación.")
            document.sections = {}
            return document

        text = document.content
        sections = self._segment_text(text)

        # Guardar las secciones dentro del documento
        document.sections = sections


        log_info(f"Segmentación completada ({len(sections)} secciones).")
        return document

    # ----------------------------------------------------------
    # NÚCLEO DE SEGMENTACIÓN
    # ----------------------------------------------------------
    def _segment_text(self, text: str) -> Dict[str, str]:
        """
        Detecta títulos de secciones en el texto y separa el contenido.
        """
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        matches: List[tuple] = []

        for i, line in enumerate(lines):
            match = self.loader.identify_section(line)
            if match:
                sec_id, method, score, line_match = match
                matches.append((i, sec_id))

        if not matches:
            log_warn("No se detectaron secciones.")
            return {}

        # Evitar duplicados → conservar última aparición
        seen = {}
        for idx, sec_id in matches:
            seen[sec_id] = idx
        ordered = sorted(seen.items(), key=lambda x: x[1])

        # Extraer segmentos
        segments: Dict[str, str] = {}
        for i, (sec_id, start) in enumerate(ordered):
            end = ordered[i + 1][1] if i + 1 < len(ordered) else len(lines)
            section_text = "\n".join(lines[start + 1:end]).strip()
            segments[sec_id] = section_text

        # Asegurar que existan todas las secciones del JSON
        for sid in self.loader.sections:
            if sid not in segments:
                segments[sid] = ""

        return segments

