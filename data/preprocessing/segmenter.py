"""Segmentador que detecta secciones de informe en texto preprocesado."""

import re
from io import StringIO
from typing import Dict, List, Tuple
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
        total_sections = len(getattr(self.loader, "sections", []) or [])
        log_info(
            f"Segmenter inicializado (tipo={tipo}, fuzzy={fuzzy}, secciones={total_sections})"
        )

    # ----------------------------------------------------------
    # FUNCIÓN PRINCIPAL
    # ----------------------------------------------------------
    def segment_document(self, document: Document) -> Document:
        """
        Divide el contenido limpio del documento en secciones.
        Devuelve un nuevo Document con atributo `sections` (dict ordenado según JSON).
        """
        if document is None:
            log_warn("Documento nulo recibido para segmentación.")
            return Document()

        if not getattr(document, "content", None):
            log_warn("Documento vacío recibido para segmentación.")
            # Si no existe documento devolvemos un contenedor mínimo
            empty = Document(content="")
            empty.sections = {}
            return empty

        content = getattr(document, "content", "")
        if not isinstance(content, str):
            content = str(content or "")

        if not content.strip():
            log_warn("Documento sin contenido recibido para segmentación.")
            setattr(document, "sections", {})
            return document

        text = content
        sections = self._segment_text(text)

        if not sections:
            log_warn(
                "Segmenter no detectó encabezados; devolviendo secciones vacías para activar el fallback."
            )
            sections = {}
            metadata = getattr(document, "metadata", None)
            if isinstance(metadata, dict):
                metadata["segmenter_missing_sections"] = True

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
        stream = StringIO(text)
        offset = 0
        hits: List[Tuple[int, str, int, bool, str]] = []  # (..., residual)

        patterns_map = getattr(self.loader, "patterns", {})
        if not isinstance(patterns_map, dict):
            patterns_map = {}

        while True:
            raw_line = stream.readline()
            if raw_line == "":
                break

            line_length = len(raw_line)
            stripped = raw_line.strip()

            if stripped:
                match = self.loader.identify_section(stripped)
                if match:
                    sec_id = match[0]

                    leading_ws = len(raw_line) - len(raw_line.lstrip())
                    heading_start = offset + leading_ws

                    heading_end = None
                    match_end = None
                    for pattern in patterns_map.get(sec_id, []):
                        if not hasattr(pattern, "search"):
                            continue
                        found = pattern.search(stripped)
                        if found:
                            heading_end = heading_start + found.end()
                            match_end = found.end()
                            break

                    if heading_end is None:
                        heading_end = offset + line_length
                        match_end = len(stripped)

                    content_start = heading_end
                    while (
                        content_start < offset + line_length
                        and text[content_start] not in "\r\n"
                        and text[content_start] in " \t:-–—.•·"
                    ):
                        content_start += 1

                    residual = (
                        stripped[match_end:].lstrip(" \t:-–—.•·")
                        if match_end is not None
                        else ""
                    )
                    has_inline_content = bool(residual)

                    hits.append((heading_start, sec_id, content_start, has_inline_content, residual))

            offset += line_length

        if not hits:
            log_warn("No se detectaron secciones.")
            return{}

        candidates: Dict[str, List[int]] = {}
        for idx, (_, sec_id, _, _, _) in enumerate(hits):
            candidates.setdefault(sec_id, []).append(idx)

        chosen_indices: List[int] = []
        for indices in candidates.values():
            pure = [i for i in indices if not hits[i][3]]
            if pure:
                chosen_indices.append(pure[-1])
            else:
                chosen_indices.append(indices[0])

        ordered_indices = sorted(chosen_indices, key=lambda i: hits[i][0])

        segments: Dict[str, str] = {}
        for pos, index in enumerate(ordered_indices):
            heading_start, sec_id, content_start, _, residual = hits[index]
            next_start = (
                hits[ordered_indices[pos + 1]][0]
                if pos + 1 < len(ordered_indices)
                else len(text)
            )
            section_text = text[content_start:next_start].strip()
            if residual and section_text.startswith(residual):
                residual = ""
            if residual:
                section_text = (
                    f"{residual}\n{section_text}".strip()
                    if section_text
                    else residual
                )
            segments[sec_id] = section_text

        # Asegurar que existan todas las secciones del JSON
        for sid in getattr(self.loader, "sections", []) or []:
            if sid not in segments:
                segments[sid] = ""

        return segments

