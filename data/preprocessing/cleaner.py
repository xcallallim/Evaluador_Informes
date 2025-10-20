# data/preprocessing/cleaner.py

import re
import unicodedata
from typing import List, Dict, Tuple, Optional
from collections import Counter

from core.logger import log_info, log_warn, log_error
from core.utils import normalize_whitespace
from core.config import CUSTOM_HEADERS, CUSTOM_FOOTERS

try:
    from rapidfuzz import fuzz
    _RAPIDFUZZ = True
except Exception:
    _RAPIDFUZZ = False


class Cleaner:

    def __init__(
        self,
        remove_headers: bool = True,
        remove_page_numbers: bool = True,
        use_custom_headers: bool = True,
    ):
        self.remove_headers = remove_headers
        self.remove_page_numbers = remove_page_numbers
        self.use_custom_headers = use_custom_headers

    # =========================================================
    # API PRINCIPAL
    # =========================================================
    def clean(self, text: str, return_report: bool = False) -> str | Tuple[str, Dict]:
        """
        Punto de entrada principal del limpiador (texto plano).
        Limpia texto completo con opción de devolver reporte.
        """
        if not isinstance(text, str) or not text.strip():
            log_warn("Texto vacío recibido para limpieza.")
            return ("", {}) if return_report else ""

        pages = self._split_pages(text)
        log_info(f"{len(pages)} páginas detectadas para limpieza.")

        # Inferir si parece OCR para decidir fuzzy matching
        is_ocr_like = getattr(self, "_force_ocr", self._infer_is_ocr_like(pages))

        # Detectar candidatos repetidos en primeras/últimas líneas
        header_candidates, footer_candidates = self._collect_candidates(pages)

        # Umbral: más del 60% de páginas (seguro)
        min_repeats = max(3, int(0.6 * len(pages)))

        # Cluster de repetidos (fuzzy para OCR, exacto para digital/DOCX)
        repeated_headers = self._cluster_repeated_lines(
            header_candidates, use_fuzzy=is_ocr_like, similarity=70, min_repeats=min_repeats,
            normalizer=self._norm_line_edges  # ¡clave!
        )
        repeated_footers = self._cluster_repeated_lines(
            footer_candidates, use_fuzzy=is_ocr_like, similarity=70, min_repeats=min_repeats,
            normalizer=self._norm_line_edges  # ¡clave!
        )



        # Contadores del reporte
        report = {
            "headers_removed": 0,
            "footers_removed": 0,
            "page_numbers_removed": 0,
            "source_lines_removed": 0,
            "digital_sign_removed": 0,
            "other_noise_removed": 0,
        }

        cleaned_pages = []
        for idx, page_text in enumerate(pages, start=1):
            cleaned, stats = self._clean_page(
                page_text=page_text,
                repeated_headers=repeated_headers,
                repeated_footers=repeated_footers,
                is_ocr_like=is_ocr_like,
            )
            cleaned_pages.append(f"=== PAGE {idx} ===\n{cleaned}")
            for k in report:
                report[k] += stats.get(k, 0)

        final_text = "\n\n".join(cleaned_pages).strip()

        # Resumen global
        log_info("✅ LIMPIEZA COMPLETA")
        log_info(f"- Encabezados repetidos eliminados: {report['headers_removed']}")
        log_info(f"- Pies de página eliminados: {report['footers_removed']}")
        log_info(f"- Números de página eliminados: {report['page_numbers_removed']}")
        log_info(f"- Líneas tipo 'Fuente/Elaboración/URL' eliminadas: {report['source_lines_removed']}")
        log_info(f"- Firmas digitales eliminadas: {report['digital_sign_removed']}")
        log_info(f"- Otras líneas de ruido eliminadas: {report['other_noise_removed']}")

        return (final_text, report) if return_report else final_text

    def clean_document(self, document, return_report: bool = False):
        """
        Limpia un objeto Document (de loader) conservando metadata.
        Usa metadata["is_ocr"] para mejorar la limpieza.
        """
        text = document.content or ""

        # Detectar si proviene de OCR según metadata del loader
        is_ocr = document.metadata.get("is_ocr", False)

        # Pasar el valor de OCR al método clean mediante flag interno temporal
        self._force_ocr = is_ocr  # guardamos una variable interna solo para esta ejecución

        result = self.clean(text, return_report=return_report)

        # limpiar flag interno
        if hasattr(self, "_force_ocr"):
            del self._force_ocr

        return result


    # =========================================================
    # BLOQUES DE LIMPIEZA
    # =========================================================
    def _clean_page(
        self,
        page_text: str,
        repeated_headers: List[str],
        repeated_footers: List[str],
        is_ocr_like: bool,
    ) -> Tuple[str, Dict]:
        """
        Limpia una página aplicando todos los bloques. Devuelve texto + estadísticas.
        """
        stats = {
            "headers_removed": 0,
            "footers_removed": 0,
            "page_numbers_removed": 0,
            "source_lines_removed": 0,
            "digital_sign_removed": 0,
            "other_noise_removed": 0,
        }

        # Normalización ligera
        page_text = normalize_whitespace(page_text)

        # 1) Eliminar encabezados/pies repetidos (según clusters previos)
        if self.remove_headers and (repeated_headers or repeated_footers):
            page_text, rm_h, rm_f = self._remove_repeated_headers_footers(
                page_text, repeated_headers, repeated_footers, is_ocr_like
            )
            stats["headers_removed"] += rm_h
            stats["footers_removed"] += rm_f

        # 2) Quitar números de página
        if self.remove_page_numbers:
            page_text, rm_pg = self._remove_page_numbers(page_text)
            stats["page_numbers_removed"] += rm_pg

        # 3) Quitar líneas sueltas tipo "Fuente:", "Elaboración", URLs, notas mecánicas
        page_text, rm_sources, rm_dsig = self._remove_source_lines_and_signs(page_text)
        stats["source_lines_removed"] += rm_sources
        stats["digital_sign_removed"] += rm_dsig

        # 4) Limpieza de ruido residual (líneas basura cortas o bloques ocr)
        page_text, rm_noise = self._remove_other_noise(page_text)
        stats["other_noise_removed"] += rm_noise

        # 5) Compactación suave (sin borrar saltos de párrafo reales)
        page_text = self._compact_blank_lines(page_text)

        return page_text.strip(), stats

    # =========================================================
    # SOPORTE: Páginas, candidatos y clustering
    # =========================================================
    def _split_pages(self, text: str) -> List[str]:
        """
        Divide el texto completo en páginas reales detectando el tag '=== PAGE X ===' del loader.
        Si no existe, intenta dividir automáticamente cada ~5000 caracteres.
        """
        if "=== PAGE" in text:
            parts = text.split("=== PAGE")
            # parts[0] puede estar vacío o tener preámbulo; reconstruimos
            pages = []
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                # restauramos el tag para mantener referencia dentro de cada página
                if not p.startswith("==="):
                    p = "=== PAGE " + p
                # quitamos la cabecera "=== PAGE N ===" dentro del contenido de la página
                page_body = re.sub(r"^=== PAGE\s+\d+\s*===\s*\n?", "", p, flags=re.IGNORECASE)
                pages.append(page_body)
            return pages
        else:
            log_warn("No se detectaron separadores de páginas. División estimada por longitud.")
            return [text[i:i + 5000] for i in range(0, len(text), 5000)]

    def _infer_is_ocr_like(self, pages: List[str]) -> bool:
        """
        Heurística simple para inferir si el texto parece provenir de OCR:
        - presencia de muchos caracteres raros,
        - alta tasa de líneas muy cortas con errores comunes.
        """
        if len(pages) == 0:
            return False
        sample = "\n".join(pages[: min(3, len(pages))]).lower()
        # patrones típicos de OCR: letras pegadas, confusión rn/m, l/1, acentos rotos
        weird = len(re.findall(r"[^\w\sáéíóúñüÁÉÍÓÚÑÜ:/(),.-]", sample))
        short_lines = sum(1 for l in sample.split("\n") if 0 < len(l.strip()) < 5)
        return (weird > 10) or (short_lines > 10)

    def _collect_candidates(self, pages: List[str]) -> Tuple[List[str], List[str]]:
        """
        Extrae primeras y últimas 3 líneas de cada página como posibles header/footer.
        Normaliza eliminando números de página al inicio/fin para robustez.
        """
        headers, footers = [], []
        for p in pages:
            lines = [l.strip() for l in p.split("\n") if l.strip()]
            if not lines:
                continue
            # primeras 3 (encabezado) → quitar números al INICIO
            for line in lines[:3]:
                headers.append(self._strip_leading_page_num(line))
            # últimas 3 (pie) → quitar números al FINAL
            for line in lines[-3:]:
                footers.append(self._strip_trailing_page_num(line))
        return headers, footers


    def _cluster_repeated_lines(
        self,
        lines: List[str],
        use_fuzzy: bool,
        similarity: int,
        min_repeats: int,
        normalizer=None,
    ) -> List[str]:
        """
        Agrupa líneas repetidas. Usa 'normalizer' para comparar (por defecto _norm_line).
        """
        if not lines:
            return []
        if normalizer is None:
            normalizer = self._norm_line

        norm_map = [(normalizer(x), x) for x in lines]

        if use_fuzzy and _RAPIDFUZZ:
            remaining = norm_map[:]
            clusters = []
            while remaining:
                base_norm, _ = remaining[0]
                cluster = [(nrm, raw) for (nrm, raw) in remaining if fuzz.ratio(base_norm, nrm) >= similarity]
                rest = [(nrm, raw) for (nrm, raw) in remaining if fuzz.ratio(base_norm, nrm) < similarity]
                if len(cluster) >= min_repeats:
                    raw_choices = [r for _, r in cluster]
                    template = Counter(raw_choices).most_common(1)[0][0]
                    clusters.append(template)
                remaining = rest
            return clusters
        else:
            counts = Counter([n for n, _ in norm_map])
            templates = []
            for nrm, cnt in counts.items():
                if cnt >= min_repeats:
                    for n, raw in norm_map:
                        if n == nrm:
                            templates.append(raw)
                            break
            return templates


    def _norm_line(self, s: str) -> str:
        """
        Normaliza una línea para comparación: minúsculas, sin acentos, espacios colapsados.
        """
        s = s.lower()
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
        s = re.sub(r"\s+", " ", s).strip()
        return s

    # =========================================================
    # APLICADORES: quitar headers/footers, números, fuentes, etc.
    # =========================================================
    def _remove_repeated_headers_footers(
        self,
        page_text: str,
        repeated_headers: List[str],
        repeated_footers: List[str],
        is_ocr_like: bool,
    ) -> Tuple[str, int, int]:
        """
        Elimina líneas que coinciden con encabezados/pies repetidos.
        - OCR: fuzzy match
        - Digital/DOCX: exact match
        """
        lines = page_text.split("\n")
        new_lines = []
        rm_h = 0
        rm_f = 0

        rep_h_norm = [self._norm_line_edges(x) for x in repeated_headers]
        rep_f_norm = [self._norm_line_edges(x) for x in repeated_footers]

        for i, line in enumerate(lines):
            norm_edge = self._norm_line_edges(line)

            def is_match(norm_line: str, rep_norm_list: List[str]) -> bool:
                if not rep_norm_list:
                    return False
                if is_ocr_like and _RAPIDFUZZ:
                    return max((fuzz.ratio(norm_line, r) for r in rep_norm_list), default=0) >= 70
                else:
                    return norm_line in rep_norm_list

            # HEADER: primeras 3 líneas
            if i <= 2 and is_match(norm_edge, rep_h_norm):
                rm_h += 1
                continue

            # FOOTER: últimas 3 líneas
            if i >= len(lines) - 3 and is_match(norm_edge, rep_f_norm):
                rm_f += 1
                continue

            new_lines.append(line)

        return "\n".join(new_lines), rm_h, rm_f


    def _remove_page_numbers(self, page_text: str) -> Tuple[str, int]:
        """
        Elimina patrones típicos de numeración de página:
        """
        removed = 0
        lines = page_text.split("\n")
        out = []
        for idx, line in enumerate(lines):
            l = line.strip().lower()
            at_edge = (idx <= 2) or (idx >= len(lines) - 3)  # solo bordes de página

            # Patrones comunes de número de página (solo si está en bordes)
            if at_edge and re.match(r"^(p[aá]g(?:ina)?\s*\d+\s*(de\s*\d+)?)$", l):
                removed += 1
                continue
            if at_edge and re.match(r"^-\s*\d+\s*-$", l):
                removed += 1
                continue
            if at_edge and re.match(r"^\d{1,4}$", l) and len(l) <= 4:
                removed += 1
                continue

            out.append(line)

        return "\n".join(out), removed

    def _remove_source_lines_and_signs(self, page_text: str) -> Tuple[str, int, int]:
        """
        Elimina líneas sueltas tipo 'Fuente: ...', 'Elaboración propia', URLs sueltas,
        y firmas digitales / sellos (si están en líneas propias).
        Devuelve: (texto, removed_sources, removed_digital_sign)
        """
        patterns_sources = [
            r"^fuente[:\s]",                 # Fuente:
            r"^elaboraci[oó]n(\s+propia|:)", # Elaboración propia / Elaboración:
            r"^elaborado\s+por",             # Elaborado por
            r"^nota[:\s]",                   # Nota: (cuando es suelta)
            r"^(https?://|www\.)",           # URLs
        ]
        # Firmas digitales / sellos
        patterns_dsig = [
            r"firmad[oa]\s+digitalmente",
            r"firma\s+electr[oó]nica",
            r"digital\s+signature",
            r"\bhash\b",
            r"\bx\.?509\b",
            r"\breniec\b",
            r"\bdni[\s-]?e\b",
            r"documento\s+generado\s+autom[aá]ticamente",
            r"certificado\s+digital",
        ]

        removed_sources = 0
        removed_dsig = 0
        out_lines = []

        for line in page_text.split("\n"):
            l = line.strip().lower()

            # Línea tipo fuente/elaboración/url
            if any(re.match(p, l) for p in patterns_sources) and len(l) < 140:
                removed_sources += 1
                continue

            # Línea firma digital/sello
            if any(re.search(p, l) for p in patterns_dsig) and len(l) < 200:
                removed_dsig += 1
                continue

            out_lines.append(line)

        return "\n".join(out_lines), removed_sources, removed_dsig

    def _remove_other_noise(self, page_text: str) -> Tuple[str, int]:
        """
        Elimina ruido OCR o líneas residuales poco informativas:
          - bloques de símbolos, cajas, etc.
          - líneas extremadamente cortas sin valor
        """
        removed = 0
        out = []
        for line in page_text.split("\n"):
            l = line.strip()

            # muchas cajas/símbolos (ruido OCR)
            if re.search(r"[█▓▒■□◆◇►◄▪•●○◘◙◆★☆✦✧]+", l):
                removed += 1
                continue

            # líneas de 1-2 caracteres no alfanuméricos
            if len(l) <= 2 and not re.match(r"[a-zA-Z0-9]", l):
                removed += 1
                continue

            out.append(line)
        return "\n".join(out), removed

    def _compact_blank_lines(self, page_text: str) -> str:
        """
        Compacta líneas en blanco múltiples a una sola; mantiene saltos de párrafo.
        """
        # Normaliza espacios internos de cada línea
        lines = [re.sub(r"[ \t]+", " ", l).rstrip() for l in page_text.split("\n")]
        text = "\n".join(lines)
        # Colapsa múltiples líneas vacías
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text
    
    def _collect_candidates(self, pages: List[str]) -> Tuple[List[str], List[str]]:
        """
        Extrae primeras y últimas líneas como candidatos de header/footer.
        Además refuerza con patrones CUSTOM_HEADERS y CUSTOM_FOOTERS desde config.py.
        """
        headers, footers = [], []

        # 1. Candidatos naturales (primeras/últimas líneas de cada página)
        for p in pages:
            lines = [l.strip() for l in p.split("\n") if l.strip()]
            if not lines:
                continue
            headers.extend(lines[:3])
            footers.extend(lines[-3:])

        # 2. Refuerzo: añadir patrones custom si están en el texto
        if self.use_custom_headers:
            for pattern in CUSTOM_HEADERS:
                for p in pages:
                    for line in p.split("\n")[:5]:  # primeras líneas
                        if re.search(pattern, line, flags=re.IGNORECASE):
                            headers.append(line.strip())

        if self.use_custom_headers:
            for pattern in CUSTOM_FOOTERS:
                for p in pages:
                    for line in p.split("\n")[-5:]:  # últimas líneas
                        if re.search(pattern, line, flags=re.IGNORECASE):
                            footers.append(line.strip())

        return headers, footers
    
    def _strip_trailing_page_num(self, s: str) -> str:
        """
        Quita numeración final típica de pie de página. Conserva el resto del texto.
        """
        s = re.sub(r"(p[aá]g(?:ina)?\s*\d+\s*(de\s*\d+)?)\s*$", "", s, flags=re.IGNORECASE).strip()
        s = re.sub(r"[\s\-–—]*\d{1,4}\s*$", "", s).strip()
        return s
    
    def _strip_trailing_page_num(self, s: str) -> str:
        """
        Quita numeración final típica de pie de página:
        - '... - 2', '... – 12', '... — 7'
        - '... 2'
        - '... Página 2' / '... Página 2 de 140'
        """
        s = re.sub(r"(p[aá]g(?:ina)?\s*\d+\s*(de\s*\d+)?)\s*$", "", s, flags=re.IGNORECASE).strip()
        s = re.sub(r"[\s\-–—]*\d{1,4}\s*$", "", s).strip()
        return s

    def _strip_leading_page_num(self, s: str) -> str:
        """
        Quita numeración al inicio típica de encabezado:
        - '2 ...', '12 — ...', '141 - ...'
        - 'Página 2 ...'
        """
        s = re.sub(r"^\s*(p[aá]g(?:ina)?\s*\d+\s*(de\s*\d+)?)[\s\-–—]*", "", s, flags=re.IGNORECASE).strip()
        s = re.sub(r"^\s*\d{1,4}[\s\-–—]*", "", s).strip()
        return s

    def _strip_edge_page_nums(self, s: str) -> str:
        """
        Normaliza una línea removiendo números de página al inicio y al final.
        """
        return self._strip_trailing_page_num(self._strip_leading_page_num(s))

    def _norm_line_edges(self, s: str) -> str:
        """
        Normalización para clustering/match de headers/footers con números variables:
        1) Quita números al inicio/fin
        2) Minúsculas, sin acentos, espacios colapsados
        """
        s = self._strip_edge_page_nums(s)
        return self._norm_line(s)


