# data/preprocessing/section_loader.py
 
import os
import json
import re
from typing import List, Dict, Optional, Tuple

from core.logger import log_info, log_warn, log_error
from core.config import CRITERIA_DIR  # Ruta donde están los JSON

try:
    # Opcional: si no está instalado, seguimos con modo sin fuzzy
    from rapidfuzz import fuzz, process  # type: ignore
    _HAS_FUZZ = True
except Exception:
    _HAS_FUZZ = False


def _normalize(s: str) -> str:
    """Normaliza para comparaciones tolerantes (solo para matching interno)."""
    import unicodedata
    s = (s or "").strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def _prefix_num_pattern() -> str:
    """
    Prefijo tolerante a numeraciones / rótulos:
    - '1. ', '2.1 ', '3 - ', 'Sección 4: ', 'CAPÍTULO II - ', 'Cap. 3: '
    """
    roman = r"(?:[IVXLCDM]+)"
    p = (
        r"^\s*(?:"
        r"(?:cap(?:itulo|\.?)\s*(?:\d+|" + roman + r")\s*[:\-]\s*)|"
        r"(?:secci[oó]n\s*\d+\s*[:\-]\s*)|"
        r"(?:\d+(?:\.\d+)*\s*[:\-\.]\s*)"
        r")?"
    )
    return p


class SectionLoader:
    """
    Carga secciones desde JSON según tipo de informe:
    - institucional -> secciones_institucional.json
    - politica      -> secciones_politica.json

    Expone:
      - get_sections(): List[str]  # claves internas (snake_case)
      - identify_section(line) -> Optional[Tuple[str, str, float, str]]
        Devuelve (section_key, match_type, score, detected_text)
    """

    def __init__(self, report_type: str = "institucional"):
        self.report_type = report_type.lower().strip()
        self.sections_file = self._select_file()
        # schema: Dict[str, Dict[str, Any]]
        self.schema: Dict[str, Dict[str, List[str] | str]] = self._load_schema()
        self.sections: List[str] = list(self.schema.keys())
        self.patterns: Dict[str, List[re.Pattern]] = self._build_patterns()
        # Para fuzzy: candidatos por sección (title + aliases)
        self._fuzzy_index: Dict[str, List[str]] = self._build_fuzzy_index()

        log_info(
            f"SectionLoader listo. Tipo='{self.report_type}', "
            f"secciones={len(self.sections)}, fuzzy={'ON' if _HAS_FUZZ else 'OFF'}"
        )

    # ----------- carga / construcción -----------

    def _select_file(self) -> str:
        mapping = {
            "institucional": "secciones_institucional.json",
            "politica": "secciones_politica.json",
        }
        if self.report_type not in mapping:
            raise ValueError(f"Tipo de informe no válido: {self.report_type}")

        filepath = os.path.join(CRITERIA_DIR, mapping[self.report_type])
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No se encontró el JSON: {filepath}")

        return filepath

    def _load_schema(self) -> Dict[str, Dict[str, List[str] | str]]:
        """
        Espera un JSON con este formato:
        {
          "resumen_ejecutivo": {
            "title": "Resumen Ejecutivo",
            "aliases": ["Resumen", "Resumen del Informe"],
            "keywords": ["síntesis", "resumen general"]
          },
          ...
        }
        """
        try:
            with open(self.sections_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("El JSON debe ser un objeto con claves por sección.")
            # Validación mínima
            for k, v in data.items():
                if not isinstance(v, dict) or "title" not in v:
                    raise ValueError(f"Sección '{k}' inválida: falta 'title'")
                v.setdefault("aliases", [])
                v.setdefault("keywords", [])
            log_info(f"Secciones cargadas para '{self.report_type}'. Total: {len(data)}.")
            return data
        except Exception as e:
            log_error(f"Error al cargar secciones desde {self.sections_file}: {e}")
            return {}

    def _build_patterns(self) -> Dict[str, List[re.Pattern]]:
        """
        Construye regex por sección para (title + aliases) con tolerancia a numeración/mayúsculas.
        """
        compiled: Dict[str, List[re.Pattern]] = {}
        prefix = _prefix_num_pattern()

        for key, obj in self.schema.items():
            variants = [obj["title"]] + list(obj.get("aliases", []))  # type: ignore
            pats: List[re.Pattern] = []
            for v in variants:
                esc = re.escape(v)
                # ^(prefijo-num)? + titulo/alias + fin de palabra
                rx = rf"{prefix}{esc}\b"
                pats.append(re.compile(rx, flags=re.IGNORECASE | re.MULTILINE))
            compiled[key] = pats

        log_info(f"Patrones regex construidos para {len(compiled)} secciones.")
        return compiled

    def _build_fuzzy_index(self) -> Dict[str, List[str]]:
        idx: Dict[str, List[str]] = {}
        for key, obj in self.schema.items():
            variants = [obj["title"]] + list(obj.get("aliases", []))  # type: ignore
            # Normalizamos para fuzzy
            idx[key] = list({ _normalize(v) for v in variants })
        return idx

    # ----------- API pública -----------

    def get_sections(self) -> List[str]:
        return self.sections

    def get_patterns(self) -> Dict[str, List[re.Pattern]]:
        return self.patterns

    def identify_section(
        self,
        line: str,
        *,
        fuzzy_threshold: int = 86,
        keyword_boost: bool = True,
        max_line_len: int = 160,
    ) -> Optional[Tuple[str, str, float, str]]:
        """
        Intenta identificar si 'line' es un título de sección.

        Retorna:
          (section_key, match_type, score, detected_text)
            - match_type: 'regex' | 'fuzzy' | 'keywords'
            - score: 100 para regex/keywords, o score de fuzzy
            - detected_text: texto original que disparó la coincidencia

        None si no parece un título.
        """
        if not isinstance(line, str):
            return None

        text = line.strip()
        if not text or len(text) > max_line_len:
            return None

        # 1) Regex exacto (con tolerancia a numeración / mayúsculas)
        for key, rx_list in self.patterns.items():
            for rx in rx_list:
                m = rx.search(text)
                if m:
                    return (key, "regex", 100.0, text)

        # 2) Fuzzy (si está disponible)
        if _HAS_FUZZ:
            # Quitar prefijos numéricos antes de fuzzy
            pref = re.compile(_prefix_num_pattern(), flags=re.IGNORECASE)
            stripped = pref.sub("", text).strip()
            norm_line = _normalize(stripped)

            best_key = None
            best_score = 0.0
            for key, candidates in self._fuzzy_index.items():
                # mejor score contra cualquiera de los candidatos de esa sección
                for cand in candidates:
                    score = fuzz.token_set_ratio(norm_line, cand)
                    if score > best_score:
                        best_score = score
                        best_key = key

            if best_key and best_score >= fuzzy_threshold:
                return (best_key, "fuzzy", float(best_score), text)

        # 3) Keywords (como refuerzo: todas las keywords deben estar o al menos 2)
        if keyword_boost:
            norm_text = _normalize(text)
            for key, obj in self.schema.items():
                kws = [ _normalize(k) for k in obj.get("keywords", []) ]  # type: ignore
                if not kws:
                    continue
                # Heurística: que aparezca al menos 1–2 keywords y que no sea línea larguísima
                hits = sum(1 for k in kws if k and k in norm_text)
                if hits >= max(1, min(2, len(kws))):
                    return (key, "keywords", 100.0, text)

        return None
