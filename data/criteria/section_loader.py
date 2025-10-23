"""Carga la definición de secciones y construye patrones de detección."""
 
import copy
import inspect
import os
import json
import re
from difflib import SequenceMatcher
from typing import List, Dict, Optional, Tuple, Any, Iterable

from core.logger import log_info, log_warn, log_error
from core.config import CRITERIA_DIR  # Ruta donde están los JSON

try:
    # Opcional: si no está instalado, seguimos con modo basado en SequenceMatcher
    from rapidfuzz import fuzz, process  # type: ignore
    _FUZZ_STRATEGY = "rapidfuzz"
except Exception:
    _FUZZ_STRATEGY = "sequence"
    fuzz = None  # type: ignore
    process = None  # type: ignore

_HAS_FUZZ = True


def _normalize(s: str, *, strip_accents: bool) -> str:
    """Normaliza cadenas para comparaciones tolerantes."""
    s = (s or "").strip()
    if strip_accents:
        import unicodedata

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


def _token_set_score(a: str, b: str, **kwargs: Any) -> float:
    """Wrapper tolerante a stubs de ``rapidfuzz.fuzz.token_set_ratio``."""

    if fuzz is None:
        return SequenceMatcher(None, a, b).ratio() * 100

    try:
        return float(fuzz.token_set_ratio(a, b, **kwargs))
    except TypeError:
        return float(fuzz.token_set_ratio(a, b))


def _can_use_process_extract() -> bool:
    if fuzz is None or process is None:
        return False

    try:
        signature = inspect.signature(fuzz.token_set_ratio)
    except (TypeError, ValueError):
        # Función C: asumimos soporte completo
        return True

    return "score_cutoff" in signature.parameters


_SCHEMA_CACHE: Dict[str, Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]] = {}
_PATTERN_CACHE: Dict[str, Dict[str, List[re.Pattern]]] = {}
_FUZZY_INDEX_CACHE: Dict[str, Dict[str, List[str]]] = {}
_FUZZY_CHOICES_CACHE: Dict[str, List[Tuple[str, str]]] = {}


class SectionLoader:
    """
    Carga secciones desde JSON según tipo de informe:
    - institucional -> secciones_institucional.json
    - politica      -> secciones_politica.json
    """

    def __init__(self, tipo: str = "institucional", fuzzy: bool = True):
        self.tipo = tipo.lower().strip()
        self.fuzzy = bool(fuzzy)
        self.sections_file = self._select_file()
        self._cache_key = self._make_cache_key()
        self.schema, self._schema_config = self._load_schema()
        self.sections: List[str] = list(self.schema.keys())
        self.patterns: Dict[str, List[re.Pattern]] = self._build_patterns()
        self._fuzzy_index: Dict[str, List[str]] = {}
        self._fuzzy_choices: List[Tuple[str, str]] = []
        self._fuzzy_cache: Dict[str, Tuple[Optional[str], float]] = {}

        self._fuzzy_enabled = False
        self._fuzzy_strategy = None
        self._use_process_extract = False
        if self.fuzzy:
            if not _HAS_FUZZ:
                log_warn(
                    "Modo fuzzy solicitado, pero el backend no está disponible. Se desactiva fuzzy."
                )
            else:
                self._fuzzy_enabled = True
                self._fuzzy_strategy = _FUZZ_STRATEGY
                if self._fuzzy_strategy != "rapidfuzz":
                    log_warn(
                        "Modo fuzzy activo con SequenceMatcher (rapidfuzz no disponible)."
                    )
                self._use_process_extract = (
                    self._fuzzy_strategy == "rapidfuzz" and _can_use_process_extract()
                )

        if self._fuzzy_enabled:
            self._fuzzy_index = self._build_fuzzy_index()
            self._fuzzy_choices = self._build_fuzzy_choices(self._fuzzy_index)

        fuzzy_state = (
            f"ON ({self._fuzzy_strategy})" if self._fuzzy_enabled else "OFF"
        )

        log_info(
            f"SectionLoader listo. Tipo='{self.tipo}', secciones={len(self.sections)}, fuzzy={fuzzy_state}"
        )

    # ----------- carga / construcción -----------

    def _select_file(self) -> str:
        """
        Selecciona el archivo JSON correcto según el tipo de informe.
        """
        file_name = f"secciones_{self.tipo}.json"
        file_path = os.path.join(CRITERIA_DIR, file_name)

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"No existe archivo de secciones para tipo '{self.tipo}': {file_path}"
            )

        return file_path


    def _make_cache_key(self) -> str:
        try:
            mtime = os.path.getmtime(self.sections_file)
            size = os.path.getsize(self.sections_file)
        except OSError:
            mtime = 0.0
            size = 0
        return f"{self.sections_file}:{mtime:.6f}:{size}"

    def _load_schema(self) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
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
        cache_key = self._cache_key
        if cache_key in _SCHEMA_CACHE:
            schema, config = _SCHEMA_CACHE[cache_key]
            return copy.deepcopy(schema), copy.deepcopy(config)
        
        try:
            with open(self.sections_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        except Exception as exc:
            log_error(
                f"No se pudo leer el archivo de secciones '{self.sections_file}': {exc}"
            )
            raise RuntimeError(
                f"No se pudo cargar la configuración de secciones para '{self.tipo}'."
            ) from exc

        if not isinstance(raw_data, dict):
            msg = (
                f"El JSON de secciones '{self.sections_file}' debe ser un objeto "
                "con claves por sección."
            )
            log_error(msg)
            raise ValueError(msg)

        config: Dict[str, Any] = {}
        schema: Dict[str, Dict[str, Any]] = {}
        for key, value in raw_data.items():
            if key.startswith("__"):
                if isinstance(value, dict):
                    config.update(value)
                continue

            if not isinstance(value, dict) or "title" not in value:
                raise ValueError(
                    f"Sección '{key}' inválida en '{self.sections_file}': falta 'title'"
                )

            value.setdefault("aliases", [])
            value.setdefault("keywords", [])
            if not isinstance(value.get("aliases"), list) or not isinstance(
                value.get("keywords"), list
            ):
                raise ValueError(
                    f"Sección '{key}' inválida en '{self.sections_file}': campos incorrectos"
                )

            schema[key] = value

        log_info(f"Secciones cargadas para '{self.tipo}'. Total: {len(schema)}.")
        _SCHEMA_CACHE[cache_key] = (copy.deepcopy(schema), copy.deepcopy(config))
        return copy.deepcopy(schema), copy.deepcopy(config)

    def _build_patterns(self) -> Dict[str, List[re.Pattern]]:
        """
        Construye regex por sección para (title + aliases) con tolerancia a numeración/mayúsculas.
        """
        cache_key = self._cache_key
        if cache_key in _PATTERN_CACHE:
            return _PATTERN_CACHE[cache_key]
        
        compiled: Dict[str, List[re.Pattern]] = {}
        prefix = _prefix_num_pattern()

        for key, obj in self.schema.items():
            variants = [obj["title"]] + list(obj.get("aliases", []))  # type: ignore
            pats: List[re.Pattern] = []
            for v in variants:
                esc = re.escape(v)
                # ^(prefijo-num)? + titulo/alias + fin de palabra
                rx = rf"{prefix}{esc}(?=\b|\W|$)"
                pats.append(re.compile(rx, flags=re.IGNORECASE | re.MULTILINE))
            compiled[key] = pats

        log_info(f"Patrones regex construidos para {len(compiled)} secciones.")
        _PATTERN_CACHE[cache_key] = compiled
        return compiled

    def _build_fuzzy_index(self) -> Dict[str, List[str]]:
        cache_key = self._cache_key
        if cache_key in _FUZZY_INDEX_CACHE:
            return copy.deepcopy(_FUZZY_INDEX_CACHE[cache_key])
        
        idx: Dict[str, List[str]] = {}
        for key, obj in self.schema.items():
            variants = [obj["title"]] + list(obj.get("aliases", []))  # type: ignore
            idx[key] = list({
                _normalize(v, strip_accents=True) for v in variants if isinstance(v, str)
            })

        _FUZZY_INDEX_CACHE[cache_key] = copy.deepcopy(idx)
        return copy.deepcopy(idx)

    def _build_fuzzy_choices(
        self, idx: Dict[str, List[str]]
    ) -> List[Tuple[str, str]]:
        cache_key = self._cache_key
        if cache_key in _FUZZY_CHOICES_CACHE:
            return list(_FUZZY_CHOICES_CACHE[cache_key])

        choices: List[Tuple[str, str]] = []
        for key, candidates in idx.items():
            for cand in candidates:
                choices.append((cand, key))

        _FUZZY_CHOICES_CACHE[cache_key] = list(choices)
        return list(choices)

    def _default_fuzzy_threshold(self) -> float:
        value = self._schema_config.get("default_fuzzy_threshold")
        if isinstance(value, (int, float)):
            return float(value)
        return 86.0

    def _section_fuzzy_threshold(self, section_key: str) -> Optional[float]:
        specific = self.schema.get(section_key, {}).get("fuzzy_threshold")
        if isinstance(specific, (int, float)):
            return float(specific)
        return None

    def _keyword_min_ratio(self, section_key: str, total_keywords: int) -> float:
        section = self.schema.get(section_key, {})
        specific = section.get("keyword_min_ratio")
        if isinstance(specific, (int, float)):
            return max(0.0, min(1.0, float(specific)))

        config_ratio = self._schema_config.get("default_keyword_min_ratio")
        if isinstance(config_ratio, (int, float)):
            return max(0.0, min(1.0, float(config_ratio)))

        # Ajuste dinámico: 1.0 para esquemas pequeños, 0.5 para listas largas
        if total_keywords <= 2:
            return 1.0
        return 0.5

    # ----------- API pública -----------

    def get_sections(self) -> List[str]:
        return self.sections

    def get_patterns(self) -> Dict[str, List[re.Pattern]]:
        return self.patterns

    def identify_section(
        self,
        line: str,
        *,
        fuzzy_threshold: Optional[float] = None,
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
                    log_info(f"SectionLoader match: type=regex section={key} score=100")
                    return (key, "regex", 100.0, text)

        # 2) Fuzzy (si está disponible)
        if self._fuzzy_enabled:
            # Quitar prefijos numéricos antes de fuzzy
            pref = re.compile(_prefix_num_pattern(), flags=re.IGNORECASE)
            stripped = pref.sub("", text).strip()
            norm_line = _normalize(stripped, strip_accents=True)

            best_key: Optional[str]
            best_score: float
            cache_result = self._fuzzy_cache.get(norm_line)
            if cache_result is not None:
                best_key, best_score = cache_result
            else:
                best_key = None
                best_score = 0.0
                if self._use_process_extract:
                    match = process.extractOne(
                        norm_line,
                        self._fuzzy_choices,
                        scorer=_token_set_score,
                        processor=lambda choice: choice[0],
                    )
                    if match:
                        (cand_text, cand_key), score, _ = match
                        best_key = cand_key
                        best_score = float(score)
                else:
                    for cand_text, cand_key in self._iter_fuzzy_candidates():
                        if self._fuzzy_strategy == "rapidfuzz":
                            score = _token_set_score(norm_line, cand_text)
                        else:
                            score = (
                                SequenceMatcher(None, norm_line, cand_text).ratio() * 100
                            )
                        if score > best_score:
                            best_score = float(score)
                            best_key = cand_key

                self._fuzzy_cache[norm_line] = (best_key, best_score)

            if best_key:
                base_threshold = (
                    float(fuzzy_threshold)
                    if fuzzy_threshold is not None
                    else self._default_fuzzy_threshold()
                )
                section_threshold = self._section_fuzzy_threshold(best_key)
                effective_threshold = (
                    section_threshold if section_threshold is not None else base_threshold
                )
                if (
                    fuzzy_threshold is not None
                    and section_threshold is not None
                    and float(fuzzy_threshold) > section_threshold
                ):
                    effective_threshold = float(fuzzy_threshold)

                if best_score >= effective_threshold:
                    log_info(
                        f"SectionLoader match: type=fuzzy section={best_key} score={best_score:.2f}"
                    )
                    return (best_key, "fuzzy", float(best_score), text)
            return None

        # 3) Keywords (como refuerzo: todas las keywords deben estar o al menos 2)
        if keyword_boost:
            norm_text = _normalize(text, strip_accents=False)
            for key, obj in self.schema.items():
                kws_raw = [k for k in obj.get("keywords", []) if isinstance(k, str)]
                if not kws_raw:
                    continue

                kws = [_normalize(k, strip_accents=False) for k in kws_raw]
                if not kws:
                    continue
           
                hits = sum(1 for k in kws if k in norm_text)
                ratio = hits / len(kws)
                min_ratio = self._keyword_min_ratio(key, len(kws))
                if hits and ratio >= min_ratio:
                    log_info(
                        f"SectionLoader match: type=keywords section={key} score=100 hits={hits}"
                    )
                    return (key, "keywords", 100.0, text)

        return None
    
    def _iter_fuzzy_candidates(self) -> Iterable[Tuple[str, str]]:
        """Itera sobre candidatos fuzzy respetando el orden cacheado."""
        if self._fuzzy_choices:
            for cand in self._fuzzy_choices:
                yield cand
        else:
            for key, candidates in self._fuzzy_index.items():
                for cand in candidates:
                    yield cand, key
