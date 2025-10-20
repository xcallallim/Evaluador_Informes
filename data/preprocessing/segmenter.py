# preprocessing/segmenter.py
# -*- coding: utf-8 -*-
import re
import unicodedata
from typing import Dict, List, Tuple

# =========================
# Utilidades de normalización / similitud
# =========================
_ARTICLES_SP = r"\b(de|del|la|las|los|el|en|para|por|con|y|o|u|a)\b"
_ONE_WORD_HEADINGS = {"conclusiones", "recomendaciones", "anexos"}  # excepciones permitidas

def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn")

def _normalize(s: str) -> str:
    """Minúsculas, sin tildes, compacta espacios, remueve artículos comunes."""
    s = _strip_accents(s.lower())
    s = re.sub(_ARTICLES_SP, " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _token_set_similarity(a: str, b: str) -> float:
    """Similaridad tipo 'token set' (0..1) sin dependencias externas."""
    ta = set(_normalize(a).split())
    tb = set(_normalize(b).split())
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union

# =========================
# Heurísticas de detección de títulos
# =========================
def _looks_like_heading(line: str) -> bool:
    """
    Heurística: línea que parece título.
    - Admite 1 palabra solo si es Conclusiones / Recomendaciones / Anexos (con o sin numeración previa).
    - Permite numeración tipo: "6.", "6.1", "6) ", "6. Anexos", etc.
    """
    t = line.strip()
    if not t:
        return False
    if t.endswith("."):
        return False
    if len(t) > 120:
        return False

    # Permitir numeración al inicio
    t_numless = re.sub(r"^\s*\d+(\.\d+)*[\)\.]?\s*", "", t).strip()

    # Caso especial: títulos de 1 palabra permitidos
    norm_numless = _normalize(t_numless)
    if norm_numless in _ONE_WORD_HEADINGS:
        return True

    # General: al menos 2 palabras con “pinta” de título
    words = [w for w in t.split() if any(ch.isalpha() for ch in w)]
    if len(words) < 2:
        return False

    cap_ratio = sum(w.isupper() for w in words) / len(words)
    title_ratio = sum((len(w) > 1 and w[0].isupper() and w[1:].islower()) for w in words) / len(words)
    starts_numbered = bool(re.match(r"^\s*\d+(\.\d+)*[\)\.]?\s+", t))
    looks_titlish = (cap_ratio >= 0.6) or (title_ratio >= 0.6)
    return starts_numbered or looks_titlish

def _merge_wrapped_headings(lines: List[str]) -> List[Tuple[str, int, int]]:
    """
    Une títulos partidos en dos líneas consecutivas:
    retorna [(texto_unido, line_start_idx, line_end_idx)]
    """
    merged: List[Tuple[str, int, int]] = []
    i = 0
    while i < len(lines):
        cur = lines[i].rstrip("\n")
        if i + 1 < len(lines):
            nxt = lines[i + 1].rstrip("\n")
            if _looks_like_heading(cur) and _looks_like_heading(nxt):
                merged.append((cur + " " + nxt, i, i + 1))
                i += 2
                continue
        merged.append((cur, i, i))
        i += 1
    return merged

def _build_char_offsets(lines: List[str]) -> List[int]:
    """Offset inicial (en chars) de cada línea, al unir con '\n'."""
    pos = 0
    offsets = []
    for ln in lines:
        offsets.append(pos)
        pos += len(ln) + 1  # + '\n'
    return offsets

# =========================
# Catálogo de secciones y variantes
# =========================
def _catalogo_secciones(tipo: str) -> List[str]:
    tipo = _normalize(tipo)
    if tipo == "institucional":
        return [
            "Resumen Ejecutivo",
            "Prioridades de la política institucional",
            "Análisis de resultados de los objetivos estratégicos institucionales (OEI)",
            "Diagnóstico sobre los OEI priorizados con bajo nivel de cumplimiento",
            "Análisis de implementación de las acciones estratégicas institucionales (AEI)",
            "Análisis de implementación de las AEI de los OEI priorizados",
            "Análisis de los productos de la AEI",
            "Análisis de la ejecución operativa en las AEI críticas",
            "Aplicación de las recomendaciones para mejorar la implementación de las AEI",
            "Conclusiones",
            "Recomendaciones",
            "Anexos",
        ]
    elif tipo == "politica nacional":
        return [
            "Resumen Ejecutivo",
            "Descripción de la política nacional",
            "Análisis de los resultados de la política nacional",
            "Análisis de implementación",
            "Conclusiones",
            "Recomendaciones",
        ]
    else:
        raise ValueError("Tipo de informe no reconocido. Usa 'institucional' o 'politica nacional'.")

def _variantes_base() -> Dict[str, List[str]]:
    return {
        "Resumen Ejecutivo": ["Resumen", "Síntesis ejecutiva", "Resumen ejecutívo"],
        "Conclusiones": ["Conclusión", "Conclusiones finales"],
        "Recomendaciones": ["Sugerencias", "Recomendación", "Recomendaciones finales"],
        "Anexos": ["Anexo"],

        "Prioridades de la política institucional": [
            "Prioridades de la política",
            "Prioridades institucionales",
            "Prioridades de política institucional",
        ],
        "Análisis de resultados de los objetivos estratégicos institucionales (OEI)": [
            "Análisis de resultados de los objetivos estratégicos institucionales",
            "Análisis de resultados de los OEI",
            "Análisis de resultados OEI",
            "Resultados de OEI",
        ],
        "Diagnóstico sobre los OEI priorizados con bajo nivel de cumplimiento": [
            "Diagnóstico de OEI priorizados con bajo nivel",
            "Diagnóstico de OEI priorizados",
        ],
        "Análisis de implementación de las acciones estratégicas institucionales (AEI)": [
            "Análisis de implementación de las AEI",
            "Implementación de AEI",
        ],
        "Análisis de implementación de las AEI de los OEI priorizados": [
            "Análisis de implementación de AEI en OEI priorizados",
            "Implementación de AEI en OEI priorizados",
        ],
        "Análisis de los productos de la AEI": [
            "Análisis de productos de AEI",
            "Productos de las AEI",
        ],
        "Análisis de la ejecución operativa en las AEI críticas": [
            "Ejecución operativa en AEI críticas",
            "Análisis de ejecución operativa AEI críticas",
        ],
        "Aplicación de las recomendaciones para mejorar la implementación de las AEI": [
            "Aplicación de recomendaciones para mejorar la implementación de las AEI",
            "Aplicación de recomendaciones AEI",
        ],
        "Descripción de la política nacional": [
            "Descripción de la política",
            "Descripción de la PN",
        ],
        "Análisis de los resultados de la política nacional": [
            "Análisis de resultados de la política",
            "Resultados de la política nacional",
        ],
        "Análisis de implementación": [
            "Análisis de la implementación",
            "Implementación de la política",
        ],
    }

def _build_variants_dict(oficiales: List[str], user_overrides: Dict[str, List[str]] = None) -> Dict[str, List[str]]:
    """
    Construye dict de variantes por sección:
    - Oficial
    - Variantes base
    - Overrides de usuario
    (normaliza para evitar duplicados con pequeñas diferencias)
    """
    base = _variantes_base()
    variants: Dict[str, List[str]] = {}
    for off in oficiales:
        lst = [off]
        if off in base:
            lst += base[off]
        if user_overrides and off in user_overrides:
            lst += user_overrides[off]
        seen = set()
        clean = []
        for v in lst:
            n = _normalize(v)
            if n and n not in seen:
                seen.add(n)
                clean.append(v)
        variants[off] = clean
    return variants

# =========================
# Detector principal
# =========================
def segment_text(
    text: str,
    tipo_informe: str,
    user_overrides: Dict[str, List[str]] = None,
    sim_threshold: float = 0.75
) -> Dict[str, str]:
    """
    Detecta secciones dentro del informe usando coincidencia tolerante (fuzzy controlado).
    - NO inventa secciones.
    - Solo segmenta títulos realmente encontrados.
    - Usa overrides del usuario si alguno título viene con otro nombre en el informe.
    - Considera títulos partidos en 2 líneas.
    - Considera títulos de 1 palabra (Conclusiones / Recomendaciones / Anexos) como válidos.
    - Considera SOLO la PRIMERA aparición de cada sección.
    """

    # 1) catálogo y variantes
    tipo = _normalize(tipo_informe)
    oficiales = _catalogo_secciones(tipo)
    variants = _build_variants_dict(oficiales, user_overrides)

    # 2) texto por líneas + offsets
    raw_lines = text.splitlines()
    joined_text = "\n".join(raw_lines)
    char_offsets = _build_char_offsets(raw_lines)

    # 3) candidatos a título (1 o 2 líneas mergeadas)
    merged_lines = _merge_wrapped_headings(raw_lines)  # [(text, line_start, line_end)]
    candidates: List[Tuple[str, int, int]] = []
    for txt, li, lj in merged_lines:
        if _looks_like_heading(txt):
            candidates.append((txt.strip(), li, lj))

    # 4) match fuzzy controlado contra catálogo
    first_hits: Dict[str, Tuple[int, int, int, float]] = {}
    for cand_text, li, lj in candidates:
        cand_norm = _normalize(cand_text)
        best_off, best_score = None, 0.0
        for off in oficiales:
            for v in variants[off]:
                score = _token_set_similarity(cand_norm, v)
                if score > best_score:
                    best_score = score
                    best_off = off

        if best_off and best_score >= sim_threshold:
            # guardar SOLO la primera aparición de ese título oficial
            if best_off not in first_hits:
                start_char = char_offsets[li]
                first_hits[best_off] = (li, lj, start_char, best_score)

    # 5) construir segmentos SOLO con los detectados (no inventar)
    segmentos: Dict[str, str] = {}
    ordered = [(off, first_hits[off]) for off in oficiales if off in first_hits]
    if not ordered:
        return segmentos  # nada detectado

    # posiciones ordenadas por aparición
    titles_positions = [(off, data[2]) for off, data in ordered]  # (nombre_oficial, start_char)
    # Para calcular el fin de cada sección: siguiente start_char o fin del doc
    doc_end = len(joined_text)

    for i, (off, start_char) in enumerate(titles_positions):
        end_char = titles_positions[i + 1][1] if i + 1 < len(titles_positions) else doc_end
        content = joined_text[start_char:end_char].strip()
        segmentos[off] = content

    return segmentos
