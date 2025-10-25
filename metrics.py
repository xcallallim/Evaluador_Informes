"""Helpers para calcular métricas globales a partir de los resultados.

El servicio de orquestación invoca estas funciones luego de que el evaluador
produce los puntajes por pregunta y los agregados ponderados por sección.
Separar los cálculos en este módulo permite incorporar metodologías
alternativas en el futuro sin modificar el servicio principal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from data.models.evaluation import EvaluationResult

__all__ = [
    "MetricsSummary",
    "calculate_metrics",
    "calculate_institutional_metrics",
    "calculate_policy_metrics",
]


@dataclass(slots=True)
class MetricsSummary:
    """Contenedor que almacena el índice global y el detalle por sección."""

    data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.data)


def _normalise(
    score: Optional[float],
    *,
    min_value: float,
    max_value: float,
    target_min: float,
    target_max: float,
) -> Optional[float]:
    if score is None:
        return None
    if max_value <= min_value or target_max <= target_min:
        return None
    ratio = (float(score) - min_value) / (max_value - min_value)
    clamped_ratio = max(0.0, min(1.0, ratio))
    return target_min + clamped_ratio * (target_max - target_min)


def _resolve_tipo_informe(evaluation: EvaluationResult) -> Optional[str]:
    """Extrae el tipo de informe desde ``evaluation`` cuando está disponible."""

    if hasattr(evaluation, "tipo_informe"):
        tipo = getattr(evaluation, "tipo_informe")  # type: ignore[attr-defined]
        if isinstance(tipo, str) and tipo.strip():
            return tipo
    if isinstance(evaluation.document_type, str) and evaluation.document_type.strip():
        return evaluation.document_type
    tipo = evaluation.metadata.get("tipo_informe")
    if isinstance(tipo, str) and tipo.strip():
        return tipo
    return None


def _section_summaries(
    evaluation: EvaluationResult,
    *,
    min_value: float,
    max_value: float,
    target_range: Tuple[float, float],
    weights: Optional[Mapping[str, float]] = None,
) -> list[Dict[str, Any]]:
    summaries: list[Dict[str, Any]] = []
    target_min, target_max = target_range
    tipo_informe = _resolve_tipo_informe(evaluation)
    for section in evaluation.sections:
        weight = section.weight
        if weights and section.section_id in weights:
            try:
                weight = float(weights[section.section_id])
            except (TypeError, ValueError):
                weight = section.weight
        summaries.append(
            {
                "section_id": section.section_id,
                "title": section.title,
                "weight": weight,
                "score": section.score,
                "tipo_informe": tipo_informe,
                "normalized_score": _normalise(
                    section.score,
                    min_value=min_value,
                    max_value=max_value,
                    target_min=target_min,
                    target_max=target_max,
                ),
            }
        )
    return summaries


def _totals(evaluation: EvaluationResult) -> Dict[str, int]:
    total_sections = len(evaluation.sections)
    sections_with_score = sum(1 for section in evaluation.sections if section.score is not None)
    sections_without_score = total_sections - sections_with_score
    return {
        "sections_total": total_sections,
        "sections_with_score": sections_with_score,
        "sections_without_score": sections_without_score,
    }


def _target_range_from_criteria(
    criteria: Mapping[str, Any],
    *,
    default_min: float,
    default_max: float,
) -> Tuple[float, float]:
    """Devuelve el rango de normalización de salida deseado."""

    if not isinstance(criteria, Mapping):
        return (default_min, default_max)
    scale = criteria.get("escala_normalizada")
    if isinstance(scale, Mapping):
        try:
            target_min = float(scale.get("min"))
            target_max = float(scale.get("max"))
        except (TypeError, ValueError):
            return (default_min, default_max)
        if target_max > target_min:
            return (target_min, target_max)
    return (default_min, default_max)


def calculate_institutional_metrics(
    evaluation: EvaluationResult,
    criteria: Mapping[str, Any],
    *,
    normalized_range: Optional[Tuple[float, float]] = None,
    weights: Optional[Mapping[str, float]] = None,
) -> MetricsSummary:
    """Calcula el índice institucional para una evaluación ya ponderada.

    Se espera que el evaluador entregue un :class:`EvaluationResult` con las
    secciones ya promediadas. Esta función únicamente normaliza esos agregados a
    la escala solicitada. ``weights`` puede sobrescribir el peso de cada sección
    al explorar ponderaciones alternativas.
    """

    scale = (
        criteria.get("metrica_global", {}).get("escala_resultado", {})
        if isinstance(criteria, Mapping)
        else {}
    )
    min_value = float(scale.get("min", 0.0))
    max_value = float(scale.get("max", 4.0))
    target_min, target_max = (
        normalized_range
        if normalized_range is not None
        else _target_range_from_criteria(criteria, default_min=0.0, default_max=100.0)
    )
    global_normalized = _normalise(
        evaluation.score,
        min_value=min_value,
        max_value=max_value,
        target_min=target_min,
        target_max=target_max,
    )

    data = {
        "methodology": "institucional",
        "global": {
            "raw_score": evaluation.score,
            "normalized_score": global_normalized,
            "scale_min": min_value,
            "scale_max": max_value,
            "normalized_min": target_min,
            "normalized_max": target_max,
        },
        "sections": _section_summaries(
            evaluation,
            min_value=min_value,
            max_value=max_value,
            target_range=(target_min, target_max),
            weights=weights,
        ),
        "totals": _totals(evaluation),
    }
    return MetricsSummary(data)


def calculate_policy_metrics(
    evaluation: EvaluationResult,
    criteria: Mapping[str, Any],
    *,
    normalized_range: Optional[Tuple[float, float]] = None,
    weights: Optional[Mapping[str, float]] = None,
) -> MetricsSummary:
    """Devuelve el índice de política para una evaluación ya ponderada.

    Se asume que las secciones contienen promedios ponderados. La función solo
    normaliza los resultados al rango objetivo solicitado. ``weights`` permite
    ajustar los pesos registrados por sección para experimentar rápidamente.
    """

    scale = criteria.get("escala", {}) if isinstance(criteria, Mapping) else {}
    min_value = float(scale.get("min", 0.0))
    max_value = float(scale.get("max", 2.0))
    target_min, target_max = (
        normalized_range
        if normalized_range is not None
        else _target_range_from_criteria(criteria, default_min=0.0, default_max=100.0)
    )
    global_normalized = _normalise(
        evaluation.score,
        min_value=min_value,
        max_value=max_value,
        target_min=target_min,
        target_max=target_max,
    )
    data = {
        "methodology": "politica_nacional",
        "global": {
            "raw_score": evaluation.score,
            "normalized_score": global_normalized,
            "scale_min": min_value,
            "scale_max": max_value,
            "normalized_min": target_min,
            "normalized_max": target_max,
        },
        "sections": _section_summaries(
            evaluation,
            min_value=min_value,
            max_value=max_value,
            target_range=(target_min, target_max),
            weights=weights,
        ),
        "totals": _totals(evaluation),
    }
    return MetricsSummary(data)


def calculate_metrics(
    evaluation: EvaluationResult,
    criteria: Mapping[str, Any],
    *,
    normalized_range: Optional[Tuple[float, float]] = None,
    weights: Optional[Mapping[str, float]] = None,
) -> MetricsSummary:
    """Despacha el cálculo de métricas según el tipo de informe.

    El :class:`EvaluationResult` recibido debe incluir los agregados ponderados
    por sección; esta función se encarga de normalizarlos y reportar totales.
    Cuando se indica ``weights`` se propaga al detalle por sección sin importar
    el tipo de informe.
    """

    report_type = ""
    if isinstance(criteria, Mapping):
        report_type = str(criteria.get("tipo_informe", "")).strip().lower()

    if report_type == "institucional":
        return calculate_institutional_metrics(
            evaluation, criteria, normalized_range=normalized_range, weights=weights
        )
    if report_type in {"politica", "politica_nacional"}:
        return calculate_policy_metrics(
            evaluation, criteria, normalized_range=normalized_range, weights=weights
        )

    # Recurso genérico: reutiliza el puntaje de evaluación sin normalizar.
    target_min, target_max = normalized_range or (0.0, 100.0)
    data = {
        "methodology": report_type or "desconocida",
        "global": {
            "raw_score": evaluation.score,
            "normalized_score": None,
            "scale_min": None,
            "scale_max": None,
            "normalized_min": target_min,
            "normalized_max": target_max,
        },
        "sections": _section_summaries(
            evaluation,
            min_value=0.0,
            max_value=1.0,
            target_range=(target_min, target_max),
            weights=weights,
        ),
        "totals": _totals(evaluation),
    }
    return MetricsSummary(data)