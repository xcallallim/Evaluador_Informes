"""Utility helpers to compute global metrics for evaluation results.

The orchestration service invokes these helpers after the evaluator has
produced the per-question scores *and* the weighted aggregates for each
section.  Keeping the metrics calculations in a separate module makes it
easier to plug alternative methodologies in the future without touching the
main service.
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
    """Container holding the global index and section level breakdown."""

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
    return target_min + ratio * (target_max - target_min)


def _resolve_tipo_informe(evaluation: EvaluationResult) -> Optional[str]:
    """Extract the report type from ``evaluation`` when available."""

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
) -> list[Dict[str, Any]]:
    summaries: list[Dict[str, Any]] = []
    target_min, target_max = target_range
    tipo_informe = _resolve_tipo_informe(evaluation)
    for section in evaluation.sections:
        summaries.append(
            {
                "section_id": section.section_id,
                "title": section.title,
                "weight": section.weight,
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
    """Return the desired output normalisation range."""

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
) -> MetricsSummary:
    """Compute the institutional index for an already weighted evaluation.

    The evaluator is expected to provide an :class:`EvaluationResult` where the
    section scores are already weighted averages.  The helper only normalises
    those aggregates to the requested output scale.
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
        ),
        "totals": _totals(evaluation),
    }
    return MetricsSummary(data)


def calculate_policy_metrics(
    evaluation: EvaluationResult,
    criteria: Mapping[str, Any],
    *,
    normalized_range: Optional[Tuple[float, float]] = None,
) -> MetricsSummary:
    """Return the policy index for an already weighted evaluation.

    Section scores are assumed to be weighted averages.  The helper simply
    normalises the results to the requested target range.
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
        ),
        "totals": _totals(evaluation),
    }
    return MetricsSummary(data)


def calculate_metrics(
    evaluation: EvaluationResult,
    criteria: Mapping[str, Any],
    *,
    normalized_range: Optional[Tuple[float, float]] = None,
) -> MetricsSummary:
    """Dispatch the metric calculation according to the report type.

    The provided :class:`EvaluationResult` must already contain weighted
    aggregates for each section; this helper focuses on normalising those
    values and reporting totals.
    """

    report_type = ""
    if isinstance(criteria, Mapping):
        report_type = str(criteria.get("tipo_informe", "")).strip().lower()

    if report_type == "institucional":
        return calculate_institutional_metrics(
            evaluation, criteria, normalized_range=normalized_range
        )
    if report_type in {"politica", "politica_nacional"}:
        return calculate_policy_metrics(
            evaluation, criteria, normalized_range=normalized_range
        )

    # Generic fallback: reuse the evaluation score without normalisation.
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
        ),
        "totals": _totals(evaluation),
    }
    return MetricsSummary(data)