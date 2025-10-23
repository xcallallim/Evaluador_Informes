"""Utility functions to compute global metrics for evaluation results.

The orchestration service invokes these helpers after the evaluator has
produced the per-question scores.  Keeping the metrics calculations in a
separate module makes it easier to plug alternative methodologies in the
future without touching the main service.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

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


def _normalise(score: Optional[float], *, min_value: float, max_value: float) -> Optional[float]:
    if score is None:
        return None
    if max_value <= min_value:
        return None
    return (float(score) - min_value) / (max_value - min_value) * 100.0


def _section_summaries(
    evaluation: EvaluationResult,
    *,
    min_value: float,
    max_value: float,
) -> list[Dict[str, Any]]:
    summaries: list[Dict[str, Any]] = []
    for section in evaluation.sections:
        summaries.append(
            {
                "section_id": section.section_id,
                "title": section.title,
                "weight": section.weight,
                "score": section.score,
                "normalized_score": _normalise(
                    section.score, min_value=min_value, max_value=max_value
                ),
            }
        )
    return summaries


def calculate_institutional_metrics(
    evaluation: EvaluationResult,
    criteria: Mapping[str, Any],
) -> MetricsSummary:
    """Compute the institutional index using the configured Likert scale."""

    scale = (
        criteria.get("metrica_global", {}).get("escala_resultado", {})
        if isinstance(criteria, Mapping)
        else {}
    )
    min_value = float(scale.get("min", 0.0))
    max_value = float(scale.get("max", 4.0))
    global_normalized = _normalise(
        evaluation.score, min_value=min_value, max_value=max_value
    )

    data = {
        "methodology": "institucional",
        "global": {
            "raw_score": evaluation.score,
            "normalized_score": global_normalized,
            "scale_min": min_value,
            "scale_max": max_value,
        },
        "sections": _section_summaries(
            evaluation, min_value=min_value, max_value=max_value
        ),
    }
    return MetricsSummary(data)


def calculate_policy_metrics(
    evaluation: EvaluationResult,
    criteria: Mapping[str, Any],
) -> MetricsSummary:
    """Return the policy index normalised to a 0-100 range."""

    scale = criteria.get("escala", {}) if isinstance(criteria, Mapping) else {}
    min_value = float(scale.get("min", 0.0))
    max_value = float(scale.get("max", 2.0))
    global_normalized = _normalise(
        evaluation.score, min_value=min_value, max_value=max_value
    )
    data = {
        "methodology": "politica_nacional",
        "global": {
            "raw_score": evaluation.score,
            "normalized_score": global_normalized,
            "scale_min": min_value,
            "scale_max": max_value,
        },
        "sections": _section_summaries(
            evaluation, min_value=min_value, max_value=max_value
        ),
    }
    return MetricsSummary(data)


def calculate_metrics(
    evaluation: EvaluationResult,
    criteria: Mapping[str, Any],
) -> MetricsSummary:
    """Dispatch the metric calculation according to the report type."""

    report_type = ""
    if isinstance(criteria, Mapping):
        report_type = str(criteria.get("tipo_informe", "")).strip().lower()

    if report_type == "institucional":
        return calculate_institutional_metrics(evaluation, criteria)
    if report_type in {"politica", "politica_nacional"}:
        return calculate_policy_metrics(evaluation, criteria)

    # Generic fallback: reuse the evaluation score without normalisation.
    data = {
        "methodology": report_type or "desconocida",
        "global": {
            "raw_score": evaluation.score,
            "normalized_score": None,
            "scale_min": None,
            "scale_max": None,
        },
        "sections": _section_summaries(
            evaluation, min_value=0.0, max_value=1.0
        ),
    }
    return MetricsSummary(data)