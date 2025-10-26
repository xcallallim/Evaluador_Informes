"""Helpers para calcular métricas globales a partir de los resultados.

El servicio de orquestación invoca estas funciones luego de que el evaluador
produce los puntajes por pregunta y los agregados ponderados por sección.
Separar los cálculos en este módulo permite incorporar metodologías
alternativas en el futuro sin modificar el servicio principal.
"""

from __future__ import annotations

from dataclasses import dataclass
import unicodedata
from typing import Any, Dict, List, Mapping, Optional, Tuple

from data.models.evaluation import EvaluationResult

__all__ = [
    "MetricsSummary",
    "calculate_metrics",
    "calculate_institutional_metrics",
    "calculate_policy_metrics",
]


INSTITUTIONAL_CRITERION_WEIGHTS: Mapping[str, float] = {
    "estructura": 0.05,
    "claridad y coherencia": 0.35,
    "pertinencia": 0.60,
}

INSTITUTIONAL_CRITERION_LABELS: Mapping[str, str] = {
    "estructura": "Estructura",
    "claridad y coherencia": "Claridad y coherencia",
    "pertinencia": "Pertinencia",
}


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

    def _coerce_from_mapping(mapping: Mapping[str, Any]) -> Optional[Tuple[float, float]]:
        try:
            target_min = float(mapping.get("min"))
            target_max = float(mapping.get("max"))
        except (TypeError, ValueError):
            return None
        if target_max > target_min:
            return (target_min, target_max)
        return None

    if not isinstance(criteria, Mapping):
        return (default_min, default_max)

    scale = criteria.get("escala_normalizada")
    if isinstance(scale, Mapping):
        coerced = _coerce_from_mapping(scale)
        if coerced:
            return coerced

    global_settings = criteria.get("metrica_global")
    if isinstance(global_settings, Mapping):
        normalised_scale = global_settings.get("escala_normalizada")
        if isinstance(normalised_scale, Mapping):
            coerced = _coerce_from_mapping(normalised_scale)
            if coerced:
                return coerced

    metodologia = criteria.get("metodologia")
    if isinstance(metodologia, Mapping):
        resultado = metodologia.get("resultado")
        if isinstance(resultado, Mapping):
            rango = resultado.get("rango")
            if isinstance(rango, (list, tuple)) and len(rango) == 2:
                try:
                    values = (float(rango[0]), float(rango[1]))
                except (TypeError, ValueError):
                    values = None
                if values and values[1] > values[0]:
                    return values
    return (default_min, default_max)


def _normalise_label(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _canonical_institutional_criterion(label: Any) -> Optional[str]:
    normalised = _normalise_label(label)
    if not normalised:
        return None
    if normalised.startswith("estructura"):
        return "estructura"
    if "claridad" in normalised and "coherencia" in normalised:
        return "claridad y coherencia"
    if normalised.startswith("pertinencia"):
        return "pertinencia"
    return normalised


def _coerce_weight(value: Any) -> float:
    if value is None:
        return 1.0
    try:
        weight = float(value)
    except (TypeError, ValueError):
        return 1.0
    return max(0.0, weight)


def _institutional_question_max(criterion: str) -> float:
    if criterion == "estructura":
        return 1.0
    return 4.0


def _aggregate_institutional_criteria(
    evaluation: EvaluationResult,
    *,
    section_weight_overrides: Optional[Mapping[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Calcula los puntajes normalizados por criterio a partir de las preguntas."""

    criteria_accumulators: Dict[str, Dict[str, Any]] = {}

    for section in evaluation.sections:
        override_weight = None
        if section_weight_overrides and section.section_id in section_weight_overrides:
            override_weight = section_weight_overrides[section.section_id]
        section_weight = _coerce_weight(override_weight if override_weight is not None else section.weight)
        if section_weight <= 0:
            continue

        for dimension in section.dimensions:
            criterion_key = _canonical_institutional_criterion(dimension.name)
            if not criterion_key:
                continue

            dimension_weight = _coerce_weight(dimension.weight)
            if dimension_weight <= 0:
                continue

            question_pairs: List[tuple[float, float]] = []
            question_count = 0

            for question in dimension.questions:
                score = question.score
                if score is None:
                    continue
                try:
                    numeric_score = float(score)
                except (TypeError, ValueError):
                    continue
                max_score = _institutional_question_max(criterion_key)
                if max_score <= 0:
                    continue
                normalised_score = max(0.0, min(1.0, numeric_score / max_score))
                question_weight = _coerce_weight(question.weight)
                if question_weight <= 0:
                    continue
                question_pairs.append((question_weight, normalised_score))
                question_count += 1

            if not question_pairs:
                continue

            total_question_weight = sum(weight for weight, _ in question_pairs)
            if total_question_weight <= 0:
                continue

            dimension_score = sum(weight * value for weight, value in question_pairs) / total_question_weight

            accumulator = criteria_accumulators.setdefault(
                criterion_key,
                {
                    "label": dimension.name or criterion_key.title(),
                    "weighted_sum": 0.0,
                    "weight_total": 0.0,
                    "questions": 0,
                },
            )

            combined_weight = section_weight * dimension_weight
            if combined_weight <= 0:
                continue

            accumulator["weighted_sum"] += dimension_score * combined_weight
            accumulator["weight_total"] += combined_weight
            accumulator["questions"] += question_count

    computed_entries: Dict[str, Dict[str, Any]] = {}
    for key, payload in criteria_accumulators.items():
        weight_total = payload["weight_total"]
        if weight_total <= 0:
            continue
        score = payload["weighted_sum"] / weight_total
        score = max(0.0, min(1.0, score))
        computed_entries[key] = {
            "key": key,
            "label": payload["label"],
            "normalized_score": score,
            "raw_score": score,
            "questions_evaluated": payload["questions"],
        }

    ordered_entries: List[Dict[str, Any]] = []
    for key in INSTITUTIONAL_CRITERION_WEIGHTS:
        entry = computed_entries.pop(key, None)
        if entry is None:
            label = INSTITUTIONAL_CRITERION_LABELS.get(key, key.title())
            entry = {
                "key": key,
                "label": label,
                "normalized_score": 0.0,
                "raw_score": 0.0,
                "questions_evaluated": 0,
            }
        ordered_entries.append(entry)

    if computed_entries:
        ordered_entries.extend(computed_entries.values())

    return ordered_entries


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
    if normalized_range is not None:
        target_min, target_max = normalized_range
    else:
        target_min, target_max = (0.0, 100.0)

    criteria_breakdown = _aggregate_institutional_criteria(
        evaluation, section_weight_overrides=weights
    )

    total_weight = 0.0
    weighted_score = 0.0
    for entry in criteria_breakdown:
        weight = INSTITUTIONAL_CRITERION_WEIGHTS.get(entry["key"])
        if weight is not None:
            entry["weight"] = weight
            entry["weighted_score"] = entry["normalized_score"] * weight
            weighted_score += entry["weighted_score"]
            total_weight += weight
        else:
            entry["weight"] = None
            entry["weighted_score"] = None

    if total_weight > 0:
        raw_global_score = weighted_score
        normalized_global_score = target_min + (target_max - target_min) * raw_global_score
        scale_min = 0.0
        scale_max = 1.0
    else:
        raw_global_score = evaluation.score
        normalized_global_score = _normalise(
            raw_global_score,
            min_value=min_value,
            max_value=max_value,
            target_min=target_min,
            target_max=target_max,
        )
        scale_min = min_value
        scale_max = max_value

    data = {
        "methodology": "institucional",
        "global": {
            "raw_score": raw_global_score,
            "normalized_score": normalized_global_score,
            "scale_min": scale_min,
            "scale_max": scale_max,
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

    if criteria_breakdown:
        data["criteria"] = criteria_breakdown
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
        else _target_range_from_criteria(criteria, default_min=0.0, default_max=20.0)
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