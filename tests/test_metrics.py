"""Pruebas para la normalización y agregación de métricas del evaluador."""

import math
import time

from data.models.evaluation import EvaluationResult, SectionResult
from metrics import (
    calculate_institutional_metrics,
    calculate_metrics,
    calculate_policy_metrics,
)


def build_evaluation(score, section_scores, *, tipo_informe="institucional"):
    return EvaluationResult(
        score=score,
        document_type=tipo_informe,
        metadata={"tipo_informe": tipo_informe},
        sections=[
            SectionResult(title=f"Section {idx}", section_id=f"s{idx}", score=value)
            for idx, value in enumerate(section_scores, start=1)
        ],
    )


def test_institutional_metrics_supports_custom_normalization_and_weights():
    evaluation = build_evaluation(3.0, [3.5, None])
    criteria = {
        "tipo_informe": "institucional",
        "metrica_global": {"escala_resultado": {"min": 1, "max": 4}},
    }

    summary = calculate_institutional_metrics(
        evaluation,
        criteria,
        normalized_range=(0.0, 1.0),
        weights={"s1": 2.5},
    ).to_dict()

    assert "normalized_min" in summary["global"]
    assert "normalized_max" in summary["global"]
    assert summary["global"]["normalized_min"] == 0.0
    assert summary["global"]["normalized_max"] == 1.0
    assert math.isclose(summary["global"]["normalized_score"], 2.0 / 3.0, rel_tol=1e-6)
    assert summary["totals"] == {
        "sections_total": 2,
        "sections_with_score": 1,
        "sections_without_score": 1,
    }
    assert math.isclose(summary["sections"][0]["normalized_score"], 5.0 / 6.0, rel_tol=1e-6)
    assert summary["sections"][1]["normalized_score"] is None
    assert summary["sections"][0]["weight"] == 2.5


def test_policy_metrics_defaults_to_percentage_range():
    evaluation = build_evaluation(1.2, [1.0, 1.4], tipo_informe="politica")
    criteria = {
        "tipo_informe": "politica",
        "escala": {"min": 0.0, "max": 2.0},
    }

    summary = calculate_policy_metrics(evaluation, criteria).to_dict()

    assert summary["global"]["normalized_min"] == 0.0
    assert summary["global"]["normalized_max"] == 100.0
    assert math.isclose(summary["global"]["normalized_score"], 60.0, rel_tol=1e-6)
    assert summary["totals"]["sections_with_score"] == 2


def test_calculate_metrics_dispatches_unknown_type_and_preserves_sections():
    evaluation = build_evaluation(0.75, [0.7], tipo_informe="desconocido")
    criteria = {"tipo_informe": "desconocido"}

    summary = calculate_metrics(
        evaluation, criteria, normalized_range=(0.0, 20.0)
    ).to_dict()

    assert summary["methodology"] == "desconocido"
    assert summary["global"]["normalized_min"] == 0.0
    assert summary["global"]["normalized_max"] == 20.0
    assert summary["totals"]["sections_total"] == 1
    assert summary["totals"]["sections_with_score"] == 1
    assert summary["totals"]["sections_without_score"] == 0
    assert "sections" in summary
    assert summary["sections"][0]["section_id"] == "s1"


def test_metrics_handle_global_none_without_normalization():
    evaluation = build_evaluation(None, [1.0])
    criteria = {"tipo_informe": "institucional"}

    summary = calculate_institutional_metrics(evaluation, criteria).to_dict()

    assert summary["global"]["raw_score"] is None
    assert summary["global"]["normalized_score"] is None


def test_normalise_clamps_out_of_range_values():
    evaluation = build_evaluation(5.0, [5.0], tipo_informe="politica")
    criteria = {
        "tipo_informe": "politica",
        "escala": {"min": 0.0, "max": 2.0},
    }

    summary = calculate_policy_metrics(
        evaluation, criteria, normalized_range=(0.0, 100.0)
    ).to_dict()

    assert summary["global"]["normalized_score"] == 100.0
    assert summary["sections"][0]["normalized_score"] == 100.0


def test_calculate_metrics_accepts_weights_mapping():
    evaluation = build_evaluation(0.5, [0.5, 0.5])
    criteria = {"tipo_informe": "institucional"}

    summary = calculate_metrics(
        evaluation, criteria, weights={"s1": 0.1, "s2": 0.9}
    ).to_dict()

    weights = {row["section_id"]: row["weight"] for row in summary["sections"]}
    assert weights == {"s1": 0.1, "s2": 0.9}


def test_metrics_performance_with_large_section_volume():
    scores = [1.5] * 1000
    evaluation = build_evaluation(1.5, scores)
    criteria = {"tipo_informe": "institucional"}

    start = time.perf_counter()
    summary = calculate_institutional_metrics(evaluation, criteria).to_dict()
    duration = time.perf_counter() - start

    assert summary["totals"]["sections_total"] == 1000
    assert duration < 0.1

# pytest tests/test_metrics.py -v
