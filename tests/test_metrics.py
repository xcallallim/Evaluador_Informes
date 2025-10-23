import math

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


def test_institutional_metrics_supports_custom_normalization():
    evaluation = build_evaluation(3.0, [3.5, None])
    criteria = {
        "tipo_informe": "institucional",
        "metrica_global": {"escala_resultado": {"min": 1, "max": 4}},
    }

    summary = calculate_institutional_metrics(
        evaluation, criteria, normalized_range=(0.0, 1.0)
    ).to_dict()

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


def test_calculate_metrics_dispatches_unknown_type():
    evaluation = build_evaluation(0.75, [0.7], tipo_informe="desconocido")
    criteria = {"tipo_informe": "desconocido"}

    summary = calculate_metrics(evaluation, criteria, normalized_range=(0.0, 20.0)).to_dict()

    assert summary["methodology"] == "desconocido"
    assert summary["global"]["normalized_min"] == 0.0
    assert summary["global"]["normalized_max"] == 20.0
    assert summary["totals"]["sections_total"] == 1
    assert summary["totals"]["sections_with_score"] == 1
    assert summary["totals"]["sections_without_score"] == 0