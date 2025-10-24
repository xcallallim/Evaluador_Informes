"""Unit tests for the evaluation service helpers."""

from services.ai_service import OpenAIService
from services.evaluation_service import MockAIService


def test_mock_ai_service_respects_model_name_and_metadata() -> None:
    service = MockAIService(model_name="mockito")

    result = service.evaluate("texto", question={"niveles": [{"valor": 2.5}]})

    assert set(result.keys()) == {"score", "justification", "relevant_text", "metadata"}
    assert result["metadata"]["mock"] is True
    assert result["metadata"]["model"] == "mockito"
    metadata = result["metadata"]
    metadata_keys = set(metadata.keys())
    assert {"model", "mock"}.issubset(metadata_keys)
    assert metadata["mock"] is True
    assert metadata["model"] == "mockito"
    assert metadata["provider"] == "mock"
    assert metadata["duration_ms"] == 0.0
    expected_keys = {
        "model",
        "mock",
        "provider",
        "duration_ms",
        "client_mode",
        "request_id",
        "timestamp",
        "report_type",
        "allowed_levels",
        "enforced_discrete",
        "raw_score",
        "adjusted_score",
    }
    assert expected_keys.issubset(metadata_keys)
    assert isinstance(result["score"], float)
    assert 0.0 <= result["score"] <= 2.5
    assert isinstance(result["justification"], str)


def test_mock_ai_service_defaults_to_standard_scale_when_no_levels() -> None:
    service = MockAIService()

    result = service.evaluate("texto", question={})

    assert isinstance(result["score"], float)
    assert 0.0 <= result["score"] <= 4.0
    assert result["metadata"]["mock"] is True


def test_openai_service_clamps_scores_to_half_points_for_pn_reports() -> None:
    service = OpenAIService(model_name="test", client=object())

    levels = service._normalise_levels(service._resolve_levels(None, {"tipo_informe": "PN"}))

    assert 0.5 in levels
    assert service._enforce_discrete_score(0.6, levels=levels) == 0.5
    assert service._enforce_discrete_score(0.74, levels=levels) == 0.5
    assert service._enforce_discrete_score(1.76, levels=levels) == 2.0


def test_openai_service_respects_question_defined_half_point_levels() -> None:
    service = OpenAIService(model_name="test", client=object())

    question = {"niveles": [{"valor": "0"}, {"valor": "0.5"}, {"valor": "2"}]}

    levels = service._normalise_levels(service._resolve_levels(question, None))

    assert levels == (0.0, 0.5, 2.0)
    assert service._enforce_discrete_score(0.8, levels=levels) == 0.5
    assert service._enforce_discrete_score(1.9, levels=levels) == 2.0

# pytest tests/test_evaluation_service.py -v