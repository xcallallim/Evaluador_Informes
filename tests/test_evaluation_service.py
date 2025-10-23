"""Unit tests for the evaluation service helpers."""

from evaluation_service import MockAIService


def test_mock_ai_service_respects_model_name_and_metadata() -> None:
    service = MockAIService(model_name="mockito")

    result = service.evaluate("texto", question={"niveles": [{"valor": 2.5}]})

    assert result["metadata"]["mock"] is True
    assert result["metadata"]["model"] == "mockito"
    assert 0.0 <= result["score"] <= 2.5
    assert isinstance(result["justification"], str)


def test_mock_ai_service_defaults_to_standard_scale_when_no_levels() -> None:
    service = MockAIService()

    result = service.evaluate("texto", question={})

    assert 0.0 <= result["score"] <= 4.0
    assert result["metadata"]["mock"] is True