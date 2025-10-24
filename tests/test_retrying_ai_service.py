# cobertura de unidad para la API por lotes del contenedor de reintentos para 
# confirmar respuestas exitosas y manejo de fallas con seguimiento de latencia.

import logging
import time
from typing import Any, Dict, List, Tuple

import pytest

from services.evaluation_service import RetryingAIService


class DummyAIService:
    def __init__(self) -> None:
        self.calls: List[Tuple[str, Dict[str, Any]]] = []

    def evaluate(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append((prompt, dict(kwargs)))
        # Pequeño retraso para observar que el wrapper devuelve latencia positiva
        time.sleep(0.01)
        return {
            "score": 1.0,
            "justification": prompt,
            "metadata": {"echo": True},
        }


class ExplodingAIService(DummyAIService):
    def evaluate(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        raise RuntimeError("boom")


@pytest.fixture
def logger() -> logging.Logger:
    logger = logging.getLogger("tests.retrying_ai")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    return logger


def test_evaluate_many_returns_latency_and_responses(logger: logging.Logger) -> None:
    inner = DummyAIService()
    service = RetryingAIService(
        inner,
        retries=0,
        backoff=1.0,
        timeout=None,
        logger=logger,
    )

    tasks = [(f"prompt-{idx}", {"index": idx}) for idx in range(3)]
    outcomes = service.evaluate_many(tasks, parallelism=2)

    assert len(outcomes) == 3
    for idx, outcome in enumerate(outcomes):
        assert outcome["response"]["score"] == 1.0
        assert outcome["response"]["justification"] == f"prompt-{idx}"
        assert outcome["response"]["metadata"]["echo"] is True
        assert outcome["latency_ms"] >= 0.0

    # El servicio interno debe haber recibido todas las llamadas
    recorded_prompts = [prompt for prompt, _ in inner.calls]
    assert recorded_prompts == [f"prompt-{idx}" for idx in range(3)]


def test_evaluate_many_handles_failures(logger: logging.Logger) -> None:
    inner = ExplodingAIService()
    service = RetryingAIService(
        inner,
        retries=1,
        backoff=1.0,
        timeout=None,
        logger=logger,
    )

    outcomes = service.evaluate_many([("prompt", {})])
    assert len(outcomes) == 1
    result = outcomes[0]["response"]
    assert result["score"] == 0
    assert result["metadata"]["score_imputed"] is True
    assert outcomes[0]["latency_ms"] >= 0.0

# pytest tests/test_retrying_ai_service.py -v