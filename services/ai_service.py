"""Core abstractions to communicate with AI providers."""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

from services.prompt_builder import (
    PromptBuilder,
    PromptContext,
    build_prompt,
)
from utils.secret_manager import SecretManagerError, get_openai_api_key

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

try:  # pragma: no cover - optional dependency
    from openai import OpenAI
except ImportError:  # pragma: no cover - compatibility with legacy client
    OpenAI = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import openai  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    openai = None  # type: ignore


_LOGGER_NAME = "evaluador.ai_service"
logger = logging.getLogger(_LOGGER_NAME)
logger.addHandler(logging.NullHandler())


def _collect_retry_exceptions() -> tuple[type[BaseException], ...]:
    candidates: set[type[BaseException]] = {TimeoutError, ConnectionError}
    if openai is not None:
        sources = [openai]
        error_module = getattr(openai, "error", None)
        if error_module is not None:
            sources.append(error_module)
        for source in sources:
            for name in (
                "APIError",
                "APIConnectionError",
                "RateLimitError",
                "ServiceUnavailableError",
                "Timeout",
                "TryAgain",
                "OpenAIError",
            ):
                exc = getattr(source, name, None)
                if isinstance(exc, type) and issubclass(exc, BaseException):
                    candidates.add(exc)
    return tuple(candidates)


_RETRYABLE_EXCEPTIONS = _collect_retry_exceptions()


@dataclass(slots=True)
class AIResponse:
    """Structured representation of an AI answer."""

    score: float
    justification: str
    relevant_text: Optional[str]
    metadata: Dict[str, Any]


class InvalidAIResponseError(RuntimeError):
    """Raised when the AI output cannot be parsed as JSON."""

    def __init__(
        self,
        message: str,
        *,
        text: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.text = text
        self.metadata = dict(metadata or {})


class BaseAIService(ABC):
    """Base contract for AI services used within the evaluator."""

    def __init__(
        self,
        *,
        model_name: str,
        prompt_builder: Callable[..., str] | None = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Instantiate the service with the shared configuration.

        Parameters
        ----------
        model_name:
            Identifier of the model used by the provider.
        prompt_builder:
            Callable that must accept the following keyword-only arguments and
            return a string prompt: ``document``, ``criteria``, ``section``,
            ``dimension``, ``question``, ``chunk_text``, ``chunk_metadata`` and
            the optional ``extra_instructions``.
        logger:
            Optional logger instance used for structured tracing.
        """
        self.model_name = model_name
        self.prompt_builder = prompt_builder or build_prompt
        base_logger = logging.getLogger(_LOGGER_NAME)
        self.logger = logger or base_logger.getChild(self.__class__.__name__)
        if not self.logger.handlers:
            self.logger.addHandler(logging.NullHandler())

    @abstractmethod
    def evaluate(self, prompt: Optional[str] = None, **kwargs: Any) -> Mapping[str, Any]:
        """Send the prompt to the model and return a mapping with the evaluation."""

    # ------------------------------------------------------------------
    # Shared helpers
    def _resolve_prompt(self, prompt: Optional[str], **kwargs: Any) -> str:
        """Return the final prompt text for the evaluation request.

        The method supports two data modes:

        * **Global (sección completa)** – requires ``document``, ``criteria``,
          ``section`` and ``chunk_text`` in ``kwargs`` (or encapsulated inside a
          :class:`~services.prompt_builder.PromptContext`). ``dimension`` and
          ``question`` can be absent or ``None``.
        * **Parcial (criterio/pregunta)** – uses the same mandatory fields and
          additionally expects ``dimension`` and ``question`` detailing the
          criterion under evaluation.

        ``chunk_metadata`` remains optional in both modes for traceability.

        If a :class:`~services.prompt_builder.PromptContext` is provided it takes
        precedence and is passed verbatim to the configured prompt builder.
        """

        if prompt:
            return prompt

        custom_builder: Callable[..., str] | None = kwargs.get("prompt_builder")
        builder = custom_builder or self.prompt_builder
        if builder is None:
            raise ValueError("No se proporcionó prompt ni prompt_builder para generar uno.")

        context = kwargs.get("prompt_context")
        if isinstance(context, PromptContext):
            return builder(
                document=context.document,
                criteria=context.criteria,
                section=context.section,
                dimension=context.dimension,
                question=context.question,
                chunk_text=context.chunk_text,
                chunk_metadata=context.chunk_metadata,
                extra_instructions=context.extra_instructions,
            )

        required_keys = ("document", "criteria", "section", "chunk_text")
        missing = [key for key in required_keys if key not in kwargs]
        if missing:
            raise ValueError(
                "Faltan datos para construir el prompt: " + ", ".join(missing)
            )

        chunk_metadata = kwargs.get("chunk_metadata") or {}

        return builder(
            document=kwargs["document"],
            criteria=kwargs["criteria"],
            section=kwargs["section"],
            dimension=kwargs.get("dimension"),
            question=kwargs.get("question"),
            chunk_text=kwargs["chunk_text"],
            chunk_metadata=chunk_metadata,
            extra_instructions=kwargs.get("extra_instructions"),
        )

    def _extract_chunk_text(self, **kwargs: Any) -> Optional[str]:
        context = kwargs.get("prompt_context")
        if isinstance(context, PromptContext):
            return context.chunk_text
        chunk = kwargs.get("chunk_text")
        if chunk is None:
            return None
        return str(chunk)

    def _is_empty_chunk(self, chunk_text: Optional[str]) -> bool:
        if chunk_text is None:
            return True
        if not isinstance(chunk_text, str):
            chunk_text = str(chunk_text)
        return not chunk_text.strip()

    def _base_metadata(self, **kwargs: Any) -> Dict[str, Any]:
        metadata = {
            "model": self.model_name,
        }
        for key in ("section", "dimension", "question", "chunk_index"):
            if key in kwargs and kwargs[key] is not None:
                metadata[key] = kwargs[key]
        return metadata

    def _log_request(self, prompt: str, **kwargs: Any) -> None:
        section = kwargs.get("section")
        dimension = self._summarise_metadata(kwargs.get("dimension"))
        question = self._summarise_metadata(kwargs.get("question"))
        report_type = kwargs.get("report_type")
        request_id = kwargs.get("request_id")
        client_mode = kwargs.get("client_mode")
        self.logger.info(
            "[%s] Invocando modelo %s (modo=%s, tipo_informe=%s, section=%s, dimension=%s, question=%s)",
            request_id,
            self.model_name,
            client_mode,
            report_type,
            section,
            dimension,
            question,
        )
        preview = self._truncate_text(prompt)
        self.logger.debug("[%s] Prompt enviado al modelo %s: %s", request_id, self.model_name, preview)
        if kwargs:
            self.logger.debug("[%s] Metadatos de solicitud: %s", request_id, kwargs)

    def _log_response(self, response: Mapping[str, Any]) -> None:
        score = response.get("score") if isinstance(response, Mapping) else None
        metadata = response.get("metadata") if isinstance(response, Mapping) else None
        request_id = None
        duration_ms = None
        retries = None
        client_mode = None
        if isinstance(metadata, Mapping):
            request_id = metadata.get("request_id")
            duration_ms = metadata.get("duration_ms")
            retries = metadata.get("retries")
            client_mode = metadata.get("client_mode")
        justification = response.get("justification") if isinstance(response, Mapping) else None
        self.logger.info(
            "[%s] Respuesta recibida del modelo %s (modo=%s, score=%s, duration_ms=%s, retries=%s)",
            request_id,
            self.model_name,
            client_mode,
            score,
            duration_ms,
            retries,
        )
        if isinstance(justification, str):
            self.logger.debug(
                "[%s] Justificación (truncada): %s",
                request_id,
                self._truncate_text(justification),
            )
        self.logger.debug(
            "[%s] Respuesta completa recibida: %s",
            request_id,
            self._truncate_text(str(response), limit=600),
        )

    def _summarise_metadata(self, value: Any) -> Any:
        if isinstance(value, Mapping):
            for key in ("id", "nombre", "name", "titulo", "title", "codigo", "code"):
                if key in value and value[key]:
                    return value[key]
            return "{...}"
        return value

    def _truncate_text(self, text: str, limit: int = 300) -> str:
        clean = text.replace("\n", " ")
        if len(clean) <= limit:
            return clean
        return clean[: limit - 3] + "..."

    def _validate_result_payload(self, payload: Mapping[str, Any]) -> None:
        if "score" not in payload:
            raise ValueError("La respuesta del servicio AI debe incluir un 'score'.")
        score = payload["score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise TypeError("'score' debe ser un número (float).")

        if "justification" not in payload:
            raise ValueError("La respuesta del servicio AI debe incluir una 'justification'.")
        justification = payload["justification"]
        if not isinstance(justification, str):
            raise TypeError("'justification' debe ser una cadena de texto.")

        metadata = payload.get("metadata")
        if not isinstance(metadata, MutableMapping):
            raise TypeError("'metadata' debe ser un mapeo con información de la evaluación.")

        required_metadata = {"model", "provider", "duration_ms"}
        missing = [key for key in required_metadata if key not in metadata]
        if missing:
            raise ValueError(
                "Faltan campos obligatorios en metadata: " + ", ".join(sorted(missing))
            )

        if not isinstance(metadata["duration_ms"], (int, float)):
            raise TypeError("'metadata.duration_ms' debe ser numérico.")

        optional_checks = {"request_id", "usage"}
        for key in optional_checks:
            if key in metadata and metadata[key] is not None:
                if key == "usage" and not isinstance(metadata[key], Mapping):
                    raise TypeError("'metadata.usage' debe ser un objeto tipo mapeo si está presente.")
                if key == "request_id" and not isinstance(metadata[key], str):
                    raise TypeError("'metadata.request_id' debe ser una cadena si está presente.")


class MockAIService(BaseAIService):
    """Generador de respuestas deterministas para pruebas y desarrollo local."""

    def __init__(self, model_name: str = "mock-model", **kwargs: Any) -> None:
        super().__init__(model_name=model_name, **kwargs)

    def evaluate(self, prompt: Optional[str] = None, **kwargs: Any) -> Mapping[str, Any]:
        chunk_text = self._extract_chunk_text(**kwargs)
        report_type = None
        criteria = kwargs.get("criteria")
        if isinstance(criteria, Mapping):
            raw_type = criteria.get("tipo_informe")
            if isinstance(raw_type, str):
                report_type = raw_type.strip().lower() or None

        metadata = {
            "model": self.model_name,
            "provider": "mock",
            "duration_ms": 0.0,
            "client_mode": "mock",
            "request_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        }
        metadata["report_type"] = report_type
        levels = list(self._extract_levels(kwargs.get("question") or {}))
        metadata["allowed_levels"] = list(levels)
        metadata["enforced_discrete"] = bool(levels)
        if prompt is None and self._is_empty_chunk(chunk_text):
            result = {
                "score": 0.0,
                "justification": "Sección no encontrada o sin contenido suficiente.",
                "relevant_text": None,
                "metadata": {
                    **metadata,
                    "mock": True,
                    "empty_chunk": True,
                    "raw_score": 0.0,
                    "adjusted_score": 0.0,
                },
            }
            self._validate_result_payload(result)
            self._log_response(result)
            return result

        prompt_text = self._resolve_prompt(prompt, **kwargs)
        self._log_request(prompt_text, **metadata)

        question = kwargs.get("question") or {}
        score = self._deterministic_score(prompt_text, levels)

        result = {
            "score": score,
            "justification": (
                "Respuesta generada por MockAIService a partir del contenido del fragmento."
            ),
            "relevant_text": None,
            "metadata": {
                **metadata,
                "mock": True,
                "raw_score": score,
                "adjusted_score": score,
            },
        }
        self._validate_result_payload(result)
        self._log_response(result)
        return result

    def _extract_levels(self, question: Mapping[str, Any]) -> Iterable[float]:
        niveles = question.get("niveles") or []
        values = []
        for level in niveles:
            if isinstance(level, Mapping) and "valor" in level:
                try:
                    values.append(float(level["valor"]))
                except (TypeError, ValueError):
                    continue
        return sorted(values)

    def _deterministic_score(self, prompt: str, levels: Iterable[float]) -> float:
        digest = hashlib.sha256(prompt.encode("utf-8")).digest()
        base = int.from_bytes(digest[:4], "big")
        candidates = list(levels)
        if candidates:
            index = base % len(candidates)
            return float(candidates[index])
        fallback = [0.0, 1.0, 2.0, 3.0, 4.0]
        return float(fallback[base % len(fallback)])


class OpenAIService(BaseAIService):
    """Real implementation using the OpenAI API."""

    DEFAULT_SYSTEM_PROMPT = (
        "Eres un evaluador experto. Analiza el fragmento del informe siguiendo los criterios "
        "proporcionados y responde exclusivamente con un objeto JSON con las claves 'score' "
        "(número) y 'justification' (texto). Puedes incluir 'relevant_text' si aplica."
    )

    _INSTITUTIONAL_LEVELS: Sequence[float] = (0.0, 1.0, 2.0, 3.0, 4.0)
    _PN_LEVELS: Sequence[float] = (0.0, 0.5, 1.0, 1.5, 2.0)

    def __init__(
        self,
        *,
        model_name: str,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_output_tokens: Optional[int] = 800,
        system_prompt: Optional[str] = None,
        client: Any = None,
        prompt_builder: Optional[PromptBuilder] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            prompt_builder=prompt_builder,
            logger=logger,
        )
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self._api_key = api_key
        self._client = None
        self._client_mode: Optional[str] = None

        if client is not None:
            self._client = client
            self._client_mode = "custom"
        else:
            self._initialise_client()

    def evaluate(self, prompt: Optional[str] = None, **kwargs: Any) -> Mapping[str, Any]:
        chunk_text = self._extract_chunk_text(**kwargs)
        metadata = self._base_metadata(**kwargs)
        metadata["service"] = "openai"
        metadata["provider"] = "openai"
        metadata["duration_ms"] = 0.0
        metadata["client_mode"] = self._client_mode or "chat.completions"
        metadata["report_type"] = self._extract_report_type(kwargs.get("criteria"))
        metadata["request_id"] = str(uuid.uuid4())
        metadata["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        levels_for_empty = self._normalise_levels(
            self._resolve_levels(kwargs.get("question"), kwargs.get("criteria"))
        )
        if prompt is None and self._is_empty_chunk(chunk_text):
            result = {
                "score": 0.0,
                "justification": "Sección no encontrada o sin contenido suficiente.",
                "relevant_text": None,
                "metadata": {
                    **metadata,
                    "empty_chunk": True,
                    "raw_score": 0.0,
                    "adjusted_score": 0.0,
                    "allowed_levels": list(levels_for_empty),
                    "enforced_discrete": bool(levels_for_empty),
                },
            }
            self._validate_result_payload(result)
            self._log_response(result)
            return result

        prompt_text = self._resolve_prompt(prompt, **kwargs)
        self._log_request(prompt_text, **metadata)

        attempts = 0
        retry_total = 0
        total_duration_ms = 0.0
        correction_error: Optional[InvalidAIResponseError] = None

        initial_retry_tracker: Dict[str, int] = {"attempts": 0}
        start = time.perf_counter()
        response = self._invoke_model(prompt_text, _retry_tracker=initial_retry_tracker)
        duration_ms = (time.perf_counter() - start) * 1000
        total_duration_ms += duration_ms
        initial_attempts = int(initial_retry_tracker.get("attempts", 0) or 0)
        if initial_attempts < 0:
            initial_attempts = 0
        attempts += initial_attempts
        retry_total += max(initial_attempts - 1, 0)
        try:
            ai_response = self._normalise_response(response)
        except InvalidAIResponseError as exc:
            correction_error = exc
            self.logger.warning(
                "La respuesta del modelo no fue un JSON válido. Se intentará una corrección."
            )
            correction_prompt = self._build_correction_prompt(prompt_text, exc.text)
            post_meta = metadata.setdefault("postprocessing", {})
            if isinstance(post_meta, MutableMapping):
                post_meta.update(
                    {
                        "invalid_json_retry": True,
                        "invalid_json_raw": exc.text,
                    }
                )
            else:
                metadata["postprocessing"] = {
                    "invalid_json_retry": True,
                    "invalid_json_raw": exc.text,
                }
            correction_retry_tracker: Dict[str, int] = {"attempts": 0}
            start = time.perf_counter()
            response = self._invoke_model(
                correction_prompt, _retry_tracker=correction_retry_tracker
            )
            duration_ms = (time.perf_counter() - start) * 1000
            total_duration_ms += duration_ms
            correction_attempts = int(correction_retry_tracker.get("attempts", 0) or 0)
            if correction_attempts < 0:
                correction_attempts = 0
            attempts += correction_attempts
            retry_total += max(correction_attempts - 1, 0)
            try:
                ai_response = self._normalise_response(response)
            except InvalidAIResponseError as exc_retry:
                raise RuntimeError(
                    "El modelo no devolvió un JSON válido tras reintentar con un prompt de corrección"
                ) from exc_retry
        levels = self._normalise_levels(
            self._resolve_levels(kwargs.get("question"), kwargs.get("criteria"))
        )
        adjusted_score = self._enforce_discrete_score(ai_response.score, levels=levels)
        metadata["attempts"] = attempts
        metadata["retries"] = retry_total
        metadata["duration_ms"] = total_duration_ms

        combined_metadata: Dict[str, Any] = dict(metadata)
        response_metadata = ai_response.metadata or {}
        if isinstance(response_metadata, Mapping):
            for key, value in response_metadata.items():
                if (
                    key == "postprocessing"
                    and key in combined_metadata
                    and isinstance(combined_metadata[key], MutableMapping)
                    and isinstance(value, Mapping)
                ):
                    combined_metadata[key].update(value)
                elif key in {"client_mode", "request_id", "timestamp", "report_type", "service"}:
                    continue
                else:
                    combined_metadata[key] = value
        combined_metadata["attempts"] = attempts
        combined_metadata["retries"] = retry_total
        combined_metadata["duration_ms"] = total_duration_ms
        combined_metadata["allowed_levels"] = list(levels)
        combined_metadata["raw_score"] = ai_response.score
        combined_metadata["adjusted_score"] = adjusted_score
        combined_metadata["enforced_discrete"] = bool(levels)
        if correction_error is not None:
            combined_metadata["correction_attempt"] = True
            if correction_error.metadata:
                previous_attempt = combined_metadata.setdefault("previous_attempt", {})
                if isinstance(previous_attempt, MutableMapping):
                    previous_attempt.update(dict(correction_error.metadata))
                else:
                    combined_metadata["previous_attempt"] = dict(correction_error.metadata)
        if (
            ai_response.score is not None
            and adjusted_score is not None
            and adjusted_score != ai_response.score
        ):
            post_meta = combined_metadata.setdefault("postprocessing", {})
            payload = {
                "raw_score": ai_response.score,
                "discrete_levels": list(levels),
            }
            if isinstance(post_meta, MutableMapping):
                post_meta.update(payload)
            else:
                combined_metadata["postprocessing"] = payload
        result = {
            "score": adjusted_score,
            "justification": ai_response.justification,
            "relevant_text": ai_response.relevant_text,
            "metadata": combined_metadata,
        }
        self._validate_result_payload(result)
        self._log_response(result)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    def _initialise_client(self) -> None:
        api_key = self._api_key
        if not api_key:
            try:
                api_key = get_openai_api_key()
            except SecretManagerError as exc:  # pragma: no cover - defensive
                raise RuntimeError("No se pudo obtener la clave API de OpenAI") from exc
        self._api_key = api_key

        if OpenAI is not None:  # pragma: no cover - depends on library availability
            self._client = OpenAI(api_key=api_key)
            self._client_mode = "chat_client"
            return
        if openai is not None and hasattr(openai, "ChatCompletion"):
            openai.api_key = api_key
            self._client = openai.ChatCompletion
            self._client_mode = "chat"
            return

        raise RuntimeError(
            "La biblioteca oficial de OpenAI no está instalada. Añádela a requirements.txt"
        )

    def _invoke_model_inner(self, prompt: str) -> Any:
        if self._client is None:
            raise RuntimeError("El cliente de OpenAI no está inicializado")

        json_mode = {"type": "json_object"}
        if self._client_mode in {"chat_client", "chat"}:  # pragma: no cover - depends on env
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": prompt})
            params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "response_format": json_mode,
            }
            if self.max_output_tokens is not None:
                params["max_tokens"] = self.max_output_tokens

            if self._client_mode == "chat_client":
                endpoint = getattr(self._client, "chat", None)
                if endpoint is None:
                    raise RuntimeError("El cliente de OpenAI no expone chat.completions")
                completions = getattr(endpoint, "completions", None)
                if completions is None:
                    raise RuntimeError(
                        "El cliente de OpenAI no expone chat.completions.create"
                    )
                create = getattr(completions, "create", None)
                if create is None:
                    raise RuntimeError(
                        "El cliente de OpenAI no expone chat.completions.create"
                    )
                return create(**params)

            try:
                return self._client.create(**params)  # type: ignore[call-arg]
            except TypeError:  # pragma: no cover - fallback for legacy clients
                params.pop("response_format", None)
                return self._client.create(**params)  # type: ignore[call-arg]

        if self._client_mode == "custom":
            try:
                return self._client(
                    prompt=prompt,
                    system_prompt=self.system_prompt,
                    response_format=json_mode,
                )
            except TypeError:  # pragma: no cover - compatibility with user-provided clients
                return self._client(prompt=prompt, system_prompt=self.system_prompt)

        raise RuntimeError(f"Modo de cliente desconocido: {self._client_mode}")

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _invoke_model(
        self, prompt: str, _retry_tracker: Optional[MutableMapping[str, int]] = None
    ) -> Any:
        if _retry_tracker is not None:
            current = _retry_tracker.get("attempts", 0)
            try:
                current_int = int(current)
            except (TypeError, ValueError):
                current_int = 0
            current_int = max(current_int, 0) + 1
            _retry_tracker["attempts"] = current_int
        return self._invoke_model_inner(prompt)

    def _build_correction_prompt(self, original_prompt: str, invalid_text: str) -> str:
        return (
            f"{original_prompt}\n\n"
            "IMPORTANTE: La respuesta anterior no fue un JSON válido. "
            "Repite la evaluación y responde exclusivamente con un objeto JSON válido con las claves "
            "'score' (número), 'justification' (texto) y opcionalmente 'relevant_text' (texto).\n"
            "Respuesta inválida previa delimitada entre <<< >>>:\n<<<\n"
            f"{invalid_text}\n"
            ">>>"
        )

    def _normalise_response(self, raw: Any) -> AIResponse:
        text = self._extract_text(raw)
        metadata = self._extract_metadata(raw)
        payload = self._parse_json_payload(text, metadata)
        score = self._coerce_score(payload.get("score"))
        if score is None:
            raise InvalidAIResponseError(
                "El JSON devuelto por el modelo no contiene un 'score' numérico.",
                text=text,
                metadata=metadata,
            )
        justification = payload.get("justification")
        if not isinstance(justification, str):
            raise InvalidAIResponseError(
                "El JSON devuelto por el modelo debe incluir una 'justification' de texto.",
                text=text,
                metadata=metadata,
            )
        relevant = payload.get("relevant_text")
        if relevant is not None and not isinstance(relevant, str):
            relevant = str(relevant)
        return AIResponse(
            score=score,
            justification=justification,
            relevant_text=relevant,
            metadata=metadata,
        )

    def _extract_text(self, raw: Any) -> str:
        if raw is None:
            return ""
        text_blocks: list[str] = []
        if hasattr(raw, "output"):  # responses API
            output = getattr(raw, "output")
            for item in output or []:
                content = getattr(item, "content", [])
                for chunk in content or []:
                    text = getattr(chunk, "text", None)
                    if text:
                        text_blocks.append(str(text))
        elif hasattr(raw, "choices"):
            choices = getattr(raw, "choices")
            for choice in choices or []:
                message = getattr(choice, "message", None)
                if isinstance(message, Mapping):
                    content = message.get("content")
                    if isinstance(content, str):
                        text_blocks.append(content)
                        continue
                    if isinstance(content, Sequence):
                        for item in content:
                            if isinstance(item, Mapping) and "text" in item:
                                text_blocks.append(str(item["text"]))
                            elif isinstance(item, str):
                                text_blocks.append(item)
                        continue
                    text = content
                else:
                    text = getattr(choice, "text", None)
                if text:
                    text_blocks.append(str(text))
        elif isinstance(raw, Mapping):
            if "output" in raw:
                for item in raw.get("output", []):
                    for chunk in item.get("content", []):
                        if "text" in chunk:
                            text_blocks.append(str(chunk["text"]))
            elif "choices" in raw:
                for choice in raw.get("choices", []):
                    message = choice.get("message")
                    if isinstance(message, Mapping) and "content" in message:
                        content = message.get("content")
                        if isinstance(content, str):
                            text_blocks.append(content)
                        elif isinstance(content, Sequence):
                            for item in content:
                                if isinstance(item, Mapping) and "text" in item:
                                    text_blocks.append(str(item["text"]))
                                elif isinstance(item, str):
                                    text_blocks.append(item)
                        else:
                            text_blocks.append(str(content))
                    elif "text" in choice:
                        text_blocks.append(str(choice["text"]))
        if not text_blocks:
            try:
                return str(raw)
            except Exception:  # pragma: no cover - defensive
                return ""
        return "\n".join(text_blocks)

    def _parse_json_payload(
        self, text: str, metadata: Optional[Mapping[str, Any]] = None
    ) -> Dict[str, Any]:
        stripped = text.strip()
        if not stripped:
            raise InvalidAIResponseError(
                "La respuesta del modelo está vacía.", text=text, metadata=metadata
            )
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise InvalidAIResponseError(
                "La respuesta del modelo no es un JSON válido.", text=text, metadata=metadata
            ) from exc
        if not isinstance(data, MutableMapping):
            raise InvalidAIResponseError(
                "El JSON devuelto por el modelo no es un objeto.",
                text=text,
                metadata=metadata,
            )
        return dict(data)

    def _extract_metadata(self, raw: Any) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {"provider": "openai"}
        request_id = getattr(raw, "id", None) or (raw.get("id") if isinstance(raw, Mapping) else None)
        if request_id:
            metadata["provider_request_id"] = str(request_id)
        usage = getattr(raw, "usage", None)
        if usage:
            usage_dict = self._object_to_dict(usage)
            if usage_dict:
                metadata["usage"] = usage_dict
        elif isinstance(raw, Mapping) and "usage" in raw:
            usage_value = raw.get("usage")
            if isinstance(usage_value, Mapping):
                metadata["usage"] = dict(usage_value)
        return metadata

    def _object_to_dict(self, value: Any) -> Optional[Dict[str, Any]]:
        if value is None:
            return None
        if isinstance(value, Mapping):
            return dict(value)
        for attr in ("model_dump", "dict"):
            method = getattr(value, attr, None)
            if callable(method):
                try:
                    data = method()
                    if isinstance(data, Mapping):
                        return dict(data)
                except Exception:  # pragma: no cover - defensive
                    continue
        if hasattr(value, "__dict__"):
            return dict(value.__dict__)
        return None

    def _coerce_score(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            self.logger.warning("Puntaje no numérico recibido desde el modelo: %s", value)
            return None

    def _normalise_levels(self, levels: Sequence[float]) -> tuple[float, ...]:
        normalised: list[float] = []
        for level in levels:
            try:
                numeric = float(level)
            except (TypeError, ValueError):
                self.logger.debug("Nivel no numérico descartado: %s", level)
                continue
            normalised.append(numeric)

        if not normalised:
            return tuple()

        return tuple(sorted({float(value) for value in normalised}))

    def _enforce_discrete_score(
        self, score: Optional[float], *, levels: Sequence[float]
    ) -> Optional[float]:
        if score is None:
            return None

        if not levels:
            return score

        closest = min(levels, key=lambda candidate: (abs(candidate - score), candidate))
        if abs(closest - score) > 1e-9:
            self.logger.debug(
                "Ajustando puntaje %s al nivel discreto permitido %s", score, closest
            )
        return float(closest)

    def _resolve_levels(
        self,
        question: Optional[Mapping[str, Any]],
        criteria: Optional[Mapping[str, Any]],
    ) -> Sequence[float]:
        question_levels = self._extract_question_levels(question)
        if question_levels:
            return question_levels

        report_type = None
        if isinstance(criteria, Mapping):
            raw = criteria.get("tipo_informe")
            if isinstance(raw, str):
                report_type = raw.strip().lower()

        if report_type == "institucional":
            return self._INSTITUTIONAL_LEVELS
        if report_type in {"pn", "politica_nacional", "política_nacional", "política nacional"}:
            return self._PN_LEVELS

        return ()

    def _extract_question_levels(
        self, question: Optional[Mapping[str, Any]]
    ) -> Sequence[float]:
        if not isinstance(question, Mapping):
            return ()
        niveles = question.get("niveles")
        if not niveles:
            return ()
        values: list[float] = []
        for level in niveles:
            candidate: Any
            if isinstance(level, Mapping):
                candidate = level.get("valor")
            else:
                candidate = level
            try:
                value = float(candidate)
            except (TypeError, ValueError):
                continue
            values.append(value)
        if not values:
            return ()
        unique_sorted = sorted({float(v) for v in values})
        return tuple(unique_sorted)

    def _extract_report_type(self, criteria: Optional[Mapping[str, Any]]) -> Optional[str]:
        if not isinstance(criteria, Mapping):
            return None
        raw = criteria.get("tipo_informe")
        if isinstance(raw, str):
            return raw.strip().lower() or None
        return None


__all__ = [
    "AIResponse",
    "BaseAIService",
    "InvalidAIResponseError",
    "MockAIService",
    "OpenAIService",
]