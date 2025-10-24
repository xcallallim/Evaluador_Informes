"""Utility helpers to assess the quality of prompts before sending them to the AI.

The validation layer implemented in this module focuses on three dimensions:

* **Estructural** – el texto debe poseer longitud suficiente, delimitadores y
  bloques esperados (fragmentos, JSON de respuesta, etc.).
* **Metodológica** – el prompt tiene que conservar las secciones clave de la
  metodología CEPLAN (objetivo, escala, instrucciones de salida, ...).
* **Semántica** – se verifica el idioma predominante, la coherencia de la escala
  con el contexto y que el truncado del texto sea explícito.

Cada regla aporta una penalización ponderada. El índice de calidad resultante se
mantiene en el rango ``[0.0, 1.0]`` y permite clasificar el prompt según los
umbrales descritos por la operación del evaluador.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence

_SPANISH_MARKERS = {
    " el ",
    " la ",
    " los ",
    " las ",
    " que ",
    " de ",
    " del ",
    " en ",
    " para ",
    " con ",
    " según ",
    " evaluación ",
}

_SPANISH_COMMON_WORDS = {
    "el",
    "la",
    "los",
    "las",
    "que",
    "de",
    "del",
    "en",
    "para",
    "con",
    "una",
    "sobre",
    "según",
    "informe",
    "evaluación",
}

_SPANISH_ACCENTED_CHARS = set("áéíóúüñ")

try:  # pragma: no cover - import opcional
    from langdetect import DetectorFactory, detect_langs  # type: ignore
except ImportError:  # pragma: no cover - fallback cuando langdetect no está instalado
    detect_langs = None  # type: ignore[assignment]
else:  # pragma: no cover - depende de la librería externa
    DetectorFactory.seed = 0


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PromptValidationResult:
    """Resultado de la validación estructural, metodológica y semántica."""

    is_valid: bool
    quality_score: float
    alerts: list[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def quality_band(self) -> str:
        """Clasificación textual del puntaje de calidad."""

        return self.metadata.get("quality_band", "")


class PromptValidator:
    """Valida prompts según reglas ponderadas y produce un índice de calidad.

    Parameters
    ----------
    min_length:
        Longitud mínima esperada para considerar que el prompt posee suficiente
        contexto.
    max_length:
        Longitud máxima tolerada antes de considerar que el prompt podría estar
        truncado o contener ruido excesivo.
    weights:
        Penalizaciones asociadas a cada regla de validación. Las llaves
        esperadas son ``length``, ``structure``, ``json``, ``scale``,
        ``language``, ``truncation`` y ``methodology``. Si no se indican, se
        utilizarán valores por defecto que suman ``1.0``.
    """

    VERSION: str = "1.0.0"

    DEFAULT_WEIGHTS: Mapping[str, float] = {
        "length": 0.20,
        "structure": 0.15,
        "json": 0.20,
        "scale": 0.15,
        "language": 0.20,
        "truncation": 0.05,
        "methodology": 0.05,
    }

    QUALITY_THRESHOLDS: Mapping[str, float] = {
        "alta": 0.90,
        "aceptable": 0.75,
        "regular": 0.60,
    }

    QUALITY_DECORATIONS: Mapping[str, Mapping[str, str]] = {
        "alta": {"symbol": "🟢", "color": "#2ecc71"},
        "aceptable": {"symbol": "🟡", "color": "#f1c40f"},
        "regular": {"symbol": "🟠", "color": "#e67e22"},
        "rechazada": {"symbol": "🔴", "color": "#e74c3c"},
    }

    def __init__(
        self,
        *,
        min_length: int = 400,
        max_length: int = 14_000,
        weights: Optional[Mapping[str, float]] = None,
    ) -> None:
        if min_length <= 0:
            raise ValueError("min_length debe ser un entero positivo")
        if max_length <= min_length:
            raise ValueError("max_length debe ser mayor que min_length")
        self.min_length = min_length
        self.max_length = max_length
        self.weights = dict(self.DEFAULT_WEIGHTS)
        if weights is not None:
            self.weights.update(weights)
        missing = set(self.DEFAULT_WEIGHTS) - set(self.weights)
        if missing:
            raise ValueError(f"Pesos faltantes para las reglas: {', '.join(sorted(missing))}")
        negative = [name for name, value in self.weights.items() if value < 0]
        if negative:
            raise ValueError(
                "Los pesos no pueden ser negativos: " + ", ".join(sorted(negative))
            )
        total_weight = sum(self.weights.values())
        if total_weight <= 0:
            raise ValueError("La suma de los pesos debe ser mayor que cero")
        self.weights = {name: value / total_weight for name, value in self.weights.items()}
        self.base_rejection_threshold = self.QUALITY_THRESHOLDS["regular"]

    # ------------------------------------------------------------------
    # API principal
    def validate(
        self,
        prompt: str,
        context: Optional[Mapping[str, Any]] = None,
    ) -> PromptValidationResult:
        """Evalúa el prompt y retorna un :class:`PromptValidationResult`.

        Parameters
        ----------
        prompt:
            Texto completo que se enviará al modelo de lenguaje.
        context:
            Información adicional para realizar comprobaciones específicas. Se
            aceptan claves como ``report_type``, ``expected_scale_values``,
            ``expected_scale_label``, ``was_truncated`` o
            ``truncation_marker``.
        """

        if not isinstance(prompt, str):
            raise TypeError("prompt debe ser una cadena de texto")
        cleaned_prompt = prompt.strip()
        if not cleaned_prompt:
            raise ValueError("prompt no puede estar vacío")

        ctx: MutableMapping[str, Any] = dict(context or {})
        alerts: list[str] = []
        penalties: list[dict[str, Any]] = []
        prompt_lower = cleaned_prompt.lower()
        metrics: Dict[str, Any] = {"chars": len(cleaned_prompt)}
        score = 1.0

        score = self._check_length(
            cleaned_prompt, prompt_lower, ctx, alerts, penalties, metrics, score
        )
        score = self._check_delimiters(
            cleaned_prompt, prompt_lower, ctx, alerts, penalties, metrics, score
        )
        score = self._check_json_block(
            cleaned_prompt, prompt_lower, ctx, alerts, penalties, metrics, score
        )
        score = self._check_scale_alignment(
            cleaned_prompt, prompt_lower, ctx, alerts, penalties, metrics, score
        )
        score = self._check_language(
            cleaned_prompt, prompt_lower, alerts, penalties, metrics, score
        )
        score = self._check_truncation(
            cleaned_prompt, prompt_lower, ctx, alerts, penalties, metrics, score
        )
        score = self._check_methodology_structure(
            cleaned_prompt, prompt_lower, ctx, alerts, penalties, metrics, score
        )

        score = max(0.0, min(1.0, round(score, 4)))
        quality_band = self._quality_band(score)
        decorations = self.QUALITY_DECORATIONS.get(quality_band, {})
        metrics["score_initial"] = 1.0
        metrics["score_final"] = score
        metrics["penalties"] = penalties
        metrics["quality_band"] = quality_band
        metrics["quality_symbol"] = decorations.get("symbol")
        metrics["quality_color"] = decorations.get("color")
        metrics["alerts_count"] = len(alerts)

        dynamic_threshold = self._dynamic_threshold(len(cleaned_prompt))
        metrics["rejection_threshold"] = dynamic_threshold
        is_valid = score >= dynamic_threshold
        return PromptValidationResult(is_valid=is_valid, quality_score=score, alerts=alerts, metadata=metrics)

    # ------------------------------------------------------------------
    # Reglas individuales
    def _check_length(
        self,
        prompt: str,
        prompt_lower: str,
        ctx: Mapping[str, Any],
        alerts: list[str],
        penalties: list[dict[str, Any]],
        metrics: Dict[str, Any],
        score: float,
    ) -> float:
        min_length = int(ctx.get("min_length", self.min_length))
        max_length = int(ctx.get("max_length", self.max_length))
        prompt_length = len(prompt)
        metrics["min_length"] = min_length
        metrics["max_length"] = max_length
        metrics["chars"] = prompt_length

        if prompt_length < min_length:
            message = (
                f"Longitud insuficiente: {prompt_length} caracteres (mínimo esperado {min_length})."
            )
            score = self._apply_penalty("length", message, alerts, penalties, score)

        if prompt_length > max_length:
            message = (
                f"Longitud excesiva: {prompt_length} caracteres (máximo recomendado {max_length})."
            )
            score = self._apply_penalty("length", message, alerts, penalties, score)

        return score

    def _check_delimiters(
        self,
        prompt: str,
        prompt_lower: str,
        ctx: Mapping[str, Any],
        alerts: list[str],
        penalties: list[dict[str, Any]],
        metrics: Dict[str, Any],
        score: float,
    ) -> float:
        start_marker = str(ctx.get("fragment_start_marker", "<<<INICIO_FRAGMENTO>>>"))
        end_marker = str(ctx.get("fragment_end_marker", "<<<FIN_FRAGMENTO>>>"))
        metrics["fragment_start_marker"] = start_marker
        metrics["fragment_end_marker"] = end_marker

        if start_marker.lower() not in prompt_lower or end_marker.lower() not in prompt_lower:
            message = "Faltan los delimitadores esperados para el fragmento del texto."
            score = self._apply_penalty("structure", message, alerts, penalties, score)
        return score

    def _check_json_block(
        self,
        prompt: str,
        prompt_lower: str,
        ctx: Mapping[str, Any],
        alerts: list[str],
        penalties: list[dict[str, Any]],
        metrics: Dict[str, Any],
        score: float,
    ) -> float:
        required_keys: Sequence[str] = tuple(
            ctx.get("json_required_keys", ("score", "justification"))
        )
        optional_keys: Sequence[str] = tuple(ctx.get("json_optional_keys", ("relevant_text",)))
        json_block = self._extract_json_block(prompt, required_keys, optional_keys)
        metrics["json_block_present"] = json_block is not None
        if json_block is None:
            message = "No se encontró el bloque de salida en formato JSON."
            return self._apply_penalty("json", message, alerts, penalties, score)

        sanitized = re.sub(r"<[^>]+>", '"placeholder"', json_block)
        sanitized = re.sub(r"\s+", " ", sanitized).strip()
        try:
            json.loads(sanitized)
        except json.JSONDecodeError:
            message = "El bloque JSON de ejemplo no es válido tras normalizar los marcadores."
            score = self._apply_penalty("json", message, alerts, penalties, score)
        else:
            metrics["json_block_valid"] = True
            metrics["json_required_keys"] = list(required_keys)
            optional_present = [
                key for key in optional_keys if key and re.search(rf'"{re.escape(key)}"', json_block)
            ]
            metrics["json_optional_keys_present"] = optional_present
            metrics["json_optional_keys_missing"] = [
                key for key in optional_keys if key and key not in optional_present
            ]
        return score

    def _check_scale_alignment(
        self,
        prompt: str,
        prompt_lower: str,
        ctx: Mapping[str, Any],
        alerts: list[str],
        penalties: list[dict[str, Any]],
        metrics: Dict[str, Any],
        score: float,
    ) -> float:
        expected_values = ctx.get("expected_scale_values")
        expected_label = ctx.get("expected_scale_label")
        metrics["expected_scale_values"] = expected_values
        metrics["expected_scale_label"] = expected_label

        if expected_label:
            if str(expected_label).lower() not in prompt_lower:
                message = "La etiqueta de escala esperada no se encuentra en el prompt."
                score = self._apply_penalty("scale", message, alerts, penalties, score)

        if expected_values:
            normalized_expected = {
                self._normalize_scale_number(value) for value in expected_values if value is not None
            }
            normalized_expected.discard(None)

            scale_section = prompt
            if expected_label:
                label_index = prompt_lower.find(str(expected_label).lower())
                if label_index != -1:
                    scale_section = prompt[label_index : label_index + 500]

            prompt_numbers = {
                self._normalize_scale_number(token)
                for token in re.findall(r"-?\d+(?:[.,]\d+)?", scale_section)
            }
            prompt_numbers.discard(None)
            metrics["scale_expected_values"] = sorted(normalized_expected)
            metrics["scale_prompt_values"] = sorted(prompt_numbers)

            if normalized_expected and not normalized_expected.issubset(prompt_numbers):
                message = "Los valores permitidos de la escala no coinciden con el prompt."
                score = self._apply_penalty("scale", message, alerts, penalties, score)
        return score

    def _check_language(
        self,
        prompt: str,
        prompt_lower: str,
        alerts: list[str],
        penalties: list[dict[str, Any]],
        metrics: Dict[str, Any],
        score: float,
    ) -> float:
        normalized_prompt = prompt.strip()
        prompt_length = len(normalized_prompt)

        text = f" {prompt_lower} "
        marker_matches = sum(1 for marker in _SPANISH_MARKERS if marker in text)
        marker_ratio = marker_matches / len(_SPANISH_MARKERS)

        words = re.findall(r"[a-záéíóúüñ]+", prompt_lower)
        spanish_word_hits = sum(1 for word in words if word in _SPANISH_COMMON_WORDS)
        word_ratio = spanish_word_hits / max(len(words), 1)

        accented_chars = sum(1 for char in prompt_lower if char in _SPANISH_ACCENTED_CHARS)
        alpha_chars = sum(1 for char in normalized_prompt if char.isalpha())
        accented_ratio = accented_chars / alpha_chars if alpha_chars else 0.0

        metrics["spanish_markers_ratio"] = round(marker_ratio, 3)
        metrics["spanish_markers_matches"] = marker_matches
        metrics["spanish_word_ratio"] = round(word_ratio, 3)
        metrics["spanish_accented_ratio"] = round(accented_ratio, 3)

        langdetect_confidence = None
        if detect_langs is not None and prompt_length >= 50:
            try:
                detections = detect_langs(normalized_prompt)
            except Exception:  # pragma: no cover - depends on external library behaviour
                detections = []
            else:
                for detection in detections:
                    if detection.lang == "es":
                        langdetect_confidence = detection.prob
                        break
        metrics["spanish_langdetect_confidence"] = (
            round(langdetect_confidence, 3) if langdetect_confidence is not None else None
        )

        evidence_score = max(marker_ratio, word_ratio)
        evidence_score = (evidence_score * 0.7) + (accented_ratio * 0.3)
        if langdetect_confidence is not None:
            evidence_score = (evidence_score + langdetect_confidence) / 2

        metrics["spanish_evidence_score"] = round(evidence_score, 3)

        min_length_for_penalty = 80
        adaptive_threshold = 0.22 if prompt_length < 250 else 0.28

        if prompt_length >= min_length_for_penalty and evidence_score < adaptive_threshold:
            message = "El contenido no parece estar mayoritariamente en español."
            score = self._apply_penalty("language", message, alerts, penalties, score)
        return score

    def _check_truncation(
        self,
        prompt: str,
        prompt_lower: str,
        ctx: Mapping[str, Any],
        alerts: list[str],
        penalties: list[dict[str, Any]],
        metrics: Dict[str, Any],
        score: float,
    ) -> float:
        truncation_marker = str(ctx.get("truncation_marker", "[texto truncado]"))
        fragment_start_marker = str(ctx.get("fragment_start_marker", "<<<INICIO_FRAGMENTO>>>"))
        fragment_end_marker = str(ctx.get("fragment_end_marker", "<<<FIN_FRAGMENTO>>>"))
        was_truncated = bool(ctx.get("was_truncated", False))
        truncation_lower = truncation_marker.lower()
        present = truncation_lower in prompt_lower
        occurrences = len(re.findall(re.escape(truncation_lower), prompt_lower))
        metrics["truncation_marker"] = truncation_marker
        metrics["was_truncated"] = was_truncated
        metrics["truncation_marker_present"] = present
        metrics["truncation_marker_occurrences"] = occurrences

        if was_truncated and not present:
            message = "El fragmento fue truncado pero el prompt no lo indica."
            score = self._apply_penalty("truncation", message, alerts, penalties, score)
        elif not was_truncated and present:
            message = "El prompt marca truncado un fragmento que no debería estarlo."
            score = self._apply_penalty("truncation", message, alerts, penalties, score)
        elif present:
            if occurrences > 1:
                message = "El marcador de truncado aparece múltiples veces en el fragmento."
                score = self._apply_penalty("truncation", message, alerts, penalties, score)
            start_index = prompt_lower.find(fragment_start_marker.lower())
            end_index = prompt_lower.find(fragment_end_marker.lower())
            marker_index = prompt_lower.find(truncation_lower)
            if marker_index != -1 and end_index != -1 and marker_index < end_index:
                tail_segment = prompt[marker_index + len(truncation_marker) : end_index]
            else:
                tail_segment = prompt[marker_index + len(truncation_marker) :] if marker_index != -1 else ""
            trailing_text = tail_segment.strip()
            metrics["truncation_trailing_chars"] = len(trailing_text)
            if trailing_text:
                trailing_alpha = sum(1 for char in trailing_text if char.isalpha())
            else:
                trailing_alpha = 0
            metrics["truncation_trailing_alpha"] = trailing_alpha
            if trailing_alpha > 40:
                message = (
                    "El marcador de truncado parece estar desplazado: aún queda mucho texto después de él."
                )
                score = self._apply_penalty("truncation", message, alerts, penalties, score)
        return score

    def _check_methodology_structure(
        self,
        prompt: str,
        prompt_lower: str,
        ctx: Mapping[str, Any],
        alerts: list[str],
        penalties: list[dict[str, Any]],
        metrics: Dict[str, Any],
        score: float,
    ) -> float:
        alias_map = self._resolve_section_aliases(ctx)
        detected: Dict[str, str] = {}
        for category, aliases in alias_map.items():
            for alias in aliases:
                if alias.lower() in prompt_lower:
                    detected[category] = alias
                    break
        coverage = len(detected) / len(alias_map) if alias_map else 1.0
        metrics["methodology_sections_detected"] = detected
        metrics["methodology_missing_sections"] = [
            category for category in alias_map if category not in detected
        ]
        metrics["methodology_coverage"] = round(coverage, 3)

        if coverage < 0.6:
            message = "El prompt no contiene todas las secciones metodológicas esperadas."
            score = self._apply_penalty("methodology", message, alerts, penalties, score)
        return score

    # ------------------------------------------------------------------
    # Utilidades
    def _apply_penalty(
        self,
        rule: str,
        message: str,
        alerts: list[str],
        penalties: list[dict[str, Any]],
        score: float,
    ) -> float:
        weight = self.weights.get(rule, 0.0)
        if weight <= 0:
            return score
        alerts.append(message)
        penalties.append({"rule": rule, "penalty": weight, "message": message})
        logger.debug("Aplicando penalización", extra={
            "rule": rule,
            "penalty": weight,
            "message": message,
        })
        return max(0.0, score - weight)

    def _dynamic_threshold(self, length: int) -> float:
        if length < 1000:
            return 0.70
        if length > 6000:
            return 0.55
        return self.base_rejection_threshold

    @staticmethod
    def _quality_band(score: float) -> str:
        if score >= PromptValidator.QUALITY_THRESHOLDS["alta"]:
            return "alta"
        if score >= PromptValidator.QUALITY_THRESHOLDS["aceptable"]:
            return "aceptable"
        if score >= PromptValidator.QUALITY_THRESHOLDS["regular"]:
            return "regular"
        return "rechazada"

    @staticmethod
    def _format_scale_value(value: Any) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        if number.is_integer():
            return str(int(number))
        return str(number).rstrip("0").rstrip(".")

    @staticmethod
    def _normalize_scale_number(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            normalized = value.strip().replace(" ", "")
            normalized = normalized.replace(",", ".")
            if not normalized:
                return None
            try:
                number = float(normalized)
            except ValueError:
                return normalized.lower()
        else:
            try:
                number = float(value)
            except (TypeError, ValueError):
                return str(value)
        if number.is_integer():
            return str(int(number))
        return str(round(number, 6)).rstrip("0").rstrip(".")

    @staticmethod
    def _extract_json_block(
        prompt: str,
        required_keys: Sequence[str],
        optional_keys: Sequence[str],
    ) -> Optional[str]:
        required_patterns = [re.compile(rf'"{re.escape(key)}"') for key in required_keys if key]
        optional_patterns = [re.compile(rf'"{re.escape(key)}"') for key in optional_keys if key]

        depth = 0
        start_index: Optional[int] = None
        for index, char in enumerate(prompt):
            if char == "{":
                if depth == 0:
                    start_index = index
                depth += 1
            elif char == "}":
                if depth == 0:
                    continue
                depth -= 1
                if depth == 0 and start_index is not None:
                    candidate = prompt[start_index : index + 1]
                    if all(pattern.search(candidate) for pattern in required_patterns):
                        if optional_patterns and not any(pattern.search(candidate) for pattern in optional_patterns):
                            # No se hallaron claves opcionales; continuar buscando una versión más completa
                            continue
                        return candidate
        # Si no se encontró un bloque que cumpla con todos los opcionales,
        # aceptar el primero que tenga únicamente las claves obligatorias.
        if required_patterns:
            fallback_pattern = re.compile(r"\{[^{}]*\}", re.DOTALL)
            for match in fallback_pattern.finditer(prompt):
                candidate = match.group(0)
                if all(pattern.search(candidate) for pattern in required_patterns):
                    return candidate
        return None

    def _resolve_section_aliases(self, ctx: Mapping[str, Any]) -> Mapping[str, Sequence[str]]:
        aliases = ctx.get("expected_sections_aliases")
        if isinstance(aliases, Mapping):
            return {
                str(category): tuple(str(alias) for alias in aliases_list if alias)
                for category, aliases_list in aliases.items()
            }

        legacy = ctx.get("expected_sections")
        if isinstance(legacy, Mapping):
            return {
                str(category): tuple(str(alias) for alias in aliases_list if alias)
                for category, aliases_list in legacy.items()
            }
        if isinstance(legacy, Sequence) and not isinstance(legacy, (str, bytes)):
            return {str(name): (str(name),) for name in legacy}

        return {
            "rol": (
                "rol asignado",
                "rol del evaluador",
                "rol encargado",
            ),
            "objetivo": (
                "objetivo",
                "propósito",
                "meta del análisis",
            ),
            "escala": (
                "escala de valoración",
                "escala de calificación",
                "escala de evaluación",
            ),
            "instrucciones": (
                "instrucciones de respuesta",
                "instrucciones de salida",
                "indicaciones para responder",
            ),
            "formato": (
                "formato de salida",
                "formato de respuesta",
                "estructura de salida",
            ),
        }


def validate_prompt(prompt: str, context: Optional[Mapping[str, Any]] = None) -> PromptValidationResult:
    """Valida un prompt usando :class:`PromptValidator` con parámetros por defecto."""

    validator = PromptValidator()
    return validator.validate(prompt, context=context)