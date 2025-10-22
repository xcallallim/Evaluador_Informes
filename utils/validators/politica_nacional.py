"""Validador de archivos de metodología de políticas nacionales."""

from __future__ import annotations

from typing import Any, Dict, List

from . import register_validator
from .base import ValidationResult, almost_equal, sum_ponderaciones


def _validate_niveles(
    *,
    niveles: Any,
    pregunta_id: str,
    bloque_nombre: str,
    escala_min: float | None,
    escala_max: float | None,
    result: ValidationResult,
) -> None:
    """Validar los niveles de preguntas asegurando la estructura y los límites numéricos."""

    if niveles is None:
        return

    if not isinstance(niveles, list) or not niveles:
        result.errors.append(
            f"La pregunta '{pregunta_id}' en el bloque '{bloque_nombre}' tiene 'niveles' inválidos o vacíos."
        )
        return

    valores: List[float] = []
    for nivel in niveles:
        if "valor" not in nivel or "descripcion" not in nivel:
            result.errors.append(
                f"Cada nivel de la pregunta '{pregunta_id}' en el bloque '{bloque_nombre}' debe tener 'valor' y 'descripcion'."
            )
            return
        valor = nivel["valor"]
        valores.append(valor)
        if escala_min is not None and valor < escala_min:
            result.errors.append(
                f"El nivel con valor {valor} de la pregunta '{pregunta_id}' está por debajo del mínimo {escala_min}."
            )
        if escala_max is not None and valor > escala_max:
            result.errors.append(
                f"El nivel con valor {valor} de la pregunta '{pregunta_id}' supera el máximo {escala_max}."
            )

    if len(valores) != len(set(valores)):
        result.warnings.append(
            f"Existen valores duplicados en los niveles de la pregunta '{pregunta_id}' del bloque '{bloque_nombre}'."
        )


def validate_politica_nacional_schema(data: Dict[str, Any]) -> ValidationResult:
    """Validar la estructura del esquema metodológico de política nacional."""

    result = ValidationResult()

    escala = data.get("escala", {})
    if not escala:
        result.errors.append("Falta la definición de 'escala'.")
        escala_min = escala_max = None
    else:
        escala_min = escala.get("min")
        escala_max = escala.get("max")
        for clave in ("tipo", "tipo_valores", "min", "max"):
            if clave not in escala:
                result.errors.append(f"Falta la clave '{clave}' dentro de 'escala'.")

        if escala_min is not None and escala_max is not None and escala_min >= escala_max:
            result.errors.append("El rango de la 'escala' es inválido: 'min' debe ser menor que 'max'.")

    bloques = data.get("bloques", [])
    if not bloques:
        result.errors.append("No se encontraron bloques de preguntas en el archivo.")
        return result

    total_bloques = sum_ponderaciones(bloques, "ponderacion")
    if not almost_equal(total_bloques, 1.0, tolerance=1e-3):
        result.errors.append(
            f"Las ponderaciones de los bloques suman {total_bloques:.6f} (deben sumar 1.0)."
        )

    for bloque in bloques:
        bloque_nombre = bloque.get("titulo") or bloque.get("id_segmenter") or "<sin_nombre>"
        preguntas = bloque.get("preguntas", [])
        if not preguntas:
            result.errors.append(f"El bloque '{bloque_nombre}' no tiene preguntas configuradas.")
            continue

        total_preguntas = sum_ponderaciones(preguntas, "ponderacion")
        if not almost_equal(total_preguntas, 1.0, tolerance=1e-3):
            result.errors.append(
                f"Las ponderaciones de las preguntas del bloque '{bloque_nombre}' suman {total_preguntas:.6f} (deben sumar 1.0)."
            )

        bloque_tipo = bloque.get("tipo")
        bloque_escala = bloque.get("tipo_escala")

        for pregunta in preguntas:
            pregunta_id = pregunta.get("id", "<sin_id>")
            niveles = pregunta.get("niveles")
            pregunta_escala = pregunta.get("tipo_escala") or bloque_escala

            if niveles is None:
                if pregunta_escala != "manual":
                    result.errors.append(
                        f"La pregunta '{pregunta_id}' en el bloque '{bloque_nombre}' no define 'niveles' ni especifica una escala manual."
                    )
                continue

            if bloque_tipo == "manual" and pregunta_escala != "manual":
                result.errors.append(
                    f"La pregunta '{pregunta_id}' en el bloque manual '{bloque_nombre}' debe declararse con 'tipo_escala' manual."
                )

            _validate_niveles(
                niveles=niveles,
                pregunta_id=pregunta_id,
                bloque_nombre=bloque_nombre,
                escala_min=escala_min,
                escala_max=escala_max,
                result=result,
            )

    return result


register_validator("politica_nacional", validate_politica_nacional_schema)