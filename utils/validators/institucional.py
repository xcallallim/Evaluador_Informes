"""Validador de archivos metodológicos institucionales."""

from __future__ import annotations

from typing import Any, Dict, List

from .base import ValidationResult, almost_equal, sum_ponderaciones
from . import register_validator


def validate_institutional_schema(data: Dict[str, Any]) -> ValidationResult:
    """Validate the institutional methodology schema structure."""

    result = ValidationResult()

    dimensiones_defecto = data.get("dimensiones_defecto", {})
    if dimensiones_defecto:
        total_dimensiones = sum(float(value) for value in dimensiones_defecto.values())
        if not almost_equal(total_dimensiones, 1.0):
            result.errors.append(
                "Las ponderaciones en 'dimensiones_defecto' suman "
                f"{total_dimensiones:.6f} (deben sumar 1.0)."
            )
    else:
        result.errors.append("Falta la clave 'dimensiones_defecto'.")

    secciones = data.get("secciones", [])
    if secciones:
        total_secciones = sum_ponderaciones(secciones, "ponderacion")
        if not almost_equal(total_secciones, 1.0):
            result.errors.append(
                f"Las ponderaciones de las secciones suman {total_secciones:.6f} (deben sumar 1.0)."
            )
    else:
        result.errors.append("No se encontraron secciones en el archivo.")

    for seccion in secciones:
        nombre = seccion.get("titulo", seccion.get("id_segmenter", "<sin_nombre>"))
        dimensiones = seccion.get("dimensiones", [])
        if not dimensiones:
            result.errors.append(f"La sección '{nombre}' no tiene dimensiones definidas.")
            continue

        total_dim = sum_ponderaciones(dimensiones, "ponderacion")
        if not almost_equal(total_dim, 1.0):
            result.errors.append(
                "Las ponderaciones de dimensiones en la sección "
                f"'{nombre}' suman {total_dim:.6f} (deben sumar 1.0)."
            )

        for dimension in dimensiones:
            dim_name = dimension.get("nombre", "<sin_nombre>")
            preguntas = dimension.get("preguntas", [])
            if not preguntas:
                result.errors.append(
                    f"La dimensión '{dim_name}' en la sección '{nombre}' no tiene preguntas configuradas."
                )
            else:
                total_preguntas = sum_ponderaciones(preguntas, "ponderacion")
                if not almost_equal(total_preguntas, 1.0):
                    result.errors.append(
                        "Las ponderaciones de preguntas en la dimensión "
                        f"'{dim_name}' de la sección '{nombre}' suman {total_preguntas:.6f} (deben sumar 1.0)."
                    )

            niveles = dimension.get("niveles")
            escala = dimension.get("tipo_escala")
            if niveles is None:
                if escala == "likert" and "escala_likert_defecto" not in data:
                    result.errors.append(
                        "La dimensión "
                        f"'{dim_name}' en la sección '{nombre}' no define 'niveles' y no existe 'escala_likert_defecto'."
                    )
                continue

            if not isinstance(niveles, list) or not niveles:
                result.errors.append(
                    f"La dimensión '{dim_name}' en la sección '{nombre}' tiene 'niveles' inválidos o vacíos."
                )
                continue

            valores: List[Any] = []
            for nivel in niveles:
                if "valor" not in nivel or "descripcion" not in nivel:
                    result.errors.append(
                        f"Cada nivel de la dimensión '{dim_name}' en la sección '{nombre}' debe tener 'valor' y 'descripcion'."
                    )
                    break
                valores.append(nivel["valor"])

            if len(valores) != len(set(valores)):
                result.warnings.append(
                    f"Existen valores duplicados en los niveles de la dimensión '{dim_name}' de la sección '{nombre}'."
                )

    return result


register_validator("institucional", validate_institutional_schema)