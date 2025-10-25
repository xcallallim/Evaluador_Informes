"""Validador de archivos metodológicos institucionales."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .base import ValidationResult, almost_equal, sum_ponderaciones
from . import register_validator


def _require_keys(
    payload: Dict[str, Any], required: Sequence[str], *, context: str, result: ValidationResult
) -> None:
    for key in required:
        if key not in payload:
            result.errors.append(f"Falta la clave obligatoria '{key}' en {context}.")


def _collect_dict_items(
    value: Any,
    *,
    context: str,
    item_label: str,
    empty_message: str,
    result: ValidationResult,
) -> List[Dict[str, Any]]:
    """Return a list of dictionaries, recording schema errors when the structure is invalid."""

    if not isinstance(value, list):
        result.errors.append(f"{context} debe ser una lista de {item_label}.")
        return []

    if not value:
        result.errors.append(empty_message)
        return []

    items: List[Dict[str, Any]] = []
    for index, element in enumerate(value, start=1):
        if not isinstance(element, dict):
            result.errors.append(
                f"El elemento #{index} en {context} debe ser un objeto JSON con la estructura esperada."
            )
            continue
        items.append(element)

    if not items:
        result.errors.append(empty_message)

    return items


@dataclass
class LikertScaleInfo:
    """Container with metadata extracted from a Likert scale definition."""

    values: List[float]
    value_range: Optional[Tuple[float, float]]


def _validate_likert_levels(
    niveles: Iterable[Dict[str, Any]],
    *,
    context: str,
    result: ValidationResult,
) -> LikertScaleInfo:
    valores: List[float] = []
    rango: Optional[Tuple[float, float]] = None
    for index, nivel in enumerate(niveles, start=1):
        if not isinstance(nivel, dict):
            result.errors.append(f"El nivel {index} en {context} debe ser un objeto con 'valor' y 'descripcion'.")
            return LikertScaleInfo(valores, None)
        _require_keys(nivel, ("valor", "descripcion"), context=f"el nivel {index} de {context}", result=result)
        try:
            valor = float(nivel.get("valor"))
        except (TypeError, ValueError):
            result.errors.append(
                f"El nivel {index} en {context} tiene un 'valor' que no es numérico: {nivel.get('valor')!r}."
            )
            continue
        descripcion = nivel.get("descripcion")
        if not isinstance(descripcion, str) or not descripcion.strip():
            result.errors.append(
                f"La 'descripcion' del nivel {index} en {context} debe ser un texto no vacío."
            )
        valores.append(valor)

    if valores:
        rango = (min(valores), max(valores))
    if len(valores) != len(set(valores)):
        result.warnings.append(f"Existen valores duplicados en los niveles definidos en {context}.")
    return LikertScaleInfo(valores, rango)


def _validate_ponderacion(
    value: Any,
    *,
    context: str,
    result: ValidationResult,
) -> Optional[float]:
    try:
        ponderacion = float(value)
    except (TypeError, ValueError):
        result.errors.append(f"La ponderación en {context} debe ser un número entre 0 y 1.")
        return None

    if not 0 <= ponderacion <= 1:
        result.errors.append(
            f"La ponderación en {context} debe estar entre 0 y 1. Se recibió {ponderacion:.6f}."
        )
    return ponderacion


def _validate_level_one(data: Dict[str, Any], result: ValidationResult) -> None:
    """Validación mínima indispensable: estructura, tipos y campos obligatorios."""

    _require_keys(
        data,
        ("tipo_informe", "version", "secciones", "metrica_global", "dimensiones_defecto"),
        context="el esquema",
        result=result,
    )

    tipo_informe = data.get("tipo_informe")
    if not isinstance(tipo_informe, str) or not tipo_informe.strip():
        result.errors.append("El campo 'tipo_informe' debe ser un texto no vacío.")

    version = data.get("version")
    if not isinstance(version, str) or not version.strip():
        result.errors.append("El campo 'version' debe ser un texto no vacío.")

    dimensiones_defecto = data.get("dimensiones_defecto")
    if not isinstance(dimensiones_defecto, dict) or not dimensiones_defecto:
        result.errors.append("'dimensiones_defecto' debe ser un objeto con ponderaciones por dimensión.")

    escala_likert_defecto = data.get("escala_likert_defecto")
    if escala_likert_defecto is not None:
        if not isinstance(escala_likert_defecto, list) or not escala_likert_defecto:
            result.errors.append("'escala_likert_defecto' debe ser una lista de niveles válidos.")
        else:
            _validate_likert_levels(
                escala_likert_defecto,
                context="'escala_likert_defecto'",
                result=result,
            )

    secciones = data.get("secciones")
    if not isinstance(secciones, list) or not secciones:
        result.errors.append("'secciones' debe ser una lista con al menos una sección.")
        return

    seccion_ids: set[str] = set()
    pregunta_ids: set[str] = set()

    for index, seccion in enumerate(secciones, start=1):
        if not isinstance(seccion, dict):
            result.errors.append(f"La sección #{index} debe ser un objeto JSON.")
            continue

        seccion_context = f"la sección #{index}"
        _require_keys(
            seccion,
            ("id_segmenter", "titulo", "ponderacion", "dimensiones"),
            context=seccion_context,
            result=result,
        )

        seccion_id = seccion.get("id_segmenter")
        if not isinstance(seccion_id, str) or not seccion_id.strip():
            result.errors.append(f"{seccion_context} debe definir un 'id_segmenter' textual.")
        elif seccion_id in seccion_ids:
            result.errors.append(f"El 'id_segmenter' '{seccion_id}' está duplicado entre secciones.")
        else:
            seccion_ids.add(seccion_id)

        titulo = seccion.get("titulo")
        if not isinstance(titulo, str) or not titulo.strip():
            result.errors.append(f"{seccion_context} debe tener un 'titulo' no vacío.")

        _validate_ponderacion(seccion.get("ponderacion"), context=seccion_context, result=result)

        dimensiones = seccion.get("dimensiones")
        if not isinstance(dimensiones, list) or not dimensiones:
            result.errors.append(f"{seccion_context} debe contener una lista de dimensiones.")
            continue

        for d_index, dimension in enumerate(dimensiones, start=1):
            if not isinstance(dimension, dict):
                result.errors.append(
                    f"La dimensión #{d_index} en {seccion_context} debe ser un objeto JSON."
                )
                continue

            dim_context = (
                f"la dimensión '{dimension.get('nombre') or dimension.get('id') or f'# {d_index}'}"  # type: ignore[arg-type]
                f" de {seccion_context}"
            )
            _require_keys(
                dimension,
                ("nombre", "tipo_escala", "ponderacion", "preguntas"),
                context=dim_context,
                result=result,
            )

            nombre = dimension.get("nombre")
            if not isinstance(nombre, str) or not nombre.strip():
                result.errors.append(f"{dim_context} debe contar con un 'nombre' no vacío.")

            _validate_ponderacion(dimension.get("ponderacion"), context=dim_context, result=result)

            metodo_agregacion = dimension.get("metodo_agregacion")
            if metodo_agregacion is None:
                result.errors.append(f"{dim_context} debe especificar un 'metodo_agregacion'.")
            elif not isinstance(metodo_agregacion, str) or not metodo_agregacion.strip():
                result.errors.append(
                    f"El 'metodo_agregacion' declarado en {dim_context} debe ser un texto no vacío."
                )

            preguntas = dimension.get("preguntas")
            if not isinstance(preguntas, list) or not preguntas:
                result.errors.append(f"{dim_context} debe incluir preguntas evaluables.")
            else:
                for p_index, pregunta in enumerate(preguntas, start=1):
                    if not isinstance(pregunta, dict):
                        result.errors.append(
                            f"La pregunta #{p_index} en {dim_context} debe ser un objeto JSON."
                        )
                        continue

                    pregunta_context = (
                        f"la pregunta '{pregunta.get('id') or f'# {p_index}'}' de {dim_context}"
                    )
                    _require_keys(
                        pregunta,
                        ("id", "texto", "ponderacion"),
                        context=pregunta_context,
                        result=result,
                    )

                    pregunta_id = pregunta.get("id")
                    if not isinstance(pregunta_id, str) or not pregunta_id.strip():
                        result.errors.append(
                            f"La pregunta #{p_index} en {dim_context} debe poseer un 'id' textual no vacío."
                        )
                    elif pregunta_id in pregunta_ids:
                        result.errors.append(
                            f"El identificador de pregunta '{pregunta_id}' está duplicado en el esquema."
                        )
                    else:
                        pregunta_ids.add(pregunta_id)

                    texto = pregunta.get("texto")
                    if not isinstance(texto, str) or not texto.strip():
                        result.errors.append(
                            f"La pregunta '{pregunta_id}' en {dim_context} debe definir un 'texto' evaluable."
                        )

                    _validate_ponderacion(
                        pregunta.get("ponderacion"), context=f"la pregunta '{pregunta_id}'", result=result
                    )

            tipo_escala = dimension.get("tipo_escala")
            niveles_propios = dimension.get("niveles")
            if niveles_propios is None:
                if tipo_escala == "likert" and not escala_likert_defecto:
                    result.errors.append(
                        f"{dim_context} utiliza una escala Likert sin definir niveles propios ni una escala por defecto."
                    )
                continue

            if not isinstance(niveles_propios, list) or not niveles_propios:
                result.errors.append(f"{dim_context} define 'niveles' inválidos o vacíos.")
                continue

            escala_dimension = _validate_likert_levels(
                niveles_propios,
                context=f"{dim_context} (niveles)",
                result=result,
            )

            if tipo_escala == "binario" and len(escala_dimension.values) != 2:
                result.errors.append(f"{dim_context} con tipo_escala binario debe tener exactamente 2 niveles.")


def _validate_level_two(data: Dict[str, Any], result: ValidationResult) -> None:
    """Validación semántica: coherencia de ponderaciones y escalas."""

    dimensiones_defecto = data.get("dimensiones_defecto") or {}
    if dimensiones_defecto:
        total_dimensiones = sum(float(value) for value in dimensiones_defecto.values())
        if not almost_equal(total_dimensiones, 1.0):
            result.errors.append(
                "Las ponderaciones en 'dimensiones_defecto' suman "
                f"{total_dimensiones:.6f} (deben sumar 1.0)."
            )
    
    secciones = _collect_dict_items(
        data.get("secciones"),
        context="'secciones'",
        item_label="secciones",
        empty_message="'secciones' debe ser una lista con al menos una sección.",
        result=result,
    )
    if not secciones:
        return
    
    total_secciones = sum_ponderaciones(secciones, "ponderacion")
    if not almost_equal(total_secciones, 1.0):
        result.errors.append(
            f"Las ponderaciones de las secciones suman {total_secciones:.6f} (deben sumar 1.0)."
        )

    escala_defecto = data.get("escala_likert_defecto")
    likert_rango_esperado: Optional[Tuple[float, float]] = None
    likert_longitud_esperada: Optional[int] = None
    if isinstance(escala_defecto, list) and escala_defecto:
        escala_defecto_info = _validate_likert_levels(
            escala_defecto,
            context="'escala_likert_defecto'",
            result=result,
        )
        if escala_defecto_info.values:
            likert_rango_esperado = (
                min(escala_defecto_info.values),
                max(escala_defecto_info.values),
            )
            likert_longitud_esperada = len(escala_defecto_info.values)

    for seccion in secciones:
        nombre_seccion = seccion.get("titulo") or seccion.get("id_segmenter") or "<sin_nombre>"
        dimensiones = _collect_dict_items(
            seccion.get("dimensiones"),
            context=f"las 'dimensiones' de la sección '{nombre_seccion}'",
            item_label="dimensiones",
            empty_message=f"La sección '{nombre_seccion}' no tiene dimensiones definidas.",
            result=result,
        )
        if not dimensiones:
            continue

        total_dim = sum_ponderaciones(dimensiones, "ponderacion")
        if not almost_equal(total_dim, 1.0):
            result.errors.append(
                "Las ponderaciones de dimensiones en la sección "
                f"'{nombre_seccion}' suman {total_dim:.6f} (deben sumar 1.0)."
            )

        for dimension in dimensiones:
            dim_name = dimension.get("nombre", "<sin_nombre>")
            preguntas = _collect_dict_items(
                dimension.get("preguntas"),
                context=f"las 'preguntas' de la dimensión '{dim_name}' en la sección '{nombre_seccion}'",
                item_label="preguntas",
                empty_message=(
                    f"La dimensión '{dim_name}' en la sección '{nombre_seccion}' no tiene preguntas configuradas."
                ),
                result=result,
            )
            if preguntas:
                total_preguntas = sum_ponderaciones(preguntas, "ponderacion")
                if not almost_equal(total_preguntas, 1.0):
                    result.errors.append(
                        "Las ponderaciones de preguntas en la dimensión "
                        f"'{dim_name}' de la sección '{nombre_seccion}' suman {total_preguntas:.6f} (deben sumar 1.0)."
                    )

            niveles = dimension.get("niveles")
            tipo_escala = dimension.get("tipo_escala")
            if isinstance(niveles, list) and niveles:
                escala_dimension = _validate_likert_levels(
                    niveles,
                    context=f"la dimensión '{dim_name}' de la sección '{nombre_seccion}'",
                    result=result,
                )
                if tipo_escala == "likert" and escala_dimension.values:
                    if (
                        likert_longitud_esperada is not None
                        and len(escala_dimension.values) != likert_longitud_esperada
                    ):
                        result.errors.append(
                            f"La dimensión '{dim_name}' en la sección '{nombre_seccion}' tiene un número de niveles Likert "
                            f"diferente al esperado ({len(escala_dimension.values)} vs {likert_longitud_esperada})."
                        )
                    if likert_rango_esperado and escala_dimension.value_range:
                        if not almost_equal(
                            escala_dimension.value_range[0], likert_rango_esperado[0]
                        ) or not almost_equal(
                            escala_dimension.value_range[1], likert_rango_esperado[1]
                        ):
                            result.errors.append(
                                f"La dimensión '{dim_name}' en la sección '{nombre_seccion}' define valores Likert fuera del "
                                f"rango {likert_rango_esperado[0]}-{likert_rango_esperado[1]}."
                            )
            elif tipo_escala == "likert" and not data.get("escala_likert_defecto"):
                result.errors.append(
                    f"La dimensión '{dim_name}' en la sección '{nombre_seccion}' utiliza escala Likert sin niveles definidos. "
                    "Configure 'niveles' o una 'escala_likert_defecto'."
                )


def _validate_level_three(data: Dict[str, Any], result: ValidationResult) -> None:
    """Validación contextual: redacción y completitud de los reactivos."""
    secciones_value = data.get("secciones")
    if not isinstance(secciones_value, list):
        return

    for seccion in secciones_value:
        if not isinstance(seccion, dict):
            continue

        nombre_seccion = seccion.get("titulo") or seccion.get("id_segmenter") or "<sin_nombre>"
        dimensiones_value = seccion.get("dimensiones")
        if not isinstance(dimensiones_value, list):
            continue

        for dimension in dimensiones_value:
            if not isinstance(dimension, dict):
                continue


            dim_name = dimension.get("nombre", "<sin_nombre>")
            preguntas_value = dimension.get("preguntas")
            if not isinstance(preguntas_value, list) or not preguntas_value:
                result.errors.append(
                    f"La dimensión '{dim_name}' en la sección '{nombre_seccion}' no tiene preguntas configuradas."
                )
                continue
            for pregunta in preguntas_value:
                if not isinstance(pregunta, dict):
                    result.errors.append(
                        f"La dimensión '{dim_name}' en la sección '{nombre_seccion}' contiene preguntas con formato inválido."
                    )
                    continue
                pregunta_id = pregunta.get("id", "<sin_id>")
                texto = pregunta.get("texto")
                if isinstance(texto, str):
                    contenido = texto.strip()
                    palabra_count = len(contenido.split())
                    if palabra_count < 3 or len(contenido) < 15:
                        result.warnings.append(
                            "La pregunta "
                            f"'{pregunta_id}' en la dimensión '{dim_name}' parece demasiado breve para evaluarse con claridad."
                        )
                else:
                    result.errors.append(
                        f"La pregunta '{pregunta_id}' en la dimensión '{dim_name}' debe tener un texto descriptivo."
                    )


def validate_institutional_schema(data: Dict[str, Any]) -> ValidationResult:
    """Validate the institutional methodology schema structure."""

    result = ValidationResult()

    _validate_level_one(data, result)
    if result.errors:
        return result

    _validate_level_two(data, result)
    if result.errors:
        return result

    _validate_level_three(data, result)
    return result


register_validator("institucional", validate_institutional_schema)
        