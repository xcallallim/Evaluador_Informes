"""CLI principal para ejecutar el Evaluador CEPLAN.

Flujo operativo esperado:
1. Usuario sube el documento (input)
2. Selecciona tipo, modo y proveedor
3. CLI ejecuta evaluación (vía EvaluationService)
4. Se genera el reporte (Excel por defecto)
5. Se guarda en data/outputs/ y se registra log
6. El sistema mantiene metadatos de trazabilidad
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

try:
    import typer
except ModuleNotFoundError as exc:  # pragma: no cover - dependencia externa opcional en entornos de prueba
    raise RuntimeError(
        "La interfaz CLI requiere la librería 'typer'. Instálala con 'pip install typer'."
    ) from exc

from services.evaluation_service import EvaluationFilters, EvaluationService, SERVICE_VERSION
from utils.secret_manager import SecretManagerError

CLI_VERSION = "0.1.0"
APP_NAME = "Evaluador CEPLAN"

INPUT_DIR = Path("data/inputs")
OUTPUT_DIR = Path("data/outputs")
CRITERIA_DIR = Path("data/criteria")
DEFAULT_CRITERIA_FILE = CRITERIA_DIR / "criterios_base.json"
LOG_DIR = Path("data/logs")
LOG_FILE = LOG_DIR / "ceplan_cli.log"

REPORT_TYPES = {
    "institucional": "institucional",
    "informe institucional": "institucional",
    "politica": "politica_nacional",
    "política": "politica_nacional",
    "política nacional": "politica_nacional",
    "politica nacional": "politica_nacional",
}

EVALUATION_MODES = {
    "global": "global",
    "parcial": "parcial",
    "incremental": "reevaluacion",
    "reevaluacion": "reevaluacion",
    "reevaluación": "reevaluacion",
}

OUTPUT_FORMATS = {
    "excel": "xlsx",
    "xlsx": "xlsx",
    "csv": "csv",
    "json": "json",
}

AI_PROVIDERS = {
    "mock": "mock",
    "simulado": "mock",
    "prueba": "mock",
    "real": "openai",
    "openai": "openai",
}

app = typer.Typer(add_completion=False, help="Interfaz asistida para ejecutar evaluaciones de informes CEPLAN.")

_run_logger: Optional[logging.Logger] = None


def _get_run_logger() -> logging.Logger:
    global _run_logger
    if _run_logger:
        return _run_logger
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ceplan_cli")
    logger.setLevel(logging.INFO)
    if not any(isinstance(handler, logging.FileHandler) and handler.baseFilename == str(LOG_FILE) for handler in logger.handlers):
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s\t%(levelname)s\t%(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    _run_logger = logger
    return logger


def _version_callback(version: bool) -> None:
    if version:
        typer.echo(f"{APP_NAME} CLI v{CLI_VERSION}")
        typer.echo(f"EvaluationService v{SERVICE_VERSION}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        help="Muestra la versión de la CLI y termina.",
        callback=_version_callback,
        is_eager=True,
        is_flag=True,
    ),
) -> None:
    """Punto de entrada principal de la CLI."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


def _normalise_choice(value: Optional[str], mapping: dict[str, str], *, label: str) -> str:
    if not value:
        raise typer.BadParameter(f"Debes especificar un valor para {label}.")
    key = value.strip().lower()
    resolved = mapping.get(key)
    if resolved:
        return resolved
    raise typer.BadParameter(f"{label} '{value}' no es reconocido. Opciones válidas: {', '.join(sorted(set(mapping.keys())))}.")


def _list_directory(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return [item for item in sorted(path.iterdir()) if item.is_file()]


def _prompt_for_choice(title: str, options: Iterable[str], default: Optional[str] = None) -> str:
    typer.echo(title)
    indexed = list(enumerate(options, start=1))
    for number, option in indexed:
        typer.echo(f"  [{number}] {option}")
    default_text = "" if default is None else str(default)
    while True:
        answer = typer.prompt("Selecciona una opción", default=default_text).strip()
        if not answer and default_text:
            answer = default_text
        if answer.isdigit():
            idx = int(answer)
            if 1 <= idx <= len(indexed):
                return indexed[idx - 1][1]
        for _, option in indexed:
            if answer.lower() == option.lower():
                return option
        typer.secho("Opción inválida, intenta nuevamente.", fg=typer.colors.YELLOW)


def _resolve_input_path(input_file: Optional[Path], guided: bool) -> Path:
    available_files = _list_directory(INPUT_DIR)
    if input_file:
        resolved = input_file if input_file.is_absolute() else Path(input_file)
        if not resolved.exists():
            candidate = INPUT_DIR / input_file
            if candidate.exists():
                resolved = candidate
        if not resolved.exists():
            raise typer.BadParameter(f"El archivo de entrada '{input_file}' no existe.")
        return resolved

    if not guided:
        raise typer.BadParameter("Debes indicar --input cuando se ejecuta en modo no guiado.")

    if not available_files:
        raise typer.BadParameter("No se encontraron archivos en data/inputs. Copia un informe antes de continuar.")

    options = [file.name for file in available_files]
    choice = _prompt_for_choice("Selecciona el informe que deseas evaluar:", options)
    return INPUT_DIR / choice


def _resolve_criteria(criteria_path: Optional[Path], guided: bool) -> Path:
    if criteria_path:
        resolved = criteria_path if criteria_path.is_absolute() else Path(criteria_path)
        if not resolved.exists():
            candidate = CRITERIA_DIR / criteria_path
            if candidate.exists():
                resolved = candidate
        if not resolved.exists():
            raise typer.BadParameter(f"El archivo de criterios '{criteria_path}' no existe.")
        return resolved

    if DEFAULT_CRITERIA_FILE.exists():
        return DEFAULT_CRITERIA_FILE

    if guided:
        raise typer.BadParameter("No se encontró un archivo de criterios por defecto. Usa --criteria para especificarlo.")
    raise typer.BadParameter("Debes proporcionar --criteria cuando no existe el archivo por defecto.")


def _resolve_tipo_informe(tipo: Optional[str], guided: bool) -> str:
    if tipo:
        return _normalise_choice(tipo, REPORT_TYPES, label="tipo de informe")

    if not guided:
        raise typer.BadParameter("Debes proporcionar --tipo-informe en modo no guiado.")

    selected = _prompt_for_choice(
        "Selecciona el tipo de informe:",
        ["Institucional", "Política Nacional"],
        default="1",
    )
    return REPORT_TYPES[selected.lower()]


def _resolve_mode(mode: Optional[str], guided: bool) -> str:
    if mode:
        return _normalise_choice(mode, EVALUATION_MODES, label="modo de evaluación")

    if not guided:
        return "global"

    selected = _prompt_for_choice(
        "Selecciona el modo de evaluación:",
        ["Global", "Parcial", "Incremental"],
        default="1",
    )
    return EVALUATION_MODES[selected.lower()]


def _resolve_output_format(fmt: Optional[str], guided: bool) -> tuple[str, str]:
    if fmt:
        extension = _normalise_choice(fmt, OUTPUT_FORMATS, label="formato")
        return fmt.strip().lower(), extension

    if not guided:
        return ("excel", "xlsx")

    selected = _prompt_for_choice(
        "Selecciona el formato de salida:",
        ["Excel", "CSV", "JSON"],
        default="1",
    )
    key = selected.lower()
    return key, OUTPUT_FORMATS[key]


def _resolve_ai_provider(provider: Optional[str], guided: bool) -> str:
    if provider:
        return _normalise_choice(provider, AI_PROVIDERS, label="proveedor de IA")

    if not guided:
        return "mock"

    selected = _prompt_for_choice(
        "Selecciona el proveedor de IA:",
        ["Mock (sin llamadas reales)", "Real (requiere API key)"]
    )
    if selected.lower().startswith("real"):
        return "openai"
    return "mock"


def _ensure_api_key(provider: str, api_key: Optional[str], guided: bool) -> None:
    if provider != "openai":
        return

    existing = os.getenv("OPENAI_API_KEY")
    key = api_key or existing
    if key:
        if not existing:
            os.environ["OPENAI_API_KEY"] = key
        return

    warning = "No se detectó una clave API para el proveedor seleccionado."
    typer.secho(warning, fg=typer.colors.YELLOW)
    if guided:
        provided = typer.prompt("Ingresa tu OPENAI_API_KEY o deja vacío para cancelar", default="").strip()
        if not provided:
            raise typer.Exit(code=1)
        os.environ["OPENAI_API_KEY"] = provided
        return

    raise typer.BadParameter("Debes proporcionar --api-key o definir OPENAI_API_KEY en el entorno.")


def _build_output_path(input_path: Path, extension: str, output_path: Optional[Path]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if output_path:
        resolved = output_path if output_path.is_absolute() else OUTPUT_DIR / output_path
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"evaluacion_{input_path.stem}_{timestamp}.{extension}"
    return OUTPUT_DIR / filename


def _log_run(**payload: object) -> None:
    logger = _get_run_logger()
    try:
        serialised = json.dumps(payload, ensure_ascii=False, default=str)
    except TypeError:
        serialised = str(payload)
    logger.info("run %s", serialised)


@app.command("evaluar")
def evaluate(
    input_file: Optional[Path] = typer.Option(None, "--input", "-i", help="Archivo del informe dentro de data/inputs."),
    criteria_path: Optional[Path] = typer.Option(None, "--criteria", "-c", help="Archivo JSON con los criterios de evaluación."),
    tipo_informe: Optional[str] = typer.Option(None, "--tipo-informe", "-t", help="Tipo de informe: institucional o política nacional."),
    modo: Optional[str] = typer.Option(None, "--modo", "-m", help="Modo de evaluación: global, parcial o incremental."),
    formato: Optional[str] = typer.Option(None, "--formato", "-f", help="Formato de salida: Excel (predeterminado), CSV o JSON."),
    proveedor: Optional[str] = typer.Option(None, "--ai-provider", help="Proveedor de IA a utilizar: mock u openai."),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Clave API para el proveedor real. Usa OPENAI_API_KEY por defecto."),
    output_path: Optional[Path] = typer.Option(None, "--output", "-o", help="Ruta personalizada para guardar los resultados."),
    guided: bool = typer.Option(True, "--guided/--no-guided", help="Activa el modo asistido con preguntas interactivas."),
    extra_instructions: Optional[str] = typer.Option(None, "--extra", help="Instrucciones adicionales para el modelo."),
) -> None:
    """Ejecuta el proceso completo de evaluación de informes."""

    try:
        resolved_input = _resolve_input_path(input_file, guided)
        resolved_criteria = _resolve_criteria(criteria_path, guided)
        resolved_tipo = _resolve_tipo_informe(tipo_informe, guided)
        resolved_mode = _resolve_mode(modo, guided)
        format_label, extension = _resolve_output_format(formato, guided)
        resolved_provider = _resolve_ai_provider(proveedor, guided)
        _ensure_api_key(resolved_provider, api_key, guided)
        final_output = _build_output_path(resolved_input, extension, output_path)

        typer.secho("Iniciando evaluación...", fg=typer.colors.CYAN)
        service = EvaluationService()
        filters = EvaluationFilters()
        config_overrides = {
            "ai_provider": resolved_provider,
            "extra_instructions": extra_instructions,
            "log_file": LOG_FILE,
        }

        _log_run(
            input=str(resolved_input),
            criteria=str(resolved_criteria),
            tipo=resolved_tipo,
            mode=resolved_mode,
            format=format_label,
            provider=resolved_provider,
            guided=guided,
            output=str(final_output),
        )

        result, context = service.run(
            input_path=resolved_input,
            criteria_path=resolved_criteria,
            tipo_informe=resolved_tipo,
            mode=resolved_mode,
            filters=filters,
            output_path=final_output,
            output_format=extension,
            config_overrides=config_overrides,
        )
    except SecretManagerError as exc:
        typer.secho(f"No se pudo obtener la clave API requerida: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    except typer.BadParameter as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=2)
    except Exception as exc:  # pragma: no cover - errores inesperados
        typer.secho(f"Ocurrió un error inesperado: {exc}", fg=typer.colors.RED)
        _get_run_logger().exception("execution failed")
        raise typer.Exit(code=1)

    questions_total = context.get("metrics", {}).get("questions_total") if isinstance(context, dict) else None
    typer.secho("Evaluación completada.", fg=typer.colors.GREEN)
    typer.echo(f"Resultados guardados en: {final_output}")
    if questions_total:
        typer.echo(f"Preguntas totales evaluadas: {questions_total}")
    metadata = getattr(result, "metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    document_metadata = metadata.get("document")
    document_title: Optional[str] = None
    if isinstance(document_metadata, dict):
        document_title = (
            document_metadata.get("title")
            or document_metadata.get("nombre")
            or document_metadata.get("titulo")
            or document_metadata.get("name")
        )

    if not document_title:
        document_title = (
            metadata.get("document_title")
            or metadata.get("document_name")
            or metadata.get("title")
            or metadata.get("nombre")
            or metadata.get("titulo")
        )

    if not document_title:
        source_path = metadata.get("source_path")
        if isinstance(source_path, str) and source_path:
            document_title = Path(source_path).name

    typer.echo("Resumen del resultado:")
    typer.echo(f"  Documento: {document_title or resolved_input.name}")
    typer.echo(f"  Proveedor IA: {resolved_provider}")
    typer.echo(f"  Formato exportado: {format_label.upper()}")

    evaluated_sections = None
    total_sections = None
    if hasattr(result, "sections") and isinstance(result.sections, list):
        total_sections = len(result.sections)
        evaluated_sections = sum(
            1 for section in result.sections if getattr(section, "score", None) is not None
        )

    average_quality = None
    prompt_validation = metadata.get("prompt_validation")
    if isinstance(prompt_validation, dict):
        raw_average = prompt_validation.get("average_quality")
        try:
            if raw_average is not None:
                average_quality = float(raw_average)
        except (TypeError, ValueError):
            average_quality = None

    summary_fragments: list[str] = []
    if evaluated_sections is not None and total_sections is not None:
        summary_fragments.append(f"Secciones evaluadas: {evaluated_sections}/{total_sections}")
    if average_quality is not None:
        summary_fragments.append(f"Promedio de calidad: {average_quality:.2f}")
    if summary_fragments:
        typer.echo(", ".join(summary_fragments))


if __name__ == "__main__":  # pragma: no cover - ejecución directa
    app()