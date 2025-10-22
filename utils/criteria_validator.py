"""Utilidades para validar esquemas JSON metodológicos utilizados por el evaluador."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

try:  # pragma: no cover - allows script execution via ``python utils/criteria_validator.py``
    from utils.validators import VALIDATORS, ValidationResult
except ModuleNotFoundError:  # pragma: no cover - fallback when package not on sys.path
    import sys

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))
    from utils.validators import VALIDATORS, ValidationResult


def validate_file(path: Path) -> ValidationResult:
    """Ejecuta el validador registrado para el tipo de informe JSON."""

    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:  # pragma: no cover - defensive logging
        result = ValidationResult()
        result.errors.append(f"No se pudo cargar '{path}': {exc}")
        return result

    tipo = data.get("tipo_informe")
    if tipo in VALIDATORS:
        return VALIDATORS[tipo](data)

    result = ValidationResult()
    result.warnings.append(
        f"No existe un validador registrado para el tipo de informe '{tipo}'. Se omitió la validación específica."
    )
    return result


def _stdout_supports(symbol: str) -> bool:
    """Return True when the active stdout encoding can render ``symbol``."""

    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        symbol.encode(encoding)
    except UnicodeEncodeError:
        return False
    return True


def _symbol(symbol: str, fallback: str) -> str:
    """Return ``symbol`` if stdout accepts it, otherwise ``fallback``."""

    return symbol if _stdout_supports(symbol) else fallback


def _format_output(path: Path, result: ValidationResult) -> List[str]:
    lines = [f"Archivo: {path}"]
    check = _symbol("✔", "[OK]")
    error = _symbol("✖", "[ERROR]")
    warning = _symbol("⚠", "[WARN]")

    if result.ok():
        lines.append(f"  {check} Validación sin errores.")
    else:
        lines.append(f"  {error} Se encontraron errores en la validación:")
        for error in result.errors:
            lines.append(f"    - {error}")

    if result.warnings:
        lines.append(f"  {warning} Advertencias:")
        for warning in result.warnings:
            lines.append(f"    - {warning}")
    return lines


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Valida archivos de criterios para el evaluador.")
    parser.add_argument("files", nargs="+", type=Path, help="Rutas de archivos JSON a validar")
    args = parser.parse_args(argv)

    exit_code = 0
    for file_path in args.files:
        result = validate_file(file_path)
        for line in _format_output(file_path, result):
            print(line)
        if not result.ok():
            exit_code = 1
    return exit_code


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())