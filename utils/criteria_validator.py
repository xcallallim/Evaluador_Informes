from __future__ import annotations

"""Utilities to validate methodological JSON schemas used by the evaluator."""

import argparse
import json
import sys
import unicodedata
from pathlib import Path, PureWindowsPath
from typing import List

try:  # pragma: no cover - allows script execution via ``python utils/criteria_validator.py``
    from utils.validators import VALIDATORS, ValidationResult
except ModuleNotFoundError:  # pragma: no cover - fallback when package not on sys.path
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))
    from utils.validators import VALIDATORS, ValidationResult


def _normalize_filename(name: str) -> str:
    """Return ``name`` lowercased and without diacritical marks."""

    normalized = unicodedata.normalize("NFKD", name)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()


def _resolve_with_normalized(candidate: Path) -> Path | None:
    """Return an existing path whose name matches ``candidate`` ignoring accents."""

    parent = candidate.parent
    try:
        entries = list(parent.iterdir())
    except FileNotFoundError:
        return None
    normalized_target = _normalize_filename(candidate.name)
    for entry in entries:
        if _normalize_filename(entry.name) == normalized_target:
            return entry
    return None


def _resolve_path(path: Path) -> Path:
    """Resolve ``path`` against common repository locations."""

    repo_root = Path(__file__).resolve().parent.parent
    repo_name = repo_root.name

    if path.exists():
        return path

    candidates: list[Path] = []

    if not path.is_absolute():
        candidates.append(Path.cwd() / path)
    else:
        try:
            parts = list(path.parts)
        except TypeError:
            parts = []
        if parts and repo_name in parts:
            suffix = Path(*parts[parts.index(repo_name) + 1 :])
            candidates.append(repo_root / suffix)
        else:
            windows_parts = list(PureWindowsPath(str(path)).parts)
            if repo_name in windows_parts:
                suffix = Path(*windows_parts[windows_parts.index(repo_name) + 1 :])
                candidates.append(repo_root / suffix)

    candidates.append(repo_root / path)

    raw = str(path)
    if repo_name in raw:
        suffix = raw.split(repo_name, 1)[1].lstrip("/\\")
        suffix = suffix.replace("\\", "/")
        if suffix:
            candidates.append(repo_root / Path(suffix))

    for candidate in candidates:
        if candidate.exists():
            return candidate
        normalized_candidate = _resolve_with_normalized(candidate)
        if normalized_candidate and normalized_candidate.exists():
            return normalized_candidate
        try:
            resolved = candidate.resolve(strict=False)
        except RuntimeError:
            resolved = candidate
        if resolved.exists():
            return resolved
        normalized_resolved = _resolve_with_normalized(resolved)
        if normalized_resolved and normalized_resolved.exists():
            return normalized_resolved

    return path


def validate_file(path: Path) -> ValidationResult:
    """Run the validator registered for the JSON report type."""

    resolved_path = _resolve_path(path)

    try:
        with resolved_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:  # pragma: no cover - defensive logging
        result = ValidationResult()
        result.errors.append(f"No se pudo cargar '{resolved_path}': {exc}")
        return result
    
    tipo = data.get("tipo_informe")
    validator = VALIDATORS.get(tipo)
    if validator is not None:
        validator_result = validator(data)
        if isinstance(validator_result, ValidationResult):
            return validator_result

        fallback = ValidationResult()
        fallback.errors.append(
            f"El validador registrado para '{tipo}' devolvió un resultado inesperado "
            f"({type(validator_result).__name__})."
        )
        return fallback

    result = ValidationResult()
    result.warnings.append(
        f"No existe un validador registrado para el tipo de informe '{tipo}'. Se omitió la validación específica."
    )
    return result


def _stdout_supports(symbol: str) -> bool:
    """Return True when the active stdout encoding can render ``symbol``."""

    encoding = getattr(sys.stdout, "encoding", None)
    if not encoding:
        return False
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
    parser.add_argument("files", nargs="*", type=Path, help="Rutas de archivos JSON a validar")
    args = parser.parse_args(argv)

    if not args.files:
        parser.print_usage(sys.stdout)
        print("No se proporcionaron archivos para validar.")
        return 0

    exit_code = 0
    for file_path in args.files:
        resolved_path = _resolve_path(file_path)
        result = validate_file(resolved_path)
        for line in _format_output(resolved_path, result):
            try:
                print(line)
            except UnicodeEncodeError:
                fallback_line = (
                    line.replace("✔", "[OK]")
                    .replace("✖", "[ERROR]")
                    .replace("⚠", "[WARN]")
                )
                print(fallback_line)
        if not result.ok():
            exit_code = 1
    return exit_code

if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())