"""Registro de validadores de criterios agrupados por tipo de informe."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Callable, Dict

from utils.validators.base import ValidationResult

Validator = Callable[[dict], ValidationResult]

_VALIDATORS: Dict[str, Validator] = {}


def register_validator(tipo_informe: str, validator: Validator) -> None:
    """Register ``validator`` to handle the given ``tipo_informe``."""

    _VALIDATORS[tipo_informe] = validator


def _autodiscover() -> None:
    """Import validator modules so they can register themselves."""

    package_dir = Path(__file__).resolve().parent
    for module_path in package_dir.glob("*.py"):
        if module_path.stem in {"__init__", "base"}:
            continue
        import_module(f"{__name__}.{module_path.stem}")


_autodiscover()

VALIDATORS: Dict[str, Validator] = _VALIDATORS

__all__ = ["VALIDATORS", "ValidationResult", "register_validator"]