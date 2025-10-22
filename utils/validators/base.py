"""Shared utilities for criteria validators."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List

TOLERANCE = 1e-6


class ValidationResult:
    """Stores validation errors and warnings for a schema."""

    def __init__(self) -> None:
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def extend(self, other: "ValidationResult") -> None:
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)

    def ok(self) -> bool:
        return not self.errors

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.ok()


def almost_equal(value: float, target: float, *, tolerance: float = TOLERANCE) -> bool:
    """Return True when ``value`` and ``target`` are nearly identical."""

    return math.isclose(value, target, rel_tol=tolerance, abs_tol=tolerance)


def sum_ponderaciones(items: Iterable[Dict[str, Any]], key: str) -> float:
    """Sum the numeric value stored at ``key`` in every mapping of ``items``."""

    return sum(float(item.get(key, 0)) for item in items)
