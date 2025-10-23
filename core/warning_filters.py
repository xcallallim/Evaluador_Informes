"""Helpers to keep noisy third-party deprecation warnings under control."""

from __future__ import annotations

import re
import warnings
from functools import lru_cache
from typing import Iterable, Sequence, Tuple, Type


_THIRD_PARTY_MODULES: Tuple[str, ...] = (
    "camelot",
    "fitz",  # PyMuPDF
    "rapidfuzz",
    "pyarrow",
)

_WARNING_CATEGORIES: Tuple[Type[Warning], ...] = (
    FutureWarning,
    DeprecationWarning,
    PendingDeprecationWarning,
)


def _register_filters(
    module_patterns: Iterable[str],
    categories: Sequence[Type[Warning]],
) -> None:
    """Register warning filters for each module/category combination."""

    for module_name in module_patterns:
        module_regex = rf"^{re.escape(module_name)}(\\.|$)"
        for category in categories:
            warnings.filterwarnings(
                "ignore",
                category=category,
                module=module_regex,
            )


@lru_cache(maxsize=None)
def suppress_external_deprecation_warnings() -> None:
    """Silence Future/Deprecation warnings emitted by third-party libraries.

    Some of the optional dependencies used by the loaders (PyMuPDF, Camelot,
    RapidFuzz, PyArrow, …) ship warnings indicating upcoming breaking changes.
    These warnings are useful when working on the libraries themselves, but they
    add a lot of noise when our test-suite imports them indirectly.  The helper
    centralises the filters so individual modules can opt-in without duplicating
    patterns or accidentally muting unrelated warnings.
    """

    _register_filters(_THIRD_PARTY_MODULES, _WARNING_CATEGORIES)


__all__ = ["suppress_external_deprecation_warnings"]