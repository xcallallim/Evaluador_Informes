from __future__ import annotations

import os
from pathlib import Path

import pytest

from utils.export_locator import (
    ExportLookup,
    ExportNotFoundError,
    find_latest_export,
)


def test_find_latest_export_returns_newest_file(tmp_path: Path) -> None:
    first = tmp_path / "resultados_demo_20240101.xlsx"
    second = tmp_path / "resultados_demo_20240102.xlsx"
    third = tmp_path / "resultados_demo_20240103.xlsx"

    for index, path in enumerate((first, second, third), start=1):
        path.write_text(f"archivo {index}")
        os.utime(path, (path.stat().st_atime, path.stat().st_mtime + index))

    lookup = find_latest_export("resultados_demo_*.xlsx", directory=tmp_path)

    assert isinstance(lookup, ExportLookup)
    assert lookup.directory == tmp_path
    assert lookup.pattern == "resultados_demo_*.xlsx"
    assert lookup.matches == (first, second, third)
    assert lookup.latest == third


def test_find_latest_export_raises_when_directory_missing(tmp_path: Path) -> None:
    missing_dir = tmp_path / "nope"

    with pytest.raises(ExportNotFoundError) as excinfo:
        find_latest_export("*.xlsx", directory=missing_dir)

    assert "no existe" in str(excinfo.value)


def test_find_latest_export_raises_when_no_matches(tmp_path: Path) -> None:
    (tmp_path / "existing.txt").write_text("contenido")

    with pytest.raises(ExportNotFoundError) as excinfo:
        find_latest_export("resultados_*.xlsx", directory=tmp_path)

    assert "No se encontraron archivos" in str(excinfo.value)