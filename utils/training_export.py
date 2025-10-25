"""Utilities to project tabular exports into the training-ready schema.

The production pipeline serialises evaluations with :class:`EvaluationRepository`
producing a workbook that contains the detailed question breakdown in the
"preguntas" sheet and a compact metadata summary in "indice_global".

This module helps notebooks and smoke tests confirm that those exports expose
the exact columns expected by the fine-tuning dataset. It provides helpers to
load the latest workbook, reshape it into the canonical schema and validate the
basic quality constraints (mandatory columns, score ranges and non-empty text
fields).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd

from utils.export_locator import ExportLookup, ExportNotFoundError, find_latest_export


class TrainingExportError(RuntimeError):
    """Exception raised when the export cannot be projected to the schema."""


REQUIRED_COLUMNS = (
    "tipo_informe",
    "seccion",
    "criterio",
    "pregunta",
    "texto",
    "score",
    "justificacion",
    "model_name",
    "pipeline_version",
    "criteria_version",
    "timestamp",
)

_SCORE_RANGES: Mapping[str, tuple[float, float]] = {
    "institucional": (0.0, 4.0),
    "politica_nacional": (0.0, 4.0),
}


@dataclass(frozen=True)
class TrainingDataset:
    """Container returned by :func:`load_training_dataset`."""

    path: Path
    dataframe: pd.DataFrame

    def validate(self) -> None:
        """Validate the dataframe against the canonical schema."""

        missing = sorted(set(REQUIRED_COLUMNS) - set(self.dataframe.columns))
        if missing:
            raise TrainingExportError(
                "Faltan columnas en el dataset exportado: " + ", ".join(missing)
            )

        if self.dataframe.empty:
            raise TrainingExportError("El dataset exportado está vacío.")

        _validate_basic_fields(self.dataframe)
        _validate_scores(self.dataframe)


def load_training_dataset(export_path: Path | str) -> TrainingDataset:
    """Load ``export_path`` and project it to the training schema."""

    workbook = Path(export_path)
    if not workbook.exists():
        raise FileNotFoundError(f"El archivo '{workbook}' no existe.")

    try:
        preguntas = pd.read_excel(workbook, sheet_name="preguntas")
        header = pd.read_excel(workbook, sheet_name="indice_global")
    except ValueError as exc:  # pragma: no cover - routed as TrainingExportError
        raise TrainingExportError(str(exc)) from exc

    if header.empty:
        raise TrainingExportError("La hoja 'indice_global' está vacía.")

    metadata = header.iloc[0].to_dict()
    tipo_informe = _clean(metadata.get("tipo_informe"))
    if not tipo_informe and "tipo_informe" in preguntas:
        tipo_informe = _clean(preguntas["tipo_informe"].iloc[0])

    dataframe = pd.DataFrame(
        {
            "tipo_informe": tipo_informe,
            "seccion": preguntas.get("section_title", pd.Series(dtype="object")),
            "criterio": preguntas.get("dimension_name", pd.Series(dtype="object")),
            "pregunta": preguntas.get("question_text", pd.Series(dtype="object")),
            "texto": preguntas.get("relevant_text", pd.Series(dtype="object")),
            "score": preguntas.get("question_score", pd.Series(dtype="float")),
            "justificacion": preguntas.get("justification", pd.Series(dtype="object")),
            "model_name": metadata.get("model_name"),
            "pipeline_version": metadata.get("pipeline_version"),
            "criteria_version": preguntas.get("criteria_version", pd.Series(dtype="object")),
            "timestamp": metadata.get("timestamp"),
        }
    )

    return TrainingDataset(path=workbook, dataframe=dataframe)


def load_latest_training_dataset(pattern: str, directory: Path | str | None = None) -> TrainingDataset:
    """Locate the latest export matching ``pattern`` and load it."""

    try:
        lookup: ExportLookup = find_latest_export(pattern=pattern, directory=directory)
    except ExportNotFoundError as exc:
        raise TrainingExportError(str(exc)) from exc

    dataset = load_training_dataset(lookup.latest)
    dataset.validate()
    return dataset


def _validate_basic_fields(dataframe: pd.DataFrame) -> None:
    """Ensure mandatory text fields are populated."""

    tipo_series = dataframe["tipo_informe"].astype(str).str.lower().str.strip()
    if tipo_series.str.len().eq(0).any():
        raise TrainingExportError("Existen filas sin 'tipo_informe'.")

    if dataframe["score"].isna().any():
        raise TrainingExportError("Existen filas sin puntaje.")

    pregunta_series = dataframe["pregunta"].astype(str)
    if pregunta_series.str.len().eq(0).any():
        raise TrainingExportError("Existen filas sin texto de pregunta.")


def _validate_scores(dataframe: pd.DataFrame) -> None:
    """Validate score ranges per methodology."""

    for tipo, bounds in _SCORE_RANGES.items():
        lower, upper = bounds
        mask = dataframe["tipo_informe"].astype(str).str.lower() == tipo
        if not mask.any():
            continue
        series = dataframe.loc[mask, "score"]
        if not series.between(lower, upper).all():
            raise TrainingExportError(
                f"Los puntajes para '{tipo}' deben estar entre {lower} y {upper}."
            )


def _clean(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


__all__ = [
    "TrainingDataset",
    "TrainingExportError",
    "load_latest_training_dataset",
    "load_training_dataset",
    "REQUIRED_COLUMNS",
]