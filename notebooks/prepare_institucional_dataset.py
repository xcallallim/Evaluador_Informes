"""Generate a reusable dataset with institutional questions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.training_export import (
    TrainingExportError,
    load_latest_training_dataset,
    load_training_dataset,
)
from utils.export_locator import find_latest_export


EXPORT_PATTERN = "dataset_entrenamiento_ml_*.xlsx"
EXPORT_DIRECTORY = Path("data/examples")
MIN_COLUMNS = ("tipo_informe", "pregunta", "texto", "justificacion", "score")
DEFAULT_OUTPUT_PATH = Path("data/processed/institucional_base.csv")


def _validate_minimum_columns(dataframe: pd.DataFrame) -> None:
    missing = [column for column in MIN_COLUMNS if column not in dataframe.columns]
    if missing:
        raise TrainingExportError(
            "El dataset exportado no contiene las columnas requeridas: "
            + ", ".join(missing)
        )


def _load_latest_dataset() -> pd.DataFrame:
    try:
        dataset = load_latest_training_dataset(
            pattern=EXPORT_PATTERN,
            directory=EXPORT_DIRECTORY,
        )
        return dataset.dataframe
    except TrainingExportError as exc:
        lookup = find_latest_export(pattern=EXPORT_PATTERN, directory=EXPORT_DIRECTORY)
        dataset = load_training_dataset(lookup.latest)
        print(
            "Advertencia: la validación del dataset falló con el mensaje:\n",
            f"  {exc}",
            "\nSe continuará con el dataset sin aplicar validaciones adicionales.",
        )
        return dataset.dataframe


def _save_institucional_dataset(dataframe: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = output_path.suffix.lower()
    if suffix == ".parquet":
        dataframe.to_parquet(output_path, index=False)
    elif suffix == ".csv":
        dataframe.to_csv(output_path, index=False)
    else:
        raise TrainingExportError(
            "Formato de archivo no soportado. Utiliza extensiones .parquet o .csv."
        )

    print(
        "Dataset institucional guardado en",
        output_path,
        "con",
        len(dataframe),
        "filas.",
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genera un dataset institucional filtrado a partir del último export "
            "disponible."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=(
            "Ruta donde se guardará el dataset filtrado. La extensión determina el "
            "formato de salida (admite .parquet o .csv)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    dataframe = _load_latest_dataset()
    _validate_minimum_columns(dataframe)

    institucional_df = dataframe[
        dataframe["tipo_informe"].astype(str).str.lower() == "institucional"
    ].copy()

    if institucional_df.empty:
        raise TrainingExportError(
            "El dataset más reciente no contiene filas del tipo 'institucional'."
        )

    _save_institucional_dataset(institucional_df, args.output)


if __name__ == "__main__":
    main()

# python notebooks/prepare_institucional_dataset.py