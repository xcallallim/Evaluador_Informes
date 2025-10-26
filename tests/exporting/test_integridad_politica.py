import json

import pandas as pd
import pytest

from services.evaluation_service import EvaluationService


def _assert_consistent_exports(
    questions_df: pd.DataFrame,
    csv_df: pd.DataFrame,
    json_df: pd.DataFrame,
    expected_tipo: str,
) -> None:
    dataframes = {
        "Excel": questions_df,
        "CSV": csv_df,
        "JSON": json_df,
    }

    lengths = {name: len(df) for name, df in dataframes.items()}
    assert len(set(lengths.values())) == 1, (
        f"La cantidad de filas difiere entre formatos: {lengths}"
    )
    assert lengths["Excel"] > 0, "Los datos exportados están vacíos"

    column_sets = {name: set(df.columns) for name, df in dataframes.items()}
    unique_column_sets = {frozenset(columns) for columns in column_sets.values()}
    assert len(unique_column_sets) == 1, (
        f"Las columnas no coinciden entre formatos: {column_sets}"
    )

    column_orders = {name: list(df.columns) for name, df in dataframes.items()}
    first_order = next(iter(column_orders.values()))
    for name, columns in column_orders.items():
        assert columns == first_order, (
            "El orden de las columnas difiere entre formatos: "
            f"{name} -> {columns}"
        )

    required_cols = {
        "document_id",
        "section_id",
        "section_title",
        "section_score",
        "section_weight",
        "dimension_name",
        "dimension_score",
        "dimension_weight",
        "question_id",
        "question_text",
        "question_score",
        "question_weight",
        "justification",
        "criteria_version",
        "tipo_informe",
        "model_name",
        "pipeline_version",
        "timestamp",
    }

    combined_columns = next(iter(unique_column_sets))
    missing_required = required_cols - set(combined_columns)
    assert not missing_required, f"Faltan columnas críticas: {missing_required}"

    string_columns = [
        "document_id",
        "section_id",
        "section_title",
        "dimension_name",
        "question_id",
        "question_text",
        "justification",
        "criteria_version",
        "tipo_informe",
        "model_name",
        "pipeline_version",
        "timestamp",
    ]
    numeric_score_columns = [
        "section_score",
        "dimension_score",
        "question_score",
    ]
    weight_columns = [
        "section_weight",
        "dimension_weight",
        "question_weight",
    ]

    for name, df in dataframes.items():
        justification_series = df.get("justification", pd.Series(index=df.index, dtype="string")).astype(str)
        allowed_missing_scores = justification_series.str.contains(
            "Evaluación omitida", case=False, na=False
        )

        for col in string_columns:
            assert col in df.columns, f"La columna '{col}' no está presente en {name}"
            series = df[col]
            manual_mask = (
                df.get("question_id", pd.Series(index=df.index, dtype="string"))
                .astype(str)
                .str.startswith("MANUAL_")
            )
            if col == "section_id":
                invalid_nulls = series.isna() & ~manual_mask
                assert not invalid_nulls.any(), (
                    f"Valores nulos en '{col}' ({name}) fuera de preguntas manuales"
                )
                non_null_series = series.dropna()
                assert not non_null_series.astype(str).str.strip().eq("").any(), (
                    f"Valores vacíos en '{col}' ({name})"
                )
            else:
                assert not series.isna().any(), f"Valores nulos en '{col}' ({name})"
                assert not series.astype(str).str.strip().eq("").any(), (
                    f"Valores vacíos en '{col}' ({name})"
                )

        for col in numeric_score_columns:
            assert col in df.columns, f"La columna '{col}' no está presente en {name}"
            numeric_series = pd.to_numeric(df[col], errors="coerce")
            null_mask = numeric_series.isna()
            assert (~null_mask | allowed_missing_scores).all(), (
                f"Valores nulos en '{col}' ({name}) sin justificación de omisión"
            )
            non_null_series = numeric_series[~null_mask]
            assert non_null_series.between(0, 4).all(), (
                f"Valores fuera de rango [0, 4] en '{col}' ({name})"
            )

        for col in weight_columns:
            assert col in df.columns, f"La columna '{col}' no está presente en {name}"
            numeric_series = pd.to_numeric(df[col], errors="coerce")
            assert numeric_series.notna().all(), (
                f"Valores no numéricos o nulos en '{col}' ({name})"
            )
            assert numeric_series.between(0, 1).all(), (
                f"Valores fuera de rango [0, 1] en '{col}' ({name})"
            )

        assert set(df["tipo_informe"].unique()) == {expected_tipo}, (
            f"Se encontraron tipos de informe no permitidos en {name}: "
            f"{set(df['tipo_informe'].unique())}"
        )

@pytest.mark.integration
def test_repository_generates_excel_and_csv(tmp_path):
    """Valida que la exportación política genere artefactos completos y consistentes."""

    output_dir = tmp_path / "exports"
    output_dir.mkdir()
    output_path = output_dir / "resultados_informe_politica_demo_test.xlsx"

    service = EvaluationService()

    evaluation, metrics = service.run(
        input_path="data/examples/informe_politica_demo.txt",
        tipo_informe="politica_nacional",
        mode="global",
        output_format="xlsx",
        output_path=output_path,
        criteria_path="data/criteria/metodología_politica_nacional.json",
    )

    assert evaluation.document_type == "politica_nacional"

    metrics_dict = metrics
    global_summary = metrics_dict["global"]

    assert metrics_dict["methodology"] == "politica_nacional"
    assert global_summary["normalized_min"] == pytest.approx(0.0)
    assert global_summary["normalized_max"] > 0
    assert global_summary["normalized_max"] >= global_summary["normalized_score"]
    assert 0.0 <= global_summary["normalized_score"] <= global_summary["normalized_max"]

    excel_path = output_path
    csv_path = output_path.with_suffix(".csv")
    json_path = output_path.with_suffix(".json")

    assert excel_path.exists(), f"No se generó el archivo Excel en {excel_path}"
    assert csv_path.exists(), f"No se generó el archivo CSV en {csv_path}"
    assert json_path.exists(), f"No se generó el archivo JSON en {json_path}"

    excel = pd.ExcelFile(excel_path)
    expected_sheets = {"preguntas", "resumen", "indice_global"}
    assert expected_sheets.issubset(set(excel.sheet_names))

    questions_df = excel.parse("preguntas")
    assert not questions_df.empty, "El archivo Excel está vacío o sin resultados."

    sections_df = excel.parse("resumen")
    if "normalized_score" in sections_df:
        assert sections_df["normalized_score"].dropna().between(0, 20).all()

    csv_df = pd.read_csv(csv_path, encoding="utf-8-sig")

    with json_path.open("r", encoding="utf-8") as handler:
        payload = json.load(handler)

    json_rows = payload.get("rows", [])
    assert json_rows, "El archivo JSON no contiene registros"
    json_df = pd.DataFrame(json_rows)

    _assert_consistent_exports(questions_df, csv_df, json_df, "politica_nacional")

    assert payload["metrics"]["global"]["normalized_max"] == pytest.approx(
        global_summary["normalized_max"]
    )
    assert payload["metrics"]["methodology"] == "politica_nacional"

# pytest tests/exporting/test_integridad_politica.py -v