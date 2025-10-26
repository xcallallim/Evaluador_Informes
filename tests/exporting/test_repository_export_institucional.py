import json

import pandas as pd
import pytest

from services.evaluation_service import EvaluationService


@pytest.mark.integration
def test_repository_generates_excel_and_csv(tmp_path):
    """Valida que la exportación institucional genere artefactos completos y consistentes."""

    #output_dir = tmp_path / "exports"
    #output_dir.mkdir(parents=True, exist_ok=True)
    from pathlib import Path
    output_dir = Path("data/examples")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "resultados_informe_institucional_demo_test.xlsx"

    service = EvaluationService()

    evaluation, metrics = service.run(
        input_path="data/examples/informe_institucional_demo.txt",
        tipo_informe="institucional",
        mode="global",
        output_format="xlsx",
        output_path=output_path,
        criteria_path="data/criteria/metodología_institucional.json",
    )

    assert evaluation.document_type == "institucional"

    metrics_dict = metrics
    global_summary = metrics_dict["global"]

    assert metrics_dict["methodology"] == "institucional"
    assert global_summary["normalized_min"] == pytest.approx(0.0)
    assert global_summary["normalized_max"] == pytest.approx(100.0)
    assert 0.0 <= global_summary["normalized_score"] <= 100.0

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
    required_cols = {
        "tipo_informe",
        "section_title",
        "dimension_name",
        "question_text",
        "question_score",
        "justification",
        "model_name",
        "pipeline_version",
        "criteria_version",
        "timestamp",
    }
    missing = required_cols - set(questions_df.columns)
    assert not missing, f"Faltan columnas obligatorias: {missing}"
    assert len(questions_df) > 0, "El archivo Excel está vacío o sin resultados."

    questions_df = questions_df.dropna(subset=["question_score"])
    structure_mask = questions_df["dimension_name"].str.lower() == "estructura"
    structure_scores = questions_df.loc[structure_mask, "question_score"]
    if not structure_scores.empty:
        assert structure_scores.isin({0, 1}).all(), "Estructura solo admite puntajes 0 o 1"
    other_scores = questions_df.loc[~structure_mask, "question_score"]
    if not other_scores.empty:
        assert other_scores.isin({0, 1, 2, 3, 4}).all(), "Puntajes de criterios deben estar en [0, 4]"

    sections_df = excel.parse("resumen")
    if "normalized_score" in sections_df:
        normalized_values = sections_df["normalized_score"].dropna()
        assert normalized_values.between(0, 100).all()

    with csv_path.open("r", encoding="utf-8-sig") as handler:
        first_line = handler.readline()
    assert "question_score" in first_line

    with json_path.open("r", encoding="utf-8") as handler:
        payload = json.load(handler)
    assert payload["metrics"]["global"]["normalized_max"] == 100.0
    assert payload["metrics"]["methodology"] == "institucional"
    criteria_entries = payload["metrics"].get("criteria", [])
    assert any(entry["key"] == "estructura" for entry in criteria_entries)

    # pytest tests/exporting/test_repository_export_institucional.py -v