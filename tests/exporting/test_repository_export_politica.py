import json

import pandas as pd
import pytest

from services.evaluation_service import EvaluationService

@pytest.mark.integration
def test_repository_generates_excel_and_csv(tmp_path):
    """Valida que la exportación política genere artefactos completos y consistentes."""

    #output_dir = tmp_path / "exports"
    #output_dir.mkdir()
    from pathlib import Path
    output_dir = Path("data/examples")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "resultados_informe_politica_demo_test.xlsx"

    service = EvaluationService()

    evaluation, metrics = service.run(
        input_path="data/examples/informe_politica_demo.txt",
        tipo_informe="institucional",
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
    assert global_summary["normalized_max"] == pytest.approx(20.0)
    assert 0.0 <= global_summary["normalized_score"] <= 20.0

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

    question_scores = questions_df["question_score"].dropna()
    assert question_scores.between(0, 2).all(), "Puntajes de preguntas fuera de rango [0, 2]"

    sections_df = excel.parse("resumen")
    if "normalized_score" in sections_df:
        assert sections_df["normalized_score"].dropna().between(0, 20).all()

    with csv_path.open("r", encoding="utf-8-sig") as handler:
        first_line = handler.readline()
    assert "question_score" in first_line

    with json_path.open("r", encoding="utf-8") as handler:
        payload = json.load(handler)
    assert payload["metrics"]["global"]["normalized_max"] == 20.0
    assert payload["metrics"]["methodology"] == "politica_nacional"

# pytest tests/exporting/test_repository_export_politica.py -v