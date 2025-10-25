import pytest
from pathlib import Path
import pandas as pd
from services.evaluation_service import EvaluationService

OUTPUT_DIR = Path("data/examples")

@pytest.mark.integration
def test_repository_generates_excel_and_csv(tmp_path):
    """
    Verifica que la exportación de resultados genera archivos Excel y CSV válidos,
    con estructura CEPLAN completa y puntajes dentro del rango esperado.
    """

    print("\n[🔍] Iniciando prueba de exportación del repositorio (Excel + CSV)...")

    # === 1. Ejecutar evaluación real (modo mock por defecto) ===
    service = EvaluationService()
    output_path = OUTPUT_DIR / "resultados_informe_institucional_demo_test.xlsx"

    print(f"[⚙️] Ejecutando evaluación simulada del informe: {output_path.stem}")

    evaluation, metrics = service.run(
        input_path="data/examples/informe_institucional_demo.txt",
        tipo_informe="institucional",
        mode="global",
        output_format="xlsx",
        output_path=output_path,
        criteria_path="data/criteria/metodología_institucional.json"
    )

    # === 2. Comprobaciones ===
    print("[🧾] Verificando generación de archivos de salida...")

    # Verifica que los archivos existen
    assert output_path.exists(), f"[❌] No se generó el archivo Excel en {output_path}"
    csv_path = output_path.with_suffix(".csv")
    assert csv_path.exists(), f"[❌] No se generó el archivo CSV complementario en {csv_path}"

    print(f"[✅] Archivos generados correctamente:")
    print(f"     - Excel: {output_path}")
    print(f"     - CSV:   {csv_path}")

    # === 3. Leer y verificar estructura del Excel ===
    print("[🔎] Validando estructura del archivo Excel...")

    excel = pd.ExcelFile(output_path)
    sheet_name = "preguntas_institucional"
    if sheet_name not in excel.sheet_names:
        sheet_name = "preguntas"
    df = excel.parse(sheet_name=sheet_name)
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

    missing = required_cols - set(df.columns)
    assert not missing, f"[❌] Faltan columnas obligatorias: {missing}"
    assert len(df) > 0, "[❌] El archivo Excel está vacío o sin resultados."
    question_scores = df["question_score"].dropna()
    assert question_scores.between(1, 4).all(), "Puntajes fuera de rango esperado"

    # === 4. Resumen final ===
    print("\n[📊] Validación completada exitosamente:")
    print(f"     - Tipo de informe: institucional")
    print(f"     - Filas evaluadas: {len(df)}")
    print(f"     - Columnas CEPLAN verificadas: {len(required_cols)}")
    print(f"     - Puntajes: dentro del rango [0, 4]")
    print(f"[🎯] Exportación verificada y consistente en Excel + CSV.\n")