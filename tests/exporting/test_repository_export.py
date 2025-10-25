import pytest
from pathlib import Path
import pandas as pd
from services.evaluation_service import EvaluationService

OUTPUT_DIR = Path("data/examples")

@pytest.mark.integration
def test_repository_generates_excel_and_csv(tmp_path):
    """
    Verifica que la exportación de resultados genera archivos Excel y CSV válidos
    con columnas CEPLAN completas.
    """

    # === 1. Ejecutar evaluación real (modo mock por defecto) ===
    service = EvaluationService()
    output_path = OUTPUT_DIR / "resultados_informe_institucional_demo_test.xlsx"

    evaluation, metrics = service.run(
        input_path="data/examples/informe_institucional_demo.txt",
        tipo_informe="institucional",
        mode="global",
        output_format="xlsx",
        output_path=output_path,
        criteria_path="data/criteria/metodologÍa_institucional.json"
    )

    # === 2. Comprobaciones ===
    # Verifica que los archivos existen
    assert output_path.exists(), "No se generó el archivo Excel"
    csv_path = output_path.with_suffix(".csv")
    assert csv_path.exists(), "No se generó el archivo CSV"

    # === 3. Leer y verificar estructura del Excel ===
    df = pd.read_excel(output_path, sheet_name="preguntas_institucional")
    required_cols = {
        "tipo_informe", "seccion", "criterio", "pregunta", "texto",
        "score", "justificacion", "model_name",
        "pipeline_version", "criteria_version", "timestamp"
    }

    assert required_cols.issubset(df.columns), f"Faltan columnas: {required_cols - set(df.columns)}"
    assert len(df) > 0, "El Excel no contiene filas de resultados"
    assert df["score"].between(0, 4).all(), "Puntajes fuera de rango esperado"

    print(f"[INFO] Archivo exportado correctamente: {output_path.name} ({len(df)} filas)")
