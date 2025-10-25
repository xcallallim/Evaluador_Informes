import pandas as pd
from pathlib import Path

# Ruta del archivo exportado
path = Path("data/outputs/resultados_informe_institucional_demo_test.xlsx")

# Verificar si el archivo existe
if not path.exists():
    print("⚠️ No se encontró el archivo en:", path.resolve())
    print("👉 Verifica si se guardó en otra carpeta, como data/outputs/")
else:
    # Cargar el archivo Excel
    xl = pd.ExcelFile(path)
    print("\n✅ Hojas detectadas:", xl.sheet_names)