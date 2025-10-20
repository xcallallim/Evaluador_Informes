# py tests/test_cleaner.py
# py -m tests.test_cleaner

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.utils import ensure_dir
from data.preprocessing.loader import DocumentLoader
from data.preprocessing.cleaner import Cleaner

def main():
    print("\n=== TEST: CLEANER ===")

    # Asegurar carpeta de salida
    ensure_dir("data/outputs")

    # 1) Cargar documento
    loader = DocumentLoader()
    cleaner = Cleaner(
        remove_headers=True,
        remove_page_numbers=True,
        use_custom_headers=True,
    )

    input_path = "data/inputs/IEI 2023 - Gob. Regional de La Libertad.pdf"
    doc = loader.load(input_path)  # sin tablas (modificar en futuro)

    # 2) Limpiar documento
    clean_text, rep = cleaner.clean_document(doc, return_report=True)

    # 3) Mostrar reporte
    print("\n✅ REPORTE DE LIMPIEZA:")
    for k, v in rep.items():
        print(f"{k}: {v}")

    # 4) Mostrar preview
    print("\n✅ TEXTO LIMPIO (primeros 500 chars):")
    print(clean_text[:500])

    # 5) Guardar a archivo
    out_path = "data/outputs/cleaned_test2.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(clean_text)
    print(f"\n📝 Archivo guardado en: {out_path}")

    print("\n=== FIN TEST CLEANER ===")

if __name__ == "__main__":
    main()
