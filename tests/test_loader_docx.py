# py tests/test_loader_docx.py
# python -m tests.test_loader_docx

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.preprocessing.loader import DocumentLoader

loader = DocumentLoader()
doc = loader.load("data/inputs/test_docx.docx")

print("✅ METADATOS DEL DOCUMENTO:")
print(doc.metadata)

print("\n✅ CONTENIDO DEL DOCUMENTO (primeros 300 caracteres):")
print(doc.content[:300])

print("\n🔎 TABLAS DETALLADAS:")
if doc.metadata.get("tables"):
    for i, table in enumerate(doc.metadata["tables"]):
        print(f"--- Tabla #{i+1} ---")
        for row in table:
            print(row)
else:
    print("⚠ No se detectaron tablas en este archivo. Puede no tener tablas verdaderas.")