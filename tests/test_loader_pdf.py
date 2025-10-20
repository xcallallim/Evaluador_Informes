# py tests/test_loader_pdf.py
# python -m tests.test_loader_pdf

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.preprocessing.loader import DocumentLoader

loader = DocumentLoader()
doc = loader.load("data/inputs/IEI 2023 - Gob. Regional de La Libertad.pdf", extract_tables=True)

print("\n✅ Tipo:", ("PDF" if doc.metadata["extension"]==".pdf" else doc.metadata["extension"]))
print("✅ Páginas (conteo):", 0 if not doc.metadata["pages"] else len(doc.metadata["pages"]))
print("✅ Tablas PDF extraídas:", len(doc.metadata.get("tables", {}).get("pdf", [])))
print("\n✅ Preview texto:", doc.content[:400])

with open("data/outputs/loaded_test.txt", "w", encoding="utf-8") as f:
    f.write(doc.content)
print("\n📝 Archivo guardado en: data/outputs/loaded_test.txt")