# py tests/test_loader_ocr.py
# python -m tests.test_loader_ocr

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.preprocessing.loader import DocumentLoader

loader = DocumentLoader()

doc = loader.load("data/inputs/test_pdf_ocr.pdf")

print("\n✅ METADATOS DEL DOCUMENTO:")
print(doc.metadata)

print("\n✅ CONTENIDO OCR (primeros 500 caracteres):")
print(doc.content[:500])

with open("data/outputs/loaded_test.txt", "w", encoding="utf-8") as f:
    f.write(doc.content)
print("\n📝 Archivo guardado en: data/outputs/loaded_test.txt")