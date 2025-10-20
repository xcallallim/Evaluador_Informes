# py tests/test_loader.py
# py -m tests.test_loader

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.preprocessing.loader import DocumentLoader

loader = DocumentLoader()
doc = loader.load("data/inputs/test_loader.txt")

print("✅ METADATOS DEL DOCUMENTO:")
print(doc.metadata)

print("\n✅ CONTENIDO DEL DOCUMENTO (primeros 200 caracteres):")
print(doc.content[:200])