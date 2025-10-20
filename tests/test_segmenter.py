# py tests/test_segmenter.py
# py -m tests.test_segmenter

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.preprocessing.loader import DocumentLoader
from data.preprocessing.cleaner import Cleaner
from data.criteria.section_loader import SectionLoader
from data.preprocessing.segmenter import Segmenter

print("\n=== TEST: SEGMENTER ===")

loader = DocumentLoader()
cleaner = Cleaner()
segmenter = Segmenter(tipo="institucional")

# Cargar y limpiar
doc = loader.load("data/inputs/IEI 2023 - Gob. Regional de La Libertad.pdf")
clean_doc, _ = cleaner.clean_document(doc, return_report=True)

# Segmentar
seg_doc = segmenter.segment_document(clean_doc)

print("\n✅ SECCIONES DETECTADAS:")
for sid, text in seg_doc.sections.items():
    resumen = text[:150].replace("\n", " ") + ("..." if len(text) > 150 else "")
    print(f"- {sid}: {len(text.split())} palabras | {resumen}")

print("\n=== FIN TEST SEGMENTER ===")

