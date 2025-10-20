# py tests/test_section_loader.py
# py -m tests.test_section_loader

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.criteria.section_loader import SectionLoader

loader = SectionLoader(tipo="politica")
for line in [
    "1. Resumen Ejecutivo",
    "Capítulo II - Análisis de los resultados de la política nacional",
    "SECCIÓN 4: CONCLUSIONES",
]:
    print(line, "=>", loader.identify_section(line))