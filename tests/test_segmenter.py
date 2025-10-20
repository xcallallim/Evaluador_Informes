# py tests/test_segmenter.py
# -*- coding: utf-8 -*-
"""
Test simple para segment_text():
- Pide ruta de archivo limpio (.txt)
- Segmenta según tipo de informe
- Muestra qué se detectó y qué no
- Guarda los segmentos encontrados en resultados/segmentados_simple/
"""

import sys, os, re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.segmentos import segment_text  # <-- Cambia según el nombre del archivo que tengas

OUTPUT_DIR = os.path.join("resultados", "segmentados_simple")

def sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name)[:60]

def run_test(file_path: str, tipo: str):
    # Leer el texto limpio
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Segmentar
    print("\n🔎 Segmentando...")
    secciones = segment_text(text, tipo)

    if not secciones:
        print("❌ No se detectaron secciones. Revisa tipo de informe o contenido.")
        return

    # Mostrar resultados
    print("\n📋 Resultado de segmentación:")
    detectadas = []
    for nombre, contenido in secciones.items():
        estado = "✅ Detectado" if contenido.strip() else "⚠️ Vacío"
        detectadas.append(nombre)
        print(f" - {nombre}: {estado}")

    # Calcular faltantes comparando con catálogo oficial
    if tipo == "institucional":
        catalogo = [
            "Resumen Ejecutivo",
            "Prioridades de la política institucional",
            "Análisis de resultados de los objetivos estratégicos institucionales",
            "Diagnóstico sobre los OEI priorizados con bajo nivel de cumplimiento",
            "Análisis de implementación de las acciones estratégicas institucionales",
            "Análisis de implementación de las AEI de los OEI priorizados",
            "Análisis de los productos de la AEI",
            "Análisis de la ejecución operativa en las AEI críticas",
            "Aplicación de las recomendaciones para mejorar la implementación de las AEI",
            "Conclusiones",
            "Recomendaciones",
            "Anexos"
        ]
    else:
        catalogo = [
            "Resumen Ejecutivo",
            "Descripción de la política nacional",
            "Análisis de los resultados de la política nacional",
            "Análisis de implementación",
            "Conclusiones",
            "Recomendaciones"
        ]

    faltantes = [c for c in catalogo if c not in secciones]
    if faltantes:
        print("\n⚠️ Secciones no detectadas:")
        for s in faltantes:
            print(f" ❌ {s}")

    # Guardar secciones detectadas como TXT
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(file_path))[0]
    for nombre, contenido in secciones.items():
        filename = f"{base}_{sanitize_filename(nombre)}.txt"
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(contenido)
        print(f" 💾 Guardado: {filename}")

    print(f"\n✅ Segmentación completada. Archivos en: {OUTPUT_DIR}")

if __name__ == "__main__":
    ruta = input("📄 Ruta del archivo .txt limpio: ").strip()
    tipo = input("📑 Tipo de informe (institucional / politica nacional): ").strip().lower()
    run_test(ruta, tipo)
