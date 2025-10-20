# main.py
import os
import sys
import traceback

from data.preprocessing.loader import load_file
from preprocessing.cleaner import clean_text
from preprocessing.segmenter import segment_text

# ---------------------------------------------
# Helpers
# ---------------------------------------------
def ask_tipo_informe() -> str:
    opciones = {"1": "institucional", "2": "politica nacional"}
    while True:
        tipo = input("Seleccione el tipo de informe (1 = institucional, 2 = política nacional): ").strip()
        if tipo in opciones:
            return opciones[tipo]
        print("❌ Opción no válida. Intente nuevamente.")

def ask_ruta_archivo() -> str:
    while True:
        ruta = input("Por favor, ingresa la ruta del archivo PDF o DOCX: ").strip().strip('"')
        if os.path.isfile(ruta):
            return os.path.abspath(ruta)
        print("❌ La ruta no existe o no es un archivo válido. Intente nuevamente.")

def ask_pagina_inicio() -> int:
    while True:
        pagina = input("📘 Indique el número de página donde inicia el contenido real (entero ≥ 1): ").strip()
        if pagina.isdigit() and int(pagina) >= 1:
            return int(pagina)
        print("❌ Número inválido. Intente nuevamente (ej. 3).")

def write_text(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text or "")

def print_resumen_secciones(secciones: dict):
    print("\n📄 Resumen de secciones detectadas:")
    for nombre, contenido in secciones.items():
        ok = isinstance(contenido, list) and len(contenido) > 0
        print(f"- {nombre}: {'✅ Detectada' if ok else '⚠️ No encontrada'}")

# ---------------------------------------------
# Main
# ---------------------------------------------
def main():
    print("Evaluador de Informes iniciado correctamente ✅")

    # 1️⃣ Entradas del usuario
    tipo_informe = ask_tipo_informe()
    ruta = ask_ruta_archivo()
    pagina_inicio = ask_pagina_inicio()

    try:
        # 2️⃣ Cargar texto desde el archivo
        texto = load_file(ruta)

        # 3️⃣ Preparar carpeta de salida
        nombre_base = os.path.splitext(os.path.basename(ruta))[0]
        carpeta_salida = os.path.join(os.getcwd(), "resultados")
        os.makedirs(carpeta_salida, exist_ok=True)

        # Guardar texto original
        ruta_original = os.path.join(carpeta_salida, f"{nombre_base}_01_original.txt")
        write_text(ruta_original, texto)
        print(f"\n📤 Texto original extraído guardado en: {ruta_original}")

        # 4️⃣ Limpiar texto a partir de la página indicada
        texto_limpio = clean_text(texto, pagina_inicio=pagina_inicio)
        ruta_limpio = os.path.join(carpeta_salida, f"{nombre_base}_02_limpio.txt")
        write_text(ruta_limpio, texto_limpio)
        print(f"🧹 Texto limpio guardado en: {ruta_limpio}")

        # 5️⃣ Segmentar el texto limpio
        secciones = segment_text(texto_limpio, tipo_informe)

        # 6️⃣ Guardar texto segmentado
        ruta_segmentado = os.path.join(carpeta_salida, f"{nombre_base}_03_segmentado.txt")
        with open(ruta_segmentado, "w", encoding="utf-8") as f:
            for nombre, contenido in secciones.items():
                f.write(f"\n\n=== {nombre.upper()} ===\n")
                if isinstance(contenido, list) and contenido:
                    for parte in contenido:
                        f.write(parte.strip() + "\n")
                else:
                    f.write("[Sección no encontrada]\n")

        # 7️⃣ Resumen de secciones detectadas
        print_resumen_secciones(secciones)

        # 8️⃣ Rutas de salida finales
        print(f"\n💾 Archivos generados en: {carpeta_salida}")
        print(f"  - 01 Original:   {ruta_original}")
        print(f"  - 02 Limpio:     {ruta_limpio}")
        print(f"  - 03 Segmentado: {ruta_segmentado}")

    except Exception as e:
        print("❌ Error durante la ejecución:")
        print(f"   {e}")
        traceback.print_exc(file=sys.stdout)

if __name__ == "__main__":
    main()
