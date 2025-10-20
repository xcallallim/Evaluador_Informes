import json
import os
from rapidfuzz import fuzz, process

# -------------------------------
# Configuración de rutas
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, "config")

ruta_secciones = os.path.join(CONFIG_DIR, "secciones_politica.json")
ruta_niveles = os.path.join(CONFIG_DIR, "niveles_politica.json")

print(f"📁 Verificando archivos en: {CONFIG_DIR}")

# -------------------------------
# Validar existencia de archivos
# -------------------------------
for ruta in [ruta_secciones, ruta_niveles]:
    if not os.path.exists(ruta):
        print(f"❌ No se encontró el archivo: {ruta}")
        exit()

# -------------------------------
# Cargar archivos JSON
# -------------------------------
try:
    with open(ruta_secciones, "r", encoding="utf-8") as f:
        secciones = json.load(f)
    with open(ruta_niveles, "r", encoding="utf-8") as f:
        niveles = json.load(f)
except json.JSONDecodeError as e:
    print(f"❌ Error al leer JSON: {e}")
    exit()

# -------------------------------
# Comparación entre archivos
# -------------------------------
print("\n🔍 Comparando preguntas entre secciones y niveles...\n")

for seccion, preguntas in secciones.items():
    print(f"📘 Sección: {seccion}")
    if seccion not in niveles:
        print("   ❌ No existe esta sección en niveles_politica.json")
        continue

    textos_niveles = [p["texto"].strip() for p in niveles[seccion]["preguntas"]]

    for pregunta in preguntas:
        mejor_coincidencia, puntaje, _ = process.extractOne(
            pregunta, textos_niveles, scorer=fuzz.token_sort_ratio
        )

        if puntaje >= 95:
            print(f"   ✅ Coincide ({puntaje:.1f}%): {pregunta[:70]}...")
        else:
            print(f"   ⚠️ No coincide ({puntaje:.1f}%): {pregunta[:70]}...")
            print(f"      ↳ Mejor coincidencia encontrada: {mejor_coincidencia[:70]} ({puntaje:.1f}%)")

print("\n✅ Revisión completada.")
