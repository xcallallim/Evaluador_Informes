import json
import os

# -------------------------------
# Configuración de rutas
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, "config")

ruta_secciones = os.path.join(CONFIG_DIR, "secciones_institucional.json")
ruta_niveles = os.path.join(CONFIG_DIR, "niveles_institucional.json")

print(f"📁 Verificando archivos en: {CONFIG_DIR}")

# -------------------------------
# Cargar archivos JSON
# -------------------------------
try:
    with open(ruta_secciones, "r", encoding="utf-8") as f:
        secciones = json.load(f)
    with open(ruta_niveles, "r", encoding="utf-8") as f:
        niveles = json.load(f)
except Exception as e:
    print(f"❌ Error al leer JSON: {e}")
    exit()

# -------------------------------
# Verificar correspondencia de criterios
# -------------------------------
print("\n🔍 Verificando compatibilidad entre secciones y niveles...\n")

criterios_en_secciones = set()
for seccion, criterios in secciones.items():
    print(f"📘 Sección: {seccion}")
    for criterio, preguntas in criterios.items():
        criterios_en_secciones.add(criterio)
        if criterio not in niveles:
            print(f"   ❌ Criterio no definido en niveles: {criterio}")
        elif not preguntas:
            print(f"   ⚠️ Criterio '{criterio}' sin preguntas definidas.")
        else:
            print(f"   ✅ {criterio}: {len(preguntas)} pregunta(s) definida(s).")

# -------------------------------
# Verificar si hay criterios en niveles que no se usan
# -------------------------------
criterios_en_niveles = set(niveles.keys())
no_usados = criterios_en_niveles - criterios_en_secciones

if no_usados:
    print("\n⚠️ Criterios definidos en niveles pero no usados en secciones:")
    for c in no_usados:
        print(f"   - {c}")
else:
    print("\n✅ Todos los criterios definidos en niveles se están utilizando.")

print("\n🔎 Verificación completada.")
