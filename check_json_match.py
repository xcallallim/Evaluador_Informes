import json

# Cargar los dos archivos
with open("config/secciones_politica.json", "r", encoding="utf-8") as f:
    secciones = json.load(f)

with open("config/niveles_politica.json", "r", encoding="utf-8") as f:
    niveles = json.load(f)

# Buscar coincidencias
for seccion, preguntas in secciones.items():
    print(f"\n📘 Sección: {seccion}")
    if seccion not in niveles:
        print("❌ No existe en niveles_politica.json")
        continue

    textos_niveles = [p["texto"].strip() for p in niveles[seccion]["preguntas"]]
    for pregunta in preguntas:
        if pregunta.strip() in textos_niveles:
            print(f"   ✅ Coincide: {pregunta[:70]}...")
        else:
            print(f"   ⚠️ No coincide: {pregunta[:70]}...")
