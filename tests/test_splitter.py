#py test/test_splitter.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.splitter import split_text, save_chunks

# Carpeta donde se encuentran los textos limpios
CARPETA_RESULTADOS = r"C:\Ceplan\03. Orden de Servicio\07. Entregable 07\Evaluador_Informes\resultados"

def main():
    print("=== TEST SPLITTER ===")
    print(f"📂 Carpeta de búsqueda: {CARPETA_RESULTADOS}")

    # Listar archivos disponibles
    txt_files = [f for f in os.listdir(CARPETA_RESULTADOS) if f.endswith(".txt")]
    if not txt_files:
        print("❌ No se encontraron archivos .txt en la carpeta de resultados.")
        return

    print("\nArchivos disponibles:")
    for i, file in enumerate(txt_files, start=1):
        print(f"  {i}. {file}")

    # Selección del archivo
    opcion = input("\nSeleccione el número del archivo a procesar: ").strip()
    if not opcion.isdigit() or int(opcion) < 1 or int(opcion) > len(txt_files):
        print("❌ Opción no válida.")
        return

    archivo_seleccionado = txt_files[int(opcion) - 1]
    ruta_archivo = os.path.join(CARPETA_RESULTADOS, archivo_seleccionado)
    print(f"\n📄 Procesando archivo: {ruta_archivo}")

    # Leer texto limpio
    with open(ruta_archivo, "r", encoding="utf-8") as f:
        texto = f.read()

    # Dividir texto
    chunks = split_text(texto, chunk_size=1200, chunk_overlap=150)

    # Guardar resultados
    output_dir = os.path.join(CARPETA_RESULTADOS, "chunks")
    save_chunks(chunks, output_dir, base_name=os.path.splitext(archivo_seleccionado)[0])

    # Resumen rápido
    print(f"\n✨ Proceso completado. Se generaron {len(chunks)} fragmentos.")
    print("📘 Vista previa:")
    for i, doc in enumerate(chunks[:3], start=1):
        preview = doc.page_content[:180].replace("\n", " ") + "..."
        print(f"\n🔹 Fragmento {i}:\n{preview}")

if __name__ == "__main__":
    main()
