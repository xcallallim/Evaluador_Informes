from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
import os

def split_text(cleaned_text: str, chunk_size: int = 1000, chunk_overlap: int = 150):
    """
    Divide el texto limpio en fragmentos (chunks) de tamaño configurable.
    Devuelve una lista de objetos Document (LangChain) con metadatos mínimos.
    """
    if not cleaned_text.strip():
        print("⚠️ No hay texto para dividir.")
        return []

    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    chunks = splitter.create_documents([cleaned_text])

    print(f"✅ Texto dividido en {len(chunks)} fragmentos.")
    return chunks


def save_chunks(chunks, output_folder: str, base_name: str = "chunks"):
    """
    Guarda los fragmentos generados en archivos de texto individuales.
    """
    os.makedirs(output_folder, exist_ok=True)
    for i, doc in enumerate(chunks, start=1):
        file_path = os.path.join(output_folder, f"{base_name}_{i:03}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(doc.page_content.strip())

    print(f"💾 Se guardaron {len(chunks)} fragmentos en '{output_folder}'.")

