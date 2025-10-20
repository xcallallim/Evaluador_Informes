import os
import json
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------------
# Configuración inicial
# -------------------------------------------------------
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

STOPWORDS_ES = set(stopwords.words("spanish"))

# -------------------------------------------------------
# Normalización de texto
# -------------------------------------------------------
def _normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-záéíóúñü\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# -------------------------------------------------------
# Construcción del índice de chunks
# -------------------------------------------------------
def build_index(secciones):
    index = []
    for nombre, partes in secciones.items():
        if not partes:
            continue
        for i, texto in enumerate(partes):
            index.append({
                "seccion": nombre,
                "chunk_id": f"{nombre}_{i+1}",
                "texto": texto.strip()
            })
    return index

# -------------------------------------------------------
# Recuperar los chunks más relevantes (TF-IDF)
# -------------------------------------------------------
def retrieve_chunks(index, query, top_k=3):
    if not index:
        return []

    docs = [_normalize_text(c["texto"]) for c in index]
    query = _normalize_text(query)

    vectorizer = TfidfVectorizer(stop_words=list(STOPWORDS_ES))
    tfidf_matrix = vectorizer.fit_transform(docs + [query])
    query_vector = tfidf_matrix[-1]

    sim = cosine_similarity(query_vector, tfidf_matrix[:-1])[0]
    top_indices = np.argsort(sim)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "chunk_id": index[idx]["chunk_id"],
            "seccion": index[idx]["seccion"],
            "texto": index[idx]["texto"],
            "similaridad": round(float(sim[idx]), 3)
        })
    return results

# -------------------------------------------------------
# Cargar rúbricas según tipo de informe
# -------------------------------------------------------
def load_rubricas(tipo_informe, config_dir="config"):
    tipo = tipo_informe.lower().strip()
    if tipo == "institucional":
        ruta_secciones = os.path.join(config_dir, "secciones_institucional.json")
        ruta_niveles = os.path.join(config_dir, "niveles_institucional.json")
    elif tipo == "politica nacional":
        ruta_secciones = os.path.join(config_dir, "secciones_politica.json")
        ruta_niveles = os.path.join(config_dir, "niveles_politica.json")
    else:
        raise ValueError("Tipo de informe no reconocido.")

    with open(ruta_secciones, "r", encoding="utf-8") as f:
        secciones = json.load(f)
    with open(ruta_niveles, "r", encoding="utf-8") as f:
        niveles = json.load(f)

    return secciones, niveles

# -------------------------------------------------------
# Recuperación de fragmentos por sección y criterio/pregunta
# -------------------------------------------------------
def retrieve_for_rubrica(secciones, tipo_informe, top_k=3):
    """
    Recupera los fragmentos relevantes para cada pregunta o criterio
    según el tipo de informe (institucional o política nacional).
    Devuelve un diccionario con los resultados más similares.
    """
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")

    # Cargar rúbricas según tipo de informe
    if tipo_informe == "institucional":
        path_secciones = os.path.join(base_dir, "secciones_institucional.json")
        path_niveles = os.path.join(base_dir, "niveles_institucional.json")
    else:
        path_secciones = os.path.join(base_dir, "secciones_politica.json")
        path_niveles = os.path.join(base_dir, "niveles_politica.json")

    with open(path_secciones, "r", encoding="utf-8") as f:
        secciones_json = json.load(f)

    # Construir índice de texto (cada sección segmentada)
    index = build_index(secciones)

    resultados = {}

    # Caso 1: Institucional → se evalúa por criterios
    if tipo_informe == "institucional":
        for seccion, criterios in secciones_json.items():
            resultados[seccion] = {}
            for criterio, preguntas in criterios.items():
                resultados[seccion][criterio] = []
                for pregunta in preguntas:
                    r = retrieve_chunks(index, pregunta, top_k)
                    resultados[seccion][criterio].append({
                        "pregunta": pregunta,
                        "resultados": r
                    })

    # Caso 2: Política Nacional → se evalúa por sección y preguntas directas
    elif tipo_informe == "politica nacional":
        for seccion, preguntas in secciones_json.items():
            resultados[seccion] = []
            for pregunta in preguntas:
                r = retrieve_chunks(index, pregunta, top_k)
                resultados[seccion].append({
                    "pregunta": pregunta,
                    "resultados": r
                })

    return resultados
