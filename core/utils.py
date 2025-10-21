# core/utils.py
# from core.utils import []

import json
import unicodedata
import re
import os

try:
    from langchain_text_splitters import CharacterTextSplitter  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    CharacterTextSplitter = None

# ====================================
# FUNCIONES DE NORMALIZACIÓN
# ====================================

def strip_accents(text: str) -> str:
    """
    Elimina tildes del texto usando Unicode.
    """
    if not isinstance(text, str):
        return text
    text = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def normalize_text(text: str) -> str:
    """
    Normaliza texto: minúsculas, sin tildes, sin símbolos raros.
    - Usado para comparar títulos en segmentación.
    - NO usar en cleaning porque destruye estructura original del contenido.
    """
    if not isinstance(text, str):
        return ""
    text = strip_accents(text.lower())
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_spaces(text: str) -> str:
    """
    Limpia espacios duplicados y saltos innecesarios.
    """
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def normalize_whitespace(text: str) -> str:
    """
    Normaliza espacios de forma segura SIN alterar el contenido semántico.
    Ideal para cleaning.py.
    - Mantiene mayúsculas/tildes
    - NO destruye estructura del texto
    - Limpia cosas raras de PDF/Word exportados
    """
    if not isinstance(text, str):
        return ""

    # Normaliza caracteres unicode
    text = unicodedata.normalize("NFKC", text)

    # Quita bullets invisibles
    text = text.replace("\uf0b7", "").replace("\u2022", "")

    # Limpieza de espacios sin tocar contenido
    text = re.sub(r"[ \t]+", " ", text)         # colapsa tabs/espacios
    text = re.sub(r"[ ]*\n[ ]*", "\n", text)    # limpia bordes de cada línea
    return text.strip()


# ====================================
# FUNCIONES PARA ARCHIVOS JSON
# ====================================
def load_json(path: str):
    """
    Carga archivos JSON desde una ruta segura.
    Retorna Python dict/list o None si falla.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] No se encontró el archivo JSON: {path}")
    except json.JSONDecodeError:
        print(f"[ERROR] El archivo JSON está mal formado: {path}")
    except Exception as e:
        print(f"[ERROR] No se pudo abrir JSON {path}: {e}")
    return None


def save_json(data, path: str):
    """
    Guarda un diccionario o lista como JSON.
    Uso:
        save_json({"a": 1}, "archivo.json")
    """
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[ERROR] No se pudo guardar JSON en {path}: {e}")


# ====================================
# UTILERÍA VISUAL
# ====================================

def print_header(title: str):
    """
    Imprime un encabezado legible en consola para separar pasos del flujo.
    Ejemplo:
        >>> print_header("Cargando Documento")
        ===== Cargando Documento =====
    """
    print(f"\n{'=' * 10} {title.upper()} {'=' * 10}")


# ====================================
# MANEJO DE CARPETAS
# ====================================

def ensure_dir(path: str):
    """
    Crea una carpeta si no existe. Evita errores al guardar archivos dinámicamente.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

# ====================================
# DIVISION POR CHUNKS
# ====================================

def split_text(text: str, chunk_size: int = 1500, chunk_overlap: int = 150) -> list:
    """
    Divide texto completo en chunks usando LangChain.
    Esta función puede ser usada desde cualquier módulo (Cleaner, Retriever, etc.)
    """
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return splitter.split_text(text)