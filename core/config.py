# core/config.py
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
# .\venv\Scripts\activate

import os

# Ruta base del proyecto (evaluador_informes)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ruta de Tesseract OCR (Windows)
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Ruta de Ghostscript para integrar Camelot y detectar tablas
GHOSTSCRIPT_PATH = r"C:\Program Files\gs\gs10.06.0\bin\gswin64c.exe"


# =========================
# CARPETAS PRINCIPALES
# =========================
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_DIR = os.path.join(DATA_DIR, "inputs")      # Documentos a evaluar
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")    # Resultados (CSV, JSON)
CRITERIA_DIR = os.path.join(DATA_DIR, "criteria") # Criterios JSON
PREPROCESSING_DIR = os.path.join(BASE_DIR, "..", "data", "preprocessing")
PIPELINE_DIR = os.path.join(BASE_DIR, "..", "data", "pipeline")

MODELS_DIR = os.path.join(BASE_DIR, "models")     # Modelos IA (embeddings, fine-tuning, etc.)
SERVICES_DIR = os.path.join(BASE_DIR, "services") # Lógica principal
LOG_DIR = os.path.join(BASE_DIR, "logs")          # Ruta para logs

TABLES_OUTPUT_DIR = os.path.join(DATA_DIR, "outputs", "tables") # Donde guardar tablas extraídas

# =========================
# CONFIGURACIÓN GLOBAL
# =========================
DEFAULT_ENCODING = "utf-8"
LOG_LEVEL = "INFO"
DEBUG = True  # Cambiar a False en producción

# =========================
# CREAR CARPETAS SI NO EXISTEN
# =========================
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CRITERIA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PIPELINE_DIR, exist_ok=True)

# Nota:
# Las carpetas models/ y services/ no se crean aquí porque contienen código,
# No datos. Se crean manualmente como parte de la estructura del proyecto.

# ============================
# CUSTOM CLEANING PATTERNS
# ============================

CUSTOM_HEADERS = [
    r"Gobierno del Perú",
    r"Oficina de Planeamiento",
    r"www\.gob\.pe",
    r"Ministerio de .*",                 
    r"República del Perú",
    r"Informe de Evaluación Institucional",
    r"Informe de Evaluación Institucional del año 2023",
    r"Informe de Evaluación Institucional al año 2023",
    r"Informe de Evaluación Institucional del año 2024",
    r"Informe de Evaluación Institucional del año 2025",
    r"Informe de Evaluación",
    r"Informe de Evaluación de la Política Nacional .*",
    r"Informe de Evaluación de la .*"
]

CUSTOM_FOOTERS = [
    r"Página \d+ de \d+",
    r"Elaborado por .*",
    r"Elaboración: .*",
    r"Fuente: .*",
    r"Revisado por .*",
    r"Documento generado electrónicamente",
]





