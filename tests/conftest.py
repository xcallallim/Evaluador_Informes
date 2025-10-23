"""Configuración compartida para las pruebas del Evaluador de Informes."""

import sys
from pathlib import Path

# Ensure the project root is on sys.path for imports like `data.preprocessing.loader`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))