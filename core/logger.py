# core/logger.py
# from core.logger import log_info, log_step

"""
Logger simple para desarrollo.
Opciones futuras:
- Guardar log en archivo .log
- Guardar log en formato JSON
- Enviar métricas a dashboard
"""

from datetime import datetime

def _log(level: str, message: str):
    """Formato base del log"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

def log_info(message: str):
    _log("INFO", message)

def log_warn(message: str):
    _log("WARN", message)

def log_error(message: str):
    _log("ERROR", message)

def log_step(title: str):
    """Resalta pasos importantes del pipeline"""
    print("\n" + "="*50)
    _log("STEP", title.upper())
    print("="*50)
