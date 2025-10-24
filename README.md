# Evaluador de Informes

Evaluador de Informes es una canalización modular que automatiza la revisión de
informes institucionales y de políticas nacionales. El proyecto combina etapas de
preprocesamiento documental, segmentación semántica y evaluación asistida por
IA para producir reportes consistentes y trazables.

## Características principales

- **Ingesta unificada de documentos**: `data.preprocessing.loader.DocumentLoader`
  normaliza archivos PDF, DOCX y TXT en una estructura común, incluida la
  paginación y los metadatos relevantes para posteriores análisis.
- **Limpieza y segmentación personalizables**: la combinación de
  `data.preprocessing.cleaner.Cleaner`, `data.preprocessing.segmenter.Segmenter`
  y `data.chunks.splitter.Splitter` aplica reglas específicas para remover ruido,
  detectar secciones y generar *chunks* listos para evaluación.
- **Evaluaciones reproducibles**: `services.evaluation_service.EvaluationService`
  orquesta la preparación del documento, construye *prompts* por criterio y
  consolida los resultados junto con métricas cuantitativas.
- **Seguridad de credenciales integrada**: el acceso a la clave de OpenAI está
  centralizado en `utils.secret_manager.get_openai_api_key()`, que evalúa fuentes
  autorizadas y aplica cifrado basado en passphrase.
- **Pruebas automatizadas**: el directorio `tests/` cubre la ingesta, limpieza,
  segmentación, métrica y servicios de evaluación para facilitar refactorizaciones
  seguras.

## Estructura del repositorio

```
Evaluador_Informes/
├── core/                 # Configuración global, rutas y patrones de limpieza
├── data/                 # Modelos, loaders y etapas de preprocesamiento
├── docs/                 # Documentación técnica complementaria
├── reporting/            # Repositorio de resultados y utilidades de exportación
├── services/             # Servicios de orquestación y lógica de IA
├── utils/                # Herramientas de soporte (secretos, validadores, etc.)
├── tests/                # Suite de pytest con casos unitarios e integrados
├── main.py               # Punto de entrada reservable para CLI externas
└── README.md             # Este archivo
```

## Requisitos previos

- Python 3.10 o superior.
- Dependencias listadas en `requirements.txt`; incluye bibliotecas para OCR
  (Tesseract), extracción de tablas (Camelot/Ghostscript) y componentes de
  LangChain.
- Acceso a una clave de OpenAI cuando se utilice el servicio real (`--real-ai`).

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate  # En Windows usar .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Si no puedes instalar binarios del sistema, consulta `docs/dependencies.md` para
obtener alternativas portátiles de Ghostscript y Tesseract.

## Configuración de seguridad

1. Crea la carpeta `secrets/` en la raíz si no existe.
2. Define la fuente principal de la clave:
   - Variable de entorno `OPENAI_API_KEY` para desarrollo.
   - Archivo temporal `secrets/openai_api_key.txt` (no versionado).
3. Para entornos compartidos ejecuta `python -m utils.seal_key` y proporciona la
   passphrase; se generará `secrets/openai_api_key.enc` usando cifrado Fernet.
4. Establece `OPENAI_KEY_PASSPHRASE` al desplegar para permitir el descifrado.

`get_openai_api_key()` es la única vía autorizada para recuperar la credencial y
fallará explícitamente si ninguna fuente válida está disponible.

## Uso rápido

### Ejecución de la evaluación principal

```bash
python -m services.evaluation_service \
  --input data/inputs/ejemplo.pdf \
  --criteria data/criteria/metodologia_institucional.json \
  --tipo-informe institucional \
  --output data/outputs/resultado.xlsx
```

Argumentos destacados:

- `--modo`: `completo`, `parcial` o `reevaluacion` para controlar el alcance.
- `--solo-seccion/--solo-bloque/--solo-criterio`: filtros cuando se realiza una
  evaluación parcial.
- `--mock-ai` / `--real-ai`: selecciona entre el simulador integrado y el
  proveedor real.
- `--previous-result`: permite reusar puntajes previos al ejecutar una
  reevaluación.

### Validación de criterios

```bash
python -m utils.criteria_validator data/criteria/*.json
```

El comando informa problemas de esquema y retorna un código de salida distinto
de cero si detecta errores.

### Sellado de la clave de OpenAI

```bash
python -m utils.seal_key
```

El asistente solicita la passphrase y crea el archivo cifrado en `secrets/`. Se
puede ejecutar en modo no interactivo con `--non-interactive` y variables de
entorno para integraciones CI/CD.

## Ejecución de pruebas

```bash
pytest
```

La suite incluye casos unitarios e integrales para loaders, limpieza,
segmentación, cálculo de métricas, repositorio de resultados y el servicio de
IA.

## Documentación adicional

- `docs/data_contract.md`: estructura del objeto `Document` y metadatos
  asociados.
- `docs/validation.md`: detalles del sistema de validación y pruebas.
- `docs/dependencies.md`: guías para dependencias con restricciones de
  instalación.
- `docs/security_config.md`: instrucciones completas de seguridad y cifrado.

## Contribuciones

1. Crea un *fork* o rama feature.
2. Asegúrate de que la suite de pruebas pase (`pytest`).
3. Actualiza la documentación correspondiente.
4. Abre un Pull Request describiendo los cambios y los pasos de validación.