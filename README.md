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
  **Exportación multiformato**: `reporting.EvaluationRepository` serializa los
  resultados en JSON, CSV, XLSX y Parquet, incorporando metadatos, métricas y
  trazabilidad por fragmento.
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
- Dependencias listadas en `requirements.txt`, agrupadas por categoría:
  - **OCR y preprocesamiento**: `pdfplumber`, `PyMuPDF`, `pytesseract`,
    `opencv-python-headless`, `Pillow`, `python-docx`, `camelot-py`.
  - **IA y orquestación**: `langchain`, `langchain-core`, `langchain-community`,
    `langchain-text-splitters`, `openai`, `tenacity`.
  - **Utilidades y análisis**: `pandas`, `numpy`, `rapidfuzz`, `scikit-learn`,
    `tabulate`, `tqdm`.
  - **Seguridad y exportación**: `cryptography` para el sellado de claves,
    `openpyxl` y `pyarrow` para los reportes tabulares.
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

## Reportes y exportación

- `metrics.py` contiene funciones reutilizables para calcular promedios
  ponderados, escalas numéricas y agregados por bloque.
- `reporting/repository.py` implementa `EvaluationRepository`, responsable de
  materializar resultados en JSON, CSV, XLSX o Parquet a partir del objeto de
  dominio.
- `data/models/evaluation.py` define las clases de datos (`SectionResult`,
  `QuestionResult`, etc.) utilizadas por el repositorio y expone métodos
  `to_dict()` para serializar sin pérdida de información.

Ejemplo de exportación directa desde Python:

```python
import json
from pathlib import Path

from metrics import calculate_metrics
from reporting.repository import EvaluationRepository

# "resultado" es un EvaluationResult devuelto por EvaluationService.
criteria = json.loads(Path("data/criteria/metodologia_institucional.json").read_text())
repository = EvaluationRepository()
repository.export(
    evaluation=resultado,
    metrics_summary=calculate_metrics(resultado, criteria).to_dict(),
    output_path=Path("data/outputs/resultado.parquet"),
    output_format="parquet",
)
```

Los formatos tabulares (`csv`, `xlsx`, `parquet`) requieren tener instalados
`pandas`, `openpyxl` y `pyarrow`. El método añade metadatos globales, métricas
por sección y los fragmentos relevantes por pregunta para facilitar auditorías.

### Sellado de la clave de OpenAI

```bash
python -m utils.seal_key
```

El asistente solicita la passphrase y crea el archivo cifrado en `secrets/`. Se
puede ejecutar en modo no interactivo con `--non-interactive` y variables de
entorno para integraciones CI/CD.

## Flujo de procesamiento

1. **Ingesta y normalización.** `data.preprocessing.loader.DocumentLoader`
   detecta automáticamente el tipo de archivo y construye un
   `data.models.document.Document` con texto plano, páginas desagregadas y
   metadatos. Los extractores especializados agregan tablas, imágenes y
   advertencias en `metadata["issues"]`.
2. **Limpieza contextual.** `data.preprocessing.cleaner.Cleaner` aplica reglas
   específicas según `metadata["extraction_method"]` para eliminar encabezados
   repetidos, reconstruir tablas a texto y marcar las secciones con OCR.
3. **Segmentación semántica.** `data.preprocessing.segmenter.Segmenter` y
   `data.chunks.splitter.Splitter` generan fragmentos consistentes, respetando
   límites de tokens y etiquetas definidas en `core/config.py`.
4. **Construcción de prompts.** `services.prompt_builder` (invocado desde
   `EvaluationService`) combina los criterios JSON con el contenido del
   documento, aplicando plantillas según el tipo de informe.
5. **Evaluación y métricas.** `services.evaluation_service.EvaluationService`
   coordina la ejecución del modelo (real o simulado), agrega métricas con
   `metrics.calculate_metrics` y consolida un Excel/JSON trazable en `reporting/`.

Cada etapa registra *artifacts* intermedios en `reporting/workdir/` cuando se
habilita el modo detallado (`--modo completo --guardar-pasos`). Esto facilita la
auditoría de decisiones y el ajuste fino de reglas.

## Integración desde Python

Además del CLI, puedes orquestar la evaluación desde tu propio código:

```python
from services.evaluation_service import EvaluationService
from utils.secret_manager import get_openai_api_key

service = EvaluationService(
    api_key=get_openai_api_key(),
    use_mock_model=True,  # Cambia a False para invocar el modelo real
)

resultado = service.evaluate_document(
    input_path="data/inputs/ejemplo.pdf",
    criteria_path="data/criteria/metodologia_institucional.json",
    report_type="institucional",
)

print(resultado.summary.score)
```

El objeto de respuesta expone el puntaje consolidado (`summary.score`), las
métricas por criterio (`criteria`) y el detalle por fragmento (`chunks`). Puedes
serializarlo mediante `resultado.to_dict()` para integrarlo con otros sistemas.

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

## Buenas prácticas operativas

- Reutiliza `utils.prompt_validator.validate_prompt()` desde un REPL o script
  para revisar la calidad de los prompts antes de enviar evaluaciones reales.
- Habilita `OPENAI_KEY_PASSPHRASE` únicamente en el entorno donde se descifrará
  la credencial; evita compartirla en texto plano.
- Versiona los criterios JSON junto con el código que depende de ellos para
  asegurar reproducciones históricas.

## Contribuciones

1. Crea un *fork* o rama feature.
2. Asegúrate de que la suite de pruebas pase (`pytest`).
3. Actualiza la documentación correspondiente.
4. Abre un Pull Request describiendo los cambios y los pasos de validación.