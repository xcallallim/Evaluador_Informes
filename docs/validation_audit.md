# Evidencia de validación y trazabilidad

## 1. Resumen de pruebas

| Tipo de prueba | Objetivo | Criterios de aprobación | Estado |
| --- | --- | --- | --- |
| Unitarias de preprocesamiento y criterios | Verificar limpieza de documentos, detección de secciones y validez de los criterios oficiales. | Los *fixtures* sintéticos deben conservar metadatos, normalizar encabezados y aceptar los esquemas institucionales y de política nacional sin errores. | ✅ Pasó (`tests/test_cleaner.py`, `tests/test_section_loader.py`, `tests/test_criteria_validator.py`).【F:tests/test_cleaner.py†L1-L109】【F:tests/test_section_loader.py†L1-L160】【F:tests/test_criteria_validator.py†L1-L149】 |
| Integración del pipeline | Asegurar que `EvaluationService` orqueste carga, limpieza, segmentación, evaluación y exportación de extremo a extremo. | Debe persistir `run_id`, modelar secciones y chunks con puntajes válidos y registrar todos los metadatos necesarios por ejecución. | ✅ Pasó (`tests/test_pipeline.py`, `tests/test_pipeline_integration.py`).【F:tests/test_pipeline.py†L374-L460】【F:tests/test_pipeline_integration.py†L676-L717】 |
| Exportación y trazabilidad | Confirmar que los artefactos JSON/CSV/XLSX incluyen columnas críticas y que los resultados son reproducibles entre ejecuciones controladas. | Las exportaciones deben contener identificadores, pesos, metadatos y hashes estables al repetir la evaluación completa. | ✅ Pasó (`tests/resilience/test_global_consistency.py`).【F:tests/resilience/test_global_consistency.py†L317-L526】 |
| Resiliencia y rendimiento | Validar que la canalización mantenga límites de tiempo, memoria y consistencia ante escenarios adversos. | Las pruebas de resiliencia ejercitan reevaluaciones, paralelismo y métricas de desempeño sin provocar errores ni desbordes. | ✅ Pasó (`tests/resilience/test_performance_characteristics.py`, `tests/resilience/test_pipeline_resilience.py`).【F:tests/resilience/test_performance_characteristics.py†L111-L200】【F:tests/resilience/test_pipeline_resilience.py†L25-L120】 |

## 2. Logs y métricas

- Registro de la última ejecución de la suite completa: [`docs/logs/pytest-2024-10-26.md`](logs/pytest-2024-10-26.md). Contiene versión de Python, número total de pruebas y duración agregada.【F:docs/logs/pytest-2024-10-26.md†L1-L18】
- Resultado resumido: 172 pruebas aprobadas en 83.97 s (`pytest`).【f269a5†L1-L4】

## 3. Configuración reproducible

- **Versión del pipeline**: `SERVICE_VERSION = "0.1.0"` se propaga a los metadatos exportados, permitiendo auditar cambios entre despliegues.【F:services/evaluation_service.py†L42-L64】【F:services/evaluation_service.py†L1074-L1178】
- **Parámetros de ejecución**: `ServiceConfig` controla modelo, `run_id`, reintentos, tiempo de espera, proveedor de IA y normalización del divisor de texto. Las pruebas usan `run_id` deterministas para repetir resultados.【F:services/evaluation_service.py†L193-L227】【F:tests/resilience/test_global_consistency.py†L317-L526】
- **Entorno Python**: `requirements.txt` fija versiones mínimas de OCR, LangChain, OpenAI, pandas, numpy y pytest para aislar dependencias de terceros.【F:requirements.txt†L1-L35】
- **Semillas y métricas estables**: la función `_canonical_payload_hash` normaliza marcas de tiempo y métricas de memoria para comparar exportaciones repetidas sin ruido determinista.【F:tests/resilience/test_global_consistency.py†L205-L225】

## 4. Manual de uso

El README incorpora un manual programático que documenta cómo instanciar `EvaluationService`, ejecutar modos `global` y `parcial` con `EvaluationFilters`, fijar `run_id` y elegir el formato de exportación.【F:README.md†L110-L181】

## 5. Flujo de trazabilidad

1. `EvaluationService.run` agrega metadatos de criterios, `run_id`, historial de ejecuciones y métricas de desempeño antes de exportar.【F:services/evaluation_service.py†L972-L1179】
2. `EvaluationRepository.flatten_evaluation` serializa por fila los identificadores de documento, sección, dimensión y pregunta junto con los pesos, puntajes y metadatos de cada chunk.【F:reporting/repository.py†L40-L106】
3. Al revisar un resultado exportado, se puede rastrear cualquier fila hasta la evaluación original usando `document_id`, `run_id` y `timestamp`, permitiendo auditorías completas.【F:reporting/repository.py†L78-L106】【F:services/evaluation_service.py†L1106-L1178】

Con esta evidencia se completa el ciclo de validación: las pruebas cubren el pipeline end-to-end, los registros documentan la ejecución, la configuración fija versiones reproducibles y la trazabilidad conecta entradas, evaluaciones y salidas exportadas.