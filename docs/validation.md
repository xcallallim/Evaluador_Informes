# Sistema de validación de criterios

El proyecto incluye un *router* de validadores en `utils/validators/` que
descubre automáticamente los módulos disponibles y los registra por
`tipo_informe`. Cada módulo de validador invoca a
`register_validator(tipo, funcion)` para inscribirse en el registro común,
permitiendo que `utils/criteria_validator.py` ejecute la verificación
específica según el JSON procesado.

## Arquitectura del registro

- `utils/validators/__init__.py` usa *autodiscovery* para cargar todos los
  módulos `.py` del paquete y poblar el diccionario `VALIDATORS`.
- Cada módulo expone una función `validate(payload: dict) -> ValidationResult`
  que encapsula las reglas de negocio del tipo de informe correspondiente.
- `utils/validators/base.py` ofrece utilidades compartidas, incluido el objeto
  `ValidationResult` (que agrega `errors` y `warnings`) y helpers como
  `almost_equal` o `sum_ponderaciones`.
- Los validadores específicos (`institucional.py`, `politica_nacional.py`, ...)
  encapsulan tanto reglas estructurales (campos obligatorios, formato de fechas)
  como reglas numéricas (ponderaciones que suman 1, escalas válidas, etc.).

Este diseño mantiene las reglas desacopladas del CLI y permite reutilizarlas
directamente desde otros módulos del pipeline (por ejemplo, antes de ejecutar la
evaluación real).

### Ejemplo básico

```python
from utils.validators import register_validator, ValidationResult


def validate_demo(payload: dict) -> ValidationResult:
    result = ValidationResult()
    if "criterios" not in payload:
        result.errors.append("Falta la lista de criterios")
    return result


register_validator("demo", validate_demo)
```

Al importarse el módulo, el validador queda disponible y puede invocarse desde
`utils.criteria_validator` o directamente mediante
`from utils.validators import VALIDATORS`.

## Uso desde consola

```bash
python -m utils.criteria_validator "data/criteria/metodología_institucional.json" \
    data/criteria/metodologia_politica_nacional.json
```

El comando imprime los resultados individuales y retorna un código de salida
`1` si se detectan errores de esquema.

## Crear un nuevo validador

1. Crea un archivo en `utils/validators/` con el nombre del tipo de informe.
2. Implementa una función `validate(datos: dict) -> ValidationResult` usando las
   utilidades de `base.py` para cálculos numéricos o composición de resultados.
3. Llama a `register_validator("mi_tipo", validate)` al final del módulo.
4. Añade casos de prueba en `tests/test_criteria_validator.py` cubriendo tanto
   un JSON válido como escenarios erróneos.

Los validadores pueden apoyarse en datos auxiliares (por ejemplo catálogos o
tablas de equivalencias) siempre que permanezcan inmutables y serializables para
facilitar su uso en CI/CD.

## Pruebas automáticas

El archivo `tests/test_criteria_validator.py` cubre la validación de ambos
metodologías, el manejo de tipos no registrados y la ejecución del CLI.
Ejecuta `pytest` para validar el flujo completo.

## Validación interactiva de prompts

Complementa la validación de criterios con
`utils.prompt_validator.validate_prompt()`, que analiza la calidad metodológica
y semántica de los prompts construidos a partir de los criterios. Puedes usarlo
desde un REPL:

```python
from utils.prompt_validator import validate_prompt

resultado = validate_prompt(open("prompt.txt", "r", encoding="utf-8").read())
print(resultado.quality_score, resultado.alerts)
```

El helper devuelve un `PromptValidationResult` con `is_valid`, `quality_score`,
metadatos y alertas, lo que permite integrarlo en pipelines de QA automáticos.

### Limpieza y segmentación

Los antiguos scripts manuales de verificación se transformaron en pruebas
unitarias que usan documentos sintéticos. Para ejecutar únicamente este
conjunto:

```bash
pytest tests/test_cleaner.py tests/test_segmenter.py tests/test_section_loader.py
```

El comando anterior se apoya en el `pytest.ini` del proyecto, por lo que no es
necesario manipular el `PYTHONPATH` ni descargar PDFs de ejemplo.