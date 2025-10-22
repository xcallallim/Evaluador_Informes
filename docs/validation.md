# Sistema de validación de criterios

El proyecto incluye un *router* de validadores en `utils/validators/` que
descubre automáticamente los módulos disponibles y los registra por
`tipo_informe`. Cada módulo de validador invoca a
`register_validator(tipo, funcion)` para inscribirse en el registro común,
permitiendo que `utils/criteria_validator.py` ejecute la verificación
específica según el JSON procesado.

## Uso desde consola

```bash
python -m utils.criteria_validator "data/criteria/metodología_institucional.json" \
    data/criteria/metodologia_politica_nacional.json
```

El comando imprime los resultados individuales y retorna un código de salida
`1` si se detectan errores de esquema.

## Pruebas automáticas

El archivo `tests/test_criteria_validator.py` cubre la validación de ambos
metodologías, el manejo de tipos no registrados y la ejecución del CLI.
Ejecuta `pytest` para validar el flujo completo.