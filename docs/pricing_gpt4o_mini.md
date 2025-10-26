# Estimación de costo por llamada usando GPT-4o-mini

Esta guía describe cómo estimar el costo por llamada al modelo **OpenAI GPT-4o-mini** cuando se evalúan informes con fragmentos (*chunks*) de 1 500 caracteres y un solape del 10 %.

## Tarifas de referencia

 Según la estructura de precios pública de OpenAI (actualizada a abril de 2024), GPT-4o-mini tiene las siguientes tarifas:

- **Entrada**: USD 0.000150 por cada 1 000 tokens (equivale a **USD 0.15 por cada 1 000 000 tokens**).
- **Salida**: USD 0.000600 por cada 1 000 tokens (equivale a **USD 0.60 por cada 1 000 000 tokens**).

> Ajusta estos valores si OpenAI modifica la tabla de precios.

## Conversión aproximada de caracteres a tokens

- 1 token ≈ 4 caracteres en texto en español/español formal.
- Un fragmento de 1 500 caracteres equivale a ~375 tokens.
- El encabezado de instrucciones que acompaña cada prompt de evaluación añade ~150 tokens.

Por lo tanto, la **entrada total** ronda los **525 tokens** (375 del fragmento + 150 del prompt base). Sustituye este supuesto por tus propias métricas si cuentas con un tokenizador distinto.

## Salida esperada

En evaluaciones operativas observamos que las justificaciones generadas por GPT-4o-mini para cada pregunta oscilan entre **120 y 180 tokens**. Para una estimación conservadora usa 180 tokens de salida por llamada.

## Fórmula de costo por llamada

```
costo ≈ (tokens_entrada / 1000) × 0.000150 + (tokens_salida / 1000) × 0.000600
```

Sustituyendo los valores aproximados:

- Entrada: 525 tokens → 0.525 × 0.000150 = **USD 0.00007875**
- Salida: 180 tokens → 0.180 × 0.000600 = **USD 0.00010800**

**Costo total estimado por llamada**: **USD 0.00018675**

## Sensibilidad al tamaño del fragmento

Si amplías el `chunk_size` a 2 500 caracteres (≈625 tokens) manteniendo el mismo encabezado y salida, el costo por llamada sube a ~USD 0.000211, pero reduces el número de fragmentos y, por ende, el número total de llamadas. Debes balancear ambos efectos según tu presupuesto.

## Recomendaciones

1. **Monitorea los tokens reales**: Registra tokens entrada/salida que devuelve la API para recalibrar estas estimaciones.
2. **Reduce llamadas innecesarias**: Ajusta filtros de preguntas y validadores para evitar prompts con baja señal.
3. **Evalúa costos indirectos**: Si incorporas embeddings u otras etapas previas, suma sus costos para obtener el total por informe.

Con estos supuestos, ejecutar 10 000 llamadas equivaldría a ~USD 1.867. Ajusta el cálculo multiplicando el costo unitario por el número de prompts efectivos en tu corrida.