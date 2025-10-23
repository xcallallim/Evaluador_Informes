# Contrato de Datos del Documento

Este documento describe la carga útil generada por la canalización de ingesta (`data.preprocessing.loader.DocumentLoader`) y consumida por las etapas posteriores (Limpiador, Segmentador, Divisor y Evaluador). El objetivo es explicitar las expectativas implícitas para que futuras refactorizaciones no interrumpan silenciosamente las integraciones entre capas.

## `data.models.document.Document`
«Document» es el portador canónico de fuentes textuales. Es una clase de datos cuyos campos se mantienen deliberadamente genéricos para que puedan reutilizarse en diferentes formatos. La fachada del cargador siempre normaliza las instancias antes de entregarlas a otros módulos.

Atributo: `content`
Tipo: `str`
Productor: Cargadores TXT/DOCX/PDF
Consumidores: Cleaner, Segmenter, Splitter
Notas: Representación en texto plano de todo el documento. Los cargadores PDF y DOCX incluyen marcas `=== PAGE n ===` para preservar la paginación.

Atributo: `metadata`
Tipo: `Dict[str, Any]`
Productor: `DocumentLoader`
Consumidores: Cleaner, Segmenter, Splitter, Evaluator, herramientas de reporte
Notas: Ver tabla dedicada más abajo. Siempre es una copia superficial (shallow copy) para evitar filtrar detalles internos del loader.

Atributo: `pages`
Tipo: `List[str]`
Productor: Loaders especializados
Consumidores: Cleaner, herramientas de reporte
Notas: Lista de páginas individuales sin los marcadores `=== PAGE`. Cleaner usa esta lista para detectar encabezados/pies de página repetidos.

Atributo: `tables`
Tipo: `List[Any]`
Productor: Loaders especializados
Consumidores: Evaluator, exportadores
Notas: Lista agregada de tablas extraídas. La estructura depende del loader (PDF produce rutas a archivos; DOCX devuelve matrices de celdas).

Atributo: `images`
Tipo: `List[Any]`
Productor: PDF loader
Consumidores: Reportes, inspección manual
Notas: Solo se llena cuando la extracción de imágenes está habilitada en `DocumentLoader`. Los valores son referencias estables devueltas por el `PDFResourceExporter` configurado.

Atributo: `sections`
Tipo: `Dict[str, str]`
Productor: Segmenter/Evaluator
Consumidores: Evaluator, herramientas de reporte
Notas: Mapeo opcional para persistir secciones semánticas de alto nivel (por ejemplo: `resumen_ejecutivo`).

Atributo: `chunks`
Tipo: `List[langchain_core.documents.Document]`
Productor: Splitter
Consumidores: Evaluator
Notas: Documentos LangChain listos para embedding o recuperación.

## Contrato de metadatos
A menos que se indique lo contrario, `DocumentLoader` garantiza la presencia de las siguientes claves dentro de `Document.metadata`. Todos los valores son serializables en JSON, por lo que el objeto puede persistir o transportarse por la red sin procesamiento adicional.

Clave: `source`
Tipo: `str`
Productor: `DocumentLoader`
Presencia: Siempre
Consumidores: Logging, exports
Descripción: Ruta absoluta del archivo ingerido. Actúa como referencia principal del documento.

Clave: `filename`
Tipo: `str`
Productor: `DocumentLoader`
Presencia: Siempre
Consumidores: Reporting, UI
Descripción: Nombre del archivo extraído de `source`.

Clave: `extension`
Tipo: `str`
Productor: `DocumentLoader`
Presencia: Siempre
Consumidores: Cleaner, heurísticas
Descripción: Extensión en minúscula incluyendo punto inicial (por ejemplo `.pdf`).

Clave: `processed_with`
Tipo: `str`
Productor: `DocumentLoader`
Presencia: Siempre
Consumidores: Diagnósticos
Descripción: Valor actual `"DocumentLoader"`. Útil cuando coexisten múltiples pipelines de ingestión.

Clave: `pages`
Tipo: `List[str]`
Productor: `DocumentLoader`
Presencia: Siempre (puede estar vacío)
Consumidores: Cleaner, reporting
Descripción: Copia directa de `Document.pages`. Se preserva para simplificar serialización y ejecución sin estado.

Clave: `tables`
Tipo: `Dict[str, Any]`
Productor: `DocumentLoader`
Presencia: Siempre (puede estar vacío)
Consumidores: Evaluator, exportadores
Descripción: Mapeo de `namespace → payload`. El namespace depende del loader (`pdf`, `docx`, `txt`). La estructura del payload depende del origen.

Clave: `is_ocr`
Tipo: `bool`
Productor: Loaders especializados vía `extra_metadata`
Presencia: Siempre (por defecto `False`)
Consumidores: Cleaner
Descripción: Bandera para habilitar heurísticas propias del OCR. PDFs escaneados la establecen en `True`.

Clave: `extraction_method`
Tipo: `str`
Productor: Loaders especializados vía `extra_metadata`
Presencia: Siempre
Consumidores: Cleaner, auditorías
Descripción: Estrategia usada para obtener texto. Valores actuales: `ocr`, `embedded_text`, `docx`, `text`.

Clave: `raw_text`
Tipo: `str`
Productor: Loader PDF vía `extra_metadata`
Presencia: Opcional
Consumidores: Debugging, QA
Descripción: Texto bruto generado antes del cleaning. En OCR contiene texto reconocido; en PDFs digitales puede duplicar `content`.

Clave: `images`
Tipo: `List[Any]`
Productor: Loader PDF vía `extra_metadata`
Presencia: Opcional
Consumidores: Reporting
Descripción: Solo presente si se activó extracción de imágenes. Generalmente contiene rutas o referencias de archivo.

Clave: `issues`
Tipo: `List[str]`
Productor: Loaders especializados vía `LoaderContext`
Presencia: Opcional
Consumidores: Logging, UI
Descripción: Advertencias deduplicadas de ingestión (tablas faltantes, fallback OCR, etc.).

Clave: `language`
Tipo: `str`
Productor: Loaders especializados vía `extra_metadata`
Presencia: Opcional
Consumidores: Evaluator, UI
Descripción: Código ISO del idioma detectado.

Clave: `detected_encoding`
Tipo: `str`
Productor: Loader TXT vía `extra_metadata`
Presencia: Opcional
Consumidores: Diagnósticos
Descripción: Codificación original reportada por el loader.

Clave: `source_hint`
Tipo: `str`
Productor: Loader PDF (passthrough en `Document.metadata`)
Presencia: Opcional
Consumidores: Heurísticas posteriores
Descripción: Pista adicional del origen del documento (por ejemplo `pdf`).

Las demás claves específicas del cargador se conservan textualmente mientras sean serializables. Esto permite nuevas funciones (p. ej., puntuaciones de confianza, metadatos geográficos) sin actualizar cada consumidor. El único nombre de clave reservado es `loader_context`, que es un área de almacenamiento interno que se elimina mediante `compose_document()` antes de que el documento abandone la fachada del cargador.

## Responsabilidades del productor

* Los cargadores especializados **deben** inyectar sus campos adicionales a través de `LoaderContext.extra_metadata`. Las escrituras directas en `Document.metadata` corren el riesgo de filtrar la estructura privada `loader_context`.
* Los cargadores son responsables de devolver `is_ocr` y `extraction_method` con valores significativos. Las rutas con OCR habilitado deben configurar `raw_text` cuando esté disponible para que las herramientas de control de calidad puedan inspeccionar el resultado del reconocimiento.
* Cuando la extracción de imágenes está deshabilitada, los cargadores de PDF deben seguir informando de los metadatos de la imagen dentro del contexto para que el usuario pueda entender que los recursos se omitieron intencionalmente.

## Expectativas del consumidor

* El limpiador se basa en `is_ocr` para seleccionar umbrales de coincidencia aproximada y en `pages`/`raw_text` al reconstruir marcadores de página.
* El segmentador y el divisor propagan `Document.metadata` en los metadatos del fragmento para que los evaluadores puedan rastrear cada fragmento hasta el documento original.
* El evaluador busca `language`, `document_metadata.id` (si se proporcionó en etapas anteriores) y cualquier clave específica del negocio para adaptar las reglas de puntuación.

Mantener estas suposiciones documentadas garantiza la seguridad de las futuras refactorizaciones y que los nuevos colaboradores puedan ampliar el proceso de ingesta sin aplicar ingeniería inversa a los contratos implícitos.