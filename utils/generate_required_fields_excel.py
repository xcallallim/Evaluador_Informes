"""Herramienta para generar un Excel con los campos requeridos por la plataforma.

El archivo resultante documenta tres conjuntos de campos:
- Atributos de ``data.models.document.Document``
- Claves obligatorias del diccionario ``Document.metadata``
- Columnas mínimas esperadas en las exportaciones de resultados
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd


@dataclass(frozen=True)
class FieldSpec:
    """Descripción de un campo requerido."""

    categoria: str
    campo: str
    tipo: str
    descripcion: str
    notas: str = ""

    def as_dict(self) -> dict[str, str]:
        """Convierte la especificación en un diccionario apto para DataFrame."""

        payload = {
            "Categoría": self.categoria,
            "Campo": self.campo,
            "Tipo": self.tipo,
            "Descripción": self.descripcion,
        }
        if self.notas:
            payload["Notas"] = self.notas
        return payload


DOCUMENT_FIELDS: Sequence[FieldSpec] = (
    FieldSpec(
        categoria="Documento",
        campo="content",
        tipo="str",
        descripcion="Texto plano completo del informe, incluyendo separadores de página.",
    ),
    FieldSpec(
        categoria="Documento",
        campo="metadata",
        tipo="Dict[str, Any]",
        descripcion="Metadatos serializables asociados al documento (ver pestaña específica).",
    ),
    FieldSpec(
        categoria="Documento",
        campo="pages",
        tipo="List[str]",
        descripcion="Páginas individuales sin los marcadores `=== PAGE n ===`.",
    ),
    FieldSpec(
        categoria="Documento",
        campo="tables",
        tipo="List[Any]",
        descripcion="Tablas extraídas por el loader; su estructura depende del origen.",
    ),
    FieldSpec(
        categoria="Documento",
        campo="images",
        tipo="List[Any]",
        descripcion="Referencias a imágenes recuperadas durante la ingestión (opcional).",
    ),
    FieldSpec(
        categoria="Documento",
        campo="sections",
        tipo="Dict[str, str]",
        descripcion="Secciones semánticas de alto nivel calculadas por el segmentador.",
    ),
    FieldSpec(
        categoria="Documento",
        campo="chunks",
        tipo="List[langchain_core.documents.Document]",
        descripcion="Fragmentos listos para evaluación/embeddings generados por el splitter.",
        notas="El servicio puede consumirlos de manera perezosa mediante iteradores.",
    ),
)

METADATA_FIELDS: Sequence[FieldSpec] = (
    FieldSpec(
        categoria="Metadata",
        campo="source",
        tipo="str",
        descripcion="Ruta absoluta del archivo ingerido (referencia principal del documento).",
    ),
    FieldSpec(
        categoria="Metadata",
        campo="filename",
        tipo="str",
        descripcion="Nombre del archivo derivado de `source`.",
    ),
    FieldSpec(
        categoria="Metadata",
        campo="extension",
        tipo="str",
        descripcion="Extensión en minúsculas con punto inicial (p. ej. `.pdf`).",
    ),
    FieldSpec(
        categoria="Metadata",
        campo="processed_with",
        tipo="str",
        descripcion="Identifica la canalización de ingestión utilizada (actualmente `DocumentLoader`).",
    ),
    FieldSpec(
        categoria="Metadata",
        campo="pages",
        tipo="List[str]",
        descripcion="Copia directa de `Document.pages` para facilitar la serialización.",
    ),
    FieldSpec(
        categoria="Metadata",
        campo="tables",
        tipo="Dict[str, Any]",
        descripcion="Mapeo `namespace → payload` con las tablas detectadas.",
    ),
    FieldSpec(
        categoria="Metadata",
        campo="is_ocr",
        tipo="bool",
        descripcion="Indica si se aplicó OCR durante la extracción (por defecto `False`).",
    ),
    FieldSpec(
        categoria="Metadata",
        campo="extraction_method",
        tipo="str",
        descripcion="Método empleado para extraer texto (`ocr`, `embedded_text`, `docx`, `text`).",
    ),
    FieldSpec(
        categoria="Metadata",
        campo="raw_text",
        tipo="str",
        descripcion="Texto bruto antes del cleaning. Disponible en pipelines con OCR.",
        notas="Opcional; útil para depuración y control de calidad.",
    ),
    FieldSpec(
        categoria="Metadata",
        campo="images",
        tipo="List[Any]",
        descripcion="Referencias a recursos de imagen cuando la extracción está habilitada.",
        notas="Opcional.",
    ),
    FieldSpec(
        categoria="Metadata",
        campo="issues",
        tipo="List[str]",
        descripcion="Advertencias deduplicadas generadas por el loader.",
        notas="Opcional.",
    ),
    FieldSpec(
        categoria="Metadata",
        campo="language",
        tipo="str",
        descripcion="Código ISO del idioma detectado para el documento.",
        notas="Opcional.",
    ),
    FieldSpec(
        categoria="Metadata",
        campo="detected_encoding",
        tipo="str",
        descripcion="Codificación original reportada por el loader de texto.",
        notas="Opcional.",
    ),
    FieldSpec(
        categoria="Metadata",
        campo="source_hint",
        tipo="str",
        descripcion="Pista adicional del origen entregada por el loader (por ejemplo `pdf`).",
        notas="Opcional.",
    ),
    FieldSpec(
        categoria="Metadata",
        campo="document_id",
        tipo="str",
        descripcion="Identificador único del documento propagado a resultados y reportes.",
        notas="Se respeta si llega en origen; de lo contrario lo genera `EvaluationService`.",
    ),
    FieldSpec(
        categoria="Metadata",
        campo="document_type",
        tipo="str",
        descripcion="Tipo de informe asociado al documento (`politica_nacional`, etc.).",
        notas="Se asigna durante la evaluación.",
    ),
)

EXPORT_COLUMNS: Sequence[FieldSpec] = (
    FieldSpec(
        categoria="Exportación",
        campo="document_id",
        tipo="str",
        descripcion="Identificador único del documento evaluado.",
    ),
    FieldSpec(
        categoria="Exportación",
        campo="section_id",
        tipo="str",
        descripcion="Identificador de la sección o capítulo evaluado.",
        notas="Puede quedar vacío en preguntas manuales (prefijo `MANUAL_`).",
    ),
    FieldSpec(
        categoria="Exportación",
        campo="section_title",
        tipo="str",
        descripcion="Nombre amigable de la sección.",
    ),
    FieldSpec(
        categoria="Exportación",
        campo="section_score",
        tipo="float",
        descripcion="Puntaje (0 a 4) asignado a la sección agregada.",
    ),
    FieldSpec(
        categoria="Exportación",
        campo="section_weight",
        tipo="float",
        descripcion="Peso relativo (0 a 1) de la sección en el índice global.",
    ),
    FieldSpec(
        categoria="Exportación",
        campo="dimension_name",
        tipo="str",
        descripcion="Nombre de la dimensión evaluada dentro de la sección.",
    ),
    FieldSpec(
        categoria="Exportación",
        campo="dimension_score",
        tipo="float",
        descripcion="Puntaje (0 a 4) de la dimensión.",
    ),
    FieldSpec(
        categoria="Exportación",
        campo="dimension_weight",
        tipo="float",
        descripcion="Peso relativo (0 a 1) de la dimensión.",
    ),
    FieldSpec(
        categoria="Exportación",
        campo="question_id",
        tipo="str",
        descripcion="Identificador estable de la pregunta aplicada.",
    ),
    FieldSpec(
        categoria="Exportación",
        campo="question_text",
        tipo="str",
        descripcion="Redacción completa de la pregunta evaluada.",
    ),
    FieldSpec(
        categoria="Exportación",
        campo="question_score",
        tipo="float",
        descripcion="Puntaje (0 a 4) asignado a la respuesta de la pregunta.",
        notas="Puede ser nulo cuando la evaluación se omite justificadamente.",
    ),
    FieldSpec(
        categoria="Exportación",
        campo="question_weight",
        tipo="float",
        descripcion="Peso relativo (0 a 1) de la pregunta dentro de su dimensión.",
    ),
    FieldSpec(
        categoria="Exportación",
        campo="justification",
        tipo="str",
        descripcion="Argumento generado por el modelo o por el analista.",
    ),
    FieldSpec(
        categoria="Exportación",
        campo="criteria_version",
        tipo="str",
        descripcion="Versión del archivo de criterios utilizado.",
    ),
    FieldSpec(
        categoria="Exportación",
        campo="tipo_informe",
        tipo="str",
        descripcion="Tipo de informe evaluado (politica_nacional, institucional, etc.).",
    ),
    FieldSpec(
        categoria="Exportación",
        campo="model_name",
        tipo="str",
        descripcion="Identificador del modelo de IA utilizado en la ejecución.",
    ),
    FieldSpec(
        categoria="Exportación",
        campo="pipeline_version",
        tipo="str",
        descripcion="Versión de la canalización de evaluación.",
    ),
    FieldSpec(
        categoria="Exportación",
        campo="timestamp",
        tipo="str",
        descripcion="Fecha y hora ISO-8601 de generación del resultado.",
    ),
)


def _to_dataframe(fields: Iterable[FieldSpec]) -> pd.DataFrame:
    """Crea un DataFrame con columnas amigables para Excel."""

    records: List[dict[str, str]] = [field.as_dict() for field in fields]
    return pd.DataFrame(records)


def generate_excel(output_dir: Path, filename: str = "campos_requeridos.xlsx") -> Path:
    """Genera el Excel con pestañas por categoría y devuelve la ruta final."""

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        _to_dataframe(DOCUMENT_FIELDS).to_excel(writer, sheet_name="documento", index=False)
        _to_dataframe(METADATA_FIELDS).to_excel(writer, sheet_name="metadata", index=False)
        _to_dataframe(EXPORT_COLUMNS).to_excel(writer, sheet_name="exportacion", index=False)

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera un Excel con los campos necesarios para ingestión y exportes."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "campos",
        help="Directorio donde se guardará el archivo (se crea si no existe).",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="campos_requeridos.xlsx",
        help="Nombre del archivo Excel generado.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = generate_excel(args.output_dir, args.filename)
    print(f"Archivo generado en: {output_path}")


if __name__ == "__main__":
    main()