"""Utilidades para serializar resultados de evaluación a distintos formatos."""

from __future__ import annotations

import json
import re   
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set

from data.models.evaluation import EvaluationResult

try:  # pragma: no cover - optional dependency for advanced exports
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - fallback when pandas is missing
    pd = None


__all__ = ["EvaluationRepository", "flatten_evaluation"]


def flatten_evaluation(evaluation: EvaluationResult) -> List[Dict[str, Any]]:
    """Return a list of rows (one per question) ready to serialise."""

    rows: List[Dict[str, Any]] = []
    evaluation_metadata: Mapping[str, Any] = (
        evaluation.metadata if isinstance(evaluation.metadata, Mapping) else {}
    )
    criteria_version = evaluation_metadata.get("criteria_version")
    tipo_informe = evaluation_metadata.get(
        "tipo_informe", evaluation.document_type
    )
    model_name = evaluation_metadata.get("model_name")
    pipeline_version = evaluation_metadata.get("pipeline_version")
    timestamp = evaluation_metadata.get("timestamp")
    for section in evaluation.sections:
        for dimension in section.dimensions:
            for question in dimension.questions:
                metadata_columns = {}
                if isinstance(question.metadata, Mapping):
                    metadata_columns = _flatten_mapping(question.metadata, prefix="metadata")
                rows.append(
                    {
                        "document_id": evaluation.document_id,
                        "section_id": section.section_id,
                        "section_title": section.title,
                        "section_score": section.score,
                        "section_weight": section.weight,
                        "dimension_name": dimension.name,
                        "dimension_score": dimension.score,
                        "dimension_weight": dimension.weight,
                        "question_id": question.question_id,
                        "question_text": question.text,
                        "question_score": question.score,
                        "question_weight": question.weight,
                        "justification": question.justification,
                        "relevant_text": question.relevant_text,
                        "chunk_results": [chunk.to_dict() for chunk in question.chunk_results],
                        "criteria_version": criteria_version,
                        "tipo_informe": tipo_informe,
                        "model_name": model_name,
                        "pipeline_version": pipeline_version,
                        "timestamp": timestamp,
                        **metadata_columns,
                    }
                )
    return rows


@dataclass(slots=True)
class EvaluationRepository:
    """Export :class:`EvaluationResult` instances to disk."""

    encoding: str = "utf-8-sig"

    def export(
        self,
        evaluation: EvaluationResult,
        metrics_summary: Mapping[str, Any],
        *,
        output_path: Path,
        output_format: str = "xlsx",
        extra_metadata: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        output_path = Path(output_path)
        output_dir = output_path.parent
        if not output_dir.exists():
            # Garantiza que los formatos solicitados se puedan exportar aun cuando el
            # directorio destino no exista. Esto facilita los usos desde scripts o
            # pruebas que escriben directamente en ``data/example`` u otras carpetas
            # efímeras sin necesidad de crearlas previamente.
            output_dir.mkdir(parents=True, exist_ok=True)

        requested_format = (output_format or "xlsx").lower()
        normalised_format = _normalise_format(requested_format)
        allowed_formats = {"", "xlsx", "json", "csv"}
        if normalised_format not in allowed_formats:
            raise ValueError(
                "Formato de salida no soportado: usa xlsx, csv o json como formato solicitado."
            )

        if pd is None:
            raise RuntimeError(
                "Exportar evaluaciones requiere pandas instalado."
            )
        
        base_name = output_path.stem if output_path.suffix else output_path.name
        base_path = output_path.parent / base_name
        excel_path = base_path.with_suffix(".xlsx")
        csv_path = base_path.with_suffix(".csv")
        json_path = base_path.with_suffix(".json")

        rows = flatten_evaluation(evaluation)
        df = pd.DataFrame(rows)  # type: ignore[arg-type]
        summary_df = pd.DataFrame(metrics_summary.get("sections", []))  # type: ignore[arg-type]
        evaluation_metadata: Mapping[str, Any] = (
            evaluation.metadata if isinstance(evaluation.metadata, Mapping) else {}
        )
        header_payload: Dict[str, Any] = {
            **metrics_summary.get("global", {}),
            "run_id": evaluation_metadata.get("run_id"),
            "model_name": evaluation_metadata.get("model_name"),
            "tipo_informe": evaluation_metadata.get(
                "tipo_informe", evaluation.document_type
            ),
            "pipeline_version": evaluation_metadata.get("pipeline_version"),
            "criteria_version": evaluation_metadata.get("criteria_version"),
            "timestamp": evaluation_metadata.get("timestamp"),
        }
        if extra_metadata:
            header_payload.update({f"extra_{k}": v for k, v in dict(extra_metadata).items()})
        header_df = pd.DataFrame([header_payload])  # type: ignore[arg-type]

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:  # type: ignore[arg-type]
            used_names: Set[str] = set()
            df.to_excel(
                writer,
                sheet_name=_safe_sheet_name("preguntas", used_names),
                index=False,
            )
            summary_df.to_excel(
                writer,
                sheet_name=_safe_sheet_name("resumen", used_names),
                index=False,
            )
            header_df.to_excel(
                writer,
                sheet_name=_safe_sheet_name("indice_global", used_names),
                index=False,
            )

        df.to_csv(csv_path, index=False, encoding=self.encoding)

        json_payload: Dict[str, Any] = {
            "rows": rows,
            "evaluation": evaluation.to_dict(),
            "metrics": dict(metrics_summary),
        }
        if extra_metadata:
            json_payload["extra"] = dict(extra_metadata)

        json_path.write_text(
            json.dumps(json_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return excel_path


def _normalise_format(fmt: str) -> str:
    mapping = {
        "xls": "xlsx",
        "excel": "xlsx",
    }
    return mapping.get(fmt, fmt)


def _safe_sheet_name(name: str, existing: Set[str]) -> str:
    """Return an Excel-compatible sheet name ensuring uniqueness."""

    sanitized = re.sub(r"[:\\/?*\[\]]", "_", name).strip()
    if not sanitized:
        sanitized = "sheet"
    sanitized = sanitized[:31]
    candidate = sanitized
    index = 1
    used_lower = {value.lower() for value in existing}
    while candidate.lower() in used_lower:
        suffix = f"_{index}"
        base_length = max(0, 31 - len(suffix))
        candidate = f"{sanitized[:base_length]}{suffix}" if base_length else suffix[-31:]
        index += 1
    existing.add(candidate)
    return candidate


def _flatten_mapping(mapping: Mapping[str, Any], *, prefix: str) -> Dict[str, Any]:
    """Flatten a nested mapping into dotted keys prefixed with ``prefix``."""

    flattened: Dict[str, Any] = {}

    def _serialise_value(value: Any) -> Any:
        if isinstance(value, (list, tuple, set)):
            return json.dumps(list(value), ensure_ascii=False)
        return value

    def _visit(items: Mapping[str, Any], parents: Iterable[str]) -> None:
        for key, value in items.items():
            path = [*parents, str(key)]
            if isinstance(value, Mapping):
                _visit(value, path)
            else:
                dotted_key = ".".join([prefix, *path]) if prefix else ".".join(path)
                flattened[dotted_key] = _serialise_value(value)

    _visit(mapping, [])
    return flattened