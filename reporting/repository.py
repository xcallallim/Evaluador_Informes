"""Persistence helpers for exporting evaluation results to tabular formats."""

from __future__ import annotations

import json
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
    criteria_version = None
    if isinstance(evaluation.metadata, Mapping):
        criteria_version = evaluation.metadata.get("criteria_version")
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
        output_format: str = "json",
        extra_metadata: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        output_path = Path(output_path)
        if not output_path.parent.exists():
            raise FileNotFoundError(
                f"La carpeta destino '{output_path.parent}' no existe."
            )

        output_format = (output_format or "json").lower()
        normalised_format = _normalise_format(output_format)
        _validate_format(normalised_format)
        rows = flatten_evaluation(evaluation)
        metadata = {
            "evaluation": evaluation.to_dict(),
            "metrics": dict(metrics_summary),
        }
        if extra_metadata:
            metadata["extra"] = dict(extra_metadata)

        if normalised_format == "json":
            output_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return output_path

        if pd is None:
            raise RuntimeError(
                "Exportar a formatos tabulares requiere pandas instalado."
            )

        df = pd.DataFrame(rows)  # type: ignore[arg-type]
        summary_df = pd.DataFrame(metrics_summary.get("sections", []))  # type: ignore[arg-type]
        evaluation_metadata: Mapping[str, Any] = (
            evaluation.metadata if isinstance(evaluation.metadata, Mapping) else {}
        )
        header_df = pd.DataFrame(
            [
                {
                    **metrics_summary.get("global", {}),
                    "run_id": evaluation_metadata.get("run_id"),
                    "model_name": evaluation_metadata.get("model_name"),
                    "tipo_informe": evaluation_metadata.get("tipo_informe", evaluation.document_type),
                    "criteria_version": evaluation_metadata.get("criteria_version"),
                }
            ]
        )  # type: ignore[arg-type]

        if normalised_format == "csv":
            df.to_csv(output_path, index=False, encoding=self.encoding)
            return output_path

        if normalised_format == "parquet":
            df.to_parquet(output_path, index=False)
            return output_path

        if normalised_format == "xlsx":
            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:  # type: ignore[arg-type]
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
            return output_path

        raise ValueError(
            f"Formato de salida no soportado: '{output_format}'. Usa json, csv, xlsx o parquet."
        )


def _normalise_format(fmt: str) -> str:
    mapping = {
        "xls": "xlsx",
        "excel": "xlsx",
    }
    return mapping.get(fmt, fmt)


def _validate_format(fmt: str) -> None:
    supported = {"json", "csv", "xlsx", "parquet"}
    if fmt not in supported:
        raise ValueError(
            f"Formato de salida no soportado: '{fmt}'. Usa uno de: {', '.join(sorted(supported))}."
        )


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