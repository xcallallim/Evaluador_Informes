from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from data.models.evaluation import EvaluationResult

try:  # pragma: no cover - optional dependency for advanced exports
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - fallback when pandas is missing
    pd = None


__all__ = ["EvaluationRepository", "flatten_evaluation"]


def flatten_evaluation(evaluation: EvaluationResult) -> List[Dict[str, Any]]:
    """Return a list of rows (one per question) ready to serialise."""

    rows: List[Dict[str, Any]] = []
    for section in evaluation.sections:
        for dimension in section.dimensions:
            for question in dimension.questions:
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
                        "metadata": question.metadata,
                        "chunk_results": [chunk.to_dict() for chunk in question.chunk_results],
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

        output_format = output_format.lower()
        rows = flatten_evaluation(evaluation)
        metadata = {
            "evaluation": evaluation.to_dict(),
            "metrics": dict(metrics_summary),
        }
        if extra_metadata:
            metadata["extra"] = dict(extra_metadata)

        if output_format == "json":
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
        header_df = pd.DataFrame([metrics_summary.get("global", {})])  # type: ignore[arg-type]

        if output_format == "csv":
            df.to_csv(output_path, index=False, encoding=self.encoding)
            return output_path

        if output_format in {"xlsx", "xls", "excel"}:
            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:  # type: ignore[arg-type]
                df.to_excel(writer, sheet_name="preguntas", index=False)
                summary_df.to_excel(writer, sheet_name="resumen", index=False)
                header_df.to_excel(writer, sheet_name="indice_global", index=False)
            return output_path

        raise ValueError(
            f"Formato de salida no soportado: '{output_format}'. Usa json, csv o xlsx."
        )