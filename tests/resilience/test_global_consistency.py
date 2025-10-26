"""Validación integral de consistencia global del pipeline de evaluación."""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import re
import math
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from data.models.document import Document
from services.evaluation_service import EvaluationService, ServiceConfig

from tests.test_pipeline_integration import (
    TRACKING_CLASSES,
    TrackingAIService,
    TrackingCleaner,
    TrackingLoader,
    TrackingPromptBuilder,
    TrackingPromptValidator,
    TrackingRepository,
    TrackingSplitter,
    _prepare_environment,
)

OPTIONAL_METADATA_KEYS = {
    "metadata.skipped",
    "metadata.skip_reason",
    "metadata.segmenter_status",
}

CRITICAL_TRACEABILITY_KEYS = (
    "pipeline_version",
    "criteria_version",
    "model_name",
    "run_id",
    "timestamp",
)

CRITICAL_NUMERIC_COLUMNS = (
    "question_score",
    "section_score",
    "dimension_score",
    "question_weight",
    "section_weight",
    "dimension_weight",
)

REQUIRED_EXPORT_COLUMNS = {
    "document_id",
    "section_id",
    "dimension_id",
    "question_id",
    "question_score",
    "justification",
}

pytest_plugins = ("tests.test_pipeline_integration",)


@pytest.fixture
def consistency_env(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Configura un entorno determinista para validar múltiples escenarios."""

    _prepare_environment(monkeypatch)

    for cls in TRACKING_CLASSES:
        instances = getattr(cls, "instances", None)
        if isinstance(instances, list):
            for instance in list(instances):
                if hasattr(instance, "calls"):
                    getattr(instance, "calls").clear()  # type: ignore[attr-defined]
                if hasattr(instance, "exports"):
                    getattr(instance, "exports").clear()  # type: ignore[attr-defined]
            instances.clear()

    loader = TrackingLoader()
    cleaner = TrackingCleaner()
    splitter = TrackingSplitter()
    repository = TrackingRepository()
    prompt_builder = TrackingPromptBuilder()
    prompt_validator = TrackingPromptValidator()
    ai_service = TrackingAIService()

    service = EvaluationService(
        config=ServiceConfig(
            ai_provider="tracking",
            run_id="consistency-validation",
            model_name="ceplan-mock",
            prompt_batch_size=1,
            retries=0,
            timeout_seconds=None,
            log_level="WARNING",
        ),
        loader=loader,
        cleaner=cleaner,
        splitter=splitter,
        repository=repository,
        ai_service_factory=lambda _: ai_service,
        prompt_builder=prompt_builder,
        prompt_validator=prompt_validator,
    )

    return SimpleNamespace(
        service=service,
        repository=repository,
        loader=loader,
        cleaner=cleaner,
        splitter=splitter,
        prompt_builder=prompt_builder,
        prompt_validator=prompt_validator,
        ai_service=ai_service,
    )


@pytest.fixture
def incomplete_document(tmp_path: Path) -> Path:
    path = tmp_path / "informe_incompleto.txt"
    path.write_text(
        "\n".join(
            [
                "[SECTION:resumen_ejecutivo]",
                "Resumen ejecutivo con hallazgos principales.",
                "Conclusiones y próximos pasos.",
            ]
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def empty_document(tmp_path: Path) -> Path:
    path = tmp_path / "informe_vacio.txt"
    path.write_text("Documento sin estructura reconocible.", encoding="utf-8")
    return path


@pytest.fixture
def corrupted_document(tmp_path: Path) -> Path:
    path = tmp_path / "informe_corrupto.txt"
    payload = (
        b"[SECTION:resumen_ejecutivo]\nContenido previo valido.\n"
        b"\xff\xfe\x00\xff"
        b"[SECTION:gestion_ceplan]\nTexto posterior valido.\n"
    )
    path.write_bytes(payload)
    return path


def _structure_signature(payload: Mapping[str, Any]) -> tuple[Any, ...]:
    rows = payload.get("rows") or []
    aggregated_row_keys: set[str] = set()
    for row in rows:
        aggregated_row_keys.update(str(key) for key in row.keys())
    normalised_row_keys = tuple(
        sorted(key for key in aggregated_row_keys if key not in OPTIONAL_METADATA_KEYS)
    )
    evaluation = payload.get("evaluation") or {}
    metrics = payload.get("metrics") or {}
    extra = payload.get("extra") or {}
    return (
        tuple(sorted(payload.keys())),
        normalised_row_keys,
        tuple(sorted(evaluation.keys())),
        tuple(sorted(metrics.keys())),
        tuple(sorted(extra.keys())),
    )


def _stable_criteria_hash(criteria_payload: Mapping[str, Any]) -> str:
    def _normalise(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                str(key): _normalise(inner)
                for key, inner in sorted(value.items(), key=lambda item: str(item[0]))
            }
        if isinstance(value, list):
            return [_normalise(item) for item in value]
        if isinstance(value, tuple):
            return tuple(_normalise(item) for item in value)
        if isinstance(value, set):
            return sorted(_normalise(item) for item in value)
        return value

    canonical = json.dumps(
        _normalise(criteria_payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _canonical_payload_hash(payload: Mapping[str, Any]) -> str:
    normalised = json.loads(json.dumps(payload, ensure_ascii=False))
    evaluation = normalised.get("evaluation")
    if isinstance(evaluation, Mapping):
        evaluation["generated_at"] = "1970-01-01T00:00:00"
    return hashlib.sha256(
        json.dumps(
            normalised,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _assert_business_rules(
    scenario: str,
    evaluation: Any,
    exported: Mapping[str, Any],
) -> None:
    rows = exported.get("rows") or []
    summary = evaluation.metadata.get("segmenter_summary", {}) if evaluation.metadata else {}
    status_counts = summary.get("status_counts", {}) if isinstance(summary, Mapping) else {}

    if scenario == "completo":
        assert status_counts.get("missing", 0) == 0
        assert any(row.get("question_score") for row in rows)
        for section in evaluation.sections:
            assert section.metadata.get("skipped") is not True
            for dimension in section.dimensions:
                for question in dimension.questions:
                    assert question.score is not None
                    assert question.score >= 0.0
                    assert question.justification
    elif scenario == "incompleto":
        assert status_counts.get("missing", 0) >= 1
        missing_sections = [
            section for section in evaluation.sections if section.metadata.get("skip_reason") == "missing_section"
        ]
        assert missing_sections, "Debe registrarse al menos una sección faltante"
        for section in missing_sections:
            assert section.score == pytest.approx(0.0)
            for dimension in section.dimensions:
                for question in dimension.questions:
                    assert question.score == pytest.approx(0.0)
                    assert question.justification == "Evaluación omitida por falta de sección"
        missing_id = missing_sections[0].section_id
        missing_rows = [row for row in rows if row.get("section_id") == missing_id]
        assert missing_rows, "Las exportaciones deben incluir filas para secciones faltantes"
        assert all(row.get("question_score") == 0 for row in missing_rows)
        assert all(row.get("justification") == "Evaluación omitida por falta de sección" for row in missing_rows)
        assert all(row.get("metadata.skipped") is True for row in missing_rows)
        assert all(row.get("metadata.skip_reason") == "missing_section" for row in missing_rows)

        present_sections = [
            section
            for section in evaluation.sections
            if section.metadata.get("skip_reason") != "missing_section"
        ]
        assert present_sections, "Debe existir al menos una sección evaluada normalmente"
        assert any(
            section.score is not None
            and section.score > 0
            and section.metadata.get("skipped") is not True
            for section in present_sections
        ), "Alguna sección presente debe conservar puntajes positivos"

        present_rows = [row for row in rows if row.get("section_id") != missing_id]
        assert any(
            float(row.get("section_score", 0)) > 0
            and row.get("metadata.skipped") is not True
            for row in present_rows
        ), "Las filas de secciones presentes deben conservar puntajes positivos"
    elif scenario == "corrupto":
        assert status_counts.get("found", 0) >= 1
        assert all(row.get("justification") for row in rows)
    elif scenario == "vacio":
        assert status_counts.get("missing", 0) >= 1 or status_counts.get("empty", 0) >= 1
        assert rows, "Incluso los informes vacíos deben generar filas exportadas"
        for row in rows:
            assert row.get("question_score") == 0
            assert row.get("justification") in {
                "Evaluación omitida por falta de sección",
                "Sin contenido para evaluar",
            }
            assert row.get("metadata.skipped") is True
            assert row.get("metadata.skip_reason") in {"missing_section", "empty_section"}
    else:  # pragma: no cover - se mantiene por seguridad ante nuevos escenarios
        raise AssertionError(f"Escenario no controlado: {scenario}")


def test_global_consistency_validation(
    consistency_env: SimpleNamespace,
    sample_document: Path,
    sample_criteria: Path,
    incomplete_document: Path,
    empty_document: Path,
    corrupted_document: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Ejecuta la validación integral sobre documentos completos e incompletos."""

    original_load = consistency_env.loader.load
    criteria_payload = json.loads(sample_criteria.read_text(encoding="utf-8"))
    expected_methodology = str(
        criteria_payload.get("tipo_informe")
        or criteria_payload.get("metodologia")
        or ""
    ).strip()
    expected_criteria_hash = _stable_criteria_hash(criteria_payload)

    class _FrozenDateTime(datetime):
        @classmethod
        def utcnow(cls) -> datetime:
            return datetime(2024, 1, 1, 0, 0, 0)

    monkeypatch.setattr("services.evaluation_service.datetime", _FrozenDateTime)
    monkeypatch.setattr("data.models.evaluation.datetime", _FrozenDateTime)

    perf_counter_holder = [itertools.count(start=0.0, step=0.05)]
    monkeypatch.setattr(
        "data.models.evaluator.time.perf_counter",
        lambda: next(perf_counter_holder[0]),
    )

    def tolerant_load(
        filepath: str,
        *,
        extract_tables: bool = True,
        extract_images: bool = True,
    ) -> Document:
        try:
            return original_load(filepath, extract_tables=extract_tables, extract_images=extract_images)
        except UnicodeDecodeError:
            raw = Path(filepath).read_text(encoding="utf-8", errors="ignore")
            document = Document(
                content=raw,
                metadata={
                    "source": filepath,
                    "filename": Path(filepath).name,
                    "extension": Path(filepath).suffix or ".txt",
                    "processed_with": "tracking-loader",
                    "pages": [raw],
                    "raw_text": raw,
                },
                pages=[raw],
            )
            consistency_env.loader.calls.append(
                {
                    "path": filepath,
                    "extract_tables": extract_tables,
                    "extract_images": extract_images,
                }
            )
            return document

    monkeypatch.setattr(consistency_env.loader, "load", tolerant_load, raising=False)

    scenarios = {
        "completo": sample_document,
        "incompleto": incomplete_document,
        "corrupto": corrupted_document,
        "vacio": empty_document,
    }

    caplog.set_level(logging.WARNING)

    structures: tuple[Any, ...] | None = None
    rows_count: int | None = None
    scores: dict[str, float] = {}
    baseline_hash: str | None = None

    for name, document_path in scenarios.items():
        initial_logs = len(caplog.records)
        output_path = tmp_path / f"resultado_{name}.json"
        evaluation, metrics = consistency_env.service.run(
            input_path=document_path,
            criteria_path=sample_criteria,
            output_path=output_path,
            output_format="json",
            extra_metadata={"scenario": name},
        )

        payload = json.loads(output_path.read_text(encoding="utf-8"))
        signature = _structure_signature(payload)
        if structures is None:
            structures = signature
        else:
            assert signature == structures, "La estructura exportada debe ser consistente"

        rows = payload.get("rows") or []
        if rows_count is None:
            rows_count = len(rows)
        else:
            assert len(rows) == rows_count, "El número de filas exportadas debe coincidir en todos los escenarios"

        assert metrics.get("methodology") == expected_methodology
        assert metrics.get("criteria_hash") == expected_criteria_hash
        global_metrics = metrics.get("global", {})
        normalized = global_metrics.get("normalized_score")
        assert normalized is not None, f"Se esperaba un puntaje normalizado en el escenario {name}"
        assert not math.isnan(float(normalized)), "Las métricas no deben contener NaN"

        normalized_min = float(global_metrics.get("normalized_min", float("nan")))
        normalized_max = float(global_metrics.get("normalized_max", float("nan")))
        assert math.isclose(normalized_min, 0.0, abs_tol=1e-6), (
            f"El mínimo normalizado debe ser 0 en el escenario {name}, se obtuvo {normalized_min}"
        )
        assert math.isclose(normalized_max, 100.0, abs_tol=1e-6), (
            f"El máximo normalizado debe ser 100 en el escenario {name}, se obtuvo {normalized_max}"
        )
        assert 0.0 <= float(normalized) <= 100.0, (
            f"El puntaje normalizado debe mantenerse en [0, 100] en el escenario {name}"
        )

        score_value = getattr(evaluation, "score", None)
        assert score_value is not None, "La evaluación debe reportar un puntaje global"
        scores[name] = float(score_value)

        exported_metrics = payload.get("metrics") or {}
        assert exported_metrics.get("methodology") == expected_methodology
        assert exported_metrics.get("criteria_hash") == expected_criteria_hash

        extra_payload = payload.get("extra") or {}
        assert isinstance(extra_payload, Mapping)
        assert extra_payload.get("scenario") == name
        assert extra_payload.get("criteria_hash") == expected_criteria_hash

        assert rows, "La exportación debe contener filas evaluadas"
        for row in rows:
            assert REQUIRED_EXPORT_COLUMNS <= set(row), (
                "La exportación debe incluir el conjunto mínimo de columnas obligatorias"
            )
        data_frame = pd.DataFrame(rows)
        missing_trace_keys = [
            key for key in CRITICAL_TRACEABILITY_KEYS if key not in data_frame.columns
        ]
        assert not missing_trace_keys, (
            "Faltan columnas críticas de trazabilidad en la exportación: "
            + ", ".join(missing_trace_keys)
        )
        for row in rows:
            for key in CRITICAL_TRACEABILITY_KEYS:
                assert key in row, f"La fila exportada debe incluir la columna {key}"
                value = row[key]
                assert not pd.isna(value), (
                    f"La columna {key} no debe contener valores nulos en la exportación"
                )
                if isinstance(value, str):
                    assert value.strip(), (
                        f"La columna {key} no debe estar vacía en la exportación"
                    )

        missing_numeric_columns = [
            column for column in CRITICAL_NUMERIC_COLUMNS if column not in data_frame.columns
        ]
        assert not missing_numeric_columns, (
            "Faltan columnas numéricas críticas en la exportación: "
            + ", ".join(missing_numeric_columns)
        )

        numeric_frame = data_frame[list(CRITICAL_NUMERIC_COLUMNS)].apply(
            pd.to_numeric, errors="coerce"
        )
        assert numeric_frame.notna().all().all(), (
            "Las columnas numéricas críticas no deben contener valores nulos"
        )
        assert np.isfinite(numeric_frame.to_numpy()).all(), (
            "Las columnas numéricas críticas deben contener valores finitos"
        )

        _assert_business_rules(name, evaluation, payload)

        if name == "incompleto":
            warning_messages = [
                re.sub(r"Sección\s+'[^']+'", "Sección", record.message)
                for record in caplog.records[initial_logs:]
            ]
            assert any(
                "Sección no encontrada" in message for message in warning_messages
            ), "Debe registrarse una advertencia por sección faltante"

        if name == "completo":
            baseline_hash = _canonical_payload_hash(payload)

    assert scores["completo"] > scores["incompleto"]
    assert scores["incompleto"] >= scores["corrupto"]
    assert scores["corrupto"] >= scores["vacio"]

    assert baseline_hash is not None
    repeat_output = tmp_path / "resultado_completo_repeat.json"
    perf_counter_holder[0] = itertools.count(start=0.0, step=0.05)
    if hasattr(consistency_env.ai_service, "calls"):
        consistency_env.ai_service.calls.clear()
    _, repeat_metrics = consistency_env.service.run(
        input_path=scenarios["completo"],
        criteria_path=sample_criteria,
        output_path=repeat_output,
        output_format="json",
        extra_metadata={"scenario": "completo"},
    )
    repeat_payload = json.loads(repeat_output.read_text(encoding="utf-8"))
    repeat_hash = _canonical_payload_hash(repeat_payload)
    assert repeat_hash == baseline_hash
    assert repeat_metrics.get("methodology") == expected_methodology
    assert repeat_metrics.get("criteria_hash") == expected_criteria_hash

# pytest tests/resilience/test_global_consistency.py -v