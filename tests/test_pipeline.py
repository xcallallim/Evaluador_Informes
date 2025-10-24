"""Integration test exercising the full evaluation pipeline."""

from __future__ import annotations

import codecs
import json
import logging
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Any, Dict, List

import pytest

from data.models.document import Document
import services.evaluation_service as evaluation_module
from services.evaluation_service import (
    PROMPT_QUALITY_THRESHOLD,
    EvaluationService,
    MockAIService,
    ServiceConfig,
    ValidatingEvaluator,
)
from utils.prompt_validator import PromptValidationResult, PromptValidator


logger = logging.getLogger("tests.pipeline")


@pytest.fixture
def deterministic_ai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force MockAIService to emit deterministic scores during the test."""

    def _deterministic_evaluate(self: MockAIService, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        question = kwargs.get("question") or {}
        levels = question.get("niveles", [])
        max_score = 4.0
        for level in levels:
            if isinstance(level, dict) and "valor" in level:
                try:
                    candidate = float(level["valor"])
                except (TypeError, ValueError):
                    continue
                max_score = max(max_score, candidate)
        score = round(0.78 * max_score, 2)
        return {
            "score": score,
            "justification": "Respuesta generada por MockAIService determinista.",
            "relevant_text": "Fragmento seleccionado",
            "metadata": {
                "model": self.model_name,
                "mock": True,
                "deterministic": True,
            },
        }

    monkeypatch.setattr(MockAIService, "evaluate", _deterministic_evaluate)


def _write_sample_document(target: Path) -> None:
    content = "\n".join(
        [
            "=== PAGE 1 ===",
            "Resumen Ejecutivo",
            "El informe resume los principales logros y retos del periodo evaluado.",
            "Incluye hallazgos clave para la toma de decisiones.",
            "",
            "=== PAGE 2 ===",
            "Prioridades de la política institucional",
            "La entidad prioriza el fortalecimiento de capacidades y servicios.",
            "Detalla iniciativas estratégicas para el siguiente año.",
        ]
    )
    target.write_text(content, encoding="utf-8")


def _write_incomplete_document(target: Path) -> None:
    """Document lacking recognizable institutional sections."""

    content = "\n".join(
        [
            "=== PAGE 1 ===",
            "Contexto general",
            "El informe incluye reflexiones dispersas sin títulos formales.",
            "No se identifican apartados específicos del manual institucional.",
            "Se mencionan logros y retos sin estructura reconocible.",
        ]
    )
    target.write_text(content, encoding="utf-8")


def _write_criteria(target: Path) -> Dict[str, Any]:
    criteria: Dict[str, Any] = {
        "version": "integracion-v1",
        "tipo_informe": "institucional",
        "metodologia": "ponderada",
        "metrica_global": {"escala_resultado": {"min": 0.0, "max": 4.0}},
        "secciones": [
            {
                "id": "resumen_ejecutivo",
                "titulo": "Resumen Ejecutivo",
                "ponderacion": 0.6,
                "dimensiones": [
                    {
                        "nombre": "Contexto General",
                        "ponderacion": 1.0,
                        "metodo_agregacion": "promedio_ponderado",
                        "preguntas": [
                            {
                                "id": "P1",
                                "texto": "¿Cuál es el panorama general del informe?",
                                "ponderacion": 0.5,
                                "niveles": [{"valor": 4.0}, {"valor": 2.0}],
                            },
                            {
                                "id": "P2",
                                "texto": "¿Qué elementos se destacan en el resumen ejecutivo?",
                                "ponderacion": 0.5,
                                "niveles": [{"valor": 4.0}],
                            },
                        ],
                    }
                ],
            },
            {
                "id": "prioridades_politica_institucional",
                "titulo": "Prioridades de la política institucional",
                "ponderacion": 0.4,
                "dimensiones": [
                    {
                        "nombre": "Planificación",
                        "ponderacion": 1.0,
                        "preguntas": [
                            {
                                "id": "P3",
                                "texto": "¿Cómo se gestionan las prioridades institucionales?",
                                "ponderacion": 1.0,
                                "niveles": [{"valor": 4.0}],
                            }
                        ],
                    }
                ],
            },
        ],
    }
    target.write_text(json.dumps(criteria, ensure_ascii=False, indent=2), encoding="utf-8")
    return criteria


def _force_prompt_batch_support(monkeypatch: pytest.MonkeyPatch) -> None:
    original_init = ValidatingEvaluator.__init__
    original_resolve = EvaluationService._resolve_ai_service
    state: Dict[str, Any] = {}

    def _capture_resolve(self: EvaluationService, config: ServiceConfig) -> Any:
        ai_service = original_resolve(self, config)
        state["ai_service"] = ai_service
        return ai_service

    def _wrapped_init(
        self: ValidatingEvaluator,
        *args: Any,
        prompt_validator: PromptValidator,
        prompt_quality_threshold: float,
        prompt_batch_size: int = 1,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("prompt_batch_size", None)
        ai_service = state.get("ai_service")
        if ai_service is None:
            raise RuntimeError("AI service must be resolved before instantiating the evaluator")
        original_init(
            self,
            ai_service,
            *args,
            prompt_validator=prompt_validator,
            prompt_quality_threshold=prompt_quality_threshold,
            **kwargs,
        )
        self.prompt_batch_size = max(1, int(prompt_batch_size))

    monkeypatch.setattr(EvaluationService, "_resolve_ai_service", _capture_resolve)
    monkeypatch.setattr(ValidatingEvaluator, "__init__", _wrapped_init)


def _log_execution_stats(evaluation: Any, metrics: Dict[str, Any]) -> None:
    prompts_metrics = metrics.get("prompts")
    if not isinstance(prompts_metrics, dict):
        prompts_metrics = {}
    if not prompts_metrics:
        summary = (
            evaluation.metadata.get("prompt_validation", {})
            if hasattr(evaluation, "metadata")
            else {}
        )
        prompts_metrics = {
            "avg_quality": summary.get("average_quality") or 0.0,
            "rejected": summary.get("rejected_prompts") or 0,
            "total": summary.get("total_prompts") or 0,
        }
        metrics["prompts"] = prompts_metrics
    avg_quality = prompts_metrics.get("avg_quality")
    rejected = prompts_metrics.get("rejected")
    total = prompts_metrics.get("total")
    try:
        avg_quality_value = float(avg_quality) if avg_quality is not None else 0.0
    except (TypeError, ValueError):
        avg_quality_value = 0.0
    try:
        rejected_value = int(rejected) if rejected is not None else 0
    except (TypeError, ValueError):
        rejected_value = 0
    try:
        total_value = int(total) if total is not None else 0
    except (TypeError, ValueError):
        total_value = 0
    logger.info(f"Total de secciones evaluadas: {len(getattr(evaluation, 'sections', []))}")
    logger.info(f"Promedio calidad de prompts: {avg_quality_value:.2f}")
    logger.info(f"Prompts rechazados: {rejected_value}/{total_value}")


class _SimpleChunk:
    def __init__(self, text: str, metadata: Dict[str, Any] | None = None) -> None:
        self.page_content = text
        self.metadata = dict(metadata or {})


def _ceplan_institutional_criteria() -> Dict[str, Any]:
    return {
        "version": "ceplan-institucional-test",
        "tipo_informe": "institucional",
        "metodologia": "ponderada",
        "metrica_global": {"escala_resultado": {"min": 0.0, "max": 4.0}},
        "secciones": [
            {
                "id": "planificacion",
                "titulo": "Planificación Institucional",
                "ponderacion": 0.05,
                "dimensiones": [
                    {
                        "nombre": "Planificación",
                        "ponderacion": 1.0,
                        "metodo_agregacion": "promedio_ponderado",
                        "preguntas": [
                            {
                                "id": "PLAN_1",
                                "texto": "¿Cómo se planifican las acciones institucionales prioritarias?",
                                "ponderacion": 1.0,
                                "niveles": [{"valor": 2.0}],
                            }
                        ],
                    }
                ],
            },
            {
                "id": "gestion",
                "titulo": "Gestión Institucional",
                "ponderacion": 0.35,
                "dimensiones": [
                    {
                        "nombre": "Gestión",
                        "ponderacion": 1.0,
                        "metodo_agregacion": "promedio_ponderado",
                        "preguntas": [
                            {
                                "id": "GESTION_1",
                                "texto": "¿Qué tan efectiva es la gestión de la política institucional?",
                                "ponderacion": 1.0,
                                "niveles": [{"valor": 3.0}],
                            }
                        ],
                    }
                ],
            },
            {
                "id": "resultados",
                "titulo": "Resultados de la Política",
                "ponderacion": 0.60,
                "dimensiones": [
                    {
                        "nombre": "Resultados",
                        "ponderacion": 1.0,
                        "metodo_agregacion": "promedio_ponderado",
                        "preguntas": [
                            {
                                "id": "RESULTADOS_1",
                                "texto": "¿Qué resultados se alcanzaron en la política institucional?",
                                "ponderacion": 1.0,
                                "niveles": [{"valor": 4.0}],
                            }
                        ],
                    }
                ],
            },
        ],
    }


def _ceplan_policy_criteria() -> Dict[str, Any]:
    return {
        "version": "ceplan-politica-test",
        "tipo_informe": "politica_nacional",
        "metodologia": "normalizada",
        "escala": {"min": 0.0, "max": 2.0},
        "escala_normalizada": {"min": 0.0, "max": 20.0},
        "bloques": [
            {
                "id": "diagnostico",
                "titulo": "Diagnóstico",
                "ponderacion": 0.4,
                "preguntas": [
                    {
                        "id": "POL_DIAG",
                        "texto": "¿Cómo caracteriza el informe el diagnóstico del problema público?",
                        "ponderacion": 1.0,
                    }
                ],
            },
            {
                "id": "implementacion",
                "titulo": "Implementación",
                "ponderacion": 0.3,
                "preguntas": [
                    {
                        "id": "POL_IMPL",
                        "texto": "¿Cómo se implementan las intervenciones propuestas?",
                        "ponderacion": 1.0,
                    }
                ],
            },
            {
                "id": "seguimiento",
                "titulo": "Seguimiento",
                "ponderacion": 0.3,
                "preguntas": [
                    {
                        "id": "POL_SEG",
                        "texto": "¿Cómo se monitorean y evalúan los avances?",
                        "ponderacion": 1.0,
                    }
                ],
            },
        ],
    }


@pytest.mark.usefixtures("deterministic_ai")
def test_full_pipeline_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    document_path = tmp_path / "documento_institucional.txt"
    criteria_path = tmp_path / "criterios.json"
    output_path = tmp_path / "resultado.json"

    _write_sample_document(document_path)
    criteria = _write_criteria(criteria_path)

    validation_calls: List[Dict[str, Any]] = []

    def _fake_validate(
        self: PromptValidator, prompt: str, context: Dict[str, Any] | None = None
    ) -> PromptValidationResult:
        call_context = dict(context or {})
        validation_calls.append({"prompt": prompt, "context": call_context})
        return PromptValidationResult(
            is_valid=True,
            quality_score=0.95,
            alerts=["estructura_completa"],
            metadata={"quality_band": "alta", "source": "test"},
        )

    monkeypatch.setattr(PromptValidator, "validate", _fake_validate)
    _force_prompt_batch_support(monkeypatch)

    config = ServiceConfig(
        ai_provider="mock",
        run_id="pipeline-integration",
        model_name="mock-integration",
        prompt_batch_size=1,
        retries=0,
        timeout_seconds=15.0,
        log_level="ERROR",
    )
    service = EvaluationService(config=config)

    evaluation, metrics = service.run(
        input_path=document_path,
        criteria_path=criteria_path,
        output_path=output_path,
        output_format="json",
        extra_metadata={"invocation": "integration-test"},
    )

    _log_execution_stats(evaluation, metrics)

    assert output_path.exists()
    exported = json.loads(output_path.read_text(encoding="utf-8"))

    assert evaluation.document_id == document_path.stem
    assert evaluation.document_type == "institucional"
    assert evaluation.metadata["criteria_version"] == criteria["version"]
    assert evaluation.metadata["model_name"] == "mock-integration"
    assert evaluation.metadata["run_id"] == "pipeline-integration"
    assert evaluation.metadata["mode"] == "completo"
    assert evaluation.metadata["prompt_batch_size"] == 1
    assert evaluation.metadata["retries"] == 0
    assert evaluation.metadata["timeout_seconds"] == 15.0
    assert evaluation.metadata["runs"][-1]["model"] == "mock-integration"
    assert evaluation.metadata["source_path"] == str(document_path)

    assert len(evaluation.sections) == 2
    section_ids = [section.section_id for section in evaluation.sections]
    assert section_ids == ["resumen_ejecutivo", "prioridades_politica_institucional"]
    first_section = evaluation.sections[0]
    second_section = evaluation.sections[1]
    assert pytest.approx(first_section.weight, rel=1e-6) == 0.6
    assert pytest.approx(second_section.weight, rel=1e-6) == 0.4

    for section in evaluation.sections:
        for dimension in section.dimensions:
            assert dimension.questions, "Cada dimensión debe contener preguntas evaluadas"
            for question in dimension.questions:
                assert question.score is not None
                assert 0.0 <= question.score <= 4.0
                assert question.chunk_results, "Cada pregunta debe incluir resultados por chunk"
                chunk_sources: List[str] = []
                for chunk in question.chunk_results:
                    assert chunk.score is not None and 0.0 <= chunk.score <= 4.0
                    metadata = chunk.metadata
                    assert metadata["prompt_was_valid"] is True
                    assert metadata["prompt_quality_score"] == pytest.approx(0.95)
                    assert metadata["prompt_quality_threshold"] == pytest.approx(
                        PROMPT_QUALITY_THRESHOLD
                    )
                    assert metadata.get("prompt_validation_alerts") == [
                        "estructura_completa"
                    ]
                    assert metadata.get("prompt_validation_metadata") == {
                        "quality_band": "alta",
                        "source": "test",
                    }
                    assert not metadata.get("prompt_rejected", False)
                    assert metadata.get("mock") is True
                    assert metadata.get("model") == "mock-integration"
                    # chunk_metadata preserves segmentation + cleaning report
                    chunk_meta = metadata.get("chunk_metadata") or {}
                    document_meta = chunk_meta.get("document_metadata") or {}
                    assert "cleaning_report" in document_meta
                    if "source_id" in chunk_meta:
                        chunk_sources.append(str(chunk_meta["source_id"]))
                # At least one chunk should come from the section that defines the question
                assert section.section_id in chunk_sources

    total_chunks = sum(
        len(question.chunk_results)
        for section in evaluation.sections
        for dimension in section.dimensions
        for question in dimension.questions
    )

    assert len(validation_calls) == total_chunks
    first_call_context = validation_calls[0]["context"]
    assert first_call_context["report_type"] == criteria["tipo_informe"]
    assert first_call_context["section_id"] == "resumen_ejecutivo"
    assert first_call_context["dimension_name"] == "Contexto General"
    assert first_call_context["question_id"] == "P1"
    assert first_call_context["chunk_index"] == 0

    first_question_metadata = first_section.dimensions[0].questions[0].metadata
    assert first_question_metadata["prompt_quality_minimum"] == pytest.approx(
        PROMPT_QUALITY_THRESHOLD
    )
    validation_block = first_question_metadata["prompt_validation"]
    assert validation_block["threshold"] == pytest.approx(PROMPT_QUALITY_THRESHOLD)
    assert len(validation_block["records"]) == len(
        first_section.dimensions[0].questions[0].chunk_results
    )
    assert all(record["was_valid"] is True for record in validation_block["records"])

    prompt_summary = evaluation.metadata.get("prompt_validation")
    assert prompt_summary
    assert prompt_summary["threshold"] == pytest.approx(PROMPT_QUALITY_THRESHOLD)
    assert prompt_summary["total_prompts"] == total_chunks
    assert prompt_summary["rejected_prompts"] == 0
    assert prompt_summary["average_quality"] == pytest.approx(0.95)
    resumen_section_summary = next(
        summary
        for summary in prompt_summary["sections"]
        if summary["label"].startswith("Resumen Ejecutivo")
    )
    assert resumen_section_summary["total_prompts"] > 0
    assert resumen_section_summary["rejected_prompts"] == 0

    assert (
        exported["evaluation"]["metadata"]["prompt_validation"]["total_prompts"]
        == total_chunks
    )

    assert evaluation.score is not None
    expected_normalised = (evaluation.score / 4.0) * 100.0
    assert metrics["methodology"] == "institucional"
    assert metrics["global"]["raw_score"] == pytest.approx(evaluation.score)
    assert metrics["global"]["normalized_score"] == pytest.approx(expected_normalised)
    assert metrics["totals"]["sections_total"] == 2
    assert metrics["sections"][0]["section_id"] == "resumen_ejecutivo"
    assert metrics["sections"][1]["section_id"] == "prioridades_politica_institucional"


@pytest.mark.usefixtures("deterministic_ai")
@pytest.mark.parametrize("output_format", ["csv", "xlsx"])
def test_pipeline_exports_tabular_formats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, output_format: str
) -> None:
    pandas = pytest.importorskip("pandas")
    if output_format == "xlsx":
        openpyxl = pytest.importorskip("openpyxl")

    document_path = tmp_path / "documento_institucional.txt"
    criteria_path = tmp_path / "criterios.json"
    output_path = tmp_path / f"resultado.{output_format}"

    _write_sample_document(document_path)
    criteria = _write_criteria(criteria_path)

    def _fake_validate(
        self: PromptValidator, prompt: str, context: Dict[str, Any] | None = None
    ) -> PromptValidationResult:
        return PromptValidationResult(
            is_valid=True,
            quality_score=0.92,
            alerts=["estructura_completa"],
            metadata={"quality_band": "alta", "source": "test"},
        )

    monkeypatch.setattr(PromptValidator, "validate", _fake_validate)
    _force_prompt_batch_support(monkeypatch)

    config = ServiceConfig(
        ai_provider="mock",
        run_id=f"pipeline-{output_format}",
        model_name="mock-integration",
        prompt_batch_size=1,
        retries=0,
        timeout_seconds=15.0,
        log_level="ERROR",
    )
    service = EvaluationService(config=config)

    evaluation, metrics = service.run(
        input_path=document_path,
        criteria_path=criteria_path,
        output_path=output_path,
        output_format=output_format,
        extra_metadata={"invocation": "tabular-export"},
    )

    _log_execution_stats(evaluation, metrics)

    assert output_path.exists()

    if output_format == "csv":
        raw_bytes = output_path.read_bytes()
        assert raw_bytes.startswith(codecs.BOM_UTF8)
        dataframe = pandas.read_csv(output_path, encoding="utf-8-sig")
        assert not dataframe.empty
        assert dataframe["document_id"].unique().tolist() == [evaluation.document_id]
        expected_columns = [
            "document_id",
            "section_id",
            "section_title",
            "section_score",
            "section_weight",
            "dimension_name",
            "dimension_score",
            "dimension_weight",
            "question_id",
        ]
        assert dataframe.columns.tolist()[: len(expected_columns)] == expected_columns
        assert "chunk_results" in dataframe.columns
    else:
        workbook = openpyxl.load_workbook(output_path)
        assert workbook.sheetnames == ["preguntas", "resumen", "indice_global"]
        preguntas_sheet = workbook["preguntas"]
        header_row = [cell.value for cell in next(preguntas_sheet.iter_rows(min_row=1, max_row=1))]
        expected_prefix = [
            "document_id",
            "section_id",
            "section_title",
            "section_score",
            "section_weight",
        ]
        assert header_row[: len(expected_prefix)] == expected_prefix
        header_df = pandas.read_excel(output_path, sheet_name="indice_global")
        assert header_df.loc[0, "run_id"] == config.run_id
        assert header_df.loc[0, "model_name"] == config.model_name
        assert header_df.loc[0, "criteria_version"] == criteria["version"]


def test_pipeline_calculates_ceplan_weighted_indices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_prompt_batch_support(monkeypatch)

    def _always_valid(
        self: PromptValidator, prompt: str, context: Dict[str, Any] | None = None
    ) -> PromptValidationResult:
        return PromptValidationResult(
            is_valid=True,
            quality_score=0.99,
            alerts=["estructura_validada"],
            metadata={"source": "ceplan-test"},
        )

    monkeypatch.setattr(PromptValidator, "validate", _always_valid)

    score_map = {
        "PLAN_1": 1.0,
        "GESTION_1": 2.0,
        "RESULTADOS_1": 3.0,
        "POL_DIAG": 0.5,
        "POL_IMPL": 1.0,
        "POL_SEG": 1.5,
    }

    def _score_evaluate(self: MockAIService, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        question = kwargs.get("question") or {}
        question_id = (
            question.get("id")
            or question.get("question_id")
            or question.get("codigo")
            or question.get("texto")
        )
        assert question_id in score_map, f"Unexpected question id {question_id}"
        score = float(score_map[question_id])
        return {
            "score": score,
            "justification": f"Puntaje CEPLAN simulado para {question_id}",
            "relevant_text": "Fragmento CEPLAN",
            "metadata": {
                "model": self.model_name,
                "mock": True,
                "ceplan_weighted": True,
            },
        }

    monkeypatch.setattr(MockAIService, "evaluate", _score_evaluate)

    config = ServiceConfig(
        ai_provider="mock",
        run_id="pipeline-ceplan",
        model_name="mock-ceplan",
        prompt_batch_size=1,
        retries=0,
        timeout_seconds=10.0,
        log_level="ERROR",
    )
    service = EvaluationService(config=config)

    institutional_document = Document(
        content="Informe CEPLAN institucional",
        metadata={
            "id": "doc-institucional-ceplan",
            "title": "Informe Institucional CEPLAN",
            "document_type": "institucional",
        },
        chunks=[
            _SimpleChunk("Síntesis de planificación", {"source_id": "planificacion"}),
            _SimpleChunk("Gestión institucional detallada", {"source_id": "gestion"}),
            _SimpleChunk("Resultados de política", {"source_id": "resultados"}),
        ],
    )

    inst_output = tmp_path / "ceplan_institucional.json"
    inst_criteria = _ceplan_institutional_criteria()

    inst_evaluation, inst_metrics = service.run(
        document=institutional_document,
        criteria_data=inst_criteria,
        tipo_informe="institucional",
        output_path=inst_output,
        output_format="json",
        extra_metadata={"escenario": "ceplan"},
    )

    _log_execution_stats(inst_evaluation, inst_metrics)

    assert inst_output.exists()
    expected_institutional = (
        0.05 * score_map["PLAN_1"]
        + 0.35 * score_map["GESTION_1"]
        + 0.60 * score_map["RESULTADOS_1"]
    )
    assert inst_evaluation.score == pytest.approx(expected_institutional)

    institutional_weights = {
        section.section_id: section.weight for section in inst_evaluation.sections
    }
    assert institutional_weights == {
        "planificacion": pytest.approx(0.05),
        "gestion": pytest.approx(0.35),
        "resultados": pytest.approx(0.60),
    }

    total_weight = sum(
        (section.weight or 0.0)
        for section in inst_evaluation.sections
        if section.score is not None
    )
    assert total_weight == pytest.approx(1.0)
    recomputed_global = sum(
        (section.weight or 0.0) * (section.score or 0.0)
        for section in inst_evaluation.sections
        if section.score is not None
    )
    assert inst_evaluation.score == pytest.approx(recomputed_global / total_weight)

    inst_section_metrics = {
        row["section_id"]: row for row in inst_metrics["sections"]
    }
    assert inst_section_metrics["planificacion"]["weight"] == pytest.approx(0.05)
    assert inst_section_metrics["gestion"]["weight"] == pytest.approx(0.35)
    assert inst_section_metrics["resultados"]["weight"] == pytest.approx(0.60)
    assert inst_metrics["global"]["raw_score"] == pytest.approx(expected_institutional)
    assert inst_metrics["global"]["normalized_score"] == pytest.approx(
        expected_institutional / 4.0 * 100.0
    )

    policy_document = Document(
        content="Informe CEPLAN política",
        metadata={
            "id": "doc-politica-ceplan",
            "title": "Política Nacional CEPLAN",
            "document_type": "politica_nacional",
        },
        chunks=[
            _SimpleChunk("Diagnóstico situacional", {"source_id": "diagnostico"}),
            _SimpleChunk("Implementación progresiva", {"source_id": "implementacion"}),
            _SimpleChunk("Seguimiento y monitoreo", {"source_id": "seguimiento"}),
        ],
    )

    pol_output = tmp_path / "ceplan_politica.json"
    pol_criteria = _ceplan_policy_criteria()

    pol_evaluation, pol_metrics = service.run(
        document=policy_document,
        criteria_data=pol_criteria,
        tipo_informe="politica_nacional",
        output_path=pol_output,
        output_format="json",
        extra_metadata={"escenario": "ceplan"},
    )

    _log_execution_stats(pol_evaluation, pol_metrics)

    assert pol_output.exists()
    expected_policy = (
        0.4 * score_map["POL_DIAG"]
        + 0.3 * score_map["POL_IMPL"]
        + 0.3 * score_map["POL_SEG"]
    )
    assert pol_evaluation.score == pytest.approx(expected_policy)

    pol_weights = {
        section.section_id: section.weight for section in pol_evaluation.sections
    }
    assert pol_weights == {
        "diagnostico": pytest.approx(0.4),
        "implementacion": pytest.approx(0.3),
        "seguimiento": pytest.approx(0.3),
    }

    pol_weight_total = sum(
        (section.weight or 0.0)
        for section in pol_evaluation.sections
        if section.score is not None
    )
    assert pol_weight_total == pytest.approx(1.0)
    pol_recomputed = sum(
        (section.weight or 0.0) * (section.score or 0.0)
        for section in pol_evaluation.sections
        if section.score is not None
    )
    assert pol_evaluation.score == pytest.approx(pol_recomputed / pol_weight_total)

    assert pol_metrics["methodology"] == "politica_nacional"
    assert pol_metrics["global"]["raw_score"] == pytest.approx(expected_policy)
    assert pol_metrics["global"]["normalized_min"] == pytest.approx(0.0)
    assert pol_metrics["global"]["normalized_max"] == pytest.approx(20.0)
    assert pol_metrics["global"]["normalized_score"] == pytest.approx(
        expected_policy / 2.0 * 20.0
    )

    pol_section_metrics = {row["section_id"]: row for row in pol_metrics["sections"]}
    assert pol_section_metrics["diagnostico"]["weight"] == pytest.approx(0.4)
    assert pol_section_metrics["implementacion"]["weight"] == pytest.approx(0.3)
    assert pol_section_metrics["seguimiento"]["weight"] == pytest.approx(0.3)

    exported_pol = json.loads(pol_output.read_text(encoding="utf-8"))
    assert (
        exported_pol["metrics"]["global"]["normalized_max"]
        == pol_metrics["global"]["normalized_max"]
    )
    assert exported_pol["metrics"]["global"]["normalized_score"] == pytest.approx(
        pol_metrics["global"]["normalized_score"]
    )


def test_pipeline_retries_transient_failures_and_timeouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document_path = tmp_path / "documento_institucional.txt"
    criteria_path = tmp_path / "criterios.json"
    output_path = tmp_path / "resultado.json"

    _write_sample_document(document_path)
    _write_criteria(criteria_path)

    monkeypatch.setattr(evaluation_module.time, "sleep", lambda _: None)

    def _always_valid(
        self: PromptValidator, prompt: str, context: Dict[str, Any] | None = None
    ) -> PromptValidationResult:
        return PromptValidationResult(
            is_valid=True,
            quality_score=0.88,
            alerts=[],
            metadata={},
        )

    monkeypatch.setattr(PromptValidator, "validate", _always_valid)

    attempts: Dict[str, int] = {}

    def _flaky_evaluate(self: MockAIService, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        question = kwargs.get("question") or {}
        question_id = str(question.get("id") or "desconocido")
        attempts[question_id] = attempts.get(question_id, 0) + 1
        if question_id == "P1" and attempts[question_id] == 1:
            raise RuntimeError("falla transitoria")
        if question_id == "P3":
            raise FuturesTimeoutError()
        base_score = {
            "P1": 3.2,
            "P2": 3.4,
            "P3": 2.5,
        }.get(question_id, 3.0)
        return {
            "score": base_score,
            "justification": f"Respuesta simulada para {question_id}",
            "relevant_text": "Fragmento relevante",
            "metadata": {"mock": True, "model": self.model_name},
        }

    monkeypatch.setattr(MockAIService, "evaluate", _flaky_evaluate)
    _force_prompt_batch_support(monkeypatch)

    config = ServiceConfig(
        ai_provider="mock",
        run_id="pipeline-retries",
        model_name="mock-flaky",
        prompt_batch_size=1,
        retries=1,
        timeout_seconds=0.01,
        log_level="ERROR",
    )
    service = EvaluationService(config=config)

    evaluation, metrics = service.run(
        input_path=document_path,
        criteria_path=criteria_path,
        output_path=output_path,
        output_format="json",
    )

    _log_execution_stats(evaluation, metrics)

    assert output_path.exists()
    assert attempts.get("P1", 0) >= 2
    assert attempts.get("P3", 0) >= 2

    p1_question = next(
        question
        for section in evaluation.sections
        for dimension in section.dimensions
        for question in dimension.questions
        if question.question_id == "P1"
    )
    assert all(
        not chunk.metadata.get("score_imputed") for chunk in p1_question.chunk_results
    )

    p3_question = next(
        question
        for section in evaluation.sections
        for dimension in section.dimensions
        for question in dimension.questions
        if question.question_id == "P3"
    )
    assert any(chunk.score == 0 for chunk in p3_question.chunk_results)
    assert all(
        chunk.metadata.get("score_imputed") is True and chunk.metadata.get("error")
        for chunk in p3_question.chunk_results
    )

    assert metrics["global"]["raw_score"] is not None


def test_pipeline_idempotence_allows_small_variation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document_path = tmp_path / "documento_institucional.txt"
    criteria_path = tmp_path / "criterios.json"
    first_output = tmp_path / "resultado_1.json"
    second_output = tmp_path / "resultado_2.json"

    _write_sample_document(document_path)
    _write_criteria(criteria_path)

    def _always_valid(
        self: PromptValidator, prompt: str, context: Dict[str, Any] | None = None
    ) -> PromptValidationResult:
        return PromptValidationResult(
            is_valid=True,
            quality_score=0.93,
            alerts=[],
            metadata={},
        )

    def _noisy_evaluate(self: MockAIService, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        question = kwargs.get("question") or {}
        question_id = str(question.get("id") or "desconocido")
        base_score = {
            "P1": 3.4,
            "P2": 3.7,
            "P3": 3.1,
        }.get(question_id, 3.2)
        offset = 0.0
        if self.model_name.endswith("variant"):
            offset = 0.08
        score = round(base_score + offset, 2)
        return {
            "score": score,
            "justification": f"Score {score} para {question_id}",
            "relevant_text": "Fragmento estable",
            "metadata": {"mock": True, "model": self.model_name},
        }

    monkeypatch.setattr(PromptValidator, "validate", _always_valid)
    monkeypatch.setattr(MockAIService, "evaluate", _noisy_evaluate)
    _force_prompt_batch_support(monkeypatch)

    base_config = ServiceConfig(
        ai_provider="mock",
        prompt_batch_size=1,
        retries=0,
        timeout_seconds=15.0,
        log_level="ERROR",
    )

    first_service = EvaluationService(
        config=base_config.with_overrides(run_id="idemp-run-1", model_name="mock-idem-base")
    )
    first_evaluation, first_metrics = first_service.run(
        input_path=document_path,
        criteria_path=criteria_path,
        output_path=first_output,
        output_format="json",
    )

    _log_execution_stats(first_evaluation, first_metrics)

    second_service = EvaluationService(
        config=base_config.with_overrides(
            run_id="idemp-run-2", model_name="mock-idem-variant"
        )
    )
    second_evaluation, second_metrics = second_service.run(
        input_path=document_path,
        criteria_path=criteria_path,
        output_path=second_output,
        output_format="json",
    )

    _log_execution_stats(second_evaluation, second_metrics)

    assert first_output.exists()
    assert second_output.exists()

    assert first_evaluation.document_id == second_evaluation.document_id
    assert len(first_evaluation.sections) == len(second_evaluation.sections)

    if first_evaluation.score is not None and second_evaluation.score is not None:
        assert first_evaluation.score == pytest.approx(
            second_evaluation.score, abs=0.1
        )

    for first_section, second_section in zip(
        first_evaluation.sections, second_evaluation.sections
    ):
        assert first_section.section_id == second_section.section_id
        if first_section.score is not None and second_section.score is not None:
            assert first_section.score == pytest.approx(
                second_section.score, abs=0.1
            )

    assert first_metrics["global"]["raw_score"] == pytest.approx(
        second_metrics["global"]["raw_score"], abs=0.1
    )


def test_pipeline_rejects_low_quality_prompts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document_path = tmp_path / "documento_institucional.txt"
    criteria_path = tmp_path / "criterios.json"
    output_path = tmp_path / "resultado.json"

    _write_sample_document(document_path)
    criteria = _write_criteria(criteria_path)

    rejection_calls: List[Dict[str, Any]] = []

    def _rejecting_validate(
        self: PromptValidator, prompt: str, context: Dict[str, Any] | None = None
    ) -> PromptValidationResult:
        rejection_calls.append({"prompt": prompt, "context": dict(context or {})})
        return PromptValidationResult(
            is_valid=False,
            quality_score=0.2,
            alerts=["prompt_demasiado_corto"],
            metadata={"quality_band": "rechazada", "source": "test"},
        )

    def _fail_if_called(
        self: MockAIService, prompt: str, **kwargs: Any
    ) -> Dict[str, Any]:
        raise AssertionError("AI no debe invocarse cuando el prompt es rechazado")

    monkeypatch.setattr(PromptValidator, "validate", _rejecting_validate)
    monkeypatch.setattr(MockAIService, "evaluate", _fail_if_called)
    _force_prompt_batch_support(monkeypatch)

    config = ServiceConfig(
        ai_provider="mock",
        run_id="pipeline-rejection",
        model_name="mock-integration",
        prompt_batch_size=1,
        retries=0,
        timeout_seconds=15.0,
        log_level="ERROR",
    )
    service = EvaluationService(config=config)

    evaluation, metrics = service.run(
        input_path=document_path,
        criteria_path=criteria_path,
        output_path=output_path,
        output_format="json",
    )

    _log_execution_stats(evaluation, metrics)

    assert output_path.exists()
    exported = json.loads(output_path.read_text(encoding="utf-8"))

    total_chunks = sum(
        len(question.chunk_results)
        for section in evaluation.sections
        for dimension in section.dimensions
        for question in dimension.questions
    )
    assert total_chunks == len(rejection_calls) > 0

    for call in rejection_calls:
        ctx = call["context"]
        assert ctx["report_type"] == criteria["tipo_informe"]
        assert ctx["section_id"] in {"resumen_ejecutivo", "prioridades_politica_institucional"}
        assert ctx["chunk_index"] >= 0

    for section in evaluation.sections:
        for dimension in section.dimensions:
            for question in dimension.questions:
                assert question.metadata["prompt_validation"]["threshold"] == pytest.approx(
                    PROMPT_QUALITY_THRESHOLD
                )
                for record in question.metadata["prompt_validation"]["records"]:
                    assert record["was_valid"] is False
                for chunk in question.chunk_results:
                    assert chunk.score is None
                    assert chunk.metadata.get("prompt_rejected") is True
                    assert chunk.metadata.get("prompt_was_valid") is False
                    assert chunk.metadata.get("prompt_validation_alerts") == [
                        "prompt_demasiado_corto"
                    ]

    prompt_summary = evaluation.metadata["prompt_validation"]
    assert prompt_summary["total_prompts"] == total_chunks
    assert prompt_summary["rejected_prompts"] == total_chunks
    assert prompt_summary["average_quality"] == pytest.approx(0.2)

    assert evaluation.score is None
    assert metrics["global"]["raw_score"] is None
    assert (
        exported["evaluation"]["metadata"]["prompt_validation"]["rejected_prompts"]
        == total_chunks
    )
    assert exported["evaluation"]["metadata"]["run_id"] == config.run_id
    assert exported["evaluation"]["metadata"]["criteria_version"] == criteria["version"]
    assert exported["metrics"]["global"]["raw_score"] is None
    assert exported["extra"]["tipo_informe"] == "institucional"
    assert exported["extra"]["criteria_version"] == criteria["version"]


@pytest.mark.usefixtures("deterministic_ai")
def test_pipeline_handles_missing_sections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document_path = tmp_path / "documento_incompleto.txt"
    criteria_path = tmp_path / "criterios.json"
    output_path = tmp_path / "resultado.json"

    _write_incomplete_document(document_path)
    criteria = _write_criteria(criteria_path)

    validation_calls: List[Dict[str, Any]] = []

    def _fake_validate(
        self: PromptValidator, prompt: str, context: Dict[str, Any] | None = None
    ) -> PromptValidationResult:
        call_context = dict(context or {})
        validation_calls.append({"prompt": prompt, "context": call_context})
        return PromptValidationResult(
            is_valid=True,
            quality_score=0.91,
            alerts=["estructura_incompleta"],
            metadata={"quality_band": "media", "source": "test"},
        )

    monkeypatch.setattr(PromptValidator, "validate", _fake_validate)
    _force_prompt_batch_support(monkeypatch)

    config = ServiceConfig(
        ai_provider="mock",
        run_id="pipeline-missing-section",
        model_name="mock-integration",
        prompt_batch_size=1,
        retries=0,
        timeout_seconds=15.0,
        log_level="ERROR",
    )
    service = EvaluationService(config=config)

    evaluation, metrics = service.run(
        input_path=document_path,
        criteria_path=criteria_path,
        output_format="json",
        extra_metadata={"invocation": "missing-section"},
    )

    _log_execution_stats(evaluation, metrics)

    assert output_path.exists()
    exported = json.loads(output_path.read_text(encoding="utf-8"))

    total_chunks = sum(
        len(question.chunk_results)
        for section in evaluation.sections
        for dimension in section.dimensions
        for question in dimension.questions
    )
    assert total_chunks == len(validation_calls) > 0

    missing_section = next(
        section
        for section in evaluation.sections
        if section.section_id == "prioridades_politica_institucional"
    )
    assert missing_section.dimensions
    missing_question = missing_section.dimensions[0].questions[0]
    assert missing_question.chunk_results

    fallback_sources = {
        chunk.metadata.get("chunk_metadata", {}).get("source_id")
        for chunk in missing_question.chunk_results
    }
    assert fallback_sources
    assert all(
        str(source).startswith("sin_clasificar") or "sin_clasificar" in str(source)
        for source in fallback_sources
        if source
    )

    all_source_ids = {
        chunk.metadata.get("chunk_metadata", {}).get("source_id")
        for section in evaluation.sections
        for dimension in section.dimensions
        for question in dimension.questions
        for chunk in question.chunk_results
    }
    assert all_source_ids == {"sin_clasificar"}

    validation_records = missing_question.metadata["prompt_validation"]["records"]
    assert validation_records
    assert all(record["alerts"] == ["estructura_incompleta"] for record in validation_records)

    missing_contexts = [
        call["context"]
        for call in validation_calls
        if call["context"].get("section_id") == "prioridades_politica_institucional"
    ]
    assert missing_contexts
    assert all(ctx["question_id"] == "P3" for ctx in missing_contexts)

    prompt_summary = evaluation.metadata["prompt_validation"]
    assert prompt_summary["total_prompts"] == total_chunks
    assert prompt_summary["rejected_prompts"] == 0

    missing_summary = next(
        summary
        for summary in prompt_summary["sections"]
        if summary["label"].startswith("Prioridades de la política institucional")
    )
    assert missing_summary["total_prompts"] == len(missing_question.chunk_results)
    assert missing_summary["rejected_prompts"] == 0

    assert metrics["sections"][1]["section_id"] == "prioridades_politica_institucional"
    exported_summary = exported["evaluation"]["metadata"]["prompt_validation"]
    assert exported_summary["total_prompts"] == total_chunks
    exported_missing = next(
        summary
        for summary in exported_summary["sections"]
        if summary["label"].startswith("Prioridades de la política institucional")
    )
    assert exported_missing["total_prompts"] == len(missing_question.chunk_results)
    assert exported_missing["rejected_prompts"] == 0
    issues = exported_summary.get("issues", [])
    assert any(
        issue.get("question_id") == "P3" and issue.get("alerts") == ["estructura_incompleta"]
        for issue in issues
    )

# pytest tests/test_pipeline.py -v