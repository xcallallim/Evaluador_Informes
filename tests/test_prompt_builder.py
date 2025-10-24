import json

import pytest

from data.models.document import Document
from services.prompt_builder import (
    InstitutionalPromptBuilder,
    PolicyPromptBuilder,
    PromptContext,
    PromptFactory,
    build_prompt,
    build_prompt_from_context,
)


def _build_document() -> Document:
    return Document(
        content="",
        metadata={
            "title": "Informe de Gestión 2024",
            "id": "doc-123",
        },
    )


def _base_section() -> dict:
    return {"titulo": "Resumen Ejecutivo"}


def _base_dimension() -> dict:
    return {"nombre": "Claridad"}


def _base_question() -> dict:
    return {"texto": "¿El resumen presenta los avances con claridad?"}


def _institutional_criteria() -> dict:
    return {
        "tipo_informe": "institucional",
        "metodologia": "Metodología CEPLAN institucional",
        "version": "2024.1",
        "descripcion": "Evaluación de informes institucionales.",
    }


def _policy_criteria() -> dict:
    return {
        "tipo_informe": "politica_nacional",
        "metodologia": "Metodología CEPLAN PN",
        "version": "2024.1",
        "descripcion": "Evaluación de políticas nacionales.",
    }


def test_institutional_prompt_contains_expected_sections() -> None:
    builder = InstitutionalPromptBuilder()
    prompt = builder(
        document=_build_document(),
        criteria=_institutional_criteria(),
        section=_base_section(),
        dimension=_base_dimension(),
        question=_base_question(),
        chunk_text="El resumen describe resultados y recomendaciones prioritarias.",
        chunk_metadata={"page": 2},
    )

    assert "Escala de valoración (0 a 4" in prompt
    assert "Pregunta orientadora" in prompt
    assert "<<<INICIO_FRAGMENTO>>>" in prompt and "<<<FIN_FRAGMENTO>>>" in prompt
    assert "justificación técnica" in prompt
    metadata = builder.last_quality_metadata
    assert metadata is not None
    assert metadata["report_type"].lower() == "institucional"


def test_policy_prompt_includes_half_point_guidance() -> None:
    builder = PolicyPromptBuilder()
    prompt = builder(
        document=_build_document(),
        criteria=_policy_criteria(),
        section={"titulo": "Resultados priorizados"},
        dimension={"nombre": "Pertinencia"},
        question={"texto": "¿La política responde a los objetivos nacionales?"},
        chunk_text="El documento vincula resultados con los objetivos nacionales con ejemplos concretos.",
        chunk_metadata=None,
    )

    assert "0.5" in prompt
    assert "incrementos de 0.5" in prompt


def test_language_validation_requires_spanish() -> None:
    builder = InstitutionalPromptBuilder()
    with pytest.raises(ValueError):
        builder(
            document=_build_document(),
            criteria=_institutional_criteria(),
            section=_base_section(),
            dimension=_base_dimension(),
            question=_base_question(),
            chunk_text="This fragment is written entirely in English without Spanish words.",
            chunk_metadata=None,
        )


def test_dynamic_scale_from_json() -> None:
    scale_json = json.dumps(
        [
            {"value": 1, "description": "Nivel inicial", "label": "Básico"},
            {"value": 2, "description": "Nivel medio", "label": "Intermedio"},
        ]
    )
    builder = InstitutionalPromptBuilder(scale_config=scale_json)

    assert len(builder.scale_levels) == 2
    assert builder.scale_levels[0].label == "Básico"


def test_langchain_payload_contains_template_and_metadata() -> None:
    builder = PolicyPromptBuilder()
    context = PromptContext(
        document=_build_document(),
        criteria=_policy_criteria(),
        section=_base_section(),
        dimension=_base_dimension(),
        question=_base_question(),
        chunk_text="El contenido describe resultados y supuestos claves.",
        chunk_metadata=None,
    )

    payload = builder.to_langchain_prompt(context)

    assert payload["type"] == "prompt"
    assert "template" in payload
    assert isinstance(payload.get("metadata"), dict)


def test_factory_selects_policy_builder() -> None:
    factory = PromptFactory()
    factory.register("politica", PolicyPromptBuilder())
    builder = factory.for_criteria({"tipo_informe": "POLITICA_NACIONAL"})
    assert isinstance(builder, PolicyPromptBuilder)


def test_chunk_truncation_is_noted_in_prompt() -> None:
    builder = InstitutionalPromptBuilder(max_chunk_chars=30)
    prompt = builder(
        document=_build_document(),
        criteria=_institutional_criteria(),
        section=_base_section(),
        dimension=_base_dimension(),
        question=_base_question(),
        chunk_text="El contenido presenta avances estratégicos y compromisos específicos." * 2,
        chunk_metadata=None,
    )

    assert "… [texto truncado]" in prompt
    assert "se truncó a 30 caracteres" in prompt


def test_empty_chunk_raises_value_error() -> None:
    builder = InstitutionalPromptBuilder()
    with pytest.raises(ValueError):
        builder(
            document=_build_document(),
            criteria=_institutional_criteria(),
            section=_base_section(),
            dimension=_base_dimension(),
            question=_base_question(),
            chunk_text="   ",
            chunk_metadata=None,
        )


def test_question_requires_dimension() -> None:
    builder = InstitutionalPromptBuilder()
    with pytest.raises(ValueError):
        builder(
            document=_build_document(),
            criteria=_institutional_criteria(),
            section=_base_section(),
            dimension=None,
            question=_base_question(),
            chunk_text="Texto válido",
            chunk_metadata=None,
        )


def test_build_prompt_from_context_uses_extra_instructions() -> None:
    context = PromptContext(
        document=_build_document(),
        criteria=_policy_criteria(),
        section=_base_section(),
        dimension=_base_dimension(),
        question=_base_question(),
        chunk_text="El contenido describe la cadena de resultados esperados.",
        chunk_metadata={"segment": "A1"},
        extra_instructions="Prioriza la identificación de brechas críticas.",
    )

    prompt = build_prompt_from_context(context)
    assert "Prioriza la identificación de brechas críticas." in prompt


def test_build_prompt_uses_factory_selection() -> None:
    prompt = build_prompt(
        document=_build_document(),
        criteria=_policy_criteria(),
        section=_base_section(),
        dimension=_base_dimension(),
        question=_base_question(),
        chunk_text="La sección presenta compromisos estratégicos y fuentes de verificación.",
        chunk_metadata={},
    )

    assert "0 a 2 (pasos de 0.5)" in prompt
    assert "<!-- prompt_quality:" in prompt
    metadata_json = prompt.split("<!-- prompt_quality: ", 1)[1].rsplit(" -->", 1)[0]
    metadata = json.loads(metadata_json)
    assert metadata["report_type"].lower() == "politica_nacional"