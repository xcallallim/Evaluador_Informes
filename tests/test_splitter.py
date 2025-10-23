"""Pruebas del pipeline de división en *chunks* del Evaluador de Informes."""

from __future__ import annotations
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import threading
from typing import Dict

import pytest

#Asegúrese de que la raíz del repositorio esté en sys.path cuando el conjunto de pruebas se ejecute directamente.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from data.chunks import splitter as splitter_module
from data.chunks.splitter import Splitter
from data.models.document import Document
from data.preprocessing.cleaner import Cleaner
from data.preprocessing.loader import DocumentLoader
from data.preprocessing.segmenter import Segmenter


@pytest.fixture()
def sample_sections() -> Dict[str, str]:
    return {
        "introduccion": (
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
            "Vestibulum id ligula porta felis euismod semper."
        ),
        "desarrollo": (
            "Proin eget tortor risus. Nulla quis lorem ut libero malesuada feugiat. "
            "Curabitur non nulla sit amet nisl tempus convallis quis ac lectus."
        ),
    }


def _assert_chunk_metadata(
    chunk,
    *,
    expected_source: str,
    expected_count: int,
    expected_overlap: int,
) -> None:
    """Valida los metadatos generados por el splitter para un chunk."""

    metadata = chunk.metadata
    assert metadata["source_id"] == expected_source
    assert metadata["source_type"] == "section"
    assert metadata["chunk_index"] >= 1
    assert metadata["chunks_in_source"] == expected_count
    assert metadata["chunk_index"] <= metadata["chunks_in_source"]
    assert metadata["length"] == len(chunk.page_content)
    assert metadata["id"].startswith(f"{expected_source}_")
    assert metadata["chunk_overlap"] == expected_overlap
    assert "document_metadata" in metadata


def _assert_chunk_sequence_covers_section(
    section_text: str,
    chunks,
    *,
    expected_overlap: int,
) -> None:
    """Comprueba que los chunks cubren la sección completa respetando el solapamiento."""

    normalized_section = section_text.strip()
    assert normalized_section, "La sección de origen no puede estar vacía"
    assert chunks, "Se esperaban chunks para verificar la cobertura"

    prev_end = 0
    cursor = 0
    for idx, chunk in enumerate(chunks, start=1):
        chunk_text = chunk.page_content.strip()
        assert chunk_text, f"El chunk {idx} está vacío"

        search_start = max(0, cursor - expected_overlap)
        start = normalized_section.find(chunk_text, search_start)
        if start == -1:
            start = normalized_section.find(chunk_text)

        assert start != -1, f"El contenido del chunk {idx} no se encontró en la sección"
        end = start + len(chunk_text)

        if idx == 1:
            assert start == 0, "El primer chunk no comienza al inicio de la sección"
        else:
            assert start <= prev_end, "Se detectó un hueco entre chunks consecutivos"

        prev_end = max(prev_end, end)
        cursor = end

    assert normalized_section.endswith(
        chunks[-1].page_content.strip()
    ), "El último chunk no cubre el final de la sección"
    assert (
        prev_end >= len(normalized_section)
    ), "La combinación de chunks no cubre toda la sección"


def test_splitter_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        Splitter(chunk_size=100, chunk_overlap=100)

    with pytest.raises(ValueError):
        Splitter(chunk_size=0, chunk_overlap=0)

    with pytest.raises(ValueError):
        Splitter(chunk_size=10, chunk_overlap=-1)


def test_split_document_generates_metadata(sample_sections: Dict[str, str]) -> None:
    document = Document(metadata={"id": "doc-123"}, sections=sample_sections)
    splitter = Splitter(chunk_size=120, chunk_overlap=20)

    splitter.split_document(document)

    assert document.chunks, "Expected chunks to be generated from the sample sections"
    assert {chunk.metadata["source_id"] for chunk in document.chunks} == set(
        sample_sections.keys()
    )

    per_section_counts = Counter(chunk.metadata["source_id"] for chunk in document.chunks)
    for source, expected_count in per_section_counts.items():
        source_chunks = [c for c in document.chunks if c.metadata["source_id"] == source]
        assert len(source_chunks) == expected_count
        for chunk in source_chunks:
            _assert_chunk_metadata(
                chunk,
                expected_source=source,
                expected_count=expected_count,
                expected_overlap=splitter.chunk_overlap,
            )
            assert chunk.metadata["document_metadata"]["id"] == "doc-123"
            assert chunk.metadata["document_id"] == "doc-123"
            assert chunk.page_content.strip(), "Chunks should not be empty"


@pytest.mark.slow
@pytest.mark.integration
def test_full_pipeline_generates_expected_chunks() -> None:
    pdf_path = os.path.join(
        REPO_ROOT, "data", "inputs", "IEI 2023 - Gob. Regional de La Libertad.pdf"
    )
    if not os.path.exists(pdf_path):
        pytest.skip("El PDF de prueba no está disponible en el repositorio")

    loader = DocumentLoader()
    cleaner = Cleaner()
    segmenter = Segmenter()
    splitter = Splitter(chunk_size=1200, chunk_overlap=200)

    document = loader.load(pdf_path)
    document = cleaner.clean_document(document)
    document = segmenter.segment_document(document)
    splitter.split_document(document)

    expected_section_counts = {
        "resumen_ejecutivo": 3,
        "prioridades_politica_institucional": 38,
        "analisis_oei": 14,
        "analisis_productos_aei": 3,
        "analisis_ejecucion_operativa": 13,
        "aplicacion_recomendaciones": 0,
        "analisis_aei_oei": 5,
        "conclusiones": 5,
        "recomendaciones": 2,
        "anexos": 4,
        "analisis_implementacion_aei": 1,
        "diagnostico_oei_priorizados": 3,
    }

    assert set(document.sections.keys()) == set(expected_section_counts.keys())
    assert len(document.chunks) == sum(expected_section_counts.values())

    generated_counts = Counter(chunk.metadata["source_id"] for chunk in document.chunks)
    for section_id, expected in expected_section_counts.items():
        assert generated_counts.get(section_id, 0) == expected

    resumen_chunks = [
        chunk for chunk in document.chunks if chunk.metadata["source_id"] == "resumen_ejecutivo"
    ]
    assert len(resumen_chunks) == 3
    for index, chunk in enumerate(resumen_chunks, start=1):
        _assert_chunk_metadata(
            chunk,
            expected_source="resumen_ejecutivo",
            expected_count=3,
            expected_overlap=splitter.chunk_overlap,
        )
        assert chunk.metadata["chunk_index"] == index
        assert chunk.page_content.strip() in document.sections["resumen_ejecutivo"]


    # Validate that chunk ordering preserves the original sequence for the section.
    assert [chunk.metadata["id"] for chunk in resumen_chunks] == [
        "resumen_ejecutivo_1",
        "resumen_ejecutivo_2",
        "resumen_ejecutivo_3",
    ]

    _assert_chunk_sequence_covers_section(
        document.sections["resumen_ejecutivo"],
        resumen_chunks,
        expected_overlap=splitter.chunk_overlap,
    )

    # Ensure no chunk lost the base metadata propagated by the splitter.
    assert all("document_metadata" in chunk.metadata for chunk in document.chunks)


def test_split_content_map_streams_without_materialising(sample_sections: Dict[str, str]) -> None:
    splitter = Splitter(chunk_size=120, chunk_overlap=20)
    base_metadata = {"document_metadata": {"id": "doc-42"}}

    streamed = splitter._split_content_map(
        sample_sections,
        origin="section",
        base_metadata=base_metadata,
        stream=True,
    )

    assert isinstance(streamed, Iterator)
    assert not isinstance(streamed, list)

    streamed_chunks = list(streamed)
    eager_chunks = splitter._split_content_map(
        sample_sections,
        origin="section",
        base_metadata=base_metadata,
    )

    assert [chunk.page_content for chunk in streamed_chunks] == [
        chunk.page_content for chunk in eager_chunks
    ]
    assert [chunk.metadata["id"] for chunk in streamed_chunks] == [
        chunk.metadata["id"] for chunk in eager_chunks
    ]


def test_iter_document_chunks_is_lazy(sample_sections: Dict[str, str]) -> None:
    document = Document(metadata={"id": "doc-99"}, sections=sample_sections)
    splitter = Splitter(chunk_size=120, chunk_overlap=20)

    chunk_iterator = splitter.iter_document_chunks(document)

    assert isinstance(chunk_iterator, Iterator)
    assert not isinstance(chunk_iterator, list)
    assert document.chunks == []

    streamed_chunks = list(chunk_iterator)
    assert streamed_chunks, "Se esperaban chunks generados por el iterador"
    assert document.chunks == []

    materialised = splitter.split_document(document)
    assert materialised.chunks
    assert [chunk.metadata["id"] for chunk in streamed_chunks] == [
        chunk.metadata["id"] for chunk in materialised.chunks
    ]
    
if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))