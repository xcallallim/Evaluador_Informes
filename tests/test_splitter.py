# py tests/test_splitter.py
# py -m tests.test_splitter

import importlib
import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

pytest.importorskip("langchain")

Document = importlib.import_module("data.models.document").Document
Splitter = importlib.import_module("data.preprocessing.splitter").Splitter


@pytest.fixture()
def sample_sections():
    return {
        "introduccion": "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        * 20,
        "desarrollo": "Proin eget tortor risus. Nulla quis lorem ut libero malesuada "
        "feugiat. "
        * 10,
    }


def test_splitter_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        Splitter(chunk_size=100, chunk_overlap=100)

    with pytest.raises(ValueError):
        Splitter(chunk_size=0, chunk_overlap=0)

    with pytest.raises(ValueError):
        Splitter(chunk_size=10, chunk_overlap=-1)


def test_split_sections_creates_metadata(sample_sections):
    splitter = Splitter(chunk_size=200, chunk_overlap=50)

    chunks = splitter.split_sections(sample_sections)

    assert len(chunks) > 0
    assert {chunk.metadata["section"] for chunk in chunks} == set(
        sample_sections.keys()
    )
    assert all(len(chunk.page_content) <= splitter.chunk_size for chunk in chunks)

    for chunk in chunks:
        assert chunk.metadata["id"].startswith(f"{chunk.metadata['section']}_")
        assert chunk.metadata["chunk_index"] >= 1
        assert chunk.metadata["length"] == len(chunk.page_content)
        assert chunk.page_content.strip()


def test_split_document_populates_document_chunks(sample_sections):
    document = Document(content="", metadata={}, sections=sample_sections)
    splitter = Splitter(chunk_size=200, chunk_overlap=50)

    result = splitter.split_document(document)

    assert result is document
    assert document.chunks
    assert {chunk.metadata["section"] for chunk in document.chunks} == set(
        sample_sections.keys()
    )


def test_split_document_handles_missing_sections():
    document = Document(content="", metadata={})
    splitter = Splitter(chunk_size=100, chunk_overlap=20)

    splitter.split_document(document)

    assert document.chunks == []

