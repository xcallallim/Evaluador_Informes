"""Unit tests for :class:`DocumentLoader` with mocked dependencies."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from data.models.document import Document
from data.preprocessing import loader as loader_module
from data.preprocessing.loader import DocumentLoader
from data.preprocessing.metadata import DocumentSummary, LoaderContext
from data.preprocessing import docx_loader, pdf_loader, txt_loader


class DummyExporter:
    """Minimal PDF resource exporter used to avoid touching the filesystem."""

    def save_table(
        self,
        stem: str,
        page_number: str,
        table_index: int,
        dataframe: Any,
    ) -> str:
        return f"table://{stem}/{page_number}/{table_index}"

    def save_image(
        self,
        stem: str,
        page_number: int,
        image_index: int,
        image: Any,
    ) -> str:
        return f"image://{stem}/{page_number}/{image_index}"


@pytest.fixture
def patch_pdf_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralise heavy optional dependencies used by the PDF loader."""

    monkeypatch.setattr(pdf_loader, "configure_ocr", lambda: None)
    monkeypatch.setattr(pdf_loader, "resolve_ghostscript", lambda: "ghostscript-bin")
    monkeypatch.setattr(pdf_loader, "_load_pdf_ocr", lambda filepath, issues: ("ocr text", ["ocr page"]))
    monkeypatch.setattr(pdf_loader, "_try_load_pdf_with_pymupdf", lambda filepath: (True, []))


class RecordingEvents(loader_module.NullLoaderEventHandler):
    """Captura eventos emitidos por :class:`DocumentLoader` durante las pruebas."""

    def __init__(self) -> None:
        self.started: List[str] = []
        self.summaries: List[DocumentSummary] = []
        self.failures: List[Tuple[str, Exception]] = []

    def on_load_started(self, filepath: str) -> None:
        self.started.append(filepath)

    def on_load_succeeded(
        self,
        filepath: str,
        document: Document,
        summary: DocumentSummary,
    ) -> None:
        self.summaries.append(summary)

    def on_load_failed(self, filepath: str, error: Exception) -> None:
        self.failures.append((filepath, error))


@pytest.fixture
def event_recorder() -> RecordingEvents:
    return RecordingEvents()


@pytest.fixture
def loader(
    patch_pdf_dependencies: None, event_recorder: RecordingEvents
) -> DocumentLoader:
    """Return a DocumentLoader configured with the lightweight exporter."""

    return DocumentLoader(resource_exporter=DummyExporter(), events=event_recorder)


def test_load_pdf_digital_compiles_metadata(
    monkeypatch: pytest.MonkeyPatch,
    loader: DocumentLoader,
    event_recorder: RecordingEvents,
    tmp_path: Path,
) -> None:

    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    pdf_call: Dict[str, Any] = {}

    def fake_pdf_load(
        filepath: str,
        *,
        extract_tables: bool,
        extract_images: bool,
        ghostscript_cmd: str,
        issues: List[str],
        detector,
        ocr_loader,
        resource_exporter,
    ) -> Document:
        pdf_call.update(
            filepath=filepath,
            extract_tables=extract_tables,
            extract_images=extract_images,
            ghostscript_cmd=ghostscript_cmd,
            issues=list(issues),
            resource_exporter=resource_exporter,
        )

        # Exercise the dynamically provided OCR callable to ensure it is wired correctly.
        ocr_text, ocr_pages = ocr_loader(filepath)
        assert ocr_text == "ocr text"
        assert ocr_pages == ["ocr page"]

        loader_context = LoaderContext(
            tables_meta={"pdf": [{"id": "t1"}]},
            issues=["tabla sin encabezado"],
            images_meta=[{"path": "image-1.png"}],
            extra_metadata={
                "is_ocr": False,
                "extraction_method": "embedded_text",
                "raw_text": "Texto completo",
                "language": "es",
            },
        )
        metadata = {"loader_context": loader_context.as_dict(), "source_hint": "pdf"}
        return Document(
            content="Contenido digital",
            pages=["Página 1"],
            tables=[{"table": "T1"}],
            images=["imagen original"],
            metadata=metadata,
        )

    monkeypatch.setattr(pdf_loader, "load", fake_pdf_load)

    document = loader.load(str(pdf_path), extract_tables=True, extract_images=True)

    assert pdf_call == {
        "filepath": str(pdf_path),
        "extract_tables": True,
        "extract_images": True,
        "ghostscript_cmd": "ghostscript-bin",
        "issues": [],
        "resource_exporter": loader._resource_exporter,
    }

    assert document.content == "Contenido digital"
    assert document.pages == ["Página 1"]
    assert document.tables == [{"table": "T1"}]

    metadata = document.metadata
    assert metadata["filename"] == pdf_path.name
    assert metadata["extension"] == ".pdf"
    assert metadata["source"] == str(pdf_path)
    assert metadata["processed_with"] == "DocumentLoader"
    assert metadata["tables"] == {"pdf": [{"id": "t1"}]}
    assert metadata["issues"] == ["tabla sin encabezado"]
    assert metadata["raw_text"] == "Texto completo"
    assert metadata["is_ocr"] is False
    assert metadata["extraction_method"] == "embedded_text"
    assert metadata["language"] == "es"
    assert metadata["source_hint"] == "pdf"
    assert metadata["images"] == [{"path": "image-1.png"}]
    assert document.images == [{"path": "image-1.png"}]

    assert len(event_recorder.summaries) == 1
    summary = event_recorder.summaries[0]
    assert summary.pages == ["Página 1"]
    assert summary.tables == {"pdf": [{"id": "t1"}]}
    assert summary.images == [{"path": "image-1.png"}]
    assert summary.issues == ["tabla sin encabezado"]


def test_load_pdf_ocr_preserves_partial_images_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
    loader: DocumentLoader,
    event_recorder: RecordingEvents,
    tmp_path: Path,
) -> None:
    """When images are disabled the partial images should be kept as-is."""

    pdf_path = tmp_path / "scanned.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")


    def fake_pdf_load(
        filepath: str,
        *,
        extract_tables: bool,
        extract_images: bool,
        ghostscript_cmd: str,
        issues: List[str],
        detector,
        ocr_loader,
        resource_exporter,
    ) -> Document:
        assert extract_tables is False
        assert extract_images is False
        assert ghostscript_cmd == "ghostscript-bin"

        loader_context = LoaderContext(
            tables_meta={},
            issues=["ocr fallback"],
            images_meta=[{"path": "ctx-image.png"}],
            extra_metadata={
                "is_ocr": True,
                "extraction_method": "ocr",
                "raw_text": "Texto OCR",
            },
        )
        metadata = {"loader_context": loader_context.as_dict()}
        return Document(
            content="",
            pages=["Página OCR"],
            tables=[],
            images=["imagen parcial"],
            metadata=metadata,
        )

    monkeypatch.setattr(pdf_loader, "load", fake_pdf_load)

    document = loader.load(str(pdf_path), extract_tables=False, extract_images=False)

    assert document.metadata["pages"] == ["Página OCR"]
    assert document.metadata["issues"] == ["ocr fallback"]
    assert document.metadata["raw_text"] == "Texto OCR"
    assert document.metadata["is_ocr"] is True
    assert "images" not in document.metadata
    assert document.images == ["imagen parcial"]

    summary = event_recorder.summaries[0]
    assert summary.pages == ["Página OCR"]
    assert summary.tables == {}
    assert summary.images is None
    assert summary.issues == ["ocr fallback"]


def test_load_docx_delegates_and_respects_extract_tables(
    monkeypatch: pytest.MonkeyPatch,
    loader: DocumentLoader,
    event_recorder: RecordingEvents,
    tmp_path: Path,
) -> None:
    """DOCX files must forward the ``extract_tables`` flag to the specialised loader."""

    docx_path = tmp_path / "sample.docx"
    docx_path.write_text("dummy", encoding="utf-8")

    called: Dict[str, Any] = {}

    def fake_docx_load(filepath: str, *, extract_tables: bool) -> Document:
        called["args"] = (filepath, extract_tables)
        loader_context = LoaderContext(
            tables_meta={"docx": [["A", "B"]]},
            issues=["tabla sin cabecera"],
            images_meta=[],
            extra_metadata={"language": "es"},
        )
        metadata = {"loader_context": loader_context.as_dict()}
        return Document(
            content="Texto DOCX",
            pages=["Página 1"],
            tables=[["A", "B"]],
            metadata=metadata,
        )

    monkeypatch.setattr(docx_loader, "load", fake_docx_load)

    document = loader.load(str(docx_path), extract_tables=False, extract_images=False)

    assert called["args"] == (str(docx_path), False)
    assert document.metadata["extension"] == ".docx"
    assert document.metadata["tables"] == {"docx": [["A", "B"]]}
    assert document.metadata["language"] == "es"
    assert document.metadata["issues"] == ["tabla sin cabecera"]
    assert "images" not in document.metadata

    summary = event_recorder.summaries[0]
    assert summary.pages == ["Página 1"]
    assert summary.tables == {"docx": [["A", "B"]]}
    assert summary.images is None
    assert summary.issues == ["tabla sin cabecera"]


def test_load_txt_merges_extra_metadata(
    monkeypatch: pytest.MonkeyPatch,
    loader: DocumentLoader,
    event_recorder: RecordingEvents,
    tmp_path: Path,
) -> None:
    """TXT files should also benefit from loader context metadata."""

    txt_path = tmp_path / "notes.txt"
    txt_path.write_text("contenido", encoding="utf-8")


    def fake_txt_load(filepath: str) -> Document:
        assert filepath == str(txt_path)
        loader_context = LoaderContext(
            tables_meta={},
            issues=[],
            images_meta=[],
            extra_metadata={"language": "es", "detected_encoding": "utf-8"},
        )
        metadata = {"loader_context": loader_context.as_dict()}
        return Document(content="Contenido TXT", pages=["Contenido TXT"], metadata=metadata)

    monkeypatch.setattr(txt_loader, "load", fake_txt_load)

    document = loader.load(str(txt_path), extract_tables=True, extract_images=True)

    assert document.content == "Contenido TXT"
    assert document.metadata["extension"] == ".txt"
    assert document.metadata["language"] == "es"
    assert document.metadata["detected_encoding"] == "utf-8"
    assert document.metadata["tables"] == {}
    assert document.metadata["pages"] == ["Contenido TXT"]

    summary = event_recorder.summaries[0]
    assert summary.pages == ["Contenido TXT"]
    assert summary.tables == {}
    assert summary.images == []
    assert summary.issues == []


def test_load_missing_file_notifies_failure(
    loader: DocumentLoader, event_recorder: RecordingEvents, tmp_path: Path
) -> None:
    """Missing files must raise ``FileNotFoundError`` and notify listeners."""

    missing_path = tmp_path / "ghost.pdf"

    with pytest.raises(FileNotFoundError):
        loader.load(str(missing_path))

    assert len(event_recorder.failures) == 1
    failure_path, failure_error = event_recorder.failures[0]
    assert failure_path == str(missing_path)
    assert isinstance(failure_error, FileNotFoundError)
    assert str(failure_error) == f"Archivo no encontrado: {missing_path}"
    assert loader.failures == 1
    traces = loader.failure_traces
    assert len(traces) == 1
    assert traces[0] == {
        "filepath": str(missing_path),
        "error_type": "FileNotFoundError",
        "message": f"Archivo no encontrado: {missing_path}",
    }


def test_load_unsupported_extension_notifies_failure(
    loader: DocumentLoader, event_recorder: RecordingEvents, tmp_path: Path
) -> None:
    """Unsupported file types should notify listeners about the failure."""

    path = tmp_path / "notes.md"
    path.write_text("contenido", encoding="utf-8")

    with pytest.raises(ValueError, match="Formato no soportado: .md"):
        loader.load(str(path))

    assert len(event_recorder.failures) == 1
    failure_path, failure_error = event_recorder.failures[0]
    assert failure_path == str(path)
    assert isinstance(failure_error, ValueError)
    assert str(failure_error) == "Formato no soportado: .md"
    assert loader.failures == 1
    traces = loader.failure_traces
    assert len(traces) == 1
    assert traces[0] == {
        "filepath": str(path),
        "error_type": "ValueError",
        "message": "Formato no soportado: .md",
    }


def test_supported_extensions_are_sorted(loader: DocumentLoader) -> None:
    """The loader should expose its supported extensions in alphabetical order."""

    assert loader.supported_extensions == [".docx", ".pdf", ".txt"]

# pytest tests/test_document_loader.py -v