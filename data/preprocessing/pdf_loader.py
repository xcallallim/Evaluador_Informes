"""Carga especializada para documentos PDF (digitales u OCR)."""

from __future__ import annotations

import os
import re
import shutil
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, TYPE_CHECKING

from core.config import TESSERACT_PATH
from core.logger import log_info, log_warn
from core.warning_filters import suppress_external_deprecation_warnings
from core.utils import ensure_dir
from data.models.document import Document
from data.preprocessing.metadata import LoaderContext
from data.preprocessing.ocr import perform_pdf_ocr

suppress_external_deprecation_warnings()

try:  # pragma: no cover - dependencia opcional
    from core.config import GHOSTSCRIPT_PATH
except Exception:  # pragma: no cover - fallback limpio
    GHOSTSCRIPT_PATH = None  # type: ignore


if TYPE_CHECKING:  # pragma: no cover - anotaciones opcionales
    from PIL import Image  # type: ignore


MEANINGFUL_TEXT_PATTERN = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]")


class PDFResourceExporter(Protocol):
    """Define cómo se persisten las tablas e imágenes extraídas del PDF."""

    def save_table(
        self,
        stem: str,
        page_number: str,
        table_index: int,
        dataframe: Any,
    ) -> str:
        """Persiste la tabla y devuelve una referencia estable (p. ej. ruta o URL)."""

    def save_image(
        self,
        stem: str,
        page_number: int,
        image_index: int,
        image: "Image",
    ) -> str:
        """Persiste la imagen y devuelve una referencia estable (p. ej. ruta o URL)."""


class FileSystemPDFResourceExporter:
    """Persistencia por defecto utilizando el sistema de archivos local."""

    def __init__(self, base_dir: Optional[str] = None) -> None:
        self._base_dir = base_dir or os.path.join("data", "outputs")

    def _normalize_path(self, path: str) -> str:
        return path.replace("\\", "/")

    def save_table(
        self,
        stem: str,
        page_number: str,
        table_index: int,
        dataframe: Any,
    ) -> str:
        base_dir = os.path.join(self._base_dir, "tables", stem)
        ensure_dir(base_dir)
        filename = f"page_{page_number}_{table_index + 1}.csv"
        output_path = os.path.join(base_dir, filename)
        dataframe.to_csv(output_path, index=False)
        return self._normalize_path(output_path)

    def save_image(
        self,
        stem: str,
        page_number: int,
        image_index: int,
        image: "Image",
    ) -> str:
        safe_stem = _slugify_filename(stem)
        base_dir = os.path.join(self._base_dir, "images", safe_stem)
        ensure_dir(base_dir)
        filename = f"page_{page_number}_img_{image_index}.png"
        output_path = os.path.join(base_dir, filename)
        image.save(output_path, format="PNG")
        return self._normalize_path(output_path)


def configure_ocr() -> None:
    """Configura la ruta de Tesseract OCR si está disponible."""

    try:
        import pytesseract

        if TESSERACT_PATH and os.path.exists(TESSERACT_PATH):
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
            log_info(f"Tesseract configurado en: {TESSERACT_PATH}")
        else:
            log_warn("Ruta de Tesseract no encontrada o inválida. OCR deshabilitado.")
    except ImportError:  # pragma: no cover - entorno sin OCR
        log_warn("pytesseract no está instalado. OCR deshabilitado.")


def resolve_ghostscript() -> Optional[str]:
    """Localiza Ghostscript permitiendo binarios descargados sin privilegios sudo."""

    candidates = [
        os.environ.get("GHOSTSCRIPT_PATH"),
        os.environ.get("GS_BIN"),
        GHOSTSCRIPT_PATH,
    ]

    for executable in ("gs", "gswin64c", "gswin32c"):
        path = shutil.which(executable)
        if path:
            candidates.append(path)

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            log_info(f"Ghostscript detectado en: {candidate}")
            return candidate

    log_warn(
        "Ghostscript no está disponible. Camelot seguirá intentando, "
        "pero la extracción de tablas puede fallar sin este binario."
    )
    return None


def load(
    filepath: str,
    *,
    extract_tables: bool = True,
    extract_images: bool = True,
    ghostscript_cmd: Optional[str] = None,
    issues: Optional[List[str]] = None,
    detector: Optional[Callable[[str], Optional[Tuple[bool, List[str]]]]] = None,
    ocr_loader: Optional[Callable[[str], Tuple[str, List[str]]]] = None,
    resource_exporter: Optional[PDFResourceExporter] = None,
) -> Document:
    """Carga un PDF y devuelve un ``Document`` parcial listo para la fachada."""

    issues = issues if issues is not None else []
    tables_meta: Dict[str, Any] = {}
    flat_tables: List[Any] = []
    resource_exporter = resource_exporter or FileSystemPDFResourceExporter()

    content, pages, is_ocr, raw_text = _load_pdf(
        filepath,
        issues,
        detector=detector,
        ocr_loader=ocr_loader,
    )

    images_meta: List[Dict[str, Any]] = []
    if extract_tables:
        pdf_tables_meta = _extract_pdf_tables(
            filepath,
            ghostscript_cmd,
            issues,
            resource_exporter,
        )
        if pdf_tables_meta:
            tables_meta["pdf"] = pdf_tables_meta
            flat_tables.extend(pdf_tables_meta)

    if extract_images:
        images_meta = _extract_pdf_images(filepath, resource_exporter)

    extra_metadata = {
        "is_ocr": is_ocr,
        "extraction_method": "ocr" if is_ocr else "embedded_text",
        "raw_text": raw_text,
    }

    context = LoaderContext(
        tables_meta=tables_meta,
        issues=issues,
        images_meta=images_meta if extract_images else [],
        extra_metadata=extra_metadata,
    )

    metadata = {"loader_context": context.as_dict()}

    return Document(
        content=content,
        pages=pages,
        tables=flat_tables,
        metadata=metadata,
    )


def _register_issue(issues: List[str], message: str) -> None:
    if not message or message in issues:
        return
    issues.append(message)


def _load_pdf(
    filepath: str,
    issues: List[str],
    *,
    detector: Optional[Callable[[str], Optional[Tuple[bool, List[str]]]]] = None,
    ocr_loader: Optional[Callable[[str], Tuple[str, List[str]]]] = None,
) -> Tuple[str, List[str], bool, str]:
    """Carga el PDF detectando si es digital u OCR."""

    log_info("📄 Analizando PDF...")

    last_pdf_raw_text = ""
    last_pdf_was_ocr = False

    if detector:
        pymupdf_detection = detector(filepath)
    else:
        pymupdf_detection = _try_load_pdf_with_pymupdf(filepath)
    pymupdf_detected: Optional[bool] = None
    pymupdf_pages: List[str] = []

    if pymupdf_detection:
        pymupdf_detected, pymupdf_pages = pymupdf_detection

    if pymupdf_detected is False:
        log_warn("⚠ PDF escaneado detectado (PyMuPDF). Ejecutando OCR...")
        last_pdf_was_ocr = True
        if ocr_loader:
            ocr_full_text, pages_text = ocr_loader(filepath)
        else:
            ocr_full_text, pages_text = perform_pdf_ocr(filepath, issues)
        last_pdf_raw_text = ocr_full_text
        return "", pages_text, last_pdf_was_ocr, last_pdf_raw_text

    pages_text: List[str] = []
    is_digital = False
    pdfplumber_failed = False

    try:
        import pdfplumber  # type: ignore
    except Exception:
        log_warn(
            "pdfplumber no está disponible o falló la importación. "
            "Se procederá al OCR si no se detecta texto con otros métodos."
        )
    else:
        if pymupdf_detected is not False:
            try:
                with pdfplumber.open(filepath) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text() or ""
                        words = []
                        try:
                            words = page.extract_words() or []
                        except Exception:
                            words = []

                        if text.strip() or words:
                            is_digital = True

                        if not text.strip() and words:
                            text = " ".join(word.get("text", "") for word in words).strip()

                        if not text.strip() and not words:
                            chars = getattr(page, "chars", []) or []
                            if chars:
                                is_digital = True
                                sorted_chars = sorted(
                                    chars,
                                    key=lambda c: (round(c.get("top", 0.0), 1), c.get("x0", 0.0)),
                                )

                                from itertools import groupby

                                lines = []
                                for _, group in groupby(
                                    sorted_chars, key=lambda c: round(c.get("top", 0.0), 1)
                                ):
                                    line_text = "".join(char.get("text", "") for char in group)
                                    if line_text.strip():
                                        lines.append(line_text)

                                if lines:
                                    text = "\n".join(lines)

                        pages_text.append(text)
            except Exception as exc:
                log_warn(
                    "No se pudo leer como PDF digital con pdfplumber. "
                    f"Detalle: {exc}. Se procederá al OCR si no hay más opciones."
                )
                pages_text = []
                pdfplumber_failed = True

    if is_digital:
        log_info("✅ PDF digital detectado (pdfplumber)")
        pages_with_tags = [f"=== PAGE {i+1} ===\n{p.strip()}" for i, p in enumerate(pages_text)]
        full_text = "\n\n".join(pages_with_tags)
        last_pdf_raw_text = full_text
        return full_text, pages_text, False, last_pdf_raw_text

    if pymupdf_detected:
        meaningful_pages = [page for page in pymupdf_pages if page.strip()]
        if meaningful_pages:
            log_info("✅ PDF digital detectado (PyMuPDF fallback)")
            pages_with_tags = [
                f"=== PAGE {i+1} ===\n{p.strip()}" for i, p in enumerate(pymupdf_pages)
            ]
            full_text = "\n\n".join(pages_with_tags)
            last_pdf_raw_text = full_text
            return full_text, pymupdf_pages, False, last_pdf_raw_text

    if pdfplumber_failed:
        log_warn(
            "⚠ PDF escaneado detectado. Ejecutando OCR tras fallo de pdfplumber..."
        )
    else:
        log_warn("⚠ PDF escaneado detectado. Ejecutando OCR...")

    last_pdf_was_ocr = True
    if ocr_loader:
        ocr_full_text, pages_text = ocr_loader(filepath)
    else:
        ocr_full_text, pages_text = perform_pdf_ocr(filepath, issues)
    last_pdf_raw_text = ocr_full_text
    return "", pages_text, last_pdf_was_ocr, last_pdf_raw_text


def _try_load_pdf_with_pymupdf(filepath: str) -> Optional[Tuple[bool, List[str]]]:
    try:
        import fitz  # type: ignore
    except Exception:
        log_warn("PyMuPDF no está disponible para el fallback de PDFs digitales.")
        return None

    try:
        pages_text: List[str] = []
        pages_with_content = 0

        with fitz.open(filepath) as pdf_document:
            for page in pdf_document:
                text = _extract_meaningful_text_from_page(page)
                pages_text.append(text)
                if text.strip():
                    pages_with_content += 1

        is_digital = pages_with_content > 0
        return is_digital, pages_text

    except Exception as exc:
        log_warn(f"PyMuPDF no pudo analizar el PDF para detección rápida: {exc}")
        return None


def _extract_meaningful_text_from_page(page: Any) -> str:
    text = ""

    try:
        text = page.get_text("text", sort=True) or ""
        text = text.strip()
        if text and MEANINGFUL_TEXT_PATTERN.search(text):
            return text
    except Exception:
        text = ""

    try:
        text = page.get_text("blocks") or []
        if text:
            blocks = [block[4] for block in text if len(block) > 4]
            text = "\n".join(block.strip() for block in blocks if block.strip())
            if text and MEANINGFUL_TEXT_PATTERN.search(text):
                return text
    except Exception:
        text = ""

    if not text:
        try:
            fallback = page.get_text("text") or ""
            fallback = fallback.strip()
            if fallback and MEANINGFUL_TEXT_PATTERN.search(fallback):
                text = fallback
        except Exception:
            text = ""

    if not text or not text.strip():
        return ""

    return text


def _slugify_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_.")
    

def _extract_pdf_tables(
    filepath: str,
    ghostscript_cmd: Optional[str],
    issues: List[str],
    exporter: PDFResourceExporter,
) -> List[Dict[str, Any]]:
    tables_meta: List[Dict[str, Any]] = []

    try:
        import camelot  # type: ignore
    except Exception:
        log_warn("Camelot no está instalado. Saltando extracción de tablas PDF.")
        _register_issue(issues, "Camelot no está disponible; no se extrajeron tablas del PDF.")
        return tables_meta

    fname = os.path.basename(filepath)
    stem, _ = os.path.splitext(fname)
    valid_count = 0

    if ghostscript_cmd:
        try:
            log_info("Extrayendo tablas (Camelot - lattice)...")
            kwargs = {"flavor": "lattice", "pages": "all", "gs": ghostscript_cmd}
            tables = camelot.read_pdf(filepath, **kwargs)
            valid_count = _save_camelot_tables(
                tables,
                stem,
                tables_meta,
                issues,
                exporter,
            )
        except Exception as exc:
            log_warn(f"Fallo 'lattice': {exc}")
            _register_issue(issues, "Camelot falló con el modo lattice.")
            valid_count = 0
    else:
        log_warn(
            "Ghostscript no disponible. Se omite la extracción de tablas con Camelot."
        )
        _register_issue(
            issues,
            "Camelot no se ejecutó por falta de Ghostscript; no se extrajeron tablas del PDF.",
        )
        return tables_meta

    if valid_count == 0:
        try:
            log_info("Reintentando tablas (Camelot - stream)...")
            kwargs = {"flavor": "stream", "pages": "all"}
            if ghostscript_cmd:
                kwargs["gs"] = ghostscript_cmd
            tables = camelot.read_pdf(filepath, **kwargs)
            valid_count = _save_camelot_tables(
                tables,
                stem,
                tables_meta,
                issues,
                exporter,
            )
        except Exception as exc:
            log_warn(f"Fallo 'stream': {exc}")
            _register_issue(issues, "Camelot falló con el modo stream.")

    log_info(f"Tablas extraídas: {len(tables_meta)}")
    return tables_meta


def _save_camelot_tables(
    tables: Any,
    stem: str,
    tables_meta: List[Dict[str, Any]],
    issues: List[str],
    exporter: PDFResourceExporter,
) -> int:
    if not tables:
        return 0

    valid_count = 0

    for index, table in enumerate(tables):
        try:
            df = getattr(table, "df", None)
            if df is None or df.empty:
                continue

            if df.shape[0] < 2 or df.shape[1] < 2:
                continue

            page_number = getattr(table, "page", None)
            if page_number is None:
                parsing_report = getattr(table, "parsing_report", None)
                if isinstance(parsing_report, dict):
                    page_number = parsing_report.get("page")

            if not page_number:
                page_number = "unknown"

            out_path = exporter.save_table(stem, str(page_number), index, df)
            tables_meta.append(
                {
                    "page": page_number,
                    "path": out_path,
                }
            )
            valid_count += 1
        except Exception as exc:  # pragma: no cover
            log_warn(f"No se pudo guardar tabla {index + 1}: {exc}")
            _register_issue(
                issues, f"Fallo al exportar tabla {index + 1} detectada por Camelot"
            )

    return valid_count


def _extract_pdf_images(
    filepath: str,
    exporter: PDFResourceExporter,
) -> List[Dict[str, Any]]:
    images_meta: List[Dict[str, Any]] = []

    try:
        import fitz  # type: ignore
        from PIL import Image  # type: ignore
        from io import BytesIO

        fname = os.path.basename(filepath)
        stem, _ = os.path.splitext(fname)
        doc = fitz.open(filepath)
        for page_index in range(len(doc)):
            page = doc[page_index]
            img_list = page.get_images(full=True)

            img_counter = 0
            for img_info in img_list:
                xref = img_info[0]
                try:
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n > 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)

                    if (pix.width * pix.height) < 30000:
                        continue

                    img_counter += 1
                    img_bytes = pix.tobytes("png")
                    image = Image.open(BytesIO(img_bytes))

                    out_path = exporter.save_image(
                        stem,
                        page_index + 1,
                        img_counter,
                        image,
                    )
                    images_meta.append(
                        {
                            "page": page_index + 1,
                            "path": out_path,
                        }
                    )
                except Exception as exc:
                    log_warn(f"No se pudo extraer imagen (pág. {page_index + 1}): {exc}")
                    continue

        doc.close()

        log_info(f"🖼️ Imágenes extraídas: {len(images_meta)}")
        return images_meta

    except Exception as exc:
        log_warn(f"No se pudieron extraer imágenes del PDF: {exc}")
        return images_meta


def _load_pdf_ocr(
    filepath: str,
    issues: Optional[List[str]] = None,
    **kwargs: Any,
) -> Tuple[str, List[str]]:
    """Mantiene compatibilidad con clientes que importaban la función previa."""

    issues_list = issues if issues is not None else []
    return perform_pdf_ocr(filepath, issues_list, **kwargs)