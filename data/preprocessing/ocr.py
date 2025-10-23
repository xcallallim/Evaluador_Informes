"""Motor OCR modular y escalable para PDFs escaneados."""

from __future__ import annotations

import concurrent.futures
import os
import tempfile
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.logger import log_error, log_info, log_warn


def _safe_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        log_warn(f"Valor inválido para {name}: {value}. Usando {default}.")
        return default
    return parsed if parsed > 0 else default


DEFAULT_LANGUAGES = os.environ.get("OCR_LANGUAGES", "spa")
DEFAULT_DPI = _safe_int_env("OCR_DPI", 200)
DEFAULT_TESSERACT_CONFIG = os.environ.get("OCR_TESSERACT_CONFIG", "--psm 4")
_DEFAULT_MAX_WORKERS = max(1, min(4, os.cpu_count() or 1))
DEFAULT_MAX_WORKERS = _safe_int_env("OCR_MAX_WORKERS", _DEFAULT_MAX_WORKERS)
MAX_PIXELS = _safe_int_env("OCR_MAX_PIXELS", 35_000_000)
MAX_QUEUE_FACTOR = 4


@dataclass
class _OCRPageResult:
    index: int
    text: str
    error: Optional[str] = None


def _register_issue(issues: List[str], message: str) -> None:
    if not message or message in issues:
        return
    issues.append(message)


def _resolve_max_workers(custom: Optional[int]) -> int:
    if custom is None:
        return DEFAULT_MAX_WORKERS
    if custom <= 0:
        log_warn(
            f"Valor inválido para max_workers ({custom}). "
            f"Se usará {DEFAULT_MAX_WORKERS}."
        )
        return DEFAULT_MAX_WORKERS
    return custom


def perform_pdf_ocr(
    filepath: str,
    issues: List[str],
    *,
    languages: str = DEFAULT_LANGUAGES,
    dpi: int = DEFAULT_DPI,
    max_workers: Optional[int] = None,
    tess_config: str = DEFAULT_TESSERACT_CONFIG,
) -> Tuple[str, List[str]]:
    """Ejecuta OCR paralelo controlado sobre un PDF escaneado."""

    try:
        import fitz  # type: ignore
    except Exception as import_error:
        log_error(f"PyMuPDF es requerido para OCR: {import_error}")
        _register_issue(issues, "PyMuPDF no está disponible para ejecutar OCR.")
        return "", []

    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore
    except Exception as import_error:
        log_error(f"Dependencias OCR incompletas: {import_error}")
        _register_issue(issues, "No se pudo iniciar el OCR por dependencias faltantes.")
        return "", []

    if dpi <= 0:
        log_warn(f"DPI inválido ({dpi}). Se usará 200.")
        dpi = 200

    max_workers_resolved = _resolve_max_workers(max_workers)
    queue_limit = max(max_workers_resolved, max_workers_resolved * MAX_QUEUE_FACTOR)

    previous_tempdir = getattr(pytesseract.pytesseract, "TEMP_DIR", None)

    try:
        with tempfile.TemporaryDirectory(prefix="tess_tmp_") as tess_tempdir:
            pytesseract.pytesseract.TEMP_DIR = tess_tempdir

            with fitz.open(filepath) as pdf_document:
                total_pages = len(pdf_document)
                if total_pages == 0:
                    log_warn("El PDF no contiene páginas. Nada que OCR.")
                    return "", []

                log_info(
                    "🧠 Iniciando OCR paralelo: "
                    f"{total_pages} páginas con {max_workers_resolved} trabajadores..."
                )

                pages_text: List[str] = [""] * total_pages
                progress = {"done": 0}
                log_every = max(1, total_pages // 5)

                def _update_progress() -> None:
                    progress["done"] += 1
                    done = progress["done"]
                    if done == 1 or done % log_every == 0 or done == total_pages:
                        log_info(f"OCR progreso: {done}/{total_pages} páginas")

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_workers_resolved
                ) as executor:
                    pending: Dict[concurrent.futures.Future[_OCRPageResult], int] = {}

                    for index, page in enumerate(pdf_document):
                        render_result = _render_page_to_png(page, dpi)
                        if render_result is None:
                            _register_issue(
                                issues,
                                f"OCR no pudo renderizar la página {index + 1}.",
                            )
                            _update_progress()
                            continue

                        if render_result["pixels"] > MAX_PIXELS:
                            log_warn(
                                "Página "
                                f"{index + 1} supera el límite de pixeles ({render_result['pixels']} > {MAX_PIXELS}). "
                                "Se omite OCR."
                            )
                            _register_issue(
                                issues,
                                f"Página {index + 1} omitida por exceder el límite de tamaño.",
                            )
                            _update_progress()
                            continue

                        future = executor.submit(
                            _run_page_ocr,
                            index,
                            render_result["bytes"],
                            languages,
                            tess_config,
                            cv2,
                            np,
                            Image,
                            pytesseract,
                        )
                        pending[future] = index

                        if len(pending) >= queue_limit:
                            _consume_completed_futures(
                                pending,
                                pages_text,
                                issues,
                                _update_progress,
                            )

                    # Esperar el resto de tareas pendientes
                    while pending:
                        _consume_completed_futures(
                            pending,
                            pages_text,
                            issues,
                            _update_progress,
                            wait_all=len(pending) == 1,
                        )

                pages_with_tags = [
                    f"=== PAGE {i + 1} ===\n{page_text.strip()}"
                    for i, page_text in enumerate(pages_text)
                ]
                full_text = "\n\n".join(pages_with_tags)

                log_info("✅ OCR completado correctamente.")
                return full_text, pages_text

    except Exception as exc:
        log_error(f"❌ Error durante OCR: {exc}")
        _register_issue(issues, "El OCR no pudo procesar el PDF escaneado.")

        try:
            with fitz.open(filepath) as pdf_document:
                fallback_pages = [""] * len(pdf_document)
        except Exception:
            fallback_pages = []
        return "", fallback_pages

    finally:
        if previous_tempdir is not None:
            pytesseract.pytesseract.TEMP_DIR = previous_tempdir
        elif hasattr(pytesseract.pytesseract, "TEMP_DIR"):
            delattr(pytesseract.pytesseract, "TEMP_DIR")


def _render_page_to_png(page: Any, dpi: int) -> Optional[Dict[str, Any]]:
    try:
        pix = page.get_pixmap(dpi=dpi)
        pixels = pix.width * pix.height
        png_bytes = pix.tobytes("png")
        return {"bytes": png_bytes, "pixels": pixels}
    except Exception as exc:
        log_warn(f"No se pudo renderizar la página para OCR: {exc}")
        return None


def _consume_completed_futures(
    pending: Dict[concurrent.futures.Future[_OCRPageResult], int],
    pages_text: List[str],
    issues: List[str],
    progress_cb: Callable[[], None],
    *,
    wait_all: bool = False,
) -> None:
    wait_mode = (
        concurrent.futures.ALL_COMPLETED if wait_all else concurrent.futures.FIRST_COMPLETED
    )
    done, _ = concurrent.futures.wait(pending.keys(), return_when=wait_mode)
    for future in done:
        index = pending.pop(future, None)
        if index is None:
            continue
        try:
            result = future.result()
        except Exception as exc:
            log_warn(f"OCR falló en la página {index + 1}: {exc}")
            _register_issue(issues, f"OCR falló en la página {index + 1}.")
            pages_text[index] = ""
            progress_cb()
            continue

        if result.error:
            log_warn(f"OCR falló en la página {index + 1}: {result.error}")
            _register_issue(issues, f"OCR falló en la página {index + 1}.")
            pages_text[index] = ""
        else:
            pages_text[index] = result.text
        progress_cb()


def _run_page_ocr(
    index: int,
    image_bytes: bytes,
    languages: str,
    tess_config: str,
    cv2_mod: Any,
    np_mod: Any,
    pil_image: Any,
    pytesseract_mod: Any,
) -> _OCRPageResult:
    try:
        with pil_image.open(BytesIO(image_bytes)) as raw_image:
            gray_image = raw_image.convert("L")
            np_image = np_mod.array(gray_image)
            filtered = cv2_mod.bilateralFilter(np_image, 9, 75, 75)
            thresh = cv2_mod.adaptiveThreshold(
                filtered,
                255,
                cv2_mod.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2_mod.THRESH_BINARY,
                25,
                7,
            )
            processed = pil_image.fromarray(thresh)
            text = pytesseract_mod.image_to_string(
                processed,
                lang=languages,
                config=tess_config,
            )
        return _OCRPageResult(index=index, text=text.strip())
    except Exception as exc:  # pragma: no cover - errores dependientes del entorno
        return _OCRPageResult(index=index, text="", error=str(exc))