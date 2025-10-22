# data/preprocessing/loader.py
# ------------------------------------------------------------
# DocumentLoader: Carga TXT, DOCX, PDF (digital + OCR)
# - DOCX: extrae texto y tablas
# - PDF digital: extrae texto con pdfplumber
# - PDF escaneado: extrae texto con OCR (Tesseract + OpenCV)
# - PDF tablas: extrae tablas con Camelot (opcional) y exporta a CSV
# - Metadatos consistentes y listos para el pipeline
# ------------------------------------------------------------

import os, re, shutil
from typing import List, Tuple, Optional, Dict, Any

from core.logger import log_info, log_warn, log_error
from core.config import TESSERACT_PATH
from core.utils import clean_spaces, ensure_dir
from data.models.document import Document

# No hacemos "from core.config import *" para mantener claridad.
try:
    from core.config import GHOSTSCRIPT_PATH
except Exception:
    GHOSTSCRIPT_PATH = None


class DocumentLoader:
    def __init__(self):
        # Extensiones soportadas
        self.supported_extensions = [".pdf", ".docx", ".txt"]
        # Configurar Tesseract si está disponible
        self._configure_tesseract()
        # Resolver ruta de Ghostscript si el entorno no permite instalación global.
        self.ghostscript_cmd = self._resolve_ghostscript()

    # ------------------------------------------------------------
    # Configuración OCR
    # ------------------------------------------------------------
    def _configure_tesseract(self):
        """Configura la ruta de Tesseract OCR si está disponible."""
        try:
            import pytesseract
            if TESSERACT_PATH and os.path.exists(TESSERACT_PATH):
                pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
                log_info(f"Tesseract configurado en: {TESSERACT_PATH}")
            else:
                log_warn("Ruta de Tesseract no encontrada o inválida. OCR deshabilitado.")
        except ImportError:
            log_warn("pytesseract no está instalado. OCR deshabilitado.")
    
    def _resolve_ghostscript(self) -> Optional[str]:
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

    # ------------------------------------------------------------
    # API principal
    # ------------------------------------------------------------
    def load(self, filepath: str, extract_tables: bool = True, extract_images: bool = True) -> Document:

        """
        Carga un archivo y devuelve un objeto Document.

        Args:
            filepath: ruta del archivo a cargar.
            extract_tables: si True, intenta extraer tablas (DOCX y PDF con Camelot).
                            Esta opción es ideal para habilitar/deshabilitar desde una UI.
        """
        if not os.path.isfile(filepath):
            log_error(f"Archivo no encontrado: {filepath}")
            raise FileNotFoundError(f"No existe el archivo: {filepath}")

        _, ext = os.path.splitext(filepath)
        ext = ext.lower()

        if ext not in self.supported_extensions:
            log_error(f"Formato no soportado: {ext}")
            raise ValueError(f"Formato no soportado: {ext}")

        log_info(f"Cargando archivo: {filepath}")

        # Estructura de tablas en metadata (multi-origen)
        tables_meta: Dict[str, Any] = {}
        flat_tables: List[Any] = []

        if ext == ".txt":
            content, pages, txt_tables = self._load_txt(filepath)
            if extract_tables and txt_tables:
                tables_meta["txt"] = txt_tables
                flat_tables.extend(txt_tables)

        elif ext == ".docx":
            content, pages, docx_tables = self._load_docx(filepath)
            if extract_tables and docx_tables:
                tables_meta["docx"] = docx_tables
                flat_tables.extend(docx_tables)

        elif ext == ".pdf":
            content, pages = self._load_pdf(filepath)

            if extract_tables:
                pdf_tables_meta = self._extract_pdf_tables(filepath)
                if pdf_tables_meta:
                    tables_meta["pdf"] = pdf_tables_meta
                    flat_tables.extend(pdf_tables_meta)

            if extract_images:
                pdf_images_meta = self._extract_pdf_images(filepath)

        else:
            # Por seguridad; no debería alcanzarse por validación previa.
            raise NotImplementedError(f"Carga para {ext} aún no implementada.")

        metadata = {
            "filename": os.path.basename(filepath),
            "extension": ext,
            "source": filepath,
            "pages": pages,             # lista de textos por página (PDF) o None
            "tables": tables_meta or {},# {"docx": [...], "pdf": [{"page": int, "path": str}, ...]}
            "processed_with": "DocumentLoader"
        }

        # Añadir imágenes al metadata
        if extract_images:
            metadata["images"] = pdf_images_meta if 'pdf_images_meta' in locals() else []

        # --- LOG RESUMEN DE CARGA ---
        log_info("📊 Resumen de carga del documento:")
        log_info(f"   • Páginas detectadas: {len(pages) if pages else 'N/A'}")
        log_info(f"   • Tablas extraídas: {len(tables_meta.get('pdf', [])) + len(tables_meta.get('docx', []))}")
        log_info(f"   • Imágenes extraídas: {len(metadata.get('images', [])) if extract_images else 0}")
        log_info("✅ Documento cargado correctamente ✅")

        # Devolvemos Document con texto completo + metadata rica
        return Document(
            content=content,
            metadata=metadata,
            pages=pages,
            tables=flat_tables,
        )

    def _split_text_into_pages(self, text: str, max_chars: int = 4000) -> List[str]:
        """
        Divide texto largo en 'páginas' aproximadas si no hay saltos naturales.
        Solo como fallback para TXT simples.
        """
        return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
    
    def _slugify_filename(self, name: str) -> str:
        """
        Convierte un nombre a un 'slug' seguro para carpetas/archivos.
        Temporal aquí; luego podemos moverlo a core/utils.py.
        """
        # Normaliza, quita tildes y caracteres raros
        import unicodedata
        name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
        name = name.lower()
        name = re.sub(r"[^a-z0-9]+", "_", name)
        return name.strip("_")


    # ------------------------------------------------------------
    # TXT
    # ------------------------------------------------------------
    def _load_txt(self, filepath: str) -> Tuple[str, List[str], List]:
        """
        Carga archivos .txt conservando estructura
        Devuelve: texto completo, lista de páginas y tablas vacías (TXT no tiene tablas).
        """
        log_info("📄 Leyendo archivo TXT...")

        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Dividir contenido en páginas si hay señales de salto
            pages = re.split(r"\f|\n=== PAGE \d+ ===", content)
            if len(pages) <= 1:
                # Si no hay saltos claros, intenta dividir por longitud como fallback
                pages = self._split_text_into_pages(content, max_chars=4000)

            # Añadir separador de página visible
            pages_with_tags = [f"=== PAGE {i+1} ===\n{p.strip()}" for i, p in enumerate(pages)]
            full_text_with_pages = "\n\n".join(pages_with_tags)

            return full_text_with_pages, pages, []  # ← importante: [] para tablas

        except Exception as e:
            log_error(f"❌ Error leyendo archivo TXT: {e}")
            raise

    # ------------------------------------------------------------
    # DOCX (texto + tablas)
    # ------------------------------------------------------------
    def _load_docx(self, filepath: str) -> Tuple[str, List[str], List]:
        """
        Carga texto y tablas desde un archivo Word (.docx),
        respetando saltos de página visibles y extrayendo tablas.
        Devuelve:
        - texto completo
        - lista de páginas (si no encuentra marcas de página, usa fallback simple)
        - tablas en estructura FILAS x COLUMNAS
        """
        log_info("📄 Cargando archivo DOCX...")

        try:
            import docx
            doc = docx.Document(filepath)

            # Extraer texto respetando párrafos
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            raw_text = "\n".join(paragraphs)

            # Detectar saltos de página utilizando marcadores explícitos de Word
            pages = []
            current_page: List[str] = []

            def _flush_page() -> None:
                if current_page:
                    pages.append("\n".join(current_page))
                    current_page.clear()
            
            for paragraph in doc.paragraphs:
                text = paragraph.text.strip()
                if text:
                    current_page.append(text)

                element = getattr(paragraph, "_p", None)
                if element is None:
                    continue

                nsmap = getattr(element, "nsmap", {}) or {}
                w_namespace = nsmap.get("w")
                if not w_namespace:
                    continue

                # Buscar cualquier salto de página declarado en el XML del párrafo
                for br in element.findall(f".//{{{w_namespace}}}br"):
                    br_type = br.get(f"{{{w_namespace}}}type")
                    if br_type == "page":
                        _flush_page()
                        break

            _flush_page()

            # Fallback: si no detectó páginas correctamente
            if not pages:
                pages = self._split_text_into_pages(raw_text, max_chars=4000)

            # Insertar separadores visibles
            pages_with_tags = [f"=== PAGE {i+1} ===\n{p}" for i, p in enumerate(pages)]
            full_text_with_pages = "\n\n".join(pages_with_tags)

            # Extraer tablas
            tables = []
            for table in doc.tables:
                table_data = []
                for row in table.rows:
                    row_cells = [cell.text.strip() for cell in row.cells]
                    if any(row_cells):
                        table_data.append(row_cells)
                if table_data:
                    tables.append(table_data)

            log_info(f"📄 DOCX cargado correctamente: {len(pages)} páginas simuladas y {len(tables)} tablas detectadas.")
            return full_text_with_pages, pages, tables

        except Exception as e:
            log_error(f"❌ Error cargando DOCX: {e}")
            raise


    # ------------------------------------------------------------
    # PDF (detección digital vs escaneado + OCR preciso)
    # ------------------------------------------------------------
    def _load_pdf(self, filepath: str) -> Tuple[str, List[str]]:
        """
        Detecta si el PDF es digital (texto embebido) o escaneado (imágenes),
        y aplica la extracción adecuada.
        Devuelve:
        - full_text: texto completo con separadores de página
        - pages: lista con texto de cada página
        """
        log_info("📄 Analizando PDF...")

        pages_text: List[str] = []
        is_digital = False

        # 1) Intentar PDF digital con pdfplumber
        try:
            import pdfplumber
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text and text.strip():
                        is_digital = True
                    # Respetar saltos tal cual vienen (limpieza ocurrirá en Cleaner)
                    pages_text.append(text if text else "")
        except Exception as e:
            log_warn(f"No se pudo leer como PDF digital. Se intentará OCR. Detalle: {e}")
            pages_text = []

        # 2) PDF DIGITAL → devolver
        if is_digital:
            log_info("✅ PDF digital detectado")
            pages_with_tags = [f"=== PAGE {i+1} ===\n{p.strip()}" for i, p in enumerate(pages_text)]
            full_text = "\n\n".join(pages_with_tags)
            return full_text, pages_text

        # 3) Sino → usar OCR
        log_warn("⚠ PDF escaneado detectado. Ejecutando OCR...")
        return self._load_pdf_ocr(filepath)


    def _load_pdf_ocr(self, filepath: str) -> Tuple[str, List[str]]:
        """
        OCR de PDF escaneado con PyMuPDF + OpenCV + Tesseract.
        Perfil 'limpio' (rápido, conserva estructura). Manejo básico de columnas con psm=4.
        """
        import fitz  # PyMuPDF
        import cv2
        import numpy as np
        import pytesseract

        pages_text: List[str] = []
        log_info("🧠 Iniciando OCR página por página...")

        try:
            pdf_document = fitz.open(filepath)
            total = len(pdf_document)

            for i, page in enumerate(pdf_document):
                if i % 5 == 0:
                    log_info(f"OCR procesando página {i+1}/{total}")

                # Render a imagen a 200 dpi (suele ser suficiente)
                pix = page.get_pixmap(dpi=200)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)

                # Preproceso básico para OCR
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.bilateralFilter(gray, 11, 17, 17)
                thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)

                # OCR en español, psm=4 respeta bloques (funciona aceptable con doble columna)
                text = pytesseract.image_to_string(thresh, lang="spa", config="--psm 4")
                pages_text.append(text.strip())

            pdf_document.close()

            # Concatenar con separadores de página
            pages_with_tags = [f"=== PAGE {i+1} ===\n{p}" for i, p in enumerate(pages_text)]
            full_text = "\n\n".join(pages_with_tags)

            log_info("✅ OCR completado correctamente.")
            return full_text, pages_text

        except Exception as e:
            log_error(f"❌ Error durante OCR: {e}")
            return "", []



    # ------------------------------------------------------------
    # Tablas PDF con Camelot (export a CSV por documento)
    # ------------------------------------------------------------
    def _extract_pdf_tables(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Extrae tablas de un PDF usando Camelot.
        - Usa Ghostscript si está configurado (GHOSTSCRIPT_PATH).
        - Intenta primero 'lattice' (tablas con líneas) y luego 'stream' (tablas sin líneas).
        - Filtra tablas válidas (>= 2 filas y >= 2 columnas).
        - Exporta cada tabla a CSV en: data/outputs/tables/<nombre_doc>/page_<n>_<idx>.csv
        - Devuelve lista de dicts: [{"page": int, "path": str}, ...]
        """
        tables_meta: List[Dict[str, Any]] = []

        # 0) Validar Camelot instalado
        try:
            import camelot  # type: ignore
        except Exception:
            log_warn("Camelot no está instalado. Saltando extracción de tablas PDF.")
            return tables_meta

        # 1) Construir carpeta de salida: data/outputs/tables/<nombre_sin_ext>/
        fname = os.path.basename(filepath)
        stem, _ = os.path.splitext(fname)
        base_dir = os.path.join("data", "outputs", "tables", stem)
        ensure_dir(base_dir)

        # 2) Intentar extracción con 'lattice' (mejor para tablas con bordes marcados)
        try:
            log_info("Extrayendo tablas (Camelot - lattice)...")
            kwargs = {"flavor": "lattice", "pages": "all"}
            if self.ghostscript_cmd:
                kwargs["gs"] = self.ghostscript_cmd

            tables = camelot.read_pdf(filepath, **kwargs)
            valid_count = self._save_camelot_tables(tables, base_dir, tables_meta)
        except Exception as e:
            log_warn(f"Fallo 'lattice': {e}")
            valid_count = 0

        # 3) Si no encontró nada útil, intentamos 'stream' (detecta columnas por espacios)
        if valid_count == 0:
            try:
                log_info("Reintentando tablas (Camelot - stream)...")
                kwargs = {"flavor": "stream", "pages": "all"}
                if self.ghostscript_cmd:
                    kwargs["gs"] = self.ghostscript_cmd

                tables = camelot.read_pdf(filepath, **kwargs)
                valid_count = self._save_camelot_tables(tables, base_dir, tables_meta)
            except Exception as e:
                log_warn(f"Fallo 'stream': {e}")

        log_info(f"Tablas extraídas: {len(tables_meta)}")
        return tables_meta
    
    # ------------------
    # Imágenes PDF 
    # ------------------

    def _extract_pdf_images(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Extrae imágenes grandes desde un PDF y las guarda en:
        data/outputs/images/<nombre_pdf_limpio>/page_<n>_img_<k>.png

        Reglas:
        - Solo guarda imágenes "útiles" (área > 30,000 px)
        - Usa nombres/carpetas 'seguros' (sin tildes ni símbolos raros)
        - Devuelve metadata: [{"page": int, "path": str}, ...]
        """
        images_meta: List[Dict[str, Any]] = []

        try:
            import fitz  # PyMuPDF
            from PIL import Image
            from io import BytesIO

            # 1) Carpeta base por PDF con nombre seguro
            fname = os.path.basename(filepath)
            stem, _ = os.path.splitext(fname)
            safe_stem = self._slugify_filename(stem)  # ⬅︎ util local (abajo)
            base_dir = os.path.join("data", "outputs", "images", safe_stem)
            ensure_dir(base_dir)

            # 2) Abrir PDF y recorrer páginas
            doc = fitz.open(filepath)
            for page_index in range(len(doc)):
                page = doc[page_index]
                img_list = page.get_images(full=True)  # [(xref, smth...), ...]

                img_counter = 0
                for img_info in img_list:
                    xref = img_info[0]
                    try:
                        pix = fitz.Pixmap(doc, xref)
                        # Convertir a RGB si es CMYK/GRAYA/etc
                        if pix.n > 4:  # CMYK u otros
                            pix = fitz.Pixmap(fitz.csRGB, pix)

                        # Filtro por tamaño (área > 30k px)
                        if (pix.width * pix.height) < 30000:
                            continue

                        img_counter += 1
                        img_bytes = pix.tobytes("png")
                        img = Image.open(BytesIO(img_bytes))

                        out_name = f"page_{page_index+1}_img_{img_counter}.png"
                        out_path = os.path.join(base_dir, out_name)
                        img.save(out_path, format="PNG")

                        images_meta.append({
                            "page": page_index + 1,
                            "path": out_path.replace("\\", "/")
                        })

                    except Exception as e:
                        log_warn(f"No se pudo extraer imagen (pág. {page_index+1}): {e}")
                        continue

            doc.close()

            log_info(f"🖼️ Imágenes extraídas: {len(images_meta)}")
            return images_meta

        except Exception as e:
            log_warn(f"No se pudieron extraer imágenes del PDF: {e}")
            return images_meta

