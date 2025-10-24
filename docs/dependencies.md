# Dependencias externas sin privilegios de superusuario

Algunos componentes del pipeline aprovechan binarios del sistema que no se distribuyen
con `pip`. Si trabajas en un entorno donde `sudo` está deshabilitado, puedes usar las
siguientes alternativas para mantener el flujo operativo.

## Ghostscript (tablas PDF con Camelot)

Camelot necesita Ghostscript para obtener buenos resultados con el modo `lattice`.
Cuando no puedes instalarlo globalmente:

1. Descarga el binario portátil desde <https://ghostscript.com/releases/index.html>.
   * En Windows descarga el instalador `.exe` y extrae su contenido con 7-Zip.
   * En Linux/macOS descarga el `.tar.gz`, descomprímelo en una carpeta de tu usuario
     (por ejemplo `~/ghostscript`).
2. Obtén la ruta absoluta al ejecutable (`gs`, `gswin64c.exe`, etc.).
3. Exporta la variable de entorno antes de lanzar el pipeline:

   ```bash
   export GHOSTSCRIPT_PATH="$HOME/ghostscript/bin/gs"
   ```

   En Windows (PowerShell):

   ```powershell
   $env:GHOSTSCRIPT_PATH = "C:\\ruta\\a\\gswin64c.exe"
   ```

4. El `DocumentLoader` detectará automáticamente esa ruta gracias a `core/config.py` y
   usará el binario descargado al llamar a Camelot.

Si prefieres no descargar Ghostscript, puedes deshabilitar la extracción de tablas al
invocar el loader con `extract_tables=False`.

## Tesseract OCR

El OCR es opcional. Si necesitas habilitarlo sin privilegios de administrador:

1. Descarga la versión portátil apropiada para tu plataforma (por ejemplo desde
   <https://github.com/UB-Mannheim/tesseract/wiki> en Windows o usando un paquete
   `tar.gz` en Linux/macOS).
2. Define la variable de entorno `TESSERACT_PATH` apuntando al ejecutable `tesseract`.
3. Reinicia el proceso para que la configuración sea tomada por `DocumentLoader`.

Si no estableces la variable, el loader continuará pero registrará que el OCR está
inactivo.

## Dependencias de OpenCV

El proyecto utiliza `opencv-python-headless`, por lo que no requiere `libGL` ni
bibliotecas gráficas adicionales. Si aún así encuentras un error mencionando
`libGL.so.1`, verifica que no tengas otra variante de OpenCV instalada en tu entorno
(`pip show opencv-python`). Desinstálala y conserva únicamente la versión "headless".

---

Al centralizar las rutas en variables de entorno, puedes trabajar en entornos sin
`sudo` manteniendo las funcionalidades principales del pipeline.
## Gestión de la clave de OpenAI

El acceso a la clave de OpenAI está centralizado en `utils.secret_manager.get_openai_api_key()`,
que abstrae las fuentes permitidas y aplica el principio de acceso controlado. Ningún otro
módulo debe leer directamente archivos o variables de entorno con la credencial.