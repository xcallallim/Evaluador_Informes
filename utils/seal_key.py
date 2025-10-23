"""Herramientas para cifrar de forma interactiva la clave API de OpenAI.

Este módulo permite generar el archivo ``secrets/openai_api_key.enc`` a partir de
una clave API introducida manualmente o leída desde ``secrets/openai_api_key.txt``.
La función principal :func:`seal_api_key` guía al usuario para capturar la
passphrase, generar el salt y delegar el proceso de cifrado en
:func:`utils.secret_manager.seal_key`, garantizando compatibilidad con el
sistema de descifrado existente.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from getpass import getpass
from pathlib import Path
from typing import Optional

try:  # pragma: no cover - import fallbacks exercised only when run as script.
    from . import secret_manager
except ImportError:  # pragma: no cover - executed when running as ``python utils/seal_key.py``
    # Ajuste del ``sys.path`` para permitir la ejecución directa del script.
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.append(str(REPO_ROOT))
    from utils import secret_manager


PLAIN_TEXT_FILE = secret_manager.PLAIN_TEXT_FILE
ENCRYPTED_FILE = secret_manager.ENCRYPTED_FILE
MIN_PASSPHRASE_LENGTH = secret_manager.MIN_PASSPHRASE_LENGTH
PASSPHRASE_ENV_VAR = secret_manager.PASSPHRASE_ENV_VAR

logger = logging.getLogger(__name__)


def _read_api_key_from_file(path: Path) -> str:
    try:
        key = path.read_text(encoding="utf-8").strip()
    except OSError as exc:  # pragma: no cover - filesystem errors are unexpected but handled.
        raise RuntimeError(f"No se pudo leer la clave API desde {path}: {exc}") from exc

    if not key:
        raise RuntimeError(
            f"El archivo {path} no contiene una clave API válida."
        )
    return key


def _prompt_yes_no(message: str) -> bool:
    while True:
        answer = input(message).strip().lower()
        if answer in {"s", "si", "sí", "y", "yes"}:
            return True
        if answer in {"n", "no", ""}:
            return False
        print("Por favor responde con 's' o 'n'.", file=sys.stderr)


def _prompt_api_key() -> str:
    print("Introduce la clave API de OpenAI. El texto no se mostrará en pantalla.")
    while True:
        api_key = getpass("Clave API: ").strip()
        if not api_key:
            print("La clave API no puede estar vacía.", file=sys.stderr)
            continue
        confirm = getpass("Repite la clave API para confirmar: ").strip()
        if api_key != confirm:
            print("Las claves introducidas no coinciden.", file=sys.stderr)
            continue
        return api_key


def _prompt_passphrase() -> str:
    print(
        "Introduce una passphrase segura (mínimo "
        f"{MIN_PASSPHRASE_LENGTH} caracteres). El texto no se mostrará en pantalla."
    )
    while True:
        passphrase = getpass("Passphrase: ")
        if len(passphrase) < MIN_PASSPHRASE_LENGTH:
            print(
                f"La passphrase debe tener al menos {MIN_PASSPHRASE_LENGTH} caracteres.",
                file=sys.stderr,
            )
            continue
        confirm = getpass("Confirma la passphrase: ")
        if passphrase != confirm:
            print("Las passphrases no coinciden.", file=sys.stderr)
            continue
        return passphrase


def seal_api_key(
    *,
    api_key: Optional[str] = None,
    passphrase: Optional[str] = None,
    plain_text_file: Optional[Path] = None,
    output_file: Optional[Path] = None,
    non_interactive: bool = False,
    passphrase_env_var: Optional[str] = None,
) -> Path:
    """Genere ``openai_api_key.enc`` interactuando con el usuario.

    Parameters
    ----------
    api_key:
        Clave API a cifrar. Si no se proporciona se solicitará al usuario.
    passphrase:
        Passphrase utilizada para derivar la clave de cifrado. Si no se proporciona
        se solicitará al usuario.
    plain_text_file:
        Ruta opcional para un archivo con la clave API en texto plano. Si es ``None``
        se utiliza :data:`PLAIN_TEXT_FILE`.
    output_file:
        Ruta de salida para el archivo cifrado. Si es ``None`` se utiliza
        :data:`ENCRYPTED_FILE`.
    non_interactive:
        Si es ``True`` el proceso no solicitará datos por consola. Requiere que
        la clave API esté disponible y que la passphrase se obtenga de una
        variable de entorno.
    passphrase_env_var:
        Nombre de la variable de entorno de la que obtener la passphrase cuando
        ``non_interactive`` es ``True``. Si es ``None`` se utiliza
        :data:`PASSPHRASE_ENV_VAR`.

    Returns
    -------
    pathlib.Path
        Ruta del archivo cifrado generado.
    """

    explicit_plain_text_file = plain_text_file is not None
    source_file = plain_text_file or PLAIN_TEXT_FILE
    destination = output_file or ENCRYPTED_FILE
    destination.parent.mkdir(parents=True, exist_ok=True)

    if api_key is None and source_file.exists():
        if non_interactive or explicit_plain_text_file:
            api_key = _read_api_key_from_file(source_file)
        elif _prompt_yes_no(
            f"¿Deseas utilizar la clave almacenada en {source_file}? [s/N]: "
        ):
            api_key = _read_api_key_from_file(source_file)

    if api_key is None:
        if non_interactive:
            raise RuntimeError(
                "En modo no interactivo se requiere proporcionar la clave API "
                "mediante --api-key, --api-key-file o un archivo existente."
            )
        api_key = _prompt_api_key()

    if passphrase is None:
        if non_interactive:
            env_var = passphrase_env_var or PASSPHRASE_ENV_VAR
            passphrase = os.getenv(env_var) if env_var else None
            if not passphrase:
                raise RuntimeError(
                    "En modo no interactivo se requiere definir la passphrase en la "
                    f"variable de entorno {env_var}."
                )
        else:
            passphrase = _prompt_passphrase()

    destination = secret_manager.seal_key(
        api_key,
        passphrase,
        output_file=destination,
    )

    logger.debug("Archivo cifrado creado en %s", destination)

    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera el archivo cifrado openai_api_key.enc para OpenAI."
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help=(
            "Ejecuta el sellado sin solicitar datos por consola. Requiere que la "
            "clave API esté disponible y que la passphrase se defina en la "
            f"variable de entorno {PASSPHRASE_ENV_VAR}."
        ),
    )
    parser.add_argument(
        "--api-key",
        help=(
            "Clave API a cifrar. Úsese con precaución, puede quedar registrada en "
            "el historial del shell."
        ),
    )
    parser.add_argument(
        "--api-key-file",
        type=Path,
        help="Ruta de un archivo desde el que leer la clave API en texto plano.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help="Ruta del archivo cifrado a generar. Por defecto se usa secrets/openai_api_key.enc.",
    )
    parser.add_argument(
        "--passphrase",
        help="Passphrase para derivar la clave de cifrado.",
    )
    parser.add_argument(
        "--passphrase-env-var",
        help=(
            "Variable de entorno de la que obtener la passphrase en modo no "
            "interactivo. Por defecto se utiliza OPENAI_KEY_PASSPHRASE."
        ),
    )

    args = parser.parse_args()

    try:
        destination = seal_api_key(
            api_key=args.api_key,
            passphrase=args.passphrase,
            plain_text_file=args.api_key_file,
            output_file=args.output_file,
            non_interactive=args.non_interactive,
            passphrase_env_var=args.passphrase_env_var,
        )
    except secret_manager.SecretManagerError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:  # pragma: no cover - user interruption
        print("\nOperación cancelada por el usuario.", file=sys.stderr)
        sys.exit(1)
    else:
        print(
            "La clave API se ha cifrado correctamente. Archivo generado en:"
            f" {destination}"
        )
        print("Por seguridad, elimina openai_api_key.txt si existe.")


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()