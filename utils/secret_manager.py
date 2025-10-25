"""Funciones para recuperar la clave API de OpenAI de forma segura.

El módulo evalúa múltiples fuentes en el siguiente orden:
1. Variable de entorno ``OPENAI_API_KEY``.
2. Archivo de texto ``secrets/openai_api_key.txt`` (solo para desarrollo local).
3. Archivo cifrado ``secrets/openai_api_key.enc`` que requiere la variable
   ``OPENAI_KEY_PASSPHRASE`` para descifrar

El archivo cifrado almacena un JSON con dos campos: ``salt`` (bytes aleatorios en
base64 usados para derivar la clave) y ``token`` (el *token* de Fernet).

Si ninguna fuente proporciona una clave se lanza ``SecretManagerError``.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class SecretManagerError(RuntimeError):
    """Se lanza cuando no se puede recuperar la clave API de OpenAI."""


@dataclass
class SecretSource:
    """Representa una posible fuente para la clave API de OpenAI."""

    name: str
    value: Optional[str] = None


BASE_DIR = Path(__file__).resolve().parent.parent
SECRETS_DIR = BASE_DIR / "secrets"
PLAIN_TEXT_FILE = SECRETS_DIR / "openai_api_key.txt"
ENCRYPTED_FILE = SECRETS_DIR / "openai_api_key.enc"
ENV_VAR = "OPENAI_API_KEY"
PASSPHRASE_ENV_VAR = "OPENAI_KEY_PASSPHRASE"
PBKDF2_ITERATIONS = 390_000
PBKDF2_LENGTH = 32
MIN_PASSPHRASE_LENGTH = 8

logger = logging.getLogger(__name__)


def _get_env_variable(name: str) -> Optional[str]:
    value = os.getenv(name)
    if value:
        value = value.strip()
    return value or None


def _load_from_env() -> Optional[SecretSource]:
    value = _get_env_variable(ENV_VAR)
    if value:
        return SecretSource(name="environment", value=value)
    return None


def _load_from_plain_text() -> Optional[SecretSource]:
    if not PLAIN_TEXT_FILE.exists():
        return None

    try:
        content = PLAIN_TEXT_FILE.read_text(encoding="utf-8").strip()
    except OSError as exc:  # pragma: no cover - improbable pero protegido.
        raise SecretManagerError(
            f"No se pudo leer el archivo de texto con la clave API: {exc}"
        ) from exc

    if not content:
        return None

    return SecretSource(name="plain_text_file", value=content)


def _validate_passphrase(passphrase: str) -> str:
    if len(passphrase) < MIN_PASSPHRASE_LENGTH:
        raise SecretManagerError(
            "La passphrase debe tener al menos 8 caracteres para proteger la clave API."
        )
    return passphrase


def _derive_key(passphrase: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=PBKDF2_LENGTH,
        salt=salt,
        iterations=PBKDF2_ITERATIONS,
    )
    return base64.urlsafe_b64encode(kdf.derive(passphrase.encode("utf-8")))



def seal_key(
    api_key: str,
    passphrase: str,
    *,
    output_file: Optional[Path] = None,
) -> Path:
    """Cifre y almacene la clave API de OpenAI en ``openai_api_key.enc``.

     Parámetros
    ----------
    api_key:
        Clave API en texto plano que se desea proteger.
    passphrase:
        Passphrase utilizada para derivar la clave criptográfica.
    output_file:
        Ruta de destino para el archivo cifrado. Si no se proporciona se utiliza
        :data:`ENCRYPTED_FILE`.

    Returns
    -------
    pathlib.Path
        Ruta del archivo cifrado generado.

    Excepciones 
    ------
    SecretManagerError
        Si la clave API está vacía o la passphrase no cumple los requisitos.
    """

    if not api_key or not api_key.strip():
        raise SecretManagerError("La clave API no puede estar vacía.")

    passphrase = _validate_passphrase(passphrase)

    salt = os.urandom(16)
    key = _derive_key(passphrase, salt)

    fernet = Fernet(key)
    token = fernet.encrypt(api_key.strip().encode("utf-8"))

    payload = {
        "salt": base64.b64encode(salt).decode("utf-8"),
        "token": token.decode("utf-8"),
    }

    destination = output_file or ENCRYPTED_FILE
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:  # pragma: no cover - errores de filesystem inesperados.
        raise SecretManagerError(
            f"No se pudo escribir el archivo cifrado con la clave API: {exc}"
        ) from exc

    return destination


def _load_from_encrypted_file() -> Optional[SecretSource]:
    if not ENCRYPTED_FILE.exists():
        return None

    passphrase = _get_env_variable(PASSPHRASE_ENV_VAR)
    if not passphrase:
        raise SecretManagerError(
            "Se requiere la variable de entorno OPENAI_KEY_PASSPHRASE para descifrar la clave."
        )

    passphrase = _validate_passphrase(passphrase)

    try:
        payload = json.loads(ENCRYPTED_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SecretManagerError("El archivo cifrado contiene un JSON inválido.") from exc
    except OSError as exc:  # pragma: no cover - errores de filesystem poco probables pero manejados.
        raise SecretManagerError(
            f"No se pudo leer el archivo cifrado con la clave API: {exc}"
        ) from exc

    salt_b64 = payload.get("salt")
    token = payload.get("token")

    if not salt_b64 or not token:
        raise SecretManagerError(
            "El archivo cifrado debe contener los campos 'salt' y 'token'."
        )

    try:
        salt = base64.b64decode(salt_b64)
    except (ValueError, TypeError) as exc:
        raise SecretManagerError("El salt del archivo cifrado no es válido.") from exc

    key = _derive_key(passphrase, salt)

    try:
        fernet = Fernet(key)
        decrypted = fernet.decrypt(token.encode("utf-8"))
    except Exception as exc:  # pragma: no cover - cryptography puede lanzar múltiples excepciones.
        raise SecretManagerError("No se pudo descifrar la clave API con la passphrase proporcionada.") from exc

    value = decrypted.decode("utf-8").strip()
    if not value:
        raise SecretManagerError("La clave API descifrada está vacía.")

    return SecretSource(name="encrypted_file", value=value)


def seal_key(api_key: str, passphrase: str, *, output_file: Path = ENCRYPTED_FILE) -> Path:
    """Cifra una clave API en un archivo compatible con :func:`get_openai_api_key`."""

    cleaned_key = (api_key or "").strip()
    if not cleaned_key:
        raise SecretManagerError("La clave API proporcionada es inválida o está vacía.")

    cleaned_passphrase = _validate_passphrase((passphrase or "").strip())

    salt = os.urandom(16)
    key = _derive_key(cleaned_passphrase, salt)
    fernet = Fernet(key)
    token = fernet.encrypt(cleaned_key.encode("utf-8"))

    payload = {
        "salt": base64.b64encode(salt).decode("utf-8"),
        "token": token.decode("utf-8"),
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload), encoding="utf-8")
    return output_file


def get_openai_api_key() -> str:
    """Obtiene la clave API de OpenAI desde las fuentes disponibles

    Returns
    -------
    str
        Clave API de OpenAI.

    Lanza
    ------
    SecretManagerError
        Si la clave no se encuentra en ninguna ubicación soportada.
    """

    for loader in (_load_from_env, _load_from_plain_text, _load_from_encrypted_file):
        source = loader()
        if source and source.value:
            logger.debug("API key cargada desde %s", source.name)
            return source.value

    raise SecretManagerError(
        "No se pudo encontrar la clave API de OpenAI en ninguna fuente disponible."
    )


__all__ = ["SecretManagerError", "get_openai_api_key", "seal_key"]
