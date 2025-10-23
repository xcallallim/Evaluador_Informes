import base64
import json

import pytest

from utils import secret_manager


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    """Ensure environment variables do not leak between tests."""
    monkeypatch.delenv(secret_manager.ENV_VAR, raising=False)
    monkeypatch.delenv(secret_manager.PASSPHRASE_ENV_VAR, raising=False)


@pytest.fixture
def temp_secret_files(monkeypatch, tmp_path):
    plain_text_file = tmp_path / "openai_api_key.txt"
    encrypted_file = tmp_path / "openai_api_key.enc"
    monkeypatch.setattr(secret_manager, "PLAIN_TEXT_FILE", plain_text_file)
    monkeypatch.setattr(secret_manager, "ENCRYPTED_FILE", encrypted_file)
    return plain_text_file, encrypted_file


def test_get_key_from_environment(monkeypatch):
    monkeypatch.setenv(secret_manager.ENV_VAR, "env-key")
    assert secret_manager.get_openai_api_key() == "env-key"


def test_get_key_from_plain_text(temp_secret_files):
    plain_text_file, _ = temp_secret_files
    plain_text_file.write_text("plain-key\n", encoding="utf-8")

    assert secret_manager.get_openai_api_key() == "plain-key"


@pytest.mark.parametrize(
    "provided_passphrase, expect_success",
    [("correcthorsebattery", True), ("wrong123", False)],
)
def test_get_key_from_encrypted_file(monkeypatch, temp_secret_files, provided_passphrase, expect_success):
    _, encrypted_file = temp_secret_files
    seal_passphrase = "correcthorsebattery"
    secret_manager.seal_key("encrypted-key", seal_passphrase, output_file=encrypted_file)

    payload = json.loads(encrypted_file.read_text(encoding="utf-8"))
    assert set(payload) == {"salt", "token"}
    # Ensure the salt can be decoded without raising.
    base64.b64decode(payload["salt"])
    assert isinstance(payload["token"], str) and payload["token"]

    monkeypatch.setenv(secret_manager.PASSPHRASE_ENV_VAR, provided_passphrase)

    if expect_success:
        assert secret_manager.get_openai_api_key() == "encrypted-key"
    else:
        with pytest.raises(secret_manager.SecretManagerError, match="passphrase"):
            secret_manager.get_openai_api_key()


@pytest.mark.parametrize(
    "payload_content, error_match",
    [
        ("not-json", "JSON inválido"),
        (json.dumps({"salt": "Zm9v"}), "salt"),
    ],
)
def test_corrupted_encrypted_file_raises_error(monkeypatch, temp_secret_files, payload_content, error_match):
    _, encrypted_file = temp_secret_files
    encrypted_file.write_text(payload_content, encoding="utf-8")

    monkeypatch.setenv(secret_manager.PASSPHRASE_ENV_VAR, "correcthorsebattery")

    with pytest.raises(secret_manager.SecretManagerError, match=error_match):
        secret_manager.get_openai_api_key()


def test_short_passphrase_rejected(temp_secret_files):
    _, encrypted_file = temp_secret_files

    with pytest.raises(secret_manager.SecretManagerError, match="passphrase"):
        secret_manager.seal_key("encrypted-key", "short", output_file=encrypted_file)


def test_get_api_key_without_sources(monkeypatch, tmp_path):
    monkeypatch.setattr(secret_manager, "PLAIN_TEXT_FILE", tmp_path / "openai_api_key.txt")
    monkeypatch.setattr(secret_manager, "ENCRYPTED_FILE", tmp_path / "openai_api_key.enc")

    with pytest.raises(
        secret_manager.SecretManagerError,
        match="No se pudo encontrar la clave API",
    ):
        secret_manager.get_openai_api_key()

# pytest tests/test_secret_manager.py -v