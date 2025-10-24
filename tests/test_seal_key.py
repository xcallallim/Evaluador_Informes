import base64
import json

import pytest

from utils import secret_manager


def _prepare_secrets_dir(monkeypatch, tmp_path):
    monkeypatch.delenv(secret_manager.ENV_VAR, raising=False)
    monkeypatch.delenv(secret_manager.PASSPHRASE_ENV_VAR, raising=False)

    secrets_dir = tmp_path / "secrets"
    encrypted_file = secrets_dir / "openai_api_key.enc"
    plain_text_file = secrets_dir / "openai_api_key.txt"

    monkeypatch.setattr(secret_manager, "ENCRYPTED_FILE", encrypted_file)
    monkeypatch.setattr(secret_manager, "PLAIN_TEXT_FILE", plain_text_file)

    return encrypted_file, plain_text_file


def test_seal_key_generates_valid_encrypted_file(monkeypatch, tmp_path):
    encrypted_file, _ = _prepare_secrets_dir(monkeypatch, tmp_path)

    api_key = "test-api-key"
    passphrase = "correcthorsebattery"

    secret_manager.seal_key(api_key, passphrase, output_file=encrypted_file)

    payload = json.loads(encrypted_file.read_text(encoding="utf-8"))
    assert set(payload) == {"salt", "token"}

    salt = base64.b64decode(payload["salt"])
    assert len(salt) == 16
    assert payload["token"]

    monkeypatch.setenv(secret_manager.PASSPHRASE_ENV_VAR, passphrase)

    source = secret_manager._load_from_encrypted_file()
    assert source is not None
    assert source.value == api_key
    assert source.name == "encrypted_file"

    assert secret_manager.get_openai_api_key() == api_key

    encrypted_file.unlink()
    assert not encrypted_file.exists()


def test_seal_key_raises_with_wrong_passphrase(monkeypatch, tmp_path):
    encrypted_file, _ = _prepare_secrets_dir(monkeypatch, tmp_path)

    api_key = "another-test-api-key"
    passphrase = "correcthorsebattery"

    secret_manager.seal_key(api_key, passphrase, output_file=encrypted_file)

    monkeypatch.setenv(secret_manager.PASSPHRASE_ENV_VAR, "wrong-passphrase")

    with pytest.raises(secret_manager.SecretManagerError):
        secret_manager.get_openai_api_key()

    encrypted_file.unlink()
    assert not encrypted_file.exists()

# pytest tests/test_seal_key.py -v