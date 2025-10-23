import pytest

from utils import secret_manager


def test_get_key_from_environment(monkeypatch):
    monkeypatch.setenv(secret_manager.ENV_VAR, "env-key")
    monkeypatch.delenv(secret_manager.PASSPHRASE_ENV_VAR, raising=False)
    assert secret_manager.get_openai_api_key() == "env-key"


def test_get_key_from_plain_text(monkeypatch, tmp_path):
    monkeypatch.delenv(secret_manager.ENV_VAR, raising=False)
    monkeypatch.delenv(secret_manager.PASSPHRASE_ENV_VAR, raising=False)
    plain_text_file = tmp_path / "openai_api_key.txt"
    plain_text_file.write_text("plain-key\n", encoding="utf-8")

    monkeypatch.setattr(secret_manager, "PLAIN_TEXT_FILE", plain_text_file)

    assert secret_manager.get_openai_api_key() == "plain-key"


def test_get_key_from_encrypted_file(monkeypatch, tmp_path):
    monkeypatch.delenv(secret_manager.ENV_VAR, raising=False)
    monkeypatch.delenv(secret_manager.PASSPHRASE_ENV_VAR, raising=False)

    encrypted_file = tmp_path / "openai_api_key.enc"
    monkeypatch.setattr(secret_manager, "ENCRYPTED_FILE", encrypted_file)
    monkeypatch.setattr(secret_manager, "PLAIN_TEXT_FILE", tmp_path / "openai_api_key.txt")

    passphrase = "correcthorsebattery"
    secret_manager.seal_key("encrypted-key", passphrase, output_file=encrypted_file)

    monkeypatch.setenv(secret_manager.PASSPHRASE_ENV_VAR, passphrase)

    assert secret_manager.get_openai_api_key() == "encrypted-key"


def test_encrypted_file_with_wrong_passphrase(monkeypatch, tmp_path):
    monkeypatch.delenv(secret_manager.ENV_VAR, raising=False)
    monkeypatch.delenv(secret_manager.PASSPHRASE_ENV_VAR, raising=False)

    encrypted_file = tmp_path / "openai_api_key.enc"
    monkeypatch.setattr(secret_manager, "ENCRYPTED_FILE", encrypted_file)
    monkeypatch.setattr(secret_manager, "PLAIN_TEXT_FILE", tmp_path / "openai_api_key.txt")

    secret_manager.seal_key("encrypted-key", "anotherpass", output_file=encrypted_file)

    monkeypatch.setenv(secret_manager.PASSPHRASE_ENV_VAR, "wrong123")

    with pytest.raises(secret_manager.SecretManagerError):
        secret_manager.get_openai_api_key()


def test_short_passphrase_rejected(monkeypatch, tmp_path):
    encrypted_file = tmp_path / "openai_api_key.enc"

    with pytest.raises(secret_manager.SecretManagerError):
        secret_manager.seal_key("encrypted-key", "short", output_file=encrypted_file)