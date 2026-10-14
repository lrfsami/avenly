from __future__ import annotations

import pytest

from avenly.config import ConfigError, get_settings


def test_default_values():
    settings = get_settings()
    assert settings.http_timeout_seconds == 10
    assert settings.fernet_key is None
    assert settings.max_cv_size_bytes == 5 * 1024 * 1024


def test_secrets_are_masked(monkeypatch):
    monkeypatch.setenv("FERNET_KEY", "super-secret-value")
    monkeypatch.setenv("LBA_API_TOKEN", "token-123")
    settings = get_settings()
    assert "super-secret-value" not in repr(settings)
    assert "token-123" not in str(settings)
    assert settings.fernet_key.get_secret_value() == "super-secret-value"


def test_empty_secret_is_none(monkeypatch):
    monkeypatch.setenv("FERNET_KEY", "   ")
    assert get_settings().fernet_key is None


def test_invalid_values_do_not_leak_input(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_SECOND", "50")
    monkeypatch.setenv("HTTP_TIMEOUT_SECONDS", "valeur-secrete")
    with pytest.raises(ConfigError) as exc:
        get_settings()
    message = str(exc.value)
    assert "RATE_LIMIT_PER_SECOND" in message
    assert "HTTP_TIMEOUT_SECONDS" in message
    assert "valeur-secrete" not in message


def test_settings_are_immutable():
    settings = get_settings()
    with pytest.raises(ValueError):
        settings.max_cv_size_mb = 100
