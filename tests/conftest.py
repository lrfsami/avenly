"""Fixtures communes : chaque test démarre avec une configuration vierge.

Les tests ne lisent jamais le vrai .env (qui peut contenir des secrets).
"""

from __future__ import annotations

import pymupdf
import pytest

from avenly import config


@pytest.fixture(autouse=True)
def clean_settings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "load_dotenv", lambda *args, **kwargs: False)
    for name in config.Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


@pytest.fixture
def make_pdf():
    """Fabrique un PDF en mémoire (rien n'est écrit sur le disque)."""

    def _make(text: str | None = "Texte du CV", pages: int = 1, password: str | None = None) -> bytes:
        doc = pymupdf.open()
        for _ in range(pages):
            page = doc.new_page()
            if text is not None:
                page.insert_text((50, 72), text, fontsize=10)
        options = {}
        if password:
            options = {
                "encryption": pymupdf.PDF_ENCRYPT_AES_256,
                "user_pw": password,
                "owner_pw": password,
            }
        return doc.tobytes(**options)

    return _make
