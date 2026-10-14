from __future__ import annotations

import time

import pytest

from avenly.security import REDACTED, CryptoManager, SecurityError, anonymize, generate_key

# --------------------------------------------------------------------------
# Anonymisation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pii",
    [
        "jean.dupont+cv@gmail.com",
        "j_dupont@etu.univ-paris.fr",
        "06 12 34 56 78",
        "06.12.34.56.78",
        "0612345678",
        "+33 6 12 34 56 78",
        "+33 (0)6 12 34 56 78",
        "0033 6 12 34 56 78",
        "+44 20 7946 0958",
        "12 rue de la Paix",
        "3 bis avenue Foch",
        "10 bd Saint-Michel",
        "75002 Paris",
        "13008 Marseille Cedex 08",
    ],
)
def test_pii_is_redacted(pii):
    result = anonymize(f"Contact : {pii} — merci")
    assert REDACTED in result
    for fragment in pii.split():
        if len(fragment) > 3:
            assert fragment not in result


@pytest.mark.parametrize(
    "useful_text",
    [
        "Expériences 2021-2023",
        "stage de 6 mois",
        "équipe de 12 personnes",
        "Python 3.12",
        "SIRET 12345678901234",
        "budget 2500 euros",
        "promo de 35000 étudiants",
        "Licence 3 Informatique",
    ],
)
def test_useful_text_is_kept(useful_text):
    assert anonymize(useful_text) == useful_text


def test_anonymize_rejects_non_str():
    with pytest.raises(TypeError):
        anonymize(b"bytes")  # type: ignore[arg-type]


def test_anonymize_resists_redos():
    """Un texte piégé ne doit pas bloquer les regex (backtracking catastrophique)."""
    payload = ("1 rue " + "a " * 50 + "@" * 100 + "0" * 1000) * 200
    start = time.perf_counter()
    anonymize(payload)
    assert time.perf_counter() - start < 1.0


# --------------------------------------------------------------------------
# Chiffrement
# --------------------------------------------------------------------------


def test_encrypt_decrypt_roundtrip():
    crypto = CryptoManager(generate_key())
    token = crypto.encrypt_text("CV confidentiel")
    assert b"CV confidentiel" not in token
    assert crypto.decrypt_text(token) == "CV confidentiel"


def test_key_is_read_from_settings(monkeypatch):
    key = generate_key()
    monkeypatch.setenv("FERNET_KEY", key)
    assert CryptoManager().decrypt_text(CryptoManager(key).encrypt_text("ok")) == "ok"


def test_wrong_key_is_rejected():
    token = CryptoManager(generate_key()).encrypt_text("secret")
    with pytest.raises(SecurityError):
        CryptoManager(generate_key()).decrypt(token)


def test_tampered_token_is_rejected():
    crypto = CryptoManager(generate_key())
    token = bytearray(crypto.encrypt_text("secret"))
    token[-5] ^= 1
    with pytest.raises(SecurityError):
        crypto.decrypt(bytes(token))


def test_invalid_key():
    with pytest.raises(SecurityError, match="invalide"):
        CryptoManager("pas-une-cle")


def test_missing_key():
    with pytest.raises(SecurityError, match="absente"):
        CryptoManager()


def test_repr_hides_key():
    key = generate_key()
    assert key not in repr(CryptoManager(key))
