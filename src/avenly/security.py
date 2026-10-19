"""Sécurité : chiffrement symétrique (Fernet) et anonymisation des PII.

Générer une clé à coller dans .env (FERNET_KEY=...) :
    PYTHONPATH=src python -m avenly.security
"""

from __future__ import annotations

import re

from cryptography.fernet import Fernet, InvalidToken

from avenly.config import get_settings

REDACTED = "[CONFIDENTIEL]"


class SecurityError(RuntimeError):
    """Erreur de chiffrement ou de configuration de sécurité."""


# --------------------------------------------------------------------------
# Chiffrement
# --------------------------------------------------------------------------


def generate_key() -> str:
    """Génère une nouvelle clé Fernet (à stocker dans .env, jamais dans le code)."""
    return Fernet.generate_key().decode("ascii")


class CryptoManager:
    """Chiffre/déchiffre des données avec Fernet (AES-128-CBC + HMAC-SHA256)."""

    def __init__(self, key: str | None = None) -> None:
        if key is None:
            secret = get_settings().fernet_key
            if secret is None:
                raise SecurityError(
                    "FERNET_KEY absente du .env. "
                    "Générez-en une avec : PYTHONPATH=src python -m avenly.security"
                )
            key = secret.get_secret_value()
        try:
            self._fernet = Fernet(key)
        except (ValueError, TypeError):
            raise SecurityError(
                "FERNET_KEY invalide (attendu : 32 octets encodés en base64 url-safe)."
            ) from None

    def __repr__(self) -> str:
        return "CryptoManager(key=***)"

    def encrypt(self, data: bytes) -> bytes:
        return self._fernet.encrypt(data)

    def decrypt(self, token: bytes, ttl: int | None = None) -> bytes:
        """Déchiffre `token`. Si `ttl` (secondes) est donné, refuse un token plus ancien."""
        try:
            return self._fernet.decrypt(token, ttl=ttl)
        except InvalidToken:
            raise SecurityError("Déchiffrement impossible : token altéré, expiré ou mauvaise clé.") from None

    def encrypt_text(self, text: str) -> bytes:
        return self.encrypt(text.encode("utf-8"))

    def decrypt_text(self, token: bytes, ttl: int | None = None) -> str:
        return self.decrypt(token, ttl=ttl).decode("utf-8")


# --------------------------------------------------------------------------
# Anonymisation
# --------------------------------------------------------------------------
# Quantificateurs bornés partout : pas de backtracking catastrophique (ReDoS)
# même sur un texte de CV piégé.

_EMAIL = re.compile(
    r"(?<![\w.+-])[\w.+-]{1,64}@[A-Za-z0-9-]{1,63}(?:\.[A-Za-z0-9-]{1,63}){0,5}\.[A-Za-z]{2,24}\b"
)

# 06 12 34 56 78 · 06.12.34.56.78 · 0612345678 · +33 6 12 34 56 78 · +33 (0)6… · 0033 6…
_PHONE_FR = re.compile(
    r"(?<![\w+])"
    r"(?:(?:\+|00)\s?33\s?(?:\(0\)\s?)?[1-9]|0[1-9])"
    r"(?:[\s.\-]?\d{2}){4}"
    r"(?!\d)"
)

# Numéros internationaux explicites : +44 20 7946 0958, +1-202-555-0143…
_PHONE_INTL = re.compile(r"(?<![\w+])\+\d{1,3}(?:[\s.\-]?\(?\d{1,4}\)?){2,5}(?!\d)")

_STREET_TYPES = (
    r"rue|avenue|av\.?|boulevard|bd|bld|place|pl\.|all[ée]e|chemin|impasse|imp\.|"
    r"route|rte|quai|cours|square|passage|faubourg|fbg|r[ée]sidence|lotissement|"
    r"hameau|lieu[- ]dit|voie|villa|cit[ée]|esplanade|promenade|parvis|"
    r"rond[- ]point|sentier|mont[ée]e|chauss[ée]e"
)

# 12 rue de la Paix · 3 bis, avenue Foch · 10 bd Saint-Michel
_STREET = re.compile(
    rf"\b\d{{1,4}}(?:[^\S\n]?(?:bis|ter|quater|[a-d])\b)?,?[^\S\n]+"
    rf"(?:{_STREET_TYPES})(?![\w])"
    rf"(?:[^\S\n]+[\w'’\-]{{1,40}}){{0,6}}",
    re.IGNORECASE,
)

# 75001 Paris · 69100 Villeurbanne · 13008 Marseille Cedex 08 · F-33000 Bordeaux
# Sensible à la casse : la ville doit commencer par une majuscule.
_POSTCODE_CITY = re.compile(
    r"\b(?:F-)?(?:0[1-9]|[1-8]\d|9[0-8])\d{3}[^\S\n]+"
    r"[A-ZÀ-Ý][\w'’\-]{1,40}"
    r"(?:[^\S\n]+(?:[A-ZÀ-Ý][\w'’\-]{0,40}|sur|sous|en|le|la|les|de|du|des|lès|\d{1,2})){0,5}"
)

# Date complète (jour/mois/année) : dans un CV, c'est presque toujours la date de naissance.
# Les dates d'expérience (« 08/2024 », « 2025-2026 ») ne sont pas concernées.
_FULL_DATE = re.compile(
    r"(?<![\d/.\-])(?:0?[1-9]|[12]\d|3[01])[/.\-](?:0?[1-9]|1[0-2])[/.\-](?:19|20)\d{2}(?![\d/.\-])"
)

_MONTHS = (
    r"janvier|février|fevrier|mars|avril|mai|juin|juillet"
    r"|août|aout|septembre|octobre|novembre|décembre|decembre"
)
# « Né le 17 décembre 2005 », « née en 2005 », « Date de naissance : … », « Âge : 20 ans », « 20 ans »
_BIRTH_MENTION = re.compile(
    rf"\b(?:n[ée]e?|date de naissance)\s*(?:le|en|:)?\s*"
    rf"(?:\d{{1,2}}(?:er)?\s+(?:{_MONTHS})\s+)?(?:19|20)\d{{2}}\b"
    r"|\b[âa]ge\s*:?\s*\d{1,2}\s*ans\b"
    r"|(?<![\w,.])(?:1[5-9]|[2-6]\d)\s*ans\b(?!\s+d['’]exp)",
    re.IGNORECASE,
)

# L'ordre compte : emails d'abord (ils peuvent contenir des chiffres).
_PATTERNS: tuple[re.Pattern[str], ...] = (
    _EMAIL,
    _FULL_DATE,
    _BIRTH_MENTION,
    _PHONE_FR,
    _PHONE_INTL,
    _STREET,
    _POSTCODE_CITY,
)


def anonymize(text: str) -> str:
    """Remplace emails, téléphones, adresses postales, dates de naissance et âge par [CONFIDENTIEL].

    Heuristique par regex : à utiliser avant tout envoi de texte à un LLM.
    Ne détecte pas les noms de personnes ni les adresses sans numéro de voie
    ni code postal.
    """
    if not isinstance(text, str):
        raise TypeError(f"anonymize() attend un str, reçu {type(text).__name__}")
    for pattern in _PATTERNS:
        text = pattern.sub(REDACTED, text)
    return text


if __name__ == "__main__":
    print(generate_key())
