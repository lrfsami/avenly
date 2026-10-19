"""Lecture sécurisée d'un CV PDF, en mémoire uniquement, puis anonymisation.

Aucune copie du CV n'est écrite sur le disque : le fichier est lu en bytes,
vérifié, ouvert par PyMuPDF depuis un io.BytesIO, et seul le texte
anonymisé ressort.
"""

from __future__ import annotations

import hashlib
import io
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import pymupdf

from avenly.config import get_settings
from avenly.security import REDACTED, anonymize

PDF_MAGIC = b"%PDF-"
MAX_PAGES = 10
MAX_TEXT_CHARS = 50_000

# Les messages de MuPDF sur stderr peuvent contenir des extraits du fichier.
pymupdf.TOOLS.mupdf_display_errors(False)
pymupdf.TOOLS.mupdf_display_warnings(False)


class CVError(ValueError):
    """CV refusé : fichier invalide, piégé ou illisible."""


@dataclass(frozen=True, slots=True)
class CVDocument:
    """CV prêt à être envoyé à un LLM : le texte est déjà anonymisé."""

    sha256: str
    page_count: int
    text: str = field(repr=False)
    truncated: bool = False


# --------------------------------------------------------------------------
# Étapes internes
# --------------------------------------------------------------------------


def _read_bytes(source: str | Path | bytes | BinaryIO, max_bytes: int) -> bytes:
    """Lit au plus `max_bytes` octets ; refuse au-delà (sans tout charger en RAM)."""
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
    elif isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise CVError(f"Fichier introuvable : {path.name}")
        if path.stat().st_size > max_bytes:
            raise CVError(f"CV trop volumineux (max {max_bytes // (1024 * 1024)} Mo).")
        with path.open("rb") as file:
            data = file.read(max_bytes + 1)
    elif hasattr(source, "read"):
        data = source.read(max_bytes + 1)
    else:
        raise TypeError(f"Source de CV non supportée : {type(source).__name__}")

    if len(data) > max_bytes:
        raise CVError(f"CV trop volumineux (max {max_bytes // (1024 * 1024)} Mo).")
    if not data:
        raise CVError("Fichier vide.")
    return data


def _check_magic_bytes(data: bytes) -> None:
    """Un vrai PDF commence par %PDF- : l'extension .pdf ne prouve rien."""
    if not data.startswith(PDF_MAGIC):
        raise CVError("Le fichier n'est pas un PDF (signature %PDF- absente).")


def _extract_text(data: bytes) -> tuple[int, str, bool]:
    """Renvoie (nombre de pages, texte brut, texte tronqué ?)."""
    try:
        doc = pymupdf.open(stream=io.BytesIO(data), filetype="pdf")
    except Exception:
        raise CVError("PDF corrompu ou illisible.") from None

    with doc:
        if doc.needs_pass:
            raise CVError("PDF protégé par mot de passe : envoyez une version non protégée.")
        if doc.page_count == 0:
            raise CVError("PDF sans aucune page.")
        if doc.page_count > MAX_PAGES:
            raise CVError(f"Trop de pages pour un CV ({doc.page_count} > {MAX_PAGES}).")

        parts: list[str] = []
        total = 0
        truncated = False
        for page in doc:
            page_text = page.get_text("text", sort=True)
            parts.append(page_text)
            total += len(page_text)
            if total > MAX_TEXT_CHARS:
                truncated = True
                break
        page_count = doc.page_count

    text = "\n".join(parts)[:MAX_TEXT_CHARS]
    return page_count, text, truncated


_MULTI_SPACES = re.compile(r"[^\S\n]+")
_MULTI_NEWLINES = re.compile(r"\n{3,}")


def _clean(text: str) -> str:
    """Normalise le texte et retire les caractères invisibles.

    Les caractères de format Unicode (zéro-largeur, inversion bidi…) servent
    souvent à cacher du texte : on les supprime avant l'envoi à un LLM.
    """
    text = unicodedata.normalize("NFKC", text)  # ligatures « ﬁ » -> « fi », etc.
    text = "".join(
        ch for ch in text if ch in "\n\t" or unicodedata.category(ch) not in ("Cc", "Cf", "Co", "Cs")
    )
    text = _MULTI_SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _MULTI_NEWLINES.sub("\n\n", text).strip()


# Variantes accentuées : « Lea » doit aussi masquer « Léa », et inversement.
_ACCENT_VARIANTS = {
    "a": "aàâä", "c": "cç", "e": "eéèêë", "i": "iîï", "o": "oôö", "u": "uùûü", "y": "yÿ",
}  # fmt: skip


def _strip_accents(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _name_parts(names: Iterable[str]) -> set[str]:
    """Découpe « Jean-Marc Dupont » en {"Jean", "Marc", "Dupont"} (parties de 2 lettres ou plus)."""
    if isinstance(names, str):  # erreur fréquente : read_cv(..., candidate_names="Jean Dupont")
        names = [names]
    parts = {p for name in names for p in re.split(r"[\s\-]+", name.strip()) if len(p) >= 2}
    if not parts:
        raise CVError("Nom du candidat obligatoire : il sert à le masquer avant tout envoi à une IA.")
    return parts


def _name_pattern(part: str) -> re.Pattern[str]:
    letters = "".join(
        f"[{_ACCENT_VARIANTS[ch]}]" if ch in _ACCENT_VARIANTS else re.escape(ch)
        for ch in _strip_accents(part).lower()
    )
    return re.compile(rf"(?<!\w){letters}(?!\w)", re.IGNORECASE)


def _redact_names(text: str, parts: set[str]) -> tuple[str, int]:
    """Masque les parties du nom (casse et accents ignorés). Renvoie (texte, nombre de remplacements)."""
    total = 0
    for part in parts:
        text, count = _name_pattern(part).subn(REDACTED, text)
        total += count
    return text, total


# --------------------------------------------------------------------------
# API publique
# --------------------------------------------------------------------------


def read_cv(
    source: str | Path | bytes | BinaryIO,
    *,
    candidate_names: Iterable[str],
) -> CVDocument:
    """Lit un CV PDF de façon sécurisée et renvoie son texte anonymisé.

    `candidate_names` (obligatoire) : nom et prénom du candidat, masqués en plus
    des emails, téléphones et adresses. Si aucun n'est retrouvé dans le CV
    (faute de frappe, oubli), la lecture échoue plutôt que de laisser passer
    le nom en clair.
    """
    name_parts = _name_parts(candidate_names)  # vérifié avant même de lire le fichier
    data = _read_bytes(source, get_settings().max_cv_size_bytes)
    _check_magic_bytes(data)
    sha256 = hashlib.sha256(data).hexdigest()

    page_count, raw_text, truncated = _extract_text(data)
    del data

    # anonymize() d'abord : masquer le nom avant casserait la détection des
    # emails du type prenom.nom@gmail.com.
    text = anonymize(_clean(raw_text))
    if not text:
        raise CVError("Aucun texte extractible : CV scanné en image (OCR non supporté).")
    text, redacted = _redact_names(text, name_parts)
    if redacted == 0:
        raise CVError(
            "Le nom fourni n'apparaît pas dans le CV : vérifie l'orthographe "
            "(le CV n'est pas utilisé tant que le nom ne peut pas être masqué)."
        )

    return CVDocument(sha256=sha256, page_count=page_count, text=text, truncated=truncated)
