"""Modèles de données partagés : CLI aujourd'hui, API FastAPI et base de données demain."""

from __future__ import annotations

import re
import unicodedata
from typing import Annotated
from urllib.parse import urlsplit

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator


def _clean_text(value: str) -> str:
    """Retire les caractères invisibles/de contrôle et normalise les espaces.

    Les données des APIs finiront dans des prompts et des emails : on ne
    laisse passer aucun caractère caché.
    """
    value = unicodedata.normalize("NFKC", value)
    value = "".join(ch for ch in value if unicodedata.category(ch) not in ("Cc", "Cf", "Co", "Cs"))
    return " ".join(value.split())


CleanStr = Annotated[str, AfterValidator(_clean_text)]

_NAF_COMPACT = re.compile(r"^(\d{2})\.?(\d{2}[A-Z])$")
# « javascript:… », « file:… », « http:… » : tout préfixe de schéma explicite.
_URL_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*:")
_HOSTNAME = re.compile(r"^(?=.{4,253}$)([a-z0-9]([a-z0-9\-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}$")


class GeoPoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class Company(BaseModel):
    """Établissement d'une entreprise, identifié par son SIRET."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    client_id: CleanStr = Field(min_length=1, max_length=64)
    siret: str = Field(pattern=r"^\d{14}$")
    name: CleanStr = Field(min_length=1, max_length=200)
    naf_code: str | None = Field(default=None, pattern=r"^\d{2}\.\d{2}[A-Z]$")
    address: CleanStr | None = Field(default=None, max_length=300)
    location: GeoPoint | None = None
    headcount: CleanStr | None = Field(default=None, max_length=50)
    website: str | None = None
    apply_url: str | None = None
    has_job_offer: bool = False
    sources: set[str] = Field(default_factory=set)

    @property
    def siren(self) -> str:
        return self.siret[:9]

    @field_validator("siret", mode="before")
    @classmethod
    def _strip_siret(cls, value: object) -> object:
        return re.sub(r"\s", "", value) if isinstance(value, str) else value

    @field_validator("naf_code", mode="before")
    @classmethod
    def _normalize_naf(cls, value: object) -> object:
        """Accepte « 6201Z » comme « 62.01Z » ; ignore un code illisible."""
        if not isinstance(value, str):
            return value
        match = _NAF_COMPACT.match(value.strip().upper())
        return f"{match[1]}.{match[2]}" if match else None

    @field_validator("website", "apply_url", mode="before")
    @classmethod
    def _safe_url(cls, value: object) -> str | None:
        """Garde uniquement les URLs http(s) valides (pas de javascript:, file:…)."""
        if not isinstance(value, str) or not value.strip() or len(value) > 500:
            return None
        url = value.strip()
        if not _URL_SCHEME.match(url):
            url = f"https://{url}"  # « www.exemple.fr » -> « https://www.exemple.fr »
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not _HOSTNAME.match(parts.hostname or ""):
            return None
        return url

    def merge(self, other: Company) -> Company:
        """Fusionne deux fiches du même SIRET : complète les champs vides."""
        if other.siret != self.siret:
            raise ValueError("Fusion impossible : SIRET différents.")
        data = self.model_dump()
        for field, value in other.model_dump().items():
            if data.get(field) in (None, "") and value not in (None, ""):
                data[field] = value
        data["sources"] = self.sources | other.sources
        data["has_job_offer"] = self.has_job_offer or other.has_job_offer
        return Company(**data)
