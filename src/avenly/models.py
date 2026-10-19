"""Modèles de données partagés : CLI aujourd'hui, API FastAPI et base de données demain."""

from __future__ import annotations

import re
import unicodedata
from typing import Annotated, Literal
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


def _clean_multiline(value: str) -> str:
    """Comme _clean_text, mais conserve les sauts de ligne (paragraphes d'un email)."""
    lines = [_clean_text(line) for line in value.replace("\r\n", "\n").split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


CleanMultilineStr = Annotated[str, AfterValidator(_clean_multiline)]

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


# --------------------------------------------------------------------------
# Profil candidat (produit par le LLM à partir du CV anonymisé)
# --------------------------------------------------------------------------
# Le schéma reste simple (c'est lui que le LLM doit respecter) ; les limites de
# taille sont appliquées après coup en tronquant, pour ne jamais rejeter un
# profil entier à cause d'un champ trop long.


def _truncate(limit: int):
    def _apply(value: str) -> str:
        return value[:limit]

    return AfterValidator(_apply)


def _cap(limit: int):
    def _apply(values: list) -> list:
        return values[:limit]

    return AfterValidator(_apply)


ShortText = Annotated[CleanStr, _truncate(120)]
LongText = Annotated[CleanStr, _truncate(600)]


class Education(BaseModel):
    degree: ShortText = Field(description="Intitulé du diplôme, ex. « Cycle ingénieur informatique »")
    school: ShortText | None = Field(default=None, description="Établissement")
    field: ShortText | None = Field(default=None, description="Spécialité")
    start_year: int | None = Field(default=None, description="Année de début (AAAA)")
    end_year: int | None = Field(default=None, description="Année de fin, prévue ou réelle (AAAA)")


_MONTH_DATE = re.compile(r"^(?:19|20)\d{2}(?:-(?:0[1-9]|1[0-2]))?$")


def _month_date(value: str | None) -> str | None:
    """Garde « 2024 » ou « 2024-08 » (et « en cours ») ; ignore tout autre format."""
    if value is None:
        return None
    value = value.strip().lower()
    if value == "en cours" or _MONTH_DATE.match(value):
        return value
    return None


MonthDate = Annotated[str | None, AfterValidator(_month_date)]
ExperienceKind = Literal["stage", "alternance", "emploi", "projet", "associatif", "autre"]


class Experience(BaseModel):
    title: ShortText = Field(description="Intitulé du poste ou du projet")
    organization: ShortText | None = Field(default=None, description="Entreprise, association ou école")
    kind: ExperienceKind = Field(description="Type d'expérience ; « autre » en cas de doute")
    start: MonthDate = Field(
        default=None, description="Début au format AAAA-MM, ou AAAA si le mois est inconnu"
    )
    end: MonthDate = Field(default=None, description="Fin au format AAAA-MM ou AAAA, ou « en cours »")
    duration_months: int | None = Field(default=None, description="Durée en mois si elle est déductible")
    description: LongText = Field(description="Missions et réalisations, en une ou deux phrases")
    skills: Annotated[list[ShortText], _cap(15)] = Field(default_factory=list)


class Language(BaseModel):
    name: ShortText
    level: ShortText | None = Field(default=None, description="Ex. : natif, C1, courant, scolaire")


def _radius(value: int | None) -> int | None:
    return value if value is not None and 0 < value <= 1000 else None


class Mobility(BaseModel):
    city: ShortText | None = Field(default=None, description="Ville de référence de la recherche")
    radius_km: Annotated[int | None, AfterValidator(_radius)] = Field(
        default=None, description="Rayon de mobilité en kilomètres, s'il est indiqué"
    )


class CandidateProfile(BaseModel):
    """Profil structuré du candidat. Ne contient aucune coordonnée personnelle."""

    headline: ShortText = Field(description="Titre de profil, ex. « Étudiant ingénieur data en 3e année »")
    summary: LongText = Field(description="Résumé du profil en 2 ou 3 phrases, à la 3e personne")
    target_roles: Annotated[list[ShortText], _cap(5)] = Field(
        default_factory=list, description="Postes visés, déduits du CV"
    )
    education: Annotated[list[Education], _cap(10)] = Field(default_factory=list)
    experiences: Annotated[list[Experience], _cap(20)] = Field(default_factory=list)
    hard_skills: Annotated[list[ShortText], _cap(40)] = Field(
        default_factory=list, description="Compétences techniques : langages, outils, méthodes"
    )
    soft_skills: Annotated[list[ShortText], _cap(15)] = Field(default_factory=list)
    languages: Annotated[list[Language], _cap(10)] = Field(default_factory=list)
    certifications: Annotated[list[ShortText], _cap(15)] = Field(default_factory=list)
    interests: Annotated[list[ShortText], _cap(15)] = Field(default_factory=list)
    mobility: Mobility | None = Field(default=None, description="Zone géographique de recherche")
    driving_licence: bool | None = Field(default=None, description="Permis de conduire mentionné")
    has_vehicle: bool | None = Field(default=None, description="Véhicule personnel mentionné")


# --------------------------------------------------------------------------
# Enrichissement des entreprises
# --------------------------------------------------------------------------


class CompanyInsight(BaseModel):
    """Résumé d'une entreprise produit par le LLM à partir de son site web."""

    summary: LongText = Field(description="Activité de l'entreprise en 2 ou 3 phrases factuelles")
    sector: ShortText | None = Field(default=None, description="Secteur d'activité principal")
    keywords: Annotated[list[ShortText], _cap(10)] = Field(
        default_factory=list, description="Produits, technologies ou métiers mis en avant"
    )
    mentions_apprenticeship: bool = Field(
        default=False, description="Le site parle d'alternance, d'apprentissage ou de stages"
    )


EnrichmentStatus = Literal["ok", "no_website", "failed"]


class Enrichment(BaseModel):
    """Ce qu'Avenly sait d'une entreprise (partagé entre tous les clients)."""

    siren: str = Field(pattern=r"^\d{9}$")
    status: EnrichmentStatus
    website: str | None = None
    website_verified: bool = False  # site confirmé par le SIREN ou le nom de l'entreprise
    careers_url: str | None = None
    generic_emails: Annotated[list[str], _cap(10)] = Field(default_factory=list)
    insight: CompanyInsight | None = None

    @field_validator("website", "careers_url", mode="before")
    @classmethod
    def _safe_urls(cls, value: object) -> str | None:
        return Company._safe_url(value)


# --------------------------------------------------------------------------
# Candidature
# --------------------------------------------------------------------------


class ApplicationContext(BaseModel):
    """Ce que le candidat recherche (saisi par lui, pas déduit du CV)."""

    current_status: ShortText | None = Field(
        default=None,
        description="Situation actuelle, prioritaire sur le CV, ex. « 3e année cycle ingénieur InSI »",
    )
    start: ShortText = Field(description="Début souhaité de l'alternance, ex. « septembre 2027 »")
    rhythm: ShortText | None = Field(
        default=None, description="Rythme, ex. « 3 semaines entreprise / 1 semaine école »"
    )
    duration: ShortText | None = Field(default=None, description="Durée du contrat, ex. « 3 ans »")


class Signature(BaseModel):
    """Coordonnées du candidat : ajoutées localement, JAMAIS envoyées au LLM."""

    full_name: ShortText = Field(min_length=2)
    email: ShortText | None = None
    phone: ShortText | None = None
    linkedin_url: str | None = None

    @field_validator("linkedin_url", mode="before")
    @classmethod
    def _safe_linkedin(cls, value: object) -> str | None:
        return Company._safe_url(value)

    def render(self) -> str:
        lines = [self.full_name, self.phone, self.email, self.linkedin_url]
        return "\n".join(line for line in lines if line)


class EmailContent(BaseModel):
    """Email produit par le LLM, avant ajout de la signature."""

    subject: ShortText = Field(description="Objet de l'email, court et précis (moins de 80 caractères)")
    body: Annotated[CleanMultilineStr, _truncate(2000)] = Field(
        description="Corps de l'email, salutation et formule de politesse comprises, SANS signature"
    )
