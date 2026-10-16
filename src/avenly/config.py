"""Configuration centralisée du moteur, chargée depuis le fichier .env.

Tous les autres modules passent par get_settings() : aucun autre fichier
ne lit os.environ directement.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    SecretStr,
    ValidationError,
    field_validator,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class ConfigError(RuntimeError):
    """Configuration absente ou invalide."""


class Settings(BaseModel):
    """Paramètres typés et validés de l'application (immuables)."""

    model_config = ConfigDict(frozen=True, extra="ignore", validate_default=True)

    # Secrets : SecretStr masque la valeur dans les print/logs/repr.
    fernet_key: SecretStr | None = None
    lba_api_token: SecretStr | None = None

    lba_api_base_url: HttpUrl = "https://api.apprentissage.beta.gouv.fr/api"
    recherche_entreprises_base_url: HttpUrl = "https://recherche-entreprises.api.gouv.fr"
    geocoding_base_url: HttpUrl = "https://data.geopf.fr/geocodage"

    http_timeout_seconds: float = Field(default=10, gt=0, le=60)
    http_max_retries: int = Field(default=3, ge=0, le=10)
    # L'API Recherche d'entreprises bloque au-delà de 7 req/s.
    rate_limit_per_second: float = Field(default=5, gt=0, le=7)

    max_cv_size_mb: int = Field(default=5, gt=0, le=20)

    # Base de données : SQLite local par défaut (dans data/, ignoré par Git).
    # SecretStr : une URL PostgreSQL contient un mot de passe.
    database_url: SecretStr = SecretStr(f"sqlite:///{DATA_DIR / 'avenly.db'}")
    # Une recherche (domaine + zone) est resservie depuis la base pendant N jours (0 = désactivé).
    sourcing_cache_days: int = Field(default=30, ge=0, le=365)
    # Nombre maximal de candidatures Avenly vers une même entreprise sur 30 jours, tous clients confondus.
    max_applications_per_company: int = Field(default=2, ge=1, le=20)

    log_level: LogLevel = "INFO"

    @field_validator("database_url")
    @classmethod
    def _supported_database(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().startswith(("sqlite://", "postgresql://", "postgresql+psycopg://")):
            raise ValueError("seuls SQLite et PostgreSQL sont supportés")
        return value

    @field_validator("fernet_key", "lba_api_token", mode="before")
    @classmethod
    def _empty_secret_is_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_log_level(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value

    @property
    def max_cv_size_bytes(self) -> int:
        return self.max_cv_size_mb * 1024 * 1024


def _format_errors(exc: ValidationError) -> str:
    """Résume les erreurs SANS les valeurs saisies (elles peuvent être des secrets)."""
    lines = []
    for error in exc.errors(include_input=False, include_url=False):
        field = ".".join(str(part) for part in error["loc"]).upper()
        lines.append(f"  - {field} : {error['msg']}")
    return "\n".join(lines)


@lru_cache(maxsize=1)
def get_settings(env_file: Path | None = None) -> Settings:
    """Charge le .env puis renvoie les paramètres validés (mis en cache).

    Les variables déjà présentes dans l'environnement ne sont pas écrasées
    par le .env.
    """
    load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)

    # Une variable vide (ex. « DATABASE_URL= ») garde sa valeur par défaut.
    raw = {
        name: os.environ[name.upper()]
        for name in Settings.model_fields
        if os.environ.get(name.upper(), "").strip()
    }
    try:
        return Settings(**raw)
    except ValidationError as exc:
        raise ConfigError(f"Configuration invalide dans .env :\n{_format_errors(exc)}") from None
