"""Base de données : catalogue d'entreprises partagé, historique des candidatures, opposition.

Ce qui est partagé entre tous les clients (données publiques) :
- `companies` : catalogue des établissements trouvés, enrichi au fil des recherches ;
- `searches` / `search_results` : recherches déjà faites (domaine + zone), resservies
  depuis la base tant qu'elles sont récentes ;
- `opt_outs` : entreprises qui ne veulent plus être contactées, exclues pour tout le monde.

Ce qui est propre à chaque client :
- `applications` : qui a contacté quelle entreprise, et quand.

Toutes les requêtes passent par SQLAlchemy (requêtes paramétrées : pas d'injection SQL).
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    TypeDecorator,
    UniqueConstraint,
    create_engine,
    func,
    select,
)
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from avenly.config import Settings, get_settings
from avenly.models import Company, CompanyInsight, Enrichment, GeoPoint
from avenly.security import CryptoManager

EARTH_RADIUS_KM = 6371.0


def utcnow() -> datetime:
    return datetime.now(UTC)


def distance_km(a: GeoPoint, b: GeoPoint) -> float:
    """Distance à vol d'oiseau (formule de haversine)."""
    lat1, lat2 = math.radians(a.latitude), math.radians(b.latitude)
    dlat = lat2 - lat1
    dlon = math.radians(b.longitude - a.longitude)
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(h))


class UTCDateTime(TypeDecorator[datetime]):
    """Stocke les dates en UTC ; SQLite ne conservant pas le fuseau, on le remet à la lecture."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("Date sans fuseau horaire refusée : utiliser utcnow().")
        return value.astimezone(UTC).replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect) -> datetime | None:
        return value.replace(tzinfo=UTC) if value is not None else None


class ApplicationStatus(StrEnum):
    DRAFT = "draft"
    SENT = "sent"
    REPLIED = "replied"
    REFUSED = "refused"


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------


class Base(DeclarativeBase):
    pass


class CompanyRecord(Base):
    __tablename__ = "companies"

    siret: Mapped[str] = mapped_column(String(14), primary_key=True)
    siren: Mapped[str] = mapped_column(String(9), index=True)
    name: Mapped[str] = mapped_column(String(200))
    naf_code: Mapped[str | None] = mapped_column(String(6), index=True)
    address: Mapped[str | None] = mapped_column(String(300))
    latitude: Mapped[float | None] = mapped_column(Float)
    longitude: Mapped[float | None] = mapped_column(Float)
    headcount: Mapped[str | None] = mapped_column(String(50))
    website: Mapped[str | None] = mapped_column(String(500))
    apply_url: Mapped[str | None] = mapped_column(String(500))
    has_job_offer: Mapped[bool] = mapped_column(Boolean, default=False)
    sources: Mapped[list[str]] = mapped_column(JSON, default=list)
    first_seen_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)

    def to_company(self, client_id: str) -> Company:
        location = (
            GeoPoint(latitude=self.latitude, longitude=self.longitude)
            if self.latitude is not None and self.longitude is not None
            else None
        )
        return Company(
            client_id=client_id,
            siret=self.siret,
            name=self.name,
            naf_code=self.naf_code,
            address=self.address,
            location=location,
            headcount=self.headcount,
            website=self.website,
            apply_url=self.apply_url,
            has_job_offer=self.has_job_offer,
            sources=set(self.sources),
        )


class SearchRecord(Base):
    __tablename__ = "searches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    domain: Mapped[str] = mapped_column(String(50), index=True)
    latitude: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float)
    radius_km: Mapped[float] = mapped_column(Float)
    requested_max: Mapped[int] = mapped_column(Integer)
    result_count: Mapped[int] = mapped_column(Integer)
    fetched_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, index=True)


class SearchResultRecord(Base):
    __tablename__ = "search_results"

    search_id: Mapped[int] = mapped_column(ForeignKey("searches.id", ondelete="CASCADE"), primary_key=True)
    siret: Mapped[str] = mapped_column(ForeignKey("companies.siret"), primary_key=True)


class ApplicationRecord(Base):
    __tablename__ = "applications"
    __table_args__ = (UniqueConstraint("client_id", "siret", name="uq_application_client_company"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    client_id: Mapped[str] = mapped_column(String(64), index=True)
    siret: Mapped[str] = mapped_column(ForeignKey("companies.siret"), index=True)
    status: Mapped[str] = mapped_column(String(20), default=ApplicationStatus.DRAFT)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class OptOutRecord(Base):
    __tablename__ = "opt_outs"

    # Par SIREN : une opposition vaut pour tous les établissements de l'entreprise.
    siren: Mapped[str] = mapped_column(String(9), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class LLMUsageRecord(Base):
    """Coût réel de chaque appel LLM : sert aux budgets mensuels (global et par client)."""

    __tablename__ = "llm_usage"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    client_id: Mapped[str] = mapped_column(String(64), index=True)
    task: Mapped[str] = mapped_column(String(50))
    model: Mapped[str] = mapped_column(String(50))
    input_tokens: Mapped[int] = mapped_column(Integer)
    output_tokens: Mapped[int] = mapped_column(Integer)
    cache_read_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cache_write_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, index=True)


class EnrichmentRecord(Base):
    """Résultat de l'enrichissement d'une entreprise (par SIREN), partagé entre tous les clients."""

    __tablename__ = "enrichments"

    siren: Mapped[str] = mapped_column(String(9), primary_key=True)
    status: Mapped[str] = mapped_column(String(20))
    website: Mapped[str | None] = mapped_column(String(500))
    website_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    careers_url: Mapped[str | None] = mapped_column(String(500))
    generic_emails: Mapped[list[str]] = mapped_column(JSON, default=list)
    insight: Mapped[dict | None] = mapped_column(JSON)
    enriched_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, index=True)

    def to_enrichment(self) -> Enrichment:
        return Enrichment(
            siren=self.siren,
            status=self.status,
            website=self.website,
            website_verified=self.website_verified,
            careers_url=self.careers_url,
            generic_emails=list(self.generic_emails or []),
            insight=CompanyInsight(**self.insight) if self.insight else None,
        )


class DraftRecord(Base):
    """Brouillon d'email. Objet et corps CHIFFRÉS (Fernet) : une lettre de motivation est personnelle."""

    __tablename__ = "drafts"
    __table_args__ = (UniqueConstraint("client_id", "siret", name="uq_draft_client_company"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    client_id: Mapped[str] = mapped_column(String(64), index=True)
    siret: Mapped[str] = mapped_column(ForeignKey("companies.siret"), index=True)
    subject_encrypted: Mapped[bytes] = mapped_column(LargeBinary)
    body_encrypted: Mapped[bytes] = mapped_column(LargeBinary)
    recipient_email: Mapped[str | None] = mapped_column(String(320))
    careers_url: Mapped[str | None] = mapped_column(String(500))
    model: Mapped[str] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, index=True)


def start_of_month(now: datetime | None = None) -> datetime:
    now = now or utcnow()
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


# --------------------------------------------------------------------------
# Accès à la base
# --------------------------------------------------------------------------


class Database:
    """Point d'accès unique à la base (catalogue, cache de recherches, candidatures, opposition)."""

    def __init__(self, url: str | None = None, *, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        url = url or self._settings.database_url.get_secret_value()
        self._prepare_sqlite_file(url)
        self.engine: Engine = create_engine(url, future=True)
        self._sessions = sessionmaker(self.engine, expire_on_commit=False)
        Base.metadata.create_all(self.engine)

    def __repr__(self) -> str:
        # L'URL peut contenir un mot de passe : on n'affiche que le type de base.
        return f"Database(backend={self.engine.dialect.name!r})"

    @staticmethod
    def _prepare_sqlite_file(url: str) -> None:
        """Crée le fichier SQLite avec des droits restreints (lecture/écriture pour toi seul)."""
        parsed = make_url(url)
        if parsed.get_backend_name() != "sqlite" or not parsed.database or parsed.database == ":memory:":
            return
        path = Path(parsed.database)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.touch(mode=0o600)
        os.chmod(path, 0o600)

    @contextmanager
    def session(self) -> Iterator[Session]:
        with self._sessions() as session, session.begin():
            yield session

    def close(self) -> None:
        self.engine.dispose()

    # ------------------------------------------------------------------ catalogue

    def upsert_companies(self, companies: Iterable[Company], session: Session | None = None) -> None:
        """Ajoute ou met à jour les établissements (les champs vides sont complétés, jamais effacés)."""
        if session is None:
            with self.session() as new_session:
                self.upsert_companies(companies, new_session)
            return

        now = utcnow()
        for company in companies:
            record = session.get(CompanyRecord, company.siret)
            if record is None:
                record = CompanyRecord(siret=company.siret, siren=company.siren, first_seen_at=now)
                session.add(record)
            record.name = company.name or record.name
            for field in ("naf_code", "address", "headcount", "website", "apply_url"):
                value = getattr(company, field)
                if value is not None:
                    setattr(record, field, value)
            if company.location is not None:
                record.latitude = company.location.latitude
                record.longitude = company.location.longitude
            record.has_job_offer = bool(record.has_job_offer) or company.has_job_offer
            record.sources = sorted(set(record.sources or []) | company.sources)
            record.last_seen_at = now

    def count_companies(self) -> int:
        with self.session() as session:
            return session.scalar(select(func.count()).select_from(CompanyRecord)) or 0

    # ------------------------------------------------------------------ cache de recherches

    def save_search(
        self,
        domain: str,
        location: GeoPoint,
        radius_km: float,
        companies: list[Company],
        *,
        requested_max: int,
    ) -> None:
        with self.session() as session:
            self.upsert_companies(companies, session)
            search = SearchRecord(
                domain=domain,
                latitude=location.latitude,
                longitude=location.longitude,
                radius_km=radius_km,
                requested_max=requested_max,
                result_count=len(companies),
            )
            session.add(search)
            session.flush()
            session.add_all(
                SearchResultRecord(search_id=search.id, siret=siret)
                for siret in {company.siret for company in companies}
            )

    def find_cached(
        self,
        domain: str,
        location: GeoPoint,
        radius_km: float,
        *,
        client_id: str,
        max_results: int,
    ) -> list[Company] | None:
        """Resservit une recherche récente qui couvre entièrement la zone demandée.

        Renvoie None si aucune recherche en cache ne couvre la zone (il faut alors
        interroger les APIs).
        """
        days = self._settings.sourcing_cache_days
        if days == 0:
            return None
        cutoff = utcnow() - timedelta(days=days)

        with self.session() as session:
            searches = session.scalars(
                select(SearchRecord)
                .where(SearchRecord.domain == domain, SearchRecord.fetched_at >= cutoff)
                .order_by(SearchRecord.fetched_at.desc())
            ).all()
            # La zone en cache doit contenir entièrement la zone demandée.
            covering = [
                s
                for s in searches
                if distance_km(location, GeoPoint(latitude=s.latitude, longitude=s.longitude)) + radius_km
                <= s.radius_km
            ]
            if not covering:
                return None
            # Recherche « complète » : l'API a renvoyé moins que le maximum demandé,
            # donc toutes les entreprises de la zone sont en base.
            complete = any(s.result_count < s.requested_max for s in covering)

            records = session.scalars(
                select(CompanyRecord)
                .join(SearchResultRecord, SearchResultRecord.siret == CompanyRecord.siret)
                .where(SearchResultRecord.search_id.in_([s.id for s in covering]))
                .distinct()
            ).all()

        companies = [
            record.to_company(client_id)
            for record in records
            if record.latitude is not None
            and distance_km(location, GeoPoint(latitude=record.latitude, longitude=record.longitude))
            <= radius_km
        ]
        if len(companies) < max_results and not complete:
            return None  # recherche tronquée et pas assez de résultats : on interroge les APIs
        return companies

    # ------------------------------------------------------------------ opposition

    def add_opt_out(self, siren: str) -> None:
        if not (len(siren) == 9 and siren.isdigit()):
            raise ValueError("SIREN invalide (9 chiffres attendus).")
        with self.session() as session:
            if session.get(OptOutRecord, siren) is None:
                session.add(OptOutRecord(siren=siren))

    def opted_out_sirens(self, sirens: Iterable[str]) -> set[str]:
        sirens = set(sirens)
        if not sirens:
            return set()
        with self.session() as session:
            return set(session.scalars(select(OptOutRecord.siren).where(OptOutRecord.siren.in_(sirens))))

    # ------------------------------------------------------------------ candidatures

    def record_application(
        self, client_id: str, siret: str, status: ApplicationStatus = ApplicationStatus.DRAFT
    ) -> None:
        """Enregistre (ou met à jour) la candidature d'un client vers un établissement."""
        with self.session() as session:
            if session.get(CompanyRecord, siret) is None:
                raise ValueError("Établissement inconnu : il doit d'abord être dans le catalogue.")
            application = session.scalar(
                select(ApplicationRecord).where(
                    ApplicationRecord.client_id == client_id, ApplicationRecord.siret == siret
                )
            )
            if application is None:
                session.add(ApplicationRecord(client_id=client_id, siret=siret, status=status))
            else:
                application.status = status
                application.updated_at = utcnow()

    def contactable(self, companies: Iterable[Company], client_id: str) -> list[Company]:
        """Filtre les établissements qu'un client peut encore contacter.

        Exclut : les entreprises en opposition, celles déjà contactées par ce
        client, et celles qui ont reçu trop de candidatures Avenly ces 30 derniers
        jours (tous clients confondus) pour éviter de les saturer.
        """
        companies = list(companies)
        sirets = {c.siret for c in companies}
        opted_out = self.opted_out_sirens(c.siren for c in companies)
        limit = self._settings.max_applications_per_company
        since = utcnow() - timedelta(days=30)

        with self.session() as session:
            already_contacted = set(
                session.scalars(
                    select(ApplicationRecord.siret).where(
                        ApplicationRecord.client_id == client_id, ApplicationRecord.siret.in_(sirets)
                    )
                )
            )
            # Saturation comptée par entreprise (SIREN), pas par établissement.
            recent = session.execute(
                select(CompanyRecord.siren, func.count())
                .join(ApplicationRecord, ApplicationRecord.siret == CompanyRecord.siret)
                .where(
                    ApplicationRecord.created_at >= since,
                    ApplicationRecord.status != ApplicationStatus.DRAFT,  # un brouillon n'a rien envoyé
                )
                .group_by(CompanyRecord.siren)
            ).all()
        saturated = {siren for siren, count in recent if count >= limit}

        return [
            c
            for c in companies
            if c.siren not in opted_out and c.siret not in already_contacted and c.siren not in saturated
        ]

    # ------------------------------------------------------------------ coûts LLM

    def record_llm_usage(
        self,
        *,
        client_id: str,
        task: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        cost_usd: float,
    ) -> None:
        with self.session() as session:
            session.add(
                LLMUsageRecord(
                    client_id=client_id,
                    task=task,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_read_tokens=cache_read_tokens,
                    cache_write_tokens=cache_write_tokens,
                    cost_usd=cost_usd,
                )
            )

    def llm_spend_this_month(self, client_id: str | None = None) -> float:
        """Dépense LLM du mois en cours, pour un client ou pour tous."""
        query = select(func.coalesce(func.sum(LLMUsageRecord.cost_usd), 0.0)).where(
            LLMUsageRecord.created_at >= start_of_month()
        )
        if client_id is not None:
            query = query.where(LLMUsageRecord.client_id == client_id)
        with self.session() as session:
            return float(session.scalar(query) or 0.0)

    # ------------------------------------------------------------------ enrichissement

    def get_enrichment(self, siren: str, *, max_age_days: int) -> Enrichment | None:
        """Enrichissement en cache s'il a moins de `max_age_days` jours."""
        cutoff = utcnow() - timedelta(days=max_age_days)
        with self.session() as session:
            record = session.get(EnrichmentRecord, siren)
            if record is None or record.enriched_at < cutoff:
                return None
            return record.to_enrichment()

    def save_enrichment(self, enrichment: Enrichment) -> None:
        with self.session() as session:
            record = session.get(EnrichmentRecord, enrichment.siren)
            if record is None:
                record = EnrichmentRecord(siren=enrichment.siren)
                session.add(record)
            record.status = enrichment.status
            record.website = enrichment.website
            record.website_verified = enrichment.website_verified
            record.careers_url = enrichment.careers_url
            record.generic_emails = list(enrichment.generic_emails)
            record.insight = enrichment.insight.model_dump() if enrichment.insight else None
            record.enriched_at = utcnow()

    # ------------------------------------------------------------------ brouillons (chiffrés)

    def save_draft(
        self,
        crypto: CryptoManager,
        *,
        client_id: str,
        siret: str,
        subject: str,
        body: str,
        recipient_email: str | None,
        careers_url: str | None,
        model: str,
    ) -> None:
        """Enregistre (ou remplace) le brouillon d'un client pour un établissement, chiffré."""
        with self.session() as session:
            draft = session.scalar(
                select(DraftRecord).where(DraftRecord.client_id == client_id, DraftRecord.siret == siret)
            )
            if draft is None:
                draft = DraftRecord(client_id=client_id, siret=siret)
                session.add(draft)
            draft.subject_encrypted = crypto.encrypt_text(subject)
            draft.body_encrypted = crypto.encrypt_text(body)
            draft.recipient_email = recipient_email
            draft.careers_url = careers_url
            draft.model = model
            draft.created_at = utcnow()

    def list_drafts(self, crypto: CryptoManager, client_id: str) -> list[dict[str, str | None]]:
        """Brouillons d'un client, déchiffrés (avec le nom de l'entreprise)."""
        with self.session() as session:
            rows = session.execute(
                select(DraftRecord, CompanyRecord.name)
                .join(CompanyRecord, CompanyRecord.siret == DraftRecord.siret)
                .where(DraftRecord.client_id == client_id)
                .order_by(DraftRecord.created_at)
            ).all()
        return [
            {
                "siret": draft.siret,
                "company": name,
                "subject": crypto.decrypt_text(draft.subject_encrypted),
                "body": crypto.decrypt_text(draft.body_encrypted),
                "recipient_email": draft.recipient_email,
                "careers_url": draft.careers_url,
            }
            for draft, name in rows
        ]
