"""Sourcing : trouver les entreprises d'un domaine métier autour d'une ville.

Sources :
- La Bonne Alternance : entreprises qui recrutent en alternance, avec ou sans
  offre publiée (marché caché). Nécessite LBA_API_TOKEN.
- Recherche d'entreprises : tous les établissements actifs d'un secteur (NAF)
  autour d'un point. Publique, sans clé.

Usage :
    from avenly.sourcing import find_companies
    companies = find_companies("informatique", "Lyon", client_id="client-001")
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Any, ClassVar
from urllib.parse import urlsplit

import requests
from pydantic import SecretStr, ValidationError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from avenly.config import Settings, get_settings
from avenly.models import Company, GeoPoint
from avenly.storage import Database, distance_km

logger = logging.getLogger(__name__)

USER_AGENT = "avenly/0.1"
MAX_RESPONSE_BYTES = 10 * 1024 * 1024


class SourcingError(RuntimeError):
    """Échec d'appel à une API de sourcing."""


# --------------------------------------------------------------------------
# Domaines métier -> codes ROME (La Bonne Alternance) et NAF (Recherche d'entreprises)
# --------------------------------------------------------------------------

_ROME = re.compile(r"^[A-N]\d{4}$")
_NAF = re.compile(r"^\d{2}\.\d{2}[A-Z]$")


@dataclass(frozen=True, slots=True)
class Domain:
    label: str
    romes: tuple[str, ...]
    naf_codes: tuple[str, ...]


DOMAINS: dict[str, Domain] = {
    "informatique": Domain(
        "Informatique / Développement",
        romes=("M1805", "M1810", "M1802", "M1806"),
        naf_codes=("62.01Z", "62.02A", "62.03Z", "62.09Z", "63.11Z", "58.29C"),
    ),
    "data": Domain(
        "Data / IA",
        # Data analyst (M1419), data scientist (M1405), data engineer (M1811), développement (M1805)
        romes=("M1419", "M1405", "M1811", "M1805"),
        # Programmation, conseil et services numériques, traitement de données et hébergement.
        # (70.22Z « conseil de gestion » et 73.20Z « études de marché » étaient trop larges.)
        naf_codes=("62.01Z", "62.02A", "62.03Z", "63.11Z", "63.12Z"),
    ),
    "marketing": Domain(
        "Marketing / Communication",
        romes=("M1705", "E1103", "E1101"),
        naf_codes=("73.11Z", "73.20Z", "70.21Z"),
    ),
    "comptabilite": Domain(
        "Comptabilité / Finance",
        romes=("M1203", "M1206", "M1201"),
        naf_codes=("69.20Z",),
    ),
    "rh": Domain(
        "Ressources humaines",
        romes=("M1503", "M1502", "M1501"),
        naf_codes=("78.10Z", "78.20Z", "70.22Z"),
    ),
}


def get_domain(key: str) -> Domain:
    try:
        return DOMAINS[key.strip().lower()]
    except KeyError:
        raise ValueError(f"Domaine inconnu : {key!r}. Domaines disponibles : {', '.join(DOMAINS)}") from None


# --------------------------------------------------------------------------
# Requête de recherche
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SearchQuery:
    client_id: str
    location: GeoPoint
    radius_km: float
    romes: tuple[str, ...] = ()
    naf_codes: tuple[str, ...] = ()
    max_results: int = 100

    def __post_init__(self) -> None:
        if not 0 < self.radius_km <= 200:
            raise ValueError("radius_km doit être entre 0 et 200.")
        if not 1 <= self.max_results <= 1000:
            raise ValueError("max_results doit être entre 1 et 1000.")
        if bad := [c for c in self.romes if not _ROME.match(c)]:
            raise ValueError(f"Codes ROME invalides : {bad}")
        if bad := [c for c in self.naf_codes if not _NAF.match(c)]:
            raise ValueError(f"Codes NAF invalides : {bad}")


# --------------------------------------------------------------------------
# Client HTTP commun
# --------------------------------------------------------------------------


class RateLimiter:
    """Impose un intervalle minimal entre deux appels (thread-safe)."""

    def __init__(self, per_second: float) -> None:
        self._interval = 1.0 / per_second
        self._lock = threading.Lock()
        self._next_call = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next_call:
                time.sleep(self._next_call - now)
                now = self._next_call
            self._next_call = now + self._interval


class ApiClient:
    """GET JSON avec timeout, relances, limitation de débit et taille de réponse bornée."""

    def __init__(
        self,
        base_url: str,
        *,
        rate_per_second: float,
        settings: Settings,
        token: SecretStr | None = None,
    ) -> None:
        self._base_url = str(base_url).rstrip("/")
        self._host = urlsplit(self._base_url).netloc
        self._timeout = settings.http_timeout_seconds
        self._limiter = RateLimiter(rate_per_second)

        retry = Retry(
            total=settings.http_max_retries,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
            raise_on_status=False,
        )
        self._session = requests.Session()
        self._session.mount("https://", HTTPAdapter(max_retries=retry))
        self._session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
        if token is not None:
            self._session.headers["Authorization"] = f"Bearer {token.get_secret_value()}"

    def __repr__(self) -> str:
        return f"ApiClient(host={self._host!r})"

    def get_json(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        self._limiter.wait()
        try:
            response = self._session.get(
                f"{self._base_url}{path}", params=params, timeout=self._timeout, stream=True
            )
        except requests.RequestException as exc:
            raise SourcingError(f"{self._host} injoignable ({type(exc).__name__}).") from None

        with response:
            if response.status_code in (401, 403):
                raise SourcingError(f"Accès refusé par {self._host} (jeton manquant ou invalide).")
            if response.status_code == 429:
                raise SourcingError(f"Limite de requêtes atteinte sur {self._host}.")
            if response.status_code >= 400:
                raise SourcingError(f"{self._host} a répondu HTTP {response.status_code}.")

            body = bytearray()
            for chunk in response.iter_content(chunk_size=65536):
                body.extend(chunk)
                if len(body) > MAX_RESPONSE_BYTES:
                    raise SourcingError(f"Réponse trop volumineuse de {self._host}.")

        try:
            data = json.loads(body)
        except ValueError:
            raise SourcingError(f"Réponse non-JSON de {self._host}.") from None
        if not isinstance(data, dict):
            raise SourcingError(f"Format de réponse inattendu de {self._host}.")
        return data

    def close(self) -> None:
        self._session.close()


# --------------------------------------------------------------------------
# Géocodage : ville -> coordonnées
# --------------------------------------------------------------------------

_CITY = re.compile(r"^[\w\s'’\-]{2,100}$")


class Geocoder:
    """Géocode une commune via l'API Géoplateforme de l'IGN."""

    def __init__(self, client: ApiClient) -> None:
        self._client = client

    def locate(self, city: str) -> GeoPoint:
        city = city.strip()
        if not _CITY.match(city):
            raise ValueError("Nom de ville invalide (lettres, chiffres, espaces, - et ' uniquement).")
        data = self._client.get_json("/search", {"q": city, "type": "municipality", "limit": 1})
        features = data.get("features") or []
        if not features or (features[0].get("properties") or {}).get("score", 0) < 0.5:
            raise SourcingError(f"Ville introuvable : {city}")
        longitude, latitude = features[0]["geometry"]["coordinates"]
        return GeoPoint(latitude=latitude, longitude=longitude)


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------


class CompanySource(ABC):
    """Interface commune : une source renvoie des `Company` pour une `SearchQuery`."""

    name: ClassVar[str]

    @abstractmethod
    def search(self, query: SearchQuery) -> list[Company]: ...


def _build_company(**fields: Any) -> Company | None:
    """Construit une Company ; ignore silencieusement une entrée mal formée."""
    try:
        return Company(**fields)
    except ValidationError:
        logger.debug("Entrée ignorée : format invalide.")
        return None


class LaBonneAlternanceSource(CompanySource):
    name = "la_bonne_alternance"
    MAX_RADIUS_KM = 200

    def __init__(self, client: ApiClient) -> None:
        self._client = client

    def search(self, query: SearchQuery) -> list[Company]:
        if not query.romes:
            return []
        data = self._client.get_json(
            "/job/v1/search",
            {
                "latitude": query.location.latitude,
                "longitude": query.location.longitude,
                "radius": min(query.radius_km, self.MAX_RADIUS_KM),
                "romes": ",".join(query.romes),
            },
        )
        companies = [
            company
            for key, has_offer in (("recruiters", False), ("jobs", True))
            for item in data.get(key) or []
            if (company := self._parse(item, query.client_id, has_offer)) is not None
        ]
        return companies

    def _parse(self, item: dict[str, Any], client_id: str, has_job_offer: bool) -> Company | None:
        workplace = item.get("workplace") or {}
        if not workplace.get("siret"):
            return None
        location = workplace.get("location") or {}
        coordinates = (location.get("geopoint") or {}).get("coordinates") or []
        naf = (workplace.get("domain") or {}).get("naf")
        return _build_company(
            client_id=client_id,
            siret=workplace["siret"],
            name=workplace.get("brand") or workplace.get("legal_name") or workplace.get("name") or "",
            naf_code=naf.get("code") if isinstance(naf, dict) else None,
            address=location.get("address"),
            location=(
                GeoPoint(longitude=coordinates[0], latitude=coordinates[1]) if len(coordinates) == 2 else None
            ),
            headcount=workplace.get("size"),
            website=workplace.get("website"),
            apply_url=(item.get("apply") or {}).get("url"),
            has_job_offer=has_job_offer,
            sources={self.name},
        )


# Tranches d'effectif INSEE (établissement).
_HEADCOUNTS = {
    "01": "1-2", "02": "3-5", "03": "6-9", "11": "10-19", "12": "20-49",
    "21": "50-99", "22": "100-199", "31": "200-249", "32": "250-499",
    "41": "500-999", "42": "1000-1999", "51": "2000-4999", "52": "5000-9999",
    "53": "10000+",
}  # fmt: skip
_NO_EMPLOYEE = {"NN", "00"}


class RechercheEntreprisesSource(CompanySource):
    name = "recherche_entreprises"
    MAX_RADIUS_KM = 50
    PER_PAGE = 25

    def __init__(self, client: ApiClient, *, employers_only: bool = True) -> None:
        self._client = client
        self._employers_only = employers_only

    def search(self, query: SearchQuery) -> list[Company]:
        if not query.naf_codes:
            return []
        companies: list[Company] = []
        page = 1
        while len(companies) < query.max_results:
            data = self._client.get_json(
                "/near_point",
                {
                    "lat": query.location.latitude,
                    "long": query.location.longitude,
                    "radius": min(query.radius_km, self.MAX_RADIUS_KM),
                    "activite_principale": ",".join(query.naf_codes),
                    "per_page": self.PER_PAGE,
                    "page": page,
                    "limite_matching_etablissements": 5,
                    # Les entreprises les plus grandes d'abord : plus de chances d'accueillir un alternant.
                    "sort_by_size": "true",
                },
            )
            for unit in data.get("results") or []:
                companies.extend(self._parse(unit, query.client_id))
            if page >= int(data.get("total_pages") or 0):
                break
            page += 1
        return companies[: query.max_results]

    def _parse(self, unit: dict[str, Any], client_id: str) -> Iterable[Company]:
        # Entreprise cessée, ou diffusion restreinte (personne ayant demandé
        # à ne pas apparaître dans SIRENE) : on n'y touche pas.
        if unit.get("etat_administratif") != "A" or unit.get("statut_diffusion") != "O":
            return
        name = unit.get("nom_complet") or unit.get("nom_raison_sociale") or ""
        # Les champs « dirigeants » (noms, dates de naissance) ne sont volontairement pas lus.
        for etab in unit.get("matching_etablissements") or []:
            if etab.get("etat_administratif") != "A" or etab.get("statut_diffusion_etablissement") != "O":
                continue
            headcount_code = etab.get("tranche_effectif_salarie")
            if self._employers_only and headcount_code in _NO_EMPLOYEE:
                continue
            try:
                location = GeoPoint(latitude=float(etab["latitude"]), longitude=float(etab["longitude"]))
            except (KeyError, TypeError, ValueError):
                location = None
            company = _build_company(
                client_id=client_id,
                siret=etab.get("siret") or "",
                name=name,
                # Code de l'entreprise (c'est sur lui que porte le filtre NAF de l'API).
                naf_code=unit.get("activite_principale") or etab.get("activite_principale"),
                address=etab.get("adresse"),
                location=location,
                headcount=_HEADCOUNTS.get(headcount_code or ""),
                sources={self.name},
            )
            if company is not None:
                yield company


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


class SourcingEngine:
    """Interroge toutes les sources configurées et fusionne les résultats par SIRET."""

    # La Bonne Alternance : 60 appels/minute documentés.
    LBA_RATE_PER_SECOND = 1.0

    def __init__(self, settings: Settings | None = None, *, database: Database | None = None) -> None:
        self._settings = settings or get_settings()
        self.database = database
        s = self._settings
        self._clients: list[ApiClient] = []

        geo_client = self._client(str(s.geocoding_base_url), s.rate_limit_per_second)
        self.geocoder = Geocoder(geo_client)

        self.sources: list[CompanySource] = []
        if s.lba_api_token is not None:
            lba_client = self._client(
                str(s.lba_api_base_url), self.LBA_RATE_PER_SECOND, token=s.lba_api_token
            )
            self.sources.append(LaBonneAlternanceSource(lba_client))
        else:
            logger.warning("LBA_API_TOKEN absent : La Bonne Alternance est désactivée.")
        re_client = self._client(str(s.recherche_entreprises_base_url), s.rate_limit_per_second)
        self.sources.append(RechercheEntreprisesSource(re_client))

    def _client(self, base_url: str, rate: float, token: SecretStr | None = None) -> ApiClient:
        client = ApiClient(base_url, rate_per_second=rate, settings=self._settings, token=token)
        self._clients.append(client)
        return client

    def __enter__(self) -> SourcingEngine:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        for client in self._clients:
            client.close()

    def find(
        self,
        domain: str,
        city: str,
        *,
        client_id: str,
        radius_km: float = 20,
        max_results: int = 100,
    ) -> list[Company]:
        target = get_domain(domain)
        # La clé de cache inclut les codes du domaine : s'ils changent, l'ancien cache est ignoré.
        codes = ",".join((*target.romes, *target.naf_codes))
        domain_key = f"{domain.strip().lower()}#{hashlib.sha256(codes.encode()).hexdigest()[:8]}"
        query = SearchQuery(
            client_id=client_id,
            location=self.geocoder.locate(city),
            radius_km=radius_km,
            romes=target.romes,
            naf_codes=target.naf_codes,
            max_results=max_results,
        )
        if self.database is None:
            return self.search(query)

        companies = self.database.find_cached(
            domain_key, query.location, radius_km, client_id=client_id, max_results=max_results
        )
        if companies is not None:
            logger.info("Recherche servie depuis la base : %d établissements.", len(companies))
        else:
            companies = self.search(query)
            self.database.save_search(
                domain_key, query.location, radius_km, companies, requested_max=max_results
            )
        # Les entreprises qui ont demandé à ne plus être contactées sont exclues pour tous.
        opted_out = self.database.opted_out_sirens(c.siren for c in companies)
        kept = (c for c in companies if c.siren not in opted_out)
        return rank_companies(kept, query.location)[:max_results]

    def search(self, query: SearchQuery) -> list[Company]:
        merged: dict[str, Company] = {}
        failures = 0
        for source in self.sources:
            try:
                found = source.search(query)
            except SourcingError as exc:
                failures += 1
                logger.warning("Source %s en échec : %s", source.name, exc)
                continue
            logger.info("Source %s : %d établissements.", source.name, len(found))
            for company in found:
                existing = merged.get(company.siret)
                merged[company.siret] = existing.merge(company) if existing else company

        if failures == len(self.sources):
            raise SourcingError("Toutes les sources ont échoué.")

        return rank_companies(merged.values(), query.location)[: query.max_results]


def headcount_floor(headcount: str | None) -> int:
    """« 250-499 » -> 250, « 10000+ » -> 10000, inconnu -> 0."""
    match = re.match(r"\s*(\d+)", headcount or "")
    return int(match.group(1)) if match else 0


def rank_companies(companies: Iterable[Company], origin: GeoPoint | None = None) -> list[Company]:
    """Classe les entreprises par chances de recruter un alternant.

    1. Recruteurs La Bonne Alternance (ils déclarent recruter en alternance) ;
    2. taille de l'établissement : une structure de 50 personnes accueille bien plus
       souvent un alternant qu'une structure de 2 personnes ;
    3. proximité (si le point de départ est connu) ; 4. nom, pour un ordre stable.
    """

    def distance(company: Company) -> float:
        if origin is None or company.location is None:
            return 0.0
        return distance_km(origin, company.location)

    return sorted(
        companies,
        key=lambda c: (
            LaBonneAlternanceSource.name not in c.sources,
            -min(headcount_floor(c.headcount), 1000),  # au-delà de 1000, la taille ne départage plus
            round(distance(c)),
            c.name.lower(),
        ),
    )


def find_companies(
    domain: str,
    city: str,
    *,
    client_id: str,
    radius_km: float = 20,
    max_results: int = 100,
    settings: Settings | None = None,
    database: Database | None = None,
) -> list[Company]:
    """Raccourci : entreprises d'un domaine métier autour d'une ville.

    Avec `database`, les recherches récentes sont resservies depuis la base et
    les entreprises en opposition sont exclues.
    """
    with SourcingEngine(settings, database=database) as engine:
        return engine.find(domain, city, client_id=client_id, radius_km=radius_km, max_results=max_results)


__all__: Sequence[str] = (
    "DOMAINS",
    "Company",
    "CompanySource",
    "Domain",
    "LaBonneAlternanceSource",
    "RechercheEntreprisesSource",
    "SearchQuery",
    "SourcingEngine",
    "SourcingError",
    "find_companies",
    "get_domain",
    "rank_companies",
)
