"""Tests du sourcing, sans réseau : les APIs sont simulées.

Les réponses simulées suivent le format des documentations officielles
(OpenAPI La Bonne Alternance et Recherche d'entreprises).
"""

from __future__ import annotations

from typing import Any

import pytest

from avenly.config import get_settings
from avenly.models import Company, GeoPoint
from avenly.sourcing import (
    ApiClient,
    CompanySource,
    Geocoder,
    LaBonneAlternanceSource,
    RechercheEntreprisesSource,
    SearchQuery,
    SourcingEngine,
    SourcingError,
    find_companies,
    get_domain,
)

LYON = GeoPoint(latitude=45.76, longitude=4.83)


class FakeClient:
    """Remplace ApiClient : renvoie des réponses préparées, enregistre les appels."""

    def __init__(self, *responses: dict[str, Any]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get_json(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((path, params))
        return self._responses.pop(0)


def make_query(**overrides) -> SearchQuery:
    fields = {
        "client_id": "client-001",
        "location": LYON,
        "radius_km": 10,
        "romes": ("M1805",),
        "naf_codes": ("62.01Z",),
    }
    return SearchQuery(**(fields | overrides))


# --------------------------------------------------------------------------
# Validation des entrées
# --------------------------------------------------------------------------


def test_unknown_domain():
    with pytest.raises(ValueError, match="Domaine inconnu"):
        get_domain("plomberie")


def test_domain_is_case_insensitive():
    assert get_domain(" Informatique ").romes


@pytest.mark.parametrize(
    "overrides",
    [
        {"radius_km": 500},
        {"radius_km": 0},
        {"max_results": 0},
        {"romes": ("DROP TABLE",)},
        {"naf_codes": ("62.01Z; rm -rf",)},
    ],
)
def test_invalid_query_is_rejected(overrides):
    with pytest.raises(ValueError):
        make_query(**overrides)


def test_invalid_city_is_rejected_before_any_call():
    client = FakeClient()
    with pytest.raises(ValueError):
        Geocoder(client).locate("Lyon; rm -rf /")
    assert client.calls == []


# --------------------------------------------------------------------------
# Géocodage
# --------------------------------------------------------------------------


def test_geocoder_returns_coordinates():
    client = FakeClient(
        {"features": [{"geometry": {"coordinates": [4.835, 45.758]}, "properties": {"score": 0.96}}]}
    )
    point = Geocoder(client).locate("Lyon")
    assert (point.latitude, point.longitude) == (45.758, 4.835)


def test_geocoder_unknown_city():
    with pytest.raises(SourcingError, match="introuvable"):
        Geocoder(FakeClient({"features": []})).locate("Zzzzqqq")


# --------------------------------------------------------------------------
# La Bonne Alternance
# --------------------------------------------------------------------------

LBA_RESPONSE = {
    "recruiters": [
        {
            "identifier": {"id": "r1"},
            "workplace": {
                "siret": "12345678901234",
                "brand": "Marque​ Test",
                "legal_name": "TEST SAS",
                "website": "www.test.fr",
                "size": "6-9",
                "location": {
                    "address": "1 rue X 69001 Lyon",
                    "geopoint": {"type": "Point", "coordinates": [4.83, 45.76]},
                },
                "domain": {"naf": {"code": "6201Z", "label": "Programmation"}},
            },
            "apply": {"url": "https://labonnealternance.apprentissage.beta.gouv.fr/x", "phone": "0600000000"},
        }
    ],
    "jobs": [
        {"workplace": {"siret": "123", "name": "SIRET invalide"}},
        {"workplace": {"siret": None, "name": "Sans SIRET"}},
        {
            "workplace": {
                "siret": "98765432100012",
                "legal_name": "OFFRE SA",
                "website": "javascript:alert(1)",
                "location": {"address": "Lyon"},
            },
            "apply": {"url": "https://exemple.fr/postuler"},
        },
    ],
}


def test_lba_parses_recruiters_and_jobs():
    client = FakeClient(LBA_RESPONSE)
    companies = LaBonneAlternanceSource(client).search(make_query())

    assert [c.siret for c in companies] == ["12345678901234", "98765432100012"]
    recruiter, job = companies
    assert recruiter.name == "Marque Test"
    assert recruiter.naf_code == "62.01Z"
    assert recruiter.website == "https://www.test.fr"
    assert recruiter.location == GeoPoint(latitude=45.76, longitude=4.83)
    assert recruiter.has_job_offer is False
    assert job.has_job_offer is True
    assert job.website is None  # javascript: rejeté

    path, params = client.calls[0]
    assert path == "/job/v1/search"
    assert params["romes"] == "M1805"


def test_lba_skipped_without_romes():
    client = FakeClient()
    assert LaBonneAlternanceSource(client).search(make_query(romes=())) == []
    assert client.calls == []


def test_lba_radius_is_capped():
    client = FakeClient({"recruiters": [], "jobs": []})
    LaBonneAlternanceSource(client).search(make_query(radius_km=200))
    assert client.calls[0][1]["radius"] == 200


# --------------------------------------------------------------------------
# Recherche d'entreprises
# --------------------------------------------------------------------------


def etab(siret: str, **overrides) -> dict[str, Any]:
    fields = {
        "siret": siret,
        "adresse": "34 RUE DE LA SOIE 69100 VILLEURBANNE",
        "etat_administratif": "A",
        "statut_diffusion_etablissement": "O",
        "tranche_effectif_salarie": "32",
        "latitude": "45.77",
        "longitude": "4.88",
        "activite_principale": "66.30Z",
    }
    return fields | overrides


def unit(name: str, *etabs: dict[str, Any], **overrides) -> dict[str, Any]:
    fields = {
        "nom_complet": name,
        "etat_administratif": "A",
        "statut_diffusion": "O",
        "activite_principale": "62.02A",
        "dirigeants": [{"nom": "SECRET", "prenoms": "PERSONNE", "date_de_naissance": "1970-01"}],
        "matching_etablissements": list(etabs),
    }
    return fields | overrides


def test_recherche_entreprises_filters_and_parses():
    response = {
        "total_pages": 1,
        "results": [
            unit(
                "ATOS FRANCE",
                etab("40802471900697"),
                etab("40802471900001", etat_administratif="F"),  # fermé
                etab("40802471900002", tranche_effectif_salarie="NN"),  # sans salarié
                etab("40802471900003", statut_diffusion_etablissement="P"),  # diffusion restreinte
            ),
            unit("CESSEE SA", etab("11111111100011"), etat_administratif="C"),
            unit("DISCRETE", etab("22222222200022"), statut_diffusion="P"),
        ],
    }
    companies = RechercheEntreprisesSource(FakeClient(response)).search(make_query())

    assert len(companies) == 1
    company = companies[0]
    assert company.siret == "40802471900697"
    assert company.name == "ATOS FRANCE"
    assert company.naf_code == "62.02A"  # code de l'entreprise, pas de l'établissement
    assert company.headcount == "250-499"
    assert "SECRET" not in company.model_dump_json()  # dirigeants jamais lus


def test_recherche_entreprises_keeps_non_employers_if_asked():
    response = {
        "total_pages": 1,
        "results": [unit("SOLO", etab("33333333300033", tranche_effectif_salarie="NN"))],
    }
    source = RechercheEntreprisesSource(FakeClient(response), employers_only=False)
    assert len(source.search(make_query())) == 1


def test_recherche_entreprises_paginates_until_max_results():
    def page(n: int) -> dict[str, Any]:
        return {"total_pages": 10, "results": [unit(f"E{n}{i}", etab(f"{n:07d}{i:07d}")) for i in range(25)]}

    client = FakeClient(page(1), page(2), page(3))
    companies = RechercheEntreprisesSource(client).search(make_query(max_results=60))

    assert len(companies) == 60
    assert [params["page"] for _, params in client.calls] == [1, 2, 3]
    assert client.calls[0][1]["radius"] == 10


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


class StaticSource(CompanySource):
    def __init__(self, name: str, companies: list[Company] | None = None, fail: bool = False) -> None:
        self.name = name  # type: ignore[misc]
        self._companies = companies or []
        self._fail = fail

    def search(self, query: SearchQuery) -> list[Company]:
        if self._fail:
            raise SourcingError("panne simulée")
        return self._companies


def company(siret: str, name: str, source: str, **extra) -> Company:
    return Company(client_id="client-001", siret=siret, name=name, sources={source}, **extra)


@pytest.fixture
def engine() -> SourcingEngine:
    engine = SourcingEngine(get_settings())
    yield engine
    engine.close()


def test_engine_merges_by_siret_and_ranks_hidden_market_first(engine):
    engine.sources = [
        StaticSource(
            "recherche_entreprises",
            [
                company("11111111111111", "AAA", "recherche_entreprises"),
                company("22222222222222", "BBB", "x"),
            ],
        ),
        StaticSource(
            "la_bonne_alternance",
            [company("22222222222222", "BBB", "la_bonne_alternance", website="https://bbb.fr")],
        ),
    ]
    results = engine.search(make_query())

    assert [c.name for c in results] == ["BBB", "AAA"]  # recruteur LBA en tête
    assert results[0].website == "https://bbb.fr"
    assert results[0].sources == {"x", "la_bonne_alternance"}


def test_engine_survives_one_failing_source(engine):
    engine.sources = [
        StaticSource("ko", fail=True),
        StaticSource("ok", [company("11111111111111", "A", "ok")]),
    ]
    assert len(engine.search(make_query())) == 1


def test_engine_fails_if_all_sources_fail(engine):
    engine.sources = [StaticSource("ko1", fail=True), StaticSource("ko2", fail=True)]
    with pytest.raises(SourcingError, match="Toutes les sources"):
        engine.search(make_query())


def test_engine_disables_lba_without_token():
    with SourcingEngine(get_settings()) as engine:
        assert [s.name for s in engine.sources] == ["recherche_entreprises"]


def test_engine_enables_lba_with_token(monkeypatch):
    monkeypatch.setenv("LBA_API_TOKEN", "faux-jeton")
    with SourcingEngine(get_settings()) as engine:
        assert "la_bonne_alternance" in [s.name for s in engine.sources]


# --------------------------------------------------------------------------
# Client HTTP
# --------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status_code: int, body: bytes = b"{}") -> None:
        self.status_code = status_code
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_content(self, chunk_size: int):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i : i + chunk_size]


@pytest.fixture
def api_client() -> ApiClient:
    client = ApiClient("https://api.exemple.fr", rate_per_second=1000, settings=get_settings())
    yield client
    client.close()


@pytest.mark.parametrize(
    ("status", "body", "message"),
    [
        (401, b"", "Accès refusé"),
        (429, b"", "Limite de requêtes"),
        (500, b"", "HTTP 500"),
        (200, b"<html>", "non-JSON"),
        (200, b"[1, 2]", "Format de réponse"),
    ],
)
def test_api_client_errors(monkeypatch, api_client, status, body, message):
    monkeypatch.setattr(api_client._session, "get", lambda *a, **k: FakeResponse(status, body))
    with pytest.raises(SourcingError, match=message):
        api_client.get_json("/x", {})


def test_api_client_rejects_huge_response(monkeypatch, api_client):
    monkeypatch.setattr("avenly.sourcing.MAX_RESPONSE_BYTES", 100)
    monkeypatch.setattr(
        api_client._session, "get", lambda *a, **k: FakeResponse(200, b"{" + b" " * 500 + b"}")
    )
    with pytest.raises(SourcingError, match="trop volumineuse"):
        api_client.get_json("/x", {})


def test_api_client_sends_token_only_when_given(monkeypatch):
    from pydantic import SecretStr

    token = SecretStr("jeton-secret-xyz")
    with_token = ApiClient("https://a.fr", rate_per_second=1, settings=get_settings(), token=token)
    without = ApiClient("https://a.fr", rate_per_second=1, settings=get_settings())
    assert with_token._session.headers["Authorization"] == "Bearer jeton-secret-xyz"
    assert "Authorization" not in without._session.headers
    assert "jeton-secret-xyz" not in repr(with_token)


# --------------------------------------------------------------------------
# Test réel (réseau) : pytest -m network
# --------------------------------------------------------------------------


@pytest.mark.network
def test_live_recherche_entreprises():
    companies = find_companies("informatique", "Lyon", client_id="test", radius_km=10, max_results=30)
    assert len(companies) == 30
    assert len({c.siret for c in companies}) == 30
