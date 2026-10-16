from __future__ import annotations

import stat
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import update

from avenly.config import get_settings
from avenly.models import Company, GeoPoint
from avenly.sourcing import CompanySource, SearchQuery, SourcingEngine
from avenly.storage import (
    ApplicationStatus,
    CompanyRecord,
    Database,
    SearchRecord,
    distance_km,
)

LYON = GeoPoint(latitude=45.7640, longitude=4.8357)
VILLEURBANNE = GeoPoint(latitude=45.7719, longitude=4.8902)  # ~4 km de Lyon
PARIS = GeoPoint(latitude=48.8566, longitude=2.3522)


@pytest.fixture
def db(tmp_path) -> Database:
    database = Database(f"sqlite:///{tmp_path / 'avenly.db'}")
    yield database
    database.close()


def company(siret: str, *, location: GeoPoint = LYON, client_id: str = "client-001", **extra) -> Company:
    fields = {"name": f"Entreprise {siret}", "sources": {"recherche_entreprises"}}
    return Company(client_id=client_id, siret=siret, location=location, **(fields | extra))


def companies_around(point: GeoPoint, count: int, prefix: str = "1") -> list[Company]:
    # SIREN (9 chiffres) différent pour chaque entreprise, puis 5 chiffres d'établissement.
    return [company(f"{prefix}{i:08d}00011", location=point) for i in range(count)]


# --------------------------------------------------------------------------
# Outils
# --------------------------------------------------------------------------


def test_distance_lyon_paris():
    assert 385 < distance_km(LYON, PARIS) < 400


def test_sqlite_file_is_private(tmp_path):
    path = tmp_path / "sub" / "avenly.db"
    Database(f"sqlite:///{path}").close()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_repr_does_not_show_url(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'secret-name.db'}")
    assert "secret-name" not in repr(database)
    database.close()


def test_naive_datetime_is_rejected(db):
    db.upsert_companies([company("12345678901234")])
    with pytest.raises(Exception, match="fuseau"), db.session() as session:
        session.execute(
            update(CompanyRecord).values(last_seen_at=datetime(2026, 1, 1))  # sans fuseau
        )


# --------------------------------------------------------------------------
# Catalogue
# --------------------------------------------------------------------------


def test_upsert_completes_fields_without_erasing(db):
    db.upsert_companies([company("12345678901234", address="1 rue X", sources={"recherche_entreprises"})])
    update_ = company(
        "12345678901234", website="https://acme.fr", has_job_offer=True, sources={"la_bonne_alternance"}
    )
    db.upsert_companies([update_])

    with db.session() as session:
        record = session.get(CompanyRecord, "12345678901234")
        assert record.address == "1 rue X"  # pas effacée par la 2e source
        assert record.website == "https://acme.fr"
        assert record.has_job_offer is True
        assert record.sources == ["la_bonne_alternance", "recherche_entreprises"]
        assert record.siren == "123456789"
        assert record.first_seen_at.tzinfo is UTC
        assert record.last_seen_at >= record.first_seen_at
    assert db.count_companies() == 1


def test_catalogue_has_no_client_id(db):
    """Le catalogue est partagé : il ne garde pas quel client a trouvé l'entreprise."""
    assert "client_id" not in CompanyRecord.__table__.columns


# --------------------------------------------------------------------------
# Cache de recherches
# --------------------------------------------------------------------------


def test_cache_empty(db):
    assert db.find_cached("data", LYON, 10, client_id="c", max_results=10) is None


def test_cache_hit_for_contained_zone(db):
    db.save_search("data", LYON, 20, companies_around(LYON, 30), requested_max=30)

    cached = db.find_cached("data", VILLEURBANNE, 10, client_id="client-002", max_results=20)

    assert cached is not None
    assert len(cached) == 30
    assert all(c.client_id == "client-002" for c in cached)  # rattachées au nouveau client


def test_cache_miss_when_zone_not_covered(db):
    db.save_search("data", LYON, 20, companies_around(LYON, 30), requested_max=30)
    assert db.find_cached("data", PARIS, 10, client_id="c", max_results=10) is None
    assert db.find_cached("data", LYON, 30, client_id="c", max_results=10) is None  # zone plus large


def test_cache_miss_for_other_domain(db):
    db.save_search("data", LYON, 20, companies_around(LYON, 30), requested_max=30)
    assert db.find_cached("marketing", LYON, 10, client_id="c", max_results=10) is None


def test_cache_miss_when_expired(db):
    db.save_search("data", LYON, 20, companies_around(LYON, 30), requested_max=30)
    with db.session() as session:
        session.execute(update(SearchRecord).values(fetched_at=datetime.now(UTC) - timedelta(days=31)))
    assert db.find_cached("data", LYON, 10, client_id="c", max_results=10) is None


def test_cache_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("SOURCING_CACHE_DAYS", "0")
    get_settings.cache_clear()
    database = Database(f"sqlite:///{tmp_path / 'a.db'}")
    database.save_search("data", LYON, 20, companies_around(LYON, 30), requested_max=30)
    assert database.find_cached("data", LYON, 10, client_id="c", max_results=10) is None
    database.close()


def test_truncated_search_with_too_few_results_is_refreshed(db):
    """Recherche tronquée (30 demandées, 30 reçues) : on ne sait pas s'il en existe d'autres."""
    db.save_search("data", LYON, 20, companies_around(LYON, 30), requested_max=30)
    assert db.find_cached("data", LYON, 10, client_id="c", max_results=50) is None


def test_complete_search_is_served_even_with_few_results(db):
    """Recherche complète (50 demandées, 5 reçues) : il n'y a que 5 entreprises dans la zone."""
    db.save_search("data", LYON, 20, companies_around(LYON, 5), requested_max=50)
    cached = db.find_cached("data", LYON, 10, client_id="c", max_results=50)
    assert cached is not None and len(cached) == 5


def test_cache_only_returns_companies_inside_the_zone(db):
    near, far = companies_around(LYON, 5, "1"), companies_around(VILLEURBANNE, 5, "2")
    db.save_search("data", LYON, 20, near + far, requested_max=50)
    cached = db.find_cached("data", LYON, 2, client_id="c", max_results=50)
    assert {c.siret for c in cached} == {c.siret for c in near}


# --------------------------------------------------------------------------
# Opposition et candidatures
# --------------------------------------------------------------------------


def test_opt_out_applies_to_whole_company(db):
    db.add_opt_out("123456789")
    db.add_opt_out("123456789")  # idempotent
    assert db.opted_out_sirens(["123456789", "987654321"]) == {"123456789"}


@pytest.mark.parametrize("siren", ["12345", "12345678901234", "abcdefghi", "123456789; DROP TABLE"])
def test_invalid_siren_is_rejected(db, siren):
    with pytest.raises(ValueError):
        db.add_opt_out(siren)


def test_application_requires_known_company(db):
    with pytest.raises(ValueError, match="inconnu"):
        db.record_application("client-001", "12345678901234")


def test_application_is_unique_per_client(db):
    db.upsert_companies([company("12345678901234")])
    db.record_application("client-001", "12345678901234")
    db.record_application("client-001", "12345678901234", ApplicationStatus.SENT)  # mise à jour

    remaining = db.contactable([company("12345678901234")], "client-001")
    assert remaining == []


def test_contactable_filters(db):
    targets = [
        company("11111111100011"),  # libre
        company("22222222200022"),  # déjà contactée par ce client
        company("33333333300033"),  # en opposition
        company("44444444400044"),  # saturée par d'autres clients (autre établissement)
    ]
    db.upsert_companies([*targets, company("44444444400099")])
    db.record_application("client-001", "22222222200022")
    db.add_opt_out("333333333")
    db.record_application("client-002", "44444444400044")
    db.record_application("client-003", "44444444400099")

    remaining = db.contactable(targets, "client-001")

    assert [c.siret for c in remaining] == ["11111111100011"]


def test_contactable_for_another_client(db):
    db.upsert_companies([company("22222222200022")])
    db.record_application("client-001", "22222222200022")
    assert len(db.contactable([company("22222222200022")], "client-002")) == 1


# --------------------------------------------------------------------------
# Intégration avec le moteur de sourcing
# --------------------------------------------------------------------------


class CountingSource(CompanySource):
    name = "recherche_entreprises"

    def __init__(self, companies: list[Company]) -> None:
        self.calls = 0
        self._companies = companies

    def search(self, query: SearchQuery) -> list[Company]:
        self.calls += 1
        return self._companies


class FixedGeocoder:
    def locate(self, city: str) -> GeoPoint:
        return LYON


def test_engine_serves_second_search_from_database(db):
    source = CountingSource(companies_around(LYON, 10))
    with SourcingEngine(get_settings(), database=db) as engine:
        engine.geocoder = FixedGeocoder()
        engine.sources = [source]

        first = engine.find("data", "Lyon", client_id="client-001", radius_km=10, max_results=50)
        second = engine.find("Data", "Lyon", client_id="client-002", radius_km=5, max_results=50)

    assert source.calls == 1  # la 2e recherche n'a pas appelé les APIs
    assert len(first) == len(second) == 10
    assert {c.client_id for c in second} == {"client-002"}


def test_engine_excludes_opted_out_companies(db):
    source = CountingSource(companies_around(LYON, 3))
    db.add_opt_out(source._companies[0].siren)
    with SourcingEngine(get_settings(), database=db) as engine:
        engine.geocoder = FixedGeocoder()
        engine.sources = [source]
        results = engine.find("data", "Lyon", client_id="c", radius_km=10, max_results=50)
    assert len(results) == 2
