"""Tests de l'enrichissement, sans réseau : les sites web sont simulés."""

from __future__ import annotations

import pytest

from avenly.enricher import (
    Enricher,
    ParsedPage,
    extract_generic_emails,
    find_key_pages,
    guess_websites,
    parse_html,
    registrable_domain,
    site_matches_company,
)
from avenly.llm import BudgetExceededError, LLMClient, LLMError
from avenly.models import Company, CompanyInsight
from avenly.storage import Database
from avenly.webfetch import FetchError, Page

HOME = """
<html><head><title>ACME Conseil - ESN à Lyon</title><script>alert('x')</script></head>
<body>
  <nav>
    <a href="/qui-sommes-nous">Qui sommes-nous ?</a>
    <a href="/carrieres">Rejoignez-nous</a>
    <a href="/mentions-legales">Mentions légales</a>
    <a href="/contact">Contact</a>
    <a href="https://www.linkedin.com/company/acme">LinkedIn</a>
    <a href="javascript:void(0)">Menu</a>
  </nav>
  <p>ACME accompagne les PME dans leurs projets data et cloud.</p>
</body></html>
"""
LEGAL = "<html><body>ACME Conseil SAS, RCS Lyon 123 456 789. Contact : contact@acme.fr</body></html>"
CAREERS = """<html><body>Nous accueillons des alternants chaque année.
Candidatures : <a href="mailto:recrutement@acme.fr?subject=Alternance">recrutement@acme.fr</a>
Responsable : jean.dupont@acme.fr</body></html>"""


def page(url: str, html: str) -> Page:
    return Page(url=url, status=200, content_type="text/html", text=html)


SITE = {
    "https://www.acme.fr": HOME,
    "https://www.acme.fr/qui-sommes-nous": "<html><body>Fondée en 2010, ACME a 80 consultants.</body></html>",
    "https://www.acme.fr/carrieres": CAREERS,
    "https://www.acme.fr/mentions-legales": LEGAL,
    "https://www.acme.fr/contact": "<html><body>Écrivez-nous à contact [at] acme [dot] fr</body></html>",
}


class FakeFetcher:
    def __init__(self, site: dict[str, str]) -> None:
        self.site = site
        self.requested: list[str] = []

    def fetch(self, url: str) -> Page:
        url = url.rstrip("/")
        self.requested.append(url)
        if url not in self.site:
            raise FetchError("introuvable")
        return page(url, self.site[url])


class FakeLLM(LLMClient):
    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[tuple[str, str]] = []
        self.error = error

    email_model = "faux-modele"

    def extract_profile(self, cv, *, client_id):  # pragma: no cover - non utilisé ici
        raise NotImplementedError

    def write_email(self, profile, company_name, insight, context, *, client_id):  # pragma: no cover
        raise NotImplementedError

    def summarize_company(self, company_name: str, site_text: str) -> CompanyInsight:
        self.calls.append((company_name, site_text))
        if self.error:
            raise self.error
        return CompanyInsight(summary="ESN spécialisée data et cloud.", mentions_apprenticeship=True)


@pytest.fixture
def db(tmp_path) -> Database:
    database = Database(f"sqlite:///{tmp_path / 'avenly.db'}")
    yield database
    database.close()


def acme(**extra) -> Company:
    fields = {
        "client_id": "c",
        "siret": "12345678900011",
        "name": "ACME CONSEIL SAS",
        "address": "1 RUE X 69003 LYON",
    }
    return Company(**(fields | extra))


# --------------------------------------------------------------------------
# Lecture du HTML
# --------------------------------------------------------------------------


def test_parse_html_removes_scripts_and_collects_links():
    parsed = parse_html(page("https://www.acme.fr", HOME))
    assert "alert" not in parsed.text
    assert parsed.title == "ACME Conseil - ESN à Lyon"
    assert ("https://www.acme.fr/carrieres", "Rejoignez-nous") in parsed.links
    assert not any(url.startswith("javascript") for url, _ in parsed.links)


def test_find_key_pages_stays_on_same_site():
    pages = find_key_pages(parse_html(page("https://www.acme.fr", HOME)))
    assert pages == {
        "about": "https://www.acme.fr/qui-sommes-nous",
        "careers": "https://www.acme.fr/carrieres",
        "legal": "https://www.acme.fr/mentions-legales",
        "contact": "https://www.acme.fr/contact",
    }


@pytest.mark.parametrize(
    ("url", "domain"),
    [
        ("https://www.acme.fr/x", "acme.fr"),
        ("https://jobs.carrieres.acme.com", "acme.com"),
        ("https://acme.fr", "acme.fr"),
    ],
)
def test_registrable_domain(url, domain):
    assert registrable_domain(url) == domain


# --------------------------------------------------------------------------
# Emails génériques (RGPD)
# --------------------------------------------------------------------------


def test_only_generic_emails_of_the_company_are_kept():
    pages = [
        ParsedPage(
            url="https://www.acme.fr",
            title="",
            text=(
                "recrutement@acme.fr jean.dupont@acme.fr marie@acme.fr contact@acme.fr "
                "rh@gmail.com jobs@concurrent.fr info@paris.acme.fr icon@2x.png"
            ),
            mailtos=["rh@acme.fr"],
        )
    ]
    assert extract_generic_emails(pages, "acme.fr") == [
        "rh@acme.fr",
        "recrutement@acme.fr",
        "contact@acme.fr",
        "info@paris.acme.fr",
    ]


def test_obfuscated_emails_are_found():
    pages = [ParsedPage(url="https://www.acme.fr", title="", text="Écrivez à contact [at] acme [dot] fr")]
    assert extract_generic_emails(pages, "acme.fr") == ["contact@acme.fr"]


# --------------------------------------------------------------------------
# Deviner et vérifier le site
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("ACME CONSEIL SAS", ["https://www.acme-conseil.fr", "https://www.acme-conseil.com"]),
        ("AKKODIS I&S SAS (AKKODIS)", ["https://www.akkodis.fr", "https://www.akkodis.com"]),
    ],
)
def test_guess_websites_tries_multi_word_domains_first(name, expected):
    guesses = guess_websites(name)
    assert guesses[:2] == expected
    assert len(guesses) <= 12
    assert guesses.index("https://www." + expected[0].split(".")[1] + ".net") >= 2  # .fr/.com d'abord


def test_one_word_of_the_name_is_not_enough():
    """« Clever » dans le titre ne prouve pas qu'il s'agit de CLEVER AGE."""
    other_site = parse_html(
        page("https://www.clever.fr", "<html><head><title>Clever - Agence web</title></head></html>")
    )
    clever_age = Company(client_id="c", siret="11122233300011", name="CLEVER AGE")
    assert not site_matches_company([other_site], clever_age)


def test_generic_words_are_ignored_in_title_check():
    """« FRANCE » est ignoré : seul « AMARIS » compte (avec la ville, car c'est un nom d'un seul mot)."""
    html = "<html><head><title>Amaris Consulting</title></head><body>Bureaux à Lyon</body></html>"
    site = parse_html(page("https://amaris.com", html))
    amaris = Company(
        client_id="c", siret="11122233300011", name="AMARIS FRANCE SAS", address="2 RUE Y 69002 LYON"
    )
    assert site_matches_company([site], amaris)


def test_guess_websites_without_usable_name():
    assert guess_websites("SAS") == []


def test_site_matches_by_siren():
    legal = parse_html(page("https://x.fr/legal", LEGAL))
    assert site_matches_company([legal], acme(name="AUTRE NOM"))


def test_site_matches_by_name_in_title():
    assert site_matches_company([parse_html(page("https://www.acme.fr", HOME))], acme())


def test_site_does_not_match_other_company():
    other = Company(client_id="c", siret="98765432100011", name="ZETA INDUSTRIE")
    assert not site_matches_company([parse_html(page("https://www.acme.fr", HOME))], other)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def test_enrich_with_known_website(db):
    fetcher, llm = FakeFetcher(SITE), FakeLLM()
    enrichment = Enricher(db, fetcher, llm=llm).enrich(acme(website="https://www.acme.fr"))

    assert enrichment.status == "ok"
    assert enrichment.website == "https://www.acme.fr"
    assert enrichment.website_verified
    assert enrichment.careers_url == "https://www.acme.fr/carrieres"
    assert enrichment.generic_emails == ["recrutement@acme.fr", "contact@acme.fr"]
    assert "jean.dupont@acme.fr" not in enrichment.generic_emails  # nominative : jamais conservée
    assert enrichment.insight.mentions_apprenticeship
    assert len(fetcher.requested) <= 5  # accueil + 4 pages maximum
    # Les mentions légales ne sont pas envoyées au LLM (inutile et coûteux).
    assert "RCS Lyon" not in llm.calls[0][1]


def test_enrich_guesses_and_verifies_website(db):
    enrichment = Enricher(db, FakeFetcher(SITE)).enrich(acme())
    assert enrichment.status == "ok"
    assert enrichment.website == "https://www.acme.fr"
    assert enrichment.insight is None  # pas de LLM fourni


def test_enrich_rejects_unverified_guess(db):
    """Un site trouvé au hasard du nom, mais qui n'est pas celui de l'entreprise."""
    homonym = {"https://www.acme.fr": "<html><head><title>Boulangerie du coin</title></head></html>"}
    enrichment = Enricher(db, FakeFetcher(homonym)).enrich(acme())
    assert enrichment.status == "no_website"


def test_enrich_without_website(db):
    enrichment = Enricher(db, FakeFetcher({})).enrich(acme())
    assert enrichment.status == "no_website"
    assert enrichment.generic_emails == []


def test_enrichment_is_cached_and_shared(db):
    fetcher = FakeFetcher(SITE)
    enricher = Enricher(db, fetcher)
    enricher.enrich(acme(website="https://www.acme.fr"))
    calls = len(fetcher.requested)

    # Un autre client, un autre établissement de la même entreprise (même SIREN).
    other = Company(client_id="autre", siret="12345678900099", name="ACME CONSEIL SAS")
    cached = enricher.enrich(other)

    assert len(fetcher.requested) == calls  # aucune nouvelle requête
    assert cached.generic_emails == ["recrutement@acme.fr", "contact@acme.fr"]


def test_enrich_many_once_per_company(db):
    fetcher = FakeFetcher(SITE)
    sites = [acme(website="https://www.acme.fr"), Company(client_id="c", siret="12345678900099", name="ACME")]
    results = Enricher(db, fetcher).enrich_many(sites)
    assert list(results) == ["123456789"]


def test_llm_error_does_not_block_enrichment(db):
    llm = FakeLLM(error=LLMError("panne"))
    enrichment = Enricher(db, FakeFetcher(SITE), llm=llm).enrich(acme(website="https://www.acme.fr"))
    assert enrichment.status == "ok"
    assert enrichment.insight is None


def test_budget_exceeded_stops_everything(db):
    llm = FakeLLM(error=BudgetExceededError("budget"))
    with pytest.raises(BudgetExceededError):
        Enricher(db, FakeFetcher(SITE), llm=llm).enrich(acme(website="https://www.acme.fr"))


def test_foreign_region_emails_are_ignored():
    """« contact.apac@ » vise l'Asie-Pacifique : inutile pour une alternance en France."""
    pages = [
        ParsedPage(
            url="https://www.acme.com",
            title="",
            text=(
                "contact.apac@acme.com jobs-us@acme.com contact.france@acme.com "
                "rh.fr@acme.com contact@acme.com"
            ),
        )
    ]
    assert extract_generic_emails(pages, "acme.com") == [
        "rh.fr@acme.com",
        "contact.france@acme.com",
        "contact@acme.com",
    ]


@pytest.mark.parametrize(
    ("final_url", "title", "ok"),
    [
        ("https://www.acme.fr/", "ACME - ESN", True),
        ("https://www.hugedomains.com/domain_profile.cfm?d=acme.fr", "Acme.fr is for sale", False),
        ("https://www.acme.fr/", "Ce nom de domaine est à vendre", False),
    ],
)
def test_parked_or_redirected_guesses_are_rejected(final_url, title, ok):
    from avenly.enricher import looks_like_real_site

    home = ParsedPage(url=final_url, title=title, text="")
    assert looks_like_real_site("https://www.acme.fr", home) is ok


@pytest.mark.parametrize(
    ("href", "label"),
    [
        ("/rejoignez-nous", "Rejoignez-nous"),
        ("/travailler-chez-nous", "Travailler chez nous"),
        ("/offres-d-emploi", "Nos offres"),
        ("https://www.welcometothejungle.com/fr/companies/acme/jobs", "Nos offres"),
        ("https://acme.teamtailor.com", "Jobs"),
    ],
)
def test_careers_pages_are_found(href, label):
    html = f'<html><body><a href="{href}">{label}</a></body></html>'
    assert "careers" in find_key_pages(parse_html(page("https://www.acme.fr", html)))


def test_external_non_ats_links_are_not_careers_pages():
    html = '<html><body><a href="https://www.linkedin.com/company/acme/jobs">Jobs</a></body></html>'
    assert "careers" not in find_key_pages(parse_html(page("https://www.acme.fr", html)))


def test_one_word_name_needs_the_company_city():
    """« BOKA » : le titre ne suffit pas, il faut aussi la ville ou le code postal de l'établissement."""
    boka = Company(client_id="c", siret="11122233300011", name="BOKA", address="3 RUE X 74000 ANNECY")
    elsewhere = parse_html(
        page("https://www.boka.com", "<html><head><title>Boka</title></head><body>Paris</body></html>")
    )
    local = parse_html(
        page("https://www.boka.fr", "<html><head><title>Boka</title></head><body>Annecy</body></html>")
    )
    by_postcode = parse_html(
        page("https://boka.io", "<html><head><title>Boka</title></head><body>74000</body></html>")
    )
    assert not site_matches_company([elsewhere], boka)
    assert site_matches_company([local], boka)
    assert site_matches_company([by_postcode], boka)


def test_blog_post_about_internships_is_not_a_careers_page():
    html = '<html><body><a href="/blog/rapport-de-stage">Conclusion de rapport de stage</a></body></html>'
    assert "careers" not in find_key_pages(parse_html(page("https://www.compilatio.net", html)))


def test_time_budget_per_company(db, monkeypatch):
    """Une fois le temps alloué à une entreprise épuisé, on n'ouvre plus de nouvelle page."""
    from avenly import enricher as enricher_module

    monkeypatch.setattr(enricher_module, "MAX_SECONDS_PER_COMPANY", -1)  # déjà épuisé
    fetcher = FakeFetcher(SITE)
    enrichment = Enricher(db, fetcher).enrich(acme(website="https://www.acme.fr"))
    assert enrichment.status == "no_website"
    assert fetcher.requested == []
