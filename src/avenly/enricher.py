"""Enrichissement des entreprises : site web, résumé d'activité, emails génériques, page carrières.

Version gratuite (sans API payante) :
1. site web : fourni par La Bonne Alternance, sinon deviné à partir du nom puis VÉRIFIÉ
   (SIREN ou nom de l'entreprise présent sur le site) ;
2. lecture de quelques pages clés (accueil, à propos, carrières, contact, mentions légales)
   via `SafeFetcher` (protection SSRF, robots.txt, limites de taille) ;
3. extraction des adresses GÉNÉRIQUES (rh@, recrutement@, contact@…) du domaine de
   l'entreprise. Les adresses nominatives (prenom.nom@) ne sont jamais conservées (RGPD) ;
4. résumé de l'activité par le LLM (optionnel) ;
5. résultat mis en cache dans le catalogue partagé (par SIREN) pour tous les clients.
"""

from __future__ import annotations

import logging
import re
import time
import unicodedata
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup

from avenly.config import Settings, get_settings
from avenly.llm import BudgetExceededError, LLMClient, LLMError
from avenly.models import Company, Enrichment
from avenly.storage import Database
from avenly.webfetch import FetchError, Page, SafeFetcher

logger = logging.getLogger(__name__)

MAX_PAGES_PER_SITE = 5
# Temps maximal consacré à une entreprise (recherche du site + lecture des pages).
MAX_SECONDS_PER_COMPANY = 60.0
# Domaines devinés testés en même temps pour une entreprise.
GUESS_WORKERS = 6
MAX_GUESSED_SITES = 12
GUESSED_TLDS = ("fr", "com", "net", "eu", "io")

# --------------------------------------------------------------------------
# Lecture du HTML
# --------------------------------------------------------------------------

PAGE_KINDS: dict[str, re.Pattern[str]] = {
    "careers": re.compile(
        r"carri[eè]re|career|recrut|emploi|jobs?\b|rejoi[ng]|join|talent|postuler|offres?[\s\-_]*d|"
        r"travailler|work[\s\-_]*with|we[\s\-_]*are[\s\-_]*hiring",
        re.I,
    ),
    "legal": re.compile(r"mentions[\s\-_]*l[ée]gales|legal|imprint|cgu", re.I),
    "about": re.compile(
        r"[àa][\s\-_]*propos|about|qui[\s\-_]*sommes|entreprise|soci[ée]t[ée]|histoire", re.I
    ),
    "contact": re.compile(r"contact", re.I),
}
# Ordre de visite (dans la limite de MAX_PAGES_PER_SITE, accueil compris).
VISIT_ORDER = ("legal", "careers", "about", "contact")


@dataclass(slots=True)
class ParsedPage:
    url: str
    title: str
    text: str
    links: list[tuple[str, str]] = field(default_factory=list)  # (url absolue, libellé)
    mailtos: list[str] = field(default_factory=list)


def parse_html(page: Page) -> ParsedPage:
    soup = BeautifulSoup(page.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "iframe", "template"]):
        tag.decompose()
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    links: list[tuple[str, str]] = []
    mailtos: list[str] = []
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        if href.lower().startswith("mailto:"):
            mailtos.append(href[7:].split("?")[0])
        elif not href.lower().startswith(("javascript:", "tel:", "data:", "#")):
            links.append((urljoin(page.url, href), anchor.get_text(" ", strip=True)[:100]))
    text = " ".join(soup.get_text(" ", strip=True).split())
    return ParsedPage(url=page.url, title=title, text=text, links=links, mailtos=mailtos)


def registrable_domain(url: str) -> str:
    """« www.carrieres.acme.fr » -> « acme.fr » (approximation suffisante pour .fr / .com)."""
    host = (urlsplit(url).hostname or "").lower()
    return ".".join(host.split(".")[-2:])


# Plateformes de recrutement où beaucoup d'entreprises publient leurs offres.
ATS_DOMAINS = {
    "welcometothejungle.com", "teamtailor.com", "lever.co", "greenhouse.io", "smartrecruiters.com",
    "workable.com", "recruitee.com", "jobaffinity.fr", "flatchr.io", "talent-soft.com", "softy.pro",
    "myworkdayjobs.com", "successfactors.com", "taleo.net", "hellowork.com", "jobteaser.com",
}  # fmt: skip


def find_key_pages(page: ParsedPage) -> dict[str, str]:
    """Repère les pages carrières, mentions légales, à propos et contact.

    Liens du même site uniquement, sauf pour les carrières : une page d'offres hébergée
    sur une plateforme de recrutement connue (Welcome to the Jungle, Teamtailor…) est acceptée.
    """
    domain = registrable_domain(page.url)
    found: dict[str, str] = {}
    for url, label in page.links:
        if url.rstrip("/") == page.url.rstrip("/"):
            continue
        link_domain = registrable_domain(url)
        if link_domain in ATS_DOMAINS and "careers" not in found:
            found["careers"] = url.split("#")[0]
            continue
        if link_domain != domain:
            continue
        haystack = f"{urlsplit(url).path} {label}"
        for kind, pattern in PAGE_KINDS.items():
            if kind not in found and pattern.search(haystack):
                found[kind] = url.split("#")[0]
                break
    return found


# --------------------------------------------------------------------------
# Emails génériques
# --------------------------------------------------------------------------

GENERIC_LOCAL_PARTS = re.compile(
    r"^(?:rh|drh|hr|recrutement|recruitment|recrute|jobs?|emplois?|carrieres?|careers?|candidatures?"
    r"|alternance|stages?|talents?|contact|info|infos|hello|bonjour|accueil|admin(?:istration)?)"
    r"(?:[.\-_][a-z0-9]+)?$"
)
# « contact.apac@ », « jobs-us@ » : adresses destinées à d'autres pays, inutiles pour la France.
FOREIGN_REGION_SUFFIX = re.compile(
    r"[.\-_](?:apac|asia|asie|emea|americas?|latam|na|us|usa|uk|gb|de|es|it|nl|be|ch|ca|au|in|cn|jp|sg|"
    r"me|mena|africa|nordics?|dach|benelux|international|intl|global)$"
)
_EMAIL_IN_TEXT = re.compile(r"[a-z0-9._%+\-]{1,64}@[a-z0-9\-]{1,63}(?:\.[a-z0-9\-]{1,63}){1,4}", re.I)
_OBFUSCATED_AT = re.compile(r"\s*(?:\[|\(|\{)\s*(?:at|arobase)\s*(?:\]|\)|\})\s*", re.I)
_OBFUSCATED_DOT = re.compile(r"\s*(?:\[|\(|\{)\s*(?:dot|point)\s*(?:\]|\)|\})\s*", re.I)


def extract_generic_emails(pages: Iterable[ParsedPage], site_domain: str) -> list[str]:
    """Adresses génériques du domaine de l'entreprise. Les adresses nominatives sont ignorées."""
    found: list[str] = []
    for page in pages:
        text = _OBFUSCATED_DOT.sub(".", _OBFUSCATED_AT.sub("@", page.text))
        for raw in [*page.mailtos, *_EMAIL_IN_TEXT.findall(text)]:
            email = raw.strip().strip(".").lower()
            local, _, domain = email.partition("@")
            if not domain or not (domain == site_domain or domain.endswith("." + site_domain)):
                continue
            if (
                GENERIC_LOCAL_PARTS.match(local)
                and not FOREIGN_REGION_SUFFIX.search(local)
                and email not in found
            ):
                found.append(email)
    # Les adresses de recrutement d'abord, puis les adresses de contact.
    return sorted(
        found,
        key=lambda e: (
            not re.match(r"(rh|drh|hr|recrut|job|emploi|carri|career|candid|altern|stage|talent)", e)
        ),
    )


# --------------------------------------------------------------------------
# Site web : deviner puis vérifier
# --------------------------------------------------------------------------

LEGAL_FORMS = {
    "sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "scop", "se", "gie",
    "et", "de", "la", "le", "les", "des", "du",
}  # fmt: skip


def _slug_words(name: str) -> list[str]:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    ascii_name = re.sub(r"\(.*?\)", " ", ascii_name)  # « AKKODIS I&S SAS (AKKODIS) » -> sans parenthèses
    return [w for w in re.findall(r"[a-z0-9]+", ascii_name) if len(w) >= 2 and w not in LEGAL_FORMS]


def guess_websites(name: str) -> list[str]:
    """Quelques domaines plausibles à partir du nom (vérifiés ensuite, jamais pris pour acquis).

    Les variantes à plusieurs mots passent en premier : un seul mot (« clever ») est
    bien plus souvent pris par une autre entreprise que « clever-age ».
    """
    words = _slug_words(name)
    if not words:
        return []
    stems = ["-".join(words[:2]), "".join(words[:2]), words[0]] if len(words) > 1 else [words[0]]
    stems = list(dict.fromkeys(s for s in stems if len(s) >= 3))
    # D'abord .fr et .com pour toutes les variantes, puis les extensions plus rares.
    common = [f"https://www.{stem}.{tld}" for stem in stems for tld in GUESSED_TLDS[:2]]
    rare = [f"https://www.{stem}.{tld}" for stem in stems for tld in GUESSED_TLDS[2:]]
    return (common + rare)[:MAX_GUESSED_SITES]


# Mots trop génériques pour identifier une entreprise par son titre de page.
GENERIC_NAME_WORDS = {
    "france", "group", "groupe", "services", "service", "consulting", "conseil", "technology",
    "technologies", "solutions", "international", "holding", "industries", "systems", "systemes",
}  # fmt: skip


def _siren_found(pages: list[ParsedPage], siren: str) -> bool:
    pattern = re.compile(rf"{siren[:3]}\s?{siren[3:6]}\s?{siren[6:]}")
    return any(pattern.search(p.text) for p in pages)


def site_matches_company(pages: Iterable[ParsedPage], company: Company) -> bool:
    """Le site est-il bien celui de l'entreprise ?

    Preuve forte : le SIREN figure sur le site (mentions légales). À défaut, TOUS les mots
    significatifs du nom doivent apparaître dans le titre des pages : « Clever » seul ne
    suffit pas à identifier « CLEVER AGE ».
    """
    pages = list(pages)
    if _siren_found(pages, company.siren):
        return True
    words = [w for w in _slug_words(company.name) if len(w) >= 3 and w not in GENERIC_NAME_WORDS]
    if not words:
        return False
    title_words = set(_slug_words(" ".join(p.title for p in pages)))
    if not all(w in title_words for w in words[:3]):
        return False
    if len(words) >= 2:
        return True
    # Nom d'un seul mot (« BOKA ») : trop d'homonymes possibles. On exige en plus que le
    # site mentionne la ville ou le code postal de l'établissement.
    return _location_found(pages, company)


def _location_found(pages: list[ParsedPage], company: Company) -> bool:
    match = re.search(r"\b(\d{5})\s+([A-Za-zÀ-ÿ'’\- ]+)", company.address or "")
    if not match:
        return False
    postcode, city = match.group(1), match.group(2).strip().split(" CEDEX")[0]
    text = " ".join(_slug_words(" ".join(f"{p.title} {p.text}" for p in pages)))
    return postcode in text or " ".join(_slug_words(city)) in text


_PARKED = re.compile(
    r"domain (?:name )?(?:is )?for sale|buy this domain|ce (?:nom de )?domaine est (?:à vendre|disponible)"
    r"|parked (?:free|domain)|hugedomains|sedo|godaddy|domaine en vente",
    re.I,
)


def looks_like_real_site(candidate_url: str, home: ParsedPage) -> bool:
    """Écarte un domaine deviné qui redirige vers un autre site ou qui est une page « domaine à vendre »."""
    if registrable_domain(home.url) != registrable_domain(candidate_url):
        return False
    return not _PARKED.search(f"{home.title} {home.text[:2000]}")


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


class Enricher:
    """Enrichit des entreprises, avec cache partagé dans la base."""

    def __init__(
        self,
        database: Database,
        fetcher: SafeFetcher,
        *,
        llm: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._db = database
        self._fetcher = fetcher
        self._llm = llm
        self._settings = settings or get_settings()

    def enrich_many(self, companies: Iterable[Company], *, workers: int = 4) -> dict[str, Enrichment]:
        """Un enrichissement par entreprise (SIREN), plusieurs entreprises EN PARALLÈLE.

        Le rythme d'une requête par seconde reste garanti pour chaque site (SafeFetcher).
        S'arrête net si le budget LLM est atteint.
        """
        unique = {c.siren: c for c in reversed(list(companies))}  # premier établissement par SIREN
        ordered = list(reversed(unique.values()))
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            results = list(pool.map(self.enrich, ordered))
        return {company.siren: enrichment for company, enrichment in zip(ordered, results, strict=True)}

    def enrich(self, company: Company) -> Enrichment:
        cached = self._db.get_enrichment(company.siren, max_age_days=self._settings.enrichment_cache_days)
        if cached is not None:
            return cached
        enrichment = self._enrich_uncached(company)
        self._db.save_enrichment(enrichment)
        return enrichment

    # ------------------------------------------------------------------ étapes

    def _read(self, url: str, deadline: float) -> ParsedPage | None:
        if time.monotonic() > deadline:
            return None  # temps alloué à cette entreprise épuisé
        try:
            page = self._fetcher.fetch(url)
        except FetchError as exc:
            logger.info("Page ignorée (%s) : %s", url, exc)
            return None
        if page.status >= 400:
            return None
        return parse_html(page)

    def _verified_home(self, candidate: str, company: Company, deadline: float) -> ParsedPage | None:
        """Lit un domaine deviné et le garde seulement s'il est bien celui de l'entreprise."""
        home = self._read(candidate, deadline)
        if home is None or not looks_like_real_site(candidate, home):
            return None
        if site_matches_company([home], company):
            return home
        # Le SIREN figure le plus souvent dans les mentions légales.
        legal_url = find_key_pages(home).get("legal")
        legal = self._read(legal_url, deadline) if legal_url else None
        if legal is not None and site_matches_company([home, legal], company):
            return home
        return None

    def _find_home(self, company: Company, deadline: float) -> tuple[ParsedPage | None, bool]:
        """Renvoie (page d'accueil, site vérifié ?)."""
        if company.website:
            home = self._read(company.website, deadline)
            if home is not None:
                return home, True  # site déclaré par l'entreprise elle-même (La Bonne Alternance)
        candidates = guess_websites(company.name)
        if not candidates:
            return None, False
        # Les domaines devinés sont testés EN PARALLÈLE (un domaine lent ne bloque plus les autres),
        # puis on garde le premier vérifié dans l'ordre de priorité.
        with ThreadPoolExecutor(max_workers=min(len(candidates), GUESS_WORKERS)) as pool:
            futures = [pool.submit(self._verified_home, c, company, deadline) for c in candidates]
            for future in futures:
                home = future.result()
                if home is not None:
                    for other in futures:
                        other.cancel()
                    return home, True
        return None, False

    def _enrich_uncached(self, company: Company) -> Enrichment:
        deadline = time.monotonic() + MAX_SECONDS_PER_COMPANY
        home, verified = self._find_home(company, deadline)
        if home is None:
            return Enrichment(siren=company.siren, status="no_website")

        pages = [home]
        key_pages = find_key_pages(home)
        for kind in VISIT_ORDER:
            if len(pages) >= MAX_PAGES_PER_SITE:
                break
            if kind in key_pages and (parsed := self._read(key_pages[kind], deadline)) is not None:
                pages.append(parsed)

        site_domain = registrable_domain(home.url)
        enrichment = Enrichment(
            siren=company.siren,
            status="ok",
            website=f"https://{urlsplit(home.url).hostname}",
            website_verified=verified,
            careers_url=key_pages.get("careers"),
            generic_emails=extract_generic_emails(pages, site_domain),
        )

        if self._llm is not None:
            site_text = "\n\n".join(p.text for p in pages if p.url != key_pages.get("legal"))
            try:
                enrichment.insight = self._llm.summarize_company(company.name, site_text)
            except BudgetExceededError:
                raise
            except LLMError as exc:
                logger.warning("Résumé impossible pour %s : %s", company.name, exc)
        return enrichment
