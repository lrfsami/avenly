"""Tests de SafeFetcher : protections SSRF, redirections, limites. Aucun appel réseau."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from avenly import webfetch
from avenly.webfetch import FetchError, SafeFetcher, check_url, is_public_ip

PUBLIC_IP = "93.184.216.34"


def public_resolver(host: str) -> list[str]:
    return [PUBLIC_IP]


@pytest.fixture(autouse=True)
def no_politeness_delay(monkeypatch):
    monkeypatch.setattr(webfetch, "PER_HOST_INTERVAL_SECONDS", 0)


class FakeResponse:
    def __init__(self, status=200, body=b"<html>ok</html>", headers=None, peer=PUBLIC_IP, encoding="utf-8"):
        self.status_code = status
        self._body = body
        self.headers = {"Content-Type": "text/html; charset=utf-8"} | (headers or {})
        self.encoding = encoding
        self.raw = SimpleNamespace(
            connection=SimpleNamespace(sock=SimpleNamespace(getpeername=lambda: (peer, 443)))
        )

    @property
    def is_redirect(self):
        return self.status_code in (301, 302, 303, 307, 308) and "Location" in self.headers

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_content(self, chunk_size):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i : i + chunk_size]


class FakeSession:
    """Répond selon l'URL demandée ; robots.txt absent par défaut."""

    def __init__(self, routes: dict[str, FakeResponse]):
        self.routes = routes
        self.headers: dict[str, str] = {}
        self.requested: list[str] = []

    def get(self, url, **kwargs):
        assert kwargs["allow_redirects"] is False  # les redirections sont suivies à la main
        self.requested.append(url)
        if url.endswith("/robots.txt") and url not in self.routes:
            return FakeResponse(status=404, body=b"")
        return self.routes[url]

    def close(self):
        pass


def make_fetcher(routes, resolver=public_resolver) -> tuple[SafeFetcher, FakeSession]:
    session = FakeSession(routes)
    return SafeFetcher(resolver=resolver, session=session), session


# --------------------------------------------------------------------------
# Validation des URLs (SSRF)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ip",
    [
        "127.0.0.1",
        "10.0.0.5",
        "192.168.1.1",
        "172.16.0.1",
        "169.254.169.254",
        "0.0.0.0",  # noqa: S104 (adresse testée, pas une écoute réseau)
        "::1",
        "fd00::1",
        "::ffff:127.0.0.1",
        "224.0.0.1",
        "100.64.0.1",
    ],
)
def test_private_ips_are_not_public(ip):
    assert not is_public_ip(ip)


def test_public_ip():
    assert is_public_ip(PUBLIC_IP)


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("file:///etc/passwd", "http et https"),
        ("ftp://acme.fr", "http et https"),
        ("http://user:pass@acme.fr", "identifiants"),
        ("http://acme.fr:8080", "Port"),
        ("http://localhost", "Nom de domaine"),
        ("http://127.0.0.1", "IP brutes"),
        ("http://[::1]/", "IP brutes"),
    ],
)
def test_forbidden_urls(url, message):
    with pytest.raises(FetchError, match=message):
        check_url(url, public_resolver)


def test_domain_pointing_to_private_ip_is_refused():
    """Un nom de domaine public qui pointe vers une adresse interne (piège classique)."""
    with pytest.raises(FetchError, match="non publique"):
        check_url("https://piege.example.com", lambda host: ["169.254.169.254"])


def test_domain_with_one_private_ip_among_others_is_refused():
    with pytest.raises(FetchError, match="non publique"):
        check_url("https://piege.example.com", lambda host: [PUBLIC_IP, "10.0.0.1"])


def test_valid_url():
    assert check_url("https://www.acme.fr/carrieres", public_resolver) == "www.acme.fr"


# --------------------------------------------------------------------------
# Requêtes
# --------------------------------------------------------------------------


def test_fetch_html_page():
    fetcher, _ = make_fetcher({"https://www.acme.fr/": FakeResponse(body="<p>Bonjour é</p>".encode())})
    page = fetcher.fetch("https://www.acme.fr/")
    assert page.status == 200
    assert "Bonjour é" in page.text


def test_dns_rebinding_is_detected_after_connection():
    """Le DNS disait « public », mais la connexion a abouti sur une adresse interne."""
    fetcher, _ = make_fetcher({"https://www.acme.fr/": FakeResponse(peer="10.0.0.7")})
    with pytest.raises(FetchError, match="non publique"):
        fetcher.fetch("https://www.acme.fr/")


def test_redirect_to_internal_address_is_refused():
    routes = {
        "https://www.acme.fr/": FakeResponse(status=302, headers={"Location": "http://127.0.0.1/admin"})
    }
    fetcher, session = make_fetcher(routes)
    with pytest.raises(FetchError, match="IP brutes"):
        fetcher.fetch("https://www.acme.fr/")
    assert "http://127.0.0.1/admin" not in session.requested


def test_redirect_is_followed():
    routes = {
        "https://acme.fr/": FakeResponse(status=301, headers={"Location": "https://www.acme.fr/"}),
        "https://www.acme.fr/": FakeResponse(body=b"<p>ok</p>"),
    }
    fetcher, _ = make_fetcher(routes)
    assert fetcher.fetch("https://acme.fr/").url == "https://www.acme.fr/"


def test_too_many_redirects():
    loop = FakeResponse(status=302, headers={"Location": "https://www.acme.fr/"})
    fetcher, _ = make_fetcher({"https://www.acme.fr/": loop})
    with pytest.raises(FetchError, match="Trop de redirections"):
        fetcher.fetch("https://www.acme.fr/")


def test_non_html_content_is_refused():
    pdf = FakeResponse(headers={"Content-Type": "application/pdf"})
    fetcher, _ = make_fetcher({"https://www.acme.fr/doc": pdf})
    with pytest.raises(FetchError, match="Type de contenu"):
        fetcher.fetch("https://www.acme.fr/doc")


def test_huge_page_is_refused(monkeypatch):
    monkeypatch.setattr(webfetch, "MAX_PAGE_BYTES", 100)
    fetcher, _ = make_fetcher({"https://www.acme.fr/": FakeResponse(body=b"x" * 500)})
    with pytest.raises(FetchError, match="volumineuse"):
        fetcher.fetch("https://www.acme.fr/")


def test_declared_huge_page_is_refused_before_reading(monkeypatch):
    big = FakeResponse(headers={"Content-Length": str(50 * 1024 * 1024)})
    fetcher, _ = make_fetcher({"https://www.acme.fr/": big})
    with pytest.raises(FetchError, match="volumineuse"):
        fetcher.fetch("https://www.acme.fr/")


def test_robots_txt_is_respected():
    routes = {
        "https://www.acme.fr/robots.txt": FakeResponse(
            body=b"User-agent: *\nDisallow: /prive\n", headers={"Content-Type": "text/plain"}
        ),
        "https://www.acme.fr/public": FakeResponse(),
    }
    fetcher, session = make_fetcher(routes)
    assert fetcher.fetch("https://www.acme.fr/public").status == 200
    with pytest.raises(FetchError, match=r"robots\.txt"):
        fetcher.fetch("https://www.acme.fr/prive/page")
    assert "https://www.acme.fr/prive/page" not in session.requested
    assert session.requested.count("https://www.acme.fr/robots.txt") == 1  # mis en cache


def test_latin1_page_without_charset_is_decoded():
    body = "Société générale".encode("cp1252")
    response = FakeResponse(body=body, headers={"Content-Type": "text/html"}, encoding="ISO-8859-1")
    fetcher, _ = make_fetcher({"https://www.acme.fr/": response})
    assert "Société" in fetcher.fetch("https://www.acme.fr/").text


def test_slow_drip_server_cannot_block_us(monkeypatch):
    """Un serveur qui envoie quelques octets à la fois (slowloris) : abandon après la durée maximale."""
    clock = iter(range(0, 1000, 5))  # chaque lecture de l'horloge avance de 5 secondes
    monkeypatch.setattr(webfetch.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(webfetch, "MAX_PAGE_SECONDS", 20)
    drip = FakeResponse(body=b"x" * 100)
    drip.iter_content = lambda chunk_size: (b"x" for _ in range(1000))
    fetcher, _ = make_fetcher({"https://www.acme.fr/": drip})
    with pytest.raises(FetchError, match="trop lente"):
        fetcher.fetch("https://www.acme.fr/")
