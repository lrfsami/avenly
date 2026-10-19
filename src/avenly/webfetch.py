"""Lecture sécurisée de pages web inconnues (sites d'entreprises).

Menace principale : la SSRF (Server-Side Request Forgery). Une URL piégée
(« http://127.0.0.1:8080 », « http://169.254.169.254/ » sur un serveur cloud, ou
un nom de domaine qui pointe vers une adresse interne) pourrait faire lire au
programme des ressources privées. Protections :

- schéma http/https et ports standards (80, 443) uniquement ;
- le nom de domaine est résolu AVANT la requête : toute adresse non publique est refusée ;
- l'adresse réellement contactée est revérifiée APRÈS connexion (contre le « DNS rebinding ») ;
- les redirections sont suivies à la main, chacune revalidée (3 au maximum) ;
- taille (2 Mo), type (HTML/texte) et durée bornés ;
- robots.txt respecté et une requête par seconde maximum par site.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import requests

from avenly.config import Settings, get_settings

logger = logging.getLogger(__name__)

USER_AGENT = "avenly/0.1 (prospection alternance ; respecte robots.txt)"
MAX_PAGE_BYTES = 2 * 1024 * 1024
MAX_REDIRECTS = 3
ALLOWED_PORTS = {None, 80, 443}
ALLOWED_TYPES = ("text/html", "application/xhtml+xml", "text/plain")
PER_HOST_INTERVAL_SECONDS = 1.0

Resolver = Callable[[str], list[str]]


class FetchError(RuntimeError):
    """Page refusée ou illisible (URL interdite, trop grosse, erreur réseau…)."""


@dataclass(frozen=True, slots=True)
class Page:
    url: str  # URL finale, après redirections
    status: int
    content_type: str
    text: str


def system_resolver(host: str) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        raise FetchError(f"Nom de domaine introuvable : {host}") from None
    return sorted({info[4][0] for info in infos})


def is_public_ip(ip: str) -> bool:
    try:
        address = ipaddress.ip_address(ip.split("%")[0])
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        address = address.ipv4_mapped
    return address.is_global and not address.is_multicast


def check_url(url: str, resolver: Resolver = system_resolver) -> str:
    """Valide une URL et vérifie que son nom de domaine ne pointe que vers des adresses publiques."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise FetchError("Seules les URLs http et https sont autorisées.")
    if parts.username or parts.password:
        raise FetchError("URL avec identifiants refusée.")
    try:
        port = parts.port
    except ValueError:
        raise FetchError("Port invalide.") from None
    if port not in ALLOWED_PORTS:
        raise FetchError("Port non standard refusé.")
    host = (parts.hostname or "").rstrip(".").lower()
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise FetchError("Les adresses IP brutes sont refusées : un nom de domaine est requis.")
    if not host or "." not in host:
        raise FetchError("Nom de domaine invalide.")
    addresses = resolver(host)
    if not addresses or not all(is_public_ip(ip) for ip in addresses):
        raise FetchError(f"{host} pointe vers une adresse non publique : accès refusé.")
    return host


def _peer_ip(response: requests.Response) -> str | None:
    """Adresse IP réellement contactée (None si on ne peut pas la lire)."""
    try:
        sock = response.raw.connection.sock  # type: ignore[union-attr]
        return sock.getpeername()[0]
    except Exception:
        return None


class SafeFetcher:
    """Client HTTP pour lire des sites inconnus sans risque de SSRF."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        resolver: Resolver = system_resolver,
        session: requests.Session | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._resolver = resolver
        self._session = session or requests.Session()
        self._session.headers.update(
            {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml;q=0.9,text/plain;q=0.5"}
        )
        self._robots: dict[str, RobotFileParser | None] = {}
        self._last_call: dict[str, float] = {}
        self._lock = threading.Lock()

    def close(self) -> None:
        self._session.close()

    # ------------------------------------------------------------------ politesse

    def _wait_turn(self, host: str) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._last_call.get(host, 0.0) + PER_HOST_INTERVAL_SECONDS - now
            self._last_call[host] = now + max(wait, 0.0)
        if wait > 0:
            time.sleep(wait)

    def allowed_by_robots(self, url: str) -> bool:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._robots:
            parser: RobotFileParser | None = RobotFileParser()
            try:
                page = self._get(urljoin(origin, "/robots.txt"), check_robots=False)
                if page.status >= 400:
                    parser = None  # pas de robots.txt : tout est permis
                else:
                    parser.parse(page.text.splitlines())
            except FetchError:
                parser = None
            self._robots[origin] = parser
        parser = self._robots[origin]
        return parser is None or parser.can_fetch(USER_AGENT, url)

    # ------------------------------------------------------------------ requêtes

    def fetch(self, url: str) -> Page:
        """Lit une page HTML/texte en suivant au plus 3 redirections, toutes revalidées."""
        if not self.allowed_by_robots(url):
            raise FetchError("Page interdite par le robots.txt du site.")
        return self._get(url, check_robots=True)

    def _get(self, url: str, *, check_robots: bool) -> Page:
        for _ in range(MAX_REDIRECTS + 1):
            host = check_url(url, self._resolver)
            self._wait_turn(host)
            try:
                response = self._session.get(
                    url,
                    timeout=(5, self._settings.http_timeout_seconds),
                    allow_redirects=False,
                    stream=True,
                )
            except requests.RequestException as exc:
                raise FetchError(f"{host} injoignable ({type(exc).__name__}).") from None

            with response:
                peer = _peer_ip(response)
                if peer is not None and not is_public_ip(peer):
                    raise FetchError(f"{host} a été résolu vers une adresse non publique : accès refusé.")

                if response.is_redirect:
                    location = response.headers.get("Location")
                    if not location:
                        raise FetchError("Redirection sans destination.")
                    url = urljoin(url, location)
                    if check_robots and not self.allowed_by_robots(url):
                        raise FetchError("Redirection vers une page interdite par robots.txt.")
                    continue

                content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
                if response.status_code < 400 and content_type not in ALLOWED_TYPES:
                    raise FetchError(f"Type de contenu refusé : {content_type or 'inconnu'}.")
                declared = response.headers.get("Content-Length")
                if declared and declared.isdigit() and int(declared) > MAX_PAGE_BYTES:
                    raise FetchError("Page trop volumineuse.")

                body = bytearray()
                for chunk in response.iter_content(chunk_size=65536):
                    body.extend(chunk)
                    if len(body) > MAX_PAGE_BYTES:
                        raise FetchError("Page trop volumineuse.")
                text = _decode(bytes(body), response.encoding)
                return Page(url=url, status=response.status_code, content_type=content_type, text=text)

        raise FetchError("Trop de redirections.")


def _decode(body: bytes, declared: str | None) -> str:
    """Décode la page. Sans charset déclaré, requests suppose ISO-8859-1 : on essaie d'abord UTF-8."""
    if declared and declared.lower() not in ("iso-8859-1", "latin-1"):
        try:
            return body.decode(declared, errors="replace")
        except LookupError:
            pass
    try:
        return body.decode("utf-8")
    except UnicodeDecodeError:
        return body.decode("cp1252", errors="replace")
