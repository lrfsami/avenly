"""Envoi des candidatures validées, depuis la boîte mail du candidat (SMTP).

Garde-fous :
- seuls les brouillons VALIDÉS par le candidat partent (`avenly valider`), après une
  confirmation explicite au moment de l'envoi ;
- connexion toujours chiffrée (SMTPS 465 ou STARTTLS 587) avec vérification du certificat ;
- protection contre l'injection d'en-têtes (destinataire et objet sans retour à la ligne) ;
- CV joint vérifié (vrai PDF, taille bornée), aucun pixel de suivi, aucun lien caché ;
- une ligne de désinscription en bas de chaque email (bonne pratique RGPD) ;
- limite quotidienne et délai aléatoire entre deux envois, arrêt immédiat en cas d'erreur.
"""

from __future__ import annotations

import contextlib
import logging
import re
import smtplib
import ssl
from collections.abc import Callable
from dataclasses import dataclass
from email.message import EmailMessage
from email.utils import formataddr, make_msgid
from pathlib import Path

from avenly.config import Settings, get_settings
from avenly.enricher import FOREIGN_REGION_SUFFIX

logger = logging.getLogger(__name__)

OPT_OUT_LINE = (
    "Si vous préférez ne plus recevoir de message de ma part, il vous suffit de me répondre : "
    "je ne vous recontacterai pas."
)
MAX_ATTACHMENT_BYTES = 5 * 1024 * 1024
_EMAIL = re.compile(r"^[A-Za-z0-9._%+\-]{1,64}@[A-Za-z0-9\-]{1,63}(?:\.[A-Za-z0-9\-]{1,63}){1,5}$")

SmtpFactory = Callable[[Settings], "smtplib.SMTP"]


class SendError(RuntimeError):
    """Envoi impossible (configuration, pièce jointe, refus du serveur…)."""


@dataclass(frozen=True, slots=True)
class OutgoingEmail:
    to: str
    subject: str
    body: str


def check_smtp_settings(settings: Settings) -> None:
    """À appeler AVANT toute autre étape : inutile de préparer un envoi impossible."""
    if not settings.smtp_user or settings.smtp_app_password is None:
        raise SendError(
            "SMTP_USER et SMTP_APP_PASSWORD manquent dans le .env "
            "(Gmail : crée un mot de passe d'application, jamais ton vrai mot de passe)."
        )


def recipient_problem(address: str | None) -> str | None:
    """Revérifie un destinataire au moment de l'envoi, avec les règles ACTUELLES.

    Un brouillon rédigé avant une correction (ex. « contact.apac@ ») ne doit pas partir.
    """
    if not address or not _EMAIL.match(address):
        return "adresse invalide"
    local = address.split("@", 1)[0].lower()
    if FOREIGN_REGION_SUFFIX.search(local):
        return "adresse destinée à une autre région du monde"
    return None


def _safe_header(value: str, label: str) -> str:
    """Refuse les retours à la ligne : ils permettraient d'injecter des en-têtes (Bcc, etc.)."""
    if "\r" in value or "\n" in value:
        raise SendError(f"{label} invalide : retour à la ligne interdit dans un en-tête.")
    return value.strip()


def load_attachment(path: str | Path) -> tuple[str, bytes]:
    """Lit le CV à joindre : un vrai PDF (signature %PDF-), 5 Mo maximum."""
    path = Path(path)
    if not path.is_file():
        raise SendError(f"CV à joindre introuvable : {path.name}")
    if path.stat().st_size > MAX_ATTACHMENT_BYTES:
        raise SendError("CV à joindre trop volumineux (5 Mo maximum).")
    data = path.read_bytes()
    if not data.startswith(b"%PDF-"):
        raise SendError("Le fichier à joindre n'est pas un PDF.")
    return path.name, data


def build_message(
    email: OutgoingEmail, *, sender_name: str, sender_email: str, attachment: tuple[str, bytes] | None
) -> EmailMessage:
    to = _safe_header(email.to, "Destinataire")
    if not _EMAIL.match(to):
        raise SendError(f"Adresse du destinataire invalide : {to!r}")
    message = EmailMessage()
    message["From"] = formataddr((_safe_header(sender_name, "Nom"), _safe_header(sender_email, "Expéditeur")))
    message["To"] = to
    message["Subject"] = _safe_header(email.subject, "Objet")
    message["Message-ID"] = make_msgid(domain=sender_email.rsplit("@", 1)[-1])
    message.set_content(f"{email.body}\n\n--\n{OPT_OUT_LINE}\n")  # texte brut : ni pixel ni lien caché
    if attachment is not None:
        filename, data = attachment
        message.add_attachment(data, maintype="application", subtype="pdf", filename=filename)
    return message


def default_smtp_factory(settings: Settings) -> smtplib.SMTP:
    """Connexion chiffrée avec vérification du certificat du serveur."""
    context = ssl.create_default_context()
    timeout = settings.http_timeout_seconds * 3
    if settings.smtp_port == 465:
        return smtplib.SMTP_SSL(settings.smtp_host, 465, context=context, timeout=timeout)
    smtp = smtplib.SMTP(settings.smtp_host, 587, timeout=timeout)
    smtp.ehlo()
    if not smtp.has_extn("starttls"):
        smtp.quit()
        raise SendError("Le serveur ne propose pas de chiffrement (STARTTLS) : envoi refusé.")
    smtp.starttls(context=context)
    smtp.ehlo()
    return smtp


class EmailSender:
    """Ouvre UNE connexion SMTP authentifiée et envoie les messages un par un."""

    def __init__(self, settings: Settings | None = None, *, smtp_factory: SmtpFactory = default_smtp_factory):
        self._settings = settings or get_settings()
        self._factory = smtp_factory
        self._smtp: smtplib.SMTP | None = None
        check_smtp_settings(self._settings)

    @property
    def sender_email(self) -> str:
        return self._settings.smtp_user or ""

    def __enter__(self) -> EmailSender:
        s = self._settings
        try:
            self._smtp = self._factory(s)
            self._smtp.login(s.smtp_user, s.smtp_app_password.get_secret_value())
        except smtplib.SMTPAuthenticationError:
            raise SendError(
                "Connexion refusée par le serveur mail : vérifie SMTP_USER et le mot de passe d'application."
            ) from None
        except (smtplib.SMTPException, OSError) as exc:
            raise SendError(f"Serveur mail injoignable ({type(exc).__name__}).") from None
        return self

    def __exit__(self, *exc: object) -> None:
        if self._smtp is not None:
            with contextlib.suppress(smtplib.SMTPException, OSError):
                self._smtp.quit()
            self._smtp = None

    def send(self, message: EmailMessage) -> None:
        if self._smtp is None:
            raise SendError("Connexion SMTP non ouverte.")
        try:
            refused = self._smtp.send_message(message)
        except smtplib.SMTPRecipientsRefused:
            raise SendError(f"Destinataire refusé par le serveur : {message['To']}") from None
        except smtplib.SMTPDataError as exc:
            # Quota atteint, message bloqué comme spam… : on s'arrête net.
            raise SendError(f"Le serveur mail a refusé l'envoi (code {exc.smtp_code}).") from None
        except (smtplib.SMTPException, OSError) as exc:
            raise SendError(f"Échec de l'envoi ({type(exc).__name__}).") from None
        if refused:
            raise SendError(f"Destinataire refusé : {', '.join(refused)}")
        logger.info("Email envoyé à %s", message["To"])
