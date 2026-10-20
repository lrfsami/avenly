"""Tests de l'envoi : aucun email réel ne part, le serveur SMTP est simulé."""

from __future__ import annotations

import smtplib
from typing import ClassVar

import pytest

from avenly import sender as sender_module
from avenly.config import get_settings
from avenly.sender import (
    OPT_OUT_LINE,
    EmailSender,
    OutgoingEmail,
    SendError,
    build_message,
    load_attachment,
)

PDF = b"%PDF-1.7\n% faux CV pour les tests\n"


class FakeSMTP:
    """Serveur SMTP simulé : enregistre les connexions et les messages."""

    instances: ClassVar[list[FakeSMTP]] = []

    def __init__(self, *, fail_login=False, data_error=False):
        self.logged_in = None
        self.sent: list = []
        self.quit_called = False
        self.fail_login = fail_login
        self.data_error = data_error
        FakeSMTP.instances.append(self)

    def login(self, user, password):
        if self.fail_login:
            raise smtplib.SMTPAuthenticationError(535, b"bad credentials")
        self.logged_in = (user, password)

    def send_message(self, message):
        if self.data_error:
            raise smtplib.SMTPDataError(550, b"quota")
        self.sent.append(message)
        return {}

    def quit(self):
        self.quit_called = True


@pytest.fixture
def smtp_env(monkeypatch):
    monkeypatch.setenv("SMTP_USER", "sami@exemple.fr")
    monkeypatch.setenv("SMTP_APP_PASSWORD", "mot-de-passe-application")
    get_settings.cache_clear()
    FakeSMTP.instances.clear()


@pytest.fixture
def cv_file(tmp_path):
    path = tmp_path / "CV_Sami.pdf"
    path.write_bytes(PDF)
    return path


EMAIL = OutgoingEmail(to="rh@acme.fr", subject="Candidature alternance", body="Madame, Monsieur,\n\nBonjour.")


# --------------------------------------------------------------------------
# Construction du message
# --------------------------------------------------------------------------


def test_message_has_attachment_and_opt_out_line(cv_file):
    message = build_message(
        EMAIL, sender_name="Sami Laarfi", sender_email="sami@exemple.fr", attachment=load_attachment(cv_file)
    )
    assert message["To"] == "rh@acme.fr"
    assert message["From"] == "Sami Laarfi <sami@exemple.fr>"
    body = message.get_body(preferencelist=("plain",)).get_content()
    assert body.startswith("Madame, Monsieur,")
    assert OPT_OUT_LINE in body
    attachments = list(message.iter_attachments())
    assert [a.get_filename() for a in attachments] == ["CV_Sami.pdf"]
    assert attachments[0].get_content() == PDF


def test_message_is_plain_text_without_tracking():
    message = build_message(EMAIL, sender_name="S", sender_email="s@exemple.fr", attachment=None)
    raw = message.as_string()
    assert "text/html" not in raw  # ni HTML, ni pixel de suivi, ni lien caché
    assert "<img" not in raw


@pytest.mark.parametrize(
    ("field", "value"),
    [("to", "rh@acme.fr\nBcc: tout-le-monde@exemple.fr"), ("subject", "Candidature\r\nBcc: x@exemple.fr")],
)
def test_header_injection_is_refused(field, value):
    email = OutgoingEmail(**({"to": EMAIL.to, "subject": EMAIL.subject, "body": EMAIL.body} | {field: value}))
    with pytest.raises(SendError, match="retour à la ligne"):
        build_message(email, sender_name="S", sender_email="s@exemple.fr", attachment=None)


@pytest.mark.parametrize("to", ["pas-un-email", "rh@", "a b@acme.fr", "rh@acme"])
def test_invalid_recipient_is_refused(to):
    email = OutgoingEmail(to=to, subject="s", body="b")
    with pytest.raises(SendError, match="invalide"):
        build_message(email, sender_name="S", sender_email="s@exemple.fr", attachment=None)


# --------------------------------------------------------------------------
# Pièce jointe
# --------------------------------------------------------------------------


def test_attachment_must_be_a_pdf(tmp_path):
    fake = tmp_path / "cv.pdf"
    fake.write_bytes(b"MZ\x90 exe renomme")
    with pytest.raises(SendError, match="pas un PDF"):
        load_attachment(fake)


def test_attachment_size_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(sender_module, "MAX_ATTACHMENT_BYTES", 10)
    big = tmp_path / "cv.pdf"
    big.write_bytes(PDF)
    with pytest.raises(SendError, match="volumineux"):
        load_attachment(big)


def test_missing_attachment(tmp_path):
    with pytest.raises(SendError, match="introuvable"):
        load_attachment(tmp_path / "absent.pdf")


# --------------------------------------------------------------------------
# Connexion et envoi
# --------------------------------------------------------------------------


def test_missing_smtp_settings():
    with pytest.raises(SendError, match="SMTP_USER"):
        EmailSender()


def test_send_through_fake_server(smtp_env):
    with EmailSender(smtp_factory=lambda s: FakeSMTP()) as sender:
        sender.send(build_message(EMAIL, sender_name="S", sender_email=sender.sender_email, attachment=None))
    server = FakeSMTP.instances[0]
    assert server.logged_in == ("sami@exemple.fr", "mot-de-passe-application")
    assert len(server.sent) == 1
    assert server.quit_called


def test_bad_credentials_message_does_not_leak_password(smtp_env):
    with pytest.raises(SendError) as exc, EmailSender(smtp_factory=lambda s: FakeSMTP(fail_login=True)):
        pass
    assert "mot de passe d'application" in str(exc.value)
    assert "mot-de-passe-application" not in str(exc.value)


def test_server_refusal_stops_sending(smtp_env):
    with EmailSender(smtp_factory=lambda s: FakeSMTP(data_error=True)) as sender:
        message = build_message(EMAIL, sender_name="S", sender_email=sender.sender_email, attachment=None)
        with pytest.raises(SendError, match="refusé"):
            sender.send(message)


@pytest.mark.parametrize("port", ["25", "2525", "80"])
def test_unencrypted_ports_are_refused(monkeypatch, port):
    from avenly.config import ConfigError

    monkeypatch.setenv("SMTP_PORT", port)
    get_settings.cache_clear()
    with pytest.raises(ConfigError, match="chiffrés"):
        get_settings()


def test_starttls_is_mandatory_on_port_587(smtp_env, monkeypatch):
    """Un serveur qui ne propose pas le chiffrement : on refuse d'envoyer quoi que ce soit."""

    class PlainServer:
        def __init__(self, host, port, timeout):
            self.closed = False

        def ehlo(self):
            pass

        def has_extn(self, name):
            return False

        def quit(self):
            self.closed = True

    monkeypatch.setenv("SMTP_PORT", "587")
    get_settings.cache_clear()
    monkeypatch.setattr(sender_module.smtplib, "SMTP", PlainServer)
    with pytest.raises(SendError, match="chiffrement"):
        sender_module.default_smtp_factory(get_settings())


def test_tls_certificate_is_verified(smtp_env, monkeypatch):
    captured = {}

    def fake_smtp_ssl(host, port, context, timeout):
        captured["context"] = context
        return FakeSMTP()

    monkeypatch.setattr(sender_module.smtplib, "SMTP_SSL", fake_smtp_ssl)
    sender_module.default_smtp_factory(get_settings())
    assert captured["context"].check_hostname is True
    assert captured["context"].verify_mode.name == "CERT_REQUIRED"


@pytest.mark.parametrize(
    ("address", "problem"),
    [
        ("rh@acme.fr", None),
        ("contact.france@acme.com", None),
        ("contact.apac@digitalrealty.com", "autre région"),
        ("jobs-us@acme.com", "autre région"),
        ("pas-une-adresse", "invalide"),
        (None, "invalide"),
    ],
)
def test_recipient_is_rechecked_at_send_time(address, problem):
    from avenly.sender import recipient_problem

    result = recipient_problem(address)
    assert (result is None) if problem is None else (problem in result)
