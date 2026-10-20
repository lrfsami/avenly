"""Tests de la ligne de commande, sans réseau ni LLM réels."""

from __future__ import annotations

import io
from typing import ClassVar

import pytest

from avenly import cli
from avenly.config import get_settings
from avenly.llm import BudgetExceededError
from avenly.models import CandidateProfile, Company, CompanyInsight, EmailContent, Enrichment
from avenly.profiler import CVDocument, CVError
from avenly.security import CryptoManager, generate_key
from avenly.storage import Database

PROFILE = CandidateProfile(headline="Étudiant ingénieur data", summary="3A IDU.")


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Base de données temporaire + clé de chiffrement de test."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'avenly.db'}")
    monkeypatch.setenv("FERNET_KEY", generate_key())
    get_settings.cache_clear()
    return tmp_path


class FakeLLM:
    email_model = "claude-sonnet-5-5"

    def __init__(self, db, error: Exception | None = None):
        self.error = error
        self.calls = 0

    def extract_profile(self, cv, *, client_id):
        if self.error:
            raise self.error
        return PROFILE

    def write_email(self, profile, company_name, insight, context, *, client_id):
        self.calls += 1
        return EmailContent(
            subject=f"Candidature {company_name}", body="Madame, Monsieur,\n\nTexte.\n\nBien cordialement,"
        )


class FakeEnricher:
    def __init__(self, db, fetcher, *, llm=None):
        pass

    def enrich_many(self, companies, *, workers=4):
        return {c.siren: self.enrich(c) for c in companies}

    def enrich(self, company):
        return Enrichment(
            siren=company.siren,
            status="ok",
            generic_emails=["rh@acme.fr"],
            insight=CompanyInsight(summary="ESN."),
        )


class FakeFetcher:
    def close(self):
        pass


def companies(n: int) -> list[Company]:
    return [Company(client_id="moi", siret=f"{i:09d}00011", name=f"ACME{i} SAS") for i in range(1, n + 1)]


@pytest.fixture
def fakes(monkeypatch):
    created: dict[str, FakeLLM] = {}

    def make_llm(db):
        created["llm"] = FakeLLM(db)
        return created["llm"]

    monkeypatch.setattr(cli, "read_cv", lambda path, candidate_names: CVDocument("h", 1, "texte"))
    monkeypatch.setattr(cli, "find_companies", lambda *a, **k: companies(5))
    monkeypatch.setattr(cli, "ClaudeClient", make_llm)
    monkeypatch.setattr(cli, "Enricher", FakeEnricher)
    monkeypatch.setattr(cli, "SafeFetcher", FakeFetcher)
    return created


CAMPAGNE = ["campagne", "--cv", "cv.pdf", "--nom", "Sami Test", "--domaine", "data", "--ville", "Annecy",
            "--debut", "septembre 2027", "--max", "3"]  # fmt: skip


def run(argv, answers=("moi@exemple.fr", "", "o")):
    out, err = io.StringIO(), io.StringIO()
    replies = iter(answers)
    code = cli.main(argv, out=out, err=err, ask=lambda prompt: next(replies))
    return code, out.getvalue(), err.getvalue()


def test_campaign_creates_drafts(env, fakes):
    code, out, _ = run(CAMPAGNE)
    assert code == 0
    assert "pour 3 brouillon(s) maximum" in out
    assert "✅ 3 brouillon(s)" in out
    assert fakes["llm"].calls == 3

    drafts = Database().list_drafts(CryptoManager(), "moi")
    assert len(drafts) == 3
    assert drafts[0]["body"].endswith("Sami Test\nmoi@exemple.fr")  # signature ajoutée localement


def test_campaign_can_be_cancelled_before_spending(env, fakes):
    code, out, _ = run(CAMPAGNE, answers=("moi@exemple.fr", "", "n"))
    assert code == 0
    assert "annulée" in out
    assert "llm" not in fakes or fakes["llm"].calls == 0


def test_signature_from_env_is_not_asked(env, fakes, monkeypatch):
    monkeypatch.setenv("SIGNATURE_EMAIL", "env@exemple.fr")
    monkeypatch.setenv("SIGNATURE_PHONE", "06 00 00 00 00")
    get_settings.cache_clear()
    code, _, _ = run([*CAMPAGNE, "--oui"], answers=())  # aucune question posée
    assert code == 0


def test_second_campaign_skips_already_drafted_companies(env, fakes):
    run([*CAMPAGNE, "--oui"])
    code, out, _ = run([*CAMPAGNE, "--oui", "--max", "10"])
    assert code == 0
    assert "2 entreprise(s) candidates" in out  # 5 trouvées, 3 déjà en brouillon


def test_user_errors_are_readable(env, fakes, monkeypatch):
    def bad_cv(path, candidate_names):
        raise CVError("Le nom fourni n'apparaît pas dans le CV")

    monkeypatch.setattr(cli, "read_cv", bad_cv)
    code, _, err = run(CAMPAGNE)
    assert code == 1
    assert "❌ Le nom fourni" in err
    assert "Traceback" not in err


def test_budget_exceeded_message(env, fakes, monkeypatch):
    monkeypatch.setattr(
        cli, "ClaudeClient", lambda db: FakeLLM(db, error=BudgetExceededError("Budget atteint"))
    )
    code, _, err = run([*CAMPAGNE, "--oui"])
    assert code == 2
    assert "⛔ Budget atteint" in err


def test_missing_encryption_key_stops_before_any_spending(env, fakes, monkeypatch):
    monkeypatch.delenv("FERNET_KEY")
    get_settings.cache_clear()
    code, _, err = run(CAMPAGNE)
    assert code == 1
    assert "FERNET_KEY" in err
    assert "llm" not in fakes  # le client LLM n'a même pas été créé


def test_invalid_domain_is_rejected():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([*CAMPAGNE[:5], "--domaine", "plomberie"])


def test_brouillons_when_empty(env):
    code, out, _ = run(["brouillons"])
    assert code == 0
    assert "Aucun brouillon" in out


def test_brouillons_lists_drafts(env, fakes):
    run([*CAMPAGNE, "--oui"])
    code, out, _ = run(["brouillons"])
    assert code == 0
    assert "ACME1 SAS  →  rh@acme.fr" in out
    assert "3 brouillon(s)" in out


def test_couts(env):
    Database().record_llm_usage(
        client_id="moi",
        task="write_email",
        model="claude-sonnet-5-5",
        input_tokens=1,
        output_tokens=1,
        cost_usd=0.01,
    )
    code, out, _ = run(["couts"])
    assert code == 0
    assert "write_email" in out
    assert "0.0100 $" in out


def test_opposition(env):
    code, out, _ = run(["opposition", "123456789"])
    assert code == 0
    assert "ne sera plus jamais contactée" in out
    assert Database().opted_out_sirens(["123456789"]) == {"123456789"}


def test_opposition_invalid_siren(env):
    code, _, err = run(["opposition", "12"])
    assert code == 1
    assert "SIREN invalide" in err


def test_companies_without_channel_are_skipped_without_drafting(env, fakes, monkeypatch):
    """Pas d'email ni de page carrières : on ne paie pas la rédaction, on passe à la suivante."""

    class HalfEnricher(FakeEnricher):
        def enrich(self, company):
            enrichment = super().enrich(company)
            if company.siren.endswith(("1", "3")):  # 2 entreprises sur 5 sans aucun canal
                enrichment.generic_emails = []
            return enrichment

    monkeypatch.setattr(cli, "Enricher", HalfEnricher)
    code, out, _ = run([*CAMPAGNE, "--oui"])
    assert code == 0
    assert out.count("ignorée") == 2
    assert "✅ 3 brouillon(s)" in out  # --max 3 atteint malgré les entreprises ignorées
    assert fakes["llm"].calls == 3


# --------------------------------------------------------------------------
# Validation et envoi
# --------------------------------------------------------------------------


class RecordingSMTP:
    sent: ClassVar[list] = []

    def login(self, user, password):
        pass

    def send_message(self, message):
        RecordingSMTP.sent.append(message)
        return {}

    def quit(self):
        pass


@pytest.fixture
def mail(env, fakes, monkeypatch, tmp_path):
    """Campagne de 3 brouillons + faux serveur mail + pas d'attente réelle entre deux envois."""
    from avenly.sender import EmailSender

    monkeypatch.setenv("SMTP_USER", "sami@exemple.fr")
    monkeypatch.setenv("SMTP_APP_PASSWORD", "app-password")
    get_settings.cache_clear()
    RecordingSMTP.sent = []
    monkeypatch.setattr(
        cli, "EmailSender", lambda settings: EmailSender(settings, smtp_factory=lambda s: RecordingSMTP())
    )
    sleeps: list[float] = []
    monkeypatch.setattr(cli, "SLEEP", sleeps.append)
    run([*CAMPAGNE, "--oui"])
    cv = tmp_path / "CV.pdf"
    cv.write_bytes(b"%PDF-1.7 cv")
    return {"cv": str(cv), "sleeps": sleeps}


def test_nothing_is_sent_without_approval(mail):
    code, out, _ = run(["envoyer", "--cv", mail["cv"]], answers=("o",))
    assert code == 0
    assert "Aucun brouillon validé" in out
    assert RecordingSMTP.sent == []


def test_approve_then_send_with_confirmation(mail):
    ids = [d["id"] for d in Database().list_drafts(CryptoManager(), "moi")]
    code, out, _ = run(["valider", str(ids[0]), str(ids[2]), "999"])
    assert "validé(s) : " in out and "#999" in out

    code, out, _ = run(["envoyer", "--cv", mail["cv"], "--nom", "Sami Test"], answers=("o",))
    assert code == 0
    assert "2 email(s) prêt(s)" in out
    assert len(RecordingSMTP.sent) == 2
    assert RecordingSMTP.sent[0]["To"] == "rh@acme.fr"
    assert len(mail["sleeps"]) == 1  # une pause entre les deux envois, pas après le dernier
    assert 120 <= mail["sleeps"][0] <= 360

    statuses = {d["id"]: d["status"] for d in Database().list_drafts(CryptoManager(), "moi")}
    assert statuses[ids[0]] == statuses[ids[2]] == "sent"
    assert statuses[ids[1]] == "draft"


def test_send_can_be_cancelled(mail):
    ids = [d["id"] for d in Database().list_drafts(CryptoManager(), "moi")]
    run(["valider", str(ids[0])])
    _, out, _ = run(["envoyer", "--cv", mail["cv"]], answers=("n",))
    assert "rien n'est parti" in out
    assert RecordingSMTP.sent == []


def test_test_mode_sends_to_myself_only(mail):
    ids = [d["id"] for d in Database().list_drafts(CryptoManager(), "moi")]
    run(["valider", str(ids[0])])
    run(["envoyer", "--cv", mail["cv"], "--essai"], answers=("o",))
    assert len(RecordingSMTP.sent) == 1
    assert RecordingSMTP.sent[0]["To"] == "sami@exemple.fr"
    assert RecordingSMTP.sent[0]["Subject"].startswith("[ESSAI Avenly]")
    # Le brouillon reste « validé » : l'essai ne compte pas comme un envoi.
    assert {d["id"]: d["status"] for d in Database().list_drafts(CryptoManager(), "moi")}[
        ids[0]
    ] == "approved"


def test_one_by_one_mode(mail):
    ids = [d["id"] for d in Database().list_drafts(CryptoManager(), "moi")]
    run(["valider", *map(str, ids)])
    # confirmation globale, puis : envoyer le 1er, passer le 2e, tout arrêter au 3e
    _, out, _ = run(["envoyer", "--cv", mail["cv"], "--un-par-un"], answers=("o", "o", "n", "q"))
    assert len(RecordingSMTP.sent) == 1
    assert "Passé." in out and "Arrêt demandé" in out


def test_daily_limit(mail, monkeypatch):
    monkeypatch.setenv("SEND_DAILY_LIMIT", "1")
    get_settings.cache_clear()
    ids = [d["id"] for d in Database().list_drafts(CryptoManager(), "moi")]
    run(["valider", *map(str, ids)])
    run(["envoyer", "--cv", mail["cv"]], answers=("o",))
    assert len(RecordingSMTP.sent) == 1
    _, out, _ = run(["envoyer", "--cv", mail["cv"]], answers=("o",))
    assert "Limite quotidienne atteinte" in out
    assert len(RecordingSMTP.sent) == 1


def test_invalid_attachment_stops_before_sending(mail, tmp_path):
    ids = [d["id"] for d in Database().list_drafts(CryptoManager(), "moi")]
    run(["valider", str(ids[0])])
    not_pdf = tmp_path / "cv.pdf"
    not_pdf.write_bytes(b"pas un pdf")
    code, _, err = run(["envoyer", "--cv", str(not_pdf)], answers=("o",))
    assert code == 1
    assert "pas un PDF" in err
    assert RecordingSMTP.sent == []


def test_missing_gmail_settings_stop_before_anything(env, fakes, monkeypatch, tmp_path):
    """Ton cas réel : sans SMTP_USER, on s'arrête AVANT le récapitulatif et la confirmation."""
    run([*CAMPAGNE, "--oui"])
    ids = [d["id"] for d in Database().list_drafts(CryptoManager(), "moi")]
    run(["valider", str(ids[0])])
    cv = tmp_path / "cv.pdf"
    cv.write_bytes(b"%PDF-1.7")
    asked: list[str] = []
    out, err = io.StringIO(), io.StringIO()
    code = cli.main(["envoyer", "--cv", str(cv)], out=out, err=err, ask=lambda p: asked.append(p) or "o")
    assert code == 1
    assert "SMTP_USER" in err.getvalue()
    assert asked == []  # aucune question posée
    assert "prêt" not in out.getvalue()


def test_foreign_region_address_is_blocked_at_send_time(mail, monkeypatch):
    """Ton cas réel : un vieux brouillon vers « contact.apac@ » ne doit pas partir."""
    drafts = Database().list_drafts(CryptoManager(), "moi")
    with Database().session() as session:
        from avenly.storage import DraftRecord

        session.get(DraftRecord, drafts[0]["id"]).recipient_email = "contact.apac@digitalrealty.com"
    run(["valider", str(drafts[0]["id"]), str(drafts[1]["id"])])

    code, out, _ = run(["envoyer", "--cv", mail["cv"]], answers=("o",))

    assert code == 0
    assert "⛔ Bloqué" in out and "contact.apac@digitalrealty.com" in out
    assert [m["To"] for m in RecordingSMTP.sent] == ["rh@acme.fr"]  # seul l'autre est parti


def test_delete_drafts(mail):
    drafts = Database().list_drafts(CryptoManager(), "moi")
    first, second = drafts[0]["id"], drafts[1]["id"]
    run(["valider", str(second)])
    run(["envoyer", "--cv", mail["cv"]], answers=("o",))  # le 2e est envoyé

    code, out, _ = run(["supprimer", str(first), str(second), "999"])

    assert code == 0
    assert f"supprimé(s) : #{first}" in out
    assert f"#{second}" in out.split("Ignoré(s)")[1]  # déjà envoyé : conservé
    remaining = {d["id"] for d in Database().list_drafts(CryptoManager(), "moi")}
    assert first not in remaining and second in remaining
    # L'entreprise du brouillon supprimé redevient contactable.
    siret = drafts[0]["siret"]
    assert Database().contactable([c for c in companies(5) if c.siret == siret], "moi")


def test_brouillons_to_review_only(mail):
    drafts = Database().list_drafts(CryptoManager(), "moi")
    run(["valider", str(drafts[0]["id"])])
    _, out, _ = run(["brouillons", "--a-relire"])
    assert f"#{drafts[0]['id']} " not in out
    assert f"#{drafts[1]['id']} [à relire]" in out
