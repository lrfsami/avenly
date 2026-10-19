"""Tests de la rédaction des candidatures, sans appel réel au LLM."""

from __future__ import annotations

import pytest

from avenly.llm import BudgetExceededError, LLMClient
from avenly.models import (
    ApplicationContext,
    CandidateProfile,
    Company,
    CompanyInsight,
    EmailContent,
    Enrichment,
    Signature,
)
from avenly.security import CryptoManager, generate_key
from avenly.storage import ApplicationStatus, Database, DraftRecord
from avenly.writer import EmailWriter, check_email, one_per_company

PROFILE = CandidateProfile(headline="Étudiant en informatique", summary="Licence 2.", hard_skills=["Linux"])
CONTEXT = ApplicationContext(start="septembre 2027", rhythm="3 semaines / 1 semaine")
SIGNATURE = Signature(full_name="Sami Laarfi", phone="06 12 34 56 78", email="sami@exemple.fr")
BODY = "Madame, Monsieur,\n\nVotre activité d'infogérance m'intéresse.\n\nBien cordialement,"


class FakeLLM(LLMClient):
    email_model = "claude-sonnet-5-5"

    def __init__(self, content: EmailContent | None = None, error: Exception | None = None) -> None:
        self.content = content or EmailContent(subject="Candidature alternance", body=BODY)
        self.error = error
        self.calls: list[dict] = []

    def extract_profile(self, cv, *, client_id):  # pragma: no cover
        raise NotImplementedError

    def summarize_company(self, company_name, site_text):  # pragma: no cover
        raise NotImplementedError

    def write_email(self, profile, company_name, insight, context, *, client_id):
        self.calls.append(
            {"company": company_name, "insight": insight, "context": context, "client_id": client_id}
        )
        if self.error:
            raise self.error
        return self.content


@pytest.fixture
def db(tmp_path) -> Database:
    database = Database(f"sqlite:///{tmp_path / 'avenly.db'}")
    yield database
    database.close()


@pytest.fixture
def crypto() -> CryptoManager:
    return CryptoManager(generate_key())


def company(siret: str, **extra) -> Company:
    return Company(**({"client_id": "moi", "siret": siret, "name": f"Entreprise {siret[:3]}"} | extra))


def enrichment(siren: str, **extra) -> Enrichment:
    fields = {
        "siren": siren,
        "status": "ok",
        "generic_emails": ["rh@acme.fr", "contact@acme.fr"],
        "careers_url": "https://acme.fr/carrieres",
        "insight": CompanyInsight(summary="ESN spécialisée en infogérance."),
    }
    return Enrichment(**(fields | extra))


def writer(db, crypto, llm=None) -> tuple[EmailWriter, FakeLLM]:
    llm = llm or FakeLLM()
    return EmailWriter(db, llm, crypto), llm


# --------------------------------------------------------------------------
# Rédaction
# --------------------------------------------------------------------------


def test_draft_adds_signature_locally(db, crypto):
    email_writer, llm = writer(db, crypto)
    draft = email_writer.draft_one(
        PROFILE, company("11111111100011"), enrichment("111111111"), CONTEXT, SIGNATURE, client_id="moi"
    )
    assert draft.body.startswith("Madame, Monsieur,")
    assert draft.body.endswith("Sami Laarfi\n06 12 34 56 78\nsami@exemple.fr")
    # Le LLM n'a reçu ni le nom ni les coordonnées du candidat.
    assert "Sami" not in str(llm.calls[0])
    assert "06 12" not in str(llm.calls[0])


def test_recruitment_email_and_careers_page_are_used(db, crypto):
    email_writer, _ = writer(db, crypto)
    draft = email_writer.draft_one(
        PROFILE, company("11111111100011"), enrichment("111111111"), CONTEXT, SIGNATURE, client_id="moi"
    )
    assert draft.recipient_email == "rh@acme.fr"
    assert draft.careers_url == "https://acme.fr/carrieres"
    assert draft.warnings == []


def test_lba_apply_url_is_used_when_no_careers_page(db, crypto):
    email_writer, _ = writer(db, crypto)
    target = company("11111111100011", apply_url="https://labonnealternance.apprentissage.beta.gouv.fr/x")
    draft = email_writer.draft_one(PROFILE, target, None, CONTEXT, SIGNATURE, client_id="moi")
    assert draft.careers_url.startswith("https://labonnealternance")
    assert any("générique" in w for w in draft.warnings)


def test_no_channel_is_reported(db, crypto):
    email_writer, _ = writer(db, crypto)
    draft = email_writer.draft_one(
        PROFILE, company("11111111100011"), None, CONTEXT, SIGNATURE, client_id="moi"
    )
    assert draft.recipient_email is None
    assert any("Aucun email" in w for w in draft.warnings)


# --------------------------------------------------------------------------
# Stockage chiffré
# --------------------------------------------------------------------------


def test_draft_is_stored_encrypted(db, crypto):
    email_writer, _ = writer(db, crypto)
    email_writer.draft_one(
        PROFILE, company("11111111100011"), enrichment("111111111"), CONTEXT, SIGNATURE, client_id="moi"
    )
    with db.session() as session:
        record = session.query(DraftRecord).one()
        assert b"Madame" not in record.body_encrypted  # illisible sans la clé
        assert b"Sami" not in record.body_encrypted

    drafts = db.list_drafts(crypto, "moi")
    assert drafts[0]["body"].startswith("Madame, Monsieur,")
    assert drafts[0]["company"] == "Entreprise 111"


def test_drafts_are_unreadable_with_another_key(db, crypto):
    from avenly.security import SecurityError

    email_writer, _ = writer(db, crypto)
    email_writer.draft_one(PROFILE, company("11111111100011"), None, CONTEXT, SIGNATURE, client_id="moi")
    with pytest.raises(SecurityError):
        db.list_drafts(CryptoManager(generate_key()), "moi")


# --------------------------------------------------------------------------
# Sélection des entreprises
# --------------------------------------------------------------------------


def test_one_per_company():
    sites = [company("11111111100011"), company("11111111100099"), company("22222222200022")]
    assert [c.siret for c in one_per_company(sites)] == ["11111111100011", "22222222200022"]


def test_draft_all_skips_contacted_and_opted_out_companies(db, crypto):
    email_writer, llm = writer(db, crypto)
    targets = [
        company("11111111100011"),
        company("11111111100099"),
        company("22222222200022"),
        company("33333333300033"),
    ]
    db.upsert_companies(targets)
    db.record_application("moi", "22222222200022", ApplicationStatus.SENT)  # déjà contactée
    db.add_opt_out("333333333")  # en opposition

    drafts = email_writer.draft_all(PROFILE, targets, {}, CONTEXT, SIGNATURE, client_id="moi")

    assert [d.company.siret for d in drafts] == ["11111111100011"]  # 1 seul établissement par SIREN
    assert len(llm.calls) == 1


def test_draft_all_respects_limit(db, crypto):
    email_writer, _ = writer(db, crypto)
    targets = [company(f"{i:09d}00011") for i in range(1, 8)]
    drafts = email_writer.draft_all(PROFILE, targets, {}, CONTEXT, SIGNATURE, client_id="moi", limit=3)
    assert len(drafts) == 3


def test_drafting_twice_does_not_duplicate(db, crypto):
    email_writer, llm = writer(db, crypto)
    targets = [company("11111111100011")]
    email_writer.draft_all(PROFILE, targets, {}, CONTEXT, SIGNATURE, client_id="moi")
    second = email_writer.draft_all(PROFILE, targets, {}, CONTEXT, SIGNATURE, client_id="moi")
    assert second == []  # déjà un brouillon pour cette entreprise
    assert len(llm.calls) == 1


def test_budget_exceeded_stops_drafting(db, crypto):
    email_writer, _ = writer(db, crypto, FakeLLM(error=BudgetExceededError("budget")))
    with pytest.raises(BudgetExceededError):
        email_writer.draft_all(PROFILE, [company("11111111100011")], {}, CONTEXT, SIGNATURE, client_id="moi")
    assert db.list_drafts(crypto, "moi") == []


# --------------------------------------------------------------------------
# Contrôles du texte
# --------------------------------------------------------------------------


def test_check_email_ok():
    assert check_email(EmailContent(subject="Candidature", body=BODY)) == []


def test_check_email_too_long():
    long_body = "Madame, Monsieur, " + "mot " * 200
    assert any("long" in w for w in check_email(EmailContent(subject="s", body=long_body)))


@pytest.mark.parametrize(
    ("body", "alert"),
    [
        (BODY + "\nJoignable au 06 12 34 56 78", "coordonnées"),
        (BODY + "\nÉcrivez à jean@exemple.fr", "coordonnées"),
        (BODY + "\n[CONFIDENTIEL]", "CONFIDENTIEL"),
    ],
)
def test_check_email_detects_leaks(body, alert):
    assert any(alert in w for w in check_email(EmailContent(subject="s", body=body)))


def test_email_length_limit_is_configurable(monkeypatch):
    from avenly.config import get_settings

    body = "Madame, Monsieur, " + "mot " * 140
    monkeypatch.setenv("EMAIL_MAX_WORDS", "150")
    get_settings.cache_clear()
    assert check_email(EmailContent(subject="s", body=body)) == []
    monkeypatch.setenv("EMAIL_MAX_WORDS", "80")
    get_settings.cache_clear()
    assert any("long" in w for w in check_email(EmailContent(subject="s", body=body)))
