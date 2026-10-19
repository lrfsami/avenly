"""Tests de la ligne de commande, sans réseau ni LLM réels."""

from __future__ import annotations

import io

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
