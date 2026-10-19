"""Rédaction des candidatures : profil + entreprise -> brouillon d'email chiffré, à relire.

Rien n'est envoyé ici : chaque brouillon est enregistré (chiffré) pour être relu par le
candidat avant tout envoi.

Garde-fous :
- une seule candidature par entreprise (SIREN), jamais vers une entreprise en opposition,
  déjà contactée par ce client ou saturée (voir `Database.contactable`) ;
- la signature (nom, téléphone, email) est ajoutée localement : elle ne passe jamais par le LLM ;
- le texte produit est contrôlé (longueur, coordonnées inventées, marqueur [CONFIDENTIEL])
  et les anomalies sont signalées au candidat au lieu d'être corrigées en silence.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from avenly.llm import EMAIL_MAX_WORDS, LLMClient
from avenly.models import (
    ApplicationContext,
    CandidateProfile,
    Company,
    EmailContent,
    Enrichment,
    Signature,
)
from avenly.security import REDACTED, CryptoManager, anonymize
from avenly.storage import ApplicationStatus, Database

logger = logging.getLogger(__name__)

# Marge de tolérance avant de signaler un email trop long (salutation et formule comprises).
WORD_TOLERANCE = 20


@dataclass(frozen=True, slots=True)
class Draft:
    company: Company
    subject: str
    body: str  # corps + signature, prêt à relire
    recipient_email: str | None
    careers_url: str | None
    warnings: list[str] = field(default_factory=list)


def check_email(content: EmailContent) -> list[str]:
    """Contrôles automatiques du texte produit par le LLM."""
    warnings: list[str] = []
    words = len(re.findall(r"\w+(?:['’]\w+)?", content.body))
    if words > EMAIL_MAX_WORDS + WORD_TOLERANCE:
        warnings.append(f"Email long ({words} mots, objectif {EMAIL_MAX_WORDS}).")
    text = f"{content.subject}\n{content.body}"
    if REDACTED in text:
        warnings.append("Le marqueur [CONFIDENTIEL] apparaît dans l'email.")
    elif anonymize(text) != text:
        warnings.append("L'email contient des coordonnées (email, téléphone, adresse) : à vérifier.")
    return warnings


def one_per_company(companies: Iterable[Company]) -> list[Company]:
    """Garde un seul établissement par entreprise (SIREN) : on ne postule pas 3 fois chez Atos."""
    seen: set[str] = set()
    result: list[Company] = []
    for company in companies:
        if company.siren not in seen:
            seen.add(company.siren)
            result.append(company)
    return result


class EmailWriter:
    def __init__(self, database: Database, llm: LLMClient, crypto: CryptoManager) -> None:
        self._db = database
        self._llm = llm
        self._crypto = crypto

    def draft_all(
        self,
        profile: CandidateProfile,
        companies: Iterable[Company],
        enrichments: Mapping[str, Enrichment],
        context: ApplicationContext,
        signature: Signature,
        *,
        client_id: str,
        limit: int = 20,
    ) -> list[Draft]:
        """Rédige au plus `limit` brouillons. S'arrête net si le budget LLM est atteint."""
        targets = one_per_company(self._db.contactable(companies, client_id))[:limit]
        return [
            self.draft_one(
                profile,
                company,
                enrichments.get(company.siren),
                context,
                signature,
                client_id=client_id,
            )
            for company in targets
        ]

    def draft_one(
        self,
        profile: CandidateProfile,
        company: Company,
        enrichment: Enrichment | None,
        context: ApplicationContext,
        signature: Signature,
        *,
        client_id: str,
    ) -> Draft:
        insight = enrichment.insight if enrichment else None
        content = self._llm.write_email(profile, company.name, insight, context, client_id=client_id)

        warnings = check_email(content)
        if insight is None:
            warnings.append("Peu d'informations sur l'entreprise : email plus générique.")

        recipient = enrichment.generic_emails[0] if enrichment and enrichment.generic_emails else None
        careers_url = (enrichment.careers_url if enrichment else None) or company.apply_url
        if recipient is None and careers_url is None:
            warnings.append("Aucun email ni page carrières trouvés : canal de candidature à chercher.")

        # Signature ajoutée ici, localement : le LLM ne voit jamais les coordonnées du candidat.
        body = f"{content.body}\n\n{signature.render()}"

        self._db.upsert_companies([company])
        self._db.save_draft(
            self._crypto,
            client_id=client_id,
            siret=company.siret,
            subject=content.subject,
            body=body,
            recipient_email=recipient,
            careers_url=careers_url,
            model=self._llm.email_model,
        )
        self._db.record_application(client_id, company.siret, ApplicationStatus.DRAFT)
        logger.info("Brouillon rédigé pour %s (%d alerte(s)).", company.name, len(warnings))
        return Draft(
            company=company,
            subject=content.subject,
            body=body,
            recipient_email=recipient,
            careers_url=careers_url,
            warnings=warnings,
        )
