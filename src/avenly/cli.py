"""Interface en ligne de commande d'Avenly.

    avenly campagne --cv data/cv.pdf --nom "Prénom Nom" --domaine data --ville Annecy --debut "septembre 2027"
    avenly brouillons            relire les brouillons rédigés
    avenly couts                 dépenses LLM du mois
    avenly opposition 123456789  ne plus jamais contacter une entreprise (SIREN)

Ce module ne contient aucune logique métier : il enchaîne les briques du moteur
(profiler, sourcing, enricher, writer), qui serviront telles quelles à l'API web.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from typing import TextIO

from sqlalchemy import func, select

from avenly.config import ConfigError, get_settings
from avenly.enricher import Enricher
from avenly.llm import PRICES_PER_MTOK, BudgetExceededError, ClaudeClient, LLMError
from avenly.models import ApplicationContext, Signature
from avenly.profiler import CVError, read_cv
from avenly.security import CryptoManager, SecurityError
from avenly.sourcing import DOMAINS, SourcingError, find_companies
from avenly.storage import Database, LLMUsageRecord, start_of_month
from avenly.webfetch import SafeFetcher
from avenly.writer import EmailWriter, has_channel, one_per_company

# Estimations prudentes (mesurées sur des appels réels), pour prévenir avant de dépenser.
ESTIMATED_COST_PROFILE = 0.002
ESTIMATED_COST_PER_COMPANY = 0.012  # résumé du site (Haiku) + email (Sonnet)
# On examine jusqu'à 5 fois plus d'entreprises que de brouillons voulus (certaines n'ont aucun canal).
MAX_ATTEMPTS_FACTOR = 5

USER_ERRORS = (CVError, LLMError, SourcingError, ConfigError, SecurityError, ValueError)


# --------------------------------------------------------------------------
# Commandes
# --------------------------------------------------------------------------


def cmd_campagne(args: argparse.Namespace, *, out: TextIO, ask: Callable[[str], str]) -> int:
    settings = get_settings()
    signature = Signature(
        full_name=args.nom,
        email=settings.signature_email
        or ask("Email de contact (affiché dans la signature) : ").strip()
        or None,
        phone=settings.signature_phone
        or ask("Téléphone (facultatif, Entrée pour passer) : ").strip()
        or None,
        linkedin_url=settings.signature_linkedin,
    )
    context = ApplicationContext(
        current_status=args.situation, start=args.debut, rhythm=args.rythme, duration=args.duree
    )

    db = Database()
    crypto = CryptoManager()  # échoue tout de suite si FERNET_KEY manque (brouillons chiffrés)
    llm = ClaudeClient(db)

    out.write("1/4 Lecture et anonymisation du CV…\n")
    cv = read_cv(args.cv, candidate_names=[args.nom])

    out.write(f"2/4 Recherche d'entreprises « {args.domaine} » autour de {args.ville} ({args.rayon} km)…\n")
    found = find_companies(
        args.domaine, args.ville, client_id=args.client, radius_km=args.rayon, max_results=100, database=db
    )
    # Une seule candidature par entreprise (et par groupe), jamais vers une entreprise
    # en opposition ou déjà contactée. On garde de la marge : certaines n'auront aucun canal.
    candidates = one_per_company(db.contactable(found, args.client))[: args.max * MAX_ATTEMPTS_FACTOR]
    if not candidates:
        out.write("Aucune nouvelle entreprise à contacter dans cette zone.\n")
        return 0

    estimate = ESTIMATED_COST_PROFILE + ESTIMATED_COST_PER_COMPANY * min(args.max, len(candidates))
    spent = db.llm_spend_this_month(args.client)
    out.write(
        f"\n{len(candidates)} entreprise(s) candidates pour {args.max} brouillon(s) maximum. "
        f"Coût estimé : environ {estimate:.2f} $ "
        f"(déjà dépensé ce mois : {spent:.4f} $ / budget {settings.llm_budget_per_client_usd:.2f} $).\n"
    )
    if not args.oui and ask("Continuer ? [o/N] ").strip().lower() not in ("o", "oui", "y", "yes"):
        out.write("Campagne annulée : rien n'a été dépensé.\n")
        return 0

    out.write("\n3/4 Analyse du profil…\n")
    profile = llm.extract_profile(cv, client_id=args.client)

    out.write("4/4 Enrichissement des entreprises et rédaction des brouillons…\n")
    fetcher = SafeFetcher()
    enricher = Enricher(db, fetcher, llm=llm)
    writer = EmailWriter(db, llm, crypto)
    drafts = skipped = 0
    try:
        for company in candidates:
            if drafts >= args.max:
                break
            enrichment = enricher.enrich(company)
            if not has_channel(company, enrichment):
                skipped += 1
                out.write(
                    f"  ·  {company.name} : aucun email ni page carrières, ignorée (pas de rédaction)\n"
                )
                continue
            draft = writer.draft_one(profile, company, enrichment, context, signature, client_id=args.client)
            drafts += 1
            out.write(
                f"  [{drafts}/{args.max}] {company.name}  →  {draft.recipient_email or draft.careers_url}\n"
            )
            for warning in draft.warnings:
                out.write(f"        ⚠️  {warning}\n")
    finally:
        fetcher.close()

    out.write(
        f"\n✅ {drafts} brouillon(s) rédigé(s). "
        f"Coût réel du mois : {db.llm_spend_this_month(args.client):.4f} $.\n"
        "Relis-les avec : avenly brouillons\n"
    )
    return 0


def cmd_brouillons(args: argparse.Namespace, *, out: TextIO, ask: Callable[[str], str]) -> int:
    drafts = Database().list_drafts(CryptoManager(), args.client)
    if not drafts:
        out.write("Aucun brouillon pour l'instant. Lance d'abord : avenly campagne …\n")
        return 0
    for number, draft in enumerate(drafts, start=1):
        channel = draft["recipient_email"] or draft["careers_url"] or "canal à trouver"
        out.write("=" * 72 + "\n")
        out.write(f"#{number} {draft['company']}  →  {channel}\n")
        out.write(f"Objet : {draft['subject']}\n\n{draft['body']}\n\n")
    out.write(f"{len(drafts)} brouillon(s).\n")
    return 0


def cmd_couts(args: argparse.Namespace, *, out: TextIO, ask: Callable[[str], str]) -> int:
    settings = get_settings()
    db = Database()
    with db.session() as session:
        rows = session.execute(
            select(LLMUsageRecord.task, LLMUsageRecord.model, func.count(), func.sum(LLMUsageRecord.cost_usd))
            .where(LLMUsageRecord.created_at >= start_of_month())
            .group_by(LLMUsageRecord.task, LLMUsageRecord.model)
            .order_by(LLMUsageRecord.task)
        ).all()
    out.write("Dépenses LLM du mois\n")
    for task, model, count, cost in rows:
        out.write(f"  {task:20} {model:20} {count:5} appel(s)  {cost:.4f} $\n")
    total = db.llm_spend_this_month()
    out.write(f"  {'TOTAL':47}           {total:.4f} $ / budget {settings.llm_monthly_budget_usd:.2f} $\n")
    out.write(f"\nTon compte « {args.client} » : {db.llm_spend_this_month(args.client):.4f} $\n")
    return 0


def cmd_opposition(args: argparse.Namespace, *, out: TextIO, ask: Callable[[str], str]) -> int:
    Database().add_opt_out(args.siren)
    out.write(
        f"Entreprise {args.siren} ajoutée à la liste d'opposition : elle ne sera plus jamais contactée.\n"
    )
    return 0


# --------------------------------------------------------------------------
# Analyse des arguments
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="avenly", description="Avenly : le marché caché de l'alternance, automatisé."
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="afficher le détail des opérations")
    parser.add_argument("--client", default="moi", help="identifiant du candidat (défaut : moi)")
    commands = parser.add_subparsers(dest="command", required=True, metavar="commande")

    campagne = commands.add_parser("campagne", help="trouver des entreprises et rédiger les brouillons")
    campagne.add_argument("--cv", required=True, help="chemin du CV en PDF")
    campagne.add_argument("--nom", required=True, help="prénom et nom du candidat (masqués avant l'IA)")
    campagne.add_argument("--domaine", required=True, choices=sorted(DOMAINS), help="domaine métier visé")
    campagne.add_argument("--ville", required=True, help="ville autour de laquelle chercher")
    campagne.add_argument("--rayon", type=float, default=20, help="rayon de recherche en km (défaut : 20)")
    campagne.add_argument("--debut", required=True, help="début souhaité, ex. « septembre 2027 »")
    campagne.add_argument("--situation", help="situation actuelle, prioritaire sur le CV")
    campagne.add_argument("--rythme", help="rythme de l'alternance, ex. « 3 semaines / 1 semaine »")
    campagne.add_argument("--duree", help="durée du contrat, ex. « 2 ans »")
    campagne.add_argument("--max", type=int, default=10, help="nombre maximum de brouillons (défaut : 10)")
    campagne.add_argument(
        "--oui", action="store_true", help="ne pas demander de confirmation avant de dépenser"
    )
    campagne.set_defaults(handler=cmd_campagne)

    brouillons = commands.add_parser("brouillons", help="relire les brouillons rédigés")
    brouillons.set_defaults(handler=cmd_brouillons)

    couts = commands.add_parser("couts", help="voir les dépenses LLM du mois")
    couts.set_defaults(handler=cmd_couts)

    opposition = commands.add_parser("opposition", help="ne plus jamais contacter une entreprise")
    opposition.add_argument("siren", help="SIREN de l'entreprise (9 chiffres)")
    opposition.set_defaults(handler=cmd_opposition)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    out: TextIO | None = None,
    err: TextIO | None = None,
    ask: Callable[[str], str] = input,
) -> int:
    out = out or sys.stdout
    err = err or sys.stderr
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="   [%(name)s] %(message)s"
    )
    try:
        return args.handler(args, out=out, ask=ask)
    except BudgetExceededError as exc:
        err.write(f"⛔ {exc}\nAucun appel supplémentaire n'a été envoyé. Ajuste les budgets dans le .env.\n")
        return 2
    except USER_ERRORS as exc:
        err.write(f"❌ {exc}\n")
        return 1
    except KeyboardInterrupt:
        err.write("\nInterrompu. Les brouillons déjà rédigés sont enregistrés.\n")
        return 130


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["PRICES_PER_MTOK", "build_parser", "main"]
