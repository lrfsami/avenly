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
import secrets
import sys
import time
from collections.abc import Callable, Sequence
from typing import TextIO

from sqlalchemy import func, select

from avenly.config import ConfigError, get_settings
from avenly.enricher import Enricher
from avenly.llm import PRICES_PER_MTOK, BudgetExceededError, ClaudeClient, LLMError
from avenly.models import ApplicationContext, Signature
from avenly.profiler import CVError, read_cv
from avenly.security import CryptoManager, SecurityError
from avenly.sender import (
    EmailSender,
    OutgoingEmail,
    SendError,
    build_message,
    check_smtp_settings,
    load_attachment,
    recipient_problem,
)
from avenly.sourcing import DOMAINS, SourcingError, find_companies
from avenly.storage import ApplicationStatus, Database, LLMUsageRecord, start_of_month
from avenly.webfetch import SafeFetcher
from avenly.writer import EmailWriter, has_channel, one_per_company

# Estimations prudentes (mesurées sur des appels réels), pour prévenir avant de dépenser.
ESTIMATED_COST_PROFILE = 0.002
ESTIMATED_COST_PER_COMPANY = 0.012  # résumé du site (Haiku) + email (Sonnet)
# On examine jusqu'à 5 fois plus d'entreprises que de brouillons voulus (certaines n'ont aucun canal).
MAX_ATTEMPTS_FACTOR = 5
# Nombre d'entreprises enrichies en même temps.
ENRICH_BATCH = 4

USER_ERRORS = (CVError, LLMError, SourcingError, ConfigError, SecurityError, SendError, ValueError)

# Remplaçable dans les tests (on n'attend pas réellement plusieurs minutes entre deux envois).
SLEEP = time.sleep
STATUS_LABELS = {
    ApplicationStatus.DRAFT: "à relire",
    ApplicationStatus.APPROVED: "validé",
    ApplicationStatus.SENT: "envoyé",
    ApplicationStatus.REPLIED: "réponse reçue",
    ApplicationStatus.REFUSED: "refusé",
}


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
        # Enrichissement par lots, en parallèle ; rédaction une par une, jusqu'au maximum demandé.
        for start in range(0, len(candidates), ENRICH_BATCH):
            if drafts >= args.max:
                break
            batch = candidates[start : start + ENRICH_BATCH]
            enrichments = enricher.enrich_many(batch, workers=ENRICH_BATCH)
            for company in batch:
                if drafts >= args.max:
                    break
                enrichment = enrichments[company.siren]
                if not has_channel(company, enrichment):
                    skipped += 1
                    out.write(
                        f"  ·  {company.name} : aucun email ni page carrières, ignorée (pas de rédaction)\n"
                    )
                    continue
                draft = writer.draft_one(
                    profile, company, enrichment, context, signature, client_id=args.client
                )
                drafts += 1
                channel = draft.recipient_email or draft.careers_url
                out.write(f"  [{drafts}/{args.max}] {company.name}  →  {channel}\n")
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
    status = ApplicationStatus.DRAFT if args.a_relire else None
    drafts = Database().list_drafts(CryptoManager(), args.client, status=status)
    if not drafts:
        out.write("Aucun brouillon à afficher. Lance une campagne : avenly campagne …\n")
        return 0
    for draft in drafts:
        channel = draft["recipient_email"] or draft["careers_url"] or "canal à trouver"
        status = STATUS_LABELS.get(draft["status"], draft["status"])
        out.write("=" * 72 + "\n")
        out.write(f"#{draft['id']} [{status}] {draft['company']}  →  {channel}\n")
        out.write(f"Objet : {draft['subject']}\n\n{draft['body']}\n\n")
    out.write(
        f"{len(drafts)} brouillon(s). Pour valider : avenly valider <numéros> (ex. : avenly valider 3 5)\n"
    )
    return 0


def cmd_valider(args: argparse.Namespace, *, out: TextIO, ask: Callable[[str], str]) -> int:
    approved = Database().approve_drafts(args.client, args.numeros)
    ignored = sorted(set(args.numeros) - set(approved))
    if approved:
        out.write(f"✅ Brouillon(s) validé(s) : {', '.join(f'#{n}' for n in approved)}\n")
    if ignored:
        out.write(
            f"Ignoré(s) : {', '.join(f'#{n}' for n in ignored)} (numéro inconnu, ou déjà validé ou envoyé).\n"
        )
    if approved:
        out.write("Envoie-les avec : avenly envoyer --cv <ton CV en PDF>\n")
    return 0


def cmd_supprimer(args: argparse.Namespace, *, out: TextIO, ask: Callable[[str], str]) -> int:
    deleted = Database().delete_drafts(args.client, args.numeros)
    ignored = sorted(set(args.numeros) - set(deleted))
    if deleted:
        out.write(f"🗑️  Brouillon(s) supprimé(s) : {', '.join(f'#{n}' for n in deleted)}\n")
    if ignored:
        out.write(f"Ignoré(s) : {', '.join(f'#{n}' for n in ignored)} (numéro inconnu ou déjà envoyé).\n")
    return 0


def cmd_envoyer(args: argparse.Namespace, *, out: TextIO, ask: Callable[[str], str]) -> int:
    settings = get_settings()
    check_smtp_settings(settings)  # d'abord : inutile de préparer un envoi impossible
    attachment = load_attachment(args.cv)
    db = Database()
    approved = db.list_drafts(CryptoManager(), args.client, status=ApplicationStatus.APPROVED)

    blocked = [
        (d, problem)
        for d in approved
        if d["recipient_email"] and (problem := recipient_problem(d["recipient_email"]))
    ]
    if blocked:
        out.write(
            "⛔ Bloqué(s), adresse refusée par les règles actuelles (supprime ou remplace ces brouillons) :\n"
        )
        for draft, problem in blocked:
            out.write(f"  #{draft['id']} {draft['company']}  →  {draft['recipient_email']} ({problem})\n")
        out.write("\n")
    blocked_ids = {d["id"] for d, _ in blocked}
    by_email = [d for d in approved if d["recipient_email"] and d["id"] not in blocked_ids]
    manual = [d for d in approved if not d["recipient_email"]]

    if manual:
        out.write("À envoyer à la main (pas d'email, seulement une page de candidature) :\n")
        for draft in manual:
            out.write(
                f"  #{draft['id']} {draft['company']}  →  {draft['careers_url'] or 'canal à trouver'}\n"
            )
        out.write("  (copie le texte affiché par « avenly brouillons »)\n\n")
    if not by_email:
        out.write(
            "Aucun brouillon validé à envoyer par email. Valide-les d'abord : avenly valider <numéros>\n"
        )
        return 0

    remaining = settings.send_daily_limit - db.sent_today(args.client)
    if not args.essai and remaining <= 0:
        out.write(f"Limite quotidienne atteinte ({settings.send_daily_limit} emails). Reprends demain.\n")
        return 0
    batch = by_email[:1] if args.essai else by_email[: min(args.max, remaining)]

    if args.essai:
        out.write(
            f"MODE ESSAI : le 1er email validé part vers TOI ({settings.smtp_user}), pas vers l'entreprise.\n"
        )
    else:
        out.write(f"{len(batch)} email(s) prêt(s) à partir :\n")
        for number, draft in enumerate(batch, start=1):
            out.write(f"  {number}. {draft['company']:<34} → {draft['recipient_email']}\n")
        out.write(
            f"Pièce jointe : {attachment[0]} · Espacement : {settings.send_min_delay_seconds // 60} à "
            f"{settings.send_max_delay_seconds // 60} min entre deux emails · "
            f"Limite : {remaining} restant(s) aujourd'hui\n"
        )
    if ask(
        f"\nEnvoyer {'cet email de test' if args.essai else f'ces {len(batch)} email(s)'} ? [o/N] "
    ).strip().lower() not in ("o", "oui"):
        out.write("Envoi annulé : rien n'est parti.\n")
        return 0

    sent = 0
    with EmailSender(settings) as sender:
        for index, draft in enumerate(batch, start=1):
            email = OutgoingEmail(
                to=sender.sender_email if args.essai else draft["recipient_email"],
                subject=("[ESSAI Avenly] " if args.essai else "") + draft["subject"],
                body=draft["body"],
            )
            message = build_message(
                email, sender_name=args.nom or "", sender_email=sender.sender_email, attachment=attachment
            )
            if args.un_par_un and not args.essai:
                out.write(f"\n──── Email {index}/{len(batch)} " + "─" * 40 + "\n")
                out.write(
                    f"À : {email.to}\nObjet : {email.subject}\n"
                    f"Pièce jointe : {attachment[0]}\n\n{email.body}\n"
                )
                answer = (
                    ask("\nEnvoyer cet email ? [o = oui / n = passer / q = tout arrêter] ").strip().lower()
                )
                if answer == "q":
                    out.write("Arrêt demandé : les emails restants ne sont pas envoyés.\n")
                    break
                if answer not in ("o", "oui"):
                    out.write("Passé.\n")
                    continue
            sender.send(message)
            sent += 1
            if not args.essai:
                db.record_application(args.client, draft["siret"], ApplicationStatus.SENT)
            out.write(f"  ✅ Envoyé : {draft['company']} → {email.to}\n")
            if not args.essai and index < len(batch):
                delay = secrets.SystemRandom().uniform(
                    settings.send_min_delay_seconds, settings.send_max_delay_seconds
                )
                out.write(f"     Prochain envoi dans {delay / 60:.1f} min (Ctrl + C pour arrêter)…\n")
                SLEEP(delay)

    out.write(f"\n{sent} email(s) envoyé(s).\n")
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
    brouillons.add_argument(
        "--a-relire", action="store_true", help="seulement les brouillons pas encore validés"
    )
    brouillons.set_defaults(handler=cmd_brouillons)

    supprimer = commands.add_parser("supprimer", help="supprimer des brouillons (pas encore envoyés)")
    supprimer.add_argument("numeros", nargs="+", type=int, help="numéros des brouillons")
    supprimer.set_defaults(handler=cmd_supprimer)

    couts = commands.add_parser("couts", help="voir les dépenses LLM du mois")
    couts.set_defaults(handler=cmd_couts)

    valider = commands.add_parser("valider", help="valider des brouillons pour l'envoi")
    valider.add_argument(
        "numeros", nargs="+", type=int, help="numéros des brouillons (voir avenly brouillons)"
    )
    valider.set_defaults(handler=cmd_valider)

    envoyer = commands.add_parser("envoyer", help="envoyer les brouillons validés (avec confirmation)")
    envoyer.add_argument("--cv", required=True, help="CV en PDF à joindre")
    envoyer.add_argument("--nom", help="ton nom, affiché comme expéditeur")
    envoyer.add_argument("--max", type=int, default=5, help="nombre maximum d'emails envoyés (défaut : 5)")
    envoyer.add_argument(
        "--essai", action="store_true", help="envoyer le 1er email validé à toi-même seulement"
    )
    envoyer.add_argument(
        "--un-par-un", action="store_true", help="afficher et confirmer chaque email avant envoi"
    )
    envoyer.set_defaults(handler=cmd_envoyer)

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
