"""Génère les variantes du CV (une page A4) à partir d'un fichier d'informations unique.

    python tools/cv_builder.py                       # toutes les variantes
    python tools/cv_builder.py data/cv/profil.json data      # une seule

Les informations personnelles restent dans data/cv/ (ignoré par Git) : seul ce script est versionné.
Mise en page sur une colonne, sans tableau ni image : lisible par les logiciels de tri de
candidatures (ATS) qui analysent les CV avant un humain.
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = ROOT / "data" / "cv" / "profil.json"
PAGE = pymupdf.paper_rect("a4")
MARGIN = 36  # points (0,5 pouce)
ACCENT = "#1f4e79"

CSS = f"""
* {{ font-family: sans-serif; }}
body {{ font-size: 9.2pt; color: #1a1a1a; line-height: 1.32; }}
h1 {{ font-size: 21pt; margin: 0; color: {ACCENT}; letter-spacing: 0.5px; }}
.titre {{ font-size: 12pt; font-weight: bold; margin: 2px 0 1px 0; }}
.recherche {{ font-size: 9.4pt; color: #444; margin: 0 0 3px 0; }}
.contact {{ font-size: 8.8pt; color: #333; margin: 0; }}
h2 {{ font-size: 10.5pt; color: {ACCENT}; text-transform: uppercase; letter-spacing: 1px;
      border-bottom: 1px solid {ACCENT}; margin: 9px 0 4px 0; padding-bottom: 1px; }}
p {{ margin: 0 0 3px 0; }}
.item {{ margin: 0 0 4px 0; }}
.item-titre {{ font-weight: bold; }}
.meta {{ color: #555; font-style: italic; }}
.dates {{ color: {ACCENT}; font-weight: bold; }}
ul {{ margin: 1px 0 3px 12px; padding: 0; }}
li {{ margin: 0 0 1px 0; }}
.label {{ font-weight: bold; }}
"""


def esc(text: str) -> str:
    return html.escape(text, quote=False)


def build_html(profile: dict, variant_key: str) -> str:
    variant = profile["variantes"][variant_key]
    ident = profile["identite"]
    parts = [
        f"<h1>{esc(ident['nom'])}</h1>",
        f"<p class='titre'>{esc(variant['titre'])}</p>",
        f"<p class='recherche'>{esc(profile['recherche'])}</p>",
        "<p class='contact'>"
        + " · ".join(
            esc(x)
            for x in (ident["ville"], ident["telephone"], ident["email"], ident["github"], ident["permis"])
            if x
        )
        + "</p>",
        "<h2>Profil</h2>",
        f"<p>{esc(variant['profil'])}</p>",
        "<h2>Projets</h2>",
    ]
    for key in variant["projets"]:
        project = profile["projets"][key]
        bullets = "".join(f"<li>{esc(point)}</li>" for point in project["points"])
        parts.append(
            f"<div class='item'><span class='item-titre'>{esc(project['titre'])}</span> "
            f"<span class='meta'>— {esc(project['contexte'])}</span><ul>{bullets}</ul></div>"
        )

    parts.append("<h2>Formation</h2>")
    for entry in profile["formation"]:
        parts.append(
            f"<div class='item'><span class='dates'>{esc(entry['dates'])}</span> &nbsp; "
            f"<span class='item-titre'>{esc(entry['titre'])}</span><br>"
            f"<span class='meta'>{esc(entry['lieu'])}</span></div>"
        )

    parts.append("<h2>Expériences</h2>")
    for entry in profile["experiences"]:
        dates = f"<span class='dates'>{esc(entry['dates'])}</span> &nbsp; " if entry["dates"] else ""
        parts.append(
            f"<div class='item'>{dates}<span class='item-titre'>{esc(entry['titre'])}</span> "
            f"<span class='meta'>— {esc(entry['lieu'])}</span><br>{esc(entry['detail'])}</div>"
        )

    parts.append("<h2>Compétences</h2>")
    for key in variant["competences"]:
        skill = profile["competences"][key]
        parts.append(f"<p><span class='label'>{esc(skill['titre'])} :</span> {esc(skill['items'])}</p>")
    parts.append(f"<p><span class='label'>Langues :</span> {esc(profile['langues'])}</p>")
    parts.append(f"<p><span class='label'>Qualités :</span> {esc(profile['qualites'])}</p>")
    parts.append(f"<p><span class='label'>Centres d'intérêt :</span> {esc(profile['interets'])}</p>")
    return "<body>" + "".join(parts) + "</body>"


def render(profile: dict, variant_key: str, output: Path) -> int:
    """Écrit le PDF et renvoie le nombre de pages (doit valoir 1)."""
    story = pymupdf.Story(html=build_html(profile, variant_key), user_css=CSS)
    writer = pymupdf.DocumentWriter(str(output))
    area = pymupdf.Rect(PAGE.x0 + MARGIN, PAGE.y0 + MARGIN, PAGE.x1 - MARGIN, PAGE.y1 - MARGIN)
    pages = 0
    more = True
    while more:
        device = writer.begin_page(PAGE)
        more, _ = story.place(area)
        story.draw(device)
        writer.end_page()
        pages += 1
    writer.close()
    with pymupdf.open(output) as doc:
        doc.set_metadata({"title": f"CV {profile['identite']['nom']}", "author": profile["identite"]["nom"]})
        doc.saveIncr()
    return pages


def main(argv: list[str]) -> int:
    profile_path = Path(argv[0]) if argv else DEFAULT_PROFILE
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    variants = argv[1:] or list(profile["variantes"])
    name = profile["identite"]["nom"].replace(" ", "_")
    for key in variants:
        output = profile_path.parent / f"CV_{name}_{key}.pdf"
        pages = render(profile, key, output)
        status = "OK" if pages == 1 else f"ATTENTION : {pages} pages, à raccourcir"
        print(f"{output.relative_to(ROOT)} : {status}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
