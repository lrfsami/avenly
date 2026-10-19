from __future__ import annotations

import io

import pytest

from avenly.profiler import CVError, _clean, read_cv

NAMES = ["Jean Test"]  # nom présent dans tous les PDF de test ci-dessous

CV_TEXT = (
    "Jean-Marc DUPONT\n"
    "12 rue de la Paix, 75002 Paris\n"
    "06 12 34 56 78  jm.dupont@gmail.com\n"
    "Développeur Python · Stage 6 mois chez Capgemini\n"
    "Compétences : Python, SQL, Docker. Contact : Dupont Jean-Marc"
)


def test_read_cv_anonymizes_everything(make_pdf):
    cv = read_cv(make_pdf(CV_TEXT), candidate_names=["Jean-Marc Dupont"])

    for leak in ("Dupont", "DUPONT", "Jean", "Marc", "06 12", "gmail", "Paix", "75002"):
        assert leak not in cv.text
    assert "Capgemini" in cv.text
    assert "Python" in cv.text
    assert cv.page_count == 1
    assert len(cv.sha256) == 64


def test_repr_does_not_show_text(make_pdf):
    cv = read_cv(make_pdf("Jean Test - Texte du CV"), candidate_names=NAMES)
    assert "Texte du CV" not in repr(cv)


@pytest.mark.parametrize("wrap", [bytes, bytearray, io.BytesIO])
def test_accepts_bytes_and_streams(make_pdf, wrap):
    cv = read_cv(wrap(make_pdf("Jean Test Hello CV")), candidate_names=NAMES)
    assert cv.text == "[CONFIDENTIEL] [CONFIDENTIEL] Hello CV"


def test_accepts_path(make_pdf, tmp_path):
    path = tmp_path / "cv.pdf"
    path.write_bytes(make_pdf("Jean Test depuis un fichier"))
    assert read_cv(path, candidate_names=NAMES).text.endswith("depuis un fichier")


def test_same_file_same_hash(make_pdf):
    data = make_pdf("Jean Test")
    assert read_cv(data, candidate_names=NAMES).sha256 == read_cv(data, candidate_names=NAMES).sha256


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"MZ\x90\x00 exe renomme en .pdf", "pas un PDF"),
        (b"", "vide"),
        (b"%PDF-1.7\n garbage garbage", "corrompu"),
        (b"%PDF-" + b"0" * (6 * 1024 * 1024), "volumineux"),
    ],
)
def test_rejects_invalid_files(data, message):
    with pytest.raises(CVError, match=message):
        read_cv(data, candidate_names=NAMES)


def test_rejects_password_protected(make_pdf):
    with pytest.raises(CVError, match="mot de passe"):
        read_cv(make_pdf("secret", password="pw"), candidate_names=NAMES)


def test_rejects_too_many_pages(make_pdf):
    with pytest.raises(CVError, match="Trop de pages"):
        read_cv(make_pdf("x", pages=11), candidate_names=NAMES)


def test_rejects_scanned_cv(make_pdf):
    with pytest.raises(CVError, match="scanné"):
        read_cv(make_pdf(text=None), candidate_names=NAMES)


def test_rejects_missing_path():
    with pytest.raises(CVError, match="introuvable"):
        read_cv("/nexiste/pas/cv.pdf", candidate_names=NAMES)


def test_clean_removes_invisible_characters():
    dirty = "a​b‮c﻿­  d\x00\n\n\n\nﬁn"
    assert _clean(dirty) == "abc d\n\nfin"


# --------------------------------------------------------------------------
# Nom du candidat : obligatoire et vérifié
# --------------------------------------------------------------------------


@pytest.mark.parametrize("names", [[], [""], ["  "], ["A"]])
def test_name_is_required(make_pdf, names):
    with pytest.raises(CVError, match="obligatoire"):
        read_cv(make_pdf("Jean Test"), candidate_names=names)


def test_name_is_required_before_reading_the_file():
    with pytest.raises(CVError, match="obligatoire"):
        read_cv("/nexiste/pas/cv.pdf", candidate_names=[])


def test_name_not_found_stops_everything(make_pdf):
    """L'erreur « Prénom Nom » : le nom fourni ne correspond pas, rien n'est renvoyé."""
    with pytest.raises(CVError, match="n'apparaît pas"):
        read_cv(make_pdf("Jean Test, développeur Python"), candidate_names=["Prénom Nom"])


def test_name_as_plain_string_is_accepted(make_pdf):
    cv = read_cv(make_pdf("Jean Test"), candidate_names="Jean Test")  # type: ignore[arg-type]
    assert "Jean" not in cv.text


@pytest.mark.parametrize(
    ("in_cv", "typed"),
    [("Léa MARTIN", "Lea Martin"), ("Lea Martin", "Léa Martin"), ("LÉA MARTIN", "léa martin")],
)
def test_name_matching_ignores_case_and_accents(make_pdf, in_cv, typed):
    cv = read_cv(make_pdf(f"{in_cv} - Data analyst"), candidate_names=[typed])
    assert cv.text == "[CONFIDENTIEL] [CONFIDENTIEL] - Data analyst"


def test_name_does_not_mask_longer_words(make_pdf):
    cv = read_cv(make_pdf("Léa Martin, stage chez Martinez et Leandre"), candidate_names=["Léa Martin"])
    assert "Martinez" in cv.text and "Leandre" in cv.text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("É T U D E S", "ÉTUDES"),
        ("D ’ I N T É R Ê T", "D’INTÉRÊT"),
        ("C O M P É T E N C E S", "COMPÉTENCES"),
        ("A B test", "A B test"),  # moins de 4 lettres : on ne touche pas
        ("C'est à dire", "C'est à dire"),
    ],
)
def test_clean_rejoins_spaced_headings(raw, expected):
    assert _clean(raw) == expected
