from __future__ import annotations

import io

import pytest

from avenly.profiler import CVError, _clean, read_cv

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
    cv = read_cv(make_pdf("Texte du CV"))
    assert "Texte du CV" not in repr(cv)


@pytest.mark.parametrize("wrap", [bytes, bytearray, io.BytesIO])
def test_accepts_bytes_and_streams(make_pdf, wrap):
    assert read_cv(wrap(make_pdf("Hello CV"))).text == "Hello CV"


def test_accepts_path(make_pdf, tmp_path):
    path = tmp_path / "cv.pdf"
    path.write_bytes(make_pdf("Depuis un fichier"))
    assert read_cv(path).text == "Depuis un fichier"


def test_same_file_same_hash(make_pdf):
    data = make_pdf("Identique")
    assert read_cv(data).sha256 == read_cv(data).sha256


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
        read_cv(data)


def test_rejects_password_protected(make_pdf):
    with pytest.raises(CVError, match="mot de passe"):
        read_cv(make_pdf("secret", password="pw"))


def test_rejects_too_many_pages(make_pdf):
    with pytest.raises(CVError, match="Trop de pages"):
        read_cv(make_pdf("x", pages=11))


def test_rejects_scanned_cv(make_pdf):
    with pytest.raises(CVError, match="scanné"):
        read_cv(make_pdf(text=None))


def test_rejects_missing_path():
    with pytest.raises(CVError, match="introuvable"):
        read_cv("/nexiste/pas/cv.pdf")


def test_clean_removes_invisible_characters():
    dirty = "a​b‮c﻿­  d\x00\n\n\n\nﬁn"
    assert _clean(dirty) == "abc d\n\nfin"
