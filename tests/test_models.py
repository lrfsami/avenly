from __future__ import annotations

import pytest
from pydantic import ValidationError

from avenly.models import Company


def make_company(**overrides) -> Company:
    fields = {"client_id": "client-001", "siret": "12345678901234", "name": "ACME"}
    return Company(**(fields | overrides))


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("www.test.fr", "https://www.test.fr"),
        ("https://ok.fr/carrieres", "https://ok.fr/carrieres"),
        ("javascript:alert(1)", None),
        ("file:///etc/passwd", None),
        ("data:text/html,x", None),
        ("ftp://a.fr", None),
        ("http://localhost:8080", None),
        ("https://127.0.0.1", None),
        ("", None),
    ],
)
def test_only_safe_urls_are_kept(url, expected):
    assert make_company(website=url).website == expected


def test_invalid_siret_is_rejected():
    with pytest.raises(ValidationError):
        make_company(siret="123")


@pytest.mark.parametrize(("raw", "expected"), [("6201Z", "62.01Z"), ("62.01Z", "62.01Z"), ("nimp", None)])
def test_naf_code_is_normalized(raw, expected):
    assert make_company(naf_code=raw).naf_code == expected


def test_invisible_characters_are_removed_from_name():
    assert make_company(name="Marque​  Test‮").name == "Marque Test"


def test_merge_fills_missing_fields():
    a = make_company(sources={"a"}, address="1 rue X")
    b = make_company(sources={"b"}, website="https://acme.fr", has_job_offer=True)
    merged = a.merge(b)
    assert merged.address == "1 rue X"
    assert merged.website == "https://acme.fr"
    assert merged.sources == {"a", "b"}
    assert merged.has_job_offer is True


def test_merge_refuses_different_siret():
    with pytest.raises(ValueError):
        make_company().merge(make_company(siret="98765432100012"))
