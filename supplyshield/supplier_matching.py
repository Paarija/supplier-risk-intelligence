"""Supplier and location entity matching."""

from __future__ import annotations

import re
from difflib import SequenceMatcher


def normalize_name(value: str) -> str:
    value = re.sub(r"[^a-z0-9 ]", " ", (value or "").lower())
    stopwords = {"limited", "ltd", "inc", "corp", "corporation", "company", "co"}
    return " ".join(token for token in value.split() if token not in stopwords)


def name_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, normalize_name(left), normalize_name(right)).ratio()


def article_matches_supplier(
    text: str,
    supplier_name: str,
    location: str,
    aliases: str = "",
    threshold: float = 0.72,
) -> bool:
    normalized_text = normalize_name(text)
    names = [supplier_name, *[alias.strip() for alias in aliases.split("|") if alias.strip()]]
    if any(normalize_name(name) in normalized_text for name in names if normalize_name(name)):
        return True
    location_tokens = set(normalize_name(location).split())
    if location_tokens and location_tokens.issubset(set(normalized_text.split())):
        return True
    title_window = " ".join(normalized_text.split()[:12])
    return any(name_similarity(name, title_window) >= threshold for name in names)
