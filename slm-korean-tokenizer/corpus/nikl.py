#!/usr/bin/env python3
"""Shared streaming reader for the NIKL Newspaper Corpus 2025.

The corpus is 3.4 GB across 20 JSON files, two of them approaching 1 GB, so it
is only ever parsed incrementally with ijson — never loaded into memory.

Two schema facts were verified against the data rather than taken from the
specification, and both matter for evaluation:

* ``document[].metadata.title`` holds exactly one distinct value per file (for
  example "굿모닝충청 2024년 기사"). It is a collection label, not a headline.
  The article headline is ``paragraph[0]``: median 30 characters, and only 6% of
  them end like a sentence, against a median 73 for ``paragraph[1]``.
* Inside ``paragraph[].form`` the only presentational markup is ``<p>``/``</p>``.
  Every other angle-bracketed run is Korean content — ``<오마이뉴스>``,
  ``<키움證>``, work titles — so nothing else is stripped.
"""
from __future__ import annotations

import glob
import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

DEFAULT_ROOT = "data/NIKL_NEWSPAPER_2025_v1.1/NIKL_NEWSPAPER_2025_v1.1"

# Only the presentational paragraph wrapper is removed.
P_WRAPPER = re.compile(r"^\s*<p>\s*|\s*</p>\s*$")

# The nine topic labels the corpus actually uses. The specification calls the
# science topic "정보통신/과학"; the data spells it "IT/과학". The observed value
# is authoritative and is what gets reported, with the alias recorded here.
TOPIC_ALIASES = {
    "정보통신/과학": "IT/과학",
}
KNOWN_TOPICS = (
    "사회", "정치", "경제", "생활", "IT/과학", "스포츠", "연예", "문화", "미용/건강",
)

# Entities safe to normalize: their meaning is unambiguous. Everything else is
# left exactly as written so the real newspaper distribution survives.
HTML_ENTITIES = {
    "&amp;": "&",
    "&lt;": "<",
    "&gt;": ">",
    "&quot;": '"',
    "&apos;": "'",
    "&nbsp;": " ",
}
ENTITY_RE = re.compile("|".join(re.escape(k) for k in HTML_ENTITIES))


@dataclass(frozen=True)
class Article:
    article_id: str
    topic: str
    original_topic: str
    publisher: str
    date: str
    title: str
    body: str

    @property
    def title_and_body(self) -> str:
        return f"{self.title}\n{self.body}" if self.title else self.body


def normalize_topic(value: str | None) -> str:
    raw = (value or "").strip()
    return TOPIC_ALIASES.get(raw, raw) or "(unlabelled)"


def clean_form(form: str | None) -> str:
    """Strip the <p> wrapper, resolve unambiguous entities, NFC-normalize."""
    if not form:
        return ""
    text = P_WRAPPER.sub("", form)
    text = P_WRAPPER.sub("", text)
    if "&" in text:
        text = ENTITY_RE.sub(lambda m: HTML_ENTITIES[m.group(0)], text)
    return unicodedata.normalize("NFC", text).strip()


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def short_digest(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


def corpus_files(root: str | Path = DEFAULT_ROOT) -> list[Path]:
    return [Path(p) for p in sorted(glob.glob(str(Path(root) / "*.json")))]


def iter_articles(paths: list[Path]) -> Iterator[Article]:
    """Yield one Article per document, streaming, never holding a whole file."""
    import ijson

    for path in paths:
        with path.open("rb") as handle:
            for document in ijson.items(handle, "document.item"):
                metadata = document.get("metadata") or {}
                forms = [clean_form(p.get("form")) for p in (document.get("paragraph") or [])]
                forms = [f for f in forms if f]
                if not forms:
                    continue
                title = forms[0]
                body = "\n".join(forms[1:])
                yield Article(
                    article_id=str(document.get("id") or ""),
                    topic=normalize_topic(metadata.get("topic")),
                    original_topic=str(metadata.get("original_topic") or ""),
                    publisher=str(metadata.get("publisher") or ""),
                    date=str(metadata.get("date") or ""),
                    title=title,
                    body=body,
                )


def character_profile(text: str) -> dict[str, int]:
    hangul = ascii_count = english = digits = 0
    for char in text:
        if "가" <= char <= "힣":
            hangul += 1
        if char.isascii():
            ascii_count += 1
            if char.isalpha():
                english += 1
        if char.isdigit():
            digits += 1
    return {"hangul": hangul, "ascii": ascii_count, "english": english, "digits": digits}
