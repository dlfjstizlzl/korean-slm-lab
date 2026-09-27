#!/usr/bin/env python3
"""Deduplication and quality filtering primitives for the Stage-2 corpus.

Three exact-dedup levels are applied by the builder:
  A  source-internal exact duplicates
  B  cross-source exact duplicates
  C  train <-> evaluation exact overlap (removed from the train side only)

Plus an optional supplemental near-duplicate pass using SimHash over character
5-gram shingles, which is O(n) and safe at 100M+ characters.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter

CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
EXCESS_WHITESPACE = re.compile(r"[ \t]{3,}")
EXCESS_NEWLINES = re.compile(r"\n{4,}")


def normalize(text: str) -> str:
    """NFC, strip control characters, collapse runaway whitespace. Nothing else.

    English, digits, URLs, brackets, punctuation and loanwords are left alone:
    they are part of the real language distribution.
    """
    text = unicodedata.normalize("NFC", text)
    text = CONTROL_CHARS.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = EXCESS_WHITESPACE.sub("  ", text)
    text = EXCESS_NEWLINES.sub("\n\n", text)
    return text.strip()


def content_hash(text: str) -> str:
    """Deterministic identity for exact dedup, insensitive to whitespace/case."""
    folded = re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip().lower()
    return hashlib.sha256(folded.encode("utf-8")).hexdigest()


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Quality filters
# --------------------------------------------------------------------------- #

SYMBOL_RE = re.compile(r"[^\w\s]", re.UNICODE)
WORD_RE = re.compile(r"\w", re.UNICODE)


def quality_verdict(text: str, *, min_chars: int, max_chars: int,
                    max_symbol_ratio: float = 0.5,
                    max_line_repeat_ratio: float = 0.5,
                    max_token_length: int = 400,
                    is_code_like: bool = False) -> str | None:
    """Return a rejection reason, or None when the document is acceptable."""
    if not text or len(text) < min_chars:
        return "too_short"
    if len(text) > max_chars:
        return "too_long"
    if "�" in text:
        return "encoding_corruption"

    if not is_code_like:
        # Code legitimately looks symbol-heavy; prose does not.
        symbols = len(SYMBOL_RE.findall(text))
        words = len(WORD_RE.findall(text))
        if words == 0:
            return "no_word_characters"
        if symbols / max(len(text), 1) > max_symbol_ratio:
            return "symbol_heavy"

    longest = max((len(part) for part in re.split(r"\s+", text) if part), default=0)
    if longest > max_token_length:
        # Minified bundles, base64 blobs and lockfile hashes all trip this. The
        # limit is per-category: compact JSON is legitimately whitespace-free and
        # needs a far higher ceiling than prose.
        return "unbroken_run"

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) >= 8:
        counts = Counter(lines)
        if counts.most_common(1)[0][1] / len(lines) > max_line_repeat_ratio:
            return "repeated_lines"
    return None


# --------------------------------------------------------------------------- #
# Repetition statistics
# --------------------------------------------------------------------------- #

def ngram_duplicate_ratio(text: str, size: int) -> float:
    words = text.split()
    if len(words) < size:
        return 0.0
    grams = [tuple(words[i:i + size]) for i in range(len(words) - size + 1)]
    return 1.0 - len(set(grams)) / len(grams)


# --------------------------------------------------------------------------- #
# Supplemental near-duplicate detection
# --------------------------------------------------------------------------- #

def simhash(text: str, shingle: int = 5, bits: int = 64) -> int:
    """SimHash over character shingles. Linear time, constant memory per doc."""
    folded = re.sub(r"\s+", " ", text).strip().lower()
    if len(folded) < shingle:
        return 0
    vector = [0] * bits
    for index in range(len(folded) - shingle + 1):
        digest = hashlib.blake2b(
            folded[index:index + shingle].encode("utf-8"), digest_size=8
        ).digest()
        value = int.from_bytes(digest, "big")
        for bit in range(bits):
            vector[bit] += 1 if (value >> bit) & 1 else -1
    result = 0
    for bit in range(bits):
        if vector[bit] > 0:
            result |= 1 << bit
    return result


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def near_duplicate_estimate(hashes: list[int], threshold: int = 3,
                            band_bits: int = 16) -> dict:
    """Banded SimHash grouping. Supplemental only; not used to drop documents."""
    if not hashes:
        return {"documents": 0, "near_duplicate_pairs": 0, "near_duplicate_ratio": 0.0}
    buckets: dict[tuple[int, int], list[int]] = {}
    bands = 64 // band_bits
    for index, value in enumerate(hashes):
        for band in range(bands):
            key = (band, (value >> (band * band_bits)) & ((1 << band_bits) - 1))
            buckets.setdefault(key, []).append(index)
    flagged: set[int] = set()
    pairs = 0
    for members in buckets.values():
        if len(members) < 2 or len(members) > 200:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                if hamming(hashes[members[i]], hashes[members[j]]) <= threshold:
                    pairs += 1
                    flagged.add(members[j])
    return {
        "documents": len(hashes),
        "near_duplicate_pairs": pairs,
        "near_duplicate_documents": len(flagged),
        "near_duplicate_ratio": len(flagged) / len(hashes),
        "threshold_bits": threshold,
    }
