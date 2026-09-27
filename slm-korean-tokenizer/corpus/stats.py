from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import unicodedata
from collections import Counter
from pathlib import Path

WORD_RE = re.compile(r"[가-힣]+|[A-Za-z_][A-Za-z0-9_]*|\d+")


def nearest_rank(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    values.sort()
    return values[max(0, math.ceil(percentile * len(values)) - 1)]


def analyze_text_file(path: str | Path, lexical: bool = True) -> dict:
    path = Path(path)
    lengths: list[int] = []
    unique_lines: set[bytes] = set()
    unique_chars: set[str] = set()
    hangul_syllables: set[str] = set()
    words: Counter[str] = Counter()
    characters = hangul = ascii_count = english = digits = whitespace = punctuation = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            text = raw.rstrip("\r\n")
            if not text:
                continue
            encoded = text.encode("utf-8")
            unique_lines.add(__import__("hashlib").blake2b(encoded, digest_size=16).digest())
            lengths.append(len(text))
            characters += len(text)
            unique_chars.update(text)
            for char in text:
                if "가" <= char <= "힣":
                    hangul += 1
                    hangul_syllables.add(char)
                if char.isascii():
                    ascii_count += 1
                if char.isascii() and char.isalpha():
                    english += 1
                if char.isdigit():
                    digits += 1
                if char.isspace():
                    whitespace += 1
                if unicodedata.category(char).startswith("P"):
                    punctuation += 1
            if lexical:
                words.update(match.group(0).lower() for match in WORD_RE.finditer(text))
    lines = len(lengths)
    denominator = characters or 1
    return {
        "path": str(path),
        "documents_or_lines": lines,
        "total_characters": characters,
        "unique_lines": len(unique_lines),
        "duplicate_ratio_percent": 100.0 * (lines - len(unique_lines)) / (lines or 1),
        "hangul_ratio_percent": 100.0 * hangul / denominator,
        "ascii_ratio_percent": 100.0 * ascii_count / denominator,
        "english_alphabet_ratio_percent": 100.0 * english / denominator,
        "digit_ratio_percent": 100.0 * digits / denominator,
        "whitespace_ratio_percent": 100.0 * whitespace / denominator,
        "punctuation_ratio_percent": 100.0 * punctuation / denominator,
        "unique_unicode_characters": len(unique_chars),
        "unique_hangul_syllables": len(hangul_syllables),
        "average_line_characters": characters / (lines or 1),
        "median_line_characters": statistics.median(lengths) if lengths else 0,
        "p95_line_characters": nearest_rank(lengths, 0.95),
        "word_like_tokens": sum(words.values()) if lexical else None,
        "unique_word_like_types": len(words) if lexical else None,
        "hapax_type_ratio_percent": 100.0 * sum(value == 1 for value in words.values()) / (len(words) or 1) if lexical else None,
        "top_100_terms": words.most_common(100) if lexical else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute streaming corpus statistics for one or more text files.")
    parser.add_argument("files", nargs="+")
    parser.add_argument("--output")
    parser.add_argument("--no-lexical", action="store_true")
    args = parser.parse_args()
    payload = {str(path): analyze_text_file(path, lexical=not args.no_lexical) for path in args.files}
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
