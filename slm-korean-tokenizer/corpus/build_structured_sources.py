#!/usr/bin/env python3
"""Extract, clean, and split the external JSON/SQL corpora.

The external files are raw StarCoder/The-Stack/Spider dumps. They are never
modified; every artifact produced here lands under ``data/processed/structured``.

JSON records arrive as whole pretty-printed GitHub files carrying StarCoder
metadata headers (``<reponame>``/``<filename>``/``<gh_stars>``), so documents are
reconstructed by brace balance after the headers are stripped. SQL arrives as
raw dump text, so statements are recovered by quote-aware ``;`` splitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

DEFAULT_JSON = "data/external/json/starcoder_json.txt"
DEFAULT_STACK_SQL = "data/external/sql/stack_sql.txt"
DEFAULT_SPIDER_SQL = "data/external/sql/spider_sql.txt"
OUTPUT_DIR = "data/processed/structured"

STARCODER_MARKERS = (
    re.compile(r"<reponame>[^\n<]*"),
    re.compile(r"<filename>[^\n<]*"),
    re.compile(r"<gh_stars>[^\n]*"),
)

# Dependency/lock artifacts and generated bundles carry almost no reusable
# structure but huge character mass, so they are dropped rather than sampled.
LOCK_KEYS = {"lockfileVersion", "integrity", "resolved"}

SQL_LEAD = re.compile(
    r"^\s*(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|WITH|REPLACE|TRUNCATE|GRANT)\b",
    re.IGNORECASE,
)
SQL_FEATURES = {
    "SELECT": r"^\s*SELECT\b",
    "INSERT": r"^\s*INSERT\b",
    "UPDATE": r"^\s*UPDATE\b",
    "DELETE": r"^\s*DELETE\b",
    "CREATE": r"^\s*CREATE\b",
    "ALTER": r"^\s*ALTER\b",
    "DROP": r"^\s*DROP\b",
    "WITH/CTE": r"^\s*WITH\b",
    "JOIN": r"\bJOIN\b",
    "GROUP BY": r"\bGROUP\s+BY\b",
    "ORDER BY": r"\bORDER\s+BY\b",
    "HAVING": r"\bHAVING\b",
    "nested subquery": r"\(\s*SELECT\b",
    "aggregate": r"\b(COUNT|SUM|AVG|MAX|MIN)\s*\(",
    "set operation": r"\b(UNION|INTERSECT|EXCEPT)\b",
    "schema-qualified": r"\b\w+\.\w+\b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cleaned external JSON/SQL structured sources.")
    parser.add_argument("--json-file", default=DEFAULT_JSON)
    parser.add_argument("--stack-sql-file", default=DEFAULT_STACK_SQL)
    parser.add_argument("--spider-sql-file", default=DEFAULT_SPIDER_SQL)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-ratio", type=float, default=0.10)
    parser.add_argument("--max-json-doc-chars", type=int, default=100_000)
    parser.add_argument("--max-sql-statement-chars", type=int, default=2_000)
    parser.add_argument("--min-sql-statement-chars", type=int, default=10)
    parser.add_argument("--max-line-chars", type=int, default=4_000)
    parser.add_argument("--report", default="outputs/structured_corpus_report.json")
    return parser.parse_args()


def normalize_line(text: str) -> str:
    return unicodedata.normalize("NFC", text.replace("\t", "    ")).strip()


def digest(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


def score(seed: int, namespace: str, value: str) -> float:
    raw = hashlib.blake2b(f"{seed}\0{namespace}\0{value}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(raw, "big") / float(1 << 64)


def ratios(text: str) -> dict[str, float]:
    denominator = max(1, len(text))
    return {
        "ascii_ratio_percent": 100.0 * sum(c.isascii() for c in text) / denominator,
        "english_alphabet_ratio_percent": 100.0 * sum(c.isascii() and c.isalpha() for c in text) / denominator,
        "digit_ratio_percent": 100.0 * sum(c.isdigit() for c in text) / denominator,
        "hangul_ratio_percent": 100.0 * sum("가" <= c <= "힣" for c in text) / denominator,
    }


def strip_markers(text: str) -> str:
    for pattern in STARCODER_MARKERS:
        text = pattern.sub("", text)
    return text


def reconstruct_json_documents(text: str) -> list[str]:
    """Recover whole JSON documents from a concatenated dump by brace balance."""
    lines = text.split("\n")
    documents: list[str] = []
    index = 0
    total = len(lines)
    while index < total:
        if lines[index][:1] not in ("{", "["):
            index += 1
            continue
        depth = 0
        in_string = False
        escaped = False
        buffer: list[str] = []
        cursor = index
        closed = False
        while cursor < total and not closed:
            for char in lines[cursor]:
                if escaped:
                    escaped = False
                    continue
                if char == "\\" and in_string:
                    escaped = True
                    continue
                if char == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if char in "{[":
                    depth += 1
                elif char in "}]":
                    depth -= 1
                    if depth == 0:
                        closed = True
            buffer.append(lines[cursor])
            cursor += 1
        documents.append("\n".join(buffer))
        index = cursor
    return documents


def classify_json(document: str, parsed, parse_ok: bool, max_chars: int) -> str:
    if not parse_ok:
        if "<KEY>" in document:
            return "malformed JSON (redacted <KEY>)"
        if any(key in document for key in LOCK_KEYS):
            return "dependency lock file"
        return "malformed JSON"
    if len(document) > max_chars:
        return "oversized generated file"
    if isinstance(parsed, list):
        return "valid JSON array"
    if isinstance(parsed, dict):
        keys = set(parsed.keys())
        if keys & LOCK_KEYS:
            return "dependency lock file"
        if {"name", "version"} <= keys and ("dependencies" in keys or "scripts" in keys):
            return "config/package metadata"
        return "valid JSON object"
    return "valid JSON scalar"


def looks_minified(document: str) -> bool:
    lines = document.split("\n")
    return len(lines) <= 3 and max((len(line) for line in lines), default=0) > 2_000


def extract_json(path: Path, args: argparse.Namespace) -> tuple[list[dict], dict]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    documents = reconstruct_json_documents(strip_markers(raw))
    categories: Counter = Counter()
    dropped: Counter = Counter()
    records: list[dict] = []
    kept_docs = 0
    for document in documents:
        try:
            parsed = json.loads(document)
            parse_ok = True
        except Exception:
            parsed, parse_ok = None, False
        category = classify_json(document, parsed, parse_ok, args.max_json_doc_chars)
        categories[category] += 1
        categories[f"__chars__{category}"] += len(document)
        if not parse_ok:
            dropped["malformed"] += 1
            dropped["malformed_chars"] += len(document)
            continue
        if category == "dependency lock file":
            dropped["lock_file"] += 1
            dropped["lock_file_chars"] += len(document)
            continue
        if len(document) > args.max_json_doc_chars:
            dropped["oversized"] += 1
            dropped["oversized_chars"] += len(document)
            continue
        if looks_minified(document):
            dropped["minified"] += 1
            dropped["minified_chars"] += len(document)
            continue
        kept_docs += 1
        document_id = f"json:{kept_docs}"
        # Compact form is what the SkillLM runtime actually transports; the
        # pretty-printed lines are what a checked-in JSON file looks like.
        compact = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
        if len(compact) <= args.max_line_chars:
            records.append({"text": compact, "source": "starcoder_json_compact", "document": document_id})
        for line in document.split("\n"):
            line = normalize_line(line)
            if line and len(line) <= args.max_line_chars:
                records.append({"text": line, "source": "starcoder_json_pretty", "document": document_id})
    stats = {
        "input_documents": len(documents),
        "input_characters": len(raw),
        "kept_documents": kept_docs,
        "categories": {k: v for k, v in categories.items() if not k.startswith("__chars__")},
        "category_characters": {k[9:]: v for k, v in categories.items() if k.startswith("__chars__")},
        "dropped": dict(dropped),
    }
    return records, stats


def split_sql_statements(text: str) -> list[str]:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = "\n".join(line for line in text.split("\n") if not line.strip().startswith(("--", "#")))
    statements: list[str] = []
    buffer: list[str] = []
    quote: str | None = None
    escaped = False
    for char in text:
        if escaped:
            buffer.append(char)
            escaped = False
            continue
        if char == "\\" and quote:
            buffer.append(char)
            escaped = True
            continue
        if quote:
            buffer.append(char)
            if char == quote:
                quote = None
            continue
        if char in "'\"`":
            quote = char
            buffer.append(char)
            continue
        if char == ";":
            candidate = "".join(buffer).strip()
            if candidate:
                statements.append(candidate + ";")
            buffer = []
            continue
        buffer.append(char)
    candidate = "".join(buffer).strip()
    if candidate:
        statements.append(candidate)
    return statements


def sql_feature_counts(statements: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name, pattern in SQL_FEATURES.items():
        counts[name] = sum(1 for s in statements if re.search(pattern, s, re.IGNORECASE))
    return counts


def extract_sql(path: Path, source: str, args: argparse.Namespace, statement_per_line: bool) -> tuple[list[dict], dict]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    if statement_per_line:
        statements = [line.strip() for line in raw.split("\n") if line.strip()]
    else:
        statements = split_sql_statements(raw)
    dropped: Counter = Counter()
    kept: list[str] = []
    for statement in statements:
        if len(statement) < args.min_sql_statement_chars:
            dropped["too_short"] += 1
            continue
        if len(statement) > args.max_sql_statement_chars:
            dropped["oversized_bulk_dump"] += 1
            dropped["oversized_bulk_dump_chars"] += len(statement)
            continue
        if not SQL_LEAD.match(statement):
            dropped["not_a_sql_statement"] += 1
            dropped["not_a_sql_statement_chars"] += len(statement)
            continue
        kept.append(statement)
    records: list[dict] = []
    for index, statement in enumerate(kept, 1):
        document_id = f"{source}:{index}"
        lines = [normalize_line(line) for line in statement.split("\n")]
        lines = [line for line in lines if line and len(line) <= args.max_line_chars]
        if len(lines) == 1:
            records.append({"text": lines[0], "source": source, "document": document_id})
            continue
        # Multi-line DDL keeps its natural shape; a collapsed single-line form is
        # added only for short statements, matching how queries appear inline.
        for line in lines:
            records.append({"text": line, "source": source, "document": document_id})
        collapsed = normalize_line(" ".join(lines))
        if len(collapsed) <= 300:
            records.append({"text": collapsed, "source": source, "document": document_id})
    stats = {
        "input_statements": len(statements),
        "input_characters": len(raw),
        "kept_statements": len(kept),
        "kept_characters": sum(map(len, kept)),
        "dropped": dict(dropped),
        "features": sql_feature_counts(kept),
    }
    return records, stats


def split_and_write(records: list[dict], args: argparse.Namespace, out_dir: Path) -> dict:
    """Document-level deterministic split; a document never straddles the split."""
    by_source: dict[str, list[dict]] = {}
    for record in records:
        by_source.setdefault(record["source"], []).append(record)
    report: dict = {}
    train_seen: set[bytes] = set()
    valid_seen: set[bytes] = set()
    train_rows: list[dict] = []
    valid_rows: list[dict] = []
    for source, rows in sorted(by_source.items()):
        counts = Counter()
        for row in rows:
            is_valid = score(args.seed, "structured-holdout", row["document"]) < args.valid_ratio
            key = digest(row["text"])
            bucket, seen = (valid_rows, valid_seen) if is_valid else (train_rows, train_seen)
            if key in seen:
                counts["duplicate_dropped"] += 1
                continue
            seen.add(key)
            bucket.append(row)
            counts["valid_lines" if is_valid else "train_lines"] += 1
            counts["valid_characters" if is_valid else "train_characters"] += len(row["text"])
        report[source] = dict(counts)
    # A line that appears in both splits is removed from validation only.
    deduped_valid = []
    cross = 0
    for row in valid_rows:
        if digest(row["text"]) in train_seen:
            cross += 1
            continue
        deduped_valid.append(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("structured_train", train_rows), ("structured_valid", deduped_valid)):
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        with (out_dir / f"{name}.txt").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(row["text"] + "\n")
    report["__split__"] = {
        "train_lines": len(train_rows),
        "train_characters": sum(len(r["text"]) for r in train_rows),
        "valid_lines": len(deduped_valid),
        "valid_characters": sum(len(r["text"]) for r in deduped_valid),
        "validation_lines_removed_for_train_overlap": cross,
        "valid_ratio": args.valid_ratio,
        "seed": args.seed,
    }
    return report


def per_source_text_stats(rows: list[dict]) -> dict:
    stats: dict[str, dict] = {}
    grouped: dict[str, list[str]] = {}
    for row in rows:
        grouped.setdefault(row["source"], []).append(row["text"])
    for source, texts in sorted(grouped.items()):
        blob = "\n".join(texts)
        stats[source] = {
            "lines": len(texts),
            "characters": sum(map(len, texts)),
            "unique_lines": len({digest(t) for t in texts}),
            "average_line_length": sum(map(len, texts)) / max(1, len(texts)),
            **ratios(blob),
        }
    return stats


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    json_records, json_stats = extract_json(Path(args.json_file), args)
    stack_records, stack_stats = extract_sql(Path(args.stack_sql_file), "stack_sql", args, statement_per_line=False)
    spider_records, spider_stats = extract_sql(Path(args.spider_sql_file), "spider_sql", args, statement_per_line=True)

    all_records = json_records + stack_records + spider_records
    split_report = split_and_write(all_records, args, out_dir)

    report = {
        "seed": args.seed,
        "inputs": {
            "json": args.json_file,
            "stack_sql": args.stack_sql_file,
            "spider_sql": args.spider_sql_file,
        },
        "json_extraction": json_stats,
        "stack_sql_extraction": stack_stats,
        "spider_sql_extraction": spider_stats,
        "record_statistics": per_source_text_stats(all_records),
        "split": split_report,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["split"]["__split__"], indent=2))
    print(f"wrote {out_dir}/structured_train.* , {out_dir}/structured_valid.* and {report_path}")


if __name__ == "__main__":
    main()
