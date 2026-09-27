from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterator

TEXT_KEYS = ("text", "content", "body", "document", "raw_content")
HTML_RE = re.compile(r"</?(?:html|head|body|div|span|p|br|script|style|article|section|a)\b[^>]*>", re.I)
SPACE_RE = re.compile(r"\s+")


def clean_text(value: str) -> str:
    value = unicodedata.normalize("NFC", value.replace("\x00", ""))
    value = html.unescape(value)
    if HTML_RE.search(value):
        value = HTML_RE.sub(" ", value)
    return SPACE_RE.sub(" ", value).strip()


def discover_fineweb_files(data_dir: str | Path) -> list[Path]:
    root = Path(data_dir)
    supported = {".txt", ".json", ".jsonl", ".parquet"}
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in supported
        and "fineweb" in unicodedata.normalize("NFC", str(path)).lower()
        and "processed" not in path.parts
    )


def _extract_text(record) -> str | None:
    if isinstance(record, str):
        return record
    if not isinstance(record, dict):
        return None
    for key in TEXT_KEYS:
        value = record.get(key)
        if isinstance(value, str):
            return value
    return None


def _iter_txt(path: Path) -> Iterator[str]:
    # FineWeb export keeps source-document boundaries as blank lines.
    parts: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            if raw.strip():
                parts.append(raw.strip())
            elif parts:
                yield " ".join(parts)
                parts.clear()
        if parts:
            yield " ".join(parts)


def _iter_json(path: Path) -> Iterator[str]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                if not raw.strip():
                    continue
                text = _extract_text(json.loads(raw))
                if text is not None:
                    yield text
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload if isinstance(payload, list) else [payload]
    for record in records:
        text = _extract_text(record)
        if text is not None:
            yield text


def _iter_parquet(path: Path) -> Iterator[str]:
    from datasets import load_dataset

    dataset = load_dataset("parquet", data_files=str(path), split="train", streaming=True)
    for record in dataset:
        text = _extract_text(record)
        if text is not None:
            yield text


def iter_fineweb_documents(paths: list[Path], min_chars: int = 20) -> Iterator[tuple[str, str]]:
    for path in paths:
        suffix = path.suffix.lower()
        if suffix == ".txt":
            iterator = _iter_txt(path)
        elif suffix in {".json", ".jsonl"}:
            iterator = _iter_json(path)
        elif suffix == ".parquet":
            iterator = _iter_parquet(path)
        else:
            continue
        for raw in iterator:
            text = clean_text(raw)
            if len(text) >= min_chars:
                yield str(path), text


def is_holdout(text: str, ratio: float = 0.05, seed: int = 42) -> bool:
    digest = hashlib.blake2b(f"{seed}\0{text}".encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big") / 2**64
    return value < ratio
