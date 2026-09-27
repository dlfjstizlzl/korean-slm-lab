from __future__ import annotations

import json
import re
import statistics
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

CONTROL_TOKENS = [
    "<검색>",
    "<요약>",
    "<SQL>",
    "<코드>",
    "<이전0>",
    "<이전1>",
    "<이전2>",
    "<종료>",
]

CATEGORIES = {"general", "technical", "skill_dsl", "mixed_ascii"}
BYTE_FALLBACK_RE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip()


@dataclass(frozen=True)
class CorpusRecord:
    text: str
    category: str | None = None


def read_corpus(path: str | Path, deduplicate: bool = False) -> list[CorpusRecord]:
    records: list[CorpusRecord] = []
    seen: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = normalize_text(raw)
            if not line or line.startswith("#"):
                continue
            category = None
            text = line
            if "\t" in line:
                candidate, remainder = line.split("\t", 1)
                if candidate in CATEGORIES:
                    category = candidate
                    text = normalize_text(remainder)
            if not text or (deduplicate and text in seen):
                continue
            seen.add(text)
            records.append(CorpusRecord(text=text, category=category))
    return records


def iter_corpus(
    path: str | Path, deduplicate: bool = False, skip_comment_lines: bool = True
) -> Iterator[CorpusRecord]:
    seen: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = normalize_text(raw)
            if not line or (skip_comment_lines and line.startswith("#")):
                continue
            category = None
            text = line
            if "\t" in line:
                candidate, remainder = line.split("\t", 1)
                if candidate in CATEGORIES:
                    category = candidate
                    text = normalize_text(remainder)
            if not text or (deduplicate and text in seen):
                continue
            if deduplicate:
                seen.add(text)
            yield CorpusRecord(text=text, category=category)


def corpus_file_stats(
    path: str | Path, deduplicate: bool = False, skip_comment_lines: bool = True
) -> dict[str, int]:
    lines = characters = 0
    for record in iter_corpus(path, deduplicate=deduplicate, skip_comment_lines=skip_comment_lines):
        lines += 1
        characters += len(record.text)
    return {"lines": lines, "characters": characters}


def corpus_file_iterator(
    path: str | Path,
    batch_size: int = 1000,
    deduplicate: bool = False,
    skip_comment_lines: bool = True,
) -> Iterator[list[str]]:
    batch: list[str] = []
    for record in iter_corpus(path, deduplicate=deduplicate, skip_comment_lines=skip_comment_lines):
        batch.append(record.text)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def corpus_iterator(records: Sequence[CorpusRecord], batch_size: int = 1000) -> Iterator[list[str]]:
    for start in range(0, len(records), batch_size):
        yield [record.text for record in records[start : start + batch_size]]


def percentile_nearest_rank(values: Sequence[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((percentile * len(ordered) + 0.999999)) - 1))
    return ordered[index]


def tokenizer_name(path_or_id: str) -> str:
    return Path(path_or_id.rstrip("/")).name or path_or_id


def special_token_strings(tokenizer) -> dict[str, str]:
    result: dict[str, str] = {}
    for attr in ("bos_token", "eos_token", "pad_token", "unk_token", "sep_token", "cls_token", "mask_token"):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            result[attr] = str(value)
    for index, value in enumerate(getattr(tokenizer, "all_special_tokens", [])):
        result.setdefault(f"all_special_tokens[{index}]", str(value))
    return result


def assert_required_special_tokens(base_tokenizer, new_tokenizer) -> None:
    missing: list[str] = []
    invalid: list[str] = []
    new_vocab = new_tokenizer.get_vocab()
    for attr, token in special_token_strings(base_tokenizer).items():
        if token not in new_vocab:
            missing.append(f"{attr}={token!r}")
            continue
        token_id = new_tokenizer.convert_tokens_to_ids(token)
        if token_id is None or not 0 <= int(token_id) < len(new_tokenizer):
            invalid.append(f"{attr}={token!r}:{token_id}")
    if missing or invalid:
        details = "; ".join([*(f"missing {item}" for item in missing), *(f"invalid {item}" for item in invalid)])
        raise ValueError(f"Required base-model special tokens are not valid in the new tokenizer: {details}")


def assert_control_tokens_atomic(tokenizer) -> dict[str, int]:
    token_ids: dict[str, int] = {}
    failures: list[str] = []
    for token in CONTROL_TOKENS:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) != 1:
            failures.append(f"{token} -> {encoded}")
        else:
            token_ids[token] = int(encoded[0])
    if failures:
        raise ValueError("Control tokens are not atomic: " + "; ".join(failures))
    return token_ids


def non_special_added_tokens(tokenizer) -> list:
    """Return base added-token objects that train_new_from_iterator otherwise drops."""
    return [token for token in tokenizer.added_tokens_decoder.values() if not token.special]


def assert_base_added_tokens_preserved(base_tokenizer, new_tokenizer) -> dict[str, int]:
    missing: list[str] = []
    split: list[str] = []
    ids: dict[str, int] = {}
    for token in base_tokenizer.added_tokens_decoder.values():
        text = str(token)
        if text not in new_tokenizer.get_vocab():
            missing.append(text)
            continue
        encoded = new_tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) != 1:
            split.append(f"{text} -> {encoded}")
        else:
            ids[text] = int(encoded[0])
    if missing or split:
        details = "; ".join([*(f"missing {item}" for item in missing), *(f"split {item}" for item in split)])
        raise ValueError(f"Base added tokens were not preserved atomically: {details}")
    return ids


class SentencePieceAdapter:
    """Small benchmark-only adapter for SentencePiece Unigram artifacts."""

    def __init__(self, directory: str | Path):
        import sentencepiece as spm

        directory = Path(directory)
        self.directory = directory
        self.processor = spm.SentencePieceProcessor(model_file=str(directory / "tokenizer.model"))
        self.name_or_path = str(directory)
        self.vocab_size = self.processor.vocab_size()

    def __len__(self) -> int:
        return self.processor.vocab_size()

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(self.processor.encode(normalize_text(text), out_type=int))

    def tokenize(self, text: str) -> list[str]:
        return list(self.processor.encode(normalize_text(text), out_type=str))

    def convert_ids_to_tokens(self, ids: Iterable[int]) -> list[str]:
        return [self.processor.id_to_piece(int(token_id)) for token_id in ids]

    def get_vocab(self) -> dict[str, int]:
        return {self.processor.id_to_piece(i): i for i in range(len(self))}


def load_tokenizer(path_or_id: str, trust_remote_code: bool = False):
    path = Path(path_or_id)
    if path.is_dir() and (path / "tokenizer.model").exists() and (path / "sentencepiece_meta.json").exists():
        return SentencePieceAdapter(path)
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path_or_id, trust_remote_code=trust_remote_code)


def token_metrics(tokenizer, records: Sequence[CorpusRecord]) -> dict:
    counts: list[int] = []
    byte_fallback = 0
    unknown = 0
    total_characters = 0
    unk_id = getattr(tokenizer, "unk_token_id", None)
    for record in records:
        ids = tokenizer.encode(record.text, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(ids)
        if isinstance(tokens, str):
            tokens = [tokens]
        counts.append(len(ids))
        total_characters += len(record.text)
        byte_fallback += sum(bool(BYTE_FALLBACK_RE.fullmatch(str(token))) for token in tokens)
        if unk_id is not None:
            unknown += sum(int(token_id) == int(unk_id) for token_id in ids)
    total_tokens = sum(counts)
    return {
        "vocab_size": len(tokenizer),
        "total_sentences": len(records),
        "total_characters": total_characters,
        "total_tokens": total_tokens,
        "average_tokens_per_sentence": (total_tokens / len(records)) if records else 0.0,
        "median_tokens_per_sentence": statistics.median(counts) if counts else 0.0,
        "p95_tokens_per_sentence": percentile_nearest_rank(counts, 0.95),
        "characters_per_token": (total_characters / total_tokens) if total_tokens else 0.0,
        "tokens_per_character": (total_tokens / total_characters) if total_characters else 0.0,
        "byte_fallback_tokens": byte_fallback,
        "byte_fallback_percent": (100.0 * byte_fallback / total_tokens) if total_tokens else 0.0,
        "unknown_tokens": unknown,
    }


def write_json(path: str | Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
