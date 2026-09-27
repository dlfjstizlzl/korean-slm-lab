#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from corpus.stats import analyze_text_file


MIXED_SOURCE_NAMES = (
    "korean_english_mixed",
    "source_code",
    "structured_data",
    "sql",
    "cli_paths",
    "urls_api",
    "identifiers_technical",
)

CODE_EXTENSIONS = {
    ".py", ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx",
    ".kt", ".java", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx",
    ".rs", ".go", ".sh", ".bash", ".zsh", ".cmake", ".md",
}
STRUCTURED_EXTENSIONS = {".json", ".jsonl", ".yaml", ".yml", ".toml", ".xml"}
SQL_EXTENSIONS = {".sql"}
SPECIAL_FILENAMES = {"dockerfile", "cmakelists.txt", "makefile", ".gitignore"}
EXCLUDED_DIR_NAMES = {
    ".git", "node_modules", "dist", "build", "target", "venv", ".venv",
    "__pycache__", "coverage", "outputs", "data", "models", ".cache",
}
EXCLUDED_FILENAMES = {"package-lock.json", "pnpm-lock.yaml", "yarn.lock"}
URL_RE = re.compile(r"https?://[^\s<>'\"`]+")
API_RE = re.compile(r"(?<![A-Za-z0-9_.-])/(?:api|v\d+)(?:/[A-Za-z0-9_.~:@!$&'()*+,;=%-]+)+(?:\?[A-Za-z0-9_.~:@!$&'()*+,;=/%?-]*)?")
IDENTIFIER_RE = re.compile(
    r"\b(?:[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+|[A-Za-z]+(?:[A-Z][A-Za-z0-9]+)+|"
    r"[A-Za-z0-9]+(?:[-.][A-Za-z0-9_]+)+|[A-Za-z_][A-Za-z0-9_]{2,})\b"
)
SQL_LINE_RE = re.compile(
    r"\b(?:SELECT\s+.{1,2048}?\s+FROM|INSERT\s+INTO|UPDATE\s+[A-Za-z0-9_.]+\s+SET|"
    r"DELETE\s+FROM|CREATE\s+TABLE|ALTER\s+TABLE)\b"
)
CLI_MARKERS = re.compile(
    r"(?:^|\s)(?:python3?|pip3?|cmake|docker|git|cargo|go|java|gradle|npm|pnpm|yarn|"
    r"\./[A-Za-z0-9_.~/-]+|[A-Za-z0-9_.-]+/[A-Za-z0-9_.~/-]+|--[a-z0-9][a-z0-9-]*)"
)
HOME_RE = re.compile(r"/(?:Users|home)/[^/\s]+")


@dataclass(frozen=True)
class ProjectFile:
    root: Path
    path: Path
    relative_id: str
    text: str
    validation: bool


class CategoryWriter:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.handles = {
            name: (output_dir / f"{name}.txt").open("w", encoding="utf-8")
            for name in MIXED_SOURCE_NAMES
        }
        self.seen: dict[str, set[bytes]] = {name: set() for name in MIXED_SOURCE_NAMES}
        self.counts: dict[str, Counter] = {name: Counter() for name in MIXED_SOURCE_NAMES}

    def add(self, category: str, text: str) -> bool:
        text = normalize_line(text)
        if not text:
            return False
        self.counts[category]["input_lines"] += 1
        self.counts[category]["input_characters"] += len(text)
        digest = text_digest(text)
        if digest in self.seen[category]:
            self.counts[category]["duplicates"] += 1
            return False
        self.seen[category].add(digest)
        self.handles[category].write(text + "\n")
        self.counts[category]["lines"] += 1
        self.counts[category]["characters"] += len(text)
        return True

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build real mixed-workload sources and controlled tokenizer corpora.")
    parser.add_argument("--base-jsonl", default="data/processed/tokenizer_train_real.jsonl")
    parser.add_argument("--fineweb-train", default="data/processed/fineweb_general_train.txt")
    parser.add_argument("--fineweb-holdout", default="data/processed/fineweb_general_holdout.txt")
    parser.add_argument("--output-dir", default="data/processed/mixed")
    parser.add_argument(
        "--code-root",
        action="append",
        help="Source root to scan; repeat for multiple roots. Defaults to this repository.",
    )
    parser.add_argument("--ascii-min", type=float, default=0.20)
    parser.add_argument("--ascii-max", type=float, default=0.80)
    parser.add_argument(
        "--english-min",
        type=float,
        default=0.10,
        help="Minimum ASCII-letter ratio, preventing whitespace-only ASCII matches",
    )
    parser.add_argument("--file-holdout-ratio", type=float, default=0.10)
    parser.add_argument("--max-file-bytes", type=int, default=1_048_576)
    parser.add_argument("--mixed-ratios", nargs="+", type=float, default=[0.10, 0.15])
    parser.add_argument("--target-chars", type=int, default=0, help="0 reuses the base corpus character count")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-json", default="outputs/mixed_corpus_report.json")
    parser.add_argument("--output-markdown", default="outputs/MIXED_CORPUS_REPORT.md")
    return parser.parse_args()


def text_digest(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


def score(seed: int, namespace: str, value: str) -> float:
    digest = hashlib.blake2b(f"{seed}\0{namespace}\0{value}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def normalize_line(text: str) -> str:
    text = unicodedata.normalize("NFC", text.replace("\x00", ""))
    return text.strip()


def generalize_home_path(text: str) -> str:
    return HOME_RE.sub("~", text)


def ratios(text: str) -> tuple[float, float, float]:
    denominator = len(text) or 1
    ascii_ratio = sum(char.isascii() for char in text) / denominator
    english_ratio = sum(char.isascii() and char.isalpha() for char in text) / denominator
    hangul_ratio = sum("가" <= char <= "힣" for char in text) / denominator
    return ascii_ratio, english_ratio, hangul_ratio


def prepare_outputs(paths: list[Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to replace existing outputs: {existing[:6]}; pass --overwrite")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def excluded_path(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True
    for part in relative.parts[:-1]:
        lower = part.lower()
        if lower in EXCLUDED_DIR_NAMES or lower.startswith("build-"):
            return True
    lower_name = path.name.lower()
    if lower_name in EXCLUDED_FILENAMES or lower_name.endswith(".min.js"):
        return True
    return False


def supported_file(path: Path) -> bool:
    return path.suffix.lower() in CODE_EXTENSIONS | STRUCTURED_EXTENSIONS | SQL_EXTENSIONS or path.name.lower() in SPECIAL_FILENAMES


def looks_generated_or_minified(text: str) -> bool:
    lines = text.splitlines()
    nonempty = [line for line in lines if line.strip()]
    if not nonempty:
        return True
    if len(text) > 10_000 and len(nonempty) <= 2:
        return True
    return max(map(len, nonempty)) > 20_000


def iter_project_files(roots: list[Path], seed: int, holdout_ratio: float, max_bytes: int) -> Iterator[ProjectFile]:
    visited: set[Path] = set()
    for root_index, root in enumerate(roots):
        root = root.resolve()
        if not root.is_dir():
            print(f"WARNING: code root is not a directory: {root}", file=sys.stderr)
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or excluded_path(path, root) or not supported_file(path):
                continue
            resolved = path.resolve()
            if resolved in visited:
                continue
            visited.add(resolved)
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size <= 0 or size > max_bytes:
                continue
            raw = path.read_bytes()
            if b"\x00" in raw[:8192]:
                continue
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                continue
            text = unicodedata.normalize("NFC", text.replace("\x00", ""))
            if looks_generated_or_minified(text):
                continue
            relative = path.relative_to(root).as_posix()
            identifier = f"root{root_index}/{root.name}/{relative}"
            yield ProjectFile(root, path, identifier, text, score(seed, "file-holdout", identifier) < holdout_ratio)


def iter_nonempty_lines(text: str) -> Iterator[str]:
    for line in text.splitlines():
        if normalized := normalize_line(line):
            yield generalize_home_path(normalized)


def clean_url(value: str) -> str:
    return value.rstrip(".,;:!?)]}>'\"")


def file_category(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in STRUCTURED_EXTENSIONS:
        return "structured_data"
    if suffix in SQL_EXTENSIONS:
        return "sql"
    return "source_code"


def project_records(item: ProjectFile) -> Iterator[tuple[str, str]]:
    primary = file_category(item.path)
    for line in iter_nonempty_lines(item.text):
        yield primary, line
        if primary != "sql" and SQL_LINE_RE.search(line):
            yield "sql", line
        if CLI_MARKERS.search(line):
            yield "cli_paths", line
        for match in URL_RE.finditer(line):
            if value := clean_url(match.group(0)):
                yield "urls_api", value
        for match in API_RE.finditer(line):
            if value := clean_url(match.group(0)):
                yield "urls_api", value
        for match in IDENTIFIER_RE.finditer(line):
            value = match.group(0)
            if 3 <= len(value) <= 100 and not value.isdigit():
                yield "identifiers_technical", value


def write_validation(handle, seen: set[bytes], text: str, source: str, document: str) -> bool:
    text = normalize_line(text)
    if not text:
        return False
    digest = text_digest(text)
    if digest in seen:
        return False
    seen.add(digest)
    handle.write(json.dumps({"text": text, "source": source, "document": document}, ensure_ascii=False) + "\n")
    return True


def extract_sources(args: argparse.Namespace, output_dir: Path, roots: list[Path]) -> tuple[dict, Path]:
    writer = CategoryWriter(output_dir)
    validation_raw = output_dir / "mixed_validation_raw.jsonl"
    validation_seen: set[bytes] = set()
    discovered = Counter()
    with validation_raw.open("w", encoding="utf-8") as validation:
        for split, path in (("train", Path(args.fineweb_train)), ("valid", Path(args.fineweb_holdout))):
            with path.open("r", encoding="utf-8") as handle:
                for index, raw in enumerate(handle):
                    text = normalize_line(raw)
                    if not text:
                        continue
                    ascii_ratio, english_ratio, hangul_ratio = ratios(text)
                    if not (
                        args.ascii_min <= ascii_ratio <= args.ascii_max
                        and english_ratio >= args.english_min
                        and hangul_ratio > 0
                    ):
                        continue
                    discovered[f"fineweb_{split}_documents"] += 1
                    discovered[f"fineweb_{split}_characters"] += len(text)
                    if split == "train":
                        writer.add("korean_english_mixed", text)
                    else:
                        write_validation(validation, validation_seen, text, "korean_english_mixed", f"fineweb-holdout:{index}")
                    for match in URL_RE.finditer(text):
                        url = clean_url(match.group(0))
                        if not url:
                            continue
                        if split == "train":
                            writer.add("urls_api", url)
                        else:
                            write_validation(validation, validation_seen, url, "urls_api", f"fineweb-holdout:{index}")

        for file_index, item in enumerate(iter_project_files(roots, args.seed, args.file_holdout_ratio, args.max_file_bytes), 1):
            discovered["project_files"] += 1
            discovered["project_file_bytes"] += item.path.stat().st_size
            discovered["project_validation_files" if item.validation else "project_train_files"] += 1
            for category, text in project_records(item):
                if item.validation:
                    write_validation(validation, validation_seen, text, category, item.relative_id)
                else:
                    writer.add(category, text)
            if file_index % 500 == 0:
                print(f"[Project] processed {file_index:,} files", flush=True)
    writer.close()
    return {
        "discovered": dict(discovered),
        "extraction_counts": {name: dict(values) for name, values in writer.counts.items()},
        "validation_raw_records": len(validation_seen),
    }, validation_raw


def build_mixed_pool(output_dir: Path) -> tuple[Path, dict]:
    pool = output_dir / "mixed_pool.jsonl"
    seen: set[bytes] = set()
    counts: dict[str, Counter] = {name: Counter() for name in MIXED_SOURCE_NAMES}
    total = Counter()
    # Code-like sources are ordered first only for stable output; hash sampling remains source-neutral.
    order = (
        "source_code", "structured_data", "sql", "cli_paths", "urls_api",
        "identifiers_technical", "korean_english_mixed",
    )
    with pool.open("w", encoding="utf-8") as output:
        for source in order:
            with (output_dir / f"{source}.txt").open("r", encoding="utf-8") as handle:
                for raw in handle:
                    text = raw.rstrip("\r\n")
                    if not text:
                        continue
                    counts[source]["input_lines"] += 1
                    digest = text_digest(text)
                    if digest in seen:
                        counts[source]["cross_source_duplicates"] += 1
                        continue
                    seen.add(digest)
                    output.write(json.dumps({"text": text, "source": source}, ensure_ascii=False) + "\n")
                    counts[source]["lines"] += 1
                    counts[source]["characters"] += len(text)
                    total["lines"] += 1
                    total["characters"] += len(text)
    return pool, {"sources": {name: dict(value) for name, value in counts.items()}, "total": dict(total)}


def base_corpus_stats(base_jsonl: Path, excluded_fineweb_hashes: set[bytes]) -> Counter:
    counts = Counter()
    with base_jsonl.open("r", encoding="utf-8") as handle:
        for raw in handle:
            record = json.loads(raw)
            text = normalize_line(str(record["text"]))
            if record.get("source") == "fineweb" and text_digest(text) in excluded_fineweb_hashes:
                counts["excluded_mixed_lines"] += 1
                counts["excluded_mixed_characters"] += len(text)
                continue
            counts["lines"] += 1
            counts["characters"] += len(text)
    return counts


def selected(probability: float, seed: int, namespace: str, text: str) -> bool:
    return score(seed, namespace, text) < probability


def build_variant(
    ratio: float,
    target_chars: int,
    seed: int,
    base_jsonl: Path,
    mixed_pool: Path,
    base_available: int,
    mixed_available: int,
    excluded_fineweb_hashes: set[bytes],
) -> tuple[dict, set[bytes]]:
    label = f"mixed{int(round(ratio * 100))}"
    output_txt = base_jsonl.parent / f"tokenizer_train_{label}.txt"
    output_jsonl = base_jsonl.parent / f"tokenizer_train_{label}.jsonl"
    mixed_target = int(target_chars * ratio)
    mixed_probability = min(1.0, mixed_target / (mixed_available or 1))
    seen: set[bytes] = set()
    counts = Counter()
    source_counts: dict[str, Counter] = {}
    with output_txt.open("w", encoding="utf-8") as txt, output_jsonl.open("w", encoding="utf-8") as jsonl:
        with mixed_pool.open("r", encoding="utf-8") as handle:
            for raw in handle:
                record = json.loads(raw)
                text = str(record["text"])
                source = str(record["source"])
                if counts["mixed_characters"] >= mixed_target:
                    continue
                if not selected(mixed_probability, seed, f"{label}-mixed", f"{source}\0{text}"):
                    continue
                digest = text_digest(text)
                if digest in seen:
                    continue
                seen.add(digest)
                txt.write(text + "\n")
                jsonl.write(json.dumps({"text": text, "source": source}, ensure_ascii=False) + "\n")
                counts["mixed_lines"] += 1
                counts["mixed_characters"] += len(text)
                source_counts.setdefault(source, Counter())["lines"] += 1
                source_counts[source]["characters"] += len(text)

        korean_target = max(0, target_chars - counts["mixed_characters"])
        korean_probability = min(1.0, korean_target / (base_available or 1))
        with base_jsonl.open("r", encoding="utf-8") as handle:
            for raw in handle:
                record = json.loads(raw)
                text = normalize_line(str(record["text"]))
                source = str(record.get("source", "unknown"))
                digest = text_digest(text)
                if source == "fineweb" and digest in excluded_fineweb_hashes:
                    continue
                if counts["korean_characters"] >= korean_target:
                    continue
                if not selected(korean_probability, seed, f"{label}-korean", f"{source}\0{text}"):
                    continue
                if digest in seen:
                    counts["duplicates_removed"] += 1
                    continue
                seen.add(digest)
                txt.write(text + "\n")
                jsonl.write(json.dumps({"text": text, "source": source}, ensure_ascii=False) + "\n")
                counts["korean_lines"] += 1
                counts["korean_characters"] += len(text)
                source_counts.setdefault(source, Counter())["lines"] += 1
                source_counts[source]["characters"] += len(text)
    counts["total_lines"] = counts["mixed_lines"] + counts["korean_lines"]
    counts["total_characters"] = counts["mixed_characters"] + counts["korean_characters"]
    return {
        "label": label,
        "ratio_target_percent": 100.0 * ratio,
        "target_characters": target_chars,
        "mixed_target_characters": mixed_target,
        "mixed_sampling_probability": mixed_probability,
        "korean_sampling_probability": korean_probability,
        **dict(counts),
        "actual_mixed_percent": 100.0 * counts["mixed_characters"] / (counts["total_characters"] or 1),
        "sources": {name: dict(value) for name, value in source_counts.items()},
        "txt": str(output_txt),
        "jsonl": str(output_jsonl),
    }, seen


def clean_validation(raw_path: Path, output_dir: Path, train_hashes: set[bytes]) -> dict:
    txt_path = output_dir / "mixed_validation.txt"
    jsonl_path = output_dir / "mixed_validation.jsonl"
    seen: set[bytes] = set()
    counts = Counter()
    with raw_path.open("r", encoding="utf-8") as source, txt_path.open("w", encoding="utf-8") as txt, jsonl_path.open("w", encoding="utf-8") as jsonl:
        for raw in source:
            record = json.loads(raw)
            text = str(record["text"])
            digest = text_digest(text)
            counts["input"] += 1
            if digest in train_hashes:
                counts["train_overlap_removed"] += 1
                continue
            if digest in seen:
                counts["duplicates_removed"] += 1
                continue
            seen.add(digest)
            txt.write(text + "\n")
            jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts["retained"] += 1
            counts["characters"] += len(text)
    return dict(counts) | {"txt": str(txt_path), "jsonl": str(jsonl_path)}


def markdown(report: dict) -> str:
    lines = [
        "# Mixed workload corpus report",
        "",
        "> All records come from existing FineWeb or local project files. No template expansion or repeated padding was used.",
        "",
        "## Discovered local sources",
        "",
        f"- Project files: {report['extraction']['discovered'].get('project_files', 0):,}",
        f"- Project bytes: {report['extraction']['discovered'].get('project_file_bytes', 0):,}",
        f"- Training files: {report['extraction']['discovered'].get('project_train_files', 0):,}",
        f"- Validation files: {report['extraction']['discovered'].get('project_validation_files', 0):,}",
        "",
        "## Mixed source statistics",
        "",
        "| Source | Lines | Characters | Duplicates | Hangul | ASCII | English | Digits |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, stats in report["source_statistics"].items():
        extraction = report["extraction"]["extraction_counts"].get(name, {})
        lines.append(
            f"| {name} | {stats['documents_or_lines']:,} | {stats['total_characters']:,} | "
            f"{extraction.get('duplicates', 0):,} | {stats['hangul_ratio_percent']:.2f}% | "
            f"{stats['ascii_ratio_percent']:.2f}% | {stats['english_alphabet_ratio_percent']:.2f}% | "
            f"{stats['digit_ratio_percent']:.2f}% |"
        )
    lines += [
        "",
        f"Globally unique mixed pool: {report['mixed_pool']['total'].get('lines', 0):,} lines / "
        f"{report['mixed_pool']['total'].get('characters', 0):,} characters.",
        "",
        "## Training variants",
        "",
        "| Variant | Total chars | Mixed chars | Actual mixed share | Lines |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in report["variants"]:
        lines.append(
            f"| {item['label']} | {item['total_characters']:,} | {item['mixed_characters']:,} | "
            f"{item['actual_mixed_percent']:.3f}% | {item['total_lines']:,} |"
        )
    lines += [
        "",
        "## Mixed validation",
        "",
        f"Retained {report['validation']['retained']:,} records / {report['validation']['characters']:,} characters after removing "
        f"{report['validation'].get('train_overlap_removed', 0):,} exact train overlaps.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not 0 <= args.ascii_min < args.ascii_max <= 1:
        raise ValueError("ASCII bounds must satisfy 0 <= min < max <= 1")
    if not 0 <= args.english_min <= 1:
        raise ValueError("--english-min must be between 0 and 1")
    if not 0 < args.file_holdout_ratio < 1:
        raise ValueError("--file-holdout-ratio must be between 0 and 1")
    if not args.mixed_ratios or any(not 0 < value < 1 for value in args.mixed_ratios):
        raise ValueError("--mixed-ratios values must be between 0 and 1")

    output_dir = Path(args.output_dir)
    roots = [Path(value) for value in args.code_root] if args.code_root else [PROJECT_ROOT]
    base_jsonl = Path(args.base_jsonl)
    required = [base_jsonl, Path(args.fineweb_train), Path(args.fineweb_holdout)]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    planned = [output_dir / f"{name}.txt" for name in MIXED_SOURCE_NAMES]
    planned += [output_dir / "mixed_pool.jsonl", output_dir / "mixed_validation_raw.jsonl", output_dir / "mixed_validation.txt", output_dir / "mixed_validation.jsonl"]
    for ratio in args.mixed_ratios:
        label = f"mixed{int(round(ratio * 100))}"
        planned += [base_jsonl.parent / f"tokenizer_train_{label}.txt", base_jsonl.parent / f"tokenizer_train_{label}.jsonl"]
    prepare_outputs(planned, args.overwrite)

    print("[Extract] FineWeb mixed documents and local project files", flush=True)
    extraction, validation_raw = extract_sources(args, output_dir, roots)
    print("[Pool] Global exact deduplication", flush=True)
    mixed_pool, pool_stats = build_mixed_pool(output_dir)

    fineweb_mixed_hashes = {
        text_digest(raw.rstrip("\r\n"))
        for raw in (output_dir / "korean_english_mixed.txt").open("r", encoding="utf-8")
        if raw.rstrip("\r\n")
    }
    base_stats = base_corpus_stats(base_jsonl, fineweb_mixed_hashes)
    if args.target_chars:
        target_chars = args.target_chars
    else:
        target_chars = base_stats["characters"] + base_stats["excluded_mixed_characters"]

    variants = []
    train_hash_union: set[bytes] = set()
    for ratio in args.mixed_ratios:
        print(f"[Compose] mixed{int(round(ratio * 100))}", flush=True)
        variant, hashes = build_variant(
            ratio,
            target_chars,
            args.seed,
            base_jsonl,
            mixed_pool,
            base_stats["characters"],
            pool_stats["total"].get("characters", 0),
            fineweb_mixed_hashes,
        )
        variants.append(variant)
        train_hash_union.update(hashes)

    validation = clean_validation(validation_raw, output_dir, train_hash_union)
    source_statistics = {
        name: analyze_text_file(output_dir / f"{name}.txt", lexical=False)
        for name in MIXED_SOURCE_NAMES
    }
    validation["statistics"] = analyze_text_file(validation["txt"], lexical=False)
    report = {
        "configuration": vars(args),
        "code_roots": [str(path) for path in roots],
        "extraction": extraction,
        "source_statistics": source_statistics,
        "mixed_pool": pool_stats,
        "base_korean_pool": dict(base_stats),
        "target_characters": target_chars,
        "variants": variants,
        "validation": validation,
        "synthetic_repetition": False,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.output_markdown).write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"mixed_pool": pool_stats["total"], "variants": variants, "validation": validation}, ensure_ascii=False, indent=2))
    print(f"wrote {args.output_json} and {args.output_markdown}")


if __name__ == "__main__":
    main()
