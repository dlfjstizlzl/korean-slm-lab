#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import tempfile
import time
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from corpus.loaders.aihub import discover_aihub_sources, extract_texts, iter_aihub_objects
from corpus.loaders.fineweb import discover_fineweb_files, is_holdout, iter_fineweb_documents
from corpus.loaders.nikl import discover_nikl_files
from corpus.stats import analyze_text_file

KINDS = ("dialogue", "documents", "queries", "summaries")


class DedupStore:
    def __init__(self, path: Path):
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute(
            "CREATE TABLE source_seen (source TEXT NOT NULL, text TEXT NOT NULL, PRIMARY KEY(source, text)) WITHOUT ROWID"
        )
        self.connection.execute("CREATE TABLE global_seen (text TEXT PRIMARY KEY, first_source TEXT NOT NULL) WITHOUT ROWID")
        self.pending = 0

    def add(self, source: str, text: str) -> tuple[bool, str | None]:
        cursor = self.connection.execute("INSERT OR IGNORE INTO source_seen(source, text) VALUES (?, ?)", (source, text))
        internal_unique = cursor.rowcount == 1
        existing = self.connection.execute("SELECT first_source FROM global_seen WHERE text = ?", (text,)).fetchone()
        first_source = existing[0] if existing else None
        if existing is None:
            self.connection.execute("INSERT INTO global_seen(text, first_source) VALUES (?, ?)", (text, source))
        self.pending += 1
        if self.pending >= 10_000:
            self.connection.commit()
            self.pending = 0
        return internal_unique, first_source

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a provenance-preserving real Korean tokenizer corpus.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--target-chars", type=int, default=100_000_000)
    parser.add_argument("--fineweb-ratio", type=float, default=0.55)
    parser.add_argument("--technical-ratio", type=float, default=0.25)
    parser.add_argument("--dialogue-ratio", type=float, default=0.12)
    parser.add_argument("--query-summary-ratio", type=float, default=0.08)
    parser.add_argument("--fineweb-holdout-ratio", type=float, default=0.05)
    parser.add_argument("--min-fineweb-chars", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-json", default="outputs/corpus_report.json")
    parser.add_argument("--output-markdown", default="outputs/CORPUS_REPORT.md")
    return parser.parse_args()


def ensure_outputs(paths: list[Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to replace existing outputs: {existing[:8]}; pass --overwrite")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def selection_probability(text: str, seed: int, source: str) -> float:
    digest = hashlib.blake2b(f"{seed}\0{source}\0{text}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def write_line(handle, text: str) -> None:
    handle.write(text.replace("\n", " ") + "\n")


def build_markdown(report: dict) -> str:
    lines = [
        "# Real corpus report",
        "",
        "> No synthetic repetition was used. NIKL is optional and was not required.",
        "",
        "## Source statistics",
        "",
        "| Source | Lines/docs | Characters | Unique | Duplicate ratio | Hangul | ASCII | Avg chars | p95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, stats in report["source_statistics"].items():
        lines.append(
            f"| {name} | {stats['documents_or_lines']:,} | {stats['total_characters']:,} | {stats['unique_lines']:,} | "
            f"{stats['duplicate_ratio_percent']:.2f}% | {stats['hangul_ratio_percent']:.2f}% | "
            f"{stats['ascii_ratio_percent']:.2f}% | {stats['average_line_characters']:.1f} | {stats['p95_line_characters']:,} |"
        )
    lines += ["", "## Final training mix", ""]
    lines += ["| Source | Lines | Characters | Share |", "|---|---:|---:|---:|"]
    for name, values in report["final_corpus"]["sources"].items():
        lines.append(f"| {name} | {values['lines']:,} | {values['characters']:,} | {values['character_share_percent']:.2f}% |")
    lines += [
        "",
        f"Total: {report['final_corpus']['lines']:,} lines / {report['final_corpus']['characters']:,} characters.",
        "",
        "## Duplicate statistics",
        "",
        f"- Internal duplicates removed: {report['duplicates']['internal_removed']:,}",
        f"- Cross-source duplicates observed: {report['duplicates']['cross_source_observed']:,}",
        f"- Final cross-source duplicates removed: {report['duplicates']['final_removed']:,}",
        "",
        "### Internal exact duplicates by raw source",
        "",
        "| Source | Raw items | Removed | Removal rate |",
        "|---|---:|---:|---:|",
    ]
    for name, removed in report["duplicates"]["internal_by_source"].items():
        raw = report["raw_counts"][name]
        lines.append(f"| {name} | {raw:,} | {removed:,} | {100.0 * removed / (raw or 1):.2f}% |")
    lines += [
        "",
        "### AI Hub validation exact-overlap removal",
        "",
        "| Type | Input | Removed as train overlap | Retained |",
        "|---|---:|---:|---:|",
    ]
    for name, values in report["duplicates"]["validation_exact_train_overlap"].items():
        lines.append(
            f"| {name} | {values['input']:,} | {values['removed']:,} | {values['retained']:,} |"
        )
    lines += [
        "",
        "## Validation scope",
        "",
        "FineWeb holdout is deterministic and document-level for the available blank-line-delimited export. AI Hub VL is in-domain validation. Neither is a source-disjoint external Korean benchmark.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    ratios = [args.fineweb_ratio, args.technical_ratio, args.dialogue_ratio, args.query_summary_ratio]
    if abs(sum(ratios) - 1.0) > 1e-9 or any(value < 0 for value in ratios):
        raise ValueError("Sampling ratios must be non-negative and sum to 1.0")
    if not 0 < args.fineweb_holdout_ratio < 1:
        raise ValueError("--fineweb-holdout-ratio must be between 0 and 1")

    processed = Path(args.processed_dir)
    outputs = {
        "fineweb_all": processed / "fineweb_general.txt",
        "fineweb_train": processed / "fineweb_general_train.txt",
        "fineweb_holdout": processed / "fineweb_general_holdout.txt",
    }
    for split in ("train", "valid"):
        for kind in KINDS:
            outputs[f"aihub_{kind}_{split}"] = processed / f"aihub_{kind}_{split}.txt"
            if split == "valid":
                outputs[f"aihub_{kind}_{split}_clean"] = processed / f"aihub_{kind}_{split}_clean.txt"
    outputs["final_txt"] = processed / "tokenizer_train_real.txt"
    outputs["final_jsonl"] = processed / "tokenizer_train_real.jsonl"
    ensure_outputs(list(outputs.values()), args.overwrite)

    fineweb_files = discover_fineweb_files(args.data_dir)
    aihub_sources = discover_aihub_sources(args.data_dir)
    if not fineweb_files:
        raise FileNotFoundError("No FineWeb source was detected")
    if not any(source.split == "train" for source in aihub_sources):
        raise FileNotFoundError("No AI Hub TL session source was detected")

    duplicate_counts = Counter()
    raw_counts = Counter()
    aihub_counts = Counter()
    cross_pairs = Counter()
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="corpus-dedup-", dir=processed) as temp_dir:
        dedup = DedupStore(Path(temp_dir) / "dedup.sqlite")
        with ExitStack() as stack:
            handles = {
                name: stack.enter_context(path.open("w", encoding="utf-8"))
                for name, path in outputs.items()
                if name not in {"final_txt", "final_jsonl"} and not name.endswith("_valid_clean")
            }
            for index, (_, text) in enumerate(iter_fineweb_documents(fineweb_files, args.min_fineweb_chars), 1):
                raw_counts["fineweb"] += 1
                unique, first = dedup.add("fineweb", text)
                if not unique:
                    duplicate_counts["fineweb"] += 1
                    continue
                if first and first != "fineweb":
                    cross_pairs["|".join(sorted((first, "fineweb")))] += 1
                write_line(handles["fineweb_all"], text)
                split = "fineweb_holdout" if is_holdout(text, args.fineweb_holdout_ratio, args.seed) else "fineweb_train"
                write_line(handles[split], text)
                if index % 5_000 == 0:
                    print(f"[FineWeb] processed {index:,} source documents", flush=True)

            for source in aihub_sources:
                file_counter = 0
                for _, payload in iter_aihub_objects(source):
                    file_counter += 1
                    raw_counts[f"aihub_json_{source.split}"] += 1
                    extracted, counts = extract_texts(payload)
                    for key, value in counts.items():
                        aihub_counts[f"{source.split}_{key}"] += value
                    for kind, values in extracted.items():
                        source_name = f"aihub_{kind}_{source.split}"
                        for text in values:
                            raw_counts[source_name] += 1
                            unique, first = dedup.add(source_name, text)
                            if not unique:
                                duplicate_counts[source_name] += 1
                                continue
                            if first and first != source_name:
                                cross_pairs["|".join(sorted((first, source_name)))] += 1
                            write_line(handles[source_name], text)
                    if file_counter % 1_000 == 0:
                        print(
                            f"[AIHub {source.split} session{source.session}] processed {file_counter:,} / {source.file_count:,} JSON",
                            flush=True,
                        )
        dedup.close()

    print("[Stats] analyzing processed sources", flush=True)
    stats_files = {
        "fineweb": outputs["fineweb_all"],
        "fineweb_train": outputs["fineweb_train"],
        "fineweb_holdout": outputs["fineweb_holdout"],
        "aihub_dialogue_train": outputs["aihub_dialogue_train"],
        "aihub_documents_train": outputs["aihub_documents_train"],
        "aihub_queries_train": outputs["aihub_queries_train"],
        "aihub_summaries_train": outputs["aihub_summaries_train"],
        "aihub_dialogue_valid": outputs["aihub_dialogue_valid"],
        "aihub_documents_valid": outputs["aihub_documents_valid"],
        "aihub_queries_valid": outputs["aihub_queries_valid"],
        "aihub_summaries_valid": outputs["aihub_summaries_valid"],
    }
    source_stats = {}
    for name, path in stats_files.items():
        print(f"[Stats] {name}", flush=True)
        source_stats[name] = analyze_text_file(path, lexical=True)

    available = {
        "fineweb": source_stats["fineweb_train"]["total_characters"],
        "aihub_document": source_stats["aihub_documents_train"]["total_characters"],
        "aihub_dialogue": source_stats["aihub_dialogue_train"]["total_characters"],
        "aihub_query": source_stats["aihub_queries_train"]["total_characters"],
        "aihub_summary": source_stats["aihub_summaries_train"]["total_characters"],
    }
    qs_total = available["aihub_query"] + available["aihub_summary"]
    quotas = {
        "fineweb": int(args.target_chars * args.fineweb_ratio),
        "aihub_document": int(args.target_chars * args.technical_ratio),
        "aihub_dialogue": int(args.target_chars * args.dialogue_ratio),
        "aihub_query": int(args.target_chars * args.query_summary_ratio * available["aihub_query"] / (qs_total or 1)),
        "aihub_summary": int(args.target_chars * args.query_summary_ratio * available["aihub_summary"] / (qs_total or 1)),
    }
    source_paths = {
        "fineweb": outputs["fineweb_train"],
        "aihub_document": outputs["aihub_documents_train"],
        "aihub_dialogue": outputs["aihub_dialogue_train"],
        "aihub_query": outputs["aihub_queries_train"],
        "aihub_summary": outputs["aihub_summaries_train"],
    }
    final_counts: dict[str, Counter] = {source: Counter() for source in source_paths}
    final_seen: set[bytes] = set()
    final_removed = 0
    with outputs["final_txt"].open("w", encoding="utf-8") as txt, outputs["final_jsonl"].open("w", encoding="utf-8") as jsonl:
        for source, path in source_paths.items():
            probability = min(1.0, quotas[source] / (available[source] or 1))
            with path.open("r", encoding="utf-8") as handle:
                for raw in handle:
                    text = raw.rstrip("\r\n")
                    if not text or selection_probability(text, args.seed, source) >= probability:
                        continue
                    if final_counts[source]["characters"] >= quotas[source]:
                        continue
                    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()
                    if digest in final_seen:
                        final_removed += 1
                        continue
                    final_seen.add(digest)
                    write_line(txt, text)
                    jsonl.write(json.dumps({"text": text, "source": source}, ensure_ascii=False) + "\n")
                    final_counts[source]["lines"] += 1
                    final_counts[source]["characters"] += len(text)
            print(
                f"[Sample] {source}: {final_counts[source]['characters']:,} / {quotas[source]:,} target chars",
                flush=True,
            )

    final_total_chars = sum(value["characters"] for value in final_counts.values())
    final_total_lines = sum(value["lines"] for value in final_counts.values())

    validation_leakage = {}
    with tempfile.TemporaryDirectory(prefix="validation-dedup-", dir=processed) as temp_dir:
        connection = sqlite3.connect(Path(temp_dir) / "train.sqlite")
        connection.execute("CREATE TABLE train_text (text TEXT PRIMARY KEY) WITHOUT ROWID")
        with outputs["final_txt"].open("r", encoding="utf-8") as handle:
            batch = []
            for raw in handle:
                text = raw.rstrip("\r\n")
                if text:
                    batch.append((text,))
                if len(batch) >= 10_000:
                    connection.executemany("INSERT OR IGNORE INTO train_text(text) VALUES (?)", batch)
                    batch.clear()
            if batch:
                connection.executemany("INSERT OR IGNORE INTO train_text(text) VALUES (?)", batch)
        connection.commit()
        for kind in KINDS:
            source_path = outputs[f"aihub_{kind}_valid"]
            clean_path = outputs[f"aihub_{kind}_valid_clean"]
            counts = Counter()
            with source_path.open("r", encoding="utf-8") as source_handle, clean_path.open("w", encoding="utf-8") as clean_handle:
                for raw in source_handle:
                    text = raw.rstrip("\r\n")
                    if not text:
                        continue
                    counts["input"] += 1
                    if connection.execute("SELECT 1 FROM train_text WHERE text = ?", (text,)).fetchone():
                        counts["removed"] += 1
                        continue
                    write_line(clean_handle, text)
                    counts["retained"] += 1
            validation_leakage[kind] = dict(counts)
        connection.close()

    for kind in KINDS:
        name = f"aihub_{kind}_valid_clean"
        print(f"[Stats] {name}", flush=True)
        source_stats[name] = analyze_text_file(outputs[name], lexical=True)
    final_sources = {
        source: {
            "lines": values["lines"],
            "characters": values["characters"],
            "available_characters": available[source],
            "target_characters": quotas[source],
            "character_share_percent": 100.0 * values["characters"] / (final_total_chars or 1),
        }
        for source, values in final_counts.items()
    }
    report = {
        "configuration": vars(args),
        "discovered": {
            "fineweb_files": [str(path) for path in fineweb_files],
            "aihub_sources": [source.__dict__ | {"path": str(source.path)} for source in aihub_sources],
            "nikl_files": [str(path) for path in discover_nikl_files(args.data_dir)],
        },
        "raw_counts": dict(raw_counts),
        "aihub_semantic_counts": dict(aihub_counts),
        "source_statistics": source_stats,
        "duplicates": {
            "internal_by_source": dict(duplicate_counts),
            "internal_removed": sum(duplicate_counts.values()),
            "cross_source_pairs": dict(cross_pairs),
            "cross_source_observed": sum(cross_pairs.values()),
            "final_removed": final_removed,
            "validation_exact_train_overlap": validation_leakage,
        },
        "final_corpus": {
            "txt": str(outputs["final_txt"]),
            "jsonl": str(outputs["final_jsonl"]),
            "lines": final_total_lines,
            "characters": final_total_chars,
            "sources": final_sources,
            "synthetic_repetition": False,
        },
        "validation": {
            "fineweb_holdout": str(outputs["fineweb_holdout"]),
            "aihub_valid_is_in_domain": True,
            "aihub_valid_clean_files": {kind: str(outputs[f"aihub_{kind}_valid_clean"]) for kind in KINDS},
            "fineweb_holdout_is_in_domain": True,
            "independent_eval": "data/korean_eval.txt",
            "independent_eval_warning": "Only 40 curated sentences; too small for a final generalization claim.",
        },
        "elapsed_seconds": time.time() - started,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.output_markdown).write_text(build_markdown(report), encoding="utf-8")
    print(json.dumps({"final_corpus": report["final_corpus"], "duplicates": report["duplicates"]}, ensure_ascii=False, indent=2))
    print(f"wrote {args.output_json} and {args.output_markdown}")


if __name__ == "__main__":
    main()
