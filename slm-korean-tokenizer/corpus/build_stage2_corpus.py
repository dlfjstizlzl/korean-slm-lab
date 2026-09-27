#!/usr/bin/env python3
"""Build the Stage-2 continued-pretraining corpus.

Document-level, provenance-preserving, deduplicated, and screened against every
held-out evaluation set. NIKL is blocked at path, source and record level and its
files are never opened.

This script does not train anything.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "corpus"))

from build_recovery_data import assert_record_allowed, assert_source_allowed  # noqa: E402
from stage2_dedup import (  # noqa: E402
    content_hash,
    ngram_duplicate_ratio,
    normalize,
    quality_verdict,
    sha256,
    simhash,
)
# Revision 2 of the skill generator: each plan step carries its own sub-task and
# later steps reference earlier ones instead of restating the query. See
# `corpus/stage2_skill_synth_v2.py` for the diagnosis that motivated it.
from stage2_skill_synth_v2 import (  # noqa: E402
    extract_candidates,
    generate as generate_skill,
    skeleton,
)

PROCESSED = Path("data/processed")

# Fallback mixture, fixed before this stage ran. Used only when the primary
# mixture cannot be met from legitimately available public data.
FALLBACK_SHARES = {
    "korean_general": 0.50,
    "korean_technical": 0.25,
    "dialogue": 0.10,
    "code_url": 0.05,
    "json_structured": 0.05,
    "sql": 0.02,
    "skill_manager": 0.03,
}

# Target mixture. Deviations are reported, never silently applied.
TARGET_SHARES = {
    "korean_general": 0.45,
    "korean_technical": 0.30,
    "dialogue": 0.10,
    "code_url": 0.05,
    "json_structured": 0.05,
    "sql": 0.02,
    "skill_manager": 0.03,
}

# Per-category bounds: (min_chars, max_chars, code_like, max_unbroken_run).
# The unbroken-run ceiling is what separates "minified bundle" from "compact but
# legitimate": prose never has a 400-character word, but whitespace-free JSON
# routinely does, so JSON gets a much higher ceiling and only giant blobs are cut.
LIMITS = {
    "korean_general": (200, 200_000, False, 400),
    "korean_technical": (150, 100_000, False, 400),
    "dialogue": (200, 100_000, False, 400),
    "code_url": (40, 40_000, True, 1_000),
    "json_structured": (40, 40_000, True, 8_000),
    "sql": (20, 40_000, True, 4_000),
    "skill_manager": (20, 20_000, True, 8_000),
}

# Every held-out evaluation surface Stage-2 training must not contain.
EVAL_SOURCES = [
    ("balanced_eval", Path("data/eval/recovery_balanced_eval.jsonl"), "jsonl"),
    ("korean_large_eval", Path("data/eval/recovery_korean_large_eval.jsonl"), "jsonl"),
    ("balanced_eval_decontaminated",
     Path("data/eval/recovery_balanced_eval_decontaminated.jsonl"), "jsonl"),
    ("korean_large_eval_decontaminated",
     Path("data/eval/recovery_korean_large_eval_decontaminated.jsonl"), "jsonl"),
    ("fineweb_holdout", PROCESSED / "fineweb_general_holdout.txt", "txt"),
    ("aihub_document_valid", PROCESSED / "aihub_documents_valid.txt", "txt"),
    ("aihub_document_valid_clean", PROCESSED / "aihub_documents_valid_clean.txt", "txt"),
    ("aihub_dialogue_valid", PROCESSED / "aihub_dialogue_valid_clean.txt", "txt"),
    ("aihub_summary_valid", PROCESSED / "aihub_summaries_valid_clean.txt", "txt"),
    ("aihub_query_valid", PROCESSED / "aihub_queries_valid_clean.txt", "txt"),
    ("mixed_validation", PROCESSED / "mixed/mixed_validation.jsonl", "jsonl"),
    ("structured_valid", PROCESSED / "structured/structured_valid.jsonl", "jsonl"),
    ("json_extra_valid", PROCESSED / "structured/json_extra_valid.jsonl", "jsonl"),
    ("recovery_train", PROCESSED / "recovery_train.jsonl", "jsonl"),
]


# --------------------------------------------------------------------------- #
# Readers (each yields provenance-carrying candidates)
# --------------------------------------------------------------------------- #

def read_txt_documents(path: Path, source: str):
    assert_source_allowed(source, path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for index, raw in enumerate(handle):
            text = raw.rstrip("\n")
            if text.strip():
                yield {"text": text, "source": source, "source_id": f"{source}:{index}"}


def read_jsonl_documents(path: Path, source_filter: set[str] | None, fallback: str):
    assert_source_allowed(path.name, path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for index, raw in enumerate(handle):
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue
            source = str(record.get("source", fallback))
            if source_filter is not None and source not in source_filter:
                continue
            text = record.get("text", "")
            if not text:
                continue
            assert_source_allowed(source, path)
            yield {
                "text": text, "source": source,
                "source_id": str(record.get("source_id") or f"{source}:{index}"),
            }


def category_readers(category: str):
    if category == "korean_general":
        return [
            lambda: read_txt_documents(PROCESSED / "fineweb_general_train.txt", "fineweb2_ko"),
            lambda: read_jsonl_documents(
                Path("data/raw/fineweb_ko_extra.jsonl"), None, "fineweb2_ko_extra"),
        ]
    if category == "korean_technical":
        return [
            lambda: read_jsonl_documents(
                PROCESSED / "stage2_aihub_technical.jsonl", None, "aihub_technical_document"),
            lambda: read_jsonl_documents(
                Path("data/raw/korean_technical_wiki.jsonl"), None, "korean_wikipedia_technical"),
        ]
    if category == "dialogue":
        return [lambda: read_jsonl_documents(
            PROCESSED / "stage2_aihub_dialogue.jsonl", None, "aihub_dialogue_session")]
    if category == "code_url":
        return [
            lambda: read_jsonl_documents(
                PROCESSED / "mixed/mixed_pool.jsonl",
                {"source_code", "identifiers_technical", "urls_api"}, "mixed_pool"),
            lambda: read_txt_documents(PROCESSED / "mixed/cli_paths.txt", "cli_paths"),
        ]
    if category == "json_structured":
        return [
            lambda: read_jsonl_documents(
                PROCESSED / "structured/json_extra_train.jsonl", None, "starcoder_json_extra"),
            lambda: read_jsonl_documents(
                PROCESSED / "structured/structured_train.jsonl",
                {"starcoder_json_pretty", "starcoder_json_compact"}, "starcoder_json"),
            lambda: read_jsonl_documents(
                Path("data/external/json/starcoder_json_extra.jsonl"), None, "starcoder_json_extra"),
            lambda: read_txt_documents(Path("data/external/json/starcoder_json.txt"), "starcoder_json"),
            lambda: read_txt_documents(PROCESSED / "mixed/structured_data.txt", "local_structured"),
            lambda: read_jsonl_documents(
                Path("data/raw/structured_extra.jsonl"), {"starcoderdata_json"}, "starcoderdata_json"),
        ]
    if category == "sql":
        return [
            lambda: read_jsonl_documents(
                PROCESSED / "structured/structured_train.jsonl",
                {"stack_sql", "spider_sql"}, "sql"),
            lambda: read_txt_documents(Path("data/external/sql/spider_sql.txt"), "spider_sql"),
            lambda: read_txt_documents(Path("data/external/sql/stack_sql.txt"), "stack_sql"),
            lambda: read_jsonl_documents(
                Path("data/raw/structured_extra.jsonl"), {"the_stack_smol_sql"}, "the_stack_smol_sql"),
        ]
    raise KeyError(category)


# --------------------------------------------------------------------------- #
# Eval protection
# --------------------------------------------------------------------------- #

def load_eval_hashes() -> tuple[set[str], dict[str, int]]:
    hashes: set[str] = set()
    counts: dict[str, int] = {}
    for name, path, kind in EVAL_SOURCES:
        assert_source_allowed(name, path)
        before = len(hashes)
        if not path.exists():
            counts[name] = 0
            continue
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                raw = raw.rstrip("\n")
                if not raw.strip():
                    continue
                if kind == "jsonl":
                    try:
                        text = json.loads(raw).get("text", "")
                    except json.JSONDecodeError:
                        continue
                else:
                    text = raw
                if text:
                    hashes.add(content_hash(text))
        counts[name] = len(hashes) - before
    return hashes, counts


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/processed/stage2_train.jsonl")
    parser.add_argument("--manifest", default="data/processed/stage2_corpus_manifest.jsonl")
    parser.add_argument("--holdout", default="data/processed/stage2_holdout.jsonl")
    parser.add_argument("--report", default="outputs/stage2_corpus_report.json")
    parser.add_argument("--source-csv", default="outputs/stage2_source_stats.csv")
    parser.add_argument("--target-chars", type=int, default=0,
                        help="0 = build the largest corpus the target mixture allows.")
    parser.add_argument("--minimum-chars", type=int, default=150_000_000)
    parser.add_argument("--holdout-ratio", type=float, default=0.01)
    parser.add_argument("--near-duplicate-sample", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixture", choices=["primary", "fallback"], default="primary",
                        help="'fallback' is the 50/25 mixture fixed before this stage ran.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng_seed = args.seed
    global TARGET_SHARES
    if args.mixture == "fallback":
        TARGET_SHARES = FALLBACK_SHARES
        print("[stage2] using the pre-defined fallback mixture (50/25)", flush=True)

    print("[stage2] hashing held-out evaluation text", flush=True)
    eval_hashes, eval_counts = load_eval_hashes()
    print(f"  {len(eval_hashes):,} evaluation hashes from {len(eval_counts)} sources", flush=True)

    # --- pass 1: collect and screen candidates per category -------------------- #
    pools: dict[str, list[dict]] = {}
    stats: dict[str, Counter] = defaultdict(Counter)
    source_stats: dict[tuple[str, str], Counter] = defaultdict(Counter)
    seen_global: set[str] = set()

    skill_groups_by_source: dict[str, list[dict]] = {}

    for category in TARGET_SHARES:
        if category == "skill_manager":
            continue
        print(f"[stage2] reading {category}", flush=True)
        pool: list[dict] = []
        min_chars, max_chars, code_like, max_run = LIMITS[category]
        seen_local: set[str] = set()
        for reader in category_readers(category):
            for candidate in reader():
                text = normalize(candidate["text"])
                source = candidate["source"]
                stats[category]["candidates"] += 1
                source_stats[(category, source)]["candidates"] += 1

                reason = quality_verdict(text, min_chars=min_chars, max_chars=max_chars,
                                         is_code_like=code_like, max_token_length=max_run)
                if reason:
                    stats[category][f"rejected_{reason}"] += 1
                    source_stats[(category, source)][f"rejected_{reason}"] += 1
                    continue

                digest = content_hash(text)
                if digest in seen_local:
                    stats[category]["dedup_source_internal"] += 1
                    continue
                if digest in seen_global:
                    stats[category]["dedup_cross_source"] += 1
                    continue
                if digest in eval_hashes:
                    stats[category]["dropped_eval_overlap"] += 1
                    source_stats[(category, source)]["dropped_eval_overlap"] += 1
                    continue
                seen_local.add(digest)
                seen_global.add(digest)
                assert_record_allowed(text, source)
                pool.append({
                    "text": text, "source": source, "source_id": candidate["source_id"],
                    "category": category, "chars": len(text), "hash": digest,
                })
                stats[category]["accepted"] += 1
                stats[category]["accepted_chars"] += len(text)
                source_stats[(category, source)]["accepted"] += 1
                source_stats[(category, source)]["accepted_chars"] += len(text)
                if category == "korean_technical":
                    bucket = skill_groups_by_source.setdefault(source, [])
                    if len(bucket) < 40000:
                        candidates = extract_candidates(text)
                        if candidates:
                            bucket.append({"candidates": candidates, "source": source,
                                           "doc_id": candidate["source_id"]})
        pools[category] = pool
        print(f"  accepted {len(pool):,} docs, {stats[category]['accepted_chars']:,} chars", flush=True)

    # --- decide the achievable total ------------------------------------------ #
    supply = {category: stats[category]["accepted_chars"] for category in pools}
    caps = {category: supply[category] / TARGET_SHARES[category] for category in pools}
    # Skill text is synthetic, so it never binds the total.
    binding = min(caps, key=caps.get)
    achievable = int(caps[binding])
    total_target = args.target_chars or achievable
    total_target = min(total_target, achievable)

    print(f"[stage2] binding category: {binding}; achievable total {achievable:,} chars", flush=True)

    # --- synthesize the skill category to its share --------------------------- #
    skill_budget = int(total_target * TARGET_SHARES["skill_manager"])
    # Deterministic stratified interleave: sources take turns, so a larger
    # technical source cannot crowd the smaller one out of the query pool.
    streams = [skill_groups_by_source[name] for name in sorted(skill_groups_by_source)]
    skill_groups: list[dict] = []
    position = 0
    while any(position < len(stream) for stream in streams):
        for stream in streams:
            if position < len(stream):
                skill_groups.append(stream[position])
        position += 1
    skill_groups = skill_groups[:60000]
    print(f"[stage2] synthesizing skill_manager ({skill_budget:,} chars) from "
          f"{len(skill_groups):,} candidate groups across {len(streams)} sources", flush=True)
    skill_records = generate_skill(skill_groups, skill_budget, seed=rng_seed)
    skill_pool = []
    skill_seen: set[str] = set()
    for record in skill_records:
        text = normalize(record["text"])
        digest = content_hash(text)
        if digest in skill_seen or digest in seen_global or digest in eval_hashes:
            stats["skill_manager"]["dedup_source_internal"] += 1
            continue
        skill_seen.add(digest)
        seen_global.add(digest)
        skill_pool.append({
            "text": text, "source": record["source"], "source_id": record["source_id"],
            "category": "skill_manager", "chars": len(text), "hash": digest,
            "shape": record["shape"], "serialization": record.get("serialization"),
            "query_source": record.get("query_source"),
        })
        stats["skill_manager"]["accepted"] += 1
        stats["skill_manager"]["accepted_chars"] += len(text)
        source_stats[("skill_manager", record["source"])]["accepted"] += 1
        source_stats[("skill_manager", record["source"])]["accepted_chars"] += len(text)
    pools["skill_manager"] = skill_pool

    # --- cut each category to its share, document-level, deterministic --------- #
    train_path, holdout_path = Path(args.out), Path(args.holdout)
    manifest_path = Path(args.manifest)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    selected_counts: dict[str, Counter] = defaultdict(Counter)
    near_dup_hashes: dict[str, list[int]] = defaultdict(list)
    repetition: dict[str, list[tuple[float, float]]] = defaultdict(list)
    duplicate_line_counts: dict[str, Counter] = defaultdict(Counter)

    with train_path.open("w", encoding="utf-8") as train_handle, \
            holdout_path.open("w", encoding="utf-8") as holdout_handle, \
            manifest_path.open("w", encoding="utf-8") as manifest_handle:
        for category, pool in pools.items():
            budget = int(total_target * TARGET_SHARES[category])
            rng = random.Random(f"{rng_seed}:{category}")
            ordered = sorted(pool, key=lambda item: item["hash"])  # deterministic
            rng.shuffle(ordered)
            written = 0
            for position, record in enumerate(ordered):
                if written >= budget:
                    break
                # Document-level split: the whole document goes to one side.
                is_holdout = rng.random() < args.holdout_ratio
                payload = {
                    "source": record["source"],
                    "source_id": record["source_id"],
                    "category": category,
                    "text": record["text"],
                    "chars": record["chars"],
                    "sha256": sha256(record["text"]),
                }
                for field in ("shape", "serialization", "query_source"):
                    if record.get(field):
                        payload[field] = record[field]
                handle = holdout_handle if is_holdout else train_handle
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                manifest_handle.write(json.dumps({
                    "sha256": payload["sha256"], "category": category,
                    "source": record["source"], "source_id": record["source_id"],
                    "chars": record["chars"], "split": "holdout" if is_holdout else "train",
                }, ensure_ascii=False) + "\n")
                if is_holdout:
                    selected_counts[category]["holdout_docs"] += 1
                    selected_counts[category]["holdout_chars"] += record["chars"]
                else:
                    selected_counts[category]["train_docs"] += 1
                    selected_counts[category]["train_chars"] += record["chars"]
                    written += record["chars"]
                if len(near_dup_hashes[category]) < args.near_duplicate_sample and position % 3 == 0:
                    near_dup_hashes[category].append(simhash(record["text"]))
                if len(repetition[category]) < 5000:
                    repetition[category].append((
                        ngram_duplicate_ratio(record["text"], 4),
                        ngram_duplicate_ratio(record["text"], 8),
                    ))
                lines = [line.strip() for line in record["text"].split("\n") if line.strip()]
                if lines:
                    duplicate_line_counts[category]["lines"] += len(lines)
                    duplicate_line_counts[category]["unique_lines"] += len(set(lines))

    # --- statistics ------------------------------------------------------------ #
    from stage2_dedup import near_duplicate_estimate

    total_train_chars = sum(selected_counts[c]["train_chars"] for c in selected_counts)
    total_train_docs = sum(selected_counts[c]["train_docs"] for c in selected_counts)

    skeletons = Counter(skeleton(record["text"]) for record in pools["skill_manager"])
    report = {
        "stage": "stage2_corpus",
        "training_performed": False,
        "seed": rng_seed,
        "mixture": args.mixture,
        "target_shares": TARGET_SHARES,
        "supply": {c: supply.get(c, stats[c]["accepted_chars"]) for c in TARGET_SHARES},
        "max_total_at_target_shares": achievable,
        "binding_category": binding,
        "minimum_required_chars": args.minimum_chars,
        "meets_minimum": total_train_chars >= args.minimum_chars,
        "shortfall_chars": max(0, args.minimum_chars - total_train_chars),
        "totals": {
            "train_documents": total_train_docs,
            "train_characters": total_train_chars,
            "holdout_documents": sum(selected_counts[c]["holdout_docs"] for c in selected_counts),
            "holdout_characters": sum(selected_counts[c]["holdout_chars"] for c in selected_counts),
        },
        "by_category": {
            category: {
                "train_documents": selected_counts[category]["train_docs"],
                "train_characters": selected_counts[category]["train_chars"],
                "char_share": selected_counts[category]["train_chars"] / total_train_chars
                if total_train_chars else 0.0,
                "target_share": TARGET_SHARES[category],
                "holdout_documents": selected_counts[category]["holdout_docs"],
                "supply_characters": supply.get(category, stats[category]["accepted_chars"]),
                "supply_utilization": (
                    selected_counts[category]["train_chars"]
                    / supply.get(category, stats[category]["accepted_chars"])
                    if supply.get(category, stats[category]["accepted_chars"]) else 0.0
                ),
            }
            for category in TARGET_SHARES
        },
        "filters": {category: dict(counter) for category, counter in stats.items()},
        "eval_protection": {
            "hashes": len(eval_hashes),
            "by_source": eval_counts,
            "dropped_by_category": {
                category: stats[category]["dropped_eval_overlap"] for category in stats
            },
        },
        "repetition": {
            category: {
                "documents_sampled": len(values),
                "mean_duplicate_4gram": sum(v[0] for v in values) / len(values) if values else 0.0,
                "mean_duplicate_8gram": sum(v[1] for v in values) / len(values) if values else 0.0,
                "duplicate_line_ratio": (
                    1.0 - duplicate_line_counts[category]["unique_lines"]
                    / duplicate_line_counts[category]["lines"]
                ) if duplicate_line_counts[category]["lines"] else 0.0,
            }
            for category, values in repetition.items()
        },
        "near_duplicates": {
            category: near_duplicate_estimate(values)
            for category, values in near_dup_hashes.items()
        },
        "skill_synthesis": {
            "generator": "stage2_skill_synth_v2",
            "query_sources": {name: len(groups)
                              for name, groups in sorted(skill_groups_by_source.items())},
            "candidate_groups": len(skill_groups),
            "serializations": dict(Counter(r.get("serialization", "?")
                                           for r in pools["skill_manager"])),
            "query_source_mix": dict(Counter(r.get("query_source", "?")
                                             for r in pools["skill_manager"])),
            "documents": len(pools["skill_manager"]),
            "distinct_skeletons": len(skeletons),
            "top_skeleton_share": (skeletons.most_common(1)[0][1] / len(pools["skill_manager"]))
            if pools["skill_manager"] else 0.0,
            "shapes": dict(Counter(r.get("shape", "?") for r in pools["skill_manager"])),
        },
        "nikl": {
            "used": False,
            "files_opened": 0,
            "guard": "assert_source_allowed blocks nikl/newspaper/nirw/nwrw in every path and "
                     "source; assert_record_allowed rejects NIRW/NWRW article ids; no NIKL path "
                     "appears in any reader.",
        },
        "outputs": {"train": args.out, "holdout": args.holdout, "manifest": args.manifest},
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with Path(args.source_csv).open("w", encoding="utf-8") as handle:
        handle.write("category,source,candidates,accepted,accepted_chars,dropped_eval_overlap\n")
        for (category, source), counter in sorted(source_stats.items()):
            handle.write(f"{category},{source},{counter['candidates']},{counter['accepted']},"
                         f"{counter['accepted_chars']},{counter['dropped_eval_overlap']}\n")

    print(json.dumps({
        "train_characters": total_train_chars,
        "train_documents": total_train_docs,
        "meets_minimum": report["meets_minimum"],
        "shortfall_chars": report["shortfall_chars"],
        "binding_category": binding,
        "by_category": {c: v["char_share"] for c, v in report["by_category"].items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
