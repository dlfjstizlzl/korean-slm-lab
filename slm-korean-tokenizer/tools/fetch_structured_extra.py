#!/usr/bin/env python3
"""Top up the JSON and SQL categories for the Stage-2 corpus.

Source: `bigcode/the-stack-smol` — permissively-licensed source files from The
Stack. Access required accepting the dataset's terms, which this account had
already done; nothing was bypassed.

Only the small shortfalls are taken. JSON and SQL are not inflated past their
mixture shares, and nothing is repeated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "corpus"))

from build_recovery_data import assert_record_allowed, assert_source_allowed  # noqa: E402
from stage2_dedup import content_hash, normalize, quality_verdict  # noqa: E402

DATASET = "bigcode/the-stack-smol"
JSON_DATASET = "bigcode/starcoderdata"   # the provider this project already used for JSON
LICENSE = ("The Stack (permissive subset); per-file licenses retained in the source "
           "dataset. Access gated by terms acceptance, already granted for this account.")

# Language directory -> (category, min_chars, max_unbroken_run)
PLAN = {
    "sql": ("sql", 20, 4_000),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/raw/structured_extra.jsonl")
    parser.add_argument("--report", default="outputs/structured_extra_acquisition.json")
    parser.add_argument("--sql-target", type=int, default=1_200_000)
    parser.add_argument("--json-target", type=int, default=3_000_000)
    parser.add_argument("--max-chars", type=int, default=40_000)
    return parser.parse_args()


def load_known() -> set[str]:
    known: set[str] = set()
    paths = [
        PROJECT_ROOT / "data/processed/stage2_train.jsonl",
        PROJECT_ROOT / "data/processed/structured/json_extra_train.jsonl",
        PROJECT_ROOT / "data/processed/structured/structured_train.jsonl",
        PROJECT_ROOT / "data/processed/structured/structured_valid.jsonl",
        PROJECT_ROOT / "data/processed/structured/json_extra_valid.jsonl",
        PROJECT_ROOT / "data/eval/recovery_balanced_eval.jsonl",
    ]
    for path in paths:
        if not path.exists():
            continue
        assert_source_allowed(path.name, path)
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    try:
                        known.add(content_hash(json.loads(line)["text"]))
                    except (json.JSONDecodeError, KeyError):
                        continue
    for name in ("data/external/json/starcoder_json.txt",
                 "data/external/sql/spider_sql.txt", "data/external/sql/stack_sql.txt"):
        path = PROJECT_ROOT / name
        if path.exists():
            with path.open(encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if line.strip():
                        known.add(content_hash(line.rstrip("\n")))
    return known


def main() -> None:
    args = parse_args()
    from huggingface_hub import hf_hub_download

    targets = {"sql": args.sql_target}
    known = load_known()
    print(f"[structured] {len(known):,} known hashes", flush=True)

    counts: dict[str, dict] = {}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rejections: dict[str, dict] = {}

    with out_path.open("w", encoding="utf-8") as handle:
        for language, (category, min_chars, max_run) in PLAN.items():
            stats = {"scanned": 0, "accepted": 0, "accepted_chars": 0, "duplicates": 0}
            reasons: dict[str, int] = {}
            try:
                local = hf_hub_download(DATASET, f"data/{language}/data.json", repo_type="dataset")
            except Exception as error:  # noqa: BLE001
                print(f"[structured] {language}: UNAVAILABLE ({type(error).__name__})", flush=True)
                counts[language] = {**stats, "status": "unavailable",
                                    "error": f"{type(error).__name__}: {error}"}
                continue
            print(f"[structured] reading {language}", flush=True)
            # These files are JSON Lines despite the .json extension.
            with open(local, encoding="utf-8", errors="replace") as source:
                payload = [json.loads(line) for line in source if line.strip()]
            for row in payload:
                if stats["accepted_chars"] >= targets[language]:
                    break
                stats["scanned"] += 1
                text = normalize(row.get("content") or "")
                reason = quality_verdict(text, min_chars=min_chars, max_chars=args.max_chars,
                                         is_code_like=True, max_token_length=max_run)
                if reason:
                    reasons[reason] = reasons.get(reason, 0) + 1
                    continue
                digest = content_hash(text)
                if digest in known:
                    stats["duplicates"] += 1
                    continue
                known.add(digest)
                assert_record_allowed(text, f"the_stack_smol_{language}")
                handle.write(json.dumps({
                    "text": text,
                    "source": f"the_stack_smol_{language}",
                    "source_id": str(row.get("hexsha") or digest[:16]),
                    "dataset": DATASET,
                    "category": category,
                    "repository": row.get("max_stars_repo_name"),
                    "path": row.get("max_stars_repo_path"),
                    "license": LICENSE,
                    "chars": len(text),
                    "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                }, ensure_ascii=False) + "\n")
                stats["accepted"] += 1
                stats["accepted_chars"] += len(text)
            stats["status"] = "ok"
            counts[language] = stats
            rejections[language] = reasons
            print(f"  {language}: {stats['accepted']:,} accepted, "
                  f"{stats['accepted_chars']:,} chars", flush=True)

    # --- JSON from starcoderdata (the-stack-smol has no JSON subset) ---------- #
    json_stats = {"scanned": 0, "accepted": 0, "accepted_chars": 0, "duplicates": 0}
    json_reasons: dict[str, int] = {}
    json_shards: list[str] = []
    try:
        from huggingface_hub import list_repo_files
        import pyarrow.parquet as pq

        shards = sorted(name for name in list_repo_files(JSON_DATASET, repo_type="dataset")
                        if name.startswith("json/") and name.endswith(".parquet"))
        with out_path.open("a", encoding="utf-8") as handle:
            for shard in shards:
                if json_stats["accepted_chars"] >= args.json_target:
                    break
                print(f"[structured] downloading {JSON_DATASET}/{shard}", flush=True)
                local = hf_hub_download(JSON_DATASET, shard, repo_type="dataset")
                json_shards.append(shard)
                for batch in pq.ParquetFile(local).iter_batches(batch_size=2000):
                    for row in batch.to_pylist():
                        if json_stats["accepted_chars"] >= args.json_target:
                            break
                        json_stats["scanned"] += 1
                        text = normalize(row.get("content") or "")
                        reason = quality_verdict(text, min_chars=40, max_chars=args.max_chars,
                                                 is_code_like=True, max_token_length=8_000)
                        if reason:
                            json_reasons[reason] = json_reasons.get(reason, 0) + 1
                            continue
                        digest = content_hash(text)
                        if digest in known:
                            json_stats["duplicates"] += 1
                            continue
                        known.add(digest)
                        assert_record_allowed(text, "starcoderdata_json")
                        handle.write(json.dumps({
                            "text": text, "source": "starcoderdata_json",
                            "source_id": str(row.get("hexsha") or digest[:16]),
                            "dataset": JSON_DATASET, "category": "json_structured",
                            "repository": row.get("max_stars_repo_name"),
                            "path": row.get("max_stars_repo_path"),
                            "license": LICENSE, "chars": len(text),
                            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        }, ensure_ascii=False) + "\n")
                        json_stats["accepted"] += 1
                        json_stats["accepted_chars"] += len(text)
                    if json_stats["scanned"] % 20000 < 2000:
                        print(f"  json scanned {json_stats['scanned']:,} accepted "
                              f"{json_stats['accepted_chars']:,} chars", flush=True)
        json_stats["status"] = "ok"
    except Exception as error:  # noqa: BLE001
        json_stats["status"] = "unavailable"
        json_stats["error"] = f"{type(error).__name__}: {error}"
        print(f"[structured] json UNAVAILABLE: {error}", flush=True)
    counts["json"] = json_stats
    rejections["json"] = json_reasons

    report = {
        "dataset": DATASET, "json_dataset": JSON_DATASET, "json_shards": json_shards,
        "provider": "BigCode",
        "access": "gated by terms acceptance; already granted for this account (not bypassed)",
        "license": LICENSE, "targets": targets,
        "by_language": counts, "rejections": rejections, "output": str(out_path),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
