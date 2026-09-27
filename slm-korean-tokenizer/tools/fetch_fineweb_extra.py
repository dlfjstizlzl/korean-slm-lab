#!/usr/bin/env python3
"""Fetch additional FineWeb2 Korean documents for the Stage-2 corpus.

Same dataset, config and split the project already used
(`HuggingFaceFW/fineweb-2`, `kor_Hang`, `train`) — only later shards, because
the original 50M-character slice was taken from the start of the stream.

`kor_Hang_removed` (the deduplication-rejected portion) is deliberately not used.
Downloading stops as soon as the usable-character target is met.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "corpus"))

from build_recovery_data import assert_source_allowed  # noqa: E402
from stage2_dedup import content_hash, normalize, quality_verdict  # noqa: E402

DATASET = "HuggingFaceFW/fineweb-2"
CONFIG = "kor_Hang"
SPLIT = "train"
LICENSE = "ODC-By 1.0 (FineWeb-2); underlying pages under their own terms"


def existing_hashes(paths: list[Path]) -> set[str]:
    """Every Korean document the project already holds, in any form."""
    seen: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        assert_source_allowed(path.name, path)
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                if path.suffix == ".jsonl":
                    try:
                        text = json.loads(line).get("text", "")
                    except json.JSONDecodeError:
                        continue
                else:
                    text = line
                if text:
                    seen.add(content_hash(text))
    return seen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/raw/fineweb_ko_extra.jsonl")
    parser.add_argument("--report", default="outputs/fineweb_extra_acquisition.json")
    parser.add_argument("--target-chars", type=int, default=34_000_000,
                        help="Usable characters to collect, after filtering and dedup.")
    parser.add_argument("--start-shard", type=int, default=1,
                        help="Shard 0 was consumed by the original 50M-character slice.")
    parser.add_argument("--max-shards", type=int, default=12)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--max-chars", type=int, default=200_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from huggingface_hub import hf_hub_download, list_repo_files
    import pyarrow.parquet as pq

    files = sorted(
        name for name in list_repo_files(DATASET, repo_type="dataset")
        if name.startswith(f"data/{CONFIG}/{SPLIT}/") and name.endswith(".parquet")
    )
    shards = files[args.start_shard:args.start_shard + args.max_shards]
    print(f"[fineweb] {len(files)} shards available; using {len(shards)} from index "
          f"{args.start_shard}", flush=True)

    print("[fineweb] indexing existing Korean documents", flush=True)
    known = existing_hashes([
        PROJECT_ROOT / "data/processed/fineweb_general_train.txt",
        PROJECT_ROOT / "data/processed/fineweb_general_holdout.txt",
        PROJECT_ROOT / "data/raw/fineweb_ko_50m.txt",
        PROJECT_ROOT / "data/processed/stage2_train.jsonl",
        PROJECT_ROOT / "data/eval/recovery_balanced_eval.jsonl",
        PROJECT_ROOT / "data/eval/recovery_korean_large_eval.jsonl",
    ])
    print(f"[fineweb] {len(known):,} known document hashes", flush=True)

    counts = {"scanned": 0, "accepted": 0, "rejected_quality": 0,
              "rejected_duplicate": 0, "accepted_chars": 0, "duplicate_chars": 0,
              "shards_downloaded": 0}
    used_shards: list[str] = []
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as handle:
        for shard in shards:
            if counts["accepted_chars"] >= args.target_chars:
                break
            print(f"[fineweb] downloading {shard}", flush=True)
            local = hf_hub_download(DATASET, shard, repo_type="dataset")
            used_shards.append(shard)
            counts["shards_downloaded"] += 1
            table = pq.ParquetFile(local)
            for batch in table.iter_batches(batch_size=2000,
                                            columns=["id", "text", "url", "language"]):
                rows = batch.to_pylist()
                for row in rows:
                    counts["scanned"] += 1
                    text = normalize(row.get("text") or "")
                    if quality_verdict(text, min_chars=args.min_chars,
                                       max_chars=args.max_chars, max_token_length=400):
                        counts["rejected_quality"] += 1
                        continue
                    digest = content_hash(text)
                    if digest in known:
                        counts["rejected_duplicate"] += 1
                        counts["duplicate_chars"] += len(text)
                        continue
                    known.add(digest)
                    handle.write(json.dumps({
                        "text": text,
                        "source": "fineweb2_ko_extra",
                        "source_id": str(row.get("id") or digest[:16]),
                        "dataset": DATASET,
                        "dataset_config": CONFIG,
                        "dataset_split": SPLIT,
                        "shard": shard,
                        "url": row.get("url"),
                        "language": row.get("language") or "kor_Hang",
                        "license": LICENSE,
                        "chars": len(text),
                        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    }, ensure_ascii=False) + "\n")
                    counts["accepted"] += 1
                    counts["accepted_chars"] += len(text)
                if counts["scanned"] % 20000 < 2000:
                    print(f"  scanned {counts['scanned']:,} | accepted {counts['accepted']:,} "
                          f"({counts['accepted_chars']:,} chars) | dup {counts['rejected_duplicate']:,} "
                          f"| rejected {counts['rejected_quality']:,}", flush=True)
                if counts["accepted_chars"] >= args.target_chars:
                    print("[fineweb] target reached; stopping", flush=True)
                    break

    report = {
        "dataset": DATASET, "config": CONFIG, "split": SPLIT,
        "provider": "HuggingFaceFW", "access": "public, ungated, huggingface_hub",
        "license": LICENSE,
        "excluded": f"data/{CONFIG}_removed/* (deduplication-rejected portion)",
        "start_shard_index": args.start_shard,
        "shards_used": used_shards,
        "target_chars": args.target_chars,
        **counts,
        "output": str(out_path),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "shards_used"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
