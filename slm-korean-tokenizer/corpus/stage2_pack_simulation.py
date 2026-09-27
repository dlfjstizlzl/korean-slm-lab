#!/usr/bin/env python3
"""Simulate 1024-token packing of the Stage-2 corpus. No training, no tensors.

Documents are concatenated with an explicit separator so unrelated documents
spliced into one block still carry a visible boundary, then cut into fixed
1024-token blocks. The trailing partial block is dropped rather than padded, so
every training position has a real label.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-lengths", default="data/processed/stage2_token_lengths.jsonl")
    parser.add_argument("--manifest-out", default="data/processed/stage2_1024_manifest.jsonl")
    parser.add_argument("--report", default="outputs/stage2_packing.json")
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--separator-tokens", type=int, default=1,
                        help="Tokens inserted between documents (<|endoftext|> = 1).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-tokens", type=int, default=60_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = [json.loads(line) for line in Path(args.token_lengths).open(encoding="utf-8")
               if line.strip()]
    rng = random.Random(args.seed)
    order = list(range(len(records)))
    rng.shuffle(order)

    sequence = args.sequence_length
    separator = args.separator_tokens

    position = 0           # tokens placed into the current block
    blocks = 0
    docs_in_block = 0
    boundary_crossings = 0
    docs_per_block: Counter = Counter()
    tokens_by_category = Counter()
    documents_spanning = 0

    manifest_path = Path(args.manifest_out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    total_stream = 0

    with manifest_path.open("w", encoding="utf-8") as handle:
        for index in order:
            record = records[index]
            length = int(record["tokens"]) + separator
            tokens_by_category[record["category"]] += int(record["tokens"])
            start_block = blocks
            start_offset = position

            remaining = length
            crossed = False
            while remaining > 0:
                space = sequence - position
                take = min(space, remaining)
                position += take
                remaining -= take
                if position == sequence:
                    docs_per_block[docs_in_block + 1] += 1
                    blocks += 1
                    position = 0
                    docs_in_block = 0
                    if remaining > 0:
                        crossed = True
                        boundary_crossings += 1
            if position > 0:
                docs_in_block += 1
            if crossed:
                documents_spanning += 1
            total_stream += length

            handle.write(json.dumps({
                "document_index": index,
                "category": record["category"],
                "source": record["source"],
                "tokens": int(record["tokens"]),
                "start_block": start_block,
                "start_offset": start_offset,
                "spans_blocks": crossed,
            }, ensure_ascii=False) + "\n")

    full_blocks = blocks
    dropped_tail = position
    packed_tokens = full_blocks * sequence

    report = {
        "sequence_length": sequence,
        "separator": "<|endoftext|> between documents",
        "separator_tokens_per_document": separator,
        "documents": len(records),
        "stream_tokens": total_stream,
        "full_blocks": full_blocks,
        "packed_tokens": packed_tokens,
        "dropped_tail_tokens": dropped_tail,
        "packing_efficiency": packed_tokens / total_stream if total_stream else 0.0,
        "document_boundary_crossings": boundary_crossings,
        "documents_spanning_blocks": documents_spanning,
        "documents_spanning_share": documents_spanning / len(records) if records else 0.0,
        "average_documents_per_block": (
            sum(count * docs for docs, count in docs_per_block.items()) / full_blocks
            if full_blocks else 0.0
        ),
        "documents_per_block_distribution": {
            str(docs): count for docs, count in sorted(docs_per_block.items())[:20]
        },
        "tokens_by_category": dict(tokens_by_category),
        "budget": {
            "target_tokens": args.target_tokens,
            "blocks_for_target": args.target_tokens // sequence,
            "epochs_required": args.target_tokens / packed_tokens if packed_tokens else None,
            "within_one_epoch": packed_tokens >= args.target_tokens,
        },
        "manifest": str(manifest_path),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("documents_per_block_distribution", "tokens_by_category")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
