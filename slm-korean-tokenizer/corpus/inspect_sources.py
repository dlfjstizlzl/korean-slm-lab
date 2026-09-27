#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from corpus.loaders.aihub import discover_aihub_sources, iter_aihub_objects, schema_paths
from corpus.loaders.fineweb import discover_fineweb_files
from corpus.loaders.nikl import discover_nikl_files


def markdown(payload: dict) -> str:
    lines = ["# Corpus source inventory", "", "## Discovered sources", ""]
    lines += ["| Source | Files | Bytes | Format | Compressed |", "|---|---:|---:|---|---|"]
    for item in payload["sources"]:
        lines.append(
            f"| {item['name']} | {item['file_count']:,} | {item['total_bytes']:,} | "
            f"{item['format']} | {str(item['compressed']).lower()} |"
        )
    lines += ["", "## AI Hub detected schema", ""]
    for path, value_type in payload["aihub_schema"]["paths"]:
        lines.append(f"- `{path}`: `{value_type}`")
    lines += ["", f"Samples inspected: {payload['aihub_schema']['samples_inspected']}", ""]
    if not payload["nikl"]["present"]:
        lines += ["## NIKL", "", "No NIKL corpus is present. This is expected and does not block the pipeline.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover FineWeb, AI Hub, and optional NIKL corpus sources.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--schema-samples", type=int, default=12)
    parser.add_argument("--output-json", default="outputs/corpus_source_inventory.json")
    parser.add_argument("--output-markdown", default="outputs/CORPUS_SOURCE_INVENTORY.md")
    args = parser.parse_args()

    fineweb = discover_fineweb_files(args.data_dir)
    aihub = discover_aihub_sources(args.data_dir)
    sources = []
    if fineweb:
        sources.append({
            "name": "FineWeb Korean",
            "file_count": len(fineweb),
            "total_bytes": sum(path.stat().st_size for path in fineweb),
            "format": ",".join(sorted({path.suffix.lower().lstrip(".") for path in fineweb})),
            "compressed": False,
            "paths": [str(path) for path in fineweb],
        })
    for source in aihub:
        sources.append({
            "name": f"AIHub {source.split.upper()} session{source.session}",
            "file_count": source.file_count,
            "total_bytes": source.total_bytes,
            "format": "zip/json" if source.compressed else "json",
            "compressed": source.compressed,
            "path": str(source.path),
        })

    schema: set[tuple[str, str]] = set()
    per_session = Counter()
    inspected = 0
    per_source_limit = max(1, args.schema_samples // max(1, len(aihub)))
    for source in aihub:
        for source_index, (_, payload) in enumerate(iter_aihub_objects(source)):
            schema.update(schema_paths(payload))
            per_session[f"{source.split}_session{source.session}"] += 1
            inspected += 1
            if source_index + 1 >= per_source_limit or inspected >= args.schema_samples:
                break
        if inspected >= args.schema_samples:
            break
    nikl = discover_nikl_files(args.data_dir)
    payload = {
        "data_dir": args.data_dir,
        "sources": sources,
        "aihub_schema": {
            "samples_inspected": inspected,
            "sample_distribution": dict(per_session),
            "paths": [list(value) for value in sorted(schema)],
        },
        "nikl": {"present": bool(nikl), "files": [str(path) for path in nikl]},
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.output_markdown).write_text(markdown(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"wrote {args.output_json} and {args.output_markdown}")


if __name__ == "__main__":
    main()
