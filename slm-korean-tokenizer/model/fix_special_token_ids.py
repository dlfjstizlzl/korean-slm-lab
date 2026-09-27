#!/usr/bin/env python3
"""Repair special-token ids that are outside the KO-32K vocabulary.

`transplant_ko32k.py` synchronized config ids from the new tokenizer with
`if value is not None`. The KO-32K tokenizer has no BOS, so `bos_token_id` was
skipped and kept Qwen's inherited 151643 — an index far outside a 32,000-entry
vocabulary.

Harmless during training and evaluation (nothing reads it), but it would break
generation started without input_ids, and it would propagate a wrong id into any
export that reads special tokens. This rewrites the affected config files to
match the tokenizer, and leaves weights untouched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

FIELDS = ("bos_token_id", "eos_token_id", "pad_token_id", "decoder_start_token_id")


def tokenizer_ids(directory: Path) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(directory)
    return {
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "vocab_size": len(tokenizer),
    }


def repair(directory: Path, apply: bool) -> dict:
    expected = tokenizer_ids(directory)
    vocab = expected["vocab_size"]
    changes: dict[str, dict] = {}
    for name in ("config.json", "generation_config.json"):
        path = directory / name
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        edits = {}
        for field in FIELDS:
            if field not in payload:
                continue
            current = payload[field]
            if current is None:
                continue
            values = current if isinstance(current, list) else [current]
            out_of_range = any(not (0 <= int(v) < vocab) for v in values)
            if not out_of_range:
                continue
            target = expected.get(field, None)
            edits[field] = {"was": current, "now": target}
            payload[field] = target
        if edits:
            changes[name] = edits
            if apply:
                path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                                encoding="utf-8")
    return {"directory": str(directory), "vocab_size": vocab,
            "tokenizer_ids": {k: v for k, v in expected.items() if k != "vocab_size"},
            "changes": changes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--report", default="outputs/special_token_id_repair.json")
    args = parser.parse_args()

    results = []
    for target in args.directories:
        directory = Path(target)
        if not (directory / "config.json").exists():
            continue
        result = repair(directory, args.apply)
        results.append(result)
        status = "FIXED" if (args.apply and result["changes"]) else (
            "would fix" if result["changes"] else "ok")
        print(f"{status:9s} {directory}")
        for name, edits in result["changes"].items():
            for field, change in edits.items():
                print(f"            {name}: {field} {change['was']} -> {change['now']}")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(
        {"applied": args.apply, "results": results}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
