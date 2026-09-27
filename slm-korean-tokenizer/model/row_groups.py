#!/usr/bin/env python3
"""Classify every KO-32K embedding row as exact-copy, decomposed, or random.

The transplant decides each row's provenance from the two tokenizers alone, so
the classification can be recomputed without loading the model. Recovery uses it
for the gradient mask and for per-group norm tracking.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from transplant_ko32k import (  # noqa: E402
    CONTROL_TOKENS,
    DECOMPOSED,
    EXACT,
    byte_level_alphabet,
    plan_token,
)

DEFAULT_CACHE = "outputs/ko32k_row_groups.json"


def compute_row_groups(base_model: str, tokenizer_path: str, old_rows: int | None = None) -> dict:
    base_tokenizer = AutoTokenizer.from_pretrained(base_model)
    new_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    old_vocab = base_tokenizer.get_vocab()
    bpe_model = base_tokenizer.backend_tokenizer.model
    alphabet = byte_level_alphabet()
    if old_rows is None:
        from transformers import AutoConfig

        old_rows = int(AutoConfig.from_pretrained(base_model).vocab_size)

    added = {str(token) for token in new_tokenizer.added_tokens_decoder.values()}
    base_added = {str(token) for token in base_tokenizer.added_tokens_decoder.values()}
    tokens = new_tokenizer.convert_ids_to_tokens(list(range(len(new_tokenizer))))

    strategies: list[str] = []
    for token in tokens:
        strategy, _, _, _ = plan_token(
            token, is_added=token in added, old_vocab=old_vocab,
            old_rows=old_rows, bpe_model=bpe_model, alphabet=alphabet,
        )
        strategies.append(strategy)

    return {
        "base_model": base_model,
        "tokenizer": str(tokenizer_path),
        "vocab_size": len(tokens),
        "strategies": strategies,
        "exact": [i for i, s in enumerate(strategies) if s == EXACT],
        "decomposed": [i for i, s in enumerate(strategies) if s == DECOMPOSED],
        "random": [i for i, s in enumerate(strategies) if s not in (EXACT, DECOMPOSED)],
        "control": [int(new_tokenizer.convert_tokens_to_ids(t)) for t in CONTROL_TOKENS],
        "base_added": sorted(int(new_tokenizer.convert_tokens_to_ids(t)) for t in base_added),
    }


def load_row_groups(base_model: str, tokenizer_path: str, cache: str | Path = DEFAULT_CACHE) -> dict:
    path = Path(cache)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("tokenizer") == str(tokenizer_path) and payload.get("base_model") == base_model:
            return payload
    payload = compute_row_groups(base_model, tokenizer_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    import sys

    groups = load_row_groups(
        sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B",
        sys.argv[2] if len(sys.argv) > 2 else "outputs/tokenizer-ko-32000-mixed10-structured-v2",
    )
    print(json.dumps({key: len(groups[key]) for key in ("exact", "decomposed", "random", "control", "base_added")},
                     indent=2))
