#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import shutil
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tokenizer"))

from common import (  # noqa: E402
    CONTROL_TOKENS,
    assert_base_added_tokens_preserved,
    assert_control_tokens_atomic,
    assert_required_special_tokens,
    special_token_strings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transplant a base causal LM to an HF-trained Korean tokenizer.")
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--tokenizer", default="outputs/tokenizer-ko-24000")
    parser.add_argument("--output-dir", default="outputs/qwen3-0.6b-ko-init")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-shard-size", default="5GB")
    return parser.parse_args()


def resolve_dtype(name: str):
    return {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def token_surface(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode(
            [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
    except TypeError:
        return tokenizer.decode([token_id], skip_special_tokens=False)


def source_ids_for_token(base_tokenizer, new_tokenizer, token: str, new_id: int, old_vocab: dict[str, int], old_rows: int):
    exact_id = old_vocab.get(token)
    if exact_id is not None and 0 <= int(exact_id) < old_rows:
        return "exact", [int(exact_id)], token_surface(new_tokenizer, new_id)
    surface = token_surface(new_tokenizer, new_id)
    if "\ufffd" in surface:
        return "random", [], surface
    ids = list(base_tokenizer.encode(surface, add_special_tokens=False)) if surface else []
    unk_id = getattr(base_tokenizer, "unk_token_id", None)
    valid = ids and all(0 <= int(token_id) < old_rows for token_id in ids)
    if unk_id is not None and any(int(token_id) == int(unk_id) for token_id in ids):
        valid = False
    return ("averaged", [int(value) for value in ids], surface) if valid else ("random", [], surface)


def initialize_weight(weight: torch.Tensor, std: float) -> None:
    torch.nn.init.normal_(weight, mean=0.0, std=std)


def apply_average(destination: torch.Tensor, row: int, source: torch.Tensor, source_ids: list[int]) -> None:
    selected = source[source_ids].to(dtype=torch.float32)
    destination[row].copy_(selected.mean(dim=0).to(dtype=destination.dtype))


def set_special_token_ids(model, tokenizer) -> None:
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            setattr(model.config, attr, value)
            if getattr(model, "generation_config", None) is not None:
                setattr(model.generation_config, attr, value)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    prepare_output(output_dir, args.overwrite)

    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    new_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=args.trust_remote_code)
    assert_required_special_tokens(base_tokenizer, new_tokenizer)
    assert_base_added_tokens_preserved(base_tokenizer, new_tokenizer)
    control_ids = assert_control_tokens_atomic(new_tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=resolve_dtype(args.dtype),
        device_map="cpu",
        trust_remote_code=args.trust_remote_code,
    )
    old_input = model.get_input_embeddings().weight.detach()
    old_output_module = model.get_output_embeddings()
    old_output = old_output_module.weight.detach() if old_output_module is not None else None
    old_rows = old_input.shape[0]
    old_vocab = base_tokenizer.get_vocab()
    tied = bool(getattr(model.config, "tie_word_embeddings", False))
    initializer_range = float(getattr(model.config, "initializer_range", 0.02))

    resize_kwargs = {}
    if "mean_resizing" in inspect.signature(model.resize_token_embeddings).parameters:
        resize_kwargs["mean_resizing"] = False
    model.resize_token_embeddings(len(new_tokenizer), **resize_kwargs)
    new_input = model.get_input_embeddings().weight
    new_output_module = model.get_output_embeddings()
    new_output = new_output_module.weight if new_output_module is not None else None

    counts = {"exact_match": 0, "averaged": 0, "random_fallback": 0}
    examples = {"exact_match": [], "averaged": [], "random_fallback": []}
    new_vocab = new_tokenizer.get_vocab()
    ordered_tokens = sorted(new_vocab.items(), key=lambda item: item[1])

    with torch.no_grad():
        initialize_weight(new_input, initializer_range)
        if new_output is not None and not tied:
            initialize_weight(new_output, initializer_range)
        for token, new_id in ordered_tokens:
            policy, source_ids, surface = source_ids_for_token(
                base_tokenizer, new_tokenizer, token, int(new_id), old_vocab, old_rows
            )
            if policy == "exact":
                new_input[new_id].copy_(old_input[source_ids[0]].to(dtype=new_input.dtype))
                if new_output is not None and not tied and old_output is not None:
                    new_output[new_id].copy_(old_output[source_ids[0]].to(dtype=new_output.dtype))
                key = "exact_match"
            elif policy == "averaged":
                apply_average(new_input, new_id, old_input, source_ids)
                if new_output is not None and not tied and old_output is not None:
                    apply_average(new_output, new_id, old_output, source_ids)
                key = "averaged"
            else:
                key = "random_fallback"
            counts[key] += 1
            if len(examples[key]) < 12:
                examples[key].append(
                    {"new_id": int(new_id), "token": token, "surface": surface, "source_ids": source_ids}
                )

    model.config.vocab_size = len(new_tokenizer)
    set_special_token_ids(model, new_tokenizer)
    if tied:
        model.tie_weights()
    if model.get_input_embeddings().weight.shape[0] != len(new_tokenizer):
        raise RuntimeError("Input embedding row count does not match tokenizer length after transplant")
    if model.get_output_embeddings() is not None and model.get_output_embeddings().weight.shape[0] != len(new_tokenizer):
        raise RuntimeError("LM head row count does not match tokenizer length after transplant")
    if tied and model.get_input_embeddings().weight.data_ptr() != model.get_output_embeddings().weight.data_ptr():
        raise RuntimeError("tie_word_embeddings=true but input and output weights are not tied")

    model.save_pretrained(output_dir, safe_serialization=True, max_shard_size=args.max_shard_size)
    new_tokenizer.save_pretrained(output_dir)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(output_dir)

    metadata = {
        "checkpoint_status": "embedding_transplant_only",
        "warning": "This checkpoint only contains transplanted embeddings. Continued pretraining is required before meaningful evaluation.",
        "base_model": args.base_model,
        "source_tokenizer": args.tokenizer,
        "original_vocab": int(old_rows),
        "new_vocab": len(new_tokenizer),
        "hidden_size": int(new_input.shape[1]),
        "tie_word_embeddings": tied,
        **counts,
        "control_token_ids": control_ids,
        "base_special_tokens": special_token_strings(base_tokenizer),
        "new_special_tokens": special_token_strings(new_tokenizer),
        "examples": examples,
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
    }
    (output_dir / "transplant_report.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"saved embedding-transplant checkpoint to {output_dir}")


if __name__ == "__main__":
    main()
