#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tokenizer"))

from common import CONTROL_TOKENS, assert_control_tokens_atomic  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a tokenizer/embedding transplant checkpoint.")
    parser.add_argument("--model", default="outputs/qwen3-0.6b-ko-init")
    parser.add_argument("--text", default="가상메모리는 물리 메모리와 독립된 주소 공간을 제공한다.")
    parser.add_argument("--output-json", default="outputs/transplant_validation.json")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def validate_special_ids(model, tokenizer) -> dict[str, int | list[int] | None]:
    checked = {}
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
        tokenizer_value = getattr(tokenizer, attr, None)
        config_value = getattr(model.config, attr, None)
        if tokenizer_value is not None and config_value != tokenizer_value:
            raise ValueError(f"{attr} mismatch: tokenizer={tokenizer_value}, config={config_value}")
        values = tokenizer_value if isinstance(tokenizer_value, list) else [tokenizer_value]
        for value in values:
            if value is not None and not 0 <= int(value) < len(tokenizer):
                raise ValueError(f"{attr}={value} is outside tokenizer vocabulary")
        checked[attr] = tokenizer_value
    return checked


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype="auto", device_map="cpu", trust_remote_code=args.trust_remote_code
    )
    input_shape = tuple(model.get_input_embeddings().weight.shape)
    output_module = model.get_output_embeddings()
    if output_module is None:
        raise ValueError("Causal LM has no output embedding / LM head")
    output_shape = tuple(output_module.weight.shape)
    if input_shape[0] != len(tokenizer):
        raise ValueError(f"embedding rows {input_shape[0]} != tokenizer length {len(tokenizer)}")
    if output_shape[0] != len(tokenizer):
        raise ValueError(f"LM head rows {output_shape[0]} != tokenizer length {len(tokenizer)}")
    if int(model.config.vocab_size) != len(tokenizer):
        raise ValueError(f"config vocab_size {model.config.vocab_size} != tokenizer length {len(tokenizer)}")
    special_ids = validate_special_ids(model, tokenizer)
    control_ids = assert_control_tokens_atomic(tokenizer)
    tied_expected = bool(getattr(model.config, "tie_word_embeddings", False))
    tied_actual = model.get_input_embeddings().weight.data_ptr() == output_module.weight.data_ptr()
    if tied_expected != tied_actual:
        raise ValueError(f"weight tying mismatch: config={tied_expected}, actual={tied_actual}")

    encoded = tokenizer(args.text, return_tensors="pt", add_special_tokens=True)
    with torch.inference_mode():
        output = model(**encoded)
    logits = output.logits
    finite = bool(torch.isfinite(logits).all().item())
    if not finite:
        raise FloatingPointError("Forward pass produced NaN or Inf logits")
    if logits.shape[-1] != len(tokenizer):
        raise ValueError(f"logit vocabulary {logits.shape[-1]} != tokenizer length {len(tokenizer)}")

    report = {
        "model_load": "PASS",
        "tokenizer_load": "PASS",
        "embedding_shape": list(input_shape),
        "lm_head_shape": list(output_shape),
        "config_vocab_size": int(model.config.vocab_size),
        "tokenizer_size": len(tokenizer),
        "special_token_ids": special_ids,
        "control_token_ids": control_ids,
        "tie_word_embeddings_expected": tied_expected,
        "tie_word_embeddings_actual": tied_actual,
        "input_shape": list(encoded["input_ids"].shape),
        "logits_shape": list(logits.shape),
        "logits_finite": finite,
        "forward_pass": "PASS",
        "quality_evaluated": False,
        "warning": "Generation quality is intentionally not evaluated; continued pretraining has not been performed.",
    }
    destination = Path(args.output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
