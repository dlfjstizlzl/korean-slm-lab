#!/usr/bin/env python3
"""Validate the KO-32K transplant checkpoint by reloading it from disk.

Runs structural checks, forward passes over Korean / mixed / Skill-DSL / JSON /
SQL samples, a short generation smoke test, and a baseline next-token loss on
held-out text for both the transplanted checkpoint and the original base model.

Generation *quality* is deliberately not judged: no recovery training has run.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tokenizer"))

from common import (  # noqa: E402
    CONTROL_TOKENS,
    assert_control_tokens_atomic,
    normalize_text,
    read_corpus,
)

DEFAULT_MODEL = "outputs/models/qwen3-0.6b-ko32k-transplanted"
DEFAULT_BASE = "Qwen/Qwen3-0.6B"

FORWARD_SAMPLES = {
    "korean": "가상메모리는 프로세스에게 물리 메모리와 독립된 주소 공간을 제공한다.",
    "mixed_technical": "Qwen3 모델을 llama.cpp에서 Q4_K_M으로 실행한다.",
    "skill_dsl": "<검색>Apple Silicon 통합 메모리 구조",
    "json": '{"skill":"search","input":"Apple Silicon unified memory"}',
    "sql": "SELECT id, name FROM users WHERE created_at >= '2026-09-01';",
}

GENERATION_PROMPT = "한국어에서 가상 메모리는"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-model", default=DEFAULT_BASE)
    parser.add_argument("--report-json", default="outputs/ko32k_transplant_validation.json")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--eval-korean", default="data/korean_eval.txt")
    parser.add_argument("--eval-extra", nargs="*", default=[
        "data/eval_skill_dsl.txt", "data/eval_code_url.txt",
    ])
    parser.add_argument("--eval-sample", nargs="*", default=[
        "data/eval_json.txt", "data/eval_sql.txt",
    ], help="Files sampled deterministically (first --eval-sample-lines lines).")
    parser.add_argument("--eval-sample-lines", type=int, default=40)
    parser.add_argument("--skip-base-baseline", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def load_eval_texts(args: argparse.Namespace) -> list[str]:
    """Deterministic held-out evaluation slice. No training touches this data."""
    texts: list[str] = [record.text for record in read_corpus(args.eval_korean)]
    for path in args.eval_extra:
        candidate = Path(path)
        if candidate.exists():
            texts.extend(record.text for record in read_corpus(candidate))
    for path in args.eval_sample:
        candidate = Path(path)
        if not candidate.exists():
            continue
        with candidate.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index >= args.eval_sample_lines:
                    break
                text = normalize_text(line)
                if text:
                    texts.append(text)
    return texts


def structural_report(model, tokenizer) -> dict:
    input_module = model.get_input_embeddings()
    output_module = model.get_output_embeddings()
    if output_module is None:
        raise ValueError("Reloaded model has no LM head")
    checks: dict[str, bool] = {}
    vocab = len(tokenizer)
    checks["tokenizer_len_32000"] = vocab == 32000
    checks["config_vocab_size"] = int(model.config.vocab_size) == vocab
    checks["embedding_rows"] = int(input_module.weight.shape[0]) == vocab
    checks["lm_head_rows"] = int(output_module.weight.shape[0]) == vocab
    checks["hidden_size"] = int(input_module.weight.shape[1]) == int(model.config.hidden_size)
    tied_expected = bool(getattr(model.config, "tie_word_embeddings", False))
    tied_actual = input_module.weight.data_ptr() == output_module.weight.data_ptr()
    checks["tied_weights"] = tied_expected == tied_actual
    added = {str(token) for token in tokenizer.added_tokens_decoder.values()}
    control_ids = assert_control_tokens_atomic(tokenizer)
    checks["control_tokens_atomic"] = len(control_ids) == len(CONTROL_TOKENS)
    checks["added_tokens_26_plus_8"] = len(added) == 34
    for attr in ("eos_token_id", "pad_token_id"):
        value = getattr(tokenizer, attr, None)
        checks[f"{attr}_valid"] = value is not None and 0 <= int(value) < vocab
        checks[f"config_{attr}_matches"] = getattr(model.config, attr, None) == value
    weight = input_module.weight.detach()
    checks["no_nan"] = not bool(torch.isnan(weight).any().item())
    checks["no_inf"] = not bool(torch.isinf(weight).any().item())
    return {
        "checks": checks,
        "all_passed": all(checks.values()),
        "vocab_size": vocab,
        "embedding_shape": list(input_module.weight.shape),
        "lm_head_shape": list(output_module.weight.shape),
        "tie_word_embeddings_config": tied_expected,
        "tie_word_embeddings_actual": tied_actual,
        "control_token_ids": control_ids,
        "added_token_count": len(added),
        "dtype": str(weight.dtype),
    }


def forward_checks(model, tokenizer) -> dict:
    results = {}
    vocab = len(tokenizer)
    for name, text in FORWARD_SAMPLES.items():
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True)
        ids = encoded["input_ids"]
        in_range = bool((ids >= 0).all().item() and (ids < vocab).all().item())
        with torch.inference_mode():
            output = model(**encoded, labels=ids)
        logits = output.logits
        loss = float(output.loss)
        results[name] = {
            "text": text,
            "token_count": int(ids.shape[1]),
            "ids_in_range": in_range,
            "logits_shape": list(logits.shape),
            "logits_vocab_dim_ok": int(logits.shape[-1]) == vocab,
            "logits_finite": bool(torch.isfinite(logits).all().item()),
            "loss": loss,
            "loss_finite": math.isfinite(loss),
            "passed": in_range and int(logits.shape[-1]) == vocab
                      and bool(torch.isfinite(logits).all().item()) and math.isfinite(loss),
        }
    return results


def generation_smoke(model, tokenizer, max_new_tokens: int) -> dict:
    encoded = tokenizer(GENERATION_PROMPT, return_tensors="pt", add_special_tokens=True)
    try:
        with torch.inference_mode():
            generated = model.generate(
                **encoded, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
    except Exception as error:  # noqa: BLE001 - smoke test reports any crash
        return {"passed": False, "error": repr(error)}
    ids = generated[0].tolist()
    return {
        "passed": all(0 <= i < len(tokenizer) for i in ids),
        "prompt": GENERATION_PROMPT,
        "generated_token_count": len(ids) - int(encoded["input_ids"].shape[1]),
        "ids_in_range": all(0 <= i < len(tokenizer) for i in ids),
        "text": tokenizer.decode(ids, skip_special_tokens=True),
        "quality_evaluated": False,
        "note": "Output quality is not judged; no recovery training has run.",
    }


def baseline_loss(model, tokenizer, texts: list[str]) -> dict:
    """Sum of token NLL over held-out text, plus tokenizer-independent per-character NLL."""
    total_nll = 0.0
    total_predicted = 0
    total_tokens = 0
    total_characters = 0
    for text in texts:
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=True)["input_ids"]
        if ids.shape[1] < 2:
            continue
        with torch.inference_mode():
            logits = model(input_ids=ids).logits.to(torch.float32)
        loss = torch.nn.functional.cross_entropy(
            logits[0, :-1, :], ids[0, 1:], reduction="sum"
        )
        total_nll += float(loss)
        total_predicted += int(ids.shape[1] - 1)
        total_tokens += int(ids.shape[1])
        total_characters += len(text)
    mean_token_loss = total_nll / total_predicted if total_predicted else float("nan")
    return {
        "samples": len(texts),
        "total_tokens": total_tokens,
        "predicted_tokens": total_predicted,
        "total_characters": total_characters,
        "total_nll": total_nll,
        "mean_loss_per_token": mean_token_loss,
        "perplexity": math.exp(mean_token_loss) if mean_token_loss < 20 else float("inf"),
        "nll_per_character": total_nll / total_characters if total_characters else float("nan"),
        "tokens_per_character": total_tokens / total_characters if total_characters else 0.0,
    }


def parameter_counts(model) -> dict:
    input_module = model.get_input_embeddings()
    output_module = model.get_output_embeddings()
    tied = output_module is not None and input_module.weight.data_ptr() == output_module.weight.data_ptr()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    embedding = int(input_module.weight.numel())
    if output_module is not None and not tied:
        embedding += int(output_module.weight.numel())
    return {
        "total": total, "trainable": trainable, "embedding": embedding,
        "non_embedding": total - embedding,
        "estimated_bytes_bf16": total * 2,
    }


def directory_size(path: Path) -> dict:
    if not path.exists():
        return {"exists": False}
    files = {f.name: f.stat().st_size for f in path.iterdir() if f.is_file()}
    total = sum(files.values())
    return {
        "exists": True,
        "total_bytes": total,
        "total_mib": total / 1024 / 1024,
        "files_mib": {name: size / 1024 / 1024 for name, size in sorted(files.items())},
    }


def resolve_base_dir(base_model: str) -> Path:
    candidate = Path(base_model)
    if candidate.exists():
        return candidate
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(base_model, local_files_only=True))
    except Exception:  # noqa: BLE001
        return candidate


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype="auto", device_map="cpu", trust_remote_code=args.trust_remote_code
    )
    model.eval()

    report: dict = {
        "model": str(model_dir),
        "base_model": args.base_model,
        "reload": {"tokenizer_load": "PASS", "model_load": "PASS"},
        "structural": structural_report(model, tokenizer),
    }
    report["forward"] = forward_checks(model, tokenizer)
    report["generation_smoke"] = generation_smoke(model, tokenizer, args.max_new_tokens)

    texts = load_eval_texts(args)
    report["baseline_eval"] = {
        "sources": {
            "korean": args.eval_korean,
            "extra": args.eval_extra,
            "sampled_first_n": {"files": args.eval_sample, "lines": args.eval_sample_lines},
        },
        "note": "Held-out evaluation text only. Nothing here is used for training.",
        "transplanted": baseline_loss(model, tokenizer, texts),
    }
    report["parameters"] = {"transplanted": parameter_counts(model)}
    report["checkpoint_size"] = {"transplanted": directory_size(model_dir)}

    if not args.skip_base_baseline:
        base_tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model, dtype="auto", device_map="cpu", trust_remote_code=args.trust_remote_code
        )
        base_model.eval()
        report["baseline_eval"]["original"] = baseline_loss(base_model, base_tokenizer, texts)
        report["baseline_eval"]["comparison_caveat"] = (
            "The two models use different tokenizers, so per-token loss and perplexity are "
            "not directly comparable. nll_per_character is the tokenizer-independent metric."
        )
        report["parameters"]["original"] = parameter_counts(base_model)
        report["checkpoint_size"]["original"] = directory_size(resolve_base_dir(args.base_model))
        original = report["parameters"]["original"]
        new = report["parameters"]["transplanted"]
        report["parameters"]["reduction"] = {
            "total": original["total"] - new["total"],
            "total_ratio": (original["total"] - new["total"]) / original["total"],
            "embedding": original["embedding"] - new["embedding"],
            "embedding_ratio": (original["embedding"] - new["embedding"]) / original["embedding"],
        }
        base_char = report["baseline_eval"]["original"]["nll_per_character"]
        new_char = report["baseline_eval"]["transplanted"]["nll_per_character"]
        report["baseline_eval"]["degradation"] = {
            "nll_per_character_delta": new_char - base_char,
            "nll_per_character_ratio": new_char / base_char if base_char else None,
        }

    checks = report["structural"]["checks"]
    forward_ok = all(entry["passed"] for entry in report["forward"].values())
    report["summary"] = {
        "structural_passed": all(checks.values()),
        "forward_passed": forward_ok,
        "generation_passed": report["generation_smoke"]["passed"],
        "failed_checks": [name for name, ok in checks.items() if not ok],
        "quality_evaluated": False,
    }

    destination = Path(args.report_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"saved validation report to {destination}")
    if not (report["summary"]["structural_passed"] and forward_ok):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
