#!/usr/bin/env python3
"""Stage-2 closure validation. No training of any kind happens here.

Subcommands (each writes one JSON under outputs/final_validation/):

  reproduce   balanced + Korean-large decontaminated eval for Original Qwen,
              Stage-1 final and Stage-2 signal2678, three ways:
                legacy  truncate=True  (exactly how the reported numbers were made)
                fixed   truncate=False (canonical from now on)
                prefix  truncate=False + <|endoftext|> context prefix (sensitivity only)
  generation  canonical generation/repetition eval (model/generation_eval.py)
  smoke       fresh-process load, forward, generation, save/reload equality,
              special-token and weight checks of the canonical checkpoint
  structure   parameter / vocabulary / disk accounting for the three models
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))

import generation_eval  # noqa: E402
from recovery_eval import bootstrap_difference, evaluate, load_eval  # noqa: E402

CANONICAL = "outputs/models/qwen3-0.6b-ko32k-stage2-signal2678"
CANONICAL_NAME = "Qwen3-0.6B-KO32K-Stage2-signal2678"
MODELS = {
    "original_qwen": "Qwen/Qwen3-0.6B",
    "stage1_final": "outputs/models/qwen3-0.6b-ko32k-recovery-stage1-extended",
    "stage2_signal2678": CANONICAL,
}
BALANCED = "data/eval/recovery_balanced_eval_decontaminated.jsonl"
KOREAN = "data/eval/recovery_korean_large_eval_decontaminated.jsonl"
# Values as reported before this closure (outputs/stage2_training_report.json).
EXPECTED = {
    "balanced": {"original_qwen": 1.7786435674753764, "stage1_final": 2.0871253721889325,
                 "stage2_signal2678": 1.4860517143041465},
    "korean": {"original_qwen": 2.1076167832868897, "stage1_final": 2.6522817975960926,
               "stage2_signal2678": 1.7540145397626896},
}
# Fixed before re-running: fp32 on MPS with a different batch composition can move
# the 4th decimal only through summation order; anything larger is a real difference.
REPRODUCTION_TOLERANCE = 5e-4
CONTROL_TOKENS = generation_eval.CONTROL_TOKENS
OUT = ROOT / "outputs" / "final_validation"


def write(name: str, payload: dict) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def load(path: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).to(device).eval()
    return model, tokenizer


def release(model, device: str) -> None:
    del model
    if device == "mps":
        torch.mps.empty_cache()


# --------------------------------------------------------------------------- #
# reproduce
# --------------------------------------------------------------------------- #

def reproduce(device: str) -> dict:
    sets = {"balanced": load_eval(BALANCED), "korean": load_eval(KOREAN)}
    variants = {
        "legacy": {"truncate": True, "prefix": False},
        "fixed": {"truncate": False, "prefix": False},
        "prefix": {"truncate": False, "prefix": True},
    }
    results: dict = {}
    per_record: dict = {}
    for name, path in MODELS.items():
        model, tokenizer = load(path, device)
        boundary = [tokenizer.convert_tokens_to_ids("<|endoftext|>")]
        for set_name, records in sets.items():
            for variant, options in variants.items():
                started = time.time()
                scored = evaluate(model, tokenizer, records, device=device, batch_size=8,
                                  truncate=options["truncate"],
                                  context_prefix_ids=boundary if options["prefix"] else None)
                per_record.setdefault(name, {}).setdefault(set_name, {})[variant] = \
                    scored.pop("per_record_nll")
                results.setdefault(name, {}).setdefault(set_name, {})[variant] = {
                    "overall": scored["overall"],
                    "by_category": {k: v["nll_per_character"]
                                    for k, v in scored["by_category"].items()},
                    "by_category_full": scored["by_category"],
                    "records_over_max_length": scored["records_over_max_length"],
                    "truncated_records": scored["truncated_records"],
                    "evaluated_records": scored["evaluated_records"],
                    "skipped_records": scored["skipped_records"],
                    "seconds": time.time() - started,
                }
                print(f"[reproduce] {name} {set_name} {variant} "
                      f"{scored['overall']['nll_per_character']:.6f}", flush=True)
        results[name]["checkpoint"] = path
        release(model, device)

    checks = {}
    for set_name, expected in EXPECTED.items():
        for name, value in expected.items():
            legacy = results[name][set_name]["legacy"]["overall"]["nll_per_character"]
            fixed = results[name][set_name]["fixed"]["overall"]["nll_per_character"]
            checks[f"{set_name}/{name}"] = {
                "expected": value, "legacy_reproduced": legacy,
                "legacy_abs_diff": abs(legacy - value),
                "within_tolerance": abs(legacy - value) <= REPRODUCTION_TOLERANCE,
                "fixed": fixed, "fixed_minus_legacy": fixed - legacy,
            }

    bootstrap = {}
    for set_name, records in sets.items():
        for other in ("original_qwen", "stage1_final"):
            bootstrap[f"{set_name}/stage2_minus_{other}"] = bootstrap_difference(
                records, per_record["stage2_signal2678"][set_name]["fixed"],
                per_record[other][set_name]["fixed"])

    category_comparison = {}
    fixed_bal = {n: results[n]["balanced"]["fixed"]["by_category"] for n in MODELS}
    for category, qwen in fixed_bal["original_qwen"].items():
        stage2 = fixed_bal["stage2_signal2678"][category]
        category_comparison[category] = {
            "original_qwen": qwen, "stage1_final": fixed_bal["stage1_final"][category],
            "stage2_signal2678": stage2, "stage2_minus_qwen": stage2 - qwen,
            "stage2_lower_than_qwen": stage2 < qwen,
        }

    payload = {
        "definition": "recovery_eval.evaluate (see its module docstring)",
        "tolerance": REPRODUCTION_TOLERANCE,
        "balanced_eval": BALANCED, "korean_eval": KOREAN,
        "records": {k: len(v) for k, v in sets.items()},
        "results": results,
        "reproduction_checks": checks,
        "all_reproduced": all(c["within_tolerance"] for c in checks.values()),
        "bootstrap_fixed": bootstrap,
        "balanced_category_comparison": category_comparison,
        "stage2_lower_than_qwen_categories": sum(v["stage2_lower_than_qwen"]
                                                  for v in category_comparison.values()),
        "balanced_categories": len(category_comparison),
    }
    write("reproduction.json", payload)
    return payload


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #

def generation(device: str) -> dict:
    prompts = generation_eval.load_prompts(ROOT / generation_eval.PROMPTS_FILE)
    runs, summaries = {}, {}
    for name, path in MODELS.items():
        model, tokenizer = load(path, device)
        started = time.time()
        runs[name] = generation_eval.run_model(model, tokenizer, prompts, device)
        summaries[name] = generation_eval.summarize(runs[name])
        print(f"[generation] {name} done in {time.time() - started:,.0f}s", flush=True)
        release(model, device)
    payload = {
        "prompts_file": generation_eval.PROMPTS_FILE,
        "prompts": len(prompts),
        "decoding": {"greedy": generation_eval.GREEDY, "sampling": generation_eval.SAMPLING,
                     "seeds": list(generation_eval.SAMPLING_SEEDS),
                     "max_new_tokens": generation_eval.MAX_NEW_TOKENS,
                     "stop_tokens": list(generation_eval.STOP_TOKEN_STRINGS),
                     "repetition_penalty": "none (1.0)"},
        "thresholds": {"loop_max_period": generation_eval.LOOP_MAX_PERIOD,
                       "loop_min_copies": generation_eval.LOOP_MIN_COPIES,
                       "loop_min_coverage": generation_eval.LOOP_MIN_COVERAGE,
                       "premature_eos_tokens": generation_eval.PREMATURE_EOS_TOKENS},
        "summary": summaries,
    }
    write("generation_summary.json", payload)
    write("generation_samples.json", {"note": "every output of every fixed prompt; nothing selected",
                                      "runs": runs})
    return payload


def rescore() -> dict:
    """Re-apply the current validators to the saved generations (no model is run)."""
    prompts = {p["id"]: p for p in generation_eval.load_prompts(ROOT / generation_eval.PROMPTS_FILE)}
    samples = json.loads((OUT / "generation_samples.json").read_text(encoding="utf-8"))
    summary = json.loads((OUT / "generation_summary.json").read_text(encoding="utf-8"))
    for name, run in samples["runs"].items():
        for mode in ("greedy", "sampled"):
            for sample in run[mode]:
                sample["structured"] = generation_eval.structured_checks(
                    prompts[sample["id"]], sample["generated"], sample["metrics"]["stopped_by_eos"])
        summary["summary"][name] = generation_eval.summarize(run)
    summary["rescored_with"] = "model/generation_eval.py (Skill DSL grammar v2, derived from the v3 generator)"
    write("generation_samples.json", samples)
    write("generation_summary.json", summary)
    return summary


# --------------------------------------------------------------------------- #
# smoke
# --------------------------------------------------------------------------- #

SMOKE_TEXTS = [
    "가상 메모리는 물리 메모리보다 큰 주소 공간을 제공한다.",
    "<검색>서울 날씨<종료>",
    '{"skill":"sql","input":"월별 매출"}',
    "SELECT name FROM users WHERE id = 3;",
]


def special_token_audit(model, tokenizer) -> dict:
    vocab = model.get_input_embeddings().weight.shape[0]
    fields = {}
    for key in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id",
                "decoder_start_token_id"):
        for source, obj in (("config", model.config), ("generation_config", model.generation_config)):
            value = getattr(obj, key, None)
            if value is not None:
                fields[f"{source}.{key}"] = value
        value = getattr(tokenizer, key, None)
        if value is not None:
            fields[f"tokenizer.{key}"] = value
    invalid = {}
    for key, value in fields.items():
        values = value if isinstance(value, list) else [value]
        bad = [v for v in values if not (0 <= int(v) < vocab)]
        if bad:
            invalid[key] = bad
    added = {token.content: token_id for token_id, token in tokenizer.added_tokens_decoder.items()}
    added_invalid = {k: v for k, v in added.items() if not (0 <= v < vocab)}
    all_ids_ok = max(tokenizer.get_vocab().values()) < vocab
    control = {}
    for token in CONTROL_TOKENS:
        ids = tokenizer(token, add_special_tokens=False)["input_ids"]
        control[token] = {"ids": ids, "atomic": len(ids) == 1,
                          "id_in_range": all(0 <= i < vocab for i in ids)}
    return {
        "embedding_rows": vocab, "config_vocab_size": model.config.vocab_size,
        "tokenizer_len": len(tokenizer), "special_ids": fields, "invalid_special_ids": invalid,
        "added_tokens": len(added), "added_tokens_out_of_range": added_invalid,
        "max_vocab_id_in_range": all_ids_ok, "control_tokens": control,
        "all_control_tokens_present_and_atomic": all(c["atomic"] and c["id_in_range"]
                                                     for c in control.values()),
    }


def weight_audit(model) -> dict:
    non_finite = []
    total = 0
    for name, tensor in model.state_dict().items():
        total += tensor.numel()
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            non_finite.append(name)
    unique = sum(p.numel() for p in model.parameters())
    embed = model.get_input_embeddings().weight
    head = model.get_output_embeddings().weight
    return {
        "unique_parameters": unique,
        "tensors_with_nan_or_inf": non_finite,
        "tied": embed.data_ptr() == head.data_ptr(),
        "embedding_shape": list(embed.shape), "lm_head_shape": list(head.shape),
        "dtype": str(embed.dtype),
    }


@torch.inference_mode()
def forward_and_generate(model, tokenizer, device: str) -> dict:
    out = {}
    for text in SMOKE_TEXTS:
        ids = tokenizer(text, return_tensors="pt").to(device)
        logits = model(**ids).logits
        out[text] = {"shape": list(logits.shape), "finite": bool(torch.isfinite(logits).all()),
                     "logits_sum": float(logits.double().sum())}
    encoded = tokenizer("대한민국의 수도는", return_tensors="pt").to(device)
    generated = model.generate(**encoded, max_new_tokens=16, do_sample=False,
                               eos_token_id=generation_eval.stop_token_ids(tokenizer),
                               pad_token_id=tokenizer.pad_token_id)
    return {"forward": out, "generation_ids": generated[0].tolist(),
            "generation_text": tokenizer.decode(generated[0], skip_special_tokens=False)}


def smoke(device: str) -> dict:
    """Runs in its own (fresh) process when invoked from the CLI."""
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(CANONICAL)
    model = AutoModelForCausalLM.from_pretrained(CANONICAL, dtype=torch.float32).to(device).eval()
    first = forward_and_generate(model, tokenizer, device)
    report = {
        "checkpoint": CANONICAL, "fresh_process_pid": os.getpid(),
        "load_seconds": time.time() - started,
        "special_tokens": special_token_audit(model, tokenizer),
        "weights": weight_audit(model),
        "first_load": first,
        "raw_tokenizer_json_truncation": json.loads(
            (ROOT / CANONICAL / "tokenizer.json").read_text(encoding="utf-8")).get("truncation"),
        "config_use_cache": model.config.use_cache,
        "generation_config_eos": model.generation_config.eos_token_id,
    }
    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        tokenizer.save_pretrained(tmp)
        tokenizer2 = AutoTokenizer.from_pretrained(tmp)
        model2 = AutoModelForCausalLM.from_pretrained(tmp, dtype=torch.float32).to(device).eval()
        second = forward_and_generate(model2, tokenizer2, device)
        max_diff = max(
            float((a - b).abs().max())
            for a, b in zip(model.state_dict().values(), model2.state_dict().values())
        )
        report["reload"] = {
            "state_dict_max_abs_diff": max_diff,
            "same_generation": first["generation_ids"] == second["generation_ids"],
            "same_logits_sum": all(
                math.isclose(first["forward"][t]["logits_sum"], second["forward"][t]["logits_sum"],
                             rel_tol=1e-6, abs_tol=1e-3) for t in SMOKE_TEXTS),
            "same_tokenization": all(tokenizer(t)["input_ids"] == tokenizer2(t)["input_ids"]
                                     for t in SMOKE_TEXTS),
            "tied_after_reload": weight_audit(model2)["tied"],
        }
    st = report["special_tokens"]
    wt = report["weights"]
    report["checks"] = {
        "loads_in_fresh_process": True,
        "forward_finite": all(v["finite"] for v in first["forward"].values()),
        "generation_nonempty": len(first["generation_ids"]) > encoded_len(tokenizer),
        "vocab_consistent": st["embedding_rows"] == st["config_vocab_size"] == st["tokenizer_len"] == 32000,
        "embedding_head_shape_consistent": wt["embedding_shape"] == wt["lm_head_shape"] == [32000, 1024],
        "no_invalid_special_ids": not st["invalid_special_ids"] and not st["added_tokens_out_of_range"]
                                   and st["max_vocab_id_in_range"],
        "control_tokens_ok": st["all_control_tokens_present_and_atomic"],
        "no_nan_inf": not wt["tensors_with_nan_or_inf"],
        "tied": wt["tied"],
        "reload_identical": report["reload"]["state_dict_max_abs_diff"] == 0.0
                            and report["reload"]["same_generation"]
                            and report["reload"]["same_tokenization"]
                            and report["reload"]["tied_after_reload"],
    }
    report["all_pass"] = all(report["checks"].values())
    write("smoke_test.json", report)
    return report


def encoded_len(tokenizer) -> int:
    return len(tokenizer("대한민국의 수도는")["input_ids"])


# --------------------------------------------------------------------------- #
# structure
# --------------------------------------------------------------------------- #

def safetensor_bytes(path: str) -> dict:
    from huggingface_hub import snapshot_download

    local = Path(path) if Path(path).exists() else Path(snapshot_download(path, local_files_only=True))
    files = sorted(local.glob("*.safetensors"))
    header_dtype = None
    try:
        from safetensors import safe_open
        with safe_open(str(files[0]), "pt") as handle:
            keys = list(handle.keys())
            header_dtype = str(handle.get_slice(keys[0]).get_dtype())
            has_lm_head = any(k.startswith("lm_head") for k in keys)
    except Exception:  # noqa: BLE001
        has_lm_head = None
    return {"files": [f.name for f in files],
            "bytes": sum(os.path.getsize(f) for f in files),
            "stored_dtype": header_dtype, "stores_separate_lm_head": has_lm_head,
            "directory": str(local)}


def structure() -> dict:
    out = {}
    for name, path in MODELS.items():
        config = AutoConfig.from_pretrained(path)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config)
        total = sum(p.numel() for p in model.parameters())
        embedding = model.get_input_embeddings().weight.numel()
        out[name] = {
            "checkpoint": path,
            "vocab_size": config.vocab_size,
            "embedding_rows": model.get_input_embeddings().weight.shape[0],
            "hidden_size": config.hidden_size,
            "tie_word_embeddings": config.tie_word_embeddings,
            "total_parameters": total,
            "embedding_parameters": embedding,
            "non_embedding_parameters": total - embedding,
            "disk": safetensor_bytes(path),
        }
    q, s = out["original_qwen"], out["stage2_signal2678"]

    def pct(a: float, b: float) -> float:
        return 100.0 * (1.0 - b / a)

    bf16_equiv = {n: v["total_parameters"] * 2 for n, v in out.items()}
    out["comparison"] = {
        "vocab_reduction_percent": pct(q["vocab_size"], s["vocab_size"]),
        "total_parameter_reduction_percent": pct(q["total_parameters"], s["total_parameters"]),
        "embedding_parameter_reduction_percent": pct(q["embedding_parameters"], s["embedding_parameters"]),
        "non_embedding_identical_count": q["non_embedding_parameters"] == s["non_embedding_parameters"],
        "parameters_removed": q["total_parameters"] - s["total_parameters"],
        "disk_bytes_as_stored": {n: out[n]["disk"]["bytes"] for n in MODELS},
        "bf16_weight_bytes_equivalent": bf16_equiv,
        "bf16_weight_bytes_reduction_percent": pct(bf16_equiv["original_qwen"],
                                                   bf16_equiv["stage2_signal2678"]),
        "note": ("Parameter and weight-byte reductions are structural. They are not runtime RAM "
                 "measurements: inference memory also depends on precision/quantization, backend, "
                 "KV cache, activations and allocator overhead, none of which was measured here."),
    }
    write("structure.json", out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["reproduce", "generation", "rescore", "smoke", "structure"])
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.command == "reproduce":
        result = reproduce(args.device)
        print(json.dumps(result["reproduction_checks"], indent=2))
    elif args.command == "generation":
        generation(args.device)
    elif args.command == "rescore":
        rescore()
    elif args.command == "smoke":
        result = smoke(args.device)
        print(json.dumps(result["checks"], indent=2), "\nall_pass:", result["all_pass"])
    else:
        result = structure()
        print(json.dumps(result["comparison"], indent=2))


if __name__ == "__main__":
    main()
