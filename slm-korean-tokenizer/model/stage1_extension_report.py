#!/usr/bin/env python3
"""Aggregate the extended Stage-1 experiment.

Evaluates the original Qwen, the 2.048M-token Stage-1 checkpoint, and the
4.096M-token extended checkpoint on the large Korean validation set and the
balanced set, measures generation repetition, folds in the saturation verdict,
and proposes Stage-2 settings.

The transformer is never unfrozen here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from recovery_eval import bootstrap_difference, embedding_norm_stats, evaluate, load_eval  # noqa: E402
from row_groups import load_row_groups  # noqa: E402
# Single source of truth for repetition metrics (moved to generation_eval.py).
from generation_eval import repetition_metrics  # noqa: E402,F401

PROMPTS = {
    "korean": "가상 메모리는 운영체제에서",
    "technical": "Apple Silicon의 unified memory는",
    "skill": "<검색>Apple Silicon 통합 메모리 구조",
    "json": '{"skill":"search","input":"',
}


def generate(model, tokenizer, device: str, max_new_tokens: int = 48) -> dict:
    out = {}
    for name, prompt in PROMPTS.items():
        encoded = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            generated = model.generate(**encoded, max_new_tokens=max_new_tokens,
                                       do_sample=False, pad_token_id=tokenizer.pad_token_id)
        ids = generated[0].tolist()
        new_ids = ids[int(encoded["input_ids"].shape[1]):]
        text = tokenizer.decode(ids, skip_special_tokens=True)
        out[name] = {"prompt": prompt, "text": text, **repetition_metrics(new_ids, text)}
    return out


def score(model_dir: str, sets: dict[str, list[dict]], device: str,
          groups: dict | None, generate_too: bool) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).to(device)
    model.eval()
    result: dict = {}
    per_record: dict[str, dict] = {}
    for name, records in sets.items():
        scored = evaluate(model, tokenizer, records, device=device, batch_size=8)
        per_record[name] = scored.pop("per_record_nll")
        result[name] = scored
    if groups:
        result["embedding_norms"] = embedding_norm_stats(model, groups)
    if generate_too:
        result["generation"] = generate(model, tokenizer, device)
    del model
    if device == "mps":
        torch.mps.empty_cache()
    return {"metrics": result, "per_record": per_record}


def stage2_plan(final_nll: float, original_nll: float, saturating: bool,
                embedding_lr: float, tokens_done: int) -> dict:
    """Stage-2 settings derived from what Stage 1 actually showed."""
    transformer_lr = embedding_lr / 5.0
    return {
        "unfreeze": "all blocks at once",
        "unfreeze_rationale": (
            "Stage 1 showed exact-copy rows must co-adapt with the rest of the table "
            "(freezing them cost 0.30 NLL/char). The same argument applies to the blocks: "
            "gradual unfreezing recreates the partial-freeze regime that already lost. "
            "Risk is controlled with a low block LR and warmup, not with freezing."
        ),
        "embedding_lr": embedding_lr,
        "transformer_lr": transformer_lr,
        "differential_lr_ratio": embedding_lr / transformer_lr,
        "lr_rationale": (
            f"The embedding has already absorbed {tokens_done:,} tokens of adaptation and is far "
            "ahead of the blocks, which are still exactly Qwen's. A lower block LR lets them move "
            "toward the new embedding without destroying pretrained structure. The embedding rate "
            "continues from where the Stage-1 cosine ended rather than restarting at its peak."
        ),
        "total_training_tokens": 60_000_000,
        "token_rationale": (
            "Roughly 15x the Stage-1 budget. Stage 1 recovered its share within ~2M tokens; "
            "transformer adaptation is a harder objective and needs an order of magnitude more. "
            "Budget by tokens, not steps, so batch/sequence changes stay comparable."
        ),
        "context_length": 1024,
        "context_rationale": (
            "Doubling from 512 once the blocks train, so attention adapts at a length closer to "
            "inference use. Stage 1 did not need it because embeddings are position-independent."
        ),
        "warmup_ratio": 0.02,
        "warmup_rationale": (
            "Lower than Stage 1's 0.05: the embedding is already adapted, so a long warmup mostly "
            "wastes budget. The blocks are protected by their low LR instead."
        ),
        "scheduler": "cosine to 10% of peak, not to zero",
        "scheduler_rationale": (
            "Stage 1's cosine hit exactly 0.0, which made an exact resume impossible and forced a "
            "schedule reconstruction. Ending at a non-zero floor keeps the run extensible."
        ),
        "validation_cadence": "every 2M tokens on the balanced set, every 10M on the large Korean set",
        "cadence_rationale": (
            "The balanced set is cheap and catches structured-category regressions early; the "
            "2.4M-character Korean set is the quality metric but costs minutes per evaluation."
        ),
        "save_optimizer_state": True,
        "must_keep": [
            "vocab_size 32000 and tie_word_embeddings=true",
            "NIKL excluded from training and from model selection",
            "NLL/char as the primary metric; per-token PPL is not comparable to Qwen",
        ],
        "entry_condition_met": saturating,
        "remaining_gap_at_entry": final_nll - original_nll,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", default="outputs/models/qwen3-0.6b-ko32k-recovery-stage1")
    parser.add_argument("--extended", default="outputs/models/qwen3-0.6b-ko32k-recovery-stage1-extended")
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--korean-eval", default="data/eval/recovery_korean_large_eval.jsonl")
    parser.add_argument("--balanced-eval", default="data/eval/recovery_balanced_eval.jsonl")
    parser.add_argument("--saturation", default="outputs/recovery_saturation.json")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output-json", default="outputs/stage1_extension_report.json")
    parser.add_argument("--stage2-embedding-lr", type=float, default=5e-5)
    parser.add_argument("--row-tokenizer", default="outputs/tokenizer-ko-32000-mixed10-structured-v2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    korean = load_eval(args.korean_eval)
    balanced = load_eval(args.balanced_eval)
    groups = load_row_groups(args.base_model, args.row_tokenizer)
    saturation = json.loads(Path(args.saturation).read_text(encoding="utf-8"))

    targets = {
        "original_qwen": (args.base_model, None, False),
        "stage1": (args.stage1, groups, True),
        "stage1_extended": (args.extended, groups, True),
    }
    scored = {}
    for name, (path, group, gen) in targets.items():
        print(f"[extension] evaluating {name}", flush=True)
        scored[name] = score(path, {"korean": korean, "balanced": balanced},
                             args.device, group, gen)
        print(f"  korean NLL/char {scored[name]['metrics']['korean']['overall']['nll_per_character']:.4f}",
              flush=True)

    korean_bootstrap = bootstrap_difference(
        korean,
        scored["stage1"]["per_record"]["korean"],
        scored["stage1_extended"]["per_record"]["korean"],
    )

    original_nll = saturation["original_qwen_nll_per_character"]
    final_nll = saturation["curve"][-1]["nll_per_character"]
    verdict = saturation["saturation"]

    report = {
        "stage": "stage1_extended_embedding_only",
        "transformer_unfrozen": False,
        "warning": "Embedding-only recovery. No transformer parameter was updated at any point.",
        "korean_large_eval": {
            name: {
                "overall": entry["metrics"]["korean"]["overall"],
                "by_category": {k: v["nll_per_character"]
                                for k, v in entry["metrics"]["korean"]["by_category"].items()},
            }
            for name, entry in scored.items()
        },
        "balanced_eval": {
            name: {
                "overall": entry["metrics"]["balanced"]["overall"],
                "by_category": {k: v["nll_per_character"]
                                for k, v in entry["metrics"]["balanced"]["by_category"].items()},
            }
            for name, entry in scored.items()
        },
        "korean_bootstrap_stage1_minus_extended": korean_bootstrap,
        "embedding_norms": {
            name: entry["metrics"].get("embedding_norms", {})
            for name, entry in scored.items() if "embedding_norms" in entry["metrics"]
        },
        "generation": {
            name: entry["metrics"]["generation"]
            for name, entry in scored.items() if "generation" in entry["metrics"]
        },
        "saturation": saturation,
        "stage2_plan": stage2_plan(
            final_nll, original_nll, verdict["saturating"],
            args.stage2_embedding_lr, saturation["curve"][-1]["tokens"],
        ),
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps({
        "korean": {k: round(v["overall"]["nll_per_character"], 4)
                   for k, v in report["korean_large_eval"].items()},
        "saturating": verdict["saturating"], "reason": verdict["reason"],
    }, ensure_ascii=False, indent=2))
    print(f"saved {args.output_json}")


if __name__ == "__main__":
    main()
