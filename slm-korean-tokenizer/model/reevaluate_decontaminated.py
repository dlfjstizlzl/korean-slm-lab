#!/usr/bin/env python3
"""Re-evaluate every existing checkpoint on the decontaminated evaluation sets.

No training happens here. Each checkpoint is loaded, scored, and released. The
point is to find out which of the earlier conclusions survive once the
contaminated evaluation records are gone.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from recovery_eval import bootstrap_difference, evaluate, load_eval  # noqa: E402
from stage1_extension_report import generate, repetition_metrics  # noqa: E402,F401

MODELS = {
    "original_qwen": "Qwen/Qwen3-0.6B",
    "transplant_mean": "outputs/models/qwen3-0.6b-ko32k-init-mean",
    "transplant_length_weighted": "outputs/models/qwen3-0.6b-ko32k-init-length-weighted",
    "transplant_norm_matched": "outputs/models/qwen3-0.6b-ko32k-init-norm-matched",
    "phase1_mean_decomposed_only": "outputs/models/recovery-phase1-mean",
    "phase1_length_weighted_decomposed_only": "outputs/models/recovery-phase1-length-weighted",
    "phase1_norm_matched_decomposed_only": "outputs/models/recovery-phase1-norm-matched",
    "stage1_2048k_all_rows": "outputs/models/qwen3-0.6b-ko32k-recovery-stage1",
    "stage1_extended_4096k_all_rows": "outputs/models/qwen3-0.6b-ko32k-recovery-stage1-extended",
}

# Only these are worth the cost on the 2.3M-character Korean set.
KOREAN_MODELS = [
    "original_qwen", "transplant_mean",
    "stage1_2048k_all_rows", "stage1_extended_4096k_all_rows",
]
GENERATION_MODELS = [
    "transplant_mean", "stage1_2048k_all_rows", "stage1_extended_4096k_all_rows",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--balanced", default="data/eval/recovery_balanced_eval_decontaminated.jsonl")
    parser.add_argument("--korean", default="data/eval/recovery_korean_large_eval_decontaminated.jsonl")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output-json", default="outputs/decontaminated_reevaluation.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    balanced = load_eval(args.balanced)
    korean = load_eval(args.korean)
    print(f"[reeval] balanced {len(balanced):,} records, korean {len(korean):,} records", flush=True)

    results: dict[str, dict] = {}
    per_record: dict[str, dict] = {}

    for name, path in MODELS.items():
        print(f"[reeval] {name}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).to(args.device)
        model.eval()

        entry: dict = {"checkpoint": path}
        scored = evaluate(model, tokenizer, balanced, device=args.device, batch_size=8)
        per_record.setdefault(name, {})["balanced"] = scored.pop("per_record_nll")
        entry["balanced"] = {
            "overall": scored["overall"],
            "by_category": {k: v["nll_per_character"] for k, v in scored["by_category"].items()},
        }

        if name in KOREAN_MODELS:
            scored_korean = evaluate(model, tokenizer, korean, device=args.device, batch_size=8)
            per_record.setdefault(name, {})["korean"] = scored_korean.pop("per_record_nll")
            entry["korean"] = {
                "overall": scored_korean["overall"],
                "by_category": {k: v["nll_per_character"]
                                for k, v in scored_korean["by_category"].items()},
            }

        if name in GENERATION_MODELS:
            entry["generation"] = generate(model, tokenizer, args.device)

        results[name] = entry
        print(f"  balanced NLL/char {entry['balanced']['overall']['nll_per_character']:.4f}"
              + (f" | korean {entry['korean']['overall']['nll_per_character']:.4f}"
                 if "korean" in entry else ""), flush=True)

        del model
        if args.device == "mps":
            torch.mps.empty_cache()

    # --- do the earlier conclusions survive? --------------------------------- #
    def nll(name: str, which: str = "balanced") -> float:
        return results[name][which]["overall"]["nll_per_character"]

    init_ranking = sorted(
        ["transplant_mean", "transplant_length_weighted", "transplant_norm_matched"], key=nll)
    phase1_ranking = sorted(
        ["phase1_mean_decomposed_only", "phase1_length_weighted_decomposed_only",
         "phase1_norm_matched_decomposed_only"], key=nll)

    comparisons = {
        "phase1_length_weighted_minus_mean": bootstrap_difference(
            balanced, per_record["phase1_length_weighted_decomposed_only"]["balanced"],
            per_record["phase1_mean_decomposed_only"]["balanced"]),
        "phase1_norm_matched_minus_mean": bootstrap_difference(
            balanced, per_record["phase1_norm_matched_decomposed_only"]["balanced"],
            per_record["phase1_mean_decomposed_only"]["balanced"]),
        "decomposed_only_minus_all_rows": bootstrap_difference(
            balanced, per_record["phase1_mean_decomposed_only"]["balanced"],
            per_record["stage1_2048k_all_rows"]["balanced"]),
        "stage1_minus_extended": bootstrap_difference(
            balanced, per_record["stage1_2048k_all_rows"]["balanced"],
            per_record["stage1_extended_4096k_all_rows"]["balanced"]),
    }
    if "korean" in per_record.get("stage1_2048k_all_rows", {}):
        comparisons["korean_stage1_minus_extended"] = bootstrap_difference(
            korean, per_record["stage1_2048k_all_rows"]["korean"],
            per_record["stage1_extended_4096k_all_rows"]["korean"])

    transplant = nll("transplant_mean")
    original = nll("original_qwen")
    gap = transplant - original
    curve = [
        {"tokens": 0, "nll_per_character": transplant},
        {"tokens": 2_048_000, "nll_per_character": nll("stage1_2048k_all_rows")},
        {"tokens": 4_096_000, "nll_per_character": nll("stage1_extended_4096k_all_rows")},
    ]
    for point in curve:
        point["recovery_ratio"] = (transplant - point["nll_per_character"]) / gap if gap else None

    report = {
        "stage": "decontaminated_reevaluation",
        "training_performed": False,
        "balanced_eval": args.balanced,
        "korean_eval": args.korean,
        "balanced_records": len(balanced),
        "korean_records": len(korean),
        "results": results,
        "conclusions": {
            "initialization_ranking": init_ranking,
            "phase1_ranking": phase1_ranking,
            "mean_still_best_at_init": init_ranking[0] == "transplant_mean",
            "mean_still_best_after_phase1": phase1_ranking[0] == "phase1_mean_decomposed_only",
            "all_rows_still_beats_decomposed_only":
                nll("stage1_2048k_all_rows") < nll("phase1_mean_decomposed_only"),
        },
        "bootstrap": comparisons,
        "recovery_curve": curve,
        "reference": {"original_qwen_nll_per_character": original,
                      "transplant_nll_per_character": transplant,
                      "recoverable_gap": gap},
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps({"conclusions": report["conclusions"], "curve": curve},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
