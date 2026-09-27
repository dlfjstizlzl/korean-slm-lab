#!/usr/bin/env python3
"""Aggregate the Stage-1 recovery ablation into one report.

Collects the per-run training histories, re-evaluates the final checkpoints with
per-record NLL so strategies can be compared with a paired bootstrap, runs a
short generation smoke test, and picks a winner.
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

STRATEGIES = ["mean", "length_weighted", "norm_matched"]

PROMPTS = {
    "korean": "가상 메모리는 운영체제에서",
    "technical": "Apple Silicon의 unified memory는",
    "skill": "<검색>Apple Silicon 통합 메모리 구조",
    "json": '{"skill":"search","input":"',
}


def generation_smoke(model, tokenizer, device: str, max_new_tokens: int = 40) -> dict:
    """Structural sanity only: repetition, broken UTF-8, token collapse."""
    out = {}
    for name, prompt in PROMPTS.items():
        encoded = tokenizer(prompt, return_tensors="pt").to(device)
        try:
            with torch.inference_mode():
                generated = model.generate(
                    **encoded, max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
        except Exception as error:  # noqa: BLE001
            out[name] = {"passed": False, "error": repr(error)}
            continue
        ids = generated[0].tolist()
        new_ids = ids[int(encoded["input_ids"].shape[1]):]
        text = tokenizer.decode(ids, skip_special_tokens=True)
        distinct = len(set(new_ids))
        # A single token repeated over and over is the collapse we care about.
        longest_run = 1
        run = 1
        for previous, current in zip(new_ids, new_ids[1:]):
            run = run + 1 if current == previous else 1
            longest_run = max(longest_run, run)
        out[name] = {
            "prompt": prompt,
            "text": text,
            "new_tokens": len(new_ids),
            "distinct_tokens": distinct,
            "longest_repeat_run": longest_run,
            "replacement_chars": text.count("�"),
            "ids_in_range": all(0 <= i < len(tokenizer) for i in ids),
            "passed": all(0 <= i < len(tokenizer) for i in ids) and text.count("�") == 0,
            "degenerate_repetition": longest_run >= 8 or distinct <= max(2, len(new_ids) // 10),
        }
    return out


def full_eval(model_dir: str, records: list[dict], device: str, groups: dict | None,
              batch_size: int = 8) -> tuple[dict, dict, dict]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).to(device)
    result = evaluate(model, tokenizer, records, device=device, batch_size=batch_size)
    per_record = result.pop("per_record_nll")
    norms = embedding_norm_stats(model, groups) if groups else {}
    generation = generation_smoke(model, tokenizer, device)
    del model
    if device == "mps":
        torch.mps.empty_cache()
    return {**result, "embedding_norms": norms}, per_record, generation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="outputs/recovery")
    parser.add_argument("--eval-file", default="data/eval/recovery_balanced_eval.jsonl")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--row-tokenizer", default="outputs/tokenizer-ko-32000-mixed10-structured-v2")
    parser.add_argument("--output-json", default="outputs/embedding_recovery_report.json")
    parser.add_argument("--phase2-runs", nargs="*", default=[],
                        help="Entries of the form run_name=checkpoint_dir.")
    parser.add_argument("--winner", default=None, help="Override the automatic Phase-1 winner.")
    parser.add_argument("--stage1-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    records = load_eval(args.eval_file)
    groups = load_row_groups(args.base_model, args.row_tokenizer)

    phase1_runs = {}
    for strategy in STRATEGIES:
        path = run_dir / f"phase1-{strategy.replace('_', '-')}-decomposed_only.json"
        if path.exists():
            phase1_runs[strategy] = json.loads(path.read_text(encoding="utf-8"))
    if not phase1_runs:
        raise SystemExit("no Phase-1 run reports found")

    print("[report] re-evaluating Phase-1 final checkpoints", flush=True)
    finals, per_record, generations = {}, {}, {}
    for strategy in phase1_runs:
        model_dir = f"outputs/models/recovery-phase1-{strategy.replace('_', '-')}"
        finals[strategy], per_record[strategy], generations[strategy] = full_eval(
            model_dir, records, args.device, groups
        )
        print(f"  {strategy}: {finals[strategy]['overall']['nll_per_character']:.4f}", flush=True)

    ranked = sorted(finals, key=lambda s: finals[s]["overall"]["nll_per_character"])
    winner = args.winner or ranked[0]

    comparisons = {}
    for other in ranked[1:]:
        comparisons[f"{other}_minus_{ranked[0]}"] = bootstrap_difference(
            records, per_record[other], per_record[ranked[0]]
        )

    phase2 = {}
    for entry in args.phase2_runs:
        name, _, model_dir = entry.partition("=")
        path = run_dir / f"{name}.json"
        if not path.exists():
            continue
        run = json.loads(path.read_text(encoding="utf-8"))
        if model_dir and Path(model_dir).is_dir():
            print(f"[report] re-evaluating {name}", flush=True)
            metrics, record_nll, generation = full_eval(model_dir, records, args.device, groups)
            per_record[name] = record_nll
            phase2[name] = {"run": run, "checkpoint": model_dir,
                            "final": metrics, "generation": generation}
        else:
            phase2[name] = {"run": run}

    original = json.loads((run_dir / "step0_original-qwen.json").read_text(encoding="utf-8"))
    step0 = {
        strategy: json.loads(
            (run_dir / f"step0_init-{strategy.replace('_', '-')}.json").read_text(encoding="utf-8")
        )
        for strategy in phase1_runs
    }

    original_nll = original["overall"]["nll_per_character"]
    transplant_nll = step0["mean"]["overall"]["nll_per_character"]
    recovered_nll = finals[winner]["overall"]["nll_per_character"]
    gap = transplant_nll - original_nll

    report = {
        "stage": "embedding_only_recovery_stage1",
        "warning": "Transformer blocks stayed frozen; this is not continued pretraining.",
        "eval_file": args.eval_file,
        "eval_characters": original["overall"]["characters"],
        "eval_records": original["evaluated_records"],
        "reference": {"original_qwen": original["overall"], "by_category": {
            k: v["nll_per_character"] for k, v in original["by_category"].items()}},
        "initialization_only": {
            strategy: {
                "nll_per_character": step0[strategy]["overall"]["nll_per_character"],
                "perplexity": step0[strategy]["overall"]["perplexity"],
                "mean_loss_per_token": step0[strategy]["overall"]["mean_loss_per_token"],
                "tokens_per_character": step0[strategy]["overall"]["tokens_per_character"],
                "by_category": {k: v["nll_per_character"]
                                for k, v in step0[strategy]["by_category"].items()},
                "embedding_norms": step0[strategy].get("embedding_norms", {}),
            }
            for strategy in step0
        },
        "phase1": {
            "mode": "decomposed_only",
            "runs": phase1_runs,
            "final": finals,
            "generation": generations,
            "ranking": ranked,
            "winner": winner,
            "bootstrap_vs_winner": comparisons,
        },
        "phase2": phase2,
        "final": {
            "winner_strategy": winner,
            "original_qwen_nll_per_character": original_nll,
            "transplant_nll_per_character": transplant_nll,
            "recovered_nll_per_character": recovered_nll,
            "recoverable_gap": gap,
            "recovered_amount": transplant_nll - recovered_nll,
            "recovery_ratio": (transplant_nll - recovered_nll) / gap if gap else None,
            "remaining_gap": recovered_nll - original_nll,
        },
        "stage1_checkpoint": args.stage1_dir,
    }

    if args.phase2_runs and phase2:
        best = min(
            (name for name in phase2 if "final" in phase2[name]),
            key=lambda n: phase2[n]["final"]["overall"]["nll_per_character"],
            default=None,
        )
        if best:
            best_nll = phase2[best]["final"]["overall"]["nll_per_character"]
            report["final"]["phase2_winner"] = best
            report["final"]["recovered_nll_per_character"] = best_nll
            report["final"]["recovered_amount"] = transplant_nll - best_nll
            report["final"]["recovery_ratio"] = (transplant_nll - best_nll) / gap if gap else None
            report["final"]["remaining_gap"] = best_nll - original_nll
            names = list(phase2)
            if len(names) == 2 and all(n in per_record for n in names):
                report["phase2"]["bootstrap"] = bootstrap_difference(
                    records, per_record[names[0]], per_record[names[1]]
                )

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["final"], ensure_ascii=False, indent=2))
    print(f"saved {args.output_json}")


if __name__ == "__main__":
    main()
