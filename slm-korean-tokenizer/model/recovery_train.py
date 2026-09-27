#!/usr/bin/env python3
"""Stage-1 embedding-only recovery training for the KO-32K transplant.

Every transformer parameter is frozen. Only the input embedding trains, and
because `tie_word_embeddings=true` the LM head is the same tensor, so one
parameter covers both.

Recovery modes:
  all_rows         every one of the 32,000 rows updates
  decomposed_only  exact-copy rows are held fixed with a gradient mask, so only
                   the decomposition-initialized rows move

This stage never unfreezes the transformer and never touches NIKL.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent))

from recovery_eval import embedding_norm_stats, evaluate, load_eval  # noqa: E402
from row_groups import load_row_groups  # noqa: E402

ALL_ROWS = "all_rows"
DECOMPOSED_ONLY = "decomposed_only"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def build_blocks(tokenizer, records: list[dict], sequence_length: int, seed: int,
                 separator_id: int) -> torch.Tensor:
    """Pack documents into fixed-length blocks separated by an explicit token.

    Documents are shuffled deterministically, tokenized, and concatenated with
    `separator_id` between them so the model sees an explicit boundary rather
    than an implicit splice. Only whole blocks are kept, so no padding is
    produced and every position carries a real label.
    """
    ordered = list(records)
    random.Random(seed).shuffle(ordered)
    stream: list[int] = []
    for record in ordered:
        ids = tokenizer(record["text"], add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        stream.extend(ids)
        stream.append(separator_id)
    usable = (len(stream) // sequence_length) * sequence_length
    if usable == 0:
        raise RuntimeError("training corpus is shorter than one sequence block")
    return torch.tensor(stream[:usable], dtype=torch.long).view(-1, sequence_length)


# --------------------------------------------------------------------------- #
# Model setup
# --------------------------------------------------------------------------- #

def freeze_transformer(model) -> dict:
    """Freeze everything, then re-enable the (tied) embedding only."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    embedding = model.get_input_embeddings().weight
    embedding.requires_grad_(True)

    head = model.get_output_embeddings()
    tied = head is not None and head.weight.data_ptr() == embedding.data_ptr()
    trainable = [p for p in model.parameters() if p.requires_grad]
    # A tied head must not appear as a second optimizer parameter.
    unique = {p.data_ptr(): p for p in trainable}
    return {
        "tied": tied,
        "trainable_tensors": len(trainable),
        "unique_trainable_tensors": len(unique),
        "trainable_parameters": sum(p.numel() for p in unique.values()),
        "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "parameters": list(unique.values()),
    }


def install_row_mask(embedding: torch.Tensor, trainable_rows: list[int], device: str):
    """Zero the gradient of every row outside `trainable_rows`.

    PyTorch has no per-row `requires_grad`, so the mask is applied to the
    gradient as it is produced. With `weight_decay=0` and Adam's zero-gradient
    state, a masked row's update is exactly zero.
    """
    mask = torch.zeros(embedding.shape[0], 1, dtype=embedding.dtype, device=device)
    mask[torch.tensor(trainable_rows, dtype=torch.long, device=device)] = 1.0
    handle = embedding.register_hook(lambda grad: grad * mask)
    return mask, handle


# --------------------------------------------------------------------------- #
# Tracking
# --------------------------------------------------------------------------- #

def row_drift(current: torch.Tensor, reference: torch.Tensor, rows: list[int]) -> dict:
    if not rows:
        return {"rows": 0}
    index = torch.tensor(rows, dtype=torch.long)
    a = current[index].to(torch.float32)
    b = reference[index].to(torch.float32)
    l2 = (a - b).norm(dim=1)
    cosine = torch.nn.functional.cosine_similarity(a, b, dim=1)
    return {
        "rows": len(rows),
        "l2_mean": float(l2.mean()), "l2_max": float(l2.max()),
        "cosine_mean": float(cosine.mean()), "cosine_min": float(cosine.min()),
        "unchanged_rows": int((l2 == 0).sum()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--mode", choices=[ALL_ROWS, DECOMPOSED_ONLY], default=DECOMPOSED_ONLY)
    parser.add_argument("--train-file", default="data/processed/recovery_train.jsonl")
    parser.add_argument("--eval-file", default="data/eval/recovery_balanced_eval.jsonl")
    parser.add_argument("--report-dir", default="outputs/recovery")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--eval-steps", type=int, nargs="*", default=[0, 50, 100, 250, 500, 1000])
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--save-dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--row-tokenizer", default="outputs/tokenizer-ko-32000-mixed10-structured-v2")
    # --- resume ------------------------------------------------------------- #
    parser.add_argument("--resume-tokens", type=int, default=0,
                        help="Tokens already trained before this run; used for cumulative accounting.")
    parser.add_argument("--resume-state", default=None,
                        help="optimizer_state.pt written by an earlier run, for an exact resume.")
    parser.add_argument("--schedule-total-steps", type=int, default=None,
                        help="Total steps the LR schedule spans (default: --max-steps). Set this to "
                             "the combined step count so a resumed run continues one cosine curve.")
    parser.add_argument("--schedule-start-step", type=int, default=0,
                        help="Fast-forward the schedule to this step before training.")
    parser.add_argument("--resume-block-cursor", type=int, default=0,
                        help="Blocks already consumed, so a resumed run does not repeat them.")
    parser.add_argument("--drift-reference", default=None,
                        help="Checkpoint whose embedding drift is measured against "
                             "(default: the starting weights of this run).")
    parser.add_argument("--save-optimizer", action="store_true",
                        help="Write optimizer_state.pt so a later run can resume exactly.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # fp32 master weights: MPS bf16 updates at lr 1e-4 lose too much precision.
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(args.device)

    groups = load_row_groups(args.base_model, args.row_tokenizer)
    exact_rows, decomposed_rows = groups["exact"], groups["decomposed"]

    setup = freeze_transformer(model)
    parameters = setup.pop("parameters")
    if not setup["tied"]:
        raise RuntimeError("expected tie_word_embeddings=true; the LM head is not tied")
    if setup["unique_trainable_tensors"] != 1:
        raise RuntimeError(f"expected exactly one trainable tensor, got {setup['unique_trainable_tensors']}")

    embedding = model.get_input_embeddings().weight
    if args.drift_reference:
        from safetensors.torch import load_file

        reference = load_file(
            Path(args.drift_reference) / "model.safetensors"
        )["model.embed_tokens.weight"].to(torch.float32)
    else:
        reference = embedding.detach().clone().cpu()

    if args.mode == DECOMPOSED_ONLY:
        trainable_rows = decomposed_rows
        mask, handle = install_row_mask(embedding, trainable_rows, args.device)
    else:
        trainable_rows = list(range(embedding.shape[0]))
        mask, handle = None, None

    separator_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    blocks = build_blocks(tokenizer, [json.loads(l) for l in Path(args.train_file).open(encoding="utf-8") if l.strip()],
                          args.sequence_length, args.seed, separator_id)
    eval_records = load_eval(args.eval_file)

    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate,
                                  weight_decay=args.weight_decay, betas=(0.9, 0.95), eps=1e-8)
    schedule_total = args.schedule_total_steps or args.max_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(schedule_total * args.warmup_ratio),
        num_training_steps=schedule_total,
    )
    resume_state = None
    if args.resume_state and Path(args.resume_state).exists():
        resume_state = torch.load(args.resume_state, map_location=args.device, weights_only=False)
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        print(f"[{args.run_name}] exact resume from {args.resume_state}", flush=True)
    elif args.schedule_start_step:
        # No optimizer state to restore: advance the schedule so this run continues
        # the same cosine curve instead of restarting it at the peak learning rate.
        for _ in range(args.schedule_start_step):
            scheduler.step()
        print(f"[{args.run_name}] schedule fast-forwarded to step {args.schedule_start_step}; "
              f"starting lr {scheduler.get_last_lr()[0]:.3e}", flush=True)
    start_lr = scheduler.get_last_lr()[0]

    eval_points = sorted({s for s in args.eval_steps if s <= args.max_steps} | {args.max_steps})
    history: list[dict] = []
    tokens_seen = args.resume_tokens

    def snapshot(step: int, loss: float | None) -> dict:
        result = evaluate(model, tokenizer, eval_records, device=args.device,
                          batch_size=args.eval_batch_size)
        result.pop("per_record_nll", None)
        current = embedding.detach().cpu()
        entry = {
            "step": step,
            "tokens_seen": tokens_seen,
            "tokens_this_run": tokens_seen - args.resume_tokens,
            "learning_rate": scheduler.get_last_lr()[0],
            "train_loss": loss,
            "nll_per_character": result["overall"]["nll_per_character"],
            "perplexity": result["overall"]["perplexity"],
            "mean_loss_per_token": result["overall"]["mean_loss_per_token"],
            "by_category": {k: v["nll_per_character"] for k, v in result["by_category"].items()},
            "embedding_norms": embedding_norm_stats(model, groups),
            "exact_row_drift": row_drift(current, reference, exact_rows[::8]),
            "decomposed_row_drift": row_drift(current, reference, decomposed_rows[::24]),
        }
        history.append(entry)
        print(f"[{args.run_name}] step {step:4d} tokens {tokens_seen:>9,} "
              f"NLL/char {entry['nll_per_character']:.4f} "
              f"decomp_norm {entry['embedding_norms']['decomposed']['mean']:.4f} "
              f"exact_drift {entry['exact_row_drift']['l2_mean']:.2e}", flush=True)
        return entry

    snapshot(0, None)

    model.train()
    order = torch.randperm(blocks.shape[0], generator=torch.Generator().manual_seed(args.seed))
    cursor = args.resume_block_cursor
    if cursor:
        print(f"[{args.run_name}] skipping {cursor} already-consumed blocks", flush=True)
    started = time.time()
    for step in range(1, args.max_steps + 1):
        if cursor + args.batch_size > order.numel():
            order = torch.randperm(blocks.shape[0], generator=torch.Generator().manual_seed(args.seed + step))
            cursor = 0
        batch = blocks[order[cursor:cursor + args.batch_size]].to(args.device)
        cursor += args.batch_size

        output = model(input_ids=batch, labels=batch)
        loss = output.loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}: {loss}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        tokens_seen += batch.numel()

        if step in eval_points:
            snapshot(step, float(loss.detach()))
            model.train()
        elif step % 25 == 0:
            print(f"[{args.run_name}] step {step:4d} loss {float(loss):.4f} "
                  f"lr {scheduler.get_last_lr()[0]:.2e}", flush=True)

    elapsed = time.time() - started
    if handle is not None:
        handle.remove()

    if not torch.isfinite(embedding.detach()).all():
        raise FloatingPointError("embedding contains NaN or Inf after recovery")

    report = {
        "run_name": args.run_name,
        "init_model": args.model,
        "mode": args.mode,
        "hyperparameters": {
            "max_steps": args.max_steps, "batch_size": args.batch_size,
            "sequence_length": args.sequence_length, "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay, "warmup_ratio": args.warmup_ratio,
            "scheduler": "cosine", "optimizer": "AdamW", "betas": [0.9, 0.95],
            "grad_clip": 1.0, "seed": args.seed, "device": args.device,
            "compute_dtype": "float32", "save_dtype": args.save_dtype,
            "document_separator": "<|endoftext|>",
        },
        "corpus": {
            "train_file": args.train_file, "blocks": int(blocks.shape[0]),
            "tokens_available": int(blocks.numel()),
            "tokens_trained": tokens_seen,
            "tokens_this_run": tokens_seen - args.resume_tokens,
            "tokens_before_run": args.resume_tokens,
            "epochs": tokens_seen / int(blocks.numel()),
            "block_cursor_start": args.resume_block_cursor,
            "block_cursor_end": cursor,
        },
        "resume": {
            "resumed_from": args.model,
            "exact_optimizer_resume": resume_state is not None,
            "schedule_total_steps": schedule_total,
            "schedule_start_step": args.schedule_start_step,
            "starting_learning_rate": start_lr,
            "final_learning_rate": scheduler.get_last_lr()[0],
            "drift_reference": args.drift_reference or args.model,
        },
        "parameters": {
            **setup,
            "updatable_rows": len(trainable_rows),
            "updatable_parameters": len(trainable_rows) * int(embedding.shape[1]),
        },
        "elapsed_seconds": elapsed,
        "history": history,
    }

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{args.run_name}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if args.output_dir:
        model.to("cpu")
        if args.save_dtype == "bfloat16":
            model = model.to(torch.bfloat16)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.save_pretrained(output_dir)
        if args.save_optimizer:
            torch.save(
                {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "tokens_seen": tokens_seen, "block_cursor": cursor,
                 "schedule_total_steps": schedule_total},
                output_dir / "optimizer_state.pt",
            )
        (output_dir / "recovery_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"saved recovery checkpoint to {output_dir}")

    print(f"[{args.run_name}] done in {elapsed / 60:.1f} min; "
          f"final NLL/char {history[-1]['nll_per_character']:.4f}")


if __name__ == "__main__":
    main()
