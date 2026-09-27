#!/usr/bin/env python3
"""Stage-2 continued pretraining: full-model training on the KO-32K corpus.

Stage 1 trained only the tied embedding while every transformer block stayed
frozen. Stage 2 unfreezes everything and trains on the v3 corpus for a fixed
token budget.

Deliberate properties:
  * resumes weights from the Stage-1 extended checkpoint, nothing else
  * refuses to reuse Stage-1's optimizer state, which is embedding-only Adam
    moments and meaningless for a full-model run
  * differential learning rates: the embedding is 4.096M tokens ahead of the
    blocks, so the blocks move more slowly
  * budget is counted in tokens, not epochs, and the corpus is never repeated
"""
from __future__ import annotations

import argparse
import json
import math
import random
import signal
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent))

from recovery_eval import evaluate, load_eval  # noqa: E402

STAGE1 = "outputs/models/qwen3-0.6b-ko32k-recovery-stage1-extended"
CORPUS = "data/processed/stage2_train_v3.jsonl"

# Set by the SIGUSR1 handler so a running job can be asked to checkpoint and
# release cached device memory without being killed.
_FLUSH_REQUESTED = False


def _request_flush(signum, frame) -> None:      # noqa: ARG001
    global _FLUSH_REQUESTED
    _FLUSH_REQUESTED = True
    print("[stage2] SIGUSR1 received: will checkpoint and flush caches", flush=True)


def release_cache(device: str) -> None:
    """Return cached blocks to the OS.

    The MPS caching allocator never releases on its own. Evaluation builds
    length-sorted batches of many different shapes, so every eval opens new
    allocation buckets; across eight evals this grew to 52 GB of device memory
    for a model needing about 8 GB, and the run thrashed into swap and stalled.
    """
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def device_memory_gb(device: str) -> float:
    """Driver-level allocation, not live-tensor bytes.

    `torch.mps.current_allocated_memory()` counts only tensors that are still
    referenced and hides the allocator's reserved pool. It reported 8 GB while
    the process footprint was 33 GB, which is why an earlier memory problem went
    unnoticed. `driver_allocated_memory()` is the number that matches what
    Activity Monitor shows.
    """
    if device == "mps":
        if hasattr(torch.mps, "driver_allocated_memory"):
            return torch.mps.driver_allocated_memory() / 1e9
        return torch.mps.current_allocated_memory() / 1e9
    if device == "cuda":
        return torch.cuda.memory_reserved() / 1e9
    return 0.0


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def build_blocks(tokenizer, corpus: str, sequence_length: int, seed: int,
                 separator_id: int, budget_tokens: int, cache: str | None) -> torch.Tensor:
    """Pack documents into fixed blocks. Only whole blocks, so no padding."""
    if cache and Path(cache).exists():
        blocks = torch.load(cache, map_location="cpu")
        print(f"[stage2] loaded packed blocks from {cache}: {tuple(blocks.shape)}", flush=True)
        return blocks

    records = [json.loads(line) for line in Path(corpus).open(encoding="utf-8") if line.strip()]
    random.Random(seed).shuffle(records)
    needed = int(budget_tokens * 1.02)  # small margin for the separator tokens
    stream: list[int] = []
    for record in records:
        ids = tokenizer(record["text"], add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        stream.extend(ids)
        stream.append(separator_id)
        if len(stream) >= needed:
            break
    usable = (len(stream) // sequence_length) * sequence_length
    blocks = torch.tensor(stream[:usable], dtype=torch.long).view(-1, sequence_length)
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        torch.save(blocks, cache)
    print(f"[stage2] packed {blocks.shape[0]:,} blocks x {sequence_length} "
          f"= {blocks.numel():,} tokens", flush=True)
    return blocks


# --------------------------------------------------------------------------- #
# Model setup
# --------------------------------------------------------------------------- #

def unfreeze_all(model) -> dict:
    """Unfreeze every parameter and report what is actually trainable."""
    for parameter in model.parameters():
        parameter.requires_grad_(True)

    embedding = model.get_input_embeddings().weight
    head = model.get_output_embeddings()
    tied = head is not None and head.weight.data_ptr() == embedding.data_ptr()

    unique: dict[int, torch.nn.Parameter] = {}
    for parameter in model.parameters():
        if parameter.requires_grad:
            unique.setdefault(parameter.data_ptr(), parameter)
    frozen = [name for name, p in model.named_parameters() if not p.requires_grad]
    return {
        "tied_embedding_preserved": tied,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in unique.values()),
        "unique_trainable_tensors": len(unique),
        "frozen_parameters": sum(p.numel() for n, p in model.named_parameters()
                                 if not p.requires_grad),
        "frozen_tensor_names": frozen,
        "embedding_parameters": embedding.numel(),
        "non_embedding_parameters": sum(p.numel() for p in unique.values()) - embedding.numel(),
    }


def parameter_groups(model, embedding_lr: float, transformer_lr: float,
                     weight_decay: float) -> list[dict]:
    """Four groups: embedding vs blocks, each split by decay eligibility.

    Biases, norms and the embedding table get no weight decay — decaying a
    lookup table pulls unseen rows toward zero for no benefit.
    """
    embedding = model.get_input_embeddings().weight
    embedding_ptr = embedding.data_ptr()
    seen: set[int] = set()
    groups = {
        "embedding": {"params": [], "lr": embedding_lr, "weight_decay": 0.0},
        "blocks_decay": {"params": [], "lr": transformer_lr, "weight_decay": weight_decay},
        "blocks_no_decay": {"params": [], "lr": transformer_lr, "weight_decay": 0.0},
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.data_ptr() in seen:
            continue
        seen.add(parameter.data_ptr())
        if parameter.data_ptr() == embedding_ptr:
            groups["embedding"]["params"].append(parameter)
        elif parameter.ndim < 2 or "norm" in name.lower() or name.endswith(".bias"):
            groups["blocks_no_decay"]["params"].append(parameter)
        else:
            groups["blocks_decay"]["params"].append(parameter)
    return [dict(name=key, **value) for key, value in groups.items() if value["params"]]


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=STAGE1)
    parser.add_argument("--corpus", default=CORPUS)
    parser.add_argument("--output-dir", default="outputs/models/qwen3-0.6b-ko32k-stage2")
    parser.add_argument("--run-name", default="stage2-continued-pretraining")
    parser.add_argument("--report-dir", default="outputs/stage2")
    parser.add_argument("--budget-tokens", type=int, default=60_000_000)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--embedding-lr", type=float, default=5e-5)
    parser.add_argument("--transformer-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.02)
    parser.add_argument("--min-lr-ratio", type=float, default=0.10)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-file", default="data/eval/recovery_balanced_eval_decontaminated.jsonl")
    parser.add_argument("--holdout-file", default="data/processed/stage2_holdout_v3.jsonl")
    parser.add_argument("--holdout-limit", type=int, default=400)
    parser.add_argument("--block-cache", default="data/processed/stage2_blocks_1024.pt")
    parser.add_argument("--max-steps", type=int, default=0, help="Smoke-test override.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--gradient-checkpointing", dest="gradient_checkpointing",
                        action="store_true", default=True,
                        help="Recompute activations in backward. Cuts the MPS footprint from "
                             "32.9 GB to 11.9 GB at micro-batch 4, for ~17%% less throughput.")
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                        action="store_false")
    parser.add_argument("--flush-every", type=int, default=100,
                        help="Release cached device memory every N optimizer steps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    signal.signal(signal.SIGUSR1, _request_flush)

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)

    # --- load Stage-1 weights and tokenizer -------------------------------- #
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(args.device)

    stage1_optimizer = Path(args.model) / "optimizer_state.pt"
    optimizer_state_ignored = stage1_optimizer.exists()

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        print("[stage2] gradient checkpointing enabled", flush=True)

    setup = unfreeze_all(model)
    if not setup["tied_embedding_preserved"]:
        raise RuntimeError("tie_word_embeddings was lost when loading the checkpoint")
    if setup["frozen_parameters"] != 0:
        raise RuntimeError(f"{setup['frozen_parameters']:,} parameters are still frozen: "
                           f"{setup['frozen_tensor_names'][:5]}")
    if len(tokenizer) != 32000 or int(model.config.vocab_size) != 32000:
        raise RuntimeError("vocabulary is not the KO-32K vocabulary")

    print(f"[stage2] total {setup['total_parameters']:,} | "
          f"trainable {setup['trainable_parameters']:,} | "
          f"frozen {setup['frozen_parameters']:,} | "
          f"tied={setup['tied_embedding_preserved']}", flush=True)
    print(f"[stage2] Stage-1 optimizer_state.pt present={optimizer_state_ignored}; "
          f"deliberately NOT reused (embedding-only Adam moments)", flush=True)

    # --- token / step arithmetic -------------------------------------------- #
    tokens_per_micro = args.micro_batch * args.sequence_length
    tokens_per_step = tokens_per_micro * args.grad_accum
    max_steps = args.max_steps or (args.budget_tokens // tokens_per_step)
    planned_tokens = max_steps * tokens_per_step
    warmup_steps = max(1, int(max_steps * args.warmup_ratio))
    print(f"[stage2] {tokens_per_micro:,} tok/micro x{args.grad_accum} = {tokens_per_step:,} "
          f"tok/step | {max_steps:,} steps | {planned_tokens:,} tokens "
          f"(budget {args.budget_tokens:,}, delta {planned_tokens - args.budget_tokens:+,})",
          flush=True)

    separator_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    blocks = build_blocks(tokenizer, args.corpus, args.sequence_length, args.seed,
                          separator_id, planned_tokens,
                          None if args.smoke else args.block_cache)
    needed_blocks = max_steps * args.micro_batch * args.grad_accum
    if blocks.shape[0] < needed_blocks:
        raise RuntimeError(f"packed {blocks.shape[0]:,} blocks but need {needed_blocks:,}; "
                           "the corpus would have to repeat")
    print(f"[stage2] using {needed_blocks:,} of {blocks.shape[0]:,} blocks "
          f"(no repetition)", flush=True)

    # --- fresh optimizer over the unfrozen model ---------------------------- #
    groups = parameter_groups(model, args.embedding_lr, args.transformer_lr, args.weight_decay)
    optimizer = torch.optim.AdamW(
        [{k: v for k, v in group.items() if k != "name"} for group in groups],
        betas=(args.beta1, args.beta2), eps=args.eps,
    )
    group_summary = [
        {"name": group["name"], "tensors": len(group["params"]),
         "parameters": sum(p.numel() for p in group["params"]),
         "lr": group["lr"], "weight_decay": group["weight_decay"]}
        for group in groups
    ]
    for entry in group_summary:
        print(f"[stage2]   group {entry['name']:17s} tensors {entry['tensors']:4d} "
              f"params {entry['parameters']:>12,} lr {entry['lr']:.1e} "
              f"wd {entry['weight_decay']}", flush=True)

    # Cosine to a floor rather than to zero, so the run stays extensible.
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    eval_records = load_eval(args.eval_file)
    holdout_records = []
    if Path(args.holdout_file).exists():
        # The corpus schema names the length field `chars`; the evaluator expects
        # `characters`. Normalize rather than loosening the evaluator.
        for line in Path(args.holdout_file).open(encoding="utf-8"):
            if not line.strip():
                continue
            record = json.loads(line)
            record.setdefault("characters", record.get("chars", len(record["text"])))
            record.setdefault("category", "holdout")
            holdout_records.append(record)
            if len(holdout_records) >= args.holdout_limit:
                break

    history: list[dict] = []
    start_step = 0
    tokens_seen = 0
    if args.resume and Path(args.resume).exists():
        state = torch.load(args.resume, map_location=args.device, weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step = state["step"]
        tokens_seen = state["tokens_seen"]
        history = state.get("history", [])
        print(f"[stage2] resumed at step {start_step:,}, {tokens_seen:,} tokens", flush=True)

    def snapshot(step: int, loss: float | None, grad_norm: float | None) -> dict:
        result = evaluate(model, tokenizer, eval_records, device=args.device, batch_size=4)
        result.pop("per_record_nll", None)
        entry = {
            "step": step, "tokens": tokens_seen, "train_loss": loss,
            "grad_norm": grad_norm,
            "eval_nll_per_character": result["overall"]["nll_per_character"],
            "eval_loss_per_token": result["overall"]["mean_loss_per_token"],
            "eval_perplexity": result["overall"]["perplexity"],
            "eval_by_category": {k: v["nll_per_character"]
                                 for k, v in result["by_category"].items()},
        }
        if holdout_records:
            held = evaluate(model, tokenizer, holdout_records, device=args.device, batch_size=4)
            held.pop("per_record_nll", None)
            entry["holdout_nll_per_character"] = held["overall"]["nll_per_character"]
            entry["holdout_loss_per_token"] = held["overall"]["mean_loss_per_token"]
        history.append(entry)
        release_cache(args.device)
        entry["device_memory_gb"] = device_memory_gb(args.device)
        (report_dir / f"{args.run_name}-history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[stage2] EVAL step {step:,} tokens {tokens_seen:,} "
              f"eval_nll/char {entry['eval_nll_per_character']:.4f}"
              + (f" holdout {entry['holdout_nll_per_character']:.4f}"
                 if holdout_records else ""), flush=True)
        model.train()
        return entry

    def save(tag: str, step: int) -> str:
        path = output_dir.parent / f"{output_dir.name}-{tag}" if tag != "final" else output_dir
        path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(path, safe_serialization=True)
        tokenizer.save_pretrained(path)
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.save_pretrained(path)
        torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "step": step, "tokens_seen": tokens_seen, "history": history},
                   path / "optimizer_state.pt")
        (path / "stage2_state.json").write_text(json.dumps({
            "step": step, "tokens_seen": tokens_seen, "run_name": args.run_name,
            "stage": "stage2_continued_pretraining", "tag": tag,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        release_cache(args.device)
        print(f"[stage2] saved checkpoint '{tag}' -> {path} "
              f"(device memory {device_memory_gb(args.device):.1f} GB)", flush=True)
        return str(path)

    config = {
        "run_name": args.run_name,
        "stage": "stage2_continued_pretraining",
        "resumed_weights_from": args.model,
        "stage1_optimizer_state_present": optimizer_state_ignored,
        "stage1_optimizer_state_reused": False,
        "corpus": args.corpus,
        "tokenizer": args.model,
        "vocab_size": len(tokenizer),
        "sequence_length": args.sequence_length,
        "micro_batch": args.micro_batch,
        "grad_accum": args.grad_accum,
        "tokens_per_micro_batch": tokens_per_micro,
        "tokens_per_optimizer_step": tokens_per_step,
        "budget_tokens": args.budget_tokens,
        "planned_tokens": planned_tokens,
        "token_delta_vs_budget": planned_tokens - args.budget_tokens,
        "max_steps": max_steps,
        "warmup_steps": warmup_steps,
        "embedding_lr": args.embedding_lr,
        "transformer_lr": args.transformer_lr,
        "differential_lr_ratio": args.embedding_lr / args.transformer_lr,
        "weight_decay": args.weight_decay,
        "betas": [args.beta1, args.beta2],
        "eps": args.eps,
        "grad_clip": args.grad_clip,
        "scheduler": f"cosine to {args.min_lr_ratio:.0%} of peak after {warmup_steps} warmup steps",
        "precision": "float32",
        "gradient_checkpointing": args.gradient_checkpointing,
        "seed": args.seed,
        "device": args.device,
        "packed_blocks_available": int(blocks.shape[0]),
        "packed_blocks_used": needed_blocks,
        "corpus_repeated": False,
        "parameters": setup,
        "parameter_groups": group_summary,
        "eval_file": args.eval_file,
        "holdout_file": args.holdout_file,
    }
    (report_dir / f"{args.run_name}-config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # --- training loop ------------------------------------------------------- #
    model.train()
    checkpoints: list[dict] = []
    cursor = start_step * args.micro_batch * args.grad_accum
    started = time.time()

    if start_step == 0:
        snapshot(0, None, None)

    for step in range(start_step + 1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(args.grad_accum):
            batch = blocks[cursor:cursor + args.micro_batch].to(args.device)
            cursor += args.micro_batch
            output = model(input_ids=batch, labels=batch)
            loss = output.loss / args.grad_accum
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            loss.backward()
            step_loss += float(loss.detach())
            tokens_seen += batch.numel()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))
        if not math.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite gradient norm at step {step}")
        optimizer.step()
        scheduler.step()

        global _FLUSH_REQUESTED
        if args.flush_every and step % args.flush_every == 0:
            release_cache(args.device)
        if _FLUSH_REQUESTED:
            _FLUSH_REQUESTED = False
            checkpoints.append({"tag": f"signal{step}", "step": step, "tokens": tokens_seen,
                                "path": save(f"signal{step}", step)})

        if step % args.log_every == 0:
            elapsed = time.time() - started
            done = step - start_step
            rate = (done * tokens_per_step) / elapsed if elapsed else 0.0
            remaining = (max_steps - step) * tokens_per_step / rate if rate else 0.0
            print(f"[stage2] step {step:,}/{max_steps:,} tokens {tokens_seen:,} "
                  f"loss {step_loss:.4f} lr {scheduler.get_last_lr()[0]:.2e} "
                  f"gnorm {grad_norm:.3f} {rate:.0f} tok/s "
                  f"mem {device_memory_gb(args.device):.1f}GB "
                  f"eta {remaining / 3600:.1f}h", flush=True)
        if args.eval_every and step % args.eval_every == 0:
            snapshot(step, step_loss, grad_norm)
        if args.checkpoint_every and step % args.checkpoint_every == 0 and step < max_steps:
            tag = f"step{step}"
            checkpoints.append({"tag": tag, "step": step, "tokens": tokens_seen,
                                "path": save(tag, step)})

    final_entry = snapshot(max_steps, None, None)
    checkpoints.append({"tag": "final", "step": max_steps, "tokens": tokens_seen,
                        "path": save("final", max_steps)})

    summary = {
        **config,
        "completed": True,
        "tokens_consumed": tokens_seen,
        "elapsed_seconds": time.time() - started,
        "checkpoints": checkpoints,
        "final": final_entry,
        "history": history,
    }
    (report_dir / f"{args.run_name}-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[stage2] done: {tokens_seen:,} tokens in "
          f"{(time.time() - started) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
