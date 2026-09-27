#!/usr/bin/env python3
"""Manager-SLM LoRA SFT on Qwen3-0.6B-KO32K-Stage2-signal2678 (train + validation only).

    python model/manager_sft_train.py stats                 # token length audit + rendered samples
    python model/manager_sft_train.py baseline              # Stage-2 base on validation (raw + constrained)
    python model/manager_sft_train.py train --run smoke     # pipeline smoke (subset, not a result)
    python model/manager_sft_train.py train --run full      # canonical run, 3 epochs
    python model/manager_sft_train.py train --run full --resume outputs/manager_sft_v1/checkpoints/epoch1

The sealed test split cannot be loaded without --allow-sealed-test, and no
subcommand here evaluates it.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import statistics
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "model"))

import manager_lora as lora  # noqa: E402
from manager_sft_data import (  # noqa: E402
    BASE_MODEL,
    BASE_MODEL_NAME,
    IGNORE_INDEX,
    SPLIT_FILES,
    assert_template_parity,
    encode_all,
    load_split,
    render_prompt,
    sealed_test_fingerprint,
    sha256_file,
    system_prompt,
)
from manager_sft_eval import MAX_NEW_TOKENS, evaluate, generate  # noqa: E402
from manager_v2_registry import load_contract  # noqa: E402

OUT = PROJECT_ROOT / "outputs" / "manager_sft_v1"
SEED = 20260927
CONFIG = {
    "base_model": BASE_MODEL_NAME,
    "base_model_path": str(BASE_MODEL.relative_to(PROJECT_ROOT)),
    "lora": {"rank": 16, "alpha": 32, "dropout": 0.05, "bias": "none",
             "targets": "every nn.Linear inside model.layers.* (discovered at runtime)",
             "frozen": "embeddings, tied lm_head, norms, all base weights"},
    "optimizer": {"type": "torch.optim.AdamW", "lr": 1e-4, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01},
    "scheduler": {"type": "linear warmup + cosine decay to 0", "warmup_fraction": 0.05},
    "grad_clip": 1.0,
    "epochs": 3,
    "effective_batch": 16,
    "micro_batch": 4,
    "loss": "token-mean cross entropy over assistant target tokens (+EOS) of each effective batch; lm_head is "
            "applied only at label positions (identical value, less memory)",
    "gradient_checkpointing": "non-reentrant torch checkpointing of every decoder layer (added after the first "
                              "smoke attempt ran out of MPS memory at micro batch 4; mathematically identical)",
    "mps_cache_release": "torch.mps.empty_cache() after every optimizer step (smoke showed allocator growth)",
    "seed": SEED,
    "dtype": "float32",
    "max_length": 4096,
    "generation": {"decoding": "greedy (temperature 0)", "max_new_tokens": MAX_NEW_TOKENS,
                   "eos": ["<|im_end|>", "<|endoftext|>"],
                   "constrained": "Runtime ManagerClient::json_grammar() via llama.cpp grammar engine "
                                  "(slm_taskplan_tool --grammar-session)"},
    "selection": {"fixed_before_training": True,
                  "primary": "constrained structural_exact (validation)",
                  "tie_breakers": ["constrained canonical_exact", "raw structural_exact", "lower validation loss"]},
    "smoke": {"train_samples": 384, "validation_samples": 96, "epochs": 1},
}


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def memory(device: str) -> dict:
    if device == "mps":
        return {"allocated_gb": round(torch.mps.current_allocated_memory() / 1e9, 2),
                "driver_gb": round(torch.mps.driver_allocated_memory() / 1e9, 2)}
    if device == "cuda":
        return {"allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
                "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    return {}


def empty_cache(device: str) -> None:
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def load_base(device: str):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    assert_template_parity(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32).to(device)
    model.config.use_cache = True
    return model, tokenizer


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def provenance() -> dict:
    contract = load_contract()
    import hashlib

    return {
        "base_model": BASE_MODEL_NAME,
        "base_model_weights_sha256": sha256_file(BASE_MODEL / "model.safetensors"),
        "tokenizer_json_sha256": sha256_file(BASE_MODEL / "tokenizer.json"),
        "train_sha256": sha256_file(SPLIT_FILES["train"]),
        "validation_sha256": sha256_file(SPLIT_FILES["validation"]),
        "sealed_test": sealed_test_fingerprint(),
        "system_prompt_sha256": hashlib.sha256(contract.system_prompt.encode("utf-8")).hexdigest(),
        "grammar_sha256": hashlib.sha256(contract.grammar.encode("utf-8")).hexdigest(),
        "runtime_contract_sources": contract.source_sha256,
    }


# --------------------------------------------------------------------------- #
# stats + rendered samples
# --------------------------------------------------------------------------- #

def describe(values):
    ordered = sorted(values)
    q = statistics.quantiles(ordered, n=100)
    return {"min": ordered[0], "median": statistics.median(ordered), "p95": round(q[94], 1), "p99": round(q[98], 1),
            "max": ordered[-1]}


def cmd_stats(args) -> None:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    assert_template_parity(tokenizer)
    report = {}
    for split in ("train", "validation"):
        encoded = encode_all(load_split(split), tokenizer, CONFIG["max_length"])
        report[split] = {"samples": len(encoded),
                         "input_tokens": describe([e.prompt_tokens for e in encoded]),
                         "target_tokens_with_eos": describe([e.target_tokens for e in encoded]),
                         "total_tokens": describe([len(e.input_ids) for e in encoded]),
                         "truncated_targets": 0,
                         "over_max_length": sum(len(e.input_ids) > CONFIG["max_length"] for e in encoded),
                         "sum_tokens": sum(len(e.input_ids) for e in encoded),
                         "sum_target_tokens": sum(e.target_tokens for e in encoded)}
    report["system_prompt_tokens"] = len(tokenizer(system_prompt(), add_special_tokens=False)["input_ids"])
    report["max_length"] = CONFIG["max_length"]
    write_json(OUT / "token_stats.json", report)

    rows = load_split("train")
    rng = random.Random(SEED)
    picks = rng.sample(rows, 10)
    prompt = system_prompt()
    lines = ["# Rendered training samples", "",
             "10 train rows chosen by `random.Random(20260927).sample`. The system prompt is the Runtime's "
             f"`ManagerClient::system_prompt()` ({report['system_prompt_tokens']} tokens, abbreviated below). "
             "Label `-100` marks masked positions; the first unmasked position is the first TaskPlan token.", ""]
    from manager_sft_data import encode

    for row in picks:
        enc = encode(row, tokenizer, prompt)
        first = next(i for i, label in enumerate(enc.labels) if label != IGNORE_INDEX)
        rendered = render_prompt(row["request"], prompt)
        shown = rendered.replace(prompt, f"<Runtime system prompt: {len(prompt)} chars>")
        tail = [(i, enc.input_ids[i], tokenizer.convert_ids_to_tokens(enc.input_ids[i]), enc.labels[i])
                for i in range(first - 3, first + 4)]
        lines += [f"## {row['id']} ({row['metadata']['taxonomy']})", "", "Rendered prompt (exact, except the system prompt):",
                  "", "```text", shown, "```", "",
                  f"- prompt tokens (masked): {enc.prompt_tokens}; target tokens incl. EOS (trained): {enc.target_tokens}",
                  f"- first target position: {first}; masked label count: {sum(l == IGNORE_INDEX for l in enc.labels)}",
                  f"- last token: `{tokenizer.convert_ids_to_tokens(enc.input_ids[-1])}` (id {enc.input_ids[-1]}, "
                  f"label {enc.labels[-1]}) = EOS",
                  f"- decoded trained span == plan_text + EOS: "
                  f"{tokenizer.decode([l for l in enc.labels if l != IGNORE_INDEX]) == row['plan_text'] + tokenizer.eos_token}",
                  "", "| position | token id | token | label |", "|---:|---:|---|---:|"]
        lines += [f"| {i} | {tid} | `{tstr}` | {lab} |" for i, tid, tstr, lab in tail]
        lines += ["", "Target:", "", "```json", row["plan_text"], "```", ""]
    (OUT / "rendered_training_samples.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items()}, ensure_ascii=False, indent=1))


# --------------------------------------------------------------------------- #
# evaluation helpers
# --------------------------------------------------------------------------- #

def target_loss_sum(model, batch) -> tuple[torch.Tensor, int]:
    """Summed next-token cross entropy over labelled (assistant) positions only."""
    hidden = model.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                         use_cache=False).last_hidden_state
    labels = batch["labels"][:, 1:]
    mask = labels != IGNORE_INDEX
    logits = model.lm_head(hidden[:, :-1][mask])
    return torch.nn.functional.cross_entropy(logits.float(), labels[mask], reduction="sum"), int(mask.sum())


def val_loss(model, encoded, device, micro_batch=4) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(encoded), micro_batch):
            batch = collate(encoded[start:start + micro_batch], device)
            loss, tokens = target_loss_sum(model, batch)
            total += loss.item()
            count += tokens
    return total / count


def run_generation_eval(model, tokenizer, rows, device, tag: str) -> dict:
    model.eval()
    results = {}
    for mode, constrained in (("raw", False), ("constrained", True)):
        outputs, timing = generate(model, tokenizer, [r["request"] for r in rows], constrained=constrained,
                                   batch_size=16, device=device)
        evaluation = evaluate(rows, outputs)
        evaluation["timing"] = timing
        evaluation["tag"] = tag
        evaluation["mode"] = mode
        results[mode] = evaluation
        print(f"[{tag}:{mode}] structural {evaluation['overall']['structural_exact']} canonical "
              f"{evaluation['overall']['canonical_exact']} schema {evaluation['overall']['schema_valid']} "
              f"({timing['seconds']}s)", flush=True)
        empty_cache(device)
    return results


def cmd_baseline(args) -> None:
    device = pick_device()
    model, tokenizer = load_base(device)
    model.eval()
    rows = load_split("validation")
    results = run_generation_eval(model, tokenizer, rows, device, "stage2_base")
    for mode in ("raw", "constrained"):
        write_json(OUT / f"baseline_val_{mode}.json", {"model": BASE_MODEL_NAME, "device": device,
                                                       "provenance": provenance(), **results[mode]})


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #

def collate(items, device):
    width = max(len(e.input_ids) for e in items)
    input_ids = torch.zeros(len(items), width, dtype=torch.long)
    labels = torch.full((len(items), width), IGNORE_INDEX, dtype=torch.long)
    mask = torch.zeros(len(items), width, dtype=torch.long)
    for row, e in enumerate(items):
        n = len(e.input_ids)
        input_ids[row, :n] = torch.tensor(e.input_ids)
        labels[row, :n] = torch.tensor(e.labels)
        mask[row, :n] = 1
    return {"input_ids": input_ids.to(device), "labels": labels.to(device), "attention_mask": mask.to(device)}


def lr_at(step: int, total: int, warmup: int, peak: float) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def select_best(history: list[dict]) -> dict:
    """Selection rule fixed in CONFIG before training."""
    return max(history, key=lambda h: (h["constrained"]["structural_exact"], h["constrained"]["canonical_exact"],
                                       h["raw"]["structural_exact"], -h["val_loss"]))


def save_checkpoint(model, optimizer, directory: Path, state: dict, info: dict) -> None:
    tmp = directory.with_name(directory.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    lora.save_adapter(model, tmp, info)
    torch.save({"optimizer": optimizer.state_dict(), "python_random": random.getstate(),
                "torch_random": torch.get_rng_state(), **state}, tmp / "trainer_state.pt")
    write_json(tmp / "trainer_state.json", state)
    if directory.exists():
        shutil.rmtree(directory)
    os.replace(tmp, directory)


def cmd_train(args) -> None:
    smoke = args.run == "smoke"
    device = pick_device()
    seed_everything(SEED)
    run_dir = OUT / ("smoke" if smoke else "")
    run_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_base(device)
    info = lora.inject(model, CONFIG["lora"]["rank"], CONFIG["lora"]["alpha"], CONFIG["lora"]["dropout"], SEED)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    train_rows = load_split("train")
    val_rows = load_split("validation")
    if smoke:
        train_rows = random.Random(SEED).sample(train_rows, CONFIG["smoke"]["train_samples"])
        val_rows = random.Random(SEED + 1).sample(val_rows, CONFIG["smoke"]["validation_samples"])
    train_enc = encode_all(train_rows, tokenizer, CONFIG["max_length"])
    val_enc = encode_all(val_rows, tokenizer, CONFIG["max_length"])
    epochs = CONFIG["smoke"]["epochs"] if smoke else CONFIG["epochs"]
    micro, effective = CONFIG["micro_batch"], CONFIG["effective_batch"]
    accumulation = effective // micro
    steps_per_epoch = math.ceil(len(train_enc) / effective)
    total_steps = steps_per_epoch * epochs
    warmup = max(1, round(total_steps * CONFIG["scheduler"]["warmup_fraction"]))
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=CONFIG["optimizer"]["lr"], betas=tuple(CONFIG["optimizer"]["betas"]),
                                  eps=CONFIG["optimizer"]["eps"], weight_decay=CONFIG["optimizer"]["weight_decay"])
    config = {**CONFIG, "run": args.run, "device": device, "lora_modules": info,
              "train_samples": len(train_enc), "validation_samples": len(val_enc),
              "steps_per_epoch": steps_per_epoch, "total_optimizer_steps": total_steps, "warmup_steps": warmup,
              "gradient_accumulation": accumulation, "provenance": provenance(),
              "torch": torch.__version__, "transformers": __import__("transformers").__version__}
    write_json(run_dir / "training_config.json", config)
    adapter_info = {"base_model": BASE_MODEL_NAME, "base_model_path": CONFIG["base_model_path"],
                    "lora": CONFIG["lora"], "target_modules": info["target_modules"], "seed": SEED,
                    "provenance": config["provenance"]}

    start_epoch, step, history = 1, 0, []
    if args.resume:
        state = torch.load(Path(args.resume) / "trainer_state.pt", map_location=device, weights_only=False)
        from safetensors.torch import load_file

        model.load_state_dict({k: v.to(device) for k, v in load_file(str(Path(args.resume) / "adapter_model.safetensors")).items()},
                              strict=False)
        optimizer.load_state_dict(state["optimizer"])
        random.setstate(state["python_random"])
        torch.set_rng_state(state["torch_random"])
        start_epoch, step, history = state["epoch"] + 1, state["step"], state["history"]
        print(f"resumed after epoch {state['epoch']} (step {step})", flush=True)

    csv_path = run_dir / "training_history.csv"
    if start_epoch == 1:
        with csv_path.open("w", newline="") as handle:
            csv.writer(handle).writerow(["epoch", "step", "lr", "train_loss", "grad_norm", "target_tokens",
                                         "step_seconds", "tokens_per_second", "allocated_gb", "driver_gb"])
    started = time.perf_counter()
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        order = list(range(len(train_enc)))
        random.Random(SEED + epoch).shuffle(order)
        epoch_losses = []
        for group_start in range(0, len(order), effective):
            group = [train_enc[i] for i in order[group_start:group_start + effective]]
            target_tokens = sum(sum(1 for l in e.labels[1:] if l != IGNORE_INDEX) for e in group)
            lr = lr_at(step, total_steps, warmup, CONFIG["optimizer"]["lr"])
            for g in optimizer.param_groups:
                g["lr"] = lr
            t0 = time.perf_counter()
            loss_sum, seen_tokens = 0.0, 0
            for micro_start in range(0, len(group), micro):
                batch = collate(group[micro_start:micro_start + micro], device)
                loss_sum_micro, _ = target_loss_sum(model, batch)
                loss = loss_sum_micro / target_tokens
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at step {step}")
                loss.backward()
                loss_sum += loss.item()
                seen_tokens += int(batch["attention_mask"].sum())
                del loss, loss_sum_micro
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, CONFIG["grad_clip"])
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite gradient norm at step {step}")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if device == "mps":
                torch.mps.synchronize()
            seconds = time.perf_counter() - t0
            # smoke run: step time grew 42s -> 58s while the MPS driver pool grew
            # 11.8 -> 18.5 GB; release cached blocks every step (as stage2_train.py does)
            empty_cache(device)
            epoch_losses.append(loss_sum)
            mem = memory(device)
            with csv_path.open("a", newline="") as handle:
                csv.writer(handle).writerow([epoch, step, f"{lr:.3e}", f"{loss_sum:.5f}", f"{float(grad_norm):.4f}",
                                             target_tokens, f"{seconds:.2f}", f"{seen_tokens / seconds:.0f}",
                                             mem.get("allocated_gb"), mem.get("driver_gb")])
            if step % 10 == 0:
                print(f"epoch {epoch} step {step}/{total_steps} lr {lr:.2e} loss {loss_sum:.4f} "
                      f"gnorm {float(grad_norm):.3f} {seen_tokens / seconds:.0f} tok/s {mem}", flush=True)
            step += 1
        empty_cache(device)
        loss_value = val_loss(model, val_enc, device)
        generations = run_generation_eval(model, tokenizer, val_rows, device, f"{args.run}_epoch{epoch}")
        entry = {"epoch": epoch, "step": step, "train_loss_mean": statistics.fmean(epoch_losses),
                 "train_loss_last": epoch_losses[-1], "val_loss": loss_value,
                 "raw": generations["raw"]["overall"], "constrained": generations["constrained"]["overall"],
                 "raw_status_recall": generations["raw"]["status_recall"],
                 "constrained_status_recall": generations["constrained"]["status_recall"],
                 "elapsed_seconds": round(time.perf_counter() - started, 1), "memory": memory(device)}
        history.append(entry)
        print(f"== epoch {epoch}: train {entry['train_loss_mean']:.4f} val {loss_value:.4f} "
              f"constrained structural {entry['constrained']['structural_exact']}", flush=True)
        ckpt = run_dir / "checkpoints" / f"epoch{epoch}"
        for mode in ("raw", "constrained"):
            write_json(ckpt.with_name(f"epoch{epoch}_val_{mode}.json"), generations[mode])
        save_checkpoint(model, optimizer, ckpt, {"epoch": epoch, "step": step, "history": history},
                        {**adapter_info, "epoch": epoch, "validation": {k: entry[k] for k in ("val_loss", "raw", "constrained")}})
        write_json(run_dir / "validation_history.json", history)
        empty_cache(device)

    best = select_best(history)
    summary = {"selected_epoch": best["epoch"], "rule": CONFIG["selection"], "history": history,
               "wall_seconds": round(time.perf_counter() - started, 1)}
    write_json(run_dir / "selection.json", summary)
    if not smoke:
        target = OUT / "best_adapter"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(run_dir / "checkpoints" / f"epoch{best['epoch']}", target)
        os.remove(target / "trainer_state.pt")
        for mode in ("raw", "constrained"):
            shutil.copy(run_dir / "checkpoints" / f"epoch{best['epoch']}_val_{mode}.json", OUT / f"best_val_{mode}.json")
    print(f"selected epoch {best['epoch']}", flush=True)


def cmd_smoke_report(args) -> None:
    """Checks the smoke run: loss trend, finite values, frozen base, adapter grads, save/reload, JSON output."""
    import copy

    device = pick_device()
    run_dir = OUT / "smoke"
    history = list(csv.DictReader((run_dir / "training_history.csv").open()))
    losses = [float(h["train_loss"]) for h in history]
    grad_norms = [float(h["grad_norm"]) for h in history]
    validation = json.loads((run_dir / "validation_history.json").read_text())[-1]
    model, tokenizer = load_base(device)
    val_rows = random.Random(SEED + 1).sample(load_split("validation"), CONFIG["smoke"]["validation_samples"])
    val_enc = encode_all(val_rows, tokenizer, CONFIG["max_length"])
    base_val_loss = val_loss(model, val_enc, device)

    # gradient flow on a fresh adapter: base frozen, adapters receive gradients
    probe = copy.deepcopy(model)
    lora.inject(probe, CONFIG["lora"]["rank"], CONFIG["lora"]["alpha"], 0.0, SEED)
    with torch.no_grad():
        for n, p in probe.named_parameters():
            if ".lora_B" in n:
                p.normal_(0, 1e-3)
    probe.train()
    loss, _ = target_loss_sum(probe, collate(val_enc[:1], device))
    loss.backward()
    base_grads = sum(1 for n, p in probe.named_parameters() if not p.requires_grad and p.grad is not None)
    adapter_grads = sum(1 for n, p in probe.named_parameters() if p.requires_grad and p.grad is not None
                        and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0)
    adapter_total = sum(1 for p in probe.parameters() if p.requires_grad)
    del probe
    empty_cache(device)

    # save -> reload twice, identical logits, differs from base
    batch = collate(val_enc[:2], device)
    with torch.no_grad():
        base_logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits[:, -40:].cpu()
    lora.load_adapter(model, run_dir / "checkpoints" / "epoch1")
    with torch.no_grad():
        first = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits[:, -40:].cpu()
    second_model, _ = load_base(device)
    lora.load_adapter(second_model, run_dir / "checkpoints" / "epoch1")
    with torch.no_grad():
        second = second_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits[:, -40:].cpu()
    del second_model
    outputs, _ = generate(model, tokenizer, [r["request"] for r in val_rows[:8]], constrained=False, batch_size=8,
                          device=device)
    from manager_v2_validator import validate as validate_plan

    json_ok = sum(validate_plan(o.strip(), mode="runtime").valid for o in outputs)
    checks = {
        "train_loss_first_step": losses[0], "train_loss_last_step": losses[-1],
        "train_loss_first4_mean": statistics.fmean(losses[:4]), "train_loss_last4_mean": statistics.fmean(losses[-4:]),
        "loss_decreased": statistics.fmean(losses[-4:]) < statistics.fmean(losses[:4]),
        "val_loss_base": base_val_loss, "val_loss_after_smoke": validation["val_loss"],
        "val_loss_decreased": validation["val_loss"] < base_val_loss,
        "all_losses_finite": all(math.isfinite(x) for x in losses + [validation["val_loss"]]),
        "all_grad_norms_finite": all(math.isfinite(x) for x in grad_norms),
        "frozen_base_params_with_grad": base_grads,
        "adapter_params_with_nonzero_grad": f"{adapter_grads}/{adapter_total}",
        "reload_identical_logits": bool(torch.allclose(first, second, atol=1e-5)),
        "adapter_changes_logits": bool((first - base_logits).abs().max() > 1e-3),
        "raw_generations_schema_valid": f"{json_ok}/8",
        "smoke_validation_constrained": validation["constrained"],
        "smoke_validation_raw": validation["raw"],
        "sample_raw_generations": outputs[:4],
    }
    passed = (checks["loss_decreased"] and checks["val_loss_decreased"] and checks["all_losses_finite"]
              and checks["all_grad_norms_finite"] and base_grads == 0 and adapter_grads == adapter_total
              and checks["reload_identical_logits"] and checks["adapter_changes_logits"] and json_ok > 0)
    checks["smoke_pass"] = passed
    write_json(run_dir / "smoke_checks.json", checks)
    lines = ["# Smoke report", "",
             f"**Smoke: {'PASS' if passed else 'FAIL'}**. Pipeline correctness only: 384 train / 96 validation rows, "
             "1 epoch. This model is not a result and is not reused.", "",
             "The first attempt ran out of MPS memory (88 GiB) at micro batch 4 without gradient checkpointing "
             "(`smoke_attempt1_oom.log`). Fix: non-reentrant gradient checkpointing and lm_head only at label positions. "
             "Both are mathematically identical; no hyperparameter changed.", "",
             "| check | value |", "|---|---|"]
    for key, value in checks.items():
        if key.startswith("sample_") or key.startswith("smoke_validation"):
            continue
        lines.append(f"| {key} | {value} |")
    lines += ["", "Smoke validation (96 rows) after one epoch on 384 rows:", "",
              f"- raw: {json.dumps({k: validation['raw'][k] for k in ('json_valid', 'schema_valid', 'structural_exact', 'canonical_exact')})}",
              f"- constrained: {json.dumps({k: validation['constrained'][k] for k in ('json_valid', 'schema_valid', 'structural_exact', 'canonical_exact')})}",
              "", "Raw generations (first 4 smoke-validation rows):", ""]
    lines += [f"- `{o[:220]}`" for o in outputs[:4]]
    (OUT / "smoke_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in checks.items() if not k.startswith("sample_")}, ensure_ascii=False, indent=1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("stats")
    sub.add_parser("baseline")
    sub.add_parser("smoke-report")
    train = sub.add_parser("train")
    train.add_argument("--run", choices=("smoke", "full"), required=True)
    train.add_argument("--resume")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"stats": cmd_stats, "baseline": cmd_baseline, "train": cmd_train,
     "smoke-report": cmd_smoke_report}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
