#!/usr/bin/env python3
"""Export Manager-SLM v1 as two standalone Hugging Face model directories.

    python model/manager_export.py

outputs/release/
  qwen3-0.6b-ko32k-stage2-signal2678-base/   inference copy of the Stage-2 base (no optimizer state)
  qwen3-0.6b-ko32k-manager-v1/               base + Manager LoRA (epoch 2) merged into the weights,
                                             plus the original unmerged adapter under adapter/

The merge is W' = W + (B @ A) * alpha / r for every adapted Linear, in float32.
It is verified on VALIDATION rows only (the sealed test is consumed and never read):
logit agreement with the unmerged adapter model, and byte-identical greedy
generations against the stored validation predictions of the selected adapter.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "model"))

import manager_lora as lora  # noqa: E402
from manager_sft_data import BASE_MODEL, BASE_MODEL_NAME, load_split, render_prompt, sha256_file, system_prompt  # noqa: E402
from manager_sft_eval import generate  # noqa: E402
from manager_v2_registry import load_contract  # noqa: E402

RELEASE = PROJECT_ROOT / "outputs" / "release"
BASE_OUT = RELEASE / "qwen3-0.6b-ko32k-stage2-signal2678-base"
MANAGER_OUT = RELEASE / "qwen3-0.6b-ko32k-manager-v1"
ADAPTER = PROJECT_ROOT / "outputs" / "manager_sft_v1" / "best_adapter"
FINAL = PROJECT_ROOT / "outputs" / "manager_final_test_v1"
EXPECTED_ADAPTER_SHA256 = "f1df598d22032169b7fd91ed6aef04f0b3e57a3bab9fe5c5f438352f25cbf7b7"
MODEL_FILES = ["model.safetensors", "config.json", "generation_config.json"]
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "added_tokens.json",
                   "special_tokens_map.json", "chat_template.jinja"]
VERIFY_ROWS = 64


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def manifest(directory: Path) -> dict:
    return {p.relative_to(directory).as_posix(): {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
            for p in sorted(directory.rglob("*")) if p.is_file() and p.name != "MANIFEST.json"}


def fresh(directory: Path) -> None:
    if directory.exists():
        raise SystemExit(f"{directory} already exists; refusing to overwrite a release")
    directory.mkdir(parents=True)


def export_base() -> None:
    fresh(BASE_OUT)
    for name in MODEL_FILES + TOKENIZER_FILES + ["canonical_metadata.json"]:
        shutil.copy2(BASE_MODEL / name, BASE_OUT / name)
    (BASE_OUT / "MODEL_CARD.md").write_text(f"""# {BASE_MODEL_NAME} (base)

Inference copy of `outputs/models/qwen3-0.6b-ko32k-stage2-signal2678/` (Stage-2 canonical, signal2678).
Training-only files (`optimizer_state.pt`, `stage2_state.json`) are not included; the weights are byte-identical.

- Architecture: Qwen3ForCausalLM, 28 layers, hidden 1024, float32, tied embeddings
- Tokenizer: KO-32K ByteLevel BPE (vocab 32,000), ChatML special tokens `<|im_start|>`/`<|im_end|>`
- Role: general Korean base; the starting point of Manager-SLM v1. Not instruction-tuned for TaskPlan output
  (raw TaskPlan schema validity on the Manager sealed test: 0.004).

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("outputs/release/qwen3-0.6b-ko32k-stage2-signal2678-base")
tokenizer = AutoTokenizer.from_pretrained("outputs/release/qwen3-0.6b-ko32k-stage2-signal2678-base")
```
""", encoding="utf-8")
    files = manifest(BASE_OUT)
    if files["model.safetensors"]["sha256"] != sha256_file(BASE_MODEL / "model.safetensors"):
        raise SystemExit("base weight copy differs from the source")
    (BASE_OUT / "MANIFEST.json").write_text(json.dumps({"model": BASE_MODEL_NAME, "created_at": now(),
                                                        "source": str(BASE_MODEL.relative_to(PROJECT_ROOT)),
                                                        "files": files}, indent=2) + "\n", encoding="utf-8")


@torch.no_grad()
def merge(model: nn.Module) -> int:
    merged = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, lora.LoRALinear):
            base = module.base
            base.weight += (module.lora_B @ module.lora_A) * module.scaling
            parent_name, _, child = name.rpartition(".")
            setattr(model.get_submodule(parent_name), child, base)
            merged += 1
    return merged


@torch.no_grad()
def export_manager(device: str) -> dict:
    if sha256_file(ADAPTER / "adapter_model.safetensors") != EXPECTED_ADAPTER_SHA256:
        raise SystemExit("adapter is not the selected epoch-2 adapter")
    fresh(MANAGER_OUT)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    rows = load_split("validation")[:VERIFY_ROWS]
    prompts = [render_prompt(r["request"], system_prompt()) for r in rows[:8]]

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32).to(device).eval()
    lora.load_adapter(model, ADAPTER)
    model.to(device).eval()
    tokenizer.padding_side = "left"
    batch = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    reference = model(**batch).logits[:, -1].float().cpu()
    count = merge(model)
    model.save_pretrained(MANAGER_OUT, safe_serialization=True)
    del model
    for name in TOKENIZER_FILES + ["generation_config.json"]:
        shutil.copy2(BASE_MODEL / name, MANAGER_OUT / name)
    adapter_dir = MANAGER_OUT / "adapter"
    shutil.copytree(ADAPTER, adapter_dir)

    reloaded = AutoModelForCausalLM.from_pretrained(MANAGER_OUT, dtype=torch.float32).to(device).eval()
    merged_logits = reloaded(**batch).logits[:, -1].float().cpu()
    max_diff = (merged_logits - reference).abs().max().item()
    argmax_equal = bool((merged_logits.argmax(-1) == reference.argmax(-1)).all())
    stored = json.loads((ADAPTER.parent / "best_val_raw.json").read_text(encoding="utf-8"))["records"]
    expected = [r["output"] for r in stored[:VERIFY_ROWS]]
    outputs, _ = generate(reloaded, tokenizer, [r["request"] for r in rows], constrained=False, batch_size=16,
                          device=device)
    identical = sum(a == b for a, b in zip(outputs, expected))
    return {"merged_modules": count, "verify_rows": "validation[:%d] (sealed test not used)" % VERIFY_ROWS,
            "last_token_logit_max_abs_diff_vs_unmerged": max_diff, "next_token_argmax_equal": argmax_equal,
            "greedy_raw_generations_identical_to_stored_validation": f"{identical}/{len(rows)}"}


def write_manager_card(verification: dict) -> None:
    contract = load_contract()
    (MANAGER_OUT / "runtime").mkdir()
    (MANAGER_OUT / "runtime" / "system_prompt.txt").write_text(contract.system_prompt, encoding="utf-8")
    (MANAGER_OUT / "runtime" / "taskplan_v2.gbnf").write_text(contract.grammar, encoding="utf-8")
    final = json.loads((FINAL / "final_manager_v1_report.json").read_text(encoding="utf-8"))
    lc = final["confidence_intervals"]["lora_constrained"]
    (MANAGER_OUT / "MODEL_CARD.md").write_text(f"""# Qwen3-0.6B-KO32K Manager-SLM v1

`{BASE_MODEL_NAME}` + Manager LoRA (epoch 2) **merged** into standalone float32 weights.
It turns a Korean user request into a TaskPlan v2 JSON plan over five skills
(search, summary, extract, compare, sql) or a clarify/unsupported answer.

## Contents
- `model.safetensors`, `config.json`: merged weights (W + B·A·α/r, r=16, α=32, 196 Linear modules)
- tokenizer files: identical to the base (KO-32K)
- `adapter/`: the original unmerged LoRA adapter (sha256 `{EXPECTED_ADAPTER_SHA256[:16]}…`) with provenance
- `runtime/system_prompt.txt`, `runtime/taskplan_v2.gbnf`: snapshots of `ManagerClient::system_prompt()` and
  `ManagerClient::json_grammar()` used in training/evaluation. The Runtime source stays the source of truth.

## Required usage
The model was trained only on the Runtime format. Use exactly:

```
<|im_start|>system\\n{{runtime/system_prompt.txt}}<|im_end|>\\n<|im_start|>user\\n{{request}}<|im_end|>\\n<|im_start|>assistant\\n
```

Greedy decoding (temperature 0), max 256 new tokens, stop on `<|im_end|>`; optionally constrain with the grammar.
Do not rely on `generation_config.json` sampling defaults (inherited from the base) and do not add a `<think>` block.

## Results (sealed test, n=461, evaluated once; `outputs/manager_final_test_v1/`)
- structural exact {lc['structural_exact']['rate']} (95% CI {lc['structural_exact']['ci95']}),
  canonical exact {lc['canonical_exact']['rate']}, schema valid {lc['schema_valid']['rate']},
  status accuracy {lc['status_accuracy']['rate']} (grammar-constrained; raw is the same within 0.3 pt)
- Stage-2 base, same setting: structural exact 0.2495
- Weak spots: held-out hard phrasing families (0.66), search entity corruption (~12% of search tasks)
- The test is generator-family held-out, not real-user OOD; production robustness is not established.

## Merge verification (validation rows only)
```json
{json.dumps(verification, indent=2)}
```
""", encoding="utf-8")
    files = manifest(MANAGER_OUT)
    (MANAGER_OUT / "MANIFEST.json").write_text(json.dumps(
        {"model": "Qwen3-0.6B-KO32K Manager-SLM v1 (merged)", "created_at": now(), "base": BASE_MODEL_NAME,
         "base_weights_sha256": sha256_file(BASE_MODEL / "model.safetensors"),
         "adapter_sha256": EXPECTED_ADAPTER_SHA256, "verification": verification, "files": files},
        indent=2) + "\n", encoding="utf-8")


def main() -> int:
    from manager_sft_train import pick_device

    device = pick_device()
    export_base()
    verification = export_manager(device)
    print(json.dumps(verification, indent=1))
    if not verification["next_token_argmax_equal"] or not verification[
            "greedy_raw_generations_identical_to_stored_validation"].startswith(f"{VERIFY_ROWS}/"):
        print("WARNING: merged model does not reproduce the adapter model exactly", flush=True)
    write_manager_card(verification)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
