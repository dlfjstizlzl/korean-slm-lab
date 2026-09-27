#!/usr/bin/env python3
"""Minimal LoRA for the Manager SFT (PEFT is not installed in this environment).

    y = W x + (B A dropout(x)) * alpha / r,   A: r x in (Kaiming-uniform),  B: out x r (zeros)

Target modules are discovered from the model, not hard-coded: every nn.Linear
inside the decoder blocks (`model.layers.*`). Embeddings and the (tied)
lm_head stay frozen. Adapters are saved separately from the base weights
(safetensors + JSON config); nothing is merged.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, dtype=base.weight.dtype,
                                               device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=base.weight.dtype,
                                               device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling


def discover_targets(model: nn.Module) -> list[str]:
    """Names of every nn.Linear inside the transformer blocks."""
    return [name for name, module in model.named_modules()
            if isinstance(module, nn.Linear) and ".layers." in f".{name}" and "lm_head" not in name]


def inject(model: nn.Module, rank: int, alpha: float, dropout: float, seed: int) -> dict:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    torch.manual_seed(seed)
    targets = discover_targets(model)
    for name in targets:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        setattr(parent, child, LoRALinear(getattr(parent, child), rank, alpha, dropout))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    suffixes = sorted({name.rsplit(".", 1)[-1] for name in targets})
    return {"target_modules": suffixes, "target_module_count": len(targets), "trainable_parameters": trainable,
            "total_parameters": total, "trainable_percent": round(100 * trainable / total, 4)}


def adapter_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()
            if ".lora_A" in name or ".lora_B" in name}


def save_adapter(model: nn.Module, directory: Path, config: dict) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_file(adapter_state(model), str(directory / "adapter_model.safetensors"))
    (directory / "adapter_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n",
                                                   encoding="utf-8")


def load_adapter(model: nn.Module, directory: Path) -> dict:
    """Inject LoRA as described by adapter_config.json and load its weights."""
    directory = Path(directory)
    config = json.loads((directory / "adapter_config.json").read_text(encoding="utf-8"))
    lora = config["lora"]
    inject(model, lora["rank"], lora["alpha"], 0.0, seed=0)
    weights = load_file(str(directory / "adapter_model.safetensors"))
    device = next(model.parameters()).device
    missing = set(adapter_state(model)) - set(weights)
    unexpected = set(weights) - set(adapter_state(model))
    if missing or unexpected:
        raise ValueError(f"adapter mismatch: missing {sorted(missing)[:3]}, unexpected {sorted(unexpected)[:3]}")
    model.load_state_dict({k: v.to(device) for k, v in weights.items()}, strict=False)
    model.eval()
    return config
