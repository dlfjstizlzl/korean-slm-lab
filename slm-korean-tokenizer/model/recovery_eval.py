#!/usr/bin/env python3
"""Evaluate a causal LM on the balanced recovery evaluation set.

The primary metric is **NLL per character**. The KO-32K and Qwen tokenizers
segment the same text differently, so per-token loss and perplexity are not
comparable across them; character-normalized NLL is. Per-token metrics are still
reported, and are comparable among KO-32K candidates, which share a tokenizer.

Canonical definition (this function is the single source of truth)
------------------------------------------------------------------
For each record ``r`` with raw text ``t_r`` (NFC, byte-identical for every model)::

    ids_r      = tokenizer(t_r, add_special_tokens=True)   # neither tokenizer adds BOS
    NLL_r      = sum_{i>=1} -log p(ids_r[i] | ids_r[:i])    # natural log, first token unscored
    NLL/char   = sum_r NLL_r / sum_r r["characters"]        # r["characters"] == len(t_r)

* Denominator: Unicode code points of the raw text only. No special/control
  token ever enters it, and it does not depend on the tokenizer.
* Padding positions carry label -100 and contribute nothing.
* ``perplexity`` = exp(NLL / predicted_tokens) is per-token and only comparable
  between models that share a tokenizer.
* Known cross-tokenizer caveat: the first token of each record is conditioned on
  nothing and is not scored, while its characters stay in the denominator. How
  many characters that first token covers depends on the tokenizer.
  ``context_prefix_ids`` (default off) prepends a document-boundary token as
  context only, so every content token is scored; it is used as a sensitivity
  analysis, never silently.

Bug fixed 2026-09-24 (Stage-2 closure): records were tokenized with
``truncation=True, max_length=1024`` while the full character count stayed in the
denominator, so a truncated record lost NLL but not characters. One Korean-large
record (1,186 KO-32K tokens) was affected, only for KO-32K models. The default is
now to score every record in full; ``truncate=True`` reproduces the legacy number.
Both are reported in ``outputs/final_validation/``.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from row_groups import load_row_groups  # noqa: E402

DEFAULT_EVAL = "data/eval/recovery_balanced_eval.jsonl"
IGNORE_INDEX = -100


def load_eval(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


@torch.inference_mode()
def evaluate(model, tokenizer, records: list[dict], device: str = "cpu",
             batch_size: int = 8, max_length: int = 1024,
             max_logit_elements: int = 96_000_000, truncate: bool = False,
             context_prefix_ids: list[int] | None = None,
             category_key: str = "category", pad_to_multiple: int = 1) -> dict:
    """Total NLL per category. Every model sees byte-identical raw text.

    The logit tensor is `batch x length x vocab`, so a 151,936-vocab model needs
    a far smaller batch than a 32,000-vocab one for the same memory. Batches are
    capped by element count rather than by row count so both fit.

    ``max_length`` only matters with ``truncate=True`` (legacy behaviour). With the
    default ``truncate=False`` every record is scored in full and the number of
    records longer than ``max_length`` is still reported.

    ``pad_to_multiple`` rounds each batch width up (padding is masked, so per-record
    NLL is unchanged). On MPS every new shape triggers a graph compile and a new
    allocator block; bucketing widths keeps both bounded on long-document sets.
    """
    was_training = model.training
    model.eval()
    prefix = list(context_prefix_ids or [])
    encoded = []
    over_length = 0
    for index, record in enumerate(records):
        ids = tokenizer(record["text"], add_special_tokens=True)["input_ids"]
        if len(ids) > max_length:
            over_length += 1
            if truncate:
                ids = ids[:max_length]
        if len(ids) >= (1 if prefix else 2):
            encoded.append((index, prefix + ids))
    # Length-sorted batching keeps padding (and wasted compute) small.
    encoded.sort(key=lambda item: len(item[1]))

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 0

    nll = defaultdict(float)
    predicted = defaultdict(int)
    tokens = defaultdict(int)
    characters = defaultdict(int)
    per_record: dict[int, float] = {}

    vocab = int(getattr(model.config, "vocab_size", 32000))
    start = 0
    while start < len(encoded):
        def padded(rows) -> int:
            longest = max(len(ids) for _, ids in rows)
            return -(-longest // pad_to_multiple) * pad_to_multiple

        batch = encoded[start:start + batch_size]
        width = padded(batch)
        while len(batch) > 1 and len(batch) * width * vocab > max_logit_elements:
            batch = batch[:max(1, len(batch) // 2)]
            width = padded(batch)
        start += len(batch)
        input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), width), IGNORE_INDEX, dtype=torch.long)
        attention = torch.zeros((len(batch), width), dtype=torch.long)
        for row, (_, ids) in enumerate(batch):
            input_ids[row, :len(ids)] = torch.tensor(ids)
            labels[row, len(prefix):len(ids)] = torch.tensor(ids[len(prefix):])
            attention[row, :len(ids)] = 1
        input_ids, labels, attention = (t.to(device) for t in (input_ids, labels, attention))

        logits = model(input_ids=input_ids, attention_mask=attention).logits
        if logits.dtype != torch.float32:
            logits = logits.float()
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        losses = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1), reduction="none", ignore_index=IGNORE_INDEX,
        ).view(shift_labels.shape)
        row_nll = losses.sum(dim=1).cpu()

        for row, (index, ids) in enumerate(batch):
            record = records[index]
            category = record[category_key]
            value = float(row_nll[row])
            content = len(ids) - len(prefix)
            nll[category] += value
            # Without a prefix the first content token is context only.
            predicted[category] += content if prefix else content - 1
            tokens[category] += content
            characters[category] += record["characters"]
            per_record[index] = value

    if was_training:
        model.train()

    def metrics(total_nll: float, total_predicted: int, total_tokens: int, total_chars: int) -> dict:
        per_token = total_nll / total_predicted if total_predicted else float("nan")
        return {
            "nll": total_nll,
            "nll_per_character": total_nll / total_chars if total_chars else float("nan"),
            "mean_loss_per_token": per_token,
            "perplexity": math.exp(per_token) if per_token < 20 else float("inf"),
            "tokens": total_tokens,
            "predicted_tokens": total_predicted,
            "characters": total_chars,
            "tokens_per_character": total_tokens / total_chars if total_chars else 0.0,
        }

    by_category = {
        category: metrics(nll[category], predicted[category], tokens[category], characters[category])
        for category in sorted(nll)
    }
    overall = metrics(sum(nll.values()), sum(predicted.values()),
                      sum(tokens.values()), sum(characters.values()))
    return {
        "overall": overall,
        "by_category": by_category,
        "evaluated_records": len(encoded),
        "skipped_records": len(records) - len(encoded),
        "records_over_max_length": over_length,
        "truncated_records": over_length if truncate else 0,
        "truncate": truncate,
        "max_length": max_length,
        "context_prefix_ids": prefix,
        "per_record_nll": per_record,
    }


def bootstrap_difference(records: list[dict], a: dict[int, float], b: dict[int, float],
                         iterations: int = 2000, seed: int = 42) -> dict:
    """Paired bootstrap over records for the NLL/char difference a - b."""
    shared = sorted(set(a) & set(b))
    if not shared:
        return {"samples": 0}
    chars = [records[index]["characters"] for index in shared]
    diffs = [a[index] - b[index] for index in shared]
    rng = random.Random(seed)
    size = len(shared)
    values = []
    for _ in range(iterations):
        picked = [rng.randrange(size) for _ in range(size)]
        numerator = sum(diffs[i] for i in picked)
        denominator = sum(chars[i] for i in picked)
        if denominator:
            values.append(numerator / denominator)
    values.sort()
    point = sum(diffs) / sum(chars)
    return {
        "samples": size,
        "iterations": len(values),
        "delta_nll_per_character": point,
        "ci95_low": values[int(0.025 * len(values))],
        "ci95_high": values[int(0.975 * len(values)) - 1],
        "significant": not (values[int(0.025 * len(values))] <= 0.0 <= values[int(0.975 * len(values)) - 1]),
    }


def describe(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "std": statistics.pstdev(ordered) if len(ordered) > 1 else 0.0,
        "median": statistics.median(ordered),
        "p5": ordered[max(0, int(0.05 * len(ordered)) - 1)],
        "p95": ordered[max(0, int(0.95 * len(ordered)) - 1)],
        "max": ordered[-1],
    }


def embedding_norm_stats(model, groups: dict) -> dict:
    weight = model.get_input_embeddings().weight.detach().to(torch.float32)
    norms = weight.norm(dim=1).cpu()
    out = {}
    for name in ("exact", "decomposed", "control", "base_added"):
        rows = groups.get(name, [])
        out[name] = describe([float(norms[row]) for row in rows])
    out["all"] = describe([float(value) for value in norms])
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--eval-file", default=DEFAULT_EVAL)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--truncate", action="store_true",
                        help="Legacy behaviour: truncate records to --max-length (biased; see module doc).")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--norm-groups", action="store_true",
                        help="Report embedding norms per transplant row group.")
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--row-tokenizer", default="outputs/tokenizer-ko-32000-mixed10-structured-v2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_eval(args.eval_file)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(args.device)
    result = evaluate(model, tokenizer, records, device=args.device,
                      batch_size=args.batch_size, max_length=args.max_length,
                      truncate=args.truncate)
    result.pop("per_record_nll", None)
    result["model"] = args.model
    result["eval_file"] = args.eval_file
    if args.norm_groups:
        groups = load_row_groups(args.base_model, args.row_tokenizer)
        result["embedding_norms"] = embedding_norm_stats(model, groups)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
