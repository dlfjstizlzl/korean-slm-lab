#!/usr/bin/env python3
"""Transplant Qwen3 embeddings onto the final KO-32K tokenizer.

Surface-based transplant: every new-vocabulary row is initialized from the base
model's embedding table by token *string*, never by token ID prefix. Both
tokenizers are ByteLevel BPE over the same 256-symbol alphabet, so a new token
string can be re-segmented directly with the base BPE model.

Strategies, in order:
  1. exact      - the new token string exists verbatim in the base vocabulary
  2. decomposed - base BPE re-segments the new token string; rows are combined
                  with `mean` (default), `length_weighted`, or `norm_matched`
  3. random     - normal(0, config.initializer_range); reported loudly

This stage performs embedding transplant only. No training of any kind.
"""
from __future__ import annotations

import argparse
import inspect
import json
import statistics
import sys
from pathlib import Path

import torch
from tokenizers import pre_tokenizers
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tokenizer"))

from common import (  # noqa: E402
    CONTROL_TOKENS,
    assert_base_added_tokens_preserved,
    assert_control_tokens_atomic,
    assert_required_special_tokens,
    percentile_nearest_rank,
    special_token_strings,
)

DEFAULT_BASE_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_TOKENIZER = "outputs/tokenizer-ko-32000-mixed10-structured-v2"
DEFAULT_OUTPUT = "outputs/models/qwen3-0.6b-ko32k-transplanted"

EXACT = "exact"
DECOMPOSED = "decomposed"
RANDOM = "random"


# --------------------------------------------------------------------------- #
# ByteLevel helpers
# --------------------------------------------------------------------------- #

def byte_level_alphabet() -> set[str]:
    return set(pre_tokenizers.ByteLevel.alphabet())


def build_byte_encoder() -> dict[int, str]:
    """Reconstruct the GPT-2 byte -> unicode map from the tokenizers alphabet."""
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    mapping = {b: chr(b) for b in printable}
    shift = 0
    for b in range(256):
        if b not in mapping:
            mapping[b] = chr(256 + shift)
            shift += 1
    return mapping


BYTE_ENCODER = build_byte_encoder()


BYTE_DECODER = {char: byte for byte, char in BYTE_ENCODER.items()}


def to_byte_level(text: str) -> str:
    return "".join(BYTE_ENCODER[b] for b in text.encode("utf-8"))


def from_byte_level(text: str) -> bytes:
    return bytes(BYTE_DECODER[char] for char in text)


def surface_weights(pieces: list[str]) -> list[float]:
    """Characters of real surface that each ByteLevel piece is responsible for.

    `len(piece)` is a *byte* count under ByteLevel, so a 3-byte Hangul syllable
    would outweigh a 1-byte ASCII character 3:1 purely because of the encoding.
    Instead, the pieces are concatenated back into the token's UTF-8 byte string
    and every character is attributed to the piece holding its leading byte
    (continuation bytes have `(b & 0xC0) == 0x80`). A piece made only of
    continuation bytes therefore weighs 0: it completes a character that another
    piece already accounts for. If no piece holds a leading byte the weighting
    is undefined, and byte lengths are used instead.
    """
    spans = [from_byte_level(piece) for piece in pieces]
    blob = b"".join(spans)
    leading = [index for index, byte in enumerate(blob) if (byte & 0xC0) != 0x80]
    weights: list[float] = []
    offset = 0
    for span in spans:
        end = offset + len(span)
        weights.append(float(sum(1 for index in leading if offset <= index < end)))
        offset = end
    if sum(weights) == 0.0:
        return [float(len(span)) for span in spans]
    return weights


def is_byte_level_string(text: str, alphabet: set[str]) -> bool:
    return bool(text) and all(char in alphabet for char in text)


# --------------------------------------------------------------------------- #
# Transplant policy
# --------------------------------------------------------------------------- #

def bpe_segments(bpe_model, text: str) -> list[str]:
    """Segment a ByteLevel string with the base BPE model, or [] if impossible."""
    try:
        pieces = [token.value for token in bpe_model.tokenize(text)]
    except Exception:  # noqa: BLE001 - the backend raises bare exceptions
        return []
    # `tokenize` silently drops symbols outside the alphabet; require exact
    # reconstruction so a lossy segmentation can never be mistaken for a match.
    return pieces if "".join(pieces) == text else []


def plan_token(
    token: str,
    *,
    is_added: bool,
    old_vocab: dict[str, int],
    old_rows: int,
    bpe_model,
    alphabet: set[str],
) -> tuple[str, list[int], list[str], str]:
    """Return (strategy, source_ids, source_pieces, byte_level_form)."""
    exact_id = old_vocab.get(token)
    if exact_id is not None and 0 <= int(exact_id) < old_rows:
        return EXACT, [int(exact_id)], [token], token

    # Added tokens are stored as raw text; normal BPE tokens are already in the
    # ByteLevel alphabet. Encode anything that is not.
    if is_added or not is_byte_level_string(token, alphabet):
        byte_form = to_byte_level(token)
    else:
        byte_form = token

    pieces = bpe_segments(bpe_model, byte_form)
    ids = [old_vocab.get(piece) for piece in pieces]
    if pieces and all(i is not None and 0 <= int(i) < old_rows for i in ids):
        return DECOMPOSED, [int(i) for i in ids], pieces, byte_form
    return RANDOM, [], [], byte_form


def combine_rows(
    source: torch.Tensor,
    source_ids: list[int],
    pieces: list[str],
    strategy: str,
) -> torch.Tensor:
    """Combine base rows into one new row. Single-source rows are strategy-free."""
    selected = source[source_ids].to(dtype=torch.float32)
    if len(source_ids) == 1:
        return selected[0].clone()

    if strategy == "length_weighted":
        raw = surface_weights(pieces)
        if sum(raw) == 0.0:
            raw = [1.0] * len(pieces)
        weights = torch.tensor(raw, dtype=torch.float32).unsqueeze(1)
        return (selected * weights).sum(dim=0) / weights.sum()

    mean = selected.mean(dim=0)
    if strategy != "norm_matched":
        return mean

    # Keep the mean direction, restore the magnitude the sources carried.
    target = float(selected.norm(dim=1).mean())
    length = float(mean.norm())
    if length < 1e-6 or target < 1e-6:
        return mean
    return mean * (target / length)


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

def describe(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": float(statistics.fmean(ordered)),
        "std": float(statistics.pstdev(ordered)) if len(ordered) > 1 else 0.0,
        "median": float(statistics.median(ordered)),
        "p5": float(ordered[max(0, int(0.05 * len(ordered)) - 1)]),
        "p95": float(ordered[max(0, int(0.95 * len(ordered)) - 1)]),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
    }


def has_hangul(text: str) -> bool:
    return any(
        "가" <= ch <= "힣" or "ᄀ" <= ch <= "ᇿ" or "㄰" <= ch <= "㆏"
        for ch in text
    )


def decoded_surface(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode(
            [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
    except TypeError:
        return tokenizer.decode([token_id], skip_special_tokens=False)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--strategy", choices=["mean", "length_weighted", "norm_matched"], default="mean"
    )
    parser.add_argument("--report-json", default="outputs/embedding_transplant_report.json")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-fallback", type=int, default=0,
                        help="Refuse to save when random-fallback rows exceed this count.")
    parser.add_argument("--cosine-sample", type=int, default=512)
    parser.add_argument("--dry-run", action="store_true", help="Compute statistics without saving.")
    return parser.parse_args()


def resolve_dtype(name: str):
    return {"auto": "auto", "float32": torch.float32,
            "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"{output_dir} already exists; pass --overwrite to replace it")

    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    new_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=args.trust_remote_code)
    assert_required_special_tokens(base_tokenizer, new_tokenizer)
    preserved_added = assert_base_added_tokens_preserved(base_tokenizer, new_tokenizer)
    control_ids = assert_control_tokens_atomic(new_tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=resolve_dtype(args.dtype),
        device_map="cpu", trust_remote_code=args.trust_remote_code,
    )
    old_input = model.get_input_embeddings().weight.detach()
    old_output_module = model.get_output_embeddings()
    old_output = old_output_module.weight.detach() if old_output_module is not None else None
    old_rows, hidden_size = int(old_input.shape[0]), int(old_input.shape[1])
    tied = bool(getattr(model.config, "tie_word_embeddings", False))
    initializer_range = float(getattr(model.config, "initializer_range", 0.02))
    original_total_params = sum(p.numel() for p in model.parameters())

    old_vocab = base_tokenizer.get_vocab()
    bpe_model = base_tokenizer.backend_tokenizer.model
    alphabet = byte_level_alphabet()
    missing_alphabet = sorted(char for char in alphabet if char not in old_vocab)
    if missing_alphabet:
        raise RuntimeError(f"Base vocabulary is missing ByteLevel symbols: {missing_alphabet}")

    new_len = len(new_tokenizer)
    new_added = {str(token) for token in new_tokenizer.added_tokens_decoder.values()}
    base_added = {str(token) for token in base_tokenizer.added_tokens_decoder.values()}

    # --- plan every row before touching any weight ------------------------- #
    plans: list[dict] = []
    id_to_token = new_tokenizer.convert_ids_to_tokens(list(range(new_len)))
    for new_id, token in enumerate(id_to_token):
        if token is None:
            raise RuntimeError(f"New tokenizer has no token string for id {new_id}")
        strategy, source_ids, pieces, byte_form = plan_token(
            token, is_added=token in new_added, old_vocab=old_vocab,
            old_rows=old_rows, bpe_model=bpe_model, alphabet=alphabet,
        )
        plans.append({
            "new_id": new_id, "token": token, "strategy": strategy,
            "source_ids": source_ids, "pieces": pieces, "byte_level": byte_form,
            "is_added": token in new_added, "is_base_added": token in base_added,
            "is_control": token in CONTROL_TOKENS,
        })

    # --- surface / duplicate diagnostics ----------------------------------- #
    surfaces: dict[str, list[int]] = {}
    empty_surface: list[int] = []
    replacement_surface: list[int] = []
    for plan in plans:
        surface = decoded_surface(new_tokenizer, plan["new_id"])
        plan["surface"] = surface
        surfaces.setdefault(surface, []).append(plan["new_id"])
        if surface == "":
            empty_surface.append(plan["new_id"])
        if "�" in surface:
            replacement_surface.append(plan["new_id"])
    duplicate_surfaces = {s: ids for s, ids in surfaces.items() if len(ids) > 1}

    counts = {EXACT: 0, DECOMPOSED: 0, RANDOM: 0}
    for plan in plans:
        counts[plan["strategy"]] += 1

    special_exact = sum(1 for p in plans if p["is_base_added"] and p["strategy"] == EXACT)
    special_not_exact = [p["token"] for p in plans if p["is_base_added"] and p["strategy"] != EXACT]
    if special_not_exact:
        print(f"WARNING: base added tokens not exact-copied: {special_not_exact}", file=sys.stderr)
    control_report = []
    for plan in (p for p in plans if p["is_control"]):
        entry = {
            "token": plan["token"], "new_id": plan["new_id"], "strategy": plan["strategy"],
            "byte_level": plan["byte_level"], "source_pieces": plan["pieces"],
            "source_ids": plan["source_ids"],
        }
        if plan["source_ids"]:
            entry["surface_weights"] = surface_weights(plan["pieces"])
            # All three strategies are reported for every control token so the
            # ablation can be compared without rebuilding a checkpoint.
            for name in ("mean", "length_weighted", "norm_matched"):
                vector = combine_rows(old_input, plan["source_ids"], plan["pieces"], name)
                entry[f"norm_{name}"] = float(vector.norm())
        control_report.append(entry)

    if counts[RANDOM] > args.max_fallback:
        print(json.dumps({"counts": counts,
                          "random_examples": [p["token"] for p in plans if p["strategy"] == RANDOM][:32]},
                         ensure_ascii=False, indent=2))
        raise RuntimeError(
            f"{counts[RANDOM]} rows fall back to random init (limit {args.max_fallback}). "
            "Investigate before saving a checkpoint."
        )

    # --- decomposition length distribution --------------------------------- #
    decomp_lengths = [len(p["pieces"]) for p in plans if p["strategy"] == DECOMPOSED]
    hangul_lengths = [len(p["pieces"]) for p in plans
                      if p["strategy"] == DECOMPOSED and has_hangul(p["surface"])]
    ascii_lengths = [len(p["pieces"]) for p in plans
                     if p["strategy"] == DECOMPOSED and p["surface"].isascii()]
    buckets = {"1": 0, "2": 0, "3": 0, "4": 0, "5+": 0}
    for length in decomp_lengths:
        buckets[str(length) if length <= 4 else "5+"] += 1

    def length_summary(values: list[int]) -> dict:
        if not values:
            return {"count": 0}
        return {
            "count": len(values), "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "p95": percentile_nearest_rank(values, 0.95), "max": max(values),
        }

    # --- build the new embedding table ------------------------------------- #
    resize_kwargs = {}
    if "mean_resizing" in inspect.signature(model.resize_token_embeddings).parameters:
        resize_kwargs["mean_resizing"] = False
    model.resize_token_embeddings(new_len, **resize_kwargs)
    new_input = model.get_input_embeddings().weight
    new_output_module = model.get_output_embeddings()
    new_output = new_output_module.weight if new_output_module is not None else None
    untied_head = new_output is not None and not tied and old_output is not None

    with torch.no_grad():
        # Every row is overwritten below; random init only survives where the
        # plan says RANDOM.
        torch.nn.init.normal_(new_input, mean=0.0, std=initializer_range)
        if untied_head:
            torch.nn.init.normal_(new_output, mean=0.0, std=initializer_range)
        for plan in plans:
            row, strategy, source_ids = plan["new_id"], plan["strategy"], plan["source_ids"]
            if strategy == EXACT:
                new_input[row].copy_(old_input[source_ids[0]].to(dtype=new_input.dtype))
                if untied_head:
                    new_output[row].copy_(old_output[source_ids[0]].to(dtype=new_output.dtype))
            elif strategy == DECOMPOSED:
                vector = combine_rows(old_input, source_ids, plan["pieces"], args.strategy)
                new_input[row].copy_(vector.to(dtype=new_input.dtype))
                if untied_head:
                    head = combine_rows(old_output, source_ids, plan["pieces"], args.strategy)
                    new_output[row].copy_(head.to(dtype=new_output.dtype))

    # --- config / tying ----------------------------------------------------- #
    model.config.vocab_size = new_len
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = getattr(new_tokenizer, attr, None)
        # Assign even when the new tokenizer has no such token: skipping would
        # leave the base model's inherited id, which points far outside the new
        # 32,000-entry vocabulary. KO-32K has no BOS, so bos_token_id must
        # become None rather than staying at Qwen's 151643.
        setattr(model.config, attr, value)
        if getattr(model, "generation_config", None) is not None:
            setattr(model.generation_config, attr, value)
    if tied:
        model.tie_weights()

    input_weight = model.get_input_embeddings().weight
    output_module = model.get_output_embeddings()

    # --- structural validation (blocks saving) ------------------------------ #
    failures: list[str] = []
    if new_len != 32000:
        failures.append(f"tokenizer length {new_len} != 32000")
    if int(model.config.vocab_size) != new_len:
        failures.append(f"config.vocab_size {model.config.vocab_size} != {new_len}")
    if int(input_weight.shape[0]) != new_len:
        failures.append(f"input embedding rows {input_weight.shape[0]} != {new_len}")
    if int(input_weight.shape[1]) != hidden_size:
        failures.append(f"hidden size changed: {input_weight.shape[1]} != {hidden_size}")
    if output_module is None:
        failures.append("model has no output embedding")
    elif int(output_module.weight.shape[0]) != new_len:
        failures.append(f"lm_head rows {output_module.weight.shape[0]} != {new_len}")
    tied_actual = (output_module is not None
                   and input_weight.data_ptr() == output_module.weight.data_ptr())
    if tied != tied_actual:
        failures.append(f"tie mismatch: config={tied}, actual={tied_actual}")
    if len(preserved_added) != len(base_added):
        failures.append(f"base added tokens {len(preserved_added)}/{len(base_added)}")
    if len(control_ids) != len(CONTROL_TOKENS):
        failures.append(f"control tokens {len(control_ids)}/{len(CONTROL_TOKENS)}")
    for attr in ("eos_token_id", "pad_token_id"):
        value = getattr(new_tokenizer, attr, None)
        if value is None or not 0 <= int(value) < new_len:
            failures.append(f"{attr}={value} outside vocabulary")
    if not torch.isfinite(input_weight).all().item():
        failures.append("input embedding contains NaN or Inf")
    if output_module is not None and not torch.isfinite(output_module.weight).all().item():
        failures.append("lm_head contains NaN or Inf")
    if counts[RANDOM] > args.max_fallback:
        failures.append(f"random fallback {counts[RANDOM]} > {args.max_fallback}")
    if failures:
        raise RuntimeError("Structural validation failed:\n  - " + "\n  - ".join(failures))

    # --- weight statistics --------------------------------------------------- #
    with torch.no_grad():
        old_norms = old_input.to(torch.float32).norm(dim=1)
        new_norms = input_weight.detach().to(torch.float32).norm(dim=1)
        norm_by_strategy = {
            strategy: describe([float(new_norms[p["new_id"]]) for p in plans
                                if p["strategy"] == strategy])
            for strategy in (EXACT, DECOMPOSED, RANDOM)
        }
        # cosine sanity on exact copies
        exact_plans = [p for p in plans if p["strategy"] == EXACT]
        step = max(1, len(exact_plans) // max(args.cosine_sample, 1))
        sampled = exact_plans[::step][: args.cosine_sample]
        cosines = []
        for plan in sampled:
            a = input_weight[plan["new_id"]].detach().to(torch.float32)
            b = old_input[plan["source_ids"][0]].to(torch.float32)
            cosines.append(float(torch.nn.functional.cosine_similarity(a, b, dim=0)))

    total_params = sum(p.numel() for p in model.parameters())
    embedding_params = int(input_weight.numel()) + (0 if tied_actual else int(output_module.weight.numel()))
    original_embedding_params = old_rows * hidden_size * (1 if tied else 2)

    report = {
        "checkpoint_status": "embedding_transplant_only",
        "warning": "Embeddings were transplanted without any training. "
                   "Recovery training is required before quality evaluation.",
        "base_model": args.base_model,
        "base_model_resolved": str(getattr(model.config, "_name_or_path", args.base_model)),
        "new_tokenizer": str(Path(args.tokenizer)),
        "output_dir": str(output_dir),
        "combine_strategy": args.strategy,
        "structure": {
            "model_class": type(model).__name__,
            "hidden_size": hidden_size,
            "dtype": str(input_weight.dtype),
            "old_vocab_rows": old_rows,
            "old_tokenizer_len": len(base_tokenizer),
            "new_vocab": new_len,
            "tie_word_embeddings": tied,
            "tied_actual": tied_actual,
            "original_embedding_shape": [old_rows, hidden_size],
            "new_embedding_shape": list(input_weight.shape),
            "lm_head_shape": list(output_module.weight.shape),
        },
        "counts": {
            "total": new_len,
            "exact_copy": counts[EXACT],
            "decomposition": counts[DECOMPOSED],
            "random_fallback": counts[RANDOM],
            "special_exact": special_exact,
            "special_not_exact": special_not_exact,
            "control_tokens": len(control_report),
            "empty_surface": len(empty_surface),
            "replacement_char_surface": len(replacement_surface),
            "duplicate_surface_groups": len(duplicate_surfaces),
        },
        "ratios": {
            "exact_copy": counts[EXACT] / new_len,
            "decomposition": counts[DECOMPOSED] / new_len,
            "random_fallback": counts[RANDOM] / new_len,
        },
        "decomposition_lengths": {
            "buckets": buckets,
            "overall": length_summary(decomp_lengths),
            "hangul": length_summary(hangul_lengths),
            "ascii": length_summary(ascii_lengths),
        },
        "control_tokens": control_report,
        "preserved_base_added_token_ids": preserved_added,
        "control_token_ids": control_ids,
        "base_special_tokens": special_token_strings(base_tokenizer),
        "new_special_tokens": special_token_strings(new_tokenizer),
        "norms": {
            "original_embedding": describe([float(v) for v in old_norms[::37]]),
            "new_embedding": describe([float(v) for v in new_norms]),
            "by_strategy": norm_by_strategy,
        },
        "exact_copy_cosine": {
            "sampled": len(cosines),
            "mean": float(statistics.fmean(cosines)) if cosines else None,
            "min": min(cosines) if cosines else None,
            "max": max(cosines) if cosines else None,
        },
        "duplicate_surface_examples": {
            surface: ids for surface, ids in list(duplicate_surfaces.items())[:16]
        },
        "empty_surface_ids": empty_surface[:32],
        "replacement_surface_ids": replacement_surface[:32],
        "parameters": {
            "original_total": original_total_params,
            "new_total": total_params,
            "original_embedding": original_embedding_params,
            "new_embedding": embedding_params,
            "non_embedding": total_params - embedding_params,
            "reduction": original_total_params - total_params,
            "reduction_ratio": (original_total_params - total_params) / original_total_params,
            "embedding_reduction_ratio": (
                (original_embedding_params - embedding_params) / original_embedding_params
            ),
        },
        "examples": collect_examples(plans, new_input=input_weight),
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
    }

    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print("dry run: nothing saved")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    new_tokenizer.save_pretrained(output_dir)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(output_dir)
    (output_dir / "transplant_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report["counts"], ensure_ascii=False, indent=2))
    print(f"saved transplant checkpoint to {output_dir}")
    print(f"saved report to {report_path}")


def collect_examples(plans: list[dict], new_input: torch.Tensor) -> dict:
    """Representative rows for the report: Korean, technical, code, JSON, control."""
    def entry(plan: dict) -> dict:
        return {
            "new_id": plan["new_id"], "token": plan["token"], "surface": plan["surface"],
            "strategy": plan["strategy"], "byte_level": plan["byte_level"],
            "source_pieces": plan["pieces"], "source_ids": plan["source_ids"],
            "norm": float(new_input[plan["new_id"]].detach().to(torch.float32).norm()),
        }

    groups: dict[str, list[dict]] = {
        "control": [], "base_added": [], "korean": [], "ascii_code": [], "json_like": [],
    }
    for plan in plans:
        surface = plan["surface"]
        if plan["is_control"]:
            groups["control"].append(entry(plan))
        elif plan["is_base_added"] and len(groups["base_added"]) < 6:
            groups["base_added"].append(entry(plan))
        elif has_hangul(surface) and plan["strategy"] == DECOMPOSED and len(groups["korean"]) < 10:
            groups["korean"].append(entry(plan))
        elif surface.strip() in {'":', '":"', "},{", '"},', "[{", "}]"} and len(groups["json_like"]) < 6:
            groups["json_like"].append(entry(plan))
        elif surface.isascii() and surface.strip().isidentifier() and len(groups["ascii_code"]) < 8:
            groups["ascii_code"].append(entry(plan))
    return groups


if __name__ == "__main__":
    main()
