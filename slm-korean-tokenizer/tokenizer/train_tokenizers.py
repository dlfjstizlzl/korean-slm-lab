#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from transformers import AutoTokenizer

from common import (
    CONTROL_TOKENS,
    assert_base_added_tokens_preserved,
    assert_control_tokens_atomic,
    assert_required_special_tokens,
    corpus_file_iterator,
    corpus_file_stats,
    iter_corpus,
    non_special_added_tokens,
    special_token_strings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Korean-focused tokenizers from an NFC-normalized text corpus.")
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--train-file", default="data/korean_train.txt")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--name-suffix", default="", help="Optional output name suffix, for example mixed10")
    parser.add_argument("--vocab-sizes", nargs="+", type=int, default=[16000, 24000, 32000])
    parser.add_argument("--trainer", choices=["hf", "sentencepiece"], default="hf")
    parser.add_argument("--min-frequency", type=int, default=2, help="HF BPE minimum pair frequency")
    parser.add_argument("--batch-size", type=int, default=1000, help="Streaming text batch size")
    parser.add_argument("--sentencepiece-model-type", choices=["unigram", "bpe"], default="unigram")
    parser.add_argument("--deduplicate", action="store_true")
    parser.add_argument(
        "--keep-comment-lines",
        action="store_true",
        help="Treat lines beginning with # as corpus text (recommended for real web corpora)",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def output_path(
    root: Path,
    size: int,
    trainer: str,
    sentencepiece_model_type: str = "unigram",
    name_suffix: str = "",
) -> Path:
    prefix = "tokenizer-ko" if trainer == "hf" else f"tokenizer-sp-{sentencepiece_model_type}-ko"
    suffix = f"-{name_suffix.strip('-')}" if name_suffix.strip("-") else ""
    return root / f"{prefix}-{size}{suffix}"


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def train_hf(args: argparse.Namespace, corpus_stats: dict[str, int], base_tokenizer) -> list[dict]:
    results = []
    root = Path(args.output_dir)
    preserved_added = non_special_added_tokens(base_tokenizer)
    for requested_size in args.vocab_sizes:
        destination = output_path(root, requested_size, "hf", name_suffix=args.name_suffix)
        prepare_output(destination, args.overwrite)
        training_vocab_size = requested_size - len(preserved_added)
        if training_vocab_size <= 0:
            raise ValueError(f"Requested vocab {requested_size} cannot reserve {len(preserved_added)} base added tokens")
        tokenizer = base_tokenizer.train_new_from_iterator(
            corpus_file_iterator(
                args.train_file,
                batch_size=args.batch_size,
                deduplicate=args.deduplicate,
                skip_comment_lines=not args.keep_comment_lines,
            ),
            vocab_size=training_vocab_size,
            new_special_tokens=CONTROL_TOKENS,
            length=corpus_stats["lines"],
            min_frequency=args.min_frequency,
        )
        existing_additional = [str(value) for value in tokenizer.additional_special_tokens]
        tokenizer.add_special_tokens(
            {"additional_special_tokens": list(dict.fromkeys(existing_additional + CONTROL_TOKENS))},
            replace_additional_special_tokens=True,
        )
        tokenizer.add_tokens(preserved_added, special_tokens=False)
        assert_required_special_tokens(base_tokenizer, tokenizer)
        preserved_added_ids = assert_base_added_tokens_preserved(base_tokenizer, tokenizer)
        control_ids = assert_control_tokens_atomic(tokenizer)
        tokenizer.save_pretrained(destination)
        actual_size = len(tokenizer)
        metadata = {
            "trainer": "hf",
            "base_model": args.base_model,
            "train_file": args.train_file,
            "name_suffix": args.name_suffix,
            "requested_vocab_size": requested_size,
            "trainer_vocab_budget": training_vocab_size,
            "min_frequency": args.min_frequency,
            "actual_vocab_size": actual_size,
            "train_lines": corpus_stats["lines"],
            "train_characters": corpus_stats["characters"],
            "streaming": True,
            "keep_comment_lines": args.keep_comment_lines,
            "control_token_ids": control_ids,
            "preserved_base_added_token_ids": preserved_added_ids,
            "base_special_tokens": special_token_strings(base_tokenizer),
        }
        (destination / "training_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if actual_size != requested_size:
            print(f"WARNING: requested {requested_size:,}, corpus produced {actual_size:,} tokens", file=sys.stderr)
        print(f"saved {destination} (requested={requested_size:,}, actual={actual_size:,})")
        results.append(metadata | {"path": str(destination)})
    return results


def train_sentencepiece(args: argparse.Namespace, corpus_stats: dict[str, int], base_tokenizer) -> list[dict]:
    import sentencepiece as spm

    base_specials = list(dict.fromkeys(special_token_strings(base_tokenizer).values()))
    user_symbols = list(dict.fromkeys(CONTROL_TOKENS + base_specials))
    results = []
    root = Path(args.output_dir)
    with tempfile.TemporaryDirectory(prefix="slm-ko-sp-") as temp_dir:
        normalized = Path(temp_dir) / "normalized.txt"
        with normalized.open("w", encoding="utf-8") as handle:
            for record in iter_corpus(
                args.train_file,
                deduplicate=args.deduplicate,
                skip_comment_lines=not args.keep_comment_lines,
            ):
                handle.write(record.text + "\n")
        for requested_size in args.vocab_sizes:
            destination = output_path(
                root,
                requested_size,
                "sentencepiece",
                args.sentencepiece_model_type,
                args.name_suffix,
            )
            prepare_output(destination, args.overwrite)
            prefix = destination / "tokenizer"
            spm.SentencePieceTrainer.train(
                input=str(normalized),
                model_prefix=str(prefix),
                vocab_size=requested_size,
                model_type=args.sentencepiece_model_type,
                character_coverage=0.9999,
                byte_fallback=True,
                user_defined_symbols=user_symbols,
                normalization_rule_name="identity",
                add_dummy_prefix=False,
                hard_vocab_limit=False,
                bos_id=-1,
                eos_id=-1,
                pad_id=-1,
            )
            processor = spm.SentencePieceProcessor(model_file=str(prefix) + ".model")
            failures = {token: processor.encode(token, out_type=int) for token in CONTROL_TOKENS if len(processor.encode(token, out_type=int)) != 1}
            if failures:
                raise RuntimeError(f"SentencePiece control tokens are not atomic: {failures}")
            metadata = {
                "trainer": "sentencepiece",
                "model_type": args.sentencepiece_model_type,
                "base_model": args.base_model,
                "train_file": args.train_file,
                "name_suffix": args.name_suffix,
                "requested_vocab_size": requested_size,
                "actual_vocab_size": processor.vocab_size(),
                "train_lines": corpus_stats["lines"],
                "train_characters": corpus_stats["characters"],
                "streaming_normalization": True,
                "keep_comment_lines": args.keep_comment_lines,
                "character_coverage": 0.9999,
                "byte_fallback": True,
                "user_defined_symbols": user_symbols,
                "note": "Benchmark artifact only; embedding transplant currently requires an HF tokenizer directory.",
            }
            (destination / "sentencepiece_meta.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print(f"saved {destination} (requested={requested_size:,}, actual={processor.vocab_size():,})")
            results.append(metadata | {"path": str(destination)})
    return results


def main() -> None:
    args = parse_args()
    stats = corpus_file_stats(
        args.train_file,
        deduplicate=args.deduplicate,
        skip_comment_lines=not args.keep_comment_lines,
    )
    if not stats["lines"]:
        raise ValueError(f"No usable records in {args.train_file}")
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    print(f"scanned {stats['lines']:,} NFC-normalized training lines / {stats['characters']:,} characters")
    print(f"base tokenizer vocab: {len(base_tokenizer):,}")
    if args.trainer == "hf":
        train_hf(args, stats, base_tokenizer)
    else:
        train_sentencepiece(args, stats, base_tokenizer)


if __name__ == "__main__":
    main()
