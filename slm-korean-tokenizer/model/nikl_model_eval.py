#!/usr/bin/env python3
"""NIKL Newspaper 2025 external *model* evaluation (NLL/char + tokenizer efficiency).

Provenance, stated precisely:
* NIKL was already used as the external validation set of the *tokenizer*
  benchmark (outputs/nikl_tokenizer_benchmark.json). It is therefore not a
  never-before-seen benchmark for this project.
* It was not in any tokenizer corpus, the Stage-1 recovery corpus or the Stage-2
  corpus (checked here by exact paragraph hashing, see ``--provenance``), and it
  was not used for Stage-2 checkpoint selection or early stopping.

Scored text is the article **body** exactly as the manifest fixed it
(``corpus/nikl.py`` cleaning, NFC). Each article is checked against the
manifest's ``chars`` and ``text_hash`` before it is scored, so every model and
the earlier tokenizer benchmark share one text and one character denominator.

NLL/char is computed by ``recovery_eval.evaluate`` — the canonical definition,
with no truncation (articles are scored in full). The default run uses no
context prefix; ``--prefix-sensitivity`` re-scores with ``<|endoftext|>`` as a
context-only prefix to measure the first-token caveat.

Work is split into shards of consecutive selected articles and each
(model, shard) result is written to disk, so the full run is resumable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "corpus"))

from nikl import DEFAULT_ROOT, corpus_files, digest, iter_articles, short_digest  # noqa: E402
from recovery_eval import evaluate  # noqa: E402

MODELS = {
    "original_qwen": "Qwen/Qwen3-0.6B",
    "stage1_final": "outputs/models/qwen3-0.6b-ko32k-recovery-stage1-extended",
    "stage2_signal2678": "outputs/models/qwen3-0.6b-ko32k-stage2-signal2678",
}
LENGTH_BUCKETS = ((0, 500), (500, 1_000), (1_000, 2_000), (2_000, 4_000), (4_000, None))
MODEL_TRAINING_CORPORA = {
    "stage1_recovery_train": "data/processed/recovery_train.jsonl",
    "stage2_train_v3": "data/processed/stage2_train_v3.jsonl",
    "stage2_holdout_v3": "data/processed/stage2_holdout_v3.jsonl",
}


def length_bucket(n: int) -> str:
    for low, high in LENGTH_BUCKETS:
        if high is None or low <= n < high:
            return f"{low}-{high}" if high is not None else f"{low}+"
    return "unknown"


def selection_bucket(article_id: str) -> int:
    """Deterministic 0..9999 bucket; ``fraction`` keeps buckets below fraction*10000."""
    return int(hashlib.sha256(article_id.encode("utf-8")).hexdigest()[:8], 16) % 10_000


def load_manifest(path: str) -> dict[str, dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return {row["article_id"]: row for row in map(json.loads, handle) if row}


def iter_selected(manifest: dict[str, dict], fraction: float, root: str):
    """Yield verified records for the selected articles, in corpus order."""
    cutoff = int(round(fraction * 10_000))
    for article in iter_articles(corpus_files(root)):
        row = manifest.get(article.article_id)
        if row is None or selection_bucket(article.article_id) >= cutoff:
            continue
        body = article.body
        if len(body) != row["chars"] or digest(body) != row["text_hash"]:
            raise RuntimeError(f"article {article.article_id} does not match the manifest")
        yield {
            "id": article.article_id,
            "text": body,
            "characters": len(body),
            "topic": row["topic"],
            "publisher": row["publisher"],
            "length_bucket": length_bucket(len(body)),
            "paragraphs": body.count("\n") + 1,
        }


def shard_path(out_dir: Path, tag: str, model: str, shard: int) -> Path:
    return out_dir / "nikl_shards" / tag / model / f"shard_{shard:05d}.json"


CHUNK_ARTICLES = 128
MAX_LOGIT_ELEMENTS = 48_000_000
PAD_TO_MULTIPLE = 128


def score_shard(models: dict, records: list[dict], device: str, prefix: bool) -> dict:
    """Score one shard, in small chunks.

    The MPS caching allocator keeps every distinct large block it has seen; long
    newspaper articles produce many distinct shapes, and a first attempt at a
    single call per shard grew the process to 77 GB and stalled on swap (the same
    failure mode as Stage-2 training run 1). Chunking plus ``empty_cache`` bounds
    it. Per-article NLL does not depend on batch composition (padding is masked).
    """
    out = {}
    for name, (model, tokenizer) in models.items():
        prefix_ids = [tokenizer.convert_tokens_to_ids("<|endoftext|>")] if prefix else None
        nll: list[float] = []
        skipped = 0
        peak = 0.0
        for start in range(0, len(records), CHUNK_ARTICLES):
            chunk = records[start:start + CHUNK_ARTICLES]
            result = evaluate(model, tokenizer, chunk, device=device, batch_size=16,
                              max_logit_elements=MAX_LOGIT_ELEMENTS, truncate=False,
                              context_prefix_ids=prefix_ids, category_key="topic",
                              pad_to_multiple=PAD_TO_MULTIPLE)
            nll += [result["per_record_nll"].get(i, 0.0) for i in range(len(chunk))]
            skipped += result["skipped_records"]
            if device == "mps":
                peak = max(peak, torch.mps.driver_allocated_memory() / 1e9)
                torch.mps.empty_cache()
        tokens = [len(tokenizer(r["text"], add_special_tokens=True)["input_ids"]) for r in records]
        out[name] = {
            "ids": [r["id"] for r in records],
            "nll": nll,
            "tokens": tokens,
            "skipped_records": skipped,
            "peak_driver_gb": peak,
        }
    return out


def load_models(names: list[str], device: str, dtype: torch.dtype) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    loaded = {}
    for name in names:
        path = MODELS[name]
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device).eval()
        loaded[name] = (model, tokenizer)
    return loaded


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    manifest = load_manifest(args.manifest)
    models = load_models(args.models, args.device, getattr(torch, args.dtype))
    shard, batch, started = 0, [], time.time()
    done_articles = 0

    def flush(records: list[dict], index: int) -> None:
        pending = {n: m for n, m in models.items()
                   if not shard_path(out_dir, args.tag, n, index).exists()}
        if not pending:
            return
        scored = score_shard(pending, records, args.device, args.prefix_sensitivity)
        meta = [{k: r[k] for k in ("id", "characters", "topic", "publisher",
                                   "length_bucket", "paragraphs")} for r in records]
        for name, payload in scored.items():
            path = shard_path(out_dir, args.tag, name, index)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"records": meta, **payload}, ensure_ascii=False),
                           encoding="utf-8")
            tmp.replace(path)
        if args.device == "mps":
            torch.mps.empty_cache()

    for record in iter_selected(manifest, args.fraction, args.root):
        batch.append(record)
        if len(batch) >= args.shard_size:
            flush(batch, shard)
            done_articles += len(batch)
            shard += 1
            batch = []
            elapsed = time.time() - started
            driver = torch.mps.driver_allocated_memory() / 1e9 if args.device == "mps" else 0.0
            print(f"[nikl] shard {shard} articles {done_articles:,} elapsed {elapsed:,.0f}s "
                  f"mps_driver {driver:.1f}GB", flush=True)
    if batch:
        flush(batch, shard)
        done_articles += len(batch)
        shard += 1
    (out_dir / "nikl_shards" / args.tag / "COMPLETE.json").write_text(json.dumps({
        "fraction": args.fraction, "shards": shard, "articles": done_articles,
        "models": args.models, "dtype": args.dtype, "prefix_sensitivity": args.prefix_sensitivity,
    }) + "\n", encoding="utf-8")
    print(f"[nikl] complete: {done_articles:,} articles in {shard} shards", flush=True)


# --------------------------------------------------------------------------- #
# Aggregation (pure; unit-tested)
# --------------------------------------------------------------------------- #

def load_shards(out_dir: Path, tag: str, model: str) -> list[dict]:
    rows = []
    for path in sorted((out_dir / "nikl_shards" / tag / model).glob("shard_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for meta, nll, tokens in zip(payload["records"], payload["nll"], payload["tokens"]):
            rows.append({**meta, "nll": nll, "tokens": tokens})
    return rows


def summarize_rows(rows: list[dict]) -> dict:
    chars = sum(r["characters"] for r in rows)
    nll = sum(r["nll"] for r in rows)
    tokens = sum(r["tokens"] for r in rows)
    predicted = sum(max(0, r["tokens"] - 1) for r in rows)
    return {
        "articles": len(rows),
        "paragraphs": sum(r["paragraphs"] for r in rows),
        "characters": chars,
        "total_nll": nll,
        "nll_per_character": nll / chars if chars else float("nan"),
        "bits_per_character": nll / chars / 0.6931471805599453 if chars else float("nan"),
        "total_tokens": tokens,
        "tokens_per_character": tokens / chars if chars else float("nan"),
        "characters_per_token": chars / tokens if tokens else float("nan"),
        "mean_nll_per_predicted_token": nll / predicted if predicted else float("nan"),
    }


def grouped(rows: list[dict], key: str) -> dict:
    groups: dict[str, list] = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return {name: summarize_rows(items) for name, items in sorted(groups.items())}


def paired_bootstrap(a: list[dict], b: list[dict], iterations: int = 1000, seed: int = 42) -> dict:
    """Article-level paired bootstrap of NLL/char(a) - NLL/char(b)."""
    import random

    by_id = {r["id"]: r for r in b}
    pairs = [(r["nll"], by_id[r["id"]]["nll"], r["characters"]) for r in a if r["id"] in by_id]
    rng = random.Random(seed)
    n = len(pairs)
    values = []
    for _ in range(iterations):
        na = nb = ch = 0.0
        for _ in range(n):
            x, y, c = pairs[rng.randrange(n)]
            na += x
            nb += y
            ch += c
        values.append((na - nb) / ch)
    values.sort()
    point = (sum(p[0] for p in pairs) - sum(p[1] for p in pairs)) / sum(p[2] for p in pairs)
    return {"articles": n, "delta_nll_per_character": point,
            "ci95_low": values[int(0.025 * iterations)],
            "ci95_high": values[int(0.975 * iterations) - 1]}


# --------------------------------------------------------------------------- #
# Provenance: exact paragraph overlap with the model-training corpora
# --------------------------------------------------------------------------- #

def provenance(args: argparse.Namespace) -> dict:
    table: dict[int, int] = {}
    names = list(MODEL_TRAINING_CORPORA)
    lines_hashed = {}
    for bit, name in enumerate(names):
        path = ROOT / MODEL_TRAINING_CORPORA[name]
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                for piece in record["text"].split("\n"):
                    piece = piece.strip()
                    if len(piece) >= 20:
                        key = short_digest(piece)
                        table[key] = table.get(key, 0) | (1 << bit)
                        count += 1
        lines_hashed[name] = count
    manifest = load_manifest(args.manifest)
    paragraphs = overlapping = articles_hit = 0
    by_corpus = {name: 0 for name in names}
    examples = []
    for record in iter_selected(manifest, 1.0, args.root):
        hit = False
        for piece in record["text"].split("\n"):
            piece = piece.strip()
            if len(piece) < 20:
                continue
            paragraphs += 1
            mask = table.get(short_digest(piece), 0)
            if mask:
                overlapping += 1
                hit = True
                for bit, name in enumerate(names):
                    if mask & (1 << bit):
                        by_corpus[name] += 1
                if len(examples) < 20:
                    examples.append({"article_id": record["id"], "paragraph": piece[:120]})
        articles_hit += hit
    return {
        "method": "blake2b-64 of every stripped line >= 20 chars in each training record vs "
                  "every stripped NIKL body paragraph >= 20 chars (exact match)",
        "training_lines_hashed": lines_hashed,
        "nikl_paragraphs_checked": paragraphs,
        "overlapping_paragraphs": overlapping,
        "articles_with_overlap": articles_hit,
        "overlap_by_corpus": by_corpus,
        "examples": examples,
    }


def tokens_only(args: argparse.Namespace) -> dict:
    """Full-corpus token counts for the model tokenizers (no model is loaded).

    Stage-1 and Stage-2 share one tokenizer (identical vocab/merges); both names are
    reported so the table is explicit.
    """
    from transformers import AutoTokenizer

    tokenizers = {"original_qwen": AutoTokenizer.from_pretrained(MODELS["original_qwen"]),
                  "ko32k_stage2": AutoTokenizer.from_pretrained(MODELS["stage2_signal2678"])}
    manifest = load_manifest(args.manifest)
    totals = {name: 0 for name in tokenizers}
    groups = {key: defaultdict(lambda: defaultdict(int)) for key in ("topic", "publisher", "length_bucket")}
    articles = characters = paragraphs = 0
    batch: list[dict] = []

    def flush(rows: list[dict]) -> None:
        nonlocal articles, characters, paragraphs
        texts = [r["text"] for r in rows]
        counts = {n: [len(ids) for ids in t(texts, add_special_tokens=True)["input_ids"]]
                  for n, t in tokenizers.items()}
        for i, row in enumerate(rows):
            articles += 1
            characters += row["characters"]
            paragraphs += row["paragraphs"]
            for key in groups:
                bucket = groups[key][row[key]]
                bucket["articles"] += 1
                bucket["characters"] += row["characters"]
                for name in tokenizers:
                    bucket[name] += counts[name][i]
            for name in tokenizers:
                totals[name] += counts[name][i]

    for record in iter_selected(manifest, args.fraction, args.root):
        batch.append(record)
        if len(batch) >= 1024:
            flush(batch)
            batch = []
    if batch:
        flush(batch)

    def block(row: dict) -> dict:
        q, k = row["original_qwen"], row["ko32k_stage2"]
        return {"articles": row["articles"], "characters": row["characters"],
                "qwen_tokens": q, "ko32k_tokens": k,
                "qwen_tokens_per_char": q / row["characters"], "ko32k_tokens_per_char": k / row["characters"],
                "token_reduction_percent": 100.0 * (1.0 - k / q)}

    report = {
        "fraction": args.fraction, "articles": articles, "paragraphs": paragraphs,
        "characters": characters,
        "overall": block({"articles": articles, "characters": characters, **totals}),
        "characters_per_token": {n: characters / totals[n] for n in totals},
        **{f"by_{key}": {name: block(row) for name, row in sorted(table.items())}
           for key, table in groups.items()},
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--manifest", default="data/eval/nikl_eval_manifest.jsonl")
    parser.add_argument("--output-dir", default="outputs/final_validation")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--tag", default="full")
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--shard-size", type=int, default=2_000)
    parser.add_argument("--prefix-sensitivity", action="store_true")
    parser.add_argument("--provenance", action="store_true",
                        help="Only run the training-corpus overlap check.")
    parser.add_argument("--tokens-only", action="store_true",
                        help="Only count tokens with the model tokenizers (no model forward).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.provenance:
        report = provenance(args)
        path = Path(args.output_dir) / "nikl_provenance.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k != "examples"}, indent=2))
        return
    if args.tokens_only:
        report = tokens_only(args)
        path = Path(args.output_dir) / "nikl_tokenizer_efficiency_full.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report["overall"], indent=2))
        return
    run(args)


if __name__ == "__main__":
    main()
