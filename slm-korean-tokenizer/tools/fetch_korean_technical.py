#!/usr/bin/env python3
"""Fetch additional technical Korean documents for the Stage-2 corpus.

Source: Korean Wikipedia (2024-05-01 dump), redistributed as
`lcw99/wikipedia-korean-20240501` — public, ungated, natural prose, stable
per-article provenance. Wikipedia text is CC BY-SA, which permits reuse with
attribution; the dataset card additionally states apache-2.0 for the packaging.

Rejected candidates and why:
  maywell/korean_textbooks      LLM-generated (claude_evol / code-alpaca subsets)
  devngho/korean-textbooks-edu  gated=auto, terms acceptance required
  namuwiki derivatives          CC BY-NC-SA: non-commercial restriction
  Open-Korean-Historical-Corpus CC BY-NC + historical, not technical

Wikipedia carries many list and stub articles, so selection is deliberately
strict: the article must read as technical prose, not as a bullet dump.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "corpus"))

from build_recovery_data import assert_record_allowed, assert_source_allowed  # noqa: E402
from stage2_dedup import content_hash, normalize, quality_verdict  # noqa: E402

DATASET = "lcw99/wikipedia-korean-20240501"
LICENSE = ("Korean Wikipedia text: CC BY-SA 3.0/4.0 (attribution required); "
           "dataset packaging card states apache-2.0")

TECH_TERMS = [
    "과학", "기술", "공학", "물리", "화학", "생물", "수학", "컴퓨터", "소프트웨어", "알고리즘",
    "프로그래밍", "데이터", "전자", "반도체", "네트워크", "통신", "의학", "약학", "천문", "지질",
    "재료", "기계공", "전기", "에너지", "인공지능", "기계 학습", "암호", "운영 체제", "데이터베이스",
    "단백질", "세포", "유전자", "분자", "원자", "양자", "화합물", "미생물", "효소", "바이러스",
    "이론", "방정식", "함수", "입자", "전압", "전류", "주파수", "궤도", "위성", "로봇", "센서", "회로",
]


def korean_ratio(text: str) -> float:
    korean = sum(1 for char in text if "가" <= char <= "힣")
    return korean / max(len(text), 1)


def bullet_ratio(text: str) -> float:
    lines = [line for line in text.split("\n") if line.strip()]
    if not lines:
        return 1.0
    return sum(1 for line in lines if line.lstrip().startswith(("*", "#", "|"))) / len(lines)


def is_technical_prose(title: str, text: str, *, min_chars: int, min_korean: float,
                       max_bullets: float, min_distinct: int, min_density: float) -> str | None:
    """Return a rejection reason, or None when the article qualifies."""
    if len(text) < min_chars:
        return "too_short"
    if korean_ratio(text) < min_korean:
        return "not_korean_enough"
    if bullet_ratio(text) > max_bullets:
        return "list_article"
    if not any(term in (title + " " + text[:300]) for term in TECH_TERMS):
        return "not_technical_topic"
    distinct = sum(1 for term in TECH_TERMS if term in text)
    density = sum(text.count(term) for term in TECH_TERMS) / (len(text) / 1000)
    if distinct < min_distinct:
        return "too_few_technical_terms"
    if density < min_density:
        return "technical_density_too_low"
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/raw/korean_technical_wiki.jsonl")
    parser.add_argument("--report", default="outputs/korean_technical_acquisition.json")
    parser.add_argument("--target-chars", type=int, default=22_000_000)
    parser.add_argument("--max-shards", type=int, default=7)
    parser.add_argument("--min-chars", type=int, default=500)
    parser.add_argument("--max-chars", type=int, default=100_000)
    parser.add_argument("--min-korean", type=float, default=0.33)
    parser.add_argument("--max-bullets", type=float, default=0.30)
    parser.add_argument("--min-distinct", type=int, default=4)
    parser.add_argument("--min-density", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from huggingface_hub import hf_hub_download, list_repo_files
    import pyarrow.parquet as pq

    assert_source_allowed("korean_wikipedia", DATASET)
    shards = sorted(name for name in list_repo_files(DATASET, repo_type="dataset")
                    if name.endswith(".parquet"))[: args.max_shards]
    print(f"[tech] {len(shards)} shards", flush=True)

    print("[tech] indexing existing corpus hashes", flush=True)
    known: set[str] = set()
    for path in [PROJECT_ROOT / "data/processed/stage2_train.jsonl",
                 PROJECT_ROOT / "data/processed/stage2_aihub_technical.jsonl",
                 PROJECT_ROOT / "data/eval/recovery_balanced_eval.jsonl",
                 PROJECT_ROOT / "data/eval/recovery_korean_large_eval.jsonl",
                 PROJECT_ROOT / "data/processed/recovery_train.jsonl"]:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    try:
                        known.add(content_hash(json.loads(line)["text"]))
                    except (json.JSONDecodeError, KeyError):
                        continue
    print(f"[tech] {len(known):,} known hashes", flush=True)

    counts = {"scanned": 0, "accepted": 0, "accepted_chars": 0, "rejected_duplicate": 0}
    rejections: dict[str, int] = {}
    used: list[str] = []
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as handle:
        for shard in shards:
            if counts["accepted_chars"] >= args.target_chars:
                break
            print(f"[tech] downloading {shard}", flush=True)
            local = hf_hub_download(DATASET, shard, repo_type="dataset")
            used.append(shard)
            for batch in pq.ParquetFile(local).iter_batches(
                    batch_size=2000, columns=["title", "text"]):
                for row in batch.to_pylist():
                    counts["scanned"] += 1
                    title = (row.get("title") or "").strip()
                    text = normalize(row.get("text") or "")
                    reason = is_technical_prose(
                        title, text, min_chars=args.min_chars, min_korean=args.min_korean,
                        max_bullets=args.max_bullets, min_distinct=args.min_distinct,
                        min_density=args.min_density)
                    if reason is None:
                        reason = quality_verdict(text, min_chars=args.min_chars,
                                                 max_chars=args.max_chars, max_token_length=400)
                    if reason:
                        rejections[reason] = rejections.get(reason, 0) + 1
                        continue
                    digest = content_hash(text)
                    if digest in known:
                        counts["rejected_duplicate"] += 1
                        continue
                    known.add(digest)
                    assert_record_allowed(text, "korean_wikipedia_technical")
                    handle.write(json.dumps({
                        "text": text,
                        "source": "korean_wikipedia_technical",
                        "source_id": title or digest[:16],
                        "dataset": DATASET,
                        "shard": shard,
                        "title": title,
                        "license": LICENSE,
                        "chars": len(text),
                        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    }, ensure_ascii=False) + "\n")
                    counts["accepted"] += 1
                    counts["accepted_chars"] += len(text)
                if counts["scanned"] % 20000 < 2000:
                    print(f"  scanned {counts['scanned']:,} | accepted {counts['accepted']:,} "
                          f"({counts['accepted_chars']:,} chars)", flush=True)
                if counts["accepted_chars"] >= args.target_chars:
                    print("[tech] target reached; stopping", flush=True)
                    break

    report = {
        "dataset": DATASET, "provider": "lcw99 (Korean Wikipedia 2024-05-01 dump)",
        "access": "public, ungated, huggingface_hub", "license": LICENSE,
        "split": "train", "shards_used": used,
        "selection": {
            "min_chars": args.min_chars, "min_korean_ratio": args.min_korean,
            "max_bullet_ratio": args.max_bullets,
            "min_distinct_technical_terms": args.min_distinct,
            "min_technical_density_per_1000_chars": args.min_density,
        },
        "target_chars": args.target_chars,
        **counts, "rejections": rejections, "output": str(out_path),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "shards_used"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
