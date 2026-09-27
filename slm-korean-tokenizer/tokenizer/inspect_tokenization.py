#!/usr/bin/env python3
from __future__ import annotations

import argparse

from common import CONTROL_TOKENS, load_tokenizer, tokenizer_name

DEFAULT_TEXTS = [
    "가상메모리는 프로세스에게 물리 메모리와 독립된 주소 공간을 제공한다.",
    "운영체제는 페이지 폴트가 발생하면 필요한 페이지를 물리 메모리에 적재한다.",
    "Apple Silicon은 CPU와 GPU가 통합 메모리를 공유한다.",
    "소형 언어모델은 제한된 자원에서 효율적인 추론을 수행할 수 있다.",
    "<검색>Apple Silicon 통합 메모리",
    "<요약><이전0>",
    "<SQL>지난달 주문량을 조회한다.",
    "https://example.com/test?q=memory",
    "SELECT name FROM users WHERE id = 123;",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect token pieces for baseline and Korean-focused tokenizers.")
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--text", action="append", help="Text to inspect; repeat for multiple texts")
    parser.add_argument(
        "--tokenizers",
        nargs="+",
        default=[
            "outputs/real/mf2/tokenizer-ko-16000",
            "outputs/real/mf2/tokenizer-ko-32000",
        ],
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    texts = args.text or DEFAULT_TEXTS
    sources = [("Original", args.base_model)] + [(tokenizer_name(path), path) for path in args.tokenizers]
    loaded = [(name, load_tokenizer(source, args.trust_remote_code)) for name, source in sources]
    for text in texts:
        print(f"\nTEXT: {text}")
        for name, tokenizer in loaded:
            ids = tokenizer.encode(text, add_special_tokens=False)
            tokens = tokenizer.convert_ids_to_tokens(ids)
            print(f"\n=== {name} ===")
            print(f"tokens: {tokens}")
            print(f"ids: {ids}")
            print(f"count: {len(ids)}")
    print("\n=== Skill control token atomicity ===")
    for name, tokenizer in loaded:
        details = []
        for token in CONTROL_TOKENS:
            ids = tokenizer.encode(token, add_special_tokens=False)
            details.append(f"{token}={ids} ({'PASS' if len(ids) == 1 else 'SPLIT'})")
        print(f"{name}: " + ", ".join(details))


if __name__ == "__main__":
    main()
