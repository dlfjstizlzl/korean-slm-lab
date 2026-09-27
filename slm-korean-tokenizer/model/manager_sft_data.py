#!/usr/bin/env python3
"""Manager SFT data: Runtime-identical rendering, assistant-only labels, guards.

Rendering is the Runtime's, not a new prompt:

* the system prompt comes from `slm_taskplan_tool --contract`, i.e. the exact
  string `ManagerClient::system_prompt()` that the Runtime sends
* the chat format is the one llama.cpp applies to this model's template in
  `LlamaCppInferenceBackend::format_prompt` (ChatML via
  `llama_chat_apply_template`, add_generation_prompt=true):
      <|im_start|>system\\n{system}<|im_end|>\\n<|im_start|>user\\n{request}<|im_end|>\\n<|im_start|>assistant\\n
  The HF tokenizer's own chat template yields byte-identical generation
  prompts (checked by `assert_template_parity`).
* the target is the row's `plan_text` (= C++ `TaskPlan::to_json()` bytes) and
  then the tokenizer's EOS `<|im_end|>`, which is what the Runtime stops on.

Prompt tokens are masked with -100; only target tokens + EOS carry loss.
The prompt and the target are tokenized separately, exactly as at inference
time, where the prompt is tokenized alone and the plan is generated token by
token after it.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "model"))

from manager_v2_registry import load_contract  # noqa: E402
from manager_v2_validator import canonical_json, validate  # noqa: E402

BASE_MODEL = PROJECT_ROOT / "outputs" / "models" / "qwen3-0.6b-ko32k-stage2-signal2678"
BASE_MODEL_NAME = "Qwen3-0.6B-KO32K-Stage2-signal2678"
DATA_DIR = PROJECT_ROOT / "data" / "manager"
SPLIT_FILES = {"train": DATA_DIR / "manager_v1_5k_train.jsonl",
               "validation": DATA_DIR / "manager_v1_5k_val.jsonl",
               "test": DATA_DIR / "manager_v1_5k_test.jsonl"}
EXPECTED_ROWS = {"train": 4085, "validation": 454, "test": 461}
SEALED_TEST_SHA256 = "6cdb3f55904cfa3f77dce9f37ec6a9bb1c3f1dea85d3b19c24e0beefffb5f98e"
IGNORE_INDEX = -100
# Written once by the one-time final evaluation (model/manager_final_test.py). While it exists the
# sealed test can only be read again with an explicit consumed-test override.
CONSUMED_MARKER = PROJECT_ROOT / "outputs" / "manager_final_test_v1" / "SEALED_TEST_CONSUMED.json"


class SealedTestError(PermissionError):
    """Raised whenever code asks for the sealed test split without explicit permission."""


class SealedTestConsumedError(SealedTestError):
    """Raised when the sealed test was already consumed by the final evaluation."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sealed_test_fingerprint() -> dict:
    """Existence + SHA-256 only. The content is never parsed here."""
    path = SPLIT_FILES["test"]
    return {"path": str(path.relative_to(PROJECT_ROOT)), "exists": path.is_file(),
            "sha256": sha256_file(path) if path.is_file() else None,
            "matches_recorded_sha256": path.is_file() and sha256_file(path) == SEALED_TEST_SHA256}


def load_split(split: str, *, allow_sealed_test: bool = False, allow_consumed_test: bool = False) -> list[dict]:
    if split == "test":
        if not allow_sealed_test:
            raise SealedTestError("the sealed Manager test split requires --allow-sealed-test")
        if CONSUMED_MARKER.exists() and not allow_consumed_test:
            raise SealedTestConsumedError(f"sealed test already consumed ({CONSUMED_MARKER}); re-evaluation refused")
        if sha256_file(SPLIT_FILES["test"]) != SEALED_TEST_SHA256:
            raise SealedTestError("sealed test SHA-256 differs from the recorded hash")
    if split not in ("train", "validation") and not (split == "test" and allow_sealed_test):
        raise ValueError(split)
    rows = [json.loads(line) for line in SPLIT_FILES[split].read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != EXPECTED_ROWS[split]:
        raise ValueError(f"{split}: expected {EXPECTED_ROWS[split]} rows, got {len(rows)}")
    for row in rows:
        if split != "test" and row["metadata"].get("sealed_test"):
            raise SealedTestError(f"{row['id']} is a sealed test row inside {split}")
    return rows


def system_prompt() -> str:
    return load_contract().system_prompt


def render_prompt(request: str, prompt: str | None = None) -> str:
    prompt = system_prompt() if prompt is None else prompt
    return (f"<|im_start|>system\n{prompt}<|im_end|>\n<|im_start|>user\n{request}<|im_end|>\n"
            f"<|im_start|>assistant\n")


def assert_template_parity(tokenizer, request: str = "테스트 요청") -> None:
    ours = render_prompt(request)
    theirs = tokenizer.apply_chat_template([{"role": "system", "content": system_prompt()},
                                            {"role": "user", "content": request}],
                                           tokenize=False, add_generation_prompt=True)
    if ours != theirs:
        raise AssertionError("Runtime ChatML rendering differs from the tokenizer chat template")


def target_text(row: dict) -> str:
    """The SFT target: canonical TaskPlan v2 bytes, verified against the Runtime serializer."""
    text = row["plan_text"]
    result = validate(text, mode="canonical")
    if not result.valid or canonical_json(result.plan) != text:
        raise ValueError(f"{row['id']}: plan_text is not canonical TaskPlan v2 ({result.code})")
    return text


@dataclass
class Encoded:
    row_id: str
    input_ids: list[int]
    labels: list[int]
    prompt_tokens: int
    target_tokens: int  # includes EOS


def encode(row: dict, tokenizer, prompt: str) -> Encoded:
    prompt_ids = tokenizer(render_prompt(row["request"], prompt), add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target_text(row), add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    return Encoded(row["id"], prompt_ids + target_ids, [IGNORE_INDEX] * len(prompt_ids) + target_ids,
                   len(prompt_ids), len(target_ids))


def encode_all(rows: list[dict], tokenizer, max_length: int) -> list[Encoded]:
    prompt = system_prompt()
    encoded = [encode(row, tokenizer, prompt) for row in rows]
    too_long = [e.row_id for e in encoded if len(e.input_ids) > max_length]
    if too_long:
        raise ValueError(f"{len(too_long)} samples exceed max_length={max_length}; refusing to truncate: {too_long[:5]}")
    return encoded
