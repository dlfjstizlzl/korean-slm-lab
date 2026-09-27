#!/usr/bin/env python3
"""Canonical base-model generation / repetition evaluation (single source of truth).

`repetition_metrics` used to live in ``stage1_extension_report.py``; it moved here
unchanged (every key it produced keeps its exact definition) and gained the
extra metrics the Stage-2 closure needs. ``stage1_extension_report`` re-imports it.

Nothing here trains anything. It measures whether a *base* model's continuations
loop, collapse, or break structure. It is not an instruction-tuning benchmark.

Definitions
-----------
Token-level (tokenizer-dependent — compare only models that share a tokenizer):
  duplicate_{4,8}gram_fraction  1 - distinct/total over contiguous token-id n-grams
  distinct_ratio (= distinct_1) distinct token ids / generated tokens
  distinct_2                    distinct token bigrams / total bigrams
  longest_single_token_run      longest run of one repeated token id
  longest_repeated_span_tokens  longest token substring occurring at least twice
                                (occurrences may overlap)
Text-level (tokenizer-independent — comparable across Qwen and KO-32K):
  word_duplicate_{4,8}gram      corpus/stage2_dedup.ngram_duplicate_ratio on the
                                generated text (whitespace words), the same
                                function the Stage-2 corpus metrics use
  loop_detected                 the generated text ends in >= LOOP_MIN_COPIES
                                consecutive copies of a unit of 2..LOOP_MAX_PERIOD
                                characters (non-whitespace content), covering at
                                least LOOP_MIN_COVERAGE of the generated characters
  loop_period_chars / loop_copies / loop_coverage
  replacement_chars             U+FFFD anywhere in the decoded text (legacy key)
  interior_replacement_chars    U+FFFD excluding a trailing run (a byte sequence
                                cut by max_new_tokens is not a model defect)
  unicode_ok                    no interior U+FFFD, no lone surrogate, no C0 control
                                other than \\t \\n \\r
Termination:
  stopped_by_eos                generation ended on a stop token
  premature_eos                 stopped_by_eos and fewer than PREMATURE_EOS_TOKENS new tokens
  empty                         generated text is empty after strip()
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
import unicodedata
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "corpus"))

from stage2_dedup import ngram_duplicate_ratio  # noqa: E402

PROMPTS_FILE = "data/eval/generation_prompts_v1.jsonl"

# Fixed before any Stage-2 generation was produced.
LOOP_MAX_PERIOD = 200
LOOP_MIN_COPIES = 3
LOOP_MIN_COVERAGE = 0.25
PREMATURE_EOS_TOKENS = 8
STOP_TOKEN_STRINGS = ("<|im_end|>", "<|endoftext|>")

GREEDY = {"do_sample": False}
# Qwen3-0.6B's own shipped sampling defaults, applied identically to every model.
SAMPLING = {"do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20}
SAMPLING_SEEDS = (0, 1, 2)
MAX_NEW_TOKENS = 160

CONTROL_TOKENS = ("<검색>", "<요약>", "<SQL>", "<코드>", "<이전0>", "<이전1>", "<이전2>", "<종료>")
SKILL_NAMES = ("search", "summary", "sql", "code")


# --------------------------------------------------------------------------- #
# Repetition metrics
# --------------------------------------------------------------------------- #

def _duplicate_ngram_fraction(sequence: list, size: int) -> float:
    if len(sequence) < size:
        return 0.0
    grams = [tuple(sequence[i:i + size]) for i in range(len(sequence) - size + 1)]
    return 1.0 - (len(set(grams)) / len(grams))


def longest_repeated_span(sequence: list) -> int:
    """Length of the longest contiguous subsequence that occurs at least twice."""
    n = len(sequence)
    if n < 2:
        return 0

    def repeated(length: int) -> bool:
        seen = set()
        for i in range(n - length + 1):
            key = tuple(sequence[i:i + length])
            if key in seen:
                return True
            seen.add(key)
        return False

    low, high = 0, n - 1
    while low < high:
        middle = (low + high + 1) // 2
        if repeated(middle):
            low = middle
        else:
            high = middle - 1
    return low


def tail_loop(text: str, max_period: int = LOOP_MAX_PERIOD) -> dict:
    """Smallest period whose repetition covers the longest tail of ``text``."""
    best = {"period": 0, "copies": 0, "span": 0}
    n = len(text)
    for period in range(2, min(max_period, n // 2) + 1):
        unit = text[n - period:]
        if not unit.strip():
            continue
        copies = 1
        while n - (copies + 1) * period >= 0 and text[n - (copies + 1) * period:n - copies * period] == unit:
            copies += 1
        span = copies * period
        if copies >= 2 and span > best["span"]:
            best = {"period": period, "copies": copies, "span": span}
    return best


def _unicode_ok(text: str) -> tuple[bool, int]:
    stripped = text.rstrip("�")
    interior = stripped.count("�")
    bad_control = any(unicodedata.category(c) == "Cc" and c not in "\t\n\r" for c in text)
    surrogate = any(0xD800 <= ord(c) <= 0xDFFF for c in text)
    return (interior == 0 and not bad_control and not surrogate), interior


def repetition_metrics(token_ids: list[int], text: str) -> dict:
    """Quantify looping. Phrase-level repetition needs n-grams, not token runs.

    The first seven keys are the historical Stage-1 definitions, unchanged.
    """
    longest_run = 1  # historical definition, kept even for an empty sequence
    run = 1
    for previous, current in zip(token_ids, token_ids[1:]):
        run = run + 1 if current == previous else 1
        longest_run = max(longest_run, run)

    loop = tail_loop(text)
    coverage = loop["span"] / len(text) if text else 0.0
    unicode_ok, interior = _unicode_ok(text)
    bigrams = list(zip(token_ids, token_ids[1:]))
    return {
        "new_tokens": len(token_ids),
        "distinct_tokens": len(set(token_ids)),
        "distinct_ratio": len(set(token_ids)) / len(token_ids) if token_ids else 0.0,
        "longest_single_token_run": longest_run,
        "duplicate_4gram_fraction": _duplicate_ngram_fraction(token_ids, 4),
        "duplicate_8gram_fraction": _duplicate_ngram_fraction(token_ids, 8),
        "replacement_chars": text.count("�"),
        # --- added for the Stage-2 closure ---
        "distinct_1": len(set(token_ids)) / len(token_ids) if token_ids else 0.0,
        "distinct_2": len(set(bigrams)) / len(bigrams) if bigrams else 0.0,
        "longest_repeated_span_tokens": longest_repeated_span(token_ids),
        "generated_characters": len(text),
        "word_duplicate_4gram": ngram_duplicate_ratio(text, 4),
        "word_duplicate_8gram": ngram_duplicate_ratio(text, 8),
        "loop_period_chars": loop["period"],
        "loop_copies": loop["copies"],
        "loop_coverage": coverage,
        "loop_detected": loop["copies"] >= LOOP_MIN_COPIES and coverage >= LOOP_MIN_COVERAGE,
        "interior_replacement_chars": interior,
        "unicode_ok": unicode_ok,
        "empty": not text.strip(),
    }


# --------------------------------------------------------------------------- #
# Stage-1 failure signatures (reported separately, never folded into a score)
# --------------------------------------------------------------------------- #

FAILURE_SIGNATURES = {
    "repeated_10000": re.compile(r"(?:10000\W*){3,}"),
    "repeated_input_sql": re.compile(r'(?:"input"\s*:\s*"sql"[^{}]*?){3,}'),
}


def failure_signatures(prompt: str, generated: str, metrics: dict) -> dict:
    out = {name: bool(pattern.search(generated)) for name, pattern in FAILURE_SIGNATURES.items()}
    out["phrase_loop"] = bool(metrics["loop_detected"])
    core = prompt.strip()
    out["prompt_echo_loop"] = len(core) >= 8 and generated.count(core) >= 2
    return out


# --------------------------------------------------------------------------- #
# Structured validity
# --------------------------------------------------------------------------- #

def json_validity(prompt: str, generated: str, schema: dict | None) -> dict:
    """Parse the first complete JSON value of prompt + continuation."""
    text = prompt + generated
    try:
        value, end = json.JSONDecoder().raw_decode(text.lstrip())
    except ValueError:
        return {"json_parse_valid": False, "json_schema_valid": False}
    schema_ok = True
    if schema:
        expected = {"object": dict, "array": list}.get(schema.get("type", ""), object)
        schema_ok = isinstance(value, expected)
        if schema_ok and isinstance(value, dict):
            schema_ok = all(key in value for key in schema.get("required", []))
            for key, allowed in (schema.get("enum") or {}).items():
                schema_ok = schema_ok and value.get(key) in allowed
    return {"json_parse_valid": True, "json_schema_valid": bool(schema_ok)}


SQL_SYNTAX_ERRORS = ("syntax error", "incomplete input", "unrecognized token")


def first_sql_statement(prompt: str, generated: str) -> str:
    """Prompt + continuation up to the first ';' or blank line, comments dropped."""
    text = prompt + generated
    lines = [line for line in text.split("\n") if not line.strip().startswith("--")]
    body = "\n".join(lines)
    body = re.split(r";|\n\s*\n", body, maxsplit=1)[0]
    return body.strip()


def sql_validity(prompt: str, generated: str) -> dict:
    """Syntax-only check with SQLite's parser against an empty database.

    Name-resolution errors (no such table/column) prove the statement parsed, so
    they count as syntactically valid; parse errors do not.
    """
    statement = first_sql_statement(prompt, generated)
    if not statement:
        return {"sql_syntax_valid": False, "sql_statement": statement}
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("EXPLAIN " + statement)
        valid = True
    except sqlite3.Error as error:
        message = str(error).lower()
        valid = not any(marker in message for marker in SQL_SYNTAX_ERRORS)
    except Exception:  # noqa: BLE001 - e.g. multiple statements
        valid = False
    finally:
        connection.close()
    return {"sql_syntax_valid": valid, "sql_statement": statement}


# Skill DSL grammar, derived from the Stage-2 v3 generator
# (corpus/stage2_skill_synth_v2.py: render / failure_case) plus the original
# data/eval_skill_dsl.txt forms. v1 of this validator (2026-09-24, first run)
# accepted only one control token per line and rejected closing lines, inline
# chains, numbered/labelled steps and multi-line JSON; see
# outputs/final_validation/generation_summary_dsl_validator_v1.json.
SKILL_TOKEN = r"(?:<검색>|<요약>|<SQL>|<코드>)"
PREV_TOKEN = r"<이전[0-2]>"
ANY_CONTROL = r"(?:<검색>|<요약>|<SQL>|<코드>|<이전[0-2]>|<종료>)"
BODY = rf"(?:(?!{ANY_CONTROL})[^\n])*"  # free text; '<'/'>' allowed (SQL), control tokens not
DSL_LINE = re.compile(
    rf"^(?:{PREV_TOKEN})*(?:{SKILL_TOKEN}(?:{PREV_TOKEN})*{BODY})+(?:<종료>)?$"     # <검색>q / inline chain
    rf"|^(?:{PREV_TOKEN})+$"                                                     # context header
    r"|^<종료>$"
    rf"|^(?!{ANY_CONTROL}){BODY}\S\s<종료>$"                                      # closing sentence
    rf"|^\d+\.\s*{SKILL_TOKEN}{BODY}$"                                            # numbered step
    r"|^(?:목표|요청|질의|작업|사용자 요청|처리 대상|사용자|계획|결과|응답|오류|재시도|확인)\s*:.*$"
)
JSON_PLAN_KEYS = ("plan", "goal", "attempts", "attempt", "context")


def _dsl_json_valid(value) -> bool:
    if isinstance(value, dict):
        return value.get("skill") in SKILL_NAMES or any(key in value for key in JSON_PLAN_KEYS)
    return False


def dsl_line_valid(line: str) -> bool:
    line = line.strip()
    if DSL_LINE.match(line):
        return True
    if line.startswith("{") or line.startswith("["):
        try:
            return _dsl_json_valid(json.loads(line))
        except ValueError:
            return False
    return False


def skill_dsl_validity(prompt: str, generated: str, complete: bool = False) -> dict:
    """Grammar check of the DSL continuation, one unit at a time.

    A unit is a line, or a whole (possibly multi-line) JSON value starting at a
    line. Only units that contain generated text are judged: lines the prompt
    already finished are skipped (they would be trivially valid), the prompt's
    unfinished last line counts once the generation extends it, and a final unit
    cut off by the token budget (``complete`` False) is dropped.
    """
    text = prompt + generated
    lines = text.split("\n")
    lines = lines[prompt.count("\n"):]
    partial = prompt.split("\n")[-1]
    if lines and partial and lines[0].strip() == partial.strip():
        lines = lines[1:]  # the generation added nothing to the prompt's own line
    cut = not complete and not text.endswith("\n")
    units: list[bool] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        last = index == len(lines) - 1
        if not line.strip():
            index += 1
            continue
        stripped = line.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            rest = "\n".join(lines[index:]).strip()
            try:
                value, end = json.JSONDecoder().raw_decode(rest)
            except ValueError:
                if cut:
                    break  # an unfinished JSON value at the budget edge is not judged
                units.append(False)
                index += 1
                continue
            consumed = rest[:end].count("\n")
            trailing = rest[end:].split("\n", 1)[0].strip()
            units.append(_dsl_json_valid(value) and not trailing)
            index += consumed + 1
            continue
        if last and cut:
            break
        units.append(dsl_line_valid(line))
        index += 1
    return {
        "dsl_lines": len(units),
        "dsl_valid_lines": sum(units),
        "dsl_first_line_valid": bool(units[0]) if units else False,
        "dsl_all_lines_valid": bool(units) and all(units),
    }


def instruction_check(generated: str, expect_regex: str | None) -> dict:
    if not expect_regex:
        return {}
    first = generated.strip().split("\n")[0] if generated.strip() else ""
    return {"instruction_match_first_line": bool(re.search(expect_regex, first))}


def structured_checks(record: dict, generated: str, complete: bool = False) -> dict:
    category = record["category"]
    prompt = record["prompt"]
    if category == "json_structured" or record["id"] == "legacy_json":
        return json_validity(prompt, generated, record.get("schema"))
    if category == "sql":
        return sql_validity(prompt, generated)
    if category == "skill_dsl" or record["id"] == "legacy_skill":
        return skill_dsl_validity(prompt, generated, complete)
    if category == "instruction":
        return instruction_check(generated, record.get("expect_regex"))
    return {}


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #

def load_prompts(path: str | Path = PROMPTS_FILE) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stop_token_ids(tokenizer) -> list[int]:
    """The same stop *strings* for every model, resolved to each tokenizer's ids."""
    ids = []
    for token in STOP_TOKEN_STRINGS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == tokenizer.unk_token_id:
            raise ValueError(f"stop token {token!r} missing from tokenizer")
        ids.append(int(token_id))
    return ids


@torch.inference_mode()
def generate_one(model, tokenizer, record: dict, device: str, decoding: dict,
                 seed: int | None, max_new_tokens: int = MAX_NEW_TOKENS) -> dict:
    encoded = tokenizer(record["prompt"], return_tensors="pt").to(device)
    stops = stop_token_ids(tokenizer)
    if seed is not None:
        torch.manual_seed(seed)
    output = model.generate(
        **encoded, max_new_tokens=max_new_tokens, eos_token_id=stops,
        pad_token_id=tokenizer.pad_token_id, use_cache=True, **decoding,
    )
    new_ids = output[0].tolist()[int(encoded["input_ids"].shape[1]):]
    stopped = bool(new_ids) and new_ids[-1] in stops
    content_ids = new_ids[:-1] if stopped else new_ids
    generated = tokenizer.decode(content_ids, skip_special_tokens=False)
    metrics = repetition_metrics(content_ids, generated)
    metrics["stopped_by_eos"] = stopped
    metrics["premature_eos"] = stopped and len(content_ids) < PREMATURE_EOS_TOKENS
    return {
        "id": record["id"],
        "category": record["category"],
        "prompt": record["prompt"],
        "seed": seed,
        "generated": generated,
        "metrics": metrics,
        "structured": structured_checks(record, generated, stopped),
        "failure_signatures": failure_signatures(record["prompt"], generated, metrics),
    }


def run_model(model, tokenizer, prompts: list[dict], device: str,
              seeds: tuple[int, ...] = SAMPLING_SEEDS,
              max_new_tokens: int = MAX_NEW_TOKENS) -> dict:
    greedy = [generate_one(model, tokenizer, p, device, GREEDY, None, max_new_tokens)
              for p in prompts]
    sampled = [generate_one(model, tokenizer, p, device, SAMPLING, seed, max_new_tokens)
               for seed in seeds for p in prompts]
    return {"greedy": greedy, "sampled": sampled}


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

MEAN_KEYS = (
    "new_tokens", "generated_characters", "duplicate_4gram_fraction", "duplicate_8gram_fraction",
    "distinct_1", "distinct_2", "longest_repeated_span_tokens", "word_duplicate_4gram",
    "word_duplicate_8gram", "loop_coverage",
)
RATE_KEYS = ("loop_detected", "premature_eos", "empty", "stopped_by_eos")


def aggregate(samples: list[dict]) -> dict:
    if not samples:
        return {"samples": 0}
    out: dict = {"samples": len(samples)}
    for key in MEAN_KEYS:
        out[f"mean_{key}"] = sum(s["metrics"][key] for s in samples) / len(samples)
    for key in RATE_KEYS:
        out[f"{key}_rate"] = sum(bool(s["metrics"][key]) for s in samples) / len(samples)
    out["unicode_ok_rate"] = sum(s["metrics"]["unicode_ok"] for s in samples) / len(samples)
    out["max_longest_repeated_span_tokens"] = max(
        s["metrics"]["longest_repeated_span_tokens"] for s in samples)
    structured: dict[str, list] = {}
    for sample in samples:
        for key, value in sample["structured"].items():
            if isinstance(value, bool):
                structured.setdefault(key, []).append(value)
    for key, values in structured.items():
        out[f"{key}_rate"] = sum(values) / len(values)
    signatures: dict[str, int] = {}
    for sample in samples:
        for key, value in sample["failure_signatures"].items():
            signatures[key] = signatures.get(key, 0) + int(value)
    out["failure_signature_counts"] = signatures
    return out


def summarize(run: dict) -> dict:
    summary = {}
    for mode in ("greedy", "sampled"):
        samples = run[mode]
        categories = sorted({s["category"] for s in samples})
        summary[mode] = {
            "overall": aggregate(samples),
            "by_category": {c: aggregate([s for s in samples if s["category"] == c])
                            for c in categories},
        }
    return summary
