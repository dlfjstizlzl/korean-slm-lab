#!/usr/bin/env python3
"""Statistics, leakage checks, pilot gate and review sheet for the Manager pilot.

Reads data/manager/pilot_v1.jsonl (+ its generation log) and writes

  outputs/manager_dataset/pilot_stats.json
  outputs/manager_dataset/pilot_report.md
  outputs/manager_dataset/pilot_review.md

Gate thresholds are declared in GATE_THRESHOLDS *before* anything is measured
and are printed verbatim in the report. Distribution thresholds are proposals
for a human to confirm, so exceeding them yields REVIEW, never a silent FAIL.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "model"))
sys.path.insert(0, str(PROJECT_ROOT / "corpus"))

from manager_plan_validator import (  # noqa: E402
    CYCLIC_DEPENDENCY,
    FUTURE_REFERENCE,
    INVALID_INPUT_FROM,
    INVALID_SKILL,
    PARITY_CASES,
    SELF_REFERENCE,
    load_registry,
    runtime_oracle,
    validate_plan_text,
)
from stage2_dedup import hamming, ngram_duplicate_ratio, simhash  # noqa: E402

DEFAULT_INPUT = PROJECT_ROOT / "data" / "manager" / "pilot_v1.jsonl"
OUT_DIR = PROJECT_ROOT / "outputs" / "manager_dataset"
QWEN_TOKENIZER = "Qwen/Qwen3-0.6B"
KO32K_TOKENIZER = PROJECT_ROOT / "outputs" / "models" / "qwen3-0.6b-ko32k-stage2-signal2678"
MANAGER_MAX_TOKENS = 256  # slm-runtime main.cpp default --manager-max-tokens
SPLITS = ("train", "validation", "test")
DEPENDENCY_CODES = {INVALID_INPUT_FROM, SELF_REFERENCE, FUTURE_REFERENCE, CYCLIC_DEPENDENCY}
REVIEW_SEED = 0

GATE_THRESHOLDS = {
    "F_max_taxonomy_share": 0.25,
    "G_max_family_share_within_taxonomy": 0.35,
    "G_max_instruction_skeleton_share": 0.05,
    "E_near_duplicate_hamming": 3,
    "L_plan_token_hard_limit": MANAGER_MAX_TOKENS,
    "L_plan_token_review_above": 200,
}

REQUIRED_METADATA = (
    "sample_id", "status", "source_type", "taxonomy", "taxonomy_group", "template_family",
    "semantic_id", "entity_family", "skills_used", "task_count", "plan_depth", "split",
    "paraphrased", "requires_human_policy", "training_candidate", "input_policy",
    "generator_version", "seed", "validation",
)

HUMAN_DECISIONS = (
    ("HD-01", "blocking", "Input span policy: what goes into `input`. Runtime few-shots disagree "
     "(whole request / instruction clause / topic only). Pilot uses a verbatim minimal content span."),
    ("HD-02", "blocking", "Request language vs runtime prompt language. Few-shots and system prompt are "
     "English; the pilot is Korean and copies Korean spans into `input` without translation."),
    ("HD-03", "blocking", "Unsupported-capability handling. The Runtime has no reject path (grammar and "
     "validator require >= 1 task). Candidates: reject before planning, nearest-skill fallback."),
    ("HD-04", "blocking", "Ambiguous / missing-payload handling: clarification outside the plan, "
     "best-guess plan, or failure."),
    ("HD-05", "blocking", "Independent multi-task requests: runtime-valid, but only the last task's "
     "output reaches the user (`[FINAL]`)."),
    ("HD-06", "non-blocking", "Cross-request references: ManagerClient::plan is stateless; no session "
     "memory exists."),
    ("HD-07", "blocking", "Topic-only tasks (summary/search with no supplied text) make the skill model "
     "supply content. Few-shots 1 and 4 do this; keep as supported?"),
    ("HD-08", "blocking", "`search` means keyword/concept extraction, not retrieval. Requests for live "
     "or web search are treated as unsupported in the pilot."),
    ("HD-09", "blocking", "Pasted text must be copied verbatim into the plan within the 256-token "
     "Manager output cap. Pilot caps payloads at 150 chars; real limit and long-input policy undecided."),
    ("HD-10", "non-blocking", "`sql` and `code` exist in the dummy benchmark, README and tokenizer DSL but "
     "not in the Manager path. Excluded from the pilot."),
    ("HD-11", "non-blocking", "The tokenizer-side Skill DSL (<검색> <요약> <이전0> ...) differs from the "
     "Runtime JSON. Pilot keeps the Runtime JSON; DSL only as a future comparison."),
    ("HD-12", "non-blocking", "Taxonomy target ratios and hard-case share (pilot: 18%) need approval."),
    ("HD-13", "non-blocking", "Multi-line requests: interactive CLI reads one line (std::getline); "
     "--prompt accepts newlines. Keep newline families?"),
    ("HD-14", "non-blocking", "SFT prompt format (runtime system prompt + chat template or bare request) and "
     "base model (Qwen3-0.6B GGUF in runtime vs KO32K Stage-2)."),
    ("HD-15", "non-blocking", "Plans deeper than 2 have no natural request with only search/summary; "
     "excluded from the taxonomy."),
    ("HD-16", "non-blocking", "Canonical target serialization = TaskPlan::to_json() compact form; the "
     "grammar also allows optional whitespace."),
)

SUMMARY_WORDS = re.compile(r"요약|간추|줄여|줄이|요지|한 줄|한 문장")
EXTRACT_WORDS = re.compile(r"키워드|개념|용어|핵심어|추출|뽑|골라|고르|나열|목록")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def skeleton(row: dict) -> str:
    text = row["request"]
    arguments = row["metadata"].get("arguments") or {}
    for key in ("text2", "text", "topic"):
        if arguments.get(key):
            text = text.replace(arguments[key], "<" + key.upper() + ">")
    return text


def seq_label(row: dict) -> str | None:
    return ">".join(row["metadata"]["skills_used"]) if row["plan"] else None


def describe(values: list[float]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    q = statistics.quantiles(ordered, n=4) if len(ordered) > 1 else [ordered[0]] * 3
    return {"count": len(values), "min": ordered[0], "p25": q[0], "median": q[1], "p75": q[2],
            "max": ordered[-1], "mean": round(statistics.fmean(values), 2)}


def histogram(values: list[int], edges: list[int]) -> dict:
    result = {}
    for low, high in zip(edges, edges[1:] + [None]):
        label = f"{low}-{high - 1}" if high else f"{low}+"
        result[label] = sum(1 for v in values if v >= low and (high is None or v < high))
    return result


def words(text: str) -> list[str]:
    return text.split()


def cross_request_ngram_share(texts: list[str], n: int) -> float:
    """Share of n-gram occurrences whose n-gram appears in >= 2 distinct requests."""
    owners: dict[tuple, set] = defaultdict(set)
    occurrences = []
    for index, text in enumerate(texts):
        tokens = words(text)
        grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        occurrences.append(grams)
        for gram in grams:
            owners[gram].add(index)
    total = sum(len(g) for g in occurrences)
    shared = sum(1 for grams in occurrences for gram in grams if len(owners[gram]) > 1)
    return round(shared / total, 4) if total else 0.0


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(set(ys)) < 2 or len(set(xs)) < 2:
        return None
    return round(statistics.correlation(xs, ys), 4)


def keyword_rule(text: str) -> str:
    s, e = SUMMARY_WORDS.search(text), EXTRACT_WORDS.search(text)
    if s and e:
        return "search>summary" if e.start() < s.start() else "summary>search"
    if e:
        return "search"
    return "summary"


def naive_bayes(train: list[tuple[list[str], str]], test: list[tuple[list[str], str]]) -> dict:
    labels = Counter(label for _, label in train)
    counts: dict[str, Counter] = defaultdict(Counter)
    vocab = set()
    for tokens, label in train:
        counts[label].update(tokens)
        vocab.update(tokens)
    totals = {label: sum(counts[label].values()) for label in labels}
    correct = 0
    for tokens, label in test:
        best = max(labels, key=lambda c: math.log(labels[c] / len(train)) + sum(
            math.log((counts[c][t] + 1) / (totals[c] + len(vocab))) for t in tokens))
        correct += best == label
    return {"train": len(train), "evaluated": len(test),
            "accuracy": round(correct / len(test), 4) if test else None}


def token_lengths(texts: list[str]) -> dict:
    result = {}
    try:
        import os

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from transformers import AutoTokenizer

        for name, source in (("qwen3_0.6b", QWEN_TOKENIZER), ("ko32k_stage2", str(KO32K_TOKENIZER))):
            try:
                tokenizer = AutoTokenizer.from_pretrained(source)
            except Exception as error:  # noqa: BLE001
                result[name] = {"error": str(error)[:200]}
                continue
            result[name] = [len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in texts]
    except Exception as error:  # noqa: BLE001
        result["error"] = str(error)[:200]
    return result


def leakage(rows: list[dict]) -> dict:
    """Group-leakage checks across train/validation/test."""
    split_of = {r["id"]: r["metadata"]["split"] for r in rows}

    def spread(key_fn):
        owners: dict[str, set] = defaultdict(set)
        for r in rows:
            if r["metadata"]["split"] in SPLITS:
                for key in key_fn(r):
                    owners[key].add(r["metadata"]["split"])
        return sorted(k for k, v in owners.items() if len(v) > 1)

    hashes = [simhash(r["request"]) for r in rows]
    near_pairs, cross_near = [], []
    for i, j in combinations(range(len(rows)), 2):
        if hamming(hashes[i], hashes[j]) <= GATE_THRESHOLDS["E_near_duplicate_hamming"]:
            pair = (rows[i]["id"], rows[j]["id"])
            near_pairs.append(pair)
            if rows[i]["metadata"]["split"] != rows[j]["metadata"]["split"]:
                cross_near.append(pair)
    return {
        "template_family_across_splits": spread(lambda r: [r["metadata"]["template_family"]]),
        "entity_family_across_splits": spread(lambda r: r["metadata"]["entity_family"]),
        "semantic_seed_across_splits": spread(lambda r: [r["metadata"]["semantic_id"]]),
        "paraphrase_sibling_across_splits": [
            r["id"] for r in rows if r["metadata"]["paraphrase_of"]
            and split_of.get(r["metadata"]["paraphrase_of"]) != r["metadata"]["split"]],
        "policy_rows_inside_training_splits": [
            r["id"] for r in rows if r["metadata"]["requires_human_policy"] and r["metadata"]["split"] in SPLITS],
        "near_duplicate_pairs": near_pairs,
        "near_duplicate_pairs_cross_split": cross_near,
    }


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #

def analyse(rows: list[dict], log: dict, run_pytest: bool = True) -> dict:
    registry = load_registry()
    labeled = [r for r in rows if r["plan"]]
    policy = [r for r in rows if r["metadata"]["requires_human_policy"]]
    meta = [r["metadata"] for r in rows]
    skeletons = [skeleton(r) for r in rows]
    stats: dict = {}

    # dataset ------------------------------------------------------------
    quotas = log["quotas"]
    taxonomy_counts = Counter(m["taxonomy"] for m in meta)
    stats["dataset"] = {
        "total_samples": len(rows),
        "labeled_training_candidates": len(labeled),
        "policy_pending_unlabeled": len(policy),
        "generation_attempts": log["attempts"],
        "rejected_attempts": log["rejected_attempts"],
        "reject_reasons": log["reject_reasons"],
        "entity_pool": log["entity_pool"],
        "source_type": dict(Counter(m["source_type"] for m in meta)),
        "status": dict(Counter(m["status"] for m in meta)),
        "taxonomy_distribution": {name: {"count": taxonomy_counts[name], "quota": quotas[name],
                                         "share": round(taxonomy_counts[name] / len(rows), 4)}
                                  for name in quotas},
        "taxonomy_group": dict(Counter(m["taxonomy_group"] for m in meta)),
        "taxonomy_groups": {m["taxonomy"]: {"group": m["taxonomy_group"]} for m in meta},
        "skill_task_frequency": dict(Counter(s for m in meta for s in m["skills_used"])),
        "skill_sequence": dict(Counter(seq_label(r) for r in labeled)),
        "task_count": dict(sorted(Counter(m["task_count"] for m in meta if m["task_count"]).items())),
        "plan_depth": dict(sorted(Counter(m["plan_depth"] for m in meta if m["task_count"]).items())),
        "register": dict(Counter(m["register"] for m in meta)),
        "source_kind": dict(Counter(m["source_kind"] for m in meta)),
        "request_chars": describe([len(r["request"]) for r in rows]),
        "request_chars_histogram": histogram([len(r["request"]) for r in rows],
                                             [0, 40, 80, 120, 160, 200, 250, 300]),
        "instruction_chars": describe([len(s) for s in skeletons]),
        "request_chars_by_taxonomy": {name: describe([len(r["request"]) for r in rows
                                                      if r["metadata"]["taxonomy"] == name]) for name in quotas},
        "multiline_requests": sum("\n" in r["request"] for r in rows),
    }

    # diversity ----------------------------------------------------------
    requests = [r["request"] for r in rows]
    from manager_dataset_generator import normalize_request

    normalized = [normalize_request(t) for t in requests]
    family_counts = Counter(m["template_family"] for m in meta)
    skeleton_counts = Counter(skeletons)
    family_share_in_taxonomy = {}
    for name in quotas:
        members = [m["template_family"] for m in meta if m["taxonomy"] == name]
        if members:
            top, count = Counter(members).most_common(1)[0]
            family_share_in_taxonomy[name] = {"family": top, "share": round(count / len(members), 4),
                                              "families_used": len(set(members))}
    skeleton_tokens = [t for s in skeletons for t in words(s)]
    vocab_by_taxonomy = {name: {t for r, s in zip(rows, skeletons) if r["metadata"]["taxonomy"] == name
                                for t in words(s)} for name in quotas}
    overlap = {}
    for a, b in combinations([n for n in quotas if vocab_by_taxonomy[n]], 2):
        union = vocab_by_taxonomy[a] | vocab_by_taxonomy[b]
        overlap[f"{a}|{b}"] = round(len(vocab_by_taxonomy[a] & vocab_by_taxonomy[b]) / len(union), 4)
    bigrams = [tuple(words(s)[i:i + 2]) for s in skeletons for i in range(len(words(s)) - 1)]
    stats["diversity"] = {
        "exact_duplicate_requests": len(requests) - len(set(requests)),
        "normalized_duplicate_requests": len(normalized) - len(set(normalized)),
        "unique_request_ratio": round(len(set(requests)) / len(requests), 4),
        "unique_normalized_request_ratio": round(len(set(normalized)) / len(requests), 4),
        "plan_text_exact_duplicate_ratio": round(
            1 - len({r["plan_text"] for r in labeled}) / len(labeled), 4) if labeled else None,
        "within_request_dup_4gram_mean": round(statistics.fmean(ngram_duplicate_ratio(t, 4) for t in requests), 4),
        "within_request_dup_8gram_mean": round(statistics.fmean(ngram_duplicate_ratio(t, 8) for t in requests), 4),
        "cross_request_shared_4gram_share": cross_request_ngram_share(requests, 4),
        "cross_request_shared_8gram_share": cross_request_ngram_share(requests, 8),
        "instruction_cross_request_shared_4gram_share": cross_request_ngram_share(skeletons, 4),
        "instruction_cross_request_shared_8gram_share": cross_request_ngram_share(skeletons, 8),
        "unique_instruction_skeleton_ratio": round(len(skeleton_counts) / len(rows), 4),
        "top_instruction_skeletons": [{"skeleton": s, "count": c} for s, c in skeleton_counts.most_common(5)],
        "template_family_count": len(family_counts),
        "template_family_distribution": dict(family_counts.most_common()),
        "max_family_share_overall": round(family_counts.most_common(1)[0][1] / len(rows), 4),
        "max_family_share_within_taxonomy": family_share_in_taxonomy,
        "instruction_type_token_ratio": round(len(set(skeleton_tokens)) / len(skeleton_tokens), 4),
        "instruction_distinct_1": len(set(skeleton_tokens)),
        "instruction_distinct_2_ratio": round(len(set(bigrams)) / len(bigrams), 4) if bigrams else None,
        "full_request_type_token_ratio": round(len({t for x in requests for t in words(x)})
                                               / sum(len(words(x)) for x in requests), 4),
        "taxonomy_instruction_vocab_jaccard": overlap,
    }

    # shortcut diagnostics ----------------------------------------------
    by_group = defaultdict(lambda: [0, 0])
    for r, s in zip(rows, skeletons):
        if r["plan"]:
            group = r["metadata"]["taxonomy_group"]
            by_group[group][0] += keyword_rule(s) == seq_label(r)
            by_group[group][1] += 1
    tokenized = [(words(s), seq_label(r), r["metadata"]["split"]) for r, s in zip(rows, skeletons) if r["plan"]]
    train_items = [(t, y) for t, y, sp in tokenized if sp == "train"]
    heldout_items = [(t, y) for t, y, sp in tokenized if sp in ("validation", "test")]
    heldout_hard = [(t, y) for (t, y, sp), r in zip(tokenized, labeled)
                    if sp in ("validation", "test") and r["metadata"]["taxonomy_group"] == "hard_labeled"]
    two_task = [len(s) for r, s in zip(rows, skeletons) if r["plan"]]
    stats["shortcut_diagnostics"] = {
        "keyword_rule_accuracy": {g: {"correct": c, "total": n, "accuracy": round(c / n, 4)}
                                  for g, (c, n) in by_group.items()},
        "bow_naive_bayes_train_to_heldout": naive_bayes(train_items, heldout_items),
        "bow_naive_bayes_train_to_heldout_hard": naive_bayes(train_items, heldout_hard),
        "corr_instruction_length_vs_task_count": pearson(two_task, [len(r["plan"]["tasks"]) for r in labeled]),
        "corr_request_length_vs_task_count": pearson([len(r["request"]) for r in labeled],
                                                     [len(r["plan"]["tasks"]) for r in labeled]),
        "mean_instruction_chars_by_task_count": {
            k: round(statistics.fmean(len(s) for r, s in zip(rows, skeletons)
                                      if r["plan"] and len(r["plan"]["tasks"]) == k), 1)
            for k in sorted({len(r["plan"]["tasks"]) for r in labeled})},
    }

    # split + leakage ----------------------------------------------------
    requests_leak = leakage(rows)
    family_leak = requests_leak["template_family_across_splits"]
    entity_leak = requests_leak["entity_family_across_splits"]
    semantic_leak = requests_leak["semantic_seed_across_splits"]
    sibling_leak = requests_leak["paraphrase_sibling_across_splits"]
    policy_in_split = requests_leak["policy_rows_inside_training_splits"]
    near_pairs = requests_leak["near_duplicate_pairs"]
    cross_near = requests_leak["near_duplicate_pairs_cross_split"]
    train_skel = {s for r, s in zip(rows, skeletons) if r["metadata"]["split"] == "train"}
    heldout_rows = [(r, s) for r, s in zip(rows, skeletons) if r["metadata"]["split"] in ("validation", "test")]
    train_8 = {tuple(words(r["request"])[i:i + 8]) for r in rows if r["metadata"]["split"] == "train"
               for i in range(len(words(r["request"])) - 7)}
    held_8 = [tuple(words(r["request"])[i:i + 8]) for r, _ in heldout_rows
              for i in range(len(words(r["request"])) - 7)]
    split_sizes = Counter(m["split"] for m in meta)
    trainable = sum(split_sizes[s] for s in SPLITS)
    stats["split"] = {
        "sizes": {s: split_sizes.get(s, 0) for s in SPLITS + ("excluded_policy",)},
        "ratios_of_trainable": {s: round(split_sizes.get(s, 0) / trainable, 4) for s in SPLITS},
        "taxonomy_by_split": {s: dict(Counter(m["taxonomy"] for m in meta if m["split"] == s)) for s in SPLITS},
        "skill_by_split": {s: dict(Counter(k for m in meta if m["split"] == s for k in m["skills_used"]))
                           for s in SPLITS},
        "task_count_by_split": {s: dict(Counter(m["task_count"] for m in meta if m["split"] == s)) for s in SPLITS},
        "families_by_split": {s: sorted({m["template_family"] for m in meta if m["split"] == s}) for s in SPLITS},
        "leakage": {
            "template_family_across_splits": family_leak,
            "entity_family_across_splits": entity_leak,
            "semantic_seed_across_splits": semantic_leak,
            "paraphrase_sibling_across_splits": sibling_leak,
            "policy_rows_inside_training_splits": policy_in_split,
            "near_duplicate_pairs_total": len(near_pairs),
            "near_duplicate_pairs": near_pairs,
            "near_duplicate_pairs_cross_split": cross_near,
            "heldout_instruction_skeleton_seen_in_train": sum(1 for _, s in heldout_rows if s in train_skel),
            "heldout_request_8gram_seen_in_train_share": round(
                sum(g in train_8 for g in held_8) / len(held_8), 4) if held_8 else 0.0,
        },
    }

    # plans --------------------------------------------------------------
    results = [validate_plan_text(r["plan_text"], registry, mode="canonical", request=r["request"])
               for r in labeled]
    runtime_results = [validate_plan_text(r["plan_text"], registry, mode="runtime") for r in labeled]
    candidate_texts = []
    for r in policy:
        for candidate in r["metadata"]["candidate_plans"] or []:
            if candidate["plan"]:
                from manager_plan_validator import canonical_json

                candidate_texts.append(canonical_json(candidate["plan"]))
    parity_texts = [c for c, _ in PARITY_CASES] + [r["plan_text"] for r in labeled] + candidate_texts
    try:
        oracle = runtime_oracle(parity_texts)
        python = [validate_plan_text(t, registry, mode="runtime") for t in parity_texts]
        disagreements = [t[:120] for t, (ok, _), p in zip(parity_texts, oracle, python) if ok != p.valid]
        serialization_mismatch = [r["id"] for r, (ok, detail) in zip(labeled, oracle[len(PARITY_CASES):])
                                  if ok and detail != r["plan_text"]]
        oracle_stats = {"available": True, "cases": len(parity_texts), "fuzz_cases": len(PARITY_CASES),
                        "dataset_plans": len(labeled), "candidate_plans": len(candidate_texts),
                        "accept_reject_disagreements": disagreements,
                        "runtime_to_json_mismatches": serialization_mismatch,
                        "runtime_accepted_dataset_plans": sum(ok for ok, _ in oracle[len(PARITY_CASES):
                                                                                     len(PARITY_CASES) + len(labeled)])}
    except Exception as error:  # noqa: BLE001
        oracle_stats = {"available": False, "error": str(error)}
    warnings = Counter(code for res in results for code in res.warning_codes)
    candidate_warnings = Counter(w["code"] for r in policy for c in (r["metadata"]["candidate_plans"] or [])
                                 if c["validation"] for w in c["validation"]["warnings"])
    tokens = token_lengths([r["plan_text"] for r in labeled])
    qwen = tokens.get("qwen3_0.6b") if isinstance(tokens.get("qwen3_0.6b"), list) else None
    ko = tokens.get("ko32k_stage2") if isinstance(tokens.get("ko32k_stage2"), list) else None
    stats["plans"] = {
        "json_parse_valid": sum(not any(c in ("JSON_PARSE_ERROR", "NO_JSON_OBJECT") for c in res.codes)
                                for res in runtime_results),
        "runtime_schema_valid": sum(res.valid for res in runtime_results),
        "canonical_valid": sum(res.valid for res in results),
        "invalid_skill": sum(INVALID_SKILL in res.codes for res in results),
        "dependency_valid": sum(not (set(res.codes) & DEPENDENCY_CODES) for res in results),
        "grounded_inputs": sum("INPUT_NOT_GROUNDED" not in res.codes for res in results),
        "labeled_total": len(labeled),
        "error_codes": dict(Counter(code for res in results for code in res.codes)),
        "warning_codes": dict(warnings),
        "policy_candidate_plan_warning_codes": dict(candidate_warnings),
        "runtime_oracle": oracle_stats,
        "plan_tokens_qwen3": describe(qwen) if qwen else tokens.get("qwen3_0.6b"),
        "plan_tokens_ko32k": describe(ko) if ko else tokens.get("ko32k_stage2"),
        "plans_over_review_tokens": [r["id"] for r, n in zip(labeled, qwen or [])
                                     if n > GATE_THRESHOLDS["L_plan_token_review_above"]],
        "plans_over_manager_max_tokens": [r["id"] for r, n in zip(labeled, qwen or []) if n > MANAGER_MAX_TOKENS],
    }
    plan_tokens = dict(zip([r["id"] for r in labeled], qwen or []))

    # hard cases ---------------------------------------------------------
    hard = [r for r in rows if r["metadata"]["taxonomy_group"] != "normal"]
    stats["hard_cases"] = {
        "count": len(hard),
        "share": round(len(hard) / len(rows), 4),
        "labeled": sum(1 for r in hard if r["plan"]),
        "requires_human_policy": len(policy),
        "by_category": dict(Counter(r["metadata"]["hard_case_category"] for r in hard)),
        "by_taxonomy": dict(Counter(r["metadata"]["taxonomy"] for r in hard)),
    }

    # provenance ---------------------------------------------------------
    missing = [r["id"] for r in rows if any(k not in r["metadata"] or r["metadata"][k] is None
                                            for k in REQUIRED_METADATA)]
    stats["provenance"] = {"required_fields": list(REQUIRED_METADATA), "rows_missing_fields": missing,
                           "gold_labels": sum(m["status"] == "gold" for m in meta)}

    # pytest -------------------------------------------------------------
    if run_pytest:
        completed = subprocess.run([sys.executable, "-m", "pytest", "tests/test_manager_dataset.py", "-q",
                                    "-k", "Validator or Oracle", "-p", "no:cacheprovider"],
                                   cwd=PROJECT_ROOT, capture_output=True, text=True)
        summary = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else completed.stderr[-200:]
        stats["validator_tests"] = {"returncode": completed.returncode, "summary": summary}
    else:
        stats["validator_tests"] = {"returncode": None, "summary": "not run"}

    stats["findings"] = findings(rows, stats)
    stats["gates"] = gates(stats)
    stats["gate_thresholds"] = GATE_THRESHOLDS
    stats["human_decisions"] = [{"id": i, "severity": s, "question": q} for i, s, q in HUMAN_DECISIONS]
    stats["_plan_tokens"] = plan_tokens
    stats["_near_pairs"] = near_pairs
    return stats


def findings(rows: list[dict], stats: dict) -> list[str]:
    """Rule-derived review notes (not gates)."""
    notes = []
    kw = stats["shortcut_diagnostics"]["keyword_rule_accuracy"]
    if "normal" in kw:
        notes.append(f"Keyword shortcut: a surface-word rule predicts the skill sequence for "
                     f"{kw['normal']['accuracy']:.1%} of normal rows (hard rows: "
                     f"{kw.get('hard_labeled', {}).get('accuracy', 0):.1%}). Normal rows alone would let a model "
                     f"route by keywords.")
    corr = stats["shortcut_diagnostics"]["corr_instruction_length_vs_task_count"]
    if corr is not None:
        notes.append(f"Plan length leaks through instruction length: corr = {corr} "
                     f"(mean chars {stats['shortcut_diagnostics']['mean_instruction_chars_by_task_count']}).")
    if stats["dataset"]["rejected_attempts"] == 0:
        notes.append("Zero rejected attempts: the duplicate/invalid rejection path was never triggered in the pilot "
                     "run (it is covered by unit tests only). It is expected to trigger at larger N.")
    for name in sorted({r["metadata"]["taxonomy"] for r in rows if r["plan"]}):
        members = [r for r in rows if r["metadata"]["taxonomy"] == name]
        train_kinds = {r["metadata"]["source_kind"] for r in members if r["metadata"]["split"] == "train"}
        held_kinds = {r["metadata"]["source_kind"] for r in members if r["metadata"]["split"] in ("validation", "test")}
        train_families = {r["metadata"]["template_family"] for r in members if r["metadata"]["split"] == "train"}
        if held_kinds and held_kinds != train_kinds:
            notes.append(f"{name}: held-out source kinds {sorted(held_kinds)} differ from train {sorted(train_kinds)}.")
        if len(train_families) == 1:
            notes.append(f"{name}: train has a single template family ({next(iter(train_families))}).")
    hard_eval = stats["shortcut_diagnostics"]["bow_naive_bayes_train_to_heldout_hard"]["evaluated"]
    notes.append(f"Held-out hard rows: {hard_eval}; far too few for any accuracy claim.")
    if stats["dataset"]["multiline_requests"]:
        notes.append(f"{stats['dataset']['multiline_requests']} multi-line requests cannot be typed into the "
                     f"interactive CLI (HD-13).")
    reused = Counter(e for r in rows for e in set(r["metadata"]["entity_family"]) if not e.startswith("authored"))
    shared = sum(1 for count in reused.values() if count > 1)
    notes.append(f"{shared} source documents are used by more than one row (topic and text, or sibling pairs); "
                 f"all stay inside one split.")
    return notes


def gates(stats: dict) -> dict:
    d, div, sp, pl, hc = (stats["dataset"], stats["diversity"], stats["split"], stats["plans"],
                          stats["hard_cases"])
    leak = sp["leakage"]
    results = {}

    def put(key, status, evidence):
        results[key] = {"status": status, "evidence": evidence}

    n = pl["labeled_total"]
    put("A_schema_validity", "PASS" if pl["canonical_valid"] == n and pl["runtime_schema_valid"] == n else "FAIL",
        f"canonical {pl['canonical_valid']}/{n}, runtime {pl['runtime_schema_valid']}/{n}")
    put("B_invalid_skill", "PASS" if pl["invalid_skill"] == 0 else "FAIL", f"{pl['invalid_skill']} plans")
    put("C_dependency_validity", "PASS" if pl["dependency_valid"] == n else "FAIL", f"{pl['dependency_valid']}/{n}")
    dup = div["exact_duplicate_requests"] + div["normalized_duplicate_requests"]
    put("D_duplicate_requests", "PASS" if dup == 0 else "FAIL",
        f"exact {div['exact_duplicate_requests']}, normalized {div['normalized_duplicate_requests']}")
    leaks = (len(leak["template_family_across_splits"]) + len(leak["entity_family_across_splits"])
             + len(leak["semantic_seed_across_splits"]) + len(leak["paraphrase_sibling_across_splits"])
             + len(leak["near_duplicate_pairs_cross_split"]))
    put("E_split_leakage", "PASS" if leaks == 0 else "FAIL",
        f"family {len(leak['template_family_across_splits'])}, entity {len(leak['entity_family_across_splits'])}, "
        f"semantic {len(leak['semantic_seed_across_splits'])}, sibling {len(leak['paraphrase_sibling_across_splits'])}, "
        f"cross-split near-dup {len(leak['near_duplicate_pairs_cross_split'])}")
    shares = d["taxonomy_distribution"]
    matches = all(v["count"] == v["quota"] for v in shares.values())
    top = max(shares.items(), key=lambda kv: kv[1]["share"])
    ok = matches and top[1]["share"] <= GATE_THRESHOLDS["F_max_taxonomy_share"]
    put("F_taxonomy_dominance", "PASS" if ok else "REVIEW",
        f"realized == quota: {matches}; largest {top[0]} {top[1]['share']:.3f} "
        f"(threshold {GATE_THRESHOLDS['F_max_taxonomy_share']}); quotas themselves need approval (HD-12)")
    worst = max(div["max_family_share_within_taxonomy"].items(), key=lambda kv: kv[1]["share"])
    skel_share = div["top_instruction_skeletons"][0]["count"] / d["total_samples"]
    ok = (worst[1]["share"] <= GATE_THRESHOLDS["G_max_family_share_within_taxonomy"]
          and skel_share <= GATE_THRESHOLDS["G_max_instruction_skeleton_share"])
    groups = {name: v["group"] for name, v in d["taxonomy_groups"].items()}
    labeled_worst = max(((k, v) for k, v in div["max_family_share_within_taxonomy"].items()
                         if groups.get(k) != "policy_pending"), key=lambda kv: kv[1]["share"])
    put("G_template_dominance", "PASS" if ok else "REVIEW",
        f"max family share within taxonomy {worst[1]['share']:.3f} ({worst[0]}/{worst[1]['family']}); "
        f"labeled taxonomies only {labeled_worst[1]['share']:.3f} ({labeled_worst[0]}/{labeled_worst[1]['family']}); "
        f"top instruction skeleton share {skel_share:.3f}. Gate population was not changed after measurement.")
    missing = stats["provenance"]["rows_missing_fields"]
    put("H_provenance", "PASS" if not missing and stats["provenance"]["gold_labels"] == 0 else "FAIL",
        f"{len(missing)} rows missing fields, {stats['provenance']['gold_labels']} rows labelled gold")
    oracle = pl["runtime_oracle"]
    tests = stats["validator_tests"]
    parity = oracle.get("available") and not oracle["accept_reject_disagreements"] and not oracle["runtime_to_json_mismatches"]
    status = "PASS" if parity and tests["returncode"] == 0 else ("REVIEW" if tests["returncode"] is None else "FAIL")
    put("I_validator_tests", status,
        f"C++ oracle parity {'ok' if parity else 'FAILED'} on {oracle.get('cases')} cases; pytest: {tests['summary']}")
    sep = (not leak["policy_rows_inside_training_splits"] and hc["requires_human_policy"] == d["policy_pending_unlabeled"])
    put("J_policy_rows_separated", "PASS" if sep else "FAIL",
        f"{hc['requires_human_policy']} rows, plan=null, split=excluded_policy, training_candidate=false")
    put("K_input_grounding", "PASS" if pl["grounded_inputs"] == n else "FAIL",
        f"{pl['grounded_inputs']}/{n} literal inputs are verbatim request spans (candidate policy HD-01)")
    if not isinstance(pl["plan_tokens_qwen3"], dict) or "max" not in pl["plan_tokens_qwen3"]:
        put("L_manager_output_budget", "REVIEW", "Qwen3 tokenizer unavailable; not measured")
    else:
        worst_tokens = pl["plan_tokens_qwen3"]["max"]
        status = ("FAIL" if pl["plans_over_manager_max_tokens"] else
                  "REVIEW" if pl["plans_over_review_tokens"] else "PASS")
        put("L_manager_output_budget", status,
            f"max plan {worst_tokens} Qwen3 tokens vs --manager-max-tokens {MANAGER_MAX_TOKENS}; "
            f"{len(pl['plans_over_review_tokens'])} plans > {GATE_THRESHOLDS['L_plan_token_review_above']}")
    blocking = [i for i, s, _ in HUMAN_DECISIONS if s == "blocking"]
    put("M_blocking_human_decisions", "REVIEW" if blocking else "PASS",
        f"{len(blocking)} blocking decisions open: {', '.join(blocking)}")
    statuses = [v["status"] for v in results.values()]
    overall = "FAIL" if "FAIL" in statuses else "REVIEW" if "REVIEW" in statuses else "PASS"
    return {"overall": overall, "items": results}


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #

def _flags(row: dict, stats: dict, near: dict) -> list[str]:
    m = row["metadata"]
    flags = []
    tokens = stats["_plan_tokens"].get(row["id"])
    if tokens is not None and tokens > GATE_THRESHOLDS["L_plan_token_review_above"]:
        flags.append(f"plan {tokens} tok")
    if "\n" in row["request"]:
        flags.append("multi-line (HD-13)")
    if m["paraphrase_of"]:
        flags.append(f"sibling of {m['paraphrase_of']}")
    if len(m["entity_family"]) and any(sum(e in r for r in stats["_entity_members"]) > 1
                                       for e in m["entity_family"]) and not m["entity_family"][0].startswith("authored"):
        flags.append("entity reused")
    if m["validation"] and m["validation"].get("warnings"):
        flags.append("warn:" + ",".join(w["code"] for w in m["validation"]["warnings"]))
    if row["id"] in near:
        flags.append("near-dup " + ",".join(near[row["id"]]))
    if m["requires_human_policy"]:
        flags.append("REQUIRES HUMAN POLICY")
    if m["hard_case_category"]:
        flags.append(m["hard_case_category"])
    return flags


def _cell(text: str, limit: int | None = 90) -> str:
    text = text.replace("\n", "⏎").replace("|", "\\|")
    return text if limit is None or len(text) <= limit else text[:limit - 1] + "…"


def _plan_cell(row: dict) -> str:
    if not row["plan"]:
        return "— (withheld: requires human policy)"
    parts = []
    for t in row["plan"]["tasks"]:
        src = f"${t['input_from']}" if "input_from" in t else f"\"{_cell(t['input'], 24)}\""
        parts.append(f"{t['id']}:{t['skill']}({src})")
    return " → ".join(parts)


def render_review(rows: list[dict], stats: dict) -> str:
    near: dict[str, list[str]] = defaultdict(list)
    for a, b in stats["_near_pairs"]:
        near[a].append(b)
        near[b].append(a)
    by_id = {r["id"]: r for r in rows}
    labeled = [r for r in rows if r["plan"]]
    skel_counts = Counter(skeleton(r) for r in rows)

    subsets = []
    subsets.append(("Random 30 (random.Random(0).sample over all 200, not curated)",
                    random.Random(REVIEW_SEED).sample(rows, 30)))
    subsets.append(("Longest 10 requests (characters)", sorted(rows, key=lambda r: (-len(r["request"]), r["id"]))[:10]))
    subsets.append(("Most complex 10 plans (task count, depth, then plan tokens)",
                    sorted(labeled, key=lambda r: (-r["metadata"]["task_count"], -r["metadata"]["plan_depth"],
                                                   -stats["_plan_tokens"].get(r["id"], 0), r["id"]))[:10]))
    subsets.append(("All hard cases", [r for r in rows if r["metadata"]["taxonomy_group"] != "normal"]))
    risk = sorted(rows, key=lambda r: (-skel_counts[skeleton(r)], -len(near.get(r["id"], [])),
                                       -(r["metadata"]["paraphrase_of"] is not None), r["id"]))
    subsets.append(("Highest repetition risk 15 (shared instruction skeleton, near-duplicates, siblings)", risk[:15]))
    borderline = [r for r in rows if _flags(r, stats, near) and any(
        f.startswith(("plan ", "multi-line", "warn:", "near-dup", "entity reused")) for f in _flags(r, stats, near))]
    subsets.append(("Validator borderline (token budget, warnings, multi-line, near-dup, reused entity)", borderline))

    lines = ["# Manager pilot v1 — human review sheet", "",
             "Every row is a **candidate**, not gold. Subsets are selected by fixed rules (listed in each "
             "heading); nothing was hand-picked. The full 200 follow the subsets.", "",
             f"Gate result: **{stats['gates']['overall']}** — see `pilot_report.md`.", "",
             "Plan notation: `id:skill(\"input\")` or `id:skill($k)` for `input_from: k`. Full JSON is in the "
             "full listing.", ""]
    for title, members in subsets:
        lines += [f"## {title} — {len(members)}", "",
                  "| ID | User request | Expected TaskPlan | Taxonomy | Source | Notes |",
                  "|---|---|---|---|---|---|"]
        for r in members:
            m = r["metadata"]
            lines.append(f"| {r['id'][-4:]} | {_cell(r['request'])} | {_plan_cell(r)} | {m['taxonomy']} | "
                         f"{m['source_type']} / {m['template_family']} / {m['split']} | "
                         f"{_cell('; '.join(_flags(r, stats, near)), 80)} |")
        lines.append("")
    lines += ["## Full listing — 200", ""]
    for r in rows:
        m = r["metadata"]
        lines += [f"### {r['id']} · {m['taxonomy']} · {m['split']}", "", "```text", r["request"], "```", ""]
        if r["plan"]:
            lines += [f"Expected TaskPlan: `{r['plan_text']}`", ""]
        else:
            lines.append("Expected TaskPlan: **withheld — REQUIRES HUMAN DECISION**. Candidates:")
            for c in m["candidate_plans"] or []:
                from manager_plan_validator import canonical_json

                plan = f"`{canonical_json(c['plan'])}`" if c["plan"] else "no plan"
                lines.append(f"- `{c['policy']}`: {plan}")
            lines.append("")
        entity = ", ".join(m["entity_family"])
        lines += [f"Source: {m['source_type']} · family `{m['template_family']}` · register {m['register']} · "
                  f"semantic `{m['semantic_id']}` · entity `{entity}`", "",
                  f"Notes: {'; '.join(_flags(r, stats, near)) or '—'}", ""]
    return "\n".join(lines) + "\n"


def _table(mapping: dict, headers=("key", "value")) -> list[str]:
    out = [f"| {headers[0]} | {headers[1]} |", "|---|---:|"]
    out += [f"| {k} | {v} |" for k, v in mapping.items()]
    return out


def render_report(rows: list[dict], stats: dict) -> str:
    d, div, sc, sp, pl, hc, g = (stats["dataset"], stats["diversity"], stats["shortcut_diagnostics"],
                                 stats["split"], stats["plans"], stats["hard_cases"], stats["gates"])
    lines = ["# Manager-SLM pilot v1 — report", "",
             f"**Pilot gate: {g['overall']}**", "",
             "This gate answers only: *is the answer definition, generator and validator trustworthy enough to "
             "start mass generation?* It is not a research threshold. No model was trained.", "",
             "## Gate", "", "| Gate | Status | Evidence |", "|---|---|---|"]
    lines += [f"| {k} | {v['status']} | {v['evidence']} |" for k, v in g["items"].items()]
    lines += ["", "Thresholds were declared in `model/manager_dataset_analysis.py::GATE_THRESHOLDS` before "
              "measurement:", "", "```json", json.dumps(stats["gate_thresholds"], indent=2), "```", "",
              "## Findings for review", ""]
    lines += [f"- {note}" for note in stats["findings"]]
    lines += ["", "## Dataset", "",
              f"- samples: {d['total_samples']} ({d['labeled_training_candidates']} labeled candidates, "
              f"{d['policy_pending_unlabeled']} policy-pending without label)",
              f"- generation attempts: {d['generation_attempts']}, rejected: {d['rejected_attempts']} "
              f"{d['reject_reasons'] or ''}",
              f"- entity pool: {d['entity_pool']['accepted']} documents from `{d['entity_pool']['source']}` "
              f"(pool-stage rejects {d['entity_pool']['rejected']})",
              f"- source_type: {d['source_type']}; status: {d['status']}",
              f"- multi-line requests: {d['multiline_requests']}", "",
              "| taxonomy | count | quota | share |", "|---|---:|---:|---:|"]
    lines += [f"| {k} | {v['count']} | {v['quota']} | {v['share']:.3f} |" for k, v in d["taxonomy_distribution"].items()]
    lines += ["", "Skill task frequency: " + json.dumps(d["skill_task_frequency"]),
              "", "Skill sequences (labeled): " + json.dumps(d["skill_sequence"]),
              "", "Task count: " + json.dumps(d["task_count"]) + " · plan depth: " + json.dumps(d["plan_depth"]),
              "", "Register: " + json.dumps(d["register"], ensure_ascii=False),
              "", "Request length (chars): " + json.dumps(d["request_chars"]),
              "", "Instruction length without arguments (chars): " + json.dumps(d["instruction_chars"]),
              "", "Request length histogram: " + json.dumps(d["request_chars_histogram"]), "",
              "## Diversity", ""]
    keys = ("exact_duplicate_requests", "normalized_duplicate_requests", "unique_request_ratio",
            "unique_normalized_request_ratio", "plan_text_exact_duplicate_ratio", "within_request_dup_4gram_mean",
            "within_request_dup_8gram_mean", "cross_request_shared_4gram_share", "cross_request_shared_8gram_share",
            "instruction_cross_request_shared_4gram_share", "instruction_cross_request_shared_8gram_share",
            "unique_instruction_skeleton_ratio", "template_family_count", "max_family_share_overall",
            "instruction_type_token_ratio", "instruction_distinct_1", "instruction_distinct_2_ratio",
            "full_request_type_token_ratio")
    lines += _table({k: div[k] for k in keys}, ("metric", "value"))
    lines += ["", "`within_request_dup_*` reuses `corpus/stage2_dedup.ngram_duplicate_ratio` (Stage-2 canonical "
              "definition). `cross_request_*` = share of word n-gram occurrences that also occur in another "
              "request; `instruction_*` measures the same after replacing topic/text arguments with placeholders, "
              "which isolates template wording from payload text.", "",
              "Top instruction skeletons:", ""]
    lines += [f"- {s['count']}× `{_cell(s['skeleton'], 120)}`" for s in div["top_instruction_skeletons"]]
    lines += ["", "Max template family share within taxonomy:", ""]
    lines += [f"- {k}: {v['family']} {v['share']:.3f} ({v['families_used']} families)"
              for k, v in div["max_family_share_within_taxonomy"].items()]
    top_overlap = sorted(div["taxonomy_instruction_vocab_jaccard"].items(), key=lambda kv: -kv[1])[:6]
    lines += ["", "Highest instruction-vocabulary Jaccard between taxonomies: "
              + ", ".join(f"{k} {v}" for k, v in top_overlap), "",
              "## Shortcut diagnostics", "",
              "- keyword rule (요약/간추/줄… → summary, 키워드/개념/뽑… → search, order by first mention): "
              + json.dumps(sc["keyword_rule_accuracy"]),
              f"- bag-of-words naive Bayes, train → validation+test: {sc['bow_naive_bayes_train_to_heldout']}",
              f"- same, hard held-out rows only: {sc['bow_naive_bayes_train_to_heldout_hard']}",
              f"- corr(instruction length, task count) = {sc['corr_instruction_length_vs_task_count']}; "
              f"corr(request length, task count) = {sc['corr_request_length_vs_task_count']}",
              f"- mean instruction chars by task count: {sc['mean_instruction_chars_by_task_count']}", "",
              "High keyword-rule accuracy on normal rows means surface words predict the plan; the hard rows exist "
              "to break that shortcut. These are diagnostics, not gates.", "",
              "## Split", "", "Group-aware by construction: template families and source documents are assigned "
              "to one split before realization; paraphrase siblings inherit their base row's split; "
              "policy-pending rows are `excluded_policy`.", "",
              "Sizes: " + json.dumps(sp["sizes"]) + " · ratios of trainable: " + json.dumps(sp["ratios_of_trainable"]),
              ""]
    for s in SPLITS:
        lines.append(f"- {s}: taxonomy {json.dumps(sp['taxonomy_by_split'][s])}; skills "
                     f"{json.dumps(sp['skill_by_split'][s])}")
    lk = sp["leakage"]
    lines += ["", "| leakage check | result |", "|---|---:|",
              f"| template family in >1 split | {len(lk['template_family_across_splits'])} |",
              f"| entity (source document) in >1 split | {len(lk['entity_family_across_splits'])} |",
              f"| semantic seed in >1 split | {len(lk['semantic_seed_across_splits'])} |",
              f"| paraphrase sibling in other split | {len(lk['paraphrase_sibling_across_splits'])} |",
              f"| policy rows inside train/val/test | {len(lk['policy_rows_inside_training_splits'])} |",
              f"| near-duplicate pairs (SimHash ≤{GATE_THRESHOLDS['E_near_duplicate_hamming']}) total / cross-split | "
              f"{lk['near_duplicate_pairs_total']} / {len(lk['near_duplicate_pairs_cross_split'])} |",
              f"| held-out instruction skeleton seen in train | {lk['heldout_instruction_skeleton_seen_in_train']} |",
              f"| held-out request 8-grams seen in train | {lk['heldout_request_8gram_seen_in_train_share']} |",
              "", "## Plans and validator", "",
              f"- JSON parse valid {pl['json_parse_valid']}/{pl['labeled_total']}, runtime-schema valid "
              f"{pl['runtime_schema_valid']}, canonical valid {pl['canonical_valid']}, dependency valid "
              f"{pl['dependency_valid']}, invalid skill {pl['invalid_skill']}, grounded {pl['grounded_inputs']}",
              f"- warnings on labeled plans: {pl['warning_codes'] or 'none'}; on policy candidate plans: "
              f"{pl['policy_candidate_plan_warning_codes'] or 'none'}",
              f"- C++ oracle (unchanged `task_plan.cpp`): {json.dumps({k: v for k, v in pl['runtime_oracle'].items() if k != 'accept_reject_disagreements'})}; "
              f"disagreements {pl['runtime_oracle'].get('accept_reject_disagreements')}",
              f"- plan length, Qwen3-0.6B tokens: {pl['plan_tokens_qwen3']}",
              f"- plan length, KO32K Stage-2 tokens: {pl['plan_tokens_ko32k']}",
              f"- plans over {GATE_THRESHOLDS['L_plan_token_review_above']} Qwen3 tokens: "
              f"{pl['plans_over_review_tokens'] or 'none'}; over {MANAGER_MAX_TOKENS}: "
              f"{pl['plans_over_manager_max_tokens'] or 'none'}",
              f"- validator pytest subset: {stats['validator_tests']['summary']}", "",
              "## Hard cases", "",
              f"- {hc['count']} ({hc['share']:.1%}); labeled {hc['labeled']}; requires_human_policy "
              f"{hc['requires_human_policy']}",
              f"- by category: {json.dumps(hc['by_category'])}", "",
              "## Unresolved human decisions", "", "| ID | severity | question |", "|---|---|---|"]
    lines += [f"| {h['id']} | {h['severity']} | {h['question']} |" for h in stats["human_decisions"]]
    lines += ["", "STOPPED BEFORE MASS DATASET GENERATION AND MANAGER-SLM TRAINING.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--skip-pytest", action="store_true")
    arguments = parser.parse_args()
    rows = load_rows(arguments.input)
    log = json.loads(arguments.input.with_suffix(".generation_log.json").read_text(encoding="utf-8"))
    stats = analyse(rows, log, run_pytest=not arguments.skip_pytest)
    stats["_entity_members"] = [r["metadata"]["entity_family"] for r in rows]
    arguments.out_dir.mkdir(parents=True, exist_ok=True)
    public = {k: v for k, v in stats.items() if not k.startswith("_")}
    public["input"] = str(arguments.input.relative_to(PROJECT_ROOT))
    (arguments.out_dir / "pilot_stats.json").write_text(json.dumps(public, ensure_ascii=False, indent=2) + "\n",
                                                        encoding="utf-8")
    (arguments.out_dir / "pilot_report.md").write_text(render_report(rows, stats), encoding="utf-8")
    (arguments.out_dir / "pilot_review.md").write_text(render_review(rows, stats), encoding="utf-8")
    print(f"pilot gate: {stats['gates']['overall']}")
    for key, value in stats["gates"]["items"].items():
        print(f"  {key}: {value['status']} — {value['evidence']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
