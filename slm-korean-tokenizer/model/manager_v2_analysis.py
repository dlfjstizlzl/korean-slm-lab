#!/usr/bin/env python3
"""Statistics, leakage, shortcut baselines and gates for Manager v2 datasets.

    python model/manager_v2_analysis.py --dataset pilot   # pilot_v2_stats.json / report / review
    python model/manager_v2_analysis.py --dataset 5k      # 5k_stats.json / report / review

All gate thresholds are module constants written *before* the pilot was
generated (see HARD_GATES / QUALITY_THRESHOLDS). The pilot decision
`proceed_to_5k` is derived mechanically from them.
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
import zlib
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "model"))
sys.path.insert(0, str(PROJECT_ROOT / "corpus"))

from manager_v2_generator import SPLITS, normalize_request  # noqa: E402
from manager_v2_registry import load_contract  # noqa: E402
from manager_v2_validator import (  # noqa: E402
    DEPENDENCY_CODES,
    OUTPUT_CODES,
    PARITY_CASES,
    canonical_json,
    cpp_grammar,
    cpp_validate,
    validate,
)
from stage2_dedup import hamming, ngram_duplicate_ratio, simhash  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "manager"
OUT_DIR = PROJECT_ROOT / "outputs" / "manager_dataset_v2"
V1_STATS = PROJECT_ROOT / "outputs" / "manager_dataset" / "pilot_stats.json"
MANAGER_MAX_TOKENS = 256
MANAGER_CONTEXT = 4096   # main.cpp: manager context_size = max(--context, 4096)
NEAR_DUP_HAMMING = 3
REVIEW_SEED = 0

# ---- pre-declared thresholds (written before Pilot v2 was generated) --------
QUALITY_THRESHOLDS = {
    "Q1_keyword_rule_accuracy_normal_max": 0.60,     # Pilot v1: 0.9817
    "Q2_abs_corr_instruction_len_task_count_max": 0.40,  # Pilot v1: 0.7781
    "Q2_abs_corr_request_len_task_count_max": 0.40,      # Pilot v1: 0.2787
    "Q3_max_family_share_within_taxonomy": 0.35,     # taxonomies with >= 3 families used
    "Q4_max_taxonomy_share": 0.12,
    "Q5_min_skill_task_share": 0.08,                 # each of the 5 skills
    "Q6_min_single_task_share": 0.25,
    "Q6_min_multi_task_share": 0.25,
    "Q6_min_three_plus_task_share": 0.08,
    "Q7_min_multi_source_task_rows_share": 0.10,
    "Q7_min_hard_share": 0.20,
}

SUMMARY_CUE = re.compile(r"요약|정리|간추|줄여|요점|요지")
EXTRACT_CUE = re.compile(r"뽑|추출|골라|고르|항목|만 알려")
SEARCH_CUE = re.compile(r"검색|찾아|찾고|찾은|조사|알아봐|알아보|최신|요즘|인터넷")
SQL_CUE = re.compile(r"조회|DB|데이터|매출|주문|집계|테이블|시스템")
COMPARE_CUE = re.compile(r"비교|vs|대조|차이|견줘|나은지")
UNSUPPORTED_CUE = re.compile(r"번역|코드|그려|그림|메일|문자|카톡|엑셀|PDF|파일|알람|리마인더|예약|예매|조명|에어컨|스피커|"
                             r"청소기|계산|재생|틀기|라디오")
DEICTIC_CUE = re.compile(r"이거|그거|이 글|아래|붙여|아까|저번|그 제품|그 모델|그 데이터|그 지표|이 두|둘이")

REQUIRED_METADATA = ("sample_id", "generator_version", "schema_version", "source_type", "taxonomy",
                     "template_family", "semantic_seed", "skills_used", "task_count", "dependency_depth",
                     "source_count", "status", "difficulty", "hard_case", "split_group", "seed", "split",
                     "entity_family", "paraphrase_of", "status_label")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def skeleton(row: dict) -> str:
    text = row["request"]
    for key in ("X", "X1", "X2", "LC"):
        value = row["metadata"]["arguments"].get(key)
        if value:
            text = text.replace(value, f"<{key}>")
    return text


def signature(plan: dict) -> str:
    if plan["status"] != "ok":
        return plan["status"]
    seen = []
    for task in plan["tasks"]:
        if task["skill"] not in seen:
            seen.append(task["skill"])
    return "ok:" + ">".join(seen)


def topology(plan: dict) -> str:
    if plan["status"] != "ok":
        return plan["status"]
    parts = [f"{t['skill']}({','.join(map(str, t['sources']))})" for t in plan["tasks"]]
    return " ".join(parts) + f" -> {plan['outputs']}"


def describe(values: list[float]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    q = statistics.quantiles(ordered, n=100) if len(ordered) > 1 else [ordered[0]] * 99
    return {"count": len(values), "min": ordered[0], "median": statistics.median(ordered), "p95": q[94],
            "max": ordered[-1], "mean": round(statistics.fmean(ordered), 2)}


def pearson(xs, ys):
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    return round(statistics.correlation(xs, ys), 4)


def keyword_rule(row: dict) -> str:
    text = skeleton(row)
    if UNSUPPORTED_CUE.search(text):
        return "unsupported"
    if not row["metadata"]["has_user_payload"] and len(text) < 30 and DEICTIC_CUE.search(text):
        return "clarify"
    found = []
    for skill, cue in (("search", SEARCH_CUE), ("sql", SQL_CUE), ("extract", EXTRACT_CUE), ("summary", SUMMARY_CUE),
                       ("compare", COMPARE_CUE)):
        match = cue.search(text)
        if match:
            found.append((match.start() if skill != "compare" else 10**6, skill))
    if not found:
        return "ok:search"
    return "ok:" + ">".join(skill for _, skill in sorted(found))


def cross_share(texts: list[str], n: int) -> float:
    owners, grams_all = defaultdict(set), []
    for i, text in enumerate(texts):
        words = text.split()
        grams = [tuple(words[j:j + n]) for j in range(len(words) - n + 1)]
        grams_all.append(grams)
        for gram in grams:
            owners[gram].add(i)
    total = sum(map(len, grams_all))
    return round(sum(1 for gs in grams_all for g in gs if len(owners[g]) > 1) / total, 4) if total else 0.0


def features(text: str, dims: int = 1 << 15) -> dict[int, float]:
    tokens = text.split()
    feats = Counter()
    for token in tokens:
        feats[zlib.crc32(("w\x1f" + token).encode()) % dims] += 1
    compact = re.sub(r"\s+", "", text)
    for i in range(len(compact) - 1):
        feats[zlib.crc32(("c\x1f" + compact[i:i + 2]).encode()) % dims] += 1
    return feats


def logistic_regression(train, test, epochs=300, lr=0.5, l2=1e-4, dims=1 << 15):
    """Softmax regression on hashed word + char-bigram features (diagnostic only)."""
    labels = sorted({y for _, y in train})
    index = {y: i for i, y in enumerate(labels)}

    def matrix(items):
        x = np.zeros((len(items), dims), dtype=np.float32)
        for row, (text, _) in enumerate(items):
            feats = features(text, dims)
            norm = math.sqrt(sum(v * v for v in feats.values())) or 1.0
            for key, value in feats.items():
                x[row, key] = value / norm
        return x

    x_train, x_test = matrix(train), matrix(test)
    y = np.array([index[label] for _, label in train])
    weights = np.zeros((dims, len(labels)), dtype=np.float32)
    onehot = np.eye(len(labels), dtype=np.float32)[y]
    for _ in range(epochs):
        logits = x_train @ weights
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        weights -= lr * (x_train.T @ (probs - onehot) / len(train) + l2 * weights)
    predictions = (x_test @ weights).argmax(axis=1)
    return [labels[i] for i in predictions]


def naive_bayes(train, test):
    counts, totals, priors, vocab = defaultdict(Counter), Counter(), Counter(), set()
    for text, label in train:
        tokens = text.split()
        counts[label].update(tokens)
        totals[label] += len(tokens)
        priors[label] += 1
        vocab.update(tokens)
    out = []
    for text, _ in test:
        best = max(priors, key=lambda c: math.log(priors[c]) + sum(
            math.log((counts[c][t] + 1) / (totals[c] + len(vocab))) for t in text.split()))
        out.append(best)
    return out


def token_counts(texts: list[str]) -> list[int] | None:
    try:
        import os

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        return [len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in texts]
    except Exception:  # noqa: BLE001
        return None


def mutations(plan_text: str, rng: random.Random) -> list[str]:
    """Invalid variants of a valid plan for differential testing."""
    plan = json.loads(plan_text)
    out = []
    if plan["status"] == "ok":
        tasks = plan["tasks"]
        broken = json.loads(plan_text)
        broken["tasks"][-1]["sources"] = [len(tasks) - 1]
        out.append(json.dumps(broken, ensure_ascii=False, separators=(",", ":")))
        broken = json.loads(plan_text)
        broken["outputs"] = [len(tasks)]
        out.append(json.dumps(broken, ensure_ascii=False, separators=(",", ":")))
        broken = json.loads(plan_text)
        broken["tasks"][0]["skill"] = rng.choice(["translate", "code", "browse", "Search"])
        out.append(json.dumps(broken, ensure_ascii=False, separators=(",", ":")))
        broken = json.loads(plan_text)
        broken["tasks"][0]["input"] = "x"
        out.append(json.dumps(broken, ensure_ascii=False, separators=(",", ":")))
        broken = json.loads(plan_text)
        broken["tasks"][0]["instruction"] = " "
        out.append(json.dumps(broken, ensure_ascii=False, separators=(",", ":")))
        broken = json.loads(plan_text)
        skill = broken["tasks"][0]["skill"]
        broken["tasks"][0]["sources"] = ["user"] if skill in ("search", "sql") else []
        out.append(json.dumps(broken, ensure_ascii=False, separators=(",", ":")))
        broken = json.loads(plan_text)
        broken["message"] = "x"
        out.append(json.dumps(broken, ensure_ascii=False, separators=(",", ":")))
        out.append(plan_text.replace('"id":0', '"id":0.0', 1))
        out.append(plan_text[:-1])
    else:
        broken = json.loads(plan_text)
        broken["tasks"] = [{"id": 0, "skill": "search", "instruction": "x", "sources": []}]
        out.append(json.dumps(broken, ensure_ascii=False, separators=(",", ":")))
        broken = json.loads(plan_text)
        del broken["message"]
        out.append(json.dumps(broken, ensure_ascii=False, separators=(",", ":")))
        out.append(plan_text.replace('"status":', '"status":"x","status":', 1))
    return out


def leakage(rows: list[dict]) -> dict:
    split_of = {r["id"]: r["metadata"]["split"] for r in rows}

    def spread(key_fn):
        owners = defaultdict(set)
        for r in rows:
            for key in key_fn(r):
                owners[key].add(r["metadata"]["split"])
        return sorted(k for k, v in owners.items() if len(v) > 1)

    hashes = [simhash(r["request"]) for r in rows]
    buckets = defaultdict(list)
    for i, h in enumerate(hashes):
        for band in range(4):
            buckets[(band, (h >> (16 * band)) & 0xFFFF)].append(i)
    pairs = set()
    for members in buckets.values():
        for i, j in combinations(members, 2):
            if hamming(hashes[i], hashes[j]) <= NEAR_DUP_HAMMING:
                pairs.add((min(i, j), max(i, j)))
    near = [(rows[i]["id"], rows[j]["id"]) for i, j in sorted(pairs)]
    cross = [(a, b) for a, b in near if split_of[a] != split_of[b]]
    # union-find over every grouping key: each component must sit in one split
    parent = list(range(len(rows)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    first = {}
    for i, r in enumerate(rows):
        m = r["metadata"]
        keys = [f"fam:{m['template_family']}", f"sem:{m['semantic_seed']}"] + [f"ent:{e}" for e in m["entity_family"]]
        if m["paraphrase_of"]:
            keys.append(f"row:{m['paraphrase_of']}")
        keys.append(f"row:{r['id']}")
        for key in keys:
            if key in first:
                parent[find(i)] = find(first[key])
            else:
                first[key] = i
    components = defaultdict(set)
    for i, r in enumerate(rows):
        components[find(i)].add(r["metadata"]["split"])
    return {
        "template_family_across_splits": spread(lambda r: [r["metadata"]["template_family"]]),
        "semantic_seed_across_splits": spread(lambda r: [r["metadata"]["semantic_seed"]]),
        "entity_family_across_splits": spread(lambda r: r["metadata"]["entity_family"]),
        "paraphrase_sibling_across_splits": [r["id"] for r in rows if r["metadata"]["paraphrase_of"]
                                             and split_of[r["metadata"]["paraphrase_of"]] != r["metadata"]["split"]],
        "near_duplicate_pairs": len(near),
        "near_duplicate_examples": near[:30],
        "near_duplicate_pairs_cross_split": cross,
        "group_components": len(components),
        "group_components_spanning_splits": sum(len(v) > 1 for v in components.values()),
    }


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #

def analyse(rows: list[dict], log: dict, dataset: str) -> dict:
    contract = load_contract()
    meta = [r["metadata"] for r in rows]
    plans = [r["plan"] for r in rows]
    n = len(rows)
    skeletons = [skeleton(r) for r in rows]
    stats: dict = {"dataset": dataset, "rows": n}

    # ---- distributions ------------------------------------------------------
    task_skill = Counter(t["skill"] for p in plans for t in p["tasks"])
    total_tasks = sum(task_skill.values())
    cooc = {a: {b: 0 for b in contract.skill_names} for a in contract.skill_names}
    for p in plans:
        used = {t["skill"] for t in p["tasks"]}
        for a in used:
            for b in used:
                cooc[a][b] += 1
    ok_rows = [p for p in plans if p["status"] == "ok"]
    stats["distribution"] = {
        "taxonomy": dict(Counter(m["taxonomy"] for m in meta)),
        "taxonomy_group": dict(Counter(m["taxonomy_group"] for m in meta)),
        "status": dict(Counter(p["status"] for p in plans)),
        "skill_task_frequency": dict(task_skill),
        "skill_task_share": {k: round(v / total_tasks, 4) for k, v in task_skill.items()},
        "skill_row_presence": {s: sum(1 for p in plans if any(t["skill"] == s for t in p["tasks"]))
                               for s in contract.skill_names},
        "skill_cooccurrence_rows": cooc,
        "task_count": dict(sorted(Counter(len(p["tasks"]) for p in plans).items())),
        "dependency_depth": dict(sorted(Counter(m["dependency_depth"] for m in meta).items())),
        "output_count": dict(sorted(Counter(len(p["outputs"]) for p in plans).items())),
        "topology": dict(Counter(topology(p) for p in plans).most_common()),
        "signature": dict(Counter(signature(p) for p in plans).most_common()),
        "source_usage": {
            "rows_with_user_source": sum(1 for p in plans if any("user" in t["sources"] for t in p["tasks"])),
            "rows_with_task_source": sum(1 for p in plans if any(s != "user" for t in p["tasks"] for s in t["sources"])),
            "rows_with_multi_source_task": sum(1 for p in plans if any(len(t["sources"]) > 1 for t in p["tasks"])),
            "tasks_by_source_count": dict(sorted(Counter(len(t["sources"]) for p in plans for t in p["tasks"]).items())),
        },
        "hard_case_category": dict(Counter(m["hard_case_category"] for m in meta if m["hard_case"])),
        "hard_share": round(sum(m["hard_case"] for m in meta) / n, 4),
        "source_type": dict(Counter(m["source_type"] for m in meta)),
        "register": dict(Counter(m["register"] for m in meta)),
        "verbosity": dict(Counter(m["verbosity"] for m in meta)),
        "family_tier": dict(Counter(m["family_tier"] for m in meta)),
        "request_chars": describe([len(r["request"]) for r in rows]),
        "request_chars_without_payload": describe([len(s) for s in skeletons]),
        "payload_chars": describe([m["payload_chars"] for m in meta if m["has_user_payload"]]),
        "instruction_chars": describe([len(t["instruction"]) for p in plans for t in p["tasks"]]),
        "single_task_share": round(sum(len(p["tasks"]) == 1 for p in plans) / n, 4),
        "multi_task_share": round(sum(len(p["tasks"]) >= 2 for p in plans) / n, 4),
        "three_plus_task_share": round(sum(len(p["tasks"]) >= 3 for p in plans) / n, 4),
    }

    # ---- plans / validator / differential ------------------------------------
    py = [validate(r["plan_text"], contract, mode="canonical",
                   payloads=[v for k, v in r["metadata"]["arguments"].items() if k in ("X", "X1", "X2")] or None,
                   names=[v for k, v in r["metadata"]["arguments"].items() if k in ("T", "TT1", "TT2")])
          for r in rows]
    rng = random.Random(0)
    mutated = [m for r in rows for m in mutations(r["plan_text"], rng)]
    corpus = [t for t, _ in PARITY_CASES] + [r["plan_text"] for r in rows] + mutated
    cpp = cpp_validate(corpus)
    py_runtime = [validate(t, contract, mode="runtime") for t in corpus]
    mismatch = [str(t)[:140] for t, (ok, code, _), p in zip(corpus, cpp, py_runtime)
                if ok != p.valid or (not ok and code != p.code)]
    serialization = [r["id"] for r, (ok, _, detail) in zip(rows, cpp[len(PARITY_CASES):]) if ok and detail != r["plan_text"]]
    grammar = cpp_grammar([r["plan_text"] for r in rows] + mutated)
    grammar_rows, grammar_mut = grammar[:n], grammar[n:]
    mutated_cpp = cpp[len(PARITY_CASES) + n:]
    grammar_accepts_invalid = sum(g for g, (ok, _, _) in zip(grammar_mut, mutated_cpp) if g and not ok)
    tokens = token_counts([r["plan_text"] for r in rows])
    prompt_tokens = token_counts([contract.system_prompt])
    request_tokens = token_counts([r["request"] for r in rows])
    stats["plans"] = {
        "schema_valid": sum(p.valid or p.code in ("PAYLOAD_COPY", "USER_SOURCE_WITHOUT_PAYLOAD", "NON_CANONICAL")
                            for p in py),
        "canonical_valid": sum(p.valid for p in py),
        "invalid_codes": dict(Counter(p.code for p in py if not p.valid)),
        "invalid_skill": sum(p.code == "INVALID_SKILL" for p in py),
        "dependency_valid": sum(p.code not in DEPENDENCY_CODES for p in py),
        "outputs_valid": sum(p.code not in OUTPUT_CODES for p in py),
        "warnings": dict(Counter(w for p in py for w in p.warnings)),
        "differential": {"cases": len(corpus), "parity_cases": len(PARITY_CASES), "dataset_plans": n,
                         "mutated_invalid_variants": len(mutated),
                         "mutated_rejected_by_cpp": sum(not ok for ok, _, _ in mutated_cpp),
                         "python_cpp_mismatches": len(mismatch), "mismatch_examples": mismatch[:10],
                         "to_json_mismatches": serialization},
        "grammar": {"dataset_plans_accepted": sum(grammar_rows), "dataset_plans": n,
                    "mutated_accepted": sum(grammar_mut), "mutated": len(mutated),
                    "grammar_accepts_validator_invalid": grammar_accepts_invalid},
        "manager_output_tokens_qwen3": describe(tokens) if tokens else None,
        "manager_output_over_256": sum(t > MANAGER_MAX_TOKENS for t in tokens) if tokens else None,
        "system_prompt_tokens_qwen3": prompt_tokens[0] if prompt_tokens else None,
        "request_tokens_qwen3": describe(request_tokens) if request_tokens else None,
        "max_prompt_budget_tokens": (prompt_tokens[0] + max(request_tokens) + 32 + MANAGER_MAX_TOKENS)
        if prompt_tokens and request_tokens else None,
        "status_paths": {s: sum(1 for p, v in zip(plans, py) if p["status"] == s and v.valid)
                         for s in ("ok", "clarify", "unsupported")},
    }
    stats["_tokens"] = dict(zip([r["id"] for r in rows], tokens or []))

    # ---- diversity ----------------------------------------------------------
    requests = [r["request"] for r in rows]
    normalized = [normalize_request(t) for t in requests]
    family_counts = Counter(m["template_family"] for m in meta)
    by_tax = defaultdict(Counter)
    for m in meta:
        by_tax[m["taxonomy"]][m["template_family"]] += 1
    within = {t: {"family": c.most_common(1)[0][0], "share": round(c.most_common(1)[0][1] / sum(c.values()), 4),
                  "families": len(c)} for t, c in by_tax.items()}
    skel_counts = Counter(skeletons)
    stats["diversity"] = {
        "exact_duplicates": n - len(set(requests)),
        "normalized_duplicates": n - len(set(normalized)),
        "unique_request_ratio": round(len(set(normalized)) / n, 4),
        "within_request_dup_4gram_mean": round(statistics.fmean(ngram_duplicate_ratio(t, 4) for t in requests), 4),
        "within_request_dup_8gram_mean": round(statistics.fmean(ngram_duplicate_ratio(t, 8) for t in requests), 4),
        "cross_request_4gram_share": cross_share(requests, 4),
        "cross_request_8gram_share": cross_share(requests, 8),
        "instruction_cross_request_4gram_share": cross_share(skeletons, 4),
        "instruction_cross_request_8gram_share": cross_share(skeletons, 8),
        "template_families_used": len(family_counts),
        "max_family_share_overall": round(family_counts.most_common(1)[0][1] / n, 4),
        "max_family_share_within_taxonomy": within,
        "unique_skeleton_ratio": round(len(skel_counts) / n, 4),
        "top_skeletons": [{"skeleton": s, "count": c} for s, c in skel_counts.most_common(8)],
        "skeleton_type_token_ratio": round(len({w for s in skeletons for w in s.split()}) /
                                           sum(len(s.split()) for s in skeletons), 4),
        "plan_text_duplicate_ratio": round(1 - len({r["plan_text"] for r in rows}) / n, 4),
    }

    # ---- shortcut diagnostics -------------------------------------------------
    groups = defaultdict(lambda: [0, 0])
    keyword_errors = []
    for r in rows:
        pred, gold = keyword_rule(r), signature(r["plan"])
        g = r["metadata"]["taxonomy_group"]
        for key in (g, "overall"):
            groups[key][0] += pred == gold
            groups[key][1] += 1
        if pred != gold:
            keyword_errors.append(r["id"])
    skeleton_len = [len(s) for s in skeletons]
    task_counts = [len(p["tasks"]) for p in plans]
    ok_idx = [i for i, p in enumerate(plans) if p["status"] == "ok"]
    v1 = json.loads(V1_STATS.read_text())["shortcut_diagnostics"] if V1_STATS.exists() else {}
    train = [(skeletons[i], signature(plans[i])) for i in range(n) if meta[i]["split"] == "train"]
    heldout_split = "test"
    test_idx = [i for i in range(n) if meta[i]["split"] == heldout_split]
    test = [(skeletons[i], signature(plans[i])) for i in test_idx]
    lr_pred = logistic_regression(train, test) if train and test else []
    nb_pred = naive_bayes(train, test) if train and test else []
    majority = Counter(y for _, y in train).most_common(1)[0][0] if train else None

    def acc(pred, idx_filter=None):
        pairs = [(p, test[k][1]) for k, p in enumerate(pred) if idx_filter is None or idx_filter(test_idx[k])]
        return {"n": len(pairs), "accuracy": round(sum(a == b for a, b in pairs) / len(pairs), 4) if pairs else None}

    stats["shortcut"] = {
        "label": "status + ordered distinct skills (e.g. ok:search>compare)",
        "keyword_rule": {k: {"correct": v[0], "total": v[1], "accuracy": round(v[0] / v[1], 4)} for k, v in groups.items()},
        "pilot_v1_keyword_rule_normal": v1.get("keyword_rule_accuracy", {}).get("normal", {}).get("accuracy"),
        "corr_instruction_length_task_count": pearson([skeleton_len[i] for i in ok_idx], [task_counts[i] for i in ok_idx]),
        "corr_request_length_task_count": pearson([len(requests[i]) for i in ok_idx], [task_counts[i] for i in ok_idx]),
        "corr_instruction_length_task_count_all_rows": pearson(skeleton_len, task_counts),
        "pilot_v1_corr_instruction_length_task_count": v1.get("corr_instruction_length_vs_task_count"),
        "pilot_v1_corr_request_length_task_count": v1.get("corr_request_length_vs_task_count"),
        "mean_instruction_chars_by_task_count": {k: round(statistics.fmean(skeleton_len[i] for i in range(n)
                                                                           if task_counts[i] == k), 1)
                                                 for k in sorted(set(task_counts))},
        "baselines_train_to_test": {
            "labels": len({y for _, y in train}),
            "majority_class": {"label": majority, **acc([majority] * len(test))},
            "naive_bayes_words": acc(nb_pred),
            "logistic_regression_words_char2": acc(lr_pred),
            "logistic_regression_hard_rows": acc(lr_pred, lambda i: meta[i]["hard_case"]),
            "logistic_regression_normal_rows": acc(lr_pred, lambda i: meta[i]["taxonomy_group"] == "normal"),
        },
    }
    stats["_keyword_errors"] = keyword_errors
    stats["_lr_errors"] = [rows[test_idx[k]]["id"] for k, p in enumerate(lr_pred) if p != test[k][1]]

    # ---- split / leakage --------------------------------------------------------
    leak = leakage(rows)
    train_8 = {tuple(w[i:i + 8]) for s, m in zip(skeletons, meta) if m["split"] == "train"
               for w in [s.split()] for i in range(len(w) - 7)}
    test_8 = [tuple(w[i:i + 8]) for s, m in zip(skeletons, meta) if m["split"] == "test"
              for w in [s.split()] for i in range(len(w) - 7)]
    train_skel = {s for s, m in zip(skeletons, meta) if m["split"] == "train"}
    stats["split"] = {
        "sizes": dict(Counter(m["split"] for m in meta)),
        "ratios": {s: round(sum(m["split"] == s for m in meta) / n, 4) for s in SPLITS},
        "taxonomy_by_split": {s: dict(Counter(m["taxonomy"] for m in meta if m["split"] == s)) for s in SPLITS},
        "status_by_split": {s: dict(Counter(m["status"] for m in meta if m["split"] == s)) for s in SPLITS},
        "skill_by_split": {s: dict(Counter(k for m in meta if m["split"] == s for k in m["skills_used"])) for s in SPLITS},
        "hard_by_split": {s: sum(m["hard_case"] for m in meta if m["split"] == s) for s in SPLITS},
        "families_by_split": {s: len({m["template_family"] for m in meta if m["split"] == s}) for s in SPLITS},
        "leakage": leak,
        "test_instruction_8gram_seen_in_train": round(sum(g in train_8 for g in test_8) / len(test_8), 4) if test_8 else 0,
        "test_skeleton_seen_in_train": sum(1 for s, m in zip(skeletons, meta) if m["split"] == "test" and s in train_skel),
        "sealed_test_rows": sum(m.get("sealed_test", False) for m in meta),
    }

    # ---- provenance / few-shots -------------------------------------------------
    fewshots = [normalize_request(f["user"]) for f in contract.few_shots]
    fewshot_hashes = [simhash(f["user"]) for f in contract.few_shots]
    stats["provenance"] = {
        "rows_missing_fields": [r["id"] for r in rows if any(k not in r["metadata"] for k in REQUIRED_METADATA)],
        "gold_labels": sum(m["status_label"] == "gold" for m in meta),
        "fewshot_exact_overlap": sum(x in fewshots for x in normalized),
        "fewshot_near_duplicates": sum(any(hamming(simhash(t), h) <= NEAR_DUP_HAMMING for h in fewshot_hashes)
                                       for t in requests),
        "generation": {k: log[k] for k in ("generator_version", "seed", "attempts", "rejected_attempts",
                                           "reject_reasons", "shortfall")},
    }
    stats["gates"] = gates(stats, contract)
    stats["thresholds"] = QUALITY_THRESHOLDS
    return stats


def gates(stats: dict, contract) -> dict:
    d, p, dv, sh, sp, pv = (stats["distribution"], stats["plans"], stats["diversity"], stats["shortcut"],
                            stats["split"], stats["provenance"])
    n = stats["rows"]
    hard, quality = {}, {}

    def put(table, key, ok, evidence, review=False):
        table[key] = {"status": "PASS" if ok else ("REVIEW" if review else "FAIL"), "evidence": evidence}

    put(hard, "A_schema_validity", p["schema_valid"] == n and p["canonical_valid"] == n,
        f"runtime-schema {p['schema_valid']}/{n}, canonical {p['canonical_valid']}/{n}")
    put(hard, "B_invalid_skill", p["invalid_skill"] == 0, f"{p['invalid_skill']}")
    put(hard, "C_dependency_source_validity", p["dependency_valid"] == n, f"{p['dependency_valid']}/{n}")
    put(hard, "D_outputs_validity", p["outputs_valid"] == n, f"{p['outputs_valid']}/{n}")
    diff = p["differential"]
    put(hard, "E_python_cpp_differential", diff["python_cpp_mismatches"] == 0 and not diff["to_json_mismatches"]
        and p["grammar"]["dataset_plans_accepted"] == n,
        f"{diff['python_cpp_mismatches']} mismatches over {diff['cases']} cases; to_json mismatches "
        f"{len(diff['to_json_mismatches'])}; grammar accepts {p['grammar']['dataset_plans_accepted']}/{n}")
    put(hard, "F_duplicate_requests", dv["exact_duplicates"] == 0 and dv["normalized_duplicates"] == 0,
        f"exact {dv['exact_duplicates']}, normalized {dv['normalized_duplicates']}")
    lk = sp["leakage"]
    leaks = (len(lk["template_family_across_splits"]) + len(lk["semantic_seed_across_splits"])
             + len(lk["entity_family_across_splits"]) + len(lk["paraphrase_sibling_across_splits"])
             + len(lk["near_duplicate_pairs_cross_split"]) + lk["group_components_spanning_splits"])
    put(hard, "G_split_leakage", leaks == 0,
        f"family {len(lk['template_family_across_splits'])}, seed {len(lk['semantic_seed_across_splits'])}, entity "
        f"{len(lk['entity_family_across_splits'])}, sibling {len(lk['paraphrase_sibling_across_splits'])}, cross-split "
        f"near-dup {len(lk['near_duplicate_pairs_cross_split'])}, spanning components "
        f"{lk['group_components_spanning_splits']}/{lk['group_components']}")
    put(hard, "H_provenance", not pv["rows_missing_fields"] and pv["gold_labels"] == 0,
        f"{len(pv['rows_missing_fields'])} rows missing fields; {pv['gold_labels']} labelled gold")
    over = p["manager_output_over_256"]
    put(hard, "I_manager_output_le_256_tokens", over == 0,
        f"{over} over 256; {p['manager_output_tokens_qwen3']}")
    sp_paths = p["status_paths"]
    put(hard, "J_clarify_unsupported_paths", sp_paths["clarify"] > 0 and sp_paths["unsupported"] > 0,
        f"valid clarify {sp_paths['clarify']}, unsupported {sp_paths['unsupported']} rows through parser/validator/"
        f"grammar; C++ runtime tests cover the executor path")
    present = [s for s in contract.skill_names if d["skill_row_presence"][s] > 0]
    put(hard, "K_all_5_skills", len(present) == 5, f"{d['skill_row_presence']}")
    put(hard, "L_hard_cases_present", d["hard_share"] > 0, f"hard share {d['hard_share']}")
    put(hard, "M_no_blocking_schema_decision", True,
        "none open; non-blocking notes in the report (attachments, partial-support requests)")

    t = QUALITY_THRESHOLDS
    kw = sh["keyword_rule"].get("normal", {}).get("accuracy")
    put(quality, "Q1_keyword_shortcut", kw is not None and kw <= t["Q1_keyword_rule_accuracy_normal_max"],
        f"normal {kw} (v1 {sh['pilot_v1_keyword_rule_normal']}), hard {sh['keyword_rule'].get('hard', {}).get('accuracy')}, "
        f"overall {sh['keyword_rule']['overall']['accuracy']}")
    c1, c2 = sh["corr_instruction_length_task_count"], sh["corr_request_length_task_count"]
    put(quality, "Q2_length_task_correlation",
        c1 is not None and abs(c1) <= t["Q2_abs_corr_instruction_len_task_count_max"]
        and abs(c2) <= t["Q2_abs_corr_request_len_task_count_max"],
        f"instruction {c1} (v1 {sh['pilot_v1_corr_instruction_length_task_count']}), request {c2} "
        f"(v1 {sh['pilot_v1_corr_request_length_task_count']})")
    worst = max(((k, v) for k, v in dv["max_family_share_within_taxonomy"].items() if v["families"] >= 3),
                key=lambda kv: kv[1]["share"])
    put(quality, "Q3_template_dominance", worst[1]["share"] <= t["Q3_max_family_share_within_taxonomy"],
        f"max {worst[1]['share']} ({worst[0]}/{worst[1]['family']}); overall max {dv['max_family_share_overall']}")
    top_tax = max(d["taxonomy"].items(), key=lambda kv: kv[1])
    put(quality, "Q4_taxonomy_dominance", top_tax[1] / n <= t["Q4_max_taxonomy_share"],
        f"largest {top_tax[0]} {top_tax[1] / n:.3f}")
    low = min(d["skill_task_share"].items(), key=lambda kv: kv[1])
    put(quality, "Q5_skill_distribution", low[1] >= t["Q5_min_skill_task_share"], f"{d['skill_task_share']}")
    put(quality, "Q6_topology_balance",
        d["single_task_share"] >= t["Q6_min_single_task_share"] and d["multi_task_share"] >= t["Q6_min_multi_task_share"]
        and d["three_plus_task_share"] >= t["Q6_min_three_plus_task_share"],
        f"single {d['single_task_share']}, multi {d['multi_task_share']}, 3+ {d['three_plus_task_share']}, "
        f"{len(d['topology'])} distinct topologies")
    put(quality, "Q7_multi_source_and_hard",
        d["source_usage"]["rows_with_multi_source_task"] / n >= t["Q7_min_multi_source_task_rows_share"]
        and d["hard_share"] >= t["Q7_min_hard_share"],
        f"multi-source rows {d['source_usage']['rows_with_multi_source_task']}/{n}, hard {d['hard_share']}; "
        f"near-dup pairs {sp['leakage']['near_duplicate_pairs']} (cross-split "
        f"{len(sp['leakage']['near_duplicate_pairs_cross_split'])})")
    hard_pass = all(v["status"] == "PASS" for v in hard.values())
    quality_pass = all(v["status"] == "PASS" for v in quality.values())
    return {"hard": hard, "quality": quality, "hard_pass": hard_pass, "quality_pass": quality_pass,
            "overall": "PASS" if hard_pass and quality_pass else ("REVIEW" if hard_pass else "FAIL")}


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #

def _cell(text: str, limit: int = 100) -> str:
    text = text.replace("\n", "⏎").replace("|", "\\|")
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _plan_cell(plan: dict) -> str:
    if plan["status"] != "ok":
        return f"**{plan['status']}**: {plan['message']}"
    tasks = " → ".join(f"{t['id']}:{t['skill']}[{','.join(map(str, t['sources']))}]({_cell(t['instruction'], 40)})"
                       for t in plan["tasks"])
    return f"{tasks} ⇒ out {plan['outputs']}"


def render_review(rows: list[dict], stats: dict, title: str, random_n: int, per_taxonomy: int,
                  full_listing: bool) -> str:
    by_id = {r["id"]: r for r in rows}
    skel = Counter(skeleton(r) for r in rows)
    near = defaultdict(list)
    for a, b in stats["split"]["leakage"]["near_duplicate_examples"]:
        near[a].append(b)
        near[b].append(a)
    rnd = random.Random(REVIEW_SEED)
    sections = [(f"Deterministic random {random_n} (random.Random({REVIEW_SEED}).sample)", rnd.sample(rows, min(random_n, len(rows))))]
    per_tax = []
    for tax in sorted({r["metadata"]["taxonomy"] for r in rows}):
        members = [r for r in rows if r["metadata"]["taxonomy"] == tax]
        per_tax += random.Random(f"{REVIEW_SEED}:{tax}").sample(members, min(per_taxonomy, len(members)))
    sections.append((f"{per_taxonomy} per taxonomy (seeded sample)", per_tax))
    sections.append(("Longest 10 requests", sorted(rows, key=lambda r: -len(r["request"]))[:10]))
    sections.append(("Most complex 10 plans (tasks, depth, sources)", sorted(
        rows, key=lambda r: (-len(r["plan"]["tasks"]), -r["metadata"]["dependency_depth"], -r["metadata"]["source_count"],
                             r["id"]))[:10]))
    multi = [r for r in rows if any(len(t["sources"]) > 1 for t in r["plan"]["tasks"])]
    sections.append((f"Multi-source plans (first 20 of {len(multi)})", multi[:20]))
    for status in ("clarify", "unsupported"):
        members = [r for r in rows if r["plan"]["status"] == status]
        sample = members if len(members) <= 30 else random.Random(f"{REVIEW_SEED}:{status}").sample(members, 30)
        sections.append((f"{status} ({len(sample)} of {len(members)})", sample))
    hard = [r for r in rows if r["metadata"]["hard_case"]]
    hard_sample = hard if len(hard) <= 60 else random.Random(f"{REVIEW_SEED}:hard").sample(hard, 60)
    sections.append((f"Hard / anti-shortcut ({len(hard_sample)} of {len(hard)})", hard_sample))
    kw_err = [by_id[i] for i in stats["_keyword_errors"]]
    sections.append((f"Keyword baseline errors (first 25 of {len(kw_err)})", kw_err[:25]))
    lr_err = [by_id[i] for i in stats["_lr_errors"]]
    sections.append((f"Logistic-regression baseline errors on test (first 25 of {len(lr_err)})", lr_err[:25]))
    risky = sorted(rows, key=lambda r: (-skel[skeleton(r)], -len(near.get(r["id"], [])), r["id"]))[:15]
    sections.append(("Repetition / near-duplicate risk 15", risky))
    lines = [f"# {title}", "", "All rows are **candidates**, not gold. Every subset below is chosen by a fixed rule stated in "
             "its heading; nothing is hand-picked.", "",
             f"Gate: **{stats['gates']['overall']}** (hard {'PASS' if stats['gates']['hard_pass'] else 'FAIL'}, quality "
             f"{'PASS' if stats['gates']['quality_pass'] else 'REVIEW'}).", "",
             "Plan notation: `id:skill[sources](instruction) ⇒ out [outputs]`.", ""]
    for heading, members in sections:
        lines += [f"## {heading}", "", "| ID | Request | Expected TaskPlan | Taxonomy / family | Split | Notes |",
                  "|---|---|---|---|---|---|"]
        for r in members:
            m = r["metadata"]
            notes = []
            if m["paraphrase_of"]:
                notes.append(f"sibling of {m['paraphrase_of'][-5:]}")
            if r["id"] in near:
                notes.append("near-dup " + ",".join(x[-5:] for x in near[r["id"]]))
            if m["hard_case"]:
                notes.append(m["hard_case_category"])
            tok = stats["_tokens"].get(r["id"])
            if tok:
                notes.append(f"{tok} tok")
            lines.append(f"| {r['id'][-5:]} | {_cell(r['request'])} | {_plan_cell(r['plan'])} | {m['taxonomy']} / "
                         f"{m['template_family']} | {m['split']} | {'; '.join(notes)} |")
        lines.append("")
    if full_listing:
        lines += ["## Full listing", ""]
        for r in rows:
            m = r["metadata"]
            lines += [f"### {r['id']} · {m['taxonomy']} · {m['split']}", "", "```text", r["request"], "```", "",
                      f"`{r['plan_text']}`", "",
                      f"{m['source_type']} · family `{m['template_family']}` · intent `{m['intent']}` · register "
                      f"{m['register']} · verbosity {m['verbosity']} · seed `{m['semantic_seed']}` · entities "
                      f"{', '.join(m['entity_family']) or '—'}", ""]
    return "\n".join(lines) + "\n"


def render_report(stats: dict, title: str) -> str:
    g, d, p, dv, sh, sp = (stats["gates"], stats["distribution"], stats["plans"], stats["diversity"],
                           stats["shortcut"], stats["split"])
    lines = [f"# {title}", "", f"**Gate: {g['overall']}** · hard gates {'PASS' if g['hard_pass'] else 'FAIL'} · "
             f"quality {'PASS' if g['quality_pass'] else 'REVIEW'}", "",
             "Candidate data only; no model was trained. Thresholds were fixed in "
             "`model/manager_v2_analysis.py` before Pilot v2 was generated.", "",
             "## Hard gates", "", "| Gate | Status | Evidence |", "|---|---|---|"]
    lines += [f"| {k} | {v['status']} | {v['evidence']} |" for k, v in g["hard"].items()]
    lines += ["", "## Quality review", "", "| Check | Status | Evidence |", "|---|---|---|"]
    lines += [f"| {k} | {v['status']} | {v['evidence']} |" for k, v in g["quality"].items()]
    lines += ["", "```json", json.dumps(stats["thresholds"], indent=2), "```", "",
              "## Shortcut diagnostics vs Pilot v1", "",
              f"- keyword rule ({sh['label']}): " + json.dumps(sh["keyword_rule"]),
              f"- Pilot v1 keyword rule, normal rows: {sh['pilot_v1_keyword_rule_normal']}",
              f"- corr(instruction length, task count), ok rows: {sh['corr_instruction_length_task_count']} "
              f"(v1 {sh['pilot_v1_corr_instruction_length_task_count']}); all rows "
              f"{sh['corr_instruction_length_task_count_all_rows']}",
              f"- corr(request length, task count), ok rows: {sh['corr_request_length_task_count']} "
              f"(v1 {sh['pilot_v1_corr_request_length_task_count']})",
              f"- mean instruction chars by task count: {sh['mean_instruction_chars_by_task_count']}",
              f"- train→test diagnostic baselines: " + json.dumps(sh["baselines_train_to_test"], ensure_ascii=False), "",
              "## Distribution", "",
              "Taxonomy: " + json.dumps(d["taxonomy"], ensure_ascii=False), "",
              "Status: " + json.dumps(d["status"]), "",
              "Skill task frequency: " + json.dumps(d["skill_task_frequency"]) + " · share " +
              json.dumps(d["skill_task_share"]), "",
              "Task count: " + json.dumps(d["task_count"]) + " · dependency depth: " + json.dumps(d["dependency_depth"]) +
              " · output count: " + json.dumps(d["output_count"]), "",
              "Source usage: " + json.dumps(d["source_usage"]), "",
              "Hard cases: " + json.dumps(d["hard_case_category"]) + f" (share {d['hard_share']})", "",
              "Skill co-occurrence (rows containing both):", "", "| | " + " | ".join(d["skill_cooccurrence_rows"]) + " |",
              "|---" * (len(d["skill_cooccurrence_rows"]) + 1) + "|"]
    lines += [f"| {a} | " + " | ".join(str(v) for v in row.values()) + " |" for a, row in d["skill_cooccurrence_rows"].items()]
    lines += ["", "Top plan topologies:", ""]
    lines += [f"- {c}× `{t}`" for t, c in list(d["topology"].items())[:15]]
    lines += ["", f"Distinct topologies: {len(d['topology'])}", "",
              "Request chars: " + json.dumps(d["request_chars"]), "",
              "Request chars without payload: " + json.dumps(d["request_chars_without_payload"]), "",
              "Instruction chars: " + json.dumps(d["instruction_chars"]), "",
              "## Manager output budget", "",
              f"- plan tokens (Qwen3-0.6B tokenizer): {p['manager_output_tokens_qwen3']}; over 256: {p['manager_output_over_256']}",
              f"- system prompt tokens: {p['system_prompt_tokens_qwen3']}; request tokens: {p['request_tokens_qwen3']}; "
              f"worst-case prompt+output budget ≈ {p['max_prompt_budget_tokens']} of {MANAGER_CONTEXT}", "",
              "## Validator / C++ differential / grammar", "",
              f"- canonical valid {p['canonical_valid']}/{stats['rows']}; invalid codes {p['invalid_codes'] or 'none'}; warnings "
              f"{p['warnings'] or 'none'}",
              f"- differential: " + json.dumps({k: v for k, v in p["differential"].items() if k != "mismatch_examples"}),
              f"- grammar (llama.cpp engine, generated grammar): " + json.dumps(p["grammar"]), "",
              "## Diversity", ""]
    lines += [f"- {k}: {v}" for k, v in dv.items() if k not in ("max_family_share_within_taxonomy", "top_skeletons")]
    lines += ["", "## Split", "", "Sizes: " + json.dumps(sp["sizes"]) + " · ratios " + json.dumps(sp["ratios"]), "",
              "Leakage: " + json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in sp["leakage"].items()
                                        if k != "near_duplicate_examples"}), "",
              f"Test instruction 8-grams seen in train: {sp['test_instruction_8gram_seen_in_train']}; test skeletons seen "
              f"in train: {sp['test_skeleton_seen_in_train']}; sealed test rows: {sp['sealed_test_rows']}", ""]
    for s in SPLITS:
        lines.append(f"- {s}: status {json.dumps(sp['status_by_split'][s])}; hard {sp['hard_by_split'][s]}; families "
                     f"{sp['families_by_split'][s]}")
    lines += ["", "## Provenance", "", json.dumps(stats["provenance"], ensure_ascii=False), ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=("pilot", "5k"), required=True)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.dataset == "pilot":
        path, prefix, title = DATA_DIR / "pilot_v2.jsonl", "pilot_v2", "Manager Pilot v2 (TaskPlan v2)"
        review_file, random_n, per_tax, full = OUT_DIR / "pilot_review.md", 30, 3, True
    else:
        path, prefix, title = DATA_DIR / "manager_v1_5k.jsonl", "5k", "Manager v1 5k candidate dataset"
        review_file, random_n, per_tax, full = OUT_DIR / "5k_review.md", 100, 4, False
    rows = load(path)
    log = json.loads(path.with_suffix(".generation_log.json").read_text(encoding="utf-8"))
    stats = analyse(rows, log, args.dataset)
    if args.dataset == "pilot":
        stats["proceed_to_5k"] = stats["gates"]["hard_pass"] and stats["gates"]["quality_pass"]
    public = {k: v for k, v in stats.items() if not k.startswith("_")}
    (OUT_DIR / f"{prefix}_stats.json").write_text(json.dumps(public, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / f"{prefix}_report.md").write_text(render_report(stats, title), encoding="utf-8")
    review_file.write_text(render_review(rows, stats, f"{title} — human review", random_n, per_tax, full), encoding="utf-8")
    print(f"{args.dataset}: {stats['gates']['overall']}")
    for table in ("hard", "quality"):
        for k, v in stats["gates"][table].items():
            print(f"  {k}: {v['status']} — {v['evidence']}")
    if args.dataset == "pilot":
        print("proceed_to_5k:", stats["proceed_to_5k"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
