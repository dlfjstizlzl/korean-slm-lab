#!/usr/bin/env python3
"""Analysis rules for the one-time Manager-SLM v1 sealed-test evaluation.

Everything here was written and hashed into `pre_test_snapshot.json` before the
sealed test was read. It consumes scored records (as produced by
`manager_sft_eval.evaluate`) and never defines a new headline metric:

* 95% confidence intervals: Wilson score interval for proportions; Newcombe's
  hybrid score interval (method 10) for the difference of two independent
  proportions (validation vs test). No other CI method is used.
* paired Base vs LoRA comparison: both / base-only / LoRA-only / neither and the
  exact two-sided McNemar test.
* supplementary failure tags (duplicate source, clarify/unsupported miss,
  entity corruption) and a rule-based instruction-fidelity classification of
  instruction mismatches (A-G, see `classify_instruction`).
* known-weakness probes, defined by metadata/regex rules fixed below.
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "model"))

from manager_v2_analysis import COMPARE_CUE, DEICTIC_CUE, SEARCH_CUE  # noqa: E402

Z95 = 1.959963984540054
ANALYSIS_PLAN = {
    "confidence_interval": "Wilson score interval, 95% (z=1.95996), for every reported proportion",
    "difference_interval": "Newcombe hybrid score interval (method 10) from the two Wilson intervals, 95%",
    "paired_test": "exact two-sided McNemar (binomial on discordant pairs)",
    "instruction_fidelity": "rule-based classes A-G (manager_final_analysis.classify_instruction); per task pair "
                            "F (search subject dropped/corrupted) is checked first, then C (similar novel string, "
                            "or request word replaced by a novel string), D, E, G, B; A = spacing/punctuation only; "
                            "row class = most severe pair in order F > C > D > E > G > B > A; entity corruption = "
                            "any C pair or F pair with a corrupted/substituted string",
    "examples": "first rows in sealed-test file order per category (no hand selection)",
    "weakness_probes": {
        "negated_compare": "request matches NEGATED_COMPARE regex",
        "compare_cue_without_compare": "COMPARE_CUE matches the request and the gold plan has no compare task",
        "search_without_cue": "gold plan has a search task and SEARCH_CUE does not match the request",
        "user_search_compare": "metadata.taxonomy == user_search_compare",
        "sql_sql_compare": "metadata.taxonomy == sql_sql_compare",
        "clarify_all": "gold status == clarify",
        "clarify_deictic": "gold status == clarify and the request matches DEICTIC_CUE or EXTRA_DEICTIC",
        "search_entity_corruption": "search tasks aligned by position with the same skill whose instruction has a "
                                    "C-class (entity corruption) difference",
    },
}

NEGATED_COMPARE = re.compile(r"비교\S*\s*(필요\s*없|안\s*해|하지\s*말|하지\s*않|말고|없이)")
EXTRA_DEICTIC = re.compile(r"저거|그 |저번|아까|거기|걔")
JOSA = ("에서는", "에서", "으로", "와의", "과의", "의", "와", "과", "를", "을", "은", "는", "이", "가", "에", "로", "도", "만", "및")
PUNCT = re.compile(r"[\s\.,·/\-\(\)\[\]'\"“”‘’!?~:;]+")
FIDELITY_CLASSES = {
    "A": "harmless wording difference (spacing/punctuation only)",
    "B": "semantically equivalent paraphrase (request content kept, only generic words differ)",
    "C": "entity/name corruption (a new string resembling a request/gold entity, or replacing request content)",
    "D": "missing constraint (request content in the gold instruction is dropped)",
    "E": "added constraint (request content absent from the gold instruction is added)",
    "F": "wrong retrieval target (search task whose subject is dropped or corrupted)",
    "G": "other",
}
SEVERITY = "FCDEGBA"


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #

def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def newcombe_diff(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float, float]:
    """Difference p2 - p1 with Newcombe's hybrid score 95% interval."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p2 - p1
    lower = d - math.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2)
    upper = d + math.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2)
    return (d, lower, upper)


def mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired(base: list[bool], lora: list[bool]) -> dict:
    if len(base) != len(lora):
        raise ValueError("paired comparison needs the same rows")
    both = sum(1 for x, y in zip(base, lora) if x and y)
    base_only = sum(1 for x, y in zip(base, lora) if x and not y)
    lora_only = sum(1 for x, y in zip(base, lora) if y and not x)
    neither = len(base) - both - base_only - lora_only
    return {"n": len(base), "both_correct": both, "base_only_correct": base_only, "lora_only_correct": lora_only,
            "both_wrong": neither, "mcnemar_p": mcnemar_p(base_only, lora_only)}


def rate_ci(flags: list[bool]) -> dict:
    k, n = sum(bool(f) for f in flags), len(flags)
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "ci95": [round(lo, 4), round(hi, 4)]}


# --------------------------------------------------------------------------- #
# instruction fidelity
# --------------------------------------------------------------------------- #

def norm(text: str) -> str:
    return PUNCT.sub("", text)


def stems(token: str) -> list[str]:
    out = [norm(token)]
    for josa in JOSA:
        if out[0].endswith(josa) and len(out[0]) > len(josa):
            out.append(out[0][: -len(josa)])
    return [s for s in out if s]


def contained(token: str, text_norm: str) -> bool:
    return any(s in text_norm for s in stems(token))


def generic_vocabulary(rows: list[dict], min_count: int = 5) -> set[str]:
    """Instruction/message words the gold labels add on their own (not copied from the request), from train."""
    counts = Counter()
    for row in rows:
        request = norm(row["request"])
        texts = [t["instruction"] for t in row["plan"]["tasks"]] + [row["plan"].get("message") or ""]
        for text in texts:
            for token in text.split():
                if not contained(token, request):
                    counts[norm(token)] += 1
    return {t for t, c in counts.items() if c >= min_count}


def _similar(a: str, b: str) -> bool:
    a, b = norm(a), norm(b)
    if not a or not b or a == b:
        return False
    if a.lower() == b.lower():
        return True
    return SequenceMatcher(None, a, b).ratio() >= 0.5 and len(set(a) & set(b)) >= 2


def classify_instruction(gold: str, pred: str, request: str, skill: str, generic: set[str]) -> dict | None:
    """One task/message pair -> fidelity class (None when identical)."""
    if gold == pred:
        return None
    if norm(gold) == norm(pred):
        return {"class": "A"}
    request_norm, gold_norm, pred_norm = norm(request), norm(gold), norm(pred)
    gold_tokens, pred_tokens = gold.split(), pred.split()
    from_request = [t for t in gold_tokens if contained(t, request_norm)]
    missing = [t for t in from_request if not contained(t, pred_norm)]
    added = [t for t in pred_tokens if contained(t, request_norm) and not contained(t, gold_norm)]
    novel = [t for t in pred_tokens if not contained(t, request_norm) and not contained(t, gold_norm)
             and not any(s in generic for s in stems(t))]
    anchors = gold_tokens + request.split()
    corrupted = [t for t in novel if any(_similar(t, a) for a in anchors)]
    # a request-derived gold word replaced by a string found in neither the request nor the gold
    substituted = bool(missing) and bool(novel)
    detail = {"missing": missing, "added": added, "novel": novel, "corrupted": corrupted}
    subject_lost = bool(from_request) and from_request[0] in missing
    if skill == "search" and subject_lost:
        return {"class": "F", **detail, "subject": from_request[0],
                "entity_corruption": bool(corrupted) or substituted}
    if corrupted or substituted:
        return {"class": "C", **detail}
    if missing:
        return {"class": "D", **detail}
    if added:
        return {"class": "E", **detail}
    if not pred_tokens:
        return {"class": "G", **detail}
    return {"class": "B", **detail}


def fidelity(row: dict, record: dict, generic: set[str]) -> dict | None:
    """Row-level instruction fidelity for predictions with a valid plan and the gold skill sequence."""
    pred = record.get("parsed_plan") if "parsed_plan" in record else record.get("pred_plan")
    gold = row["plan"]
    if not pred or record["canonical_exact"] or pred["status"] != gold["status"]:
        return None
    pairs = []
    for g, p in zip(gold["tasks"], pred["tasks"]):
        if g["skill"] == p["skill"]:
            result = classify_instruction(g["instruction"], p["instruction"], row["request"], g["skill"], generic)
            if result:
                pairs.append({"task": g["id"], "skill": g["skill"], "gold": g["instruction"],
                              "pred": p["instruction"], **result})
    if gold.get("message") != pred.get("message") and gold.get("message") and pred.get("message"):
        result = classify_instruction(gold["message"], pred["message"], row["request"], "message", generic)
        if result:
            pairs.append({"task": "message", "skill": "message", "gold": gold["message"], "pred": pred["message"],
                          **result})
    if not pairs:
        return None
    worst = min(pairs, key=lambda p: SEVERITY.index(p["class"]))["class"]
    return {"class": worst, "entity_corruption": any(p["class"] == "C" or p.get("entity_corruption") for p in pairs),
            "pairs": pairs}


# --------------------------------------------------------------------------- #
# supplementary failure tags and probes
# --------------------------------------------------------------------------- #

def duplicate_source(output: str) -> bool:
    try:
        plan = json.loads(output.strip())
    except (ValueError, TypeError):
        return False
    tasks = plan.get("tasks") if isinstance(plan, dict) else None
    if not isinstance(tasks, list):
        return False
    for task in tasks:
        sources = task.get("sources") if isinstance(task, dict) else None
        if isinstance(sources, list):
            keys = [json.dumps(s) for s in sources]
            if len(keys) != len(set(keys)):
                return True
    return False


def failure_tags(row: dict, record: dict, fid: dict | None) -> list[str]:
    tags = list(record["failures"])
    gold = row["plan"]["status"]
    if gold == "clarify" and record["pred_status"] != "clarify":
        tags.append("clarify_miss")
    if gold == "unsupported" and record["pred_status"] != "unsupported":
        tags.append("unsupported_miss")
    if duplicate_source(record["output"]):
        tags.append("duplicate_source")
    if fid and fid["entity_corruption"]:
        tags.append("entity_corruption")
    return tags


def probes(row: dict) -> list[str]:
    plan, request, meta = row["plan"], row["request"], row["metadata"]
    skills = {t["skill"] for t in plan["tasks"]}
    out = []
    if NEGATED_COMPARE.search(request):
        out.append("negated_compare")
    if COMPARE_CUE.search(request) and "compare" not in skills:
        out.append("compare_cue_without_compare")
    if "search" in skills and not SEARCH_CUE.search(request):
        out.append("search_without_cue")
    if meta["taxonomy"] in ("user_search_compare", "sql_sql_compare"):
        out.append(meta["taxonomy"])
    if plan["status"] == "clarify":
        out.append("clarify_all")
        if DEICTIC_CUE.search(request) or EXTRA_DEICTIC.search(request):
            out.append("clarify_deictic")
    return out
