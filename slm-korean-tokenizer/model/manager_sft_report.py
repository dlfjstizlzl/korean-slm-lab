#!/usr/bin/env python3
"""Base vs LoRA comparison, error analysis, shortcut analysis and readiness for Manager SFT v1.

Reads only validation artifacts (baseline_val_*.json, best_val_*.json, validation history) and
the validation split. The sealed test split is fingerprinted, never read.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "model"))

from manager_sft_data import load_split, sealed_test_fingerprint  # noqa: E402
from manager_v2_analysis import COMPARE_CUE, EXTRACT_CUE, SEARCH_CUE, SQL_CUE, SUMMARY_CUE, keyword_rule, signature  # noqa: E402

OUT = Path(os.environ.get("MANAGER_SFT_OUT", PROJECT_ROOT / "outputs" / "manager_sft_v1"))
KEY_METRICS = ["json_valid", "schema_valid", "canonical_valid", "grammar_valid", "status_accuracy",
               "task_count_accuracy", "skill_sequence_accuracy", "skill_accuracy_aligned", "sources_exact",
               "outputs_exact", "topology_exact", "structural_exact", "instruction_exact",
               "task_instruction_accuracy_aligned", "canonical_exact", "dependency_edge_precision",
               "dependency_edge_recall", "dependency_edge_f1"]
CATEGORY_ORDER = ["skill:search", "skill:summary", "skill:extract", "skill:compare", "skill:sql",
                  "tasks:0", "tasks:1", "tasks:2", "tasks:3+", "sources:single-source", "sources:multi-source",
                  "outputs:multiple", "status:ok", "status:clarify", "status:unsupported", "normal", "hard",
                  "group:normal", "group:special", "group:hard"]
CUES = {"search": SEARCH_CUE, "summary": SUMMARY_CUE, "extract": EXTRACT_CUE, "compare": COMPARE_CUE, "sql": SQL_CUE}


def load(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar test (binomial on discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def fmt(x):
    return "–" if x is None else f"{x:.4f}" if isinstance(x, float) else str(x)


def table(rows: list[tuple], headers: list[str]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(fmt(c) for c in r) + " |" for r in rows]
    return out


def main() -> int:
    val = load_split("validation")
    by_id = {r["id"]: r for r in val}
    base = {m: load(f"baseline_val_{m}.json") for m in ("raw", "constrained")}
    lora = {m: load(f"best_val_{m}.json") for m in ("raw", "constrained")}
    selection = load("selection.json")
    history = load("validation_history.json")
    config = load("training_config.json")
    criteria = load("readiness_criteria.json")

    # ---------------------------------------------------------------- comparison
    lines = ["# Stage-2 base vs Manager LoRA (validation, 454 rows)", "",
             f"LoRA = best checkpoint, epoch {selection['selected_epoch']} (rule fixed before training: "
             f"{selection['rule']['primary']}; tie-breakers {', '.join(selection['rule']['tie_breakers'])}).",
             "Both use the Runtime system prompt, greedy decoding, max 256 new tokens.", ""]
    stats = {}
    for mode in ("raw", "constrained"):
        b, l = base[mode]["overall"], lora[mode]["overall"]
        lines += [f"## {mode.upper()}", ""]
        lines += table([(k, b[k], l[k], (l[k] - b[k]) if b[k] is not None and l[k] is not None else None)
                        for k in KEY_METRICS], ["metric", "Stage-2 base", "LoRA", "Δ"])
        sr_b, sr_l = base[mode]["status_recall"], lora[mode]["status_recall"]
        lines += ["", "Status recall (per gold status):", ""]
        lines += table([(s, sr_b[s], sr_l[s]) for s in ("ok", "clarify", "unsupported")], ["gold status", "base", "LoRA"])
        pairs = [(rb["structural_exact"], rl["structural_exact"])
                 for rb, rl in zip(base[mode]["records"], lora[mode]["records"])]
        b_only = sum(1 for x, y in pairs if x and not y)
        l_only = sum(1 for x, y in pairs if y and not x)
        p = mcnemar_p(b_only, l_only)
        stats[mode] = {"structural_base": b["structural_exact"], "structural_lora": l["structural_exact"],
                       "only_base_correct": b_only, "only_lora_correct": l_only, "mcnemar_p": p}
        lines += ["", f"Paired structural exact: only base correct {b_only}, only LoRA correct {l_only}, "
                      f"exact McNemar p = {p:.3g}", ""]
        cat_rows = []
        for tag in CATEGORY_ORDER + sorted(t for t in lora[mode]["by_category"] if t.startswith("taxonomy:")):
            if tag not in lora[mode]["by_category"]:
                continue
            bc, lc = base[mode]["by_category"][tag], lora[mode]["by_category"][tag]
            cat_rows.append((tag, lc["n"], bc["schema_valid"], lc["schema_valid"], bc["structural_exact"],
                             lc["structural_exact"], bc["canonical_exact"], lc["canonical_exact"]))
        lines += [f"### {mode} by category", ""]
        lines += table(cat_rows, ["category", "n", "schema base", "schema LoRA", "structural base", "structural LoRA",
                                  "canonical base", "canonical LoRA"])
        lines.append("")
    (OUT / "base_vs_lora.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---------------------------------------------------------------- errors
    error = {}
    md = ["# Error analysis — best LoRA on validation", "",
          "Failure categories are assigned automatically (`manager_sft_eval.failure_categories`); a row can carry "
          "several. Rates are over all 454 rows. Examples are the first rows in validation order for each category "
          "(no hand selection).", ""]
    for mode in ("constrained", "raw"):
        records = lora[mode]["records"]
        counts = Counter(f for r in records for f in r["failures"])
        failed = sum(1 for r in records if r["failures"])
        examples = defaultdict(list)
        for r in records:
            for f in r["failures"]:
                if len(examples[f]) < 3:
                    row = by_id[r["id"]]
                    examples[f].append({"id": r["id"], "taxonomy": row["metadata"]["taxonomy"],
                                        "request": row["request"], "gold": row["plan_text"], "prediction": r["output"]})
        status_confusion = Counter((by_id[r["id"]]["plan"]["status"], r["pred_status"]) for r in records)
        error[mode] = {"rows": len(records), "rows_with_failure": failed, "canonical_exact": lora[mode]["overall"]["canonical_exact"],
                       "categories": {k: {"count": v, "rate": round(v / len(records), 4)} for k, v in counts.most_common()},
                       "status_confusion": {f"{g}->{p}": n for (g, p), n in sorted(status_confusion.items())},
                       "examples": examples}
        md += [f"## {mode}", "", f"Rows with any failure (not canonical-exact): {failed}/{len(records)}", ""]
        md += table([(k, v["count"], v["rate"]) for k, v in error[mode]["categories"].items()], ["category", "count", "rate"])
        md += ["", "Status confusion (gold → predicted): " +
               ", ".join(f"{k}: {v}" for k, v in error[mode]["status_confusion"].items()), ""]
        if mode == "constrained":
            for cat, items in examples.items():
                md += [f"### {cat}", ""]
                for e in items:
                    md += [f"- `{e['id']}` ({e['taxonomy']})", f"  - request: {e['request'][:200].replace(chr(10), ' ⏎ ')}",
                           f"  - gold: `{e['gold']}`", f"  - pred: `{e['prediction'].strip()[:400]}`"]
                md.append("")
    (OUT / "error_analysis.json").write_text(json.dumps(error, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ---------------------------------------------------------------- shortcut analysis
    shortcut = {}
    recs = {r["id"]: r for r in lora["constrained"]["records"]}
    base_recs = {r["id"]: r for r in base["constrained"]["records"]}
    for group in ("normal", "special", "hard"):
        ids = [r["id"] for r in val if r["metadata"]["taxonomy_group"] == group]
        shortcut[f"structural_{group}"] = {
            "n": len(ids), "base": round(sum(base_recs[i]["structural_exact"] for i in ids) / len(ids), 4),
            "lora": round(sum(recs[i]["structural_exact"] for i in ids) / len(ids), 4)}
    cue = {}
    for skill, pattern in CUES.items():
        rows = [r for r in val if any(t["skill"] == skill for t in r["plan"]["tasks"])]
        with_cue = [r for r in rows if pattern.search(r["request"])]
        without = [r for r in rows if not pattern.search(r["request"])]
        def acc(rs):
            return round(sum(skill in [t["skill"] for t in (recs[r["id"]]["pred_plan"] or {"tasks": []})["tasks"]]
                             for r in rs) / len(rs), 4) if rs else None
        false_rows = [r for r in val if not any(t["skill"] == skill for t in r["plan"]["tasks"]) and pattern.search(r["request"])]
        false_rate = round(sum(skill in [t["skill"] for t in (recs[r["id"]]["pred_plan"] or {"tasks": []})["tasks"]]
                               for r in false_rows) / len(false_rows), 4) if false_rows else None
        cue[skill] = {"gold_has_skill_with_cue": {"n": len(with_cue), "lora_predicts_skill": acc(with_cue)},
                      "gold_has_skill_without_cue": {"n": len(without), "lora_predicts_skill": acc(without)},
                      "cue_present_but_skill_absent": {"n": len(false_rows), "lora_predicts_skill_anyway": false_rate}}
    shortcut["keyword_cue_dependence"] = cue
    wrong = [r for r in val if not recs[r["id"]]["structural_exact"] and recs[r["id"]]["pred_plan"]]
    follows_rule = sum(signature(recs[r["id"]]["pred_plan"]) == keyword_rule(r) for r in wrong)
    shortcut["errors_matching_keyword_rule"] = {"wrong_valid_predictions": len(wrong), "equal_to_keyword_rule_output": follows_rule}
    shortcut["keyword_rule_accuracy_on_validation"] = round(sum(keyword_rule(r) == signature(r["plan"]) for r in val) / len(val), 4)
    shortcut["lora_signature_accuracy_on_validation"] = round(sum(bool(recs[r["id"]]["pred_plan"]) and
                                                                  signature(recs[r["id"]]["pred_plan"]) == signature(r["plan"])
                                                                  for r in val) / len(val), 4)
    hard_tax = defaultdict(lambda: [0, 0, 0])
    for r in val:
        if r["metadata"]["hard_case"]:
            t = hard_tax[r["metadata"]["taxonomy"]]
            t[0] += 1
            t[1] += base_recs[r["id"]]["structural_exact"]
            t[2] += recs[r["id"]]["structural_exact"]
    shortcut["hard_taxonomies"] = {k: {"n": n, "base": round(b / n, 4), "lora": round(l / n, 4)} for k, (n, b, l) in hard_tax.items()}
    error["shortcut"] = shortcut
    (OUT / "error_analysis.json").write_text(json.dumps(error, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md += ["## Shortcut analysis (constrained)", "", "```json", json.dumps(shortcut, ensure_ascii=False, indent=2), "```", ""]
    (OUT / "error_analysis.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---------------------------------------------------------------- samples
    import random

    picks = random.Random(0).sample(val, 30)
    samples = []
    for r in picks:
        entry = {"id": r["id"], "taxonomy": r["metadata"]["taxonomy"], "request": r["request"], "gold": r["plan_text"]}
        for name, source in (("base", base), ("lora", lora)):
            for mode in ("raw", "constrained"):
                rec = next(x for x in source[mode]["records"] if x["id"] == r["id"])
                entry[f"{name}_{mode}"] = rec["output"]
                entry[f"{name}_{mode}_structural_exact"] = rec["structural_exact"]
        samples.append(entry)
    (OUT / "sample_generations.json").write_text(json.dumps({"selection": "random.Random(0).sample(validation, 30)",
                                                             "samples": samples}, ensure_ascii=False, indent=2) + "\n",
                                                 encoding="utf-8")

    # ---------------------------------------------------------------- readiness
    lc, bc = lora["constrained"]["overall"], base["constrained"]["overall"]
    lr_, br = lora["raw"]["overall"], base["raw"]["overall"]
    outputs = Counter(r["output"].strip() for r in lora["constrained"]["records"])
    top_share = outputs.most_common(1)[0][1] / len(val)
    finite = all(math.isfinite(h["val_loss"]) and math.isfinite(h["train_loss_mean"]) for h in history)
    with (OUT / "training_history.csv").open() as handle:
        steps = list(csv.DictReader(handle))
    finite = finite and all(math.isfinite(float(s["train_loss"])) and math.isfinite(float(s["grad_norm"])) for s in steps)
    hard = shortcut["structural_hard"]
    checks = {
        "C1_pipeline_ok": len(history) == config["epochs"] and finite,
        "C3_meaningful_improvement": (lc["structural_exact"] - bc["structural_exact"] >= 0.20
                                      and stats["constrained"]["mcnemar_p"] < 0.01),
        "C4_schema_usable": lc["schema_valid"] >= 0.99 and lr_["schema_valid"] >= 0.95,
        "C5_structural_clearly_better": lc["structural_exact"] > bc["structural_exact"] and lr_["structural_exact"] > br["structural_exact"],
        "C6_hard_signal": hard["lora"] - hard["base"] >= 0.10,
        "C7_no_collapse": top_share <= 0.10 and all(v > 0 for v in lora["constrained"]["status_recall"].values()),
        "C8_sealed_test_unused": sealed_test_fingerprint()["matches_recorded_sha256"],
    }
    readiness = {"criteria_file": "readiness_criteria.json", "declared_at": criteria["declared_at"],
                 "checks": checks, "evidence": {
                     "constrained_structural": [bc["structural_exact"], lc["structural_exact"]],
                     "raw_structural": [br["structural_exact"], lr_["structural_exact"]],
                     "schema_valid": {"constrained": lc["schema_valid"], "raw": lr_["schema_valid"]},
                     "mcnemar": stats, "hard": hard, "top_output_share": round(top_share, 4),
                     "status_recall": lora["constrained"]["status_recall"], "finite": finite},
                 "note": "C2 (adapter save/reload) and C9 (tests) are verified by the test suite and recorded in readiness_report.md"}
    (OUT / "readiness_checks.json").write_text(json.dumps(readiness, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(readiness, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
