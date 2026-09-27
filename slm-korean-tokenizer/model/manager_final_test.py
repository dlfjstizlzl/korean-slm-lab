#!/usr/bin/env python3
"""One-time FINAL sealed-test evaluation of Manager-SLM v1 (Stage-2 base vs LoRA epoch 2).

    python model/manager_final_test.py snapshot --tests-status FILE   # provenance, before the test is read
    python model/manager_final_test.py run --consume-sealed-test      # the only sealed-test read
    python model/manager_final_test.py report                         # analysis from saved predictions only

`run` is the single entrypoint allowed to unlock the sealed test
(`load_split("test", allow_sealed_test=True)`). It reuses the validation
pipeline unchanged: `manager_sft_eval.generate` (same prompt, rendering,
greedy decoding, 256-token budget, Runtime grammar) and
`manager_sft_eval.evaluate` (same parser, validator and metrics). The only
difference from validation is the input split.

After the four evaluations it writes `SEALED_TEST_CONSUMED.json`; from then on
`load_split("test", ...)` refuses unless the consumed-test override is set, which no
code path here sets (`--override-consumed` exists but is not used).
`report` never reads the test file: it works from the saved predictions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import stat
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "model"))

import manager_final_analysis as fa  # noqa: E402
import manager_sft_data as data  # noqa: E402
from manager_sft_data import BASE_MODEL, BASE_MODEL_NAME, SPLIT_FILES, sha256_file  # noqa: E402
from manager_sft_eval import MAX_NEW_TOKENS, categories, evaluate, generate  # noqa: E402

OUT = Path(os.environ.get("MANAGER_FINAL_OUT", PROJECT_ROOT / "outputs" / "manager_final_test_v1"))
SFT = PROJECT_ROOT / "outputs" / "manager_sft_v1"
ADAPTER = SFT / "best_adapter"
EXPECTED_ADAPTER_SHA256 = "f1df598d22032169b7fd91ed6aef04f0b3e57a3bab9fe5c5f438352f25cbf7b7"
MODES = [("base", "raw"), ("base", "constrained"), ("lora", "raw"), ("lora", "constrained")]
EVAL_CONFIG = {
    "order": [f"{m}_{d}" for m, d in MODES],
    "split": "sealed test (data/manager/manager_v1_5k_test.jsonl), 461 rows, each evaluated once per mode",
    "pipeline": "manager_sft_eval.generate + manager_sft_eval.evaluate (identical to validation)",
    "decoding": "greedy (do_sample=False), temperature/top-p/top-k unset",
    "max_new_tokens": MAX_NEW_TOKENS,
    "batch_size": 16,
    "padding": "left",
    "eos": ["<|im_end|>", "<|endoftext|>"],
    "dtype": "float32",
    "raw": "no grammar",
    "constrained": "Runtime ManagerClient::json_grammar() via slm_taskplan_tool --grammar-session, greedy over "
                   "allowed tokens",
    "system_prompt": "ManagerClient::system_prompt() via slm_taskplan_tool --contract (unchanged)",
    "base": f"{BASE_MODEL_NAME} (outputs/models/qwen3-0.6b-ko32k-stage2-signal2678)",
    "lora": "outputs/manager_sft_v1/best_adapter (epoch 2), loaded unmerged",
}
KEY_METRICS = ["json_valid", "schema_valid", "canonical_valid", "grammar_valid", "status_accuracy",
               "task_count_accuracy", "skill_sequence_accuracy", "skill_accuracy_aligned", "sources_exact",
               "outputs_exact", "topology_exact", "structural_exact", "instruction_exact",
               "task_instruction_accuracy_aligned", "canonical_exact", "dependency_edge_precision",
               "dependency_edge_recall", "dependency_edge_f1"]
# record-level flag behind each proportion metric (aggregate-only metrics have none)
FLAG = {"json_valid": "json_valid", "schema_valid": "schema_valid", "canonical_valid": "canonical_valid",
        "grammar_valid": "grammar_valid", "status_accuracy": "status_correct",
        "task_count_accuracy": "task_count_correct", "skill_sequence_accuracy": "skill_sequence_correct",
        "sources_exact": "sources_exact", "outputs_exact": "outputs_exact", "topology_exact": "topology_exact",
        "structural_exact": "structural_exact", "instruction_exact": "instruction_exact",
        "canonical_exact": "canonical_exact"}
SOURCE_FILES = ["model/manager_final_test.py", "model/manager_final_analysis.py", "model/manager_sft_eval.py",
                "model/manager_sft_data.py", "model/manager_sft_train.py", "model/manager_lora.py",
                "model/manager_v2_validator.py", "model/manager_v2_registry.py", "model/manager_v2_analysis.py",
                "tests/test_manager_sft.py", "tests/test_manager_final_test.py"]


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, obj, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise FileExistsError(f"{path} is immutable and already exists")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def freeze(path: Path) -> None:
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def count_lines(path: Path) -> int:
    """Row count without parsing any row."""
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


# --------------------------------------------------------------------------- #
# snapshot
# --------------------------------------------------------------------------- #

def critical_hashes() -> dict:
    from manager_sft_train import provenance

    prov = provenance()
    return {
        "base_model_weights_sha256": prov["base_model_weights_sha256"],
        "base_model_config_sha256": sha256_file(BASE_MODEL / "config.json"),
        "tokenizer_json_sha256": prov["tokenizer_json_sha256"],
        "adapter_safetensors_sha256": sha256_file(ADAPTER / "adapter_model.safetensors"),
        "adapter_config_sha256": sha256_file(ADAPTER / "adapter_config.json"),
        "system_prompt_sha256": prov["system_prompt_sha256"],
        "grammar_sha256": prov["grammar_sha256"],
        "train_sha256": prov["train_sha256"],
        "validation_sha256": prov["validation_sha256"],
        "sealed_test_sha256": sha256_file(SPLIT_FILES["test"]),
        "runtime_contract_sources": prov["runtime_contract_sources"],
    }


def cmd_snapshot(args) -> None:
    path = OUT / "pre_test_snapshot.json"
    if path.exists():
        raise SystemExit(f"{path} already exists and must not change")
    if data.CONSUMED_MARKER.exists():
        raise SystemExit("sealed test already consumed")
    test_path = SPLIT_FILES["test"]
    hashes = critical_hashes()
    if hashes["sealed_test_sha256"] != data.SEALED_TEST_SHA256:
        raise SystemExit("sealed test hash mismatch -> FINAL_EVALUATION_INVALID")
    if hashes["adapter_safetensors_sha256"] != EXPECTED_ADAPTER_SHA256:
        raise SystemExit("adapter hash differs from the selected epoch-2 adapter")
    selection = read_json(SFT / "selection.json")
    training = read_json(SFT / "training_config.json")
    snapshot = {
        "timestamp": now(),
        "purpose": "one-time final sealed-test evaluation of Manager-SLM v1; recorded before the test is read",
        "base_model": {"name": BASE_MODEL_NAME, "path": str(BASE_MODEL.relative_to(PROJECT_ROOT))},
        "adapter": {"path": str(ADAPTER.relative_to(PROJECT_ROOT)), "selected_epoch": selection["selected_epoch"],
                    "equals_checkpoint_epoch2": sha256_file(ADAPTER / "adapter_model.safetensors")
                    == sha256_file(SFT / "checkpoints" / "epoch2" / "adapter_model.safetensors"),
                    "merged": False},
        "hashes": hashes,
        "sealed_test": {"path": str(test_path.relative_to(PROJECT_ROOT)), "exists": test_path.is_file(),
                        "rows": count_lines(test_path), "expected_rows": data.EXPECTED_ROWS["test"],
                        "sha256_matches_recorded": hashes["sealed_test_sha256"] == data.SEALED_TEST_SHA256,
                        "previously_read": False,
                        "consumed_marker_present": data.CONSUMED_MARKER.exists()},
        "selection_rule": selection["rule"],
        "lora_config": training["lora"],
        "training_config_sha256": sha256_file(SFT / "training_config.json"),
        "evaluation_config": EVAL_CONFIG,
        "analysis_plan": fa.ANALYSIS_PLAN,
        "validation_reference": {mode: read_json(SFT / f"best_val_{mode}.json")["overall"] for mode in ("raw", "constrained")},
        "source_sha256": {f: sha256_file(PROJECT_ROOT / f) for f in SOURCE_FILES if (PROJECT_ROOT / f).is_file()},
        "taskplan_tool_binary_sha256": sha256_file(PROJECT_ROOT.parent / "slm-runtime" / "build" / "slm_taskplan_tool"),
        "git": "not a git repository; source SHA-256 recorded instead",
        "tests_status": Path(args.tests_status).read_text(encoding="utf-8").strip() if args.tests_status else None,
    }
    if snapshot["sealed_test"]["rows"] != data.EXPECTED_ROWS["test"]:
        raise SystemExit("sealed test row count mismatch -> FINAL_EVALUATION_INVALID")
    write_json(path, snapshot, exclusive=True)
    freeze(path)
    print(json.dumps({k: snapshot[k] for k in ("timestamp", "sealed_test")}, indent=1))


# --------------------------------------------------------------------------- #
# run (the only sealed-test read)
# --------------------------------------------------------------------------- #

def leakage_check(test_rows: list[dict]) -> dict:
    train, val = data.load_split("train"), data.load_split("validation")
    seen = train + val
    result = {}
    for key, get in (("id", lambda r: r["id"]), ("request", lambda r: r["request"]),
                     ("split_group", lambda r: r["metadata"].get("split_group")),
                     ("semantic_seed", lambda r: r["metadata"].get("semantic_seed"))):
        known = {get(r) for r in seen if get(r) is not None}
        result[key] = sorted({str(get(r)) for r in test_rows if get(r) is not None and get(r) in known})[:20]
    result["leak_free"] = not any(result[k] for k in ("id", "request", "split_group", "semantic_seed"))
    return result


def serialize(row: dict, record: dict, model: str, mode: str) -> dict:
    keep = ("json_valid", "schema_valid", "canonical_valid", "grammar_valid", "error_code", "status_correct",
            "pred_status", "task_count_correct", "skill_sequence_correct", "sources_exact", "outputs_exact",
            "topology_exact", "structural_exact", "instruction_exact", "canonical_exact", "aligned_skill_matches",
            "aligned_instruction_matches", "gold_tasks", "edges_tp", "edges_pred", "edges_gold", "failures")
    meta = row["metadata"]
    return {"id": row["id"], "model": model, "mode": mode, "request": row["request"], "target": row["plan_text"],
            "target_plan": row["plan"], "generated": record["output"], "parsed_plan": record["pred_plan"],
            **{k: record[k] for k in keep}, "categories": categories(row),
            "metadata": {k: meta.get(k) for k in ("taxonomy", "taxonomy_group", "hard_case", "intent",
                                                  "template_family", "has_user_payload", "status")}}


def cmd_run(args) -> None:
    if not args.consume_sealed_test:
        raise SystemExit("refusing: pass --consume-sealed-test to run the one-time final evaluation")
    snapshot_path = OUT / "pre_test_snapshot.json"
    if not snapshot_path.exists():
        raise SystemExit("write pre_test_snapshot.json first")
    snapshot = read_json(snapshot_path)
    current = critical_hashes()
    changed = [k for k in current if current[k] != snapshot["hashes"][k]]
    if changed:
        raise SystemExit(f"provenance changed since the snapshot: {changed} -> FINAL_EVALUATION_INVALID")
    rows = data.load_split("test", allow_sealed_test=True, allow_consumed_test=args.override_consumed)
    ids = [r["id"] for r in rows]
    integrity = {"rows": len(rows), "unique_ids": len(set(ids)), "all_marked_sealed": all(r["metadata"].get("sealed_test")
                                                                                          for r in rows),
                 "sha256": sha256_file(SPLIT_FILES["test"]), "leakage": leakage_check(rows), "checked_at": now()}
    write_json(OUT / "test_integrity.json", integrity)
    if not (integrity["rows"] == integrity["unique_ids"] == data.EXPECTED_ROWS["test"] and integrity["all_marked_sealed"]
            and integrity["leakage"]["leak_free"]):
        raise SystemExit("sealed test integrity/leakage check failed -> FINAL_EVALUATION_INVALID")

    import torch

    import manager_lora as lora
    from manager_sft_train import empty_cache, load_base, pick_device

    device = pick_device()
    model, tokenizer = load_base(device)
    model.eval()
    requests = [r["request"] for r in rows]
    pred_dir = OUT / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    adapter_loaded = False
    for name, mode in MODES:
        pred_path = pred_dir / f"{name}_{mode}.jsonl"
        metrics_path = OUT / f"{name}_{mode}_metrics.json"
        if pred_path.exists() and metrics_path.exists():
            print(f"[{name}_{mode}] already evaluated in this run; kept unchanged", flush=True)
            continue
        if pred_path.exists() or metrics_path.exists():
            raise SystemExit(f"partial artifacts for {name}_{mode}; refusing to overwrite")
        if name == "lora" and not adapter_loaded:
            lora.load_adapter(model, ADAPTER)
            model.to(device).eval()
            adapter_loaded = True
        outputs, timing = generate(model, tokenizer, requests, constrained=(mode == "constrained"),
                                   batch_size=EVAL_CONFIG["batch_size"], device=device)
        result = evaluate(rows, outputs)
        if len(result["records"]) != len(rows) or [r["id"] for r in result["records"]] != ids:
            raise SystemExit("evaluation did not cover every row exactly once -> FINAL_EVALUATION_INVALID")
        lines = [json.dumps(serialize(row, rec, name, mode), ensure_ascii=False, sort_keys=True)
                 for row, rec in zip(rows, result["records"])]
        tmp = pred_path.with_suffix(".jsonl.tmp")
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(tmp, pred_path)
        freeze(pred_path)
        metrics = {"model": BASE_MODEL_NAME if name == "base" else f"{BASE_MODEL_NAME} + Manager LoRA epoch 2",
                   "mode": mode, "split": "sealed_test", "device": device, "evaluated_at": now(),
                   "adapter_sha256": EXPECTED_ADAPTER_SHA256 if name == "lora" else None,
                   "overall": result["overall"], "status_recall": result["status_recall"],
                   "by_category": result["by_category"], "failure_counts": result["failure_counts"], "timing": timing}
        write_json(metrics_path, metrics, exclusive=True)
        freeze(metrics_path)
        o = result["overall"]
        print(f"[{name}_{mode}] schema {o['schema_valid']} structural {o['structural_exact']} canonical "
              f"{o['canonical_exact']} ({timing['seconds']}s)", flush=True)
        empty_cache(device)

    prediction_hashes = {p.name: sha256_file(p) for p in sorted(pred_dir.glob("*.jsonl"))}
    metric_hashes = {f"{n}_{m}_metrics.json": sha256_file(OUT / f"{n}_{m}_metrics.json") for n, m in MODES}
    write_json(OUT / "prediction_hashes.json", {"predictions": prediction_hashes, "metrics": metric_hashes,
                                                "computed_at": now()})
    marker = {"status": "consumed", "test_path": str(SPLIT_FILES["test"].relative_to(PROJECT_ROOT)),
              "test_sha256": integrity["sha256"], "rows": integrity["rows"],
              "adapter_sha256": EXPECTED_ADAPTER_SHA256, "base_model_weights_sha256": current["base_model_weights_sha256"],
              "evaluation_timestamp": now(), "modes": EVAL_CONFIG["order"], "prediction_hashes": prediction_hashes,
              "metric_hashes": metric_hashes, "pre_test_snapshot_sha256": sha256_file(snapshot_path),
              "policy": "The sealed test must not be used again for model selection, tuning or data work. "
                        "load_split('test') refuses while this marker exists unless the consumed-test override is set."}
    write_json(data.CONSUMED_MARKER, marker, exclusive=not args.override_consumed)
    freeze(data.CONSUMED_MARKER)
    print(json.dumps({"consumed": str(data.CONSUMED_MARKER), "predictions": prediction_hashes}, indent=1))


# --------------------------------------------------------------------------- #
# report (from saved predictions only)
# --------------------------------------------------------------------------- #

def load_predictions() -> dict[str, list[dict]]:
    return {f"{n}_{m}": [json.loads(line) for line in (OUT / "predictions" / f"{n}_{m}.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()] for n, m in MODES}


def as_row(p: dict) -> dict:
    return {"id": p["id"], "request": p["request"], "plan": p["target_plan"], "plan_text": p["target"],
            "metadata": p["metadata"]}


def as_record(p: dict) -> dict:
    return {**p, "output": p["generated"], "pred_plan": p["parsed_plan"]}


def fmt(x) -> str:
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def md_table(rows, headers) -> list[str]:
    return ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] + [
        "| " + " | ".join(fmt(c) for c in r) + " |" for r in rows]


def ci_str(flags) -> str:
    c = fa.rate_ci(flags)
    return f"{c['rate']:.4f} [{c['ci95'][0]:.4f}, {c['ci95'][1]:.4f}]" if c["n"] else "–"


def extra_groups(p: dict) -> list[str]:
    tags = list(p["categories"])
    plan = p["target_plan"]
    if plan["status"] == "ok" and len(plan["outputs"]) == 1:
        tags.append("outputs:single")
    return tags


def breakdown_rows(preds: dict, prefixes: tuple[str, ...], exact: tuple[str, ...] = ()) -> list[dict]:
    out = []
    for key, items in preds.items():
        groups = defaultdict(list)
        for p in items:
            for tag in extra_groups(p):
                if tag.startswith(prefixes) or tag in exact:
                    groups[tag].append(p)
        for tag in sorted(groups):
            g = groups[tag]
            s = fa.rate_ci([x["structural_exact"] for x in g])
            c = fa.rate_ci([x["canonical_exact"] for x in g])
            tp = sum(x["edges_tp"] for x in g)
            pr = tp / max(1, sum(x["edges_pred"] for x in g))
            rc = tp / max(1, sum(x["edges_gold"] for x in g))
            out.append({"model_mode": key, "category": tag, "n": len(g),
                        "schema_valid": fa.rate_ci([x["schema_valid"] for x in g])["rate"],
                        "status_accuracy": fa.rate_ci([x["status_correct"] for x in g])["rate"],
                        "skill_sequence_accuracy": fa.rate_ci([x["skill_sequence_correct"] for x in g])["rate"],
                        "structural_exact": s["rate"], "structural_ci95_low": s["ci95"][0],
                        "structural_ci95_high": s["ci95"][1], "canonical_exact": c["rate"],
                        "canonical_ci95_low": c["ci95"][0], "canonical_ci95_high": c["ci95"][1],
                        "dependency_edge_f1": round(2 * pr * rc / (pr + rc), 4) if pr + rc else 0.0})
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validity_checks(preds: dict, snapshot: dict) -> dict:
    checks = {}
    ids = [p["id"] for p in preds["lora_constrained"]]
    for key, items in preds.items():
        checks[f"{key}_rows"] = len(items)
        checks[f"{key}_unique_ids"] = len({p["id"] for p in items})
        checks[f"{key}_same_order"] = [p["id"] for p in items] == ids
        stored = read_json(OUT / f"{key}_metrics.json")["overall"]
        recomputed = evaluate([as_row(p) for p in items], [p["generated"] for p in items])["overall"]
        checks[f"{key}_metrics_reproduce"] = recomputed == stored
    hashes = read_json(OUT / "prediction_hashes.json")["predictions"]
    checks["prediction_hashes_match"] = all(sha256_file(OUT / "predictions" / n) == h for n, h in hashes.items())
    checks["snapshot_hashes_unchanged"] = critical_hashes() == snapshot["hashes"]
    checks["consumed_marker_present"] = data.CONSUMED_MARKER.exists()
    integrity = read_json(OUT / "test_integrity.json")
    checks["leak_free"] = integrity["leakage"]["leak_free"]
    checks["all_valid"] = (all(checks[f"{k}_rows"] == checks[f"{k}_unique_ids"] == 461 and checks[f"{k}_same_order"]
                               and checks[f"{k}_metrics_reproduce"] for k in preds)
                           and checks["prediction_hashes_match"] and checks["snapshot_hashes_unchanged"]
                           and checks["consumed_marker_present"] and checks["leak_free"])
    return checks


def cmd_report(args) -> None:
    snapshot = read_json(OUT / "pre_test_snapshot.json")
    preds = load_predictions()
    metrics = {k: read_json(OUT / f"{k}_metrics.json") for k in preds}
    val = {("lora", m): read_json(SFT / f"best_val_{m}.json") for m in ("raw", "constrained")}
    val.update({("base", m): read_json(SFT / f"baseline_val_{m}.json") for m in ("raw", "constrained")})
    generic = fa.generic_vocabulary(data.load_split("train"))
    checks = validity_checks(preds, snapshot)
    report: dict = {"validity_checks": checks, "generated_at": now()}

    # ------------------------------------------------ confidence intervals
    ci = {}
    for key, items in preds.items():
        ci[key] = {m: fa.rate_ci([p[FLAG[m]] for p in items]) for m in FLAG}
        for status in ("ok", "clarify", "unsupported"):
            ci[key][f"recall_{status}"] = fa.rate_ci([p["status_correct"] for p in items
                                                      if p["target_plan"]["status"] == status])
    write_json(OUT / "confidence_intervals.json", {"method": fa.ANALYSIS_PLAN["confidence_interval"], "intervals": ci})
    report["confidence_intervals"] = ci

    # ------------------------------------------------ validation vs test
    lines = ["# Validation vs sealed test", "",
             "Same models, same pipeline; validation numbers are the ones fixed before the test was read "
             "(LoRA = epoch 2, selected on validation). Test CI: Wilson 95%. Δ CI: Newcombe hybrid score 95%.", ""]
    vt = {}
    for name in ("lora", "base"):
        for mode in ("raw", "constrained"):
            key = f"{name}_{mode}"
            v, t = val[(name, mode)]["overall"], metrics[key]["overall"]
            nv, nt = v["n"], t["n"]
            rows_, entry = [], {}
            for m in KEY_METRICS:
                if m in FLAG:
                    kv, kt = round(v[m] * nv), ci[key][m]["k"]
                    d, lo, hi = fa.newcombe_diff(kv, nv, kt, nt)
                    entry[m] = {"validation": v[m], "test": t[m], "delta": round(d, 4),
                                "delta_ci95": [round(lo, 4), round(hi, 4)], "test_ci95": ci[key][m]["ci95"]}
                    rows_.append((m, v[m], t[m], round(d, 4), f"[{lo:.4f}, {hi:.4f}]",
                                  f"[{ci[key][m]['ci95'][0]:.4f}, {ci[key][m]['ci95'][1]:.4f}]"))
                else:
                    entry[m] = {"validation": v[m], "test": t[m], "delta": round(t[m] - v[m], 4)}
                    rows_.append((m, v[m], t[m], round(t[m] - v[m], 4), "–", "–"))
            vs, ts = val[(name, mode)]["status_recall"], metrics[key]["status_recall"]
            for s in ("ok", "clarify", "unsupported"):
                entry[f"recall_{s}"] = {"validation": vs[s], "test": ts[s], "delta": round(ts[s] - vs[s], 4)}
                rows_.append((f"recall_{s}", vs[s], ts[s], round(ts[s] - vs[s], 4), "–",
                              f"[{ci[key]['recall_' + s]['ci95'][0]:.4f}, {ci[key]['recall_' + s]['ci95'][1]:.4f}]"))
            vt[key] = entry
            lines += [f"## {name.upper()} {mode.upper()} (validation n={nv}, test n={nt})", ""]
            lines += md_table(rows_, ["metric", "validation", "sealed test", "Δ (test − val)", "Δ 95% CI", "test 95% CI"])
            lines.append("")
    (OUT / "validation_vs_test.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report["validation_vs_test"] = vt

    # ------------------------------------------------ base vs lora (paired)
    lines = ["# Stage-2 base vs Manager LoRA — sealed test (461 rows, paired)", "",
             "Both models were fixed before the test was read. Paired counts are per test row; p = exact two-sided "
             "McNemar. This comparison is descriptive and is not used for any selection.", ""]
    bl = {}
    for mode in ("raw", "constrained"):
        b, l = metrics[f"base_{mode}"]["overall"], metrics[f"lora_{mode}"]["overall"]
        lines += [f"## {mode.upper()}", ""]
        lines += md_table([(m, b[m], l[m], round(l[m] - b[m], 4)) for m in KEY_METRICS],
                          ["metric", "Stage-2 base", "LoRA", "Δ"])
        pairs = {}
        for m in ("structural_exact", "canonical_exact", "schema_valid", "status_accuracy", "skill_sequence_accuracy",
                  "sources_exact", "outputs_exact", "topology_exact"):
            flag = FLAG[m]
            pairs[m] = fa.paired([p[flag] for p in preds[f"base_{mode}"]], [p[flag] for p in preds[f"lora_{mode}"]])
        lines += ["", "Paired:", ""]
        lines += md_table([(m, v["both_correct"], v["base_only_correct"], v["lora_only_correct"], v["both_wrong"],
                            f"{v['mcnemar_p']:.3g}") for m, v in pairs.items()],
                          ["metric", "both correct", "base only", "LoRA only", "both wrong", "McNemar p"])
        sb, sl = metrics[f"base_{mode}"]["status_recall"], metrics[f"lora_{mode}"]["status_recall"]
        lines += ["", "Status recall:", ""]
        lines += md_table([(s, sb[s], sl[s]) for s in ("ok", "clarify", "unsupported")], ["gold status", "base", "LoRA"])
        lines.append("")
        bl[mode] = {"base": b, "lora": l, "paired": pairs, "status_recall": {"base": sb, "lora": sl}}
    (OUT / "base_vs_lora_test.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report["base_vs_lora"] = bl

    # ------------------------------------------------ breakdown CSVs
    skill = breakdown_rows(preds, ("skill:",))
    topo = breakdown_rows(preds, ("tasks:", "sources:", "outputs:"))
    tax = breakdown_rows(preds, ("taxonomy:",))
    status = breakdown_rows(preds, ("status:", "group:"), ("normal", "hard"))
    write_csv(OUT / "skill_breakdown.csv", skill)
    write_csv(OUT / "topology_breakdown.csv", topo)
    write_csv(OUT / "taxonomy_breakdown.csv", tax)
    write_csv(OUT / "status_breakdown.csv", status)
    report["breakdowns"] = {"skill": skill, "topology": topo, "taxonomy": tax, "status_difficulty": status}

    # ------------------------------------------------ instruction fidelity
    fid_out = {"classes": fa.FIDELITY_CLASSES, "rule": fa.ANALYSIS_PLAN["instruction_fidelity"]}
    for mode in ("constrained", "raw"):
        items = preds[f"lora_{mode}"]
        struct_only = Counter()
        all_rows = Counter()
        entity_rows, examples = 0, defaultdict(list)
        search_pairs = search_corrupt = 0
        per_skill = defaultdict(Counter)
        for p in items:
            row, rec = as_row(p), as_record(p)
            f = fa.fidelity(row, rec, generic)
            if p["parsed_plan"] and p["status_correct"]:
                for g, q in zip(p["target_plan"]["tasks"], p["parsed_plan"]["tasks"]):
                    if g["skill"] == q["skill"] == "search":
                        search_pairs += 1
            if not f:
                continue
            all_rows[f["class"]] += 1
            if p["structural_exact"]:
                struct_only[f["class"]] += 1
            entity_rows += f["entity_corruption"]
            for pair in f["pairs"]:
                per_skill[pair["skill"]][pair["class"]] += 1
                if pair["skill"] == "search" and (pair["class"] == "C" or pair.get("entity_corruption")):
                    search_corrupt += 1
            if len(examples[f["class"]]) < 5:
                examples[f["class"]].append({"id": p["id"], "taxonomy": p["metadata"]["taxonomy"],
                                             "structural_exact": p["structural_exact"],
                                             "pairs": [{k: v for k, v in pair.items()} for pair in f["pairs"]]})
        struct_mismatch = sum(1 for p in items if p["structural_exact"] and not p["canonical_exact"])
        fid_out[mode] = {
            "structural_exact_but_not_canonical": struct_mismatch,
            "classes_among_structural_exact": dict(sorted(struct_only.items())),
            "classes_among_all_same_status_rows": dict(sorted(all_rows.items())),
            "semantic_instruction_errors_structural_exact": sum(struct_only[c] for c in "CDEF"),
            "harmless_or_paraphrase_structural_exact": struct_only["A"] + struct_only["B"],
            "rows_with_entity_corruption": entity_rows,
            "search_task_pairs": search_pairs, "search_task_pairs_with_entity_corruption": search_corrupt,
            "per_skill_pair_classes": {k: dict(v) for k, v in per_skill.items()},
            "examples_first_in_file_order": examples}
    write_json(OUT / "instruction_fidelity.json", fid_out)
    report["instruction_fidelity"] = {m: {k: v for k, v in fid_out[m].items() if k != "examples_first_in_file_order"}
                                      for m in ("constrained", "raw")}

    # ------------------------------------------------ error analysis
    err = {}
    md = ["# Error analysis — Manager LoRA on the sealed test", "",
          "Tags: `manager_sft_eval.failure_categories` (same as validation) plus supplementary tags from "
          "`manager_final_analysis.failure_tags` (clarify_miss, unsupported_miss, duplicate_source, entity_corruption). "
          "A row can carry several. Rates are over all 461 rows. Examples: first rows in test-file order per tag "
          "(no hand selection).", ""]
    for mode in ("constrained", "raw"):
        items = preds[f"lora_{mode}"]
        counts, examples = Counter(), defaultdict(list)
        for p in items:
            f = fa.fidelity(as_row(p), as_record(p), generic)
            tags = fa.failure_tags(as_row(p), as_record(p), f)
            counts.update(tags)
            for t in tags:
                if len(examples[t]) < 3:
                    examples[t].append({"id": p["id"], "taxonomy": p["metadata"]["taxonomy"], "request": p["request"],
                                        "gold": p["target"], "prediction": p["generated"].strip()})
        confusion = Counter(f"{p['target_plan']['status']}->{p['pred_status']}" for p in items)
        failed = sum(1 for p in items if not p["canonical_exact"])
        err[mode] = {"rows": len(items), "rows_not_canonical_exact": failed,
                     "rows_not_structural_exact": sum(1 for p in items if not p["structural_exact"]),
                     "tags": {k: {"count": v, "rate": round(v / len(items), 4)} for k, v in counts.most_common()},
                     "status_confusion": dict(sorted(confusion.items())), "examples": examples}
        md += [f"## {mode}", "", f"Not canonical-exact: {failed}/{len(items)}; not structural-exact: "
               f"{err[mode]['rows_not_structural_exact']}/{len(items)}", ""]
        md += md_table([(k, v["count"], v["rate"]) for k, v in err[mode]["tags"].items()], ["tag", "count", "rate"])
        md += ["", "Status confusion (gold → predicted): " + ", ".join(f"{k}: {v}" for k, v in err[mode]["status_confusion"].items()), ""]
        if mode == "constrained":
            for tag, exs in examples.items():
                md += [f"### {tag}", ""]
                for e in exs:
                    md += [f"- `{e['id']}` ({e['taxonomy']})",
                           f"  - request: {e['request'][:220].replace(chr(10), ' ⏎ ')}",
                           f"  - gold: `{e['gold']}`", f"  - pred: `{e['prediction'][:400]}`"]
                md.append("")

    # ------------------------------------------------ known weakness probes
    probes = {}
    for probe in ("negated_compare", "compare_cue_without_compare", "search_without_cue", "user_search_compare",
                  "sql_sql_compare", "clarify_all", "clarify_deictic"):
        entry = {}
        for key in ("lora_constrained", "lora_raw", "base_constrained"):
            sel = [p for p in preds[key] if probe in fa.probes(as_row(p))]
            if not sel:
                continue
            e = {"n": len(sel), "structural_exact": fa.rate_ci([p["structural_exact"] for p in sel]),
                 "canonical_exact": fa.rate_ci([p["canonical_exact"] for p in sel])}
            skills = [{t["skill"] for t in (p["parsed_plan"] or {"tasks": []})["tasks"]} for p in sel]
            if probe in ("negated_compare", "compare_cue_without_compare"):
                e["predicts_compare_anyway"] = fa.rate_ci(["compare" in s for s in skills])
            if probe == "search_without_cue":
                e["predicts_search"] = fa.rate_ci(["search" in s for s in skills])
            if probe.startswith("clarify"):
                e["clarify_recall"] = fa.rate_ci([p["pred_status"] == "clarify" for p in sel])
            entry[key] = e
        vsel = [r for r in data.load_split("validation") if probe in fa.probes(r)]
        entry["validation_n"] = len(vsel)
        probes[probe] = entry
    fi = fid_out["constrained"]
    probes["search_entity_corruption"] = {
        "lora_constrained": {"search_task_pairs": fi["search_task_pairs"],
                             "with_entity_corruption": fi["search_task_pairs_with_entity_corruption"],
                             "rate": fa.rate_ci([True] * fi["search_task_pairs_with_entity_corruption"] +
                                                [False] * (fi["search_task_pairs"] - fi["search_task_pairs_with_entity_corruption"]))}}
    err["weakness_probes"] = probes
    md += ["## Known weakness probes (subgroups of the existing test rows; no rows added)", "", "```json",
           json.dumps(probes, ensure_ascii=False, indent=2), "```", ""]
    write_json(OUT / "error_analysis.json", err)
    (OUT / "error_analysis.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    report["error_analysis"] = {m: {k: v for k, v in err[m].items() if k != "examples"} for m in ("constrained", "raw")}
    report["weakness_probes"] = probes

    # ------------------------------------------------ raw vs constrained (LoRA)
    rc = fa.paired([p["structural_exact"] for p in preds["lora_raw"]],
                   [p["structural_exact"] for p in preds["lora_constrained"]])
    same_text = sum(a["generated"].strip() == b["generated"].strip()
                    for a, b in zip(preds["lora_raw"], preds["lora_constrained"]))
    report["lora_raw_vs_constrained"] = {"structural_paired_raw_as_base": rc, "identical_generations": same_text,
                                         "raw_structural": metrics["lora_raw"]["overall"]["structural_exact"],
                                         "constrained_structural": metrics["lora_constrained"]["overall"]["structural_exact"],
                                         "raw_grammar_valid": metrics["lora_raw"]["overall"]["grammar_valid"]}
    report["timing"] = {k: metrics[k]["timing"] for k in metrics}
    report["status"] = "MANAGER_SLM_V1_FINALIZED" if checks["all_valid"] else "FINAL_EVALUATION_INVALID"
    write_json(OUT / "final_manager_v1_report.json", report)
    print(json.dumps({"status": report["status"], "checks": checks,
                      "lora_raw_vs_constrained": report["lora_raw_vs_constrained"]}, indent=1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    snap = sub.add_parser("snapshot")
    snap.add_argument("--tests-status", help="file with the current pytest/ctest summary")
    run = sub.add_parser("run")
    run.add_argument("--consume-sealed-test", action="store_true")
    run.add_argument("--override-consumed", action="store_true",
                     help="re-read a consumed sealed test (never used for Manager-SLM v1)")
    sub.add_parser("report")
    args = parser.parse_args()
    {"snapshot": cmd_snapshot, "run": cmd_run, "report": cmd_report}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
