#!/usr/bin/env python3
"""Assemble the Stage-2 closure report and the Manager-SLM readiness gate.

Reads only artifacts already on disk (nothing is trained or evaluated here):
outputs/stage2_training_report.json and outputs/final_validation/*.

Gate thresholds below were written before any Stage-2 generation output existed
(2026-09-24) and must not be edited to change a verdict.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))

from nikl_model_eval import MODELS as NIKL_MODELS  # noqa: E402
from nikl_model_eval import grouped, load_shards, paired_bootstrap, summarize_rows  # noqa: E402
from recovery_saturation import classify, interval_rates  # noqa: E402

FV = ROOT / "outputs" / "final_validation"
CANONICAL_DIR = "outputs/models/qwen3-0.6b-ko32k-stage2-signal2678"
CANONICAL_NAME = "Qwen3-0.6B-KO32K-Stage2-signal2678"
MODEL_LABELS = {"original_qwen": "Original Qwen3-0.6B", "stage1_final": "KO32K Stage-1 final",
                "stage2_signal2678": "KO32K Stage-2 signal2678"}

# ---- pre-registered gate thresholds (do not tune) -------------------------- #
GATE_E_GREEDY_LOOP_MARGIN = 0.20      # Stage-2 greedy loop rate <= Qwen + 0.20
GATE_E_SAMPLED_LOOP_MARGIN = 0.10     # Stage-2 sampled loop rate <= Qwen + 0.10
GATE_E_MAX_EMPTY_RATE = 0.05
GATE_E_MIN_UNICODE_OK_RATE = 0.98
GATE_G_MIN_RATIO_TO_QWEN = 0.5        # JSON parse / SQL syntax validity vs Qwen
GATE_G_MIN_DSL_FIRST_LINE = 0.5       # absolute; Qwen has no Skill control tokens


def read(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# learning curve
# --------------------------------------------------------------------------- #

def learning_curve(training: dict) -> dict:
    points = [(h["tokens"], h["eval_nll_per_character"]) for h in training["history"]]
    final = training["final_evaluation"]["balanced_decontaminated"]["overall"]["nll_per_character"]
    points.append((training["tokens_trained"], final))
    rates = interval_rates(points)
    start = points[0][1]
    for rate in rates:
        rate["relative_improvement_percent"] = 100.0 * rate["improvement"] / rate["from_nll"]
        rate["share_of_total_gain_percent"] = 100.0 * rate["improvement"] / (start - final)
    halves = {}
    for label, (lo, hi) in {"first_half_0_to_20.5M": (0, 20_480_000),
                            "second_half_20.5M_to_41.0M": (20_480_000, 40_960_000)}.items():
        a = dict(points)[lo]
        b = dict(points)[hi]
        halves[label] = {"improvement": a - b, "improvement_per_1m": (a - b) * 1e6 / (hi - lo)}
    return {
        "eval_set": "data/eval/recovery_balanced_eval_decontaminated.jsonl (2,888 records)",
        "points": [{"tokens": t, "nll_per_character": v} for t, v in points],
        "intervals": rates,
        "halves": halves,
        "last_interval": rates[-1],
        "stage1_saturation_rule_applied_for_reference": classify(rates),
        "not_measured": "No checkpoint beyond 43,876,352 tokens exists; the 60M-token outcome was "
                        "never measured and is not extrapolated here.",
    }


# --------------------------------------------------------------------------- #
# NIKL
# --------------------------------------------------------------------------- #

def nikl_manifest_articles() -> int | None:
    """Article count of the local NIKL manifest (one article per line); None if absent."""
    path = ROOT / "data/eval/nikl_eval_manifest.jsonl"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def nikl_block(tag: str) -> dict | None:
    complete = read(FV / "nikl_shards" / tag / "COMPLETE.json")
    rows = {name: load_shards(FV, tag, name) for name in NIKL_MODELS}
    if not all(rows.values()):
        return None
    ids = [{r["id"] for r in rows[n]} for n in NIKL_MODELS]
    same_articles = all(s == ids[0] for s in ids)
    out = {
        "tag": tag, "complete": bool(complete), "completion_record": complete,
        "same_articles_for_all_models": same_articles,
        "overall": {n: summarize_rows(r) for n, r in rows.items()},
        "by_topic": {n: grouped(r, "topic") for n, r in rows.items()},
        "by_publisher": {n: grouped(r, "publisher") for n, r in rows.items()},
        "by_length_bucket": {n: grouped(r, "length_bucket") for n, r in rows.items()},
        "bootstrap": {
            "stage2_minus_original_qwen": paired_bootstrap(rows["stage2_signal2678"], rows["original_qwen"]),
            "stage2_minus_stage1": paired_bootstrap(rows["stage2_signal2678"], rows["stage1_final"]),
        },
    }
    q = out["overall"]["original_qwen"]["total_tokens"]
    k = out["overall"]["stage2_signal2678"]["total_tokens"]
    out["token_reduction_percent_vs_qwen"] = 100.0 * (1.0 - k / q)
    return out


def write_nikl_csvs(block: dict, prefix: str) -> list[str]:
    written = []
    for key, name in (("by_topic", "category"), ("by_length_bucket", "length"),
                      ("by_publisher", "publisher")):
        path = FV / f"{prefix}_{name}_breakdown.csv"
        groups = sorted(block[key]["original_qwen"])
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([name, "articles", "characters",
                             "qwen_nll_per_char", "stage1_nll_per_char", "stage2_nll_per_char",
                             "stage2_minus_qwen", "qwen_tokens", "ko32k_tokens", "token_reduction_percent"])
            for group in groups:
                q = block[key]["original_qwen"][group]
                s1 = block[key]["stage1_final"][group]
                s2 = block[key]["stage2_signal2678"][group]
                writer.writerow([group, q["articles"], q["characters"],
                                 f"{q['nll_per_character']:.6f}", f"{s1['nll_per_character']:.6f}",
                                 f"{s2['nll_per_character']:.6f}",
                                 f"{s2['nll_per_character'] - q['nll_per_character']:+.6f}",
                                 q["total_tokens"], s2["total_tokens"],
                                 f"{100.0 * (1 - s2['total_tokens'] / q['total_tokens']):.3f}"])
        written.append(str(path.relative_to(ROOT)))
    return written


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #

def pooled(summary: dict, model: str, category: str, key: str) -> float | None:
    values, weights = [], []
    for mode in ("greedy", "sampled"):
        block = summary[model][mode]["by_category"].get(category)
        if block and key in block:
            values.append(block[key] * block["samples"])
            weights.append(block["samples"])
    return sum(values) / sum(weights) if weights else None


def failure_table(samples: dict) -> list[dict]:
    rows = []
    for model, run in samples["runs"].items():
        for mode in ("greedy", "sampled"):
            for sample in run[mode]:
                if sample["category"] != "legacy_stage1" and not any(sample["failure_signatures"].values()):
                    continue
                rows.append({"model": model, "mode": mode, "id": sample["id"], "seed": sample["seed"],
                             **sample["failure_signatures"]})
    return rows


# --------------------------------------------------------------------------- #
# gate
# --------------------------------------------------------------------------- #

def gate(ctx: dict) -> dict:
    g: dict[str, dict] = {}
    smoke = ctx["smoke"] or {}
    g["A_checkpoint_reproducibly_loads"] = {
        "pass": bool(smoke.get("all_pass")) and bool(smoke.get("checks", {}).get("reload_identical")),
        "evidence": "outputs/final_validation/smoke_test.json"}
    rep = ctx["reproduction"] or {}
    checks = rep.get("reproduction_checks", {})
    g["B_balanced_eval_reproducible"] = {
        "pass": bool(checks) and all(v["within_tolerance"] for k, v in checks.items() if k.startswith("balanced/")),
        "evidence": "outputs/final_validation/reproduction.json"}
    g["C_korean_large_eval_reproducible"] = {
        "pass": bool(checks) and all(v["within_tolerance"] for k, v in checks.items() if k.startswith("korean/")),
        "evidence": "outputs/final_validation/reproduction.json"}
    full = ctx["nikl_full"]
    g["D_nikl_full_model_evaluation_completed"] = {
        "pass": bool(full and full["complete"] and full["same_articles_for_all_models"]
                     and ctx.get("nikl_manifest_articles") is not None
                     and full["overall"]["original_qwen"]["articles"] == ctx["nikl_manifest_articles"]),
        "evidence": "outputs/final_validation/nikl_shards/full/COMPLETE.json",
        "note": None if full and full["complete"] else
        "Only the deterministic 1% pilot is complete; the full-manifest run is not."}

    gen = ctx["generation"]
    if gen:
        s = gen["summary"]
        q_g = s["original_qwen"]["greedy"]["overall"]
        q_s = s["original_qwen"]["sampled"]["overall"]
        k_g = s["stage2_signal2678"]["greedy"]["overall"]
        k_s = s["stage2_signal2678"]["sampled"]["overall"]
        n = k_g["samples"] + k_s["samples"]
        empty = (k_g["empty_rate"] * k_g["samples"] + k_s["empty_rate"] * k_s["samples"]) / n
        unicode_ok = (k_g["unicode_ok_rate"] * k_g["samples"] + k_s["unicode_ok_rate"] * k_s["samples"]) / n
        sig = k_g["failure_signature_counts"]
        e = {
            "greedy_loop_rate": k_g["loop_detected_rate"], "qwen_greedy_loop_rate": q_g["loop_detected_rate"],
            "sampled_loop_rate": k_s["loop_detected_rate"], "qwen_sampled_loop_rate": q_s["loop_detected_rate"],
            "empty_rate": empty, "unicode_ok_rate": unicode_ok,
            "greedy_repeated_10000": sig.get("repeated_10000", 0),
            "greedy_repeated_input_sql": sig.get("repeated_input_sql", 0),
        }
        e_pass = (e["greedy_loop_rate"] <= e["qwen_greedy_loop_rate"] + GATE_E_GREEDY_LOOP_MARGIN
                  and e["sampled_loop_rate"] <= e["qwen_sampled_loop_rate"] + GATE_E_SAMPLED_LOOP_MARGIN
                  and e["greedy_repeated_10000"] == 0 and e["greedy_repeated_input_sql"] == 0
                  and empty <= GATE_E_MAX_EMPTY_RATE and unicode_ok >= GATE_E_MIN_UNICODE_OK_RATE)
        g["E_no_severe_systematic_generation_collapse"] = {"pass": e_pass, "measurements": e,
                                                            "evidence": "outputs/final_validation/generation_summary.json"}
        structured = {
            "stage2_json_parse_valid": pooled(s, "stage2_signal2678", "json_structured", "json_parse_valid_rate"),
            "qwen_json_parse_valid": pooled(s, "original_qwen", "json_structured", "json_parse_valid_rate"),
            "stage2_sql_syntax_valid": pooled(s, "stage2_signal2678", "sql", "sql_syntax_valid_rate"),
            "qwen_sql_syntax_valid": pooled(s, "original_qwen", "sql", "sql_syntax_valid_rate"),
            "stage2_dsl_first_line_valid": pooled(s, "stage2_signal2678", "skill_dsl", "dsl_first_line_valid_rate"),
            "stage1_dsl_first_line_valid": pooled(s, "stage1_final", "skill_dsl", "dsl_first_line_valid_rate"),
        }

        def ratio_ok(a, b):
            return a is not None and b is not None and a >= GATE_G_MIN_RATIO_TO_QWEN * b

        g_pass = (ratio_ok(structured["stage2_json_parse_valid"], structured["qwen_json_parse_valid"])
                  and ratio_ok(structured["stage2_sql_syntax_valid"], structured["qwen_sql_syntax_valid"])
                  and (structured["stage2_dsl_first_line_valid"] or 0.0) >= GATE_G_MIN_DSL_FIRST_LINE)
        g["G_no_catastrophic_structured_collapse"] = {
            "pass": g_pass, "measurements": structured,
            "evidence": "outputs/final_validation/generation_summary.json",
            "note": None if g_pass else (
                f"JSON parse {structured['stage2_json_parse_valid']:.2f} (Qwen {structured['qwen_json_parse_valid']:.2f}) and "
                f"SQL syntax {structured['stage2_sql_syntax_valid']:.2f} (Qwen {structured['qwen_sql_syntax_valid']:.2f}) pass; "
                f"Skill DSL first-line validity {structured['stage2_dsl_first_line_valid']:.2f} is below the pre-registered "
                f"{GATE_G_MIN_DSL_FIRST_LINE}. Needs a decision: either accept that DSL format is taught by the Manager SFT "
                "(and re-state gate G), or run a DSL probe in the v3 prompt shapes before Manager-SLM work.")}
    else:
        g["E_no_severe_systematic_generation_collapse"] = {"pass": False, "note": "generation eval missing"}
        g["G_no_catastrophic_structured_collapse"] = {"pass": False, "note": "generation eval missing"}

    checks_s = smoke.get("checks", {})
    audit = ctx["special_audit"]
    g["F_no_nan_inf_or_invalid_special_ids"] = {
        "pass": bool(checks_s.get("no_nan_inf") and checks_s.get("no_invalid_special_ids")
                     and checks_s.get("control_tokens_ok") and audit and not audit["checkpoints_with_invalid_ids"]),
        "evidence": "smoke_test.json + special_token_audit.json"}
    g["H_artifacts_and_report_complete"] = {"pass": all(ctx["artifacts_present"].values()),
                                            "missing": [k for k, v in ctx["artifacts_present"].items() if not v]}
    tests = ctx["pytest"]
    g["I_pytest_all_pass"] = {"pass": bool(tests and tests["failures"] == 0 and tests["errors"] == 0
                                           and tests["tests"] > 0), "result": tests}
    prov = ctx["nikl_provenance"]
    g["J_provenance_documented"] = {"pass": bool(prov and ctx["canonical"]),
                                    "evidence": "nikl_provenance.json + canonical_checkpoint.json"}
    passed = all(v["pass"] for v in g.values())
    blockers = [k for k, v in g.items() if not v["pass"]]
    decision = ctx.get("closure_decision") or {}
    waived = sorted(set(blockers) & set(decision.get("waived_criteria", {})))
    closed_by_decision = not passed and bool(blockers) and set(blockers) <= set(waived)
    if passed:
        verdict = "Stage-2 closed — ready for Manager-SLM dataset/SFT design"
    elif closed_by_decision:
        verdict = ("Stage-2 closed by project-owner decision — ready for Manager-SLM dataset/SFT design "
                   f"(waived, not passed: {', '.join(waived)})")
    else:
        verdict = "Stage-2 not closed — blockers remain"
    return {"criteria": g, "pass": passed, "blockers": blockers, "waived": waived,
            "closed": passed or closed_by_decision, "closed_by_decision": closed_by_decision,
            "decision": decision or None, "verdict": verdict}


# --------------------------------------------------------------------------- #
# special-token audit of every checkpoint (JSON only, no weights)
# --------------------------------------------------------------------------- #

def special_token_audit() -> dict:
    rows, bad_dirs = [], []
    for directory in sorted((ROOT / "outputs" / "models").glob("*/")):
        config = read(directory / "config.json")
        if not config:
            continue
        vocab = config["vocab_size"]
        bad = []
        for name in ("config.json", "generation_config.json"):
            data = read(directory / name) or {}
            for key, value in data.items():
                if key.endswith("token_id") and value is not None:
                    for item in value if isinstance(value, list) else [value]:
                        if not 0 <= int(item) < vocab:
                            bad.append(f"{name}:{key}={item}")
        tokenizer = read(directory / "tokenizer.json") or {"added_tokens": [], "model": {"vocab": {}}}
        bad += [f"tokenizer.json:{t['content']}={t['id']}" for t in tokenizer["added_tokens"] if t["id"] >= vocab]
        tok_cfg = read(directory / "tokenizer_config.json") or {}
        bad += [f"tokenizer_config:{i}" for i in tok_cfg.get("added_tokens_decoder", {}) if int(i) >= vocab]
        rows.append({"checkpoint": str(directory.relative_to(ROOT)), "vocab_size": vocab,
                     "invalid": bad, "tokenizer_json_truncation": tokenizer.get("truncation")})
        if bad:
            bad_dirs.append(str(directory.relative_to(ROOT)))
    return {"checkpoints": rows, "checkpoints_with_invalid_ids": bad_dirs}


def pytest_result() -> dict | None:
    path = FV / "pytest.xml"
    if not path.exists():
        return None
    root = ET.parse(path).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    return {k: int(suite.get(k, 0)) for k in ("tests", "failures", "errors", "skipped")}


# --------------------------------------------------------------------------- #
# canonical metadata
# --------------------------------------------------------------------------- #

def canonical_metadata(training: dict, structure: dict | None) -> dict:
    cfg = read(ROOT / "outputs/stage2/stage2-continued-pretraining-config.json") or {}
    model_dir = ROOT / CANONICAL_DIR
    code_files = ["model/stage2_train.py", "model/recovery_eval.py", "model/generation_eval.py",
                  "model/nikl_model_eval.py", "model/stage2_final_validation.py",
                  "model/stage2_final_report.py", "corpus/build_stage2_corpus.py", "corpus/nikl.py"]
    return {
        "name": CANONICAL_NAME,
        "checkpoint": CANONICAL_DIR,
        "status": "canonical Stage-2 final candidate (43.876M-token Stage-2 checkpoint; planned budget 60M not completed)",
        "base_model": "Qwen/Qwen3-0.6B (HF snapshot c1899de289a04d12100db370d81485cdf75e47ca)",
        "tokenizer": "KO-32K-mixed10-structured-v2 (outputs/tokenizer-ko-32000-mixed10-structured-v2; vocab.json/merges.txt byte-identical)",
        "vocab_size": 32000,
        "parameters": (structure or {}).get("stage2_signal2678", {}).get("total_parameters"),
        "embedding_parameters": (structure or {}).get("stage2_signal2678", {}).get("embedding_parameters"),
        "tie_word_embeddings": True,
        "stage1_source_checkpoint": "outputs/models/qwen3-0.6b-ko32k-recovery-stage1-extended (embedding-only, 4.096M tokens)",
        "stage2_corpus": {"version": "v3", "path": cfg.get("corpus"),
                          "documents": 241_749, "characters": 151_105_132,
                          "packed_blocks_available": cfg.get("packed_blocks_available"),
                          "packed_blocks_planned": cfg.get("packed_blocks_used"),
                          "corpus_repeated": cfg.get("corpus_repeated")},
        "tokens_trained": training["tokens_trained"],
        "steps": training["steps"],
        "planned_steps": training["planned_steps"],
        "budget_tokens": training["budget_tokens"],
        "budget_fraction": training["budget_fraction"],
        "optimizer": {"type": "AdamW", "betas": cfg.get("betas"), "eps": cfg.get("eps"),
                      "weight_decay": cfg.get("weight_decay"), "grad_clip": cfg.get("grad_clip"),
                      "parameter_groups": cfg.get("parameter_groups")},
        "scheduler": {"description": cfg.get("scheduler"), "warmup_steps": cfg.get("warmup_steps"),
                      "embedding_peak_lr": cfg.get("embedding_lr"), "transformer_peak_lr": cfg.get("transformer_lr"),
                      "note": "cosine was planned over 3,662 steps; the run stopped at step 2,678, so the "
                              "LR never reached its 10% floor"},
        "batch": {"micro_batch": cfg.get("micro_batch"), "grad_accum": cfg.get("grad_accum"),
                  "tokens_per_optimizer_step": cfg.get("tokens_per_optimizer_step")},
        "context_length": cfg.get("sequence_length"),
        "precision": cfg.get("precision"),
        "gradient_checkpointing": cfg.get("gradient_checkpointing"),
        "seed": cfg.get("seed"),
        "backend": "PyTorch 2.14.0 / transformers 4.57.6 on Apple MPS (M5 Pro, 64 GB)",
        "stop_reason": "stopped by user schedule via SIGUSR1 at step 2,678 (73.1% of the 60M-token budget); "
                       "weights and optimizer state saved normally",
        "optimizer_state_saved": (model_dir / "optimizer_state.pt").exists(),
        "resume_history": "run 1 stalled (MPS allocator growth) and was resumed from step1500 with "
                          "gradient checkpointing; step 1750 re-measured 1.5142 vs 1.5141",
        "git_commit": "unavailable — the project directory is not a git repository; "
                      "source file sha256 values are recorded instead",
        "source_sha256": {f: sha256(ROOT / f) for f in code_files},
        "artifact_sha256": {
            "model.safetensors": sha256(model_dir / "model.safetensors"),
            "tokenizer.json": sha256(model_dir / "tokenizer.json"),
            "vocab.json": sha256(model_dir / "vocab.json"),
            "merges.txt": sha256(model_dir / "merges.txt"),
            "config.json": sha256(model_dir / "config.json"),
            "generation_config.json": sha256(model_dir / "generation_config.json"),
        },
        "data_sha256": {
            "stage2_train_v3.jsonl": sha256(ROOT / "data/processed/stage2_train_v3.jsonl"),
            "stage2_holdout_v3.jsonl": sha256(ROOT / "data/processed/stage2_holdout_v3.jsonl"),
            "recovery_balanced_eval_decontaminated.jsonl": sha256(ROOT / "data/eval/recovery_balanced_eval_decontaminated.jsonl"),
            "recovery_korean_large_eval_decontaminated.jsonl": sha256(ROOT / "data/eval/recovery_korean_large_eval_decontaminated.jsonl"),
            "nikl_eval_manifest.jsonl": sha256(ROOT / "data/eval/nikl_eval_manifest.jsonl"),
            "generation_prompts_v1.jsonl": sha256(ROOT / "data/eval/generation_prompts_v1.jsonl"),
        },
        "metadata_fixes_applied": "outputs/final_validation/checkpoint_metadata_fix.json (no weight change)",
    }


# --------------------------------------------------------------------------- #
# markdown
# --------------------------------------------------------------------------- #

def f4(x) -> str:
    return "n/a" if x is None else f"{x:.4f}"


def markdown(r: dict) -> str:  # noqa: C901 - a report is a long, flat sequence
    rep = r["reproduction"]
    lc = r["learning_curve"]
    st = r["structure"]
    L = []
    add = L.append
    add(f"# Stage-2 final report — {CANONICAL_NAME}")
    add("")
    add("> Evaluation, verification and documentation only. No training, SFT, LoRA or Manager-SLM work "
        "was started. The Stage-2 run was not resumed.")
    add(">")
    add(f"> **Measured gate: {'PASS' if r['gate']['pass'] else 'FAIL'}** — {r['gate']['verdict']}"
        + ("" if r["gate"]["pass"] or r["gate"]["closed_by_decision"] else f": {', '.join(r['gate']['blockers'])}"))
    add("")
    add("Facts are stated as measured; interpretation is marked **Interpretation**.")
    add("")

    add("## 1. Research question")
    add("Can Qwen3-0.6B's 151,936-token multilingual vocabulary be replaced by a 32,000-token Korean/Skill-"
        "focused tokenizer, with embedding transplant plus continued pretraining, so that the model uses "
        "fewer parameters and shorter Korean sequences while matching or beating the original model's "
        "character-normalized likelihood on Korean — without catastrophic loss on the structured workloads "
        "(JSON, SQL, code/URL, Skill DSL) the Manager-SLM will need?")
    add("")

    add("## 2. Original model")
    q = st["original_qwen"]
    add(f"Qwen/Qwen3-0.6B: vocab {q['vocab_size']:,}, hidden {q['hidden_size']}, tied embedding/output, "
        f"{q['total_parameters']:,} parameters ({q['embedding_parameters']:,} embedding, "
        f"{q['non_embedding_parameters']:,} non-embedding).")
    add("")

    add("## 3. KO32K tokenizer design")
    add("Qwen ByteLevel BPE pipeline retrained at 32,000 entries on 92.5M characters (FineWeb2 Korean + AI Hub "
        "technical/dialogue) with a 10% mixed budget containing code, URL, JSON (3.0%) and SQL "
        "(`KO-32K-mixed10-structured-v2`). All 26 Qwen added tokens and 8 atomic Skill control tokens "
        "(`<검색> <요약> <SQL> <코드> <이전0> <이전1> <이전2> <종료>`) are preserved.")
    add("")

    add("## 4. Embedding transplant")
    add("Rows initialised by token string: 7,644 exact copies, 24,356 decomposition means, 0 random. "
        f"Transplant-only balanced NLL/char (decontaminated): {f4(r['training']['baselines']['transplant_balanced'])}.")
    add("")

    add("## 5. Stage-1 embedding recovery")
    add("Embedding-only training (transformer frozen) for 4.096M tokens "
        "(`qwen3-0.6b-ko32k-recovery-stage1-extended`): balanced 2.0871, Korean-large 2.6523 — still worse "
        "than Original Qwen (1.7786 / 2.1076). Stage-1 greedy generations looped (`10000` repetition, "
        "`\"input\":\"sql\"` repetition, repeated questions); see §13.")
    add("")

    add("## 6. Contamination discovery and decontamination")
    add("AI Hub retrieved-knowledge documents were shared between Training and Validation splits; exact-line "
        "checks could not see containment. 65 balanced and 148 Korean-large records were removed (2,953 → 2,888; "
        "4,530 → 4,382); the Stage-2 corpus was additionally screened with 120-character eval probes "
        "(93 training documents removed). All numbers in this report use the decontaminated sets. "
        "Details: `outputs/STAGE2_DECONTAMINATED_EVAL_REPORT.md`.")
    add("")

    add("## 7. Stage-2 corpus")
    add("v3 corpus: 241,749 documents, 151.1M characters; target shares korean_general 45%, korean_technical 30%, "
        "dialogue 10%, code_url 5%, json 5%, skill_manager 3%, sql 2%. The v3 Skill generator was rewritten to "
        "remove templated repetition (dup-4gram 0.2356 → 0.0234). NIKL is blocked at path, source and record level.")
    add("")

    add("## 8. Stage-2 training setup")
    c = r["canonical"]
    add(f"Full unfreeze, tied embedding kept; AdamW (betas {c['optimizer']['betas']}, wd {c['optimizer']['weight_decay']}, "
        f"clip {c['optimizer']['grad_clip']}); embedding LR {c['scheduler']['embedding_peak_lr']}, transformer LR "
        f"{c['scheduler']['transformer_peak_lr']}; {c['scheduler']['description']}; context {c['context_length']}; "
        f"{c['batch']['tokens_per_optimizer_step']:,} tokens/step; {c['precision']}; seed {c['seed']}; {c['backend']}.")
    add("")

    add("## 9. Stop point: 43.876M / 60M tokens")
    add(f"Stopped at step {c['steps']:,} / {c['planned_steps']:,}: **{c['tokens_trained']:,} tokens "
        f"({100 * c['budget_fraction']:.1f}% of 60M)**, via SIGUSR1 on schedule; weights and optimizer state were "
        "saved. The 60M-token endpoint was never reached, and the LR never reached its planned cosine floor.")
    add("")
    add("| Step | Tokens | Balanced NLL/char | Δ vs previous | Δ per 1M tokens | Relative Δ |")
    add("|---:|---:|---:|---:|---:|---:|")
    add(f"| 0 | 0 | {lc['points'][0]['nll_per_character']:.4f} | | | |")
    for rate in lc["intervals"]:
        step = round(rate["to_tokens"] / 16_384)
        add(f"| {step:,} | {rate['to_tokens'] / 1e6:.1f}M | {rate['to_nll']:.4f} | {-rate['improvement']:+.4f} | "
            f"{-rate['improvement_per_1m']:+.4f} | {-rate['relative_improvement_percent']:.2f}% |")
    last = lc["last_interval"]
    h = lc["halves"]
    add("")
    add(f"Facts: the last interval ({last['span_tokens'] / 1e6:.2f}M tokens) changed NLL/char by "
        f"{-last['improvement']:+.4f} ({-last['improvement_per_1m']:+.4f} per 1M tokens). The first half of the run "
        f"(0–20.5M) improved by {h['first_half_0_to_20.5M']['improvement']:.4f} "
        f"({h['first_half_0_to_20.5M']['improvement_per_1m']:.4f}/1M); the second half (20.5–41.0M) by "
        f"{h['second_half_20.5M_to_41.0M']['improvement']:.4f} ({h['second_half_20.5M_to_41.0M']['improvement_per_1m']:.4f}/1M).")
    add("")
    add("**Interpretation:** marginal improvement declined steadily, so the curve shows a saturation tendency on "
        "this eval set. The result at 60M tokens was not measured and is not claimed.")
    add("")

    add("## 10. Balanced eval (decontaminated, 2,888 records)")
    add(_reproduction_table(rep, "balanced"))
    add("")
    add(_category_table(rep))
    add("")
    n_low = rep["stage2_lower_than_qwen_categories"]
    add(f"Fact: Stage-2 had lower NLL/char than Original Qwen in **{n_low} of {rep['balanced_categories']}** "
        "evaluated categories. `json_structured` is the exception (Stage-2 higher, i.e. worse).")
    add("")
    add("Note on `skill_dsl`: Stage-1's 1.1062 is **Interpretation:** likely inflated by similarity between the "
        "v1 synthetic Skill templates in Stage-1 data and the eval records; the Stage-2 v3 generator was "
        "diversified, so the Stage-1 → Stage-2 rise on this category is not read as a capability loss.")
    add("")

    add("## 11. Korean-large eval (decontaminated, 4,382 records)")
    add(_reproduction_table(rep, "korean"))
    add("")
    kb = rep["bootstrap_fixed"]["korean/stage2_minus_original_qwen"]
    add(f"Paired bootstrap (fixed definition) Stage-2 − Qwen: {kb['delta_nll_per_character']:+.4f} "
        f"[95% CI {kb['ci95_low']:+.4f}, {kb['ci95_high']:+.4f}].")
    add("")
    add(_bugfix_paragraph(rep))
    add("")

    add("## 12. NIKL external evaluation")
    add(_nikl_section(r))
    add("")

    add("## 13. Generation / repetition evaluation")
    add(_generation_section(r))
    add("")

    add("## 14. Structured / code / SQL regression analysis")
    add(_structured_section(r))
    add("")

    add("## 15. Parameter / token efficiency")
    add(_efficiency_section(r))
    add("")

    add("## 16. Limitations")
    for item in r["limitations"]:
        add(f"- {item}")
    add("")

    add("## 17. Final conclusion")
    for item in r["conclusion"]:
        add(f"- {item}")
    add("")

    add("## 18. Readiness for Manager-SLM specialization")
    add("| Criterion | Result |")
    add("|---|---|")
    for key, value in r["gate"]["criteria"].items():
        status = "PASS" if value["pass"] else ("FAIL — waived by decision" if key in r["gate"]["waived"] else "FAIL")
        add(f"| {key} | {status} |")
    add("")
    if r["gate"]["closed_by_decision"]:
        d = r["gate"]["decision"]
        add(f"**{r['gate']['verdict']}.** (Nothing beyond this was started.)")
        add("")
        add(f"Decision recorded {d['date']} by {d['decided_by']} (`outputs/final_validation/closure_decision.json`). "
            "The waived criteria are still FAIL as measured; no threshold or metric was changed.")
        for key in r["gate"]["waived"]:
            add(f"- {key}: {d['waived_criteria'][key]}")
    elif r["gate"]["pass"]:
        add("**Stage-2 closed — ready for Manager-SLM dataset/SFT design.** (Nothing beyond this was started.)")
    else:
        add(f"**Stage-2 not closed — blockers remain: {', '.join(r['gate']['blockers'])}.**")
        for key in r["gate"]["blockers"]:
            note = r["gate"]["criteria"][key].get("note")
            if note:
                add(f"- {key}: {note}")
    add("")
    add("## Bugs found and fixed during closure")
    for bug in r["bugs"]:
        add(f"- **{bug['title']}** — cause: {bug['cause']} Before: {bug['before']} After: {bug['after']}")
    add("")
    add("## GGUF export")
    add(r["gguf"]["summary"])
    add("")
    return "\n".join(L) + "\n"


def _reproduction_table(rep: dict, set_name: str) -> str:
    rows = ["| Model | Expected (reported) | Legacy re-run | abs Δ | Fixed (no truncation) | Boundary-prefix sensitivity |",
            "|---|---:|---:|---:|---:|---:|"]
    for name, label in MODEL_LABELS.items():
        chk = rep["reproduction_checks"][f"{set_name}/{name}"]
        prefix = rep["results"][name][set_name]["prefix"]["overall"]["nll_per_character"]
        rows.append(f"| {label} | {chk['expected']:.4f} | {chk['legacy_reproduced']:.4f} | "
                    f"{chk['legacy_abs_diff']:.1e} | {chk['fixed']:.4f} | {prefix:.4f} |")
    return "\n".join(rows)


def _category_table(rep: dict) -> str:
    rows = ["| Category | Original Qwen | Stage-1 | Stage-2 | Stage-2 − Qwen |", "|---|---:|---:|---:|---:|"]
    for category, v in rep["balanced_category_comparison"].items():
        rows.append(f"| {category} | {v['original_qwen']:.4f} | {v['stage1_final']:.4f} | "
                    f"{v['stage2_signal2678']:.4f} | {v['stage2_minus_qwen']:+.4f} |")
    return "\n".join(rows)


def _bugfix_paragraph(rep: dict) -> str:
    over = {n: rep["results"][n]["korean"]["legacy"]["records_over_max_length"] for n in MODEL_LABELS}
    diffs = {n: rep["reproduction_checks"][f"korean/{n}"]["fixed_minus_legacy"] for n in MODEL_LABELS}
    return ("Truncation fix: records over 1,024 tokens per model = "
            + ", ".join(f"{MODEL_LABELS[n]} {over[n]}" for n in MODEL_LABELS)
            + ". Fixed − legacy NLL/char: "
            + ", ".join(f"{MODEL_LABELS[n]} {diffs[n]:+.6f}" for n in MODEL_LABELS) + ".")


def _nikl_section(r: dict) -> str:
    out = []
    nikl = r["nikl"]
    out.append("Provenance: NIKL Newspaper 2025 v1.1 was previously used as the **external validation set of the "
               "tokenizer benchmark**, so it is not a never-before-seen benchmark for this project. It was not "
               "used in any tokenizer corpus, the Stage-1 corpus or the Stage-2 corpus, and not for Stage-2 "
               "checkpoint selection or stopping.")
    prov = r["nikl_provenance"]
    if prov:
        out.append("")
        out.append(f"Exact-paragraph overlap check against the model-training corpora: "
                   f"{prov['overlapping_paragraphs']} of {prov['nikl_paragraphs_checked']:,} NIKL paragraphs "
                   f"({prov['articles_with_overlap']} articles), by corpus {prov['overlap_by_corpus']}; every match is "
                   "the stock sentence 「경찰은 자세한 사고 경위를 조사하고 있다.」. Left in the eval set (negligible).")
    out.append("")
    out.append("Metric: NLL/char = Σ article NLL (natural log, canonical `recovery_eval.evaluate`, no truncation, "
               "articles scored in full) ÷ Σ body characters (Unicode code points of the manifest body, identical for "
               "every model and for the tokenizer benchmark). No BOS; no special token enters the denominator. "
               "bits/char = NLL/char ÷ ln 2 is reported as the perplexity-like quantity; per-token perplexity is not "
               "comparable across tokenizers and is not used.")
    for tag, block in (("full", nikl.get("full")), ("pilot_1pct", nikl.get("pilot"))):
        if not block:
            continue
        out.append("")
        title = ("Full evaluation (all manifest articles)" if tag == "full" else
                 "Deterministic 1% pilot (sha256(article_id) bucket < 100 of 10,000)")
        out.append(f"### {title}{'' if block['complete'] else ' — INCOMPLETE'}")
        out.append("")
        out.append("| Model | Articles | Paragraphs | Characters | Total NLL | NLL/char | bits/char | Tokens | Tokens/char | Chars/token |")
        out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for name, label in MODEL_LABELS.items():
            s = block["overall"][name]
            out.append(f"| {label} | {s['articles']:,} | {s['paragraphs']:,} | {s['characters']:,} | "
                       f"{s['total_nll']:,.0f} | {s['nll_per_character']:.4f} | {s['bits_per_character']:.4f} | "
                       f"{s['total_tokens']:,} | {s['tokens_per_character']:.4f} | {s['characters_per_token']:.3f} |")
        b = block["bootstrap"]
        out.append("")
        out.append(f"Stage-2 − Qwen: {b['stage2_minus_original_qwen']['delta_nll_per_character']:+.4f} "
                   f"[95% CI {b['stage2_minus_original_qwen']['ci95_low']:+.4f}, {b['stage2_minus_original_qwen']['ci95_high']:+.4f}]; "
                   f"Stage-2 − Stage-1: {b['stage2_minus_stage1']['delta_nll_per_character']:+.4f} "
                   f"[{b['stage2_minus_stage1']['ci95_low']:+.4f}, {b['stage2_minus_stage1']['ci95_high']:+.4f}] "
                   "(article-level paired bootstrap, 1,000 resamples, seed 42).")
        out.append("")
        out.append("By topic (NLL/char; n = articles):")
        out.append("")
        out.append("| Topic | n | Qwen | Stage-1 | Stage-2 | Stage-2 − Qwen |")
        out.append("|---|---:|---:|---:|---:|---:|")
        for topic, q in sorted(block["by_topic"]["original_qwen"].items(), key=lambda kv: -kv[1]["articles"]):
            s1 = block["by_topic"]["stage1_final"][topic]
            s2 = block["by_topic"]["stage2_signal2678"][topic]
            out.append(f"| {topic} | {q['articles']:,} | {q['nll_per_character']:.4f} | {s1['nll_per_character']:.4f} | "
                       f"{s2['nll_per_character']:.4f} | {s2['nll_per_character'] - q['nll_per_character']:+.4f} |")
        out.append("")
        out.append("By article length (characters):")
        out.append("")
        out.append("| Length | n | Qwen | Stage-1 | Stage-2 | Stage-2 − Qwen |")
        out.append("|---|---:|---:|---:|---:|---:|")
        for bucket in sorted(block["by_length_bucket"]["original_qwen"], key=lambda s: int(s.split("-")[0].rstrip("+"))):
            q = block["by_length_bucket"]["original_qwen"][bucket]
            s1 = block["by_length_bucket"]["stage1_final"][bucket]
            s2 = block["by_length_bucket"]["stage2_signal2678"][bucket]
            out.append(f"| {bucket} | {q['articles']:,} | {q['nll_per_character']:.4f} | {s1['nll_per_character']:.4f} | "
                       f"{s2['nll_per_character']:.4f} | {s2['nll_per_character'] - q['nll_per_character']:+.4f} |")
        pubs = block["by_publisher"]["stage2_signal2678"]
        wins = sum(pubs[p]["nll_per_character"] < block["by_publisher"]["original_qwen"][p]["nll_per_character"]
                   for p in pubs)
        out.append("")
        out.append(f"Publishers: Stage-2 lower than Qwen for {wins} of {len(pubs)} publishers "
                   f"(per-publisher table in the CSV).")
    sens = nikl.get("prefix")
    if sens:
        out.append("")
        out.append("First-token sensitivity (0.25% subset, `<|endoftext|>` as context-only prefix so every content "
                   "token is scored): " + ", ".join(
                       f"{MODEL_LABELS[n]} {sens['overall'][n]['nll_per_character']:.4f}" for n in MODEL_LABELS)
                   + ".")
    tok = r["nikl_tokens_full"]
    if tok:
        o = tok["overall"]
        out.append("")
        out.append(f"Tokenizer efficiency on the **full** corpus ({tok['articles']:,} articles, {tok['characters']:,} "
                   f"characters, tokens-only pass): Qwen {o['qwen_tokens']:,} tokens ({o['qwen_tokens_per_char']:.4f}/char), "
                   f"KO32K {o['ko32k_tokens']:,} ({o['ko32k_tokens_per_char']:.4f}/char) — "
                   f"**{o['token_reduction_percent']:.2f}% fewer tokens**.")
    return "\n".join(out)


def _generation_section(r: dict) -> str:
    gen = r["generation"]
    if not gen:
        return "Generation evaluation missing."
    s = gen["summary"]
    out = [f"{gen['prompts']} fixed prompts (`{gen['prompts_file']}`, 7 categories × 5 + the 4 verbatim Stage-1 "
           "prompts), greedy plus sampling (temperature 0.6, top-p 0.95, top-k 20 — Qwen3's shipped defaults — "
           f"seeds {gen['decoding']['seeds']}), max {gen['decoding']['max_new_tokens']} new tokens, no repetition "
           "penalty, identical stop strings for every model. Every output is saved in "
           "`outputs/final_validation/generation_samples.json`; nothing was selected.", ""]
    out.append("| Model | Mode | Loop rate | Word dup-4 | Word dup-8 | Token dup-4 | distinct-2 | Longest repeated span (tok, mean) | Premature EOS | Empty | Unicode OK |")
    out.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, label in MODEL_LABELS.items():
        for mode in ("greedy", "sampled"):
            o = s[name][mode]["overall"]
            out.append(f"| {label} | {mode} | {o['loop_detected_rate']:.2f} | {o['mean_word_duplicate_4gram']:.3f} | "
                       f"{o['mean_word_duplicate_8gram']:.3f} | {o['mean_duplicate_4gram_fraction']:.3f} | "
                       f"{o['mean_distinct_2']:.3f} | {o['mean_longest_repeated_span_tokens']:.1f} | "
                       f"{o['premature_eos_rate']:.2f} | {o['empty_rate']:.2f} | {o['unicode_ok_rate']:.2f} |")
    out.append("")
    out.append("Loop rate by category (greedy):")
    out.append("")
    cats = sorted(s["stage2_signal2678"]["greedy"]["by_category"])
    out.append("| Category | " + " | ".join(MODEL_LABELS.values()) + " |")
    out.append("|---|" + "---:|" * len(MODEL_LABELS))
    for cat in cats:
        out.append(f"| {cat} | " + " | ".join(
            f"{s[n]['greedy']['by_category'][cat]['loop_detected_rate']:.2f}" for n in MODEL_LABELS) + " |")
    out.append("")
    out.append("### Stage-1 failure signatures")
    out.append("")
    out.append("| Model | Mode | `10000` repeat | `\"input\":\"sql\"` repeat | phrase loop | prompt-echo loop |")
    out.append("|---|---|---:|---:|---:|---:|")
    for name, label in MODEL_LABELS.items():
        for mode in ("greedy", "sampled"):
            c = s[name][mode]["overall"]["failure_signature_counts"]
            n = s[name][mode]["overall"]["samples"]
            out.append(f"| {label} | {mode} | {c.get('repeated_10000', 0)}/{n} | {c.get('repeated_input_sql', 0)}/{n} | "
                       f"{c.get('phrase_loop', 0)}/{n} | {c.get('prompt_echo_loop', 0)}/{n} |")
    k_g = s["stage2_signal2678"]["greedy"]["overall"]
    k_s = s["stage2_signal2678"]["sampled"]["overall"]
    q_g = s["original_qwen"]["greedy"]["overall"]
    q_s = s["original_qwen"]["sampled"]["overall"]
    k_sig = {m: s["stage2_signal2678"][m]["overall"]["failure_signature_counts"] for m in ("greedy", "sampled")}
    s1_sig = {m: s["stage1_final"][m]["overall"]["failure_signature_counts"] for m in ("greedy", "sampled")}
    out.append("")
    out.append("Findings (facts):")
    out.append(f"- The Stage-1-specific signatures did not reproduce in Stage-2: `10000` repeat "
               f"{k_sig['greedy'].get('repeated_10000', 0) + k_sig['sampled'].get('repeated_10000', 0)} "
               f"(Stage-1: {s1_sig['greedy'].get('repeated_10000', 0) + s1_sig['sampled'].get('repeated_10000', 0)}), "
               f"`\"input\":\"sql\"` repeat "
               f"{k_sig['greedy'].get('repeated_input_sql', 0) + k_sig['sampled'].get('repeated_input_sql', 0)} "
               f"(Stage-1: {s1_sig['greedy'].get('repeated_input_sql', 0) + s1_sig['sampled'].get('repeated_input_sql', 0)}) "
               "across all greedy and sampled decodes; the legacy JSON prompt now closes into valid JSON.")
    out.append(f"- Generic phrase looping was **not** eliminated: Stage-2 greedy loop rate {k_g['loop_detected_rate']:.2f} "
               f"vs Original Qwen {q_g['loop_detected_rate']:.2f}; sampled {k_s['loop_detected_rate']:.2f} vs "
               f"{q_s['loop_detected_rate']:.2f}. The legacy prompts still loop under greedy decoding "
               "(e.g. a repeated date list), so greedy base-model looping sits at roughly Original-Qwen level.")
    out.append("- Stage-1's lower greedy loop rate coincides with far more early stops (EOS rate "
               f"{s['stage1_final']['greedy']['overall']['stopped_by_eos_rate']:.2f} greedy, "
               f"{s['stage1_final']['sampled']['overall']['stopped_by_eos_rate']:.2f} sampled) and shorter outputs.")
    out.append(f"- Unicode: 1 of {k_s['samples']} Stage-2 sampled outputs (`ko_tech_05`, seed 0) contained interior "
               "U+FFFD from byte-fragment tokens that did not form valid UTF-8; 0 in greedy, 0 for Qwen and Stage-1. "
               "KO-32K has 870 byte-fragment tokens, so a Manager decoder should validate or constrain UTF-8.")
    out.append("")
    out.append("Legacy Stage-1 prompts, greedy (verbatim first 160 characters of each continuation):")
    out.append("")
    for row in r["legacy_greedy_samples"]:
        text = row["generated"][:160].replace("\n", "⏎").replace("|", "\\|")
        out.append(f"- {MODEL_LABELS[row['model']]} / `{row['id']}`: {text}")
    return "\n".join(out)


def _structured_section(r: dict) -> str:
    gen = r["generation"]
    rep = r["reproduction"]["balanced_category_comparison"]
    out = ["Likelihood (balanced eval, Stage-2 − Qwen NLL/char): " + ", ".join(
        f"{c} {rep[c]['stage2_minus_qwen']:+.4f}" for c in ("json_structured", "sql", "code_url", "skill_dsl")) + ".",
        ""]
    if gen:
        s = gen["summary"]
        out.append("| Validity (greedy + sampled pooled) | " + " | ".join(MODEL_LABELS.values()) + " |")
        out.append("|---|" + "---:|" * len(MODEL_LABELS))
        for label, cat, key in (("JSON parse valid", "json_structured", "json_parse_valid_rate"),
                                ("JSON schema valid", "json_structured", "json_schema_valid_rate"),
                                ("SQL syntax valid (SQLite parser)", "sql", "sql_syntax_valid_rate"),
                                ("Skill DSL first generated line valid", "skill_dsl", "dsl_first_line_valid_rate"),
                                ("Skill DSL all judged lines valid", "skill_dsl", "dsl_all_lines_valid_rate"),
                                ("Instruction expected answer on first line", "instruction",
                                 "instruction_match_first_line_rate")):
            out.append(f"| {label} | " + " | ".join(f4(pooled(s, n, cat, key)) for n in MODEL_LABELS) + " |")
        v1 = read(FV / "generation_summary_dsl_validator_v1.json")
        if v1:
            out.append("")
            out.append("Skill DSL validator correction: the first grammar (v1) accepted one control token per line and "
                       "rejected closing lines (`… <종료>`), inline chains, numbered/labelled steps and multi-line JSON — "
                       "forms the v3 generator emits. v2 is derived from `corpus/stage2_skill_synth_v2.py` and accepts all "
                       "21,099 v3 training Skill documents. First-line validity v1 → v2: " + ", ".join(
                           f"{MODEL_LABELS[n]} {pooled(v1['summary'], n, 'skill_dsl', 'dsl_first_line_valid_rate'):.2f} → "
                           f"{pooled(s, n, 'skill_dsl', 'dsl_first_line_valid_rate'):.2f}" for n in MODEL_LABELS)
                       + ". The pre-registered gate threshold (0.5) was not changed.")
            out.append("")
            out.append("**Interpretation:** Stage-2 produces well-formed v3 lines after `<요약><이전0>` (e.g. "
                       "`오류: …` / `재시도: <SQL>…` / `… <종료>`), but after `<검색>…`, `<SQL>…` or a bare `<코드>` it "
                       "drifts into prose or date lists. Those four prompts follow the older `eval_skill_dsl.txt` shape, which "
                       "is rare in the v3 corpus, so part of the gap may be prompt/format mismatch; either way the base model "
                       "does not reliably continue Skill DSL, and gate G fails as specified.")
        out.append("")
        out.append("Qwen's tokenizer has no Skill control tokens, so its Skill DSL numbers are shown for reference "
                   "only. These are base-model continuation checks on 5 prompts × 4 decodes per category, not "
                   "instruction-following benchmarks.")
    out.append("")
    out.append("**Interpretation:** JSON likelihood is slightly worse than Qwen (the one balanced category where "
               "Stage-2 trails); SQL and code/URL likelihood are better. These evaluations cover the listed "
               "categories only; broad English ability was not evaluated and is not claimed to be preserved.")
    return "\n".join(out)


def _efficiency_section(r: dict) -> str:
    c = r["structure"]["comparison"]
    st = r["structure"]
    out = ["| | Original Qwen | KO32K Stage-2 | Reduction |", "|---|---:|---:|---:|",
           f"| Vocabulary | {st['original_qwen']['vocab_size']:,} | {st['stage2_signal2678']['vocab_size']:,} | {c['vocab_reduction_percent']:.2f}% |",
           f"| Total parameters | {st['original_qwen']['total_parameters']:,} | {st['stage2_signal2678']['total_parameters']:,} | {c['total_parameter_reduction_percent']:.2f}% |",
           f"| Embedding parameters (tied) | {st['original_qwen']['embedding_parameters']:,} | {st['stage2_signal2678']['embedding_parameters']:,} | {c['embedding_parameter_reduction_percent']:.2f}% |",
           f"| Non-embedding parameters | {st['original_qwen']['non_embedding_parameters']:,} | {st['stage2_signal2678']['non_embedding_parameters']:,} | identical |",
           f"| Weight bytes at bf16 (unique params × 2) | {c['bf16_weight_bytes_equivalent']['original_qwen']:,} | {c['bf16_weight_bytes_equivalent']['stage2_signal2678']:,} | {c['bf16_weight_bytes_reduction_percent']:.2f}% |",
           f"| safetensors as stored | {c['disk_bytes_as_stored']['original_qwen']:,} (bf16, includes a duplicate tied `lm_head`) | {c['disk_bytes_as_stored']['stage2_signal2678']:,} (fp32) | not comparable |",
           ""]
    tok = r["nikl_tokens_full"]
    if tok:
        out.append(f"NIKL full-corpus token reduction vs Qwen tokenizer: **{tok['overall']['token_reduction_percent']:.2f}%** "
                   f"(earlier tokenizer benchmark for the same vocabulary: 35.005%).")
    out.append("Earlier tokenizer benchmarks (same KO-32K vocabulary): FineWeb holdout 34.63%, AI Hub VL 39.97%, "
               "independent 40 sentences 37.77% fewer tokens than Qwen; code/URL and JSON still need more tokens than "
               "Qwen (`outputs/TOKENIZER_STRUCTURED_V2_BENCHMARK.md`).")
    out.append("")
    out.append(c["note"] + " The as-stored Stage-2 file is larger than Qwen's because it is fp32.")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #

def build() -> dict:
    training = read(ROOT / "outputs/stage2_training_report.json")
    structure = read(FV / "structure.json")
    reproduction = read(FV / "reproduction.json")
    generation = read(FV / "generation_summary.json")
    samples = read(FV / "generation_samples.json")
    smoke = read(FV / "smoke_test.json")
    smoke_before = read(FV / "smoke_test_before_metadata_fix.json")
    metadata_fix = read(FV / "checkpoint_metadata_fix.json")
    provenance = read(FV / "nikl_provenance.json")
    tokens_full = read(FV / "nikl_tokenizer_efficiency_full.json")
    gguf = read(FV / "gguf_export.json") or {"attempted": False,
                                              "summary": "Not performed (optional, not a blocker)."}

    canonical = canonical_metadata(training, structure)
    (FV / "canonical_checkpoint.json").write_text(json.dumps(canonical, ensure_ascii=False, indent=2) + "\n",
                                                  encoding="utf-8")
    (ROOT / CANONICAL_DIR / "canonical_metadata.json").write_text(
        json.dumps(canonical, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    audit = special_token_audit()
    (FV / "special_token_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
                                                 encoding="utf-8")

    nikl = {"pilot": nikl_block("pilot_1pct"), "full": nikl_block("full"), "prefix": nikl_block("prefix_0p25pct")}
    csvs = []
    main = nikl["full"] if nikl["full"] and nikl["full"]["complete"] else nikl["pilot"]
    if main:
        prefix = "nikl" if main is nikl["full"] else "nikl_pilot"
        csvs = write_nikl_csvs(main, prefix)
        name = "nikl_full_results.json" if main is nikl["full"] else "nikl_pilot_results.json"
        (FV / name).write_text(json.dumps(main, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    legacy = []
    if samples:
        for model, run in samples["runs"].items():
            legacy += [{"model": model, "id": x["id"], "generated": x["generated"]}
                       for x in run["greedy"] if x["category"] == "legacy_stage1"]

    bugs = [
        {"title": "Evaluation truncation bias (recovery_eval.evaluate)",
         "cause": "records tokenized with truncation=True, max_length=1024 while the full character count stayed in "
                  "the denominator; only one Korean-large record (1,186 KO-32K tokens) exceeded it, and only for "
                  "KO-32K models, which it favoured.",
         "before": "legacy numbers (reproduced exactly, see §10–11)", "after": "fixed numbers in §11; default is now no truncation"},
        {"title": "tokenizer.json carried a serialized truncation state",
         "cause": "the same truncation=True call left the fast tokenizer's backend truncation set, and save_pretrained "
                  "wrote it into every trained KO checkpoint.",
         "before": "truncation {max_length: 1024} in tokenizer.json (harmless through transformers; raw `tokenizers` users would truncate)",
         "after": "null in the canonical checkpoint; tokenization of all 7,270 eval records verified identical"},
        {"title": "generation_config dropped the document-boundary stop token",
         "cause": "transplant kept eos_token_id=2 (<|im_end|>) only; Qwen3 stops on [<|im_end|>, <|endoftext|>].",
         "before": "eos_token_id 2", "after": "eos_token_id [2, 0] (string-equivalent of Qwen's [151645, 151643])"},
        {"title": "config.use_cache=false left over from gradient-checkpointed training",
         "cause": "Stage-2 training disabled the KV cache and saved the flag.", "before": "false", "after": "true"},
        {"title": "NIKL leakage check did not cover model-training corpora",
         "cause": "the earlier check compared NIKL only with the four tokenizer corpora.",
         "before": "unchecked for Stage-1/Stage-2 corpora",
         "after": "checked: 8 of 8.4M paragraphs match one stock sentence in stage2_train_v3 (negligible, kept)"},
        {"title": "Skill DSL validator grammar too narrow (found during this closure)",
         "cause": "the first grammar was written from data/eval_skill_dsl.txt and missed the v3 generator's closing, "
                  "inline-chain, numbered, labelled and multi-line JSON forms (and rejected '<'/'>' inside SQL bodies).",
         "before": "recall against the generator was never checked; Stage-2 DSL first-line validity 0.10 (pooled)",
         "after": "grammar derived from corpus/stage2_skill_synth_v2.py accepts 21,099/21,099 v3 documents; Stage-2 0.20 "
                  "(saved generations re-scored, no regeneration; v1 summary kept as generation_summary_dsl_validator_v1.json)"},
        {"title": "NIKL scorer exhausted memory (MPS allocator growth)",
         "cause": "one evaluate() call per 2,000-article shard with long articles let the MPS caching allocator grow to 77 GB "
                  "and stall on swap (same mechanism as the Stage-2 training incident).",
         "before": "pilot stalled before its first shard; no result written",
         "after": "128-article chunks with torch.mps.empty_cache() between them; peak driver memory logged per shard"},
        {"title": "Duplicate repetition-metric implementation",
         "cause": "repetition_metrics lived in stage1_extension_report.py; the new generation eval would have copied it.",
         "before": "one private definition", "after": "model/generation_eval.py is the single source; the old module imports it"},
    ]

    limitations = [
        "Stage-2 stopped at 43.876M of the planned 60M tokens; what the remaining 16.1M tokens would have done was not measured.",
        "The 60M endpoint, and whether the LR floor would have changed the curve, are unknown.",
        "NIKL was used earlier to validate the tokenizer, so it is an external but not a never-seen benchmark.",
        "Broad English capability was not evaluated. code/URL likelihood is measured; general English prose, reasoning and multilingual ability are not.",
        "JSON likelihood is slightly worse than Original Qwen on the balanced eval.",
        "NLL/char leaves each record's first token unscored while counting its characters; the boundary-prefix sensitivity quantifies this and does not change any ranking reported here.",
        "Generation checks are base-model continuation sanity checks on 39 prompts; they are not instruction-following or task-quality benchmarks.",
        "Parameter/weight-byte reductions are structural; runtime RAM, latency and quantized behaviour were not measured.",
        "No git history exists for this project; provenance relies on file hashes.",
    ]
    if not (nikl["full"] and nikl["full"]["complete"]):
        limitations.insert(0, "NIKL model likelihood was measured on the deterministic 1% pilot only; the full-corpus model "
                              "evaluation had not completed when this report was generated.")

    present = {
        "reproduction.json": reproduction is not None,
        "generation_summary.json": generation is not None,
        "generation_samples.json": samples is not None,
        "smoke_test.json": smoke is not None,
        "structure.json": structure is not None,
        "nikl_provenance.json": provenance is not None,
        "nikl_tokenizer_efficiency_full.json": tokens_full is not None,
        "nikl model results": main is not None,
        "canonical_checkpoint.json": True,
    }
    ctx = {"smoke": smoke, "reproduction": reproduction, "nikl_full": nikl["full"], "generation": generation,
           "special_audit": audit, "artifacts_present": present, "pytest": pytest_result(),
           "nikl_provenance": provenance, "canonical": canonical,
           "nikl_manifest_articles": nikl_manifest_articles(),
           "closure_decision": read(FV / "closure_decision.json")}
    decision = gate(ctx)

    conclusion = []
    if reproduction:
        b = reproduction["results"]
        conclusion.append(
            "Stage-2 achieved lower NLL/char than Original Qwen on both decontaminated evaluations: balanced "
            f"{b['stage2_signal2678']['balanced']['fixed']['overall']['nll_per_character']:.4f} vs "
            f"{b['original_qwen']['balanced']['fixed']['overall']['nll_per_character']:.4f}, Korean-large "
            f"{b['stage2_signal2678']['korean']['fixed']['overall']['nll_per_character']:.4f} vs "
            f"{b['original_qwen']['korean']['fixed']['overall']['nll_per_character']:.4f}.")
        conclusion.append(f"{reproduction['stage2_lower_than_qwen_categories']} of "
                          f"{reproduction['balanced_categories']} balanced categories had lower NLL/char than "
                          "Original Qwen; JSON did not.")
    if main:
        o = main["overall"]
        conclusion.append(
            f"On NIKL ({'full corpus' if main is nikl['full'] else '1% deterministic pilot'}), Stage-2 NLL/char "
            f"{o['stage2_signal2678']['nll_per_character']:.4f} vs Qwen {o['original_qwen']['nll_per_character']:.4f}.")
    conclusion.append("No evidence of catastrophic forgetting was observed on the evaluated categories. This does not "
                      "establish improvement on every capability, and English ability was not evaluated.")
    if generation:
        g = generation["summary"]
        conclusion.append(
            "Generation: the Stage-1 loop signatures (`10000`, `\"input\":\"sql\"`) did not recur, but greedy phrase "
            f"looping remains at roughly Original-Qwen level ({g['stage2_signal2678']['greedy']['overall']['loop_detected_rate']:.2f} "
            f"vs {g['original_qwen']['greedy']['overall']['loop_detected_rate']:.2f}), and base-model Skill DSL continuation "
            f"is weak ({pooled(g, 'stage2_signal2678', 'skill_dsl', 'dsl_first_line_valid_rate'):.2f} first-line validity).")
    conclusion.append("The model uses 20.60% fewer parameters (78.94% fewer embedding parameters) and about 35% "
                      "fewer tokens on Korean newspaper text than Original Qwen.")

    report = {
        "canonical_model": CANONICAL_NAME, "canonical": canonical,
        "training": {"status": training["status"], "tokens_trained": training["tokens_trained"],
                     "budget_tokens": training["budget_tokens"], "baselines": training["baselines"]},
        "learning_curve": learning_curve(training),
        "reproduction": reproduction, "structure": structure,
        "nikl": {k: v for k, v in nikl.items()}, "nikl_provenance": provenance,
        "nikl_tokens_full": tokens_full, "nikl_csvs": csvs,
        "generation": generation, "legacy_greedy_samples": legacy,
        "failure_rows": failure_table(samples) if samples else [],
        "smoke": smoke, "smoke_before_metadata_fix": smoke_before, "metadata_fix": metadata_fix,
        "special_token_audit": audit, "gguf": gguf, "bugs": bugs, "limitations": limitations,
        "conclusion": conclusion, "gate": decision,
        "not_started": ["Manager-SLM dataset", "LoRA config", "SFT", "Manager model training",
                        "Stage-2 resume to 60M"],
    }
    return report


def write_nikl_summary(report: dict) -> None:
    body = ["# NIKL external model evaluation — summary", "", _nikl_section(report), ""]
    (FV / "nikl_summary.md").write_text("\n".join(body), encoding="utf-8")


def main() -> None:
    report = build()
    md = markdown(report)
    # The JSON report drops the bulky per-group NIKL tables (they live in their own files).
    slim = dict(report)
    slim["nikl"] = {k: ({kk: vv for kk, vv in v.items() if kk not in ("by_publisher",)} if v else None)
                    for k, v in report["nikl"].items()}
    (ROOT / "outputs" / "stage2_final_report.json").write_text(
        json.dumps(slim, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ROOT / "outputs" / "stage2_final_report.md").write_text(md, encoding="utf-8")
    write_nikl_summary(report)
    print(json.dumps({"gate": report["gate"]["pass"], "blockers": report["gate"]["blockers"]}, indent=2))


if __name__ == "__main__":
    main()
