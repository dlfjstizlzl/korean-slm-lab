#!/usr/bin/env python3
"""Manager evaluation: RAW and Runtime-grammar-constrained greedy generation,
structural metrics, category breakdowns and failure taxonomy.

Constrained decoding uses the Runtime's own grammar engine: `slm_taskplan_tool
--grammar-session` keeps a llama.cpp grammar state per sequence (built from
`ManagerClient::json_grammar()`), and at every step the highest-logit token
whose bytes the grammar accepts is chosen. End-of-generation is allowed only
when the grammar is complete. That is what llama.cpp's grammar sampler + greedy
selection does in the Runtime (temperature 0).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import LogitsProcessor, LogitsProcessorList

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "model"))

from manager_sft_data import render_prompt, system_prompt  # noqa: E402
from manager_v2_registry import ensure_tool  # noqa: E402
from manager_v2_validator import PlanError, canonical_json, cpp_grammar, parse, validate  # noqa: E402

MAX_NEW_TOKENS = 256  # Runtime --manager-max-tokens default
FIRST_CANDIDATES = 64


# --------------------------------------------------------------------------- #
# token bytes (what llama.cpp's grammar sees for each token)
# --------------------------------------------------------------------------- #

def _byte_decoder() -> dict[str, int]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def token_bytes_table(tokenizer) -> list[bytes]:
    decoder = _byte_decoder()
    table = [b""] * len(tokenizer)
    added = {i: t.content for i, t in tokenizer.added_tokens_decoder.items()}
    for token, index in tokenizer.get_vocab().items():
        if index in added:
            table[index] = added[index].encode("utf-8")
        else:
            table[index] = bytes(decoder[c] for c in token)
    return table


# --------------------------------------------------------------------------- #
# grammar-constrained greedy decoding
# --------------------------------------------------------------------------- #

class GrammarSession:
    def __init__(self):
        self.process = subprocess.Popen([str(ensure_tool()), "--grammar-session"], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, text=True, bufsize=1)

    def call(self, line: str) -> str:
        self.process.stdin.write(line + "\n")
        self.process.stdin.flush()
        return self.process.stdout.readline().strip()

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=10)


class RuntimeGrammarProcessor(LogitsProcessor):
    def __init__(self, session: GrammarSession, table: list[bytes], eos_ids: set[int], batch: int, prompt_len: int):
        self.session, self.table, self.eos_ids = session, table, eos_ids
        self.prompt_len = prompt_len
        self.done = [False] * batch
        self.hex = [b.hex() if b else None for b in table]
        self.all_ids = [i for i, h in enumerate(self.hex) if h is not None or i in eos_ids]
        self.stats = Counter()
        for row in range(batch):
            assert session.call(f"N r{row}") == "OK"

    def _item(self, token: int) -> str:
        return "-" if token in self.eos_ids else self.hex[token]

    def _choose(self, row: int, scores: torch.Tensor) -> int:
        top = torch.topk(scores, FIRST_CANDIDATES).indices.tolist()
        top = [t for t in top if t in self.eos_ids or self.hex[t] is not None]
        verdict = self.session.call(f"T r{row} " + ",".join(self._item(t) for t in top))
        for token, ok in zip(top, verdict):
            if ok == "1":
                self.stats["fast"] += 1
                return token
        self.stats["full_scan"] += 1
        verdict = self.session.call(f"T r{row} " + ",".join(self._item(t) for t in self.all_ids))
        allowed = [t for t, ok in zip(self.all_ids, verdict) if ok == "1"]
        if not allowed:
            self.stats["dead_end"] += 1
            return next(iter(self.eos_ids))
        return max(allowed, key=lambda t: scores[t].item())

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        step = input_ids.shape[1] - self.prompt_len
        cpu_scores = scores.detach().float().cpu()
        for row in range(input_ids.shape[0]):
            if self.done[row]:
                continue
            if step > 0:
                last = int(input_ids[row, -1])
                if last in self.eos_ids:
                    self.done[row] = True
                    continue
                reply = self.session.call(f"A r{row} {self.hex[last]}")
                if reply != "OK":
                    raise RuntimeError(f"grammar rejected a chosen token: {reply}")
            chosen = self._choose(row, cpu_scores[row])
            scores[row].fill_(float("-inf"))
            scores[row, chosen] = 0.0
        return scores


@torch.no_grad()
def generate(model, tokenizer, requests: list[str], *, constrained: bool, batch_size: int = 16,
             device: str = "mps", table: list[bytes] | None = None) -> tuple[list[str], dict]:
    prompt = system_prompt()
    eos_ids = {tokenizer.eos_token_id, tokenizer.pad_token_id}
    tokenizer.padding_side = "left"
    outputs: list[str] = []
    session = GrammarSession() if constrained else None
    table = table or (token_bytes_table(tokenizer) if constrained else None)
    stats = Counter()
    started = time.perf_counter()
    generated_tokens = 0
    try:
        for start in range(0, len(requests), batch_size):
            chunk = [render_prompt(r, prompt) for r in requests[start:start + batch_size]]
            batch = tokenizer(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            processors = LogitsProcessorList()
            processor = None
            if constrained:
                processor = RuntimeGrammarProcessor(session, table, eos_ids, len(chunk), batch["input_ids"].shape[1])
                processors.append(processor)
            result = model.generate(**batch, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, temperature=None,
                                    top_p=None, top_k=None, eos_token_id=sorted(eos_ids),
                                    pad_token_id=tokenizer.pad_token_id, logits_processor=processors)
            new = result[:, batch["input_ids"].shape[1]:]
            for row in new.tolist():
                kept = []
                for token in row:
                    if token in eos_ids:
                        break
                    kept.append(token)
                generated_tokens += len(kept)
                outputs.append(tokenizer.decode(kept, skip_special_tokens=False))
            if processor:
                stats.update(processor.stats)
                for row in range(len(chunk)):
                    session.call(f"D r{row}")
    finally:
        if session:
            session.close()
    elapsed = time.perf_counter() - started
    return outputs, {"seconds": round(elapsed, 1), "generated_tokens": generated_tokens,
                     "tokens_per_second": round(generated_tokens / elapsed, 2) if elapsed else None,
                     "grammar_steps": dict(stats)}


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def _structure(plan: dict) -> tuple:
    return (plan["status"], tuple(t["skill"] for t in plan["tasks"]),
            tuple(tuple(map(str, t["sources"])) for t in plan["tasks"]), tuple(plan["outputs"]))


def _edges(plan: dict | None) -> set:
    if not plan or plan["status"] != "ok":
        return set()
    edges = {(str(s), t["id"]) for t in plan["tasks"] for s in t["sources"]}
    return edges | {("out", o) for o in plan["outputs"]}


def score_row(row: dict, output: str, grammar_ok: bool) -> dict:
    gold = row["plan"]
    text = output.strip()
    record = {"id": row["id"], "output": output, "grammar_valid": grammar_ok}
    try:
        parse(text)
        record["json_valid"] = True
    except PlanError as error:
        record["json_valid"] = error.code not in ("NO_JSON_OBJECT", "JSON_PARSE_ERROR", "INVALID_TEXT")
    result = validate(text, mode="runtime")
    pred = result.plan if result.valid else None
    record["schema_valid"] = result.valid
    record["error_code"] = None if result.valid else result.code
    record["canonical_valid"] = bool(pred) and text == canonical_json(pred)
    record["pred_plan"] = pred
    record["canonical_exact"] = text == row["plan_text"]
    record["status_correct"] = bool(pred) and pred["status"] == gold["status"]
    record["pred_status"] = pred["status"] if pred else "invalid"
    if pred:
        gs, ps = _structure(gold), _structure(pred)
        record["task_count_correct"] = len(pred["tasks"]) == len(gold["tasks"])
        record["skill_sequence_correct"] = gs[1] == ps[1]
        record["sources_exact"] = gs[2] == ps[2]
        record["outputs_exact"] = gs[3] == ps[3]
        record["topology_exact"] = gs[1:] == ps[1:]
        record["structural_exact"] = gs == ps
        aligned = list(zip(gold["tasks"], pred["tasks"]))
        record["aligned_skill_matches"] = sum(g["skill"] == p["skill"] for g, p in aligned)
        record["aligned_instruction_matches"] = sum(g["skill"] == p["skill"] and g["instruction"] == p["instruction"]
                                                    for g, p in aligned)
        record["instruction_exact"] = record["structural_exact"] and (
            [t["instruction"] for t in gold["tasks"]] == [t["instruction"] for t in pred["tasks"]]
            and gold.get("message") == pred.get("message"))
    else:
        for key in ("task_count_correct", "skill_sequence_correct", "sources_exact", "outputs_exact", "topology_exact",
                    "structural_exact", "instruction_exact"):
            record[key] = False
        record["aligned_skill_matches"] = record["aligned_instruction_matches"] = 0
    record["gold_tasks"] = len(gold["tasks"])
    gold_edges, pred_edges = _edges(gold), _edges(pred)
    record["edges_tp"] = len(gold_edges & pred_edges)
    record["edges_pred"] = len(pred_edges)
    record["edges_gold"] = len(gold_edges)
    record["failures"] = failure_categories(row, record)
    return record


def failure_categories(row: dict, record: dict) -> list[str]:
    if record["canonical_exact"]:
        return []
    gold, pred = row["plan"], record["pred_plan"]
    out = []
    if not record["json_valid"]:
        return ["malformed_json"]
    if pred is None:
        return ["invalid_taskplan_schema"]
    if pred["status"] != gold["status"]:
        out.append("wrong_status")
        if gold["status"] == "ok" and pred["status"] == "clarify":
            out.append("clarify_instead_of_plan")
        if gold["status"] == "clarify" and pred["status"] == "ok":
            out.append("plan_instead_of_clarify")
        if "unsupported" in (gold["status"], pred["status"]):
            out.append("unsupported_confusion")
        return out
    if record["structural_exact"]:
        return ["instruction_mismatch"]
    gold_skills = [t["skill"] for t in gold["tasks"]]
    pred_skills = [t["skill"] for t in pred["tasks"]]
    if len(pred_skills) < len(gold_skills):
        out.append("missing_task")
    if len(pred_skills) > len(gold_skills):
        out.append("extra_task")
    if set(pred_skills) - set(gold_skills):
        out.append("unnecessary_skill")
    if len(pred_skills) == len(gold_skills) and pred_skills != gold_skills:
        out.append("wrong_ordering" if sorted(pred_skills) == sorted(gold_skills) else "wrong_skill")
    elif len(pred_skills) != len(gold_skills) and Counter(pred_skills) != Counter(gold_skills) \
            and not (set(pred_skills) - set(gold_skills)):
        out.append("wrong_skill")
    if len(pred_skills) == len(gold_skills) and pred_skills == gold_skills:
        if [t["sources"] for t in pred["tasks"]] != [t["sources"] for t in gold["tasks"]]:
            out.append("wrong_source")
            if any(len(p["sources"]) < len(g["sources"]) for g, p in zip(gold["tasks"], pred["tasks"])):
                out.append("missing_source")
        if pred["outputs"] != gold["outputs"]:
            out.append("wrong_output")
    out.append("valid_but_semantically_wrong")
    return out


def _rate(records, key):
    return round(sum(bool(r[key]) for r in records) / len(records), 4) if records else None


def aggregate(records: list[dict]) -> dict:
    if not records:
        return {"n": 0}
    gold_tasks = sum(r["gold_tasks"] for r in records)
    tp = sum(r["edges_tp"] for r in records)
    precision = tp / max(1, sum(r["edges_pred"] for r in records))
    recall = tp / max(1, sum(r["edges_gold"] for r in records))
    return {
        "n": len(records),
        "json_valid": _rate(records, "json_valid"),
        "schema_valid": _rate(records, "schema_valid"),
        "canonical_valid": _rate(records, "canonical_valid"),
        "grammar_valid": _rate(records, "grammar_valid"),
        "status_accuracy": _rate(records, "status_correct"),
        "task_count_accuracy": _rate(records, "task_count_correct"),
        "skill_sequence_accuracy": _rate(records, "skill_sequence_correct"),
        "skill_accuracy_aligned": round(sum(r["aligned_skill_matches"] for r in records) / gold_tasks, 4) if gold_tasks else None,
        "sources_exact": _rate(records, "sources_exact"),
        "outputs_exact": _rate(records, "outputs_exact"),
        "topology_exact": _rate(records, "topology_exact"),
        "structural_exact": _rate(records, "structural_exact"),
        "instruction_exact": _rate(records, "instruction_exact"),
        "task_instruction_accuracy_aligned": round(sum(r["aligned_instruction_matches"] for r in records) / gold_tasks, 4)
        if gold_tasks else None,
        "canonical_exact": _rate(records, "canonical_exact"),
        "dependency_edge_precision": round(precision, 4),
        "dependency_edge_recall": round(recall, 4),
        "dependency_edge_f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
    }


def categories(row: dict) -> list[str]:
    plan, meta = row["plan"], row["metadata"]
    tags = [f"status:{plan['status']}", f"group:{meta['taxonomy_group']}", f"taxonomy:{meta['taxonomy']}"]
    tags += [f"skill:{s}" for s in sorted({t['skill'] for t in plan["tasks"]})]
    n = len(plan["tasks"])
    tags.append("tasks:0" if n == 0 else "tasks:1" if n == 1 else "tasks:2" if n == 2 else "tasks:3+")
    if plan["status"] == "ok":
        tags.append("sources:multi-source" if any(len(t["sources"]) > 1 for t in plan["tasks"]) else "sources:single-source")
        if len(plan["outputs"]) > 1:
            tags.append("outputs:multiple")
    tags.append("hard" if meta["hard_case"] else "normal")
    return tags


def evaluate(rows: list[dict], outputs: list[str]) -> dict:
    grammar = cpp_grammar([o.strip() for o in outputs])
    records = [score_row(row, out, g) for row, out, g in zip(rows, outputs, grammar)]
    groups = defaultdict(list)
    for row, record in zip(rows, records):
        for tag in categories(row):
            groups[tag].append(record)
    per_status = {s: _rate([r for row, r in zip(rows, records) if row["plan"]["status"] == s], "status_correct")
                  for s in ("ok", "clarify", "unsupported")}
    failures = Counter(f for r in records for f in r["failures"])
    return {"overall": aggregate(records),
            "status_recall": per_status,
            "by_category": {tag: aggregate(group) for tag, group in sorted(groups.items())},
            "failure_counts": dict(failures.most_common()),
            "records": records}
