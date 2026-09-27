#!/usr/bin/env python3
"""Reproducible Manager-SLM pilot dataset generator (candidates only).

Pipeline per sample
-------------------
1. semantic spec   taxonomy + entity (topic or text span) + split, chosen first
2. canonical plan  built from the spec alone (`plan_for`), never from the text
3. realization     a family (syntactic frame) renders the spec into Korean
4. validation      Runtime-parity TaskPlan validator + grounding + UTF-8
5. deduplication   exact and normalized request duplicates are rejected
6. split           decided *before* realization: every template family and
                   every source document belongs to exactly one split, and
                   paraphrase siblings reuse their base sample's split

Nothing here is gold. Every row is `status: candidate` until a human approves
it. Rows whose correct plan depends on an undecided policy carry
`requires_human_policy: true`, `plan: null`, split `excluded_policy`, and
`training_candidate: false`.

The generator stops at exactly `--count` accepted rows. It never over-generates
to pick the best ones; rejected attempts are counted with their reason.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "model"))
sys.path.insert(0, str(PROJECT_ROOT / "corpus"))

from manager_dataset_templates import (  # noqa: E402
    FAMILIES,
    REGISTERS,
    TAXONOMY,
    TAXONOMY_BY_NAME,
    Family,
    Realizer,
    families_for,
    referenced_skills,
    tidy,
)
from manager_plan_validator import canonical_json, plan_depth, validate_plan  # noqa: E402
from manager_runtime_registry import DEFAULT_RUNTIME_ROOT, load_registry  # noqa: E402
from stage2_skill_synth_v2 import SENTENCE_SPLIT, clean_candidate  # noqa: E402

GENERATOR_VERSION = "manager-pilot-gen-0.1.0"
DEFAULT_SEED = 20260924
DEFAULT_COUNT = 200
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "processed" / "aihub_documents_train.txt"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "manager" / "pilot_v1.jsonl"
DEFAULT_TAXONOMY_OUT = PROJECT_ROOT / "outputs" / "manager_dataset" / "task_taxonomy.json"

SPLITS = ("train", "validation", "test")
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
POLICY_SPLIT = "excluded_policy"
POOL_SIZE = 420
POOL_SHARES = {"train": 0.62, "validation": 0.15, "test": 0.15, POLICY_SPLIT: 0.08}
PAYLOAD_CHARS = (30, 150)
TOPIC_CHARS = (2, 20)
MAX_REQUEST_CHARS = 420
MAX_ATTEMPTS = 25
SIBLING_SHARE = 0.10
ENTITY_REUSE_PROBABILITY = 0.15
INPUT_POLICY = "verbatim_minimal_span_v0"
LABEL_STATUS = "candidate"

# --------------------------------------------------------------------------- #
# Source pool
# --------------------------------------------------------------------------- #


def _split_title(line: str) -> tuple[str, str] | None:
    """AI Hub document lines start with the title repeated: 'T T()는 ...'."""
    words = line.split(" ")
    for size in range(1, 7):
        title = " ".join(words[:size])
        rest = " ".join(words[size:])
        if rest.startswith(title) and len(rest) > len(title):
            return title, rest
    return None


def _hangul_ratio(text: str) -> float:
    return sum("가" <= char <= "힣" for char in text) / max(1, len(text))


def build_pool(path: Path, seed: int, size: int = POOL_SIZE) -> tuple[list[dict], dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    order = list(range(len(lines)))
    random.Random(f"{seed}:pool").shuffle(order)
    rejects: Counter = Counter()
    seen_topics: set[str] = set()
    pool: list[dict] = []
    for line_no in order:
        if len(pool) >= size:
            break
        line = unicodedata.normalize("NFC", lines[line_no])
        split = _split_title(line)
        if split is None:
            rejects["no_repeated_title"] += 1
            continue
        title, body = split
        title = clean_candidate(title)
        if not TOPIC_CHARS[0] <= len(title) <= TOPIC_CHARS[1]:
            rejects["topic_length"] += 1
            continue
        if not ("가" <= title[-1] <= "힣") or re.search(r"[()\[\]{}\"'“”‘’:/]", title):
            rejects["topic_shape"] += 1
            continue
        if title in seen_topics:
            rejects["duplicate_topic"] += 1
            continue
        sentences = [clean_candidate(piece) for piece in SENTENCE_SPLIT.split(body)]
        sentences = [piece for piece in sentences if len(piece) >= 12]
        rng = random.Random(f"{seed}:pool:{line_no}")
        take = rng.choice((1, 2)) if len(sentences) >= 2 else 1
        payload = ". ".join(sentences[:take]).strip()
        if payload:
            payload += "."
        if not PAYLOAD_CHARS[0] <= len(payload) <= PAYLOAD_CHARS[1]:
            if take == 2 and sentences:
                payload = sentences[0] + "."
            if not PAYLOAD_CHARS[0] <= len(payload) <= PAYLOAD_CHARS[1]:
                rejects["payload_length"] += 1
                continue
        if _hangul_ratio(payload) < 0.45:
            rejects["payload_not_korean"] += 1
            continue
        if re.search(r"[\"“”\\�\x00-\x1f]", payload) or "�" in title:
            rejects["payload_characters"] += 1
            continue
        seen_topics.add(title)
        pool.append({"doc_id": f"{path.name}:L{line_no + 1}", "topic": title, "text": payload})
    stats = {"source": str(path.relative_to(PROJECT_ROOT)), "source_lines": len(lines),
             "requested": size, "accepted": len(pool), "rejected": dict(rejects)}
    return pool, stats


def partition_pool(pool: list[dict]) -> dict[str, list[dict]]:
    names = list(POOL_SHARES)
    counts = largest_remainder(len(pool), POOL_SHARES)
    result, start = {}, 0
    for name in names:
        result[name] = pool[start:start + counts[name]]
        start += counts[name]
    return result


# --------------------------------------------------------------------------- #
# Quotas and splits
# --------------------------------------------------------------------------- #


def largest_remainder(total: int, weights: dict[str, float]) -> dict[str, int]:
    norm = sum(weights.values())
    raw = {key: total * value / norm for key, value in weights.items()}
    counts = {key: math.floor(value) for key, value in raw.items()}
    leftover = total - sum(counts.values())
    for key in sorted(raw, key=lambda k: (-(raw[k] - counts[k]), list(weights).index(k)))[:leftover]:
        counts[key] += 1
    return counts


def scaled_quotas(total: int) -> dict[str, int]:
    return largest_remainder(total, {item.name: item.quota for item in TAXONOMY})


def assign_family_splits(seed: int) -> dict[str, str]:
    """Each labeled template family lives in exactly one split."""
    assignment: dict[str, str] = {}
    for taxonomy in TAXONOMY:
        families = [family.id for family in families_for(taxonomy.name)]
        if taxonomy.group == "policy_pending":
            assignment.update({family: POLICY_SPLIT for family in families})
            continue
        random.Random(f"{seed}:families:{taxonomy.name}").shuffle(families)
        n = len(families)
        n_val = max(1, round(n * SPLIT_RATIOS["validation"]))
        n_test = max(1, round(n * SPLIT_RATIOS["test"]))
        for index, family in enumerate(families):
            if index < n_val:
                assignment[family] = "validation"
            elif index < n_val + n_test:
                assignment[family] = "test"
            else:
                assignment[family] = "train"
    return assignment


# --------------------------------------------------------------------------- #
# Semantic spec -> canonical plan
# --------------------------------------------------------------------------- #


def steps_for(taxonomy_name: str, family: Family) -> tuple[tuple[str, str], ...]:
    return family.steps or TAXONOMY_BY_NAME[taxonomy_name].steps


def plan_for(steps, args: dict[str, str]) -> dict:
    tasks = []
    for index, (skill, source) in enumerate(steps):
        task: dict = {"id": index, "skill": skill}
        if source.startswith("task:"):
            task["input_from"] = int(source.split(":", 1)[1])
        else:
            task["input"] = args[source]
        tasks.append(task)
    return {"tasks": tasks}


def normalize_request(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    return "".join(char for char in text if char.isalnum())


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #


class PilotGenerator:
    def __init__(self, seed: int, count: int, source: Path, runtime_root: Path):
        self.seed = seed
        self.count = count
        self.registry = load_registry(runtime_root)
        unknown = referenced_skills() - set(self.registry.skills)
        if unknown:
            raise RuntimeError(f"taxonomy references skills absent from the Runtime: {sorted(unknown)}")
        self.pool, self.pool_stats = build_pool(source, seed)
        self.partitions = partition_pool(self.pool)
        self.family_split = assign_family_splits(seed)
        self.quotas = scaled_quotas(count)
        self.cursor = {name: 0 for name in self.partitions}
        self.doc_uses: dict[str, set] = defaultdict(set)
        self.exact: set[str] = set()
        self.normalized: set[str] = set()
        self.rows: list[dict] = []
        self.attempts = 0
        self.rejects: Counter = Counter()
        self.rejects_by_taxonomy: dict[str, Counter] = defaultdict(Counter)
        self.shortfall: list[dict] = []

    # entities -------------------------------------------------------------
    def draw_doc(self, split: str, taxonomy: str, kind: str, rng: random.Random,
                 exclude: set[str] = frozenset()) -> dict:
        pool = self.partitions[split]
        reusable = [doc for doc in pool[:self.cursor[split]]
                    if doc["doc_id"] not in exclude and len(self.doc_uses[doc["doc_id"]]) < 2
                    and (taxonomy, kind) not in self.doc_uses[doc["doc_id"]]
                    and all(t != taxonomy for t, _ in self.doc_uses[doc["doc_id"]])]
        if reusable and rng.random() < ENTITY_REUSE_PROBABILITY:
            doc = reusable[rng.randrange(len(reusable))]
        else:
            if self.cursor[split] >= len(pool):
                raise RuntimeError(f"entity pool for split {split} exhausted")
            doc = pool[self.cursor[split]]
            self.cursor[split] += 1
        self.doc_uses[doc["doc_id"]].add((taxonomy, kind))
        return doc

    def make_spec(self, index: int, taxonomy: str, family: Family, split: str) -> dict:
        rng = random.Random(f"{self.seed}:spec:{index}")
        args: dict[str, str] = {}
        entities: list[str] = []
        if family.source == "authored":
            used = sum(1 for row in self.rows if row["metadata"]["template_family"] == family.id)
            args["authored"] = family.authored[used % len(family.authored)]
            entities.append(f"authored:{family.id}:{used % len(family.authored)}")
        elif family.source == "text2":
            first = self.draw_doc(split, taxonomy, "text", rng)
            second = self.draw_doc(split, taxonomy, "text2", rng, exclude={first["doc_id"]})
            args.update(text=first["text"], text2=second["text"])
            entities += [first["doc_id"], second["doc_id"]]
        else:
            doc = self.draw_doc(split, taxonomy, family.source, rng)
            args[family.source] = doc[family.source]
            args["source"] = doc[family.source]
            entities.append(doc["doc_id"])
        return {"semantic_id": f"sem-{index:04d}", "taxonomy": taxonomy, "source_kind": family.source,
                "args": args, "entity_family": entities, "split": split}

    # realization ----------------------------------------------------------
    def realize(self, spec: dict, family: Family, source_type: str, paraphrase_of: str | None) -> bool:
        taxonomy = TAXONOMY_BY_NAME[spec["taxonomy"]]
        policy = taxonomy.group == "policy_pending"
        slot = len(self.rows)
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self.attempts += 1
            rng = random.Random(f"{self.seed}:realize:{spec['semantic_id']}:{family.id}:{slot}:{attempt}")
            register = REGISTERS[rng.randrange(len(REGISTERS))]
            realizer = Realizer(rng, register, dict(spec["args"]))
            request = tidy(unicodedata.normalize("NFC", family.render(realizer)))

            reason = None
            plan = None
            validation = None
            candidate_plans = []
            if "�" in request:
                reason = "UTF8_ERROR"
            elif len(request) > MAX_REQUEST_CHARS:
                reason = "REQUEST_TOO_LONG"
            elif request in self.exact:
                reason = "DUPLICATE_REQUEST_EXACT"
            elif normalize_request(request) in self.normalized:
                reason = "DUPLICATE_REQUEST_NORMALIZED"
            if reason is None and not policy:
                plan = plan_for(steps_for(taxonomy.name, family), spec["args"])
                result = validate_plan(plan, self.registry, request=request)
                validation = result.to_dict()
                if not result.valid:
                    reason = "INVALID_PLAN:" + ",".join(result.codes)
            if reason is None and policy:
                args = dict(spec["args"], request=request)
                for name, steps in taxonomy.candidate_plans:
                    candidate = plan_for(steps, args) if steps else None
                    check = validate_plan(candidate, self.registry, request=request).to_dict() if candidate else None
                    if check and not check["valid"]:
                        reason = "INVALID_CANDIDATE_PLAN"
                    candidate_plans.append({"policy": name, "plan": candidate, "validation": check})
                validation = {"valid": True, "errors": [], "warnings": [],
                              "note": "no label; plan withheld pending human policy"}
            if reason:
                self.rejects[reason.split(":")[0]] += 1
                self.rejects_by_taxonomy[taxonomy.name][reason] += 1
                continue

            self.exact.add(request)
            self.normalized.add(normalize_request(request))
            number = len(self.rows) + 1
            skills = [task["skill"] for task in plan["tasks"]] if plan else []
            self.rows.append({
                "id": f"mgr-pilot-v1-{number:04d}",
                "request": request,
                "plan": plan,
                "plan_text": canonical_json(plan) if plan else None,
                "metadata": {
                    "sample_id": f"mgr-pilot-v1-{number:04d}",
                    "status": LABEL_STATUS,
                    "source_type": source_type,
                    "taxonomy": taxonomy.name,
                    "taxonomy_group": taxonomy.group,
                    "hard_case_category": taxonomy.hard_case_category,
                    "template_family": family.id,
                    "semantic_id": spec["semantic_id"],
                    "entity_family": spec["entity_family"],
                    "source_kind": spec["source_kind"],
                    "arguments": {key: value for key, value in spec["args"].items()
                                  if key in ("topic", "text", "text2")},
                    "skills_used": skills,
                    "task_count": len(skills),
                    "plan_depth": plan_depth(plan) if plan else 0,
                    "split": spec["split"],
                    "paraphrased": source_type == "paraphrased",
                    "paraphrase_of": paraphrase_of,
                    "requires_human_policy": policy,
                    "training_candidate": not policy,
                    "candidate_plans": candidate_plans or None,
                    "input_policy": INPUT_POLICY,
                    "register": register,
                    "realization_trace": realizer.trace,
                    "generator_version": GENERATOR_VERSION,
                    "seed": self.seed,
                    "attempt": attempt,
                    "validation": validation,
                },
            })
            return True
        self.shortfall.append({"taxonomy": taxonomy.name, "family": family.id,
                               "semantic_id": spec["semantic_id"]})
        return False

    # driver ---------------------------------------------------------------
    def generate(self) -> list[dict]:
        spec_index = 0
        for taxonomy in TAXONOMY:
            quota = self.quotas[taxonomy.name]
            if quota == 0:
                continue
            families = families_for(taxonomy.name)
            if taxonomy.group == "policy_pending":
                order = list(families)
                random.Random(f"{self.seed}:order:{taxonomy.name}").shuffle(order)
                for k in range(quota):
                    family = order[k % len(order)]
                    spec = self.make_spec(spec_index, taxonomy.name, family, POLICY_SPLIT)
                    spec_index += 1
                    self.realize(spec, family, "hard_case", None)
                continue
            split_quota = largest_remainder(quota, SPLIT_RATIOS)
            source_type = "hard_case" if taxonomy.group == "hard_labeled" else "rule_generated"
            for split in SPLITS:
                cell = split_quota[split]
                split_families = [f for f in families if self.family_split[f.id] == split]
                random.Random(f"{self.seed}:order:{taxonomy.name}:{split}").shuffle(split_families)
                siblings = math.floor(cell * SIBLING_SHARE) if taxonomy.group == "normal" else 0
                bases = []
                for k in range(cell - siblings):
                    family = split_families[k % len(split_families)]
                    spec = self.make_spec(spec_index, taxonomy.name, family, split)
                    spec_index += 1
                    if self.realize(spec, family, source_type, None):
                        bases.append((spec, family, self.rows[-1]["id"]))
                rng = random.Random(f"{self.seed}:siblings:{taxonomy.name}:{split}")
                for _ in range(siblings):
                    spec, base_family, base_id = bases[rng.randrange(len(bases))]
                    others = [f for f in split_families if f.id != base_family.id and f.source == base_family.source]
                    family = others[rng.randrange(len(others))] if others else base_family
                    self.realize(spec, family, "paraphrased", base_id)
        return self.rows

    def log(self) -> dict:
        return {
            "generator_version": GENERATOR_VERSION,
            "seed": self.seed,
            "requested_count": self.count,
            "accepted_count": len(self.rows),
            "attempts": self.attempts,
            "rejected_attempts": sum(self.rejects.values()),
            "reject_reasons": dict(self.rejects),
            "reject_reasons_by_taxonomy": {k: dict(v) for k, v in self.rejects_by_taxonomy.items()},
            "shortfall": self.shortfall,
            "quotas": self.quotas,
            "family_split": self.family_split,
            "entity_pool": self.pool_stats,
            "entity_pool_partition": {k: len(v) for k, v in self.partitions.items()},
            "entity_pool_used": {k: self.cursor[k] for k in self.cursor},
            "runtime_registry": self.registry.to_dict(),
            "config": {"split_ratios": SPLIT_RATIOS, "pool_shares": POOL_SHARES,
                       "payload_chars": PAYLOAD_CHARS, "max_request_chars": MAX_REQUEST_CHARS,
                       "max_attempts": MAX_ATTEMPTS, "sibling_share": SIBLING_SHARE,
                       "entity_reuse_probability": ENTITY_REUSE_PROBABILITY,
                       "input_policy": INPUT_POLICY},
        }


def taxonomy_document(generator: PilotGenerator) -> dict:
    items = []
    for taxonomy in TAXONOMY:
        families = families_for(taxonomy.name)
        example = next((row for row in generator.rows
                        if row["metadata"]["taxonomy"] == taxonomy.name
                        and row["metadata"]["source_type"] != "paraphrased"), None)
        skills = sorted({s for f in families for s, _ in steps_for(taxonomy.name, f) if s != "?"}
                        | {s for s, _ in taxonomy.steps if s != "?"}
                        | {s for _, p in taxonomy.candidate_plans if p for s, _ in p})
        depth = max((plan_depth(plan_for(steps_for(taxonomy.name, f),
                                         {"topic": "t", "text": "t", "source": "t", "text2": "t"}))
                     for f in families if steps_for(taxonomy.name, f)), default=0)
        items.append({
            "name": taxonomy.name,
            "group": taxonomy.group,
            "description": taxonomy.description,
            "difficulty": taxonomy.difficulty,
            "skills": skills,
            "dependency_depth": depth,
            "plan_shape": [list(step) for step in taxonomy.steps],
            "hard_case_category": taxonomy.hard_case_category,
            "requires_human_policy": taxonomy.group == "policy_pending",
            "candidate_plans": [{"policy": name, "shape": [list(s) for s in plan] if plan else None}
                                for name, plan in taxonomy.candidate_plans],
            "target_ratio_candidate": round(taxonomy.quota / sum(t.quota for t in TAXONOMY), 4),
            "pilot_quota": generator.quotas[taxonomy.name],
            "families": [{"id": f.id, "source": f.source, "split": generator.family_split[f.id],
                          "plan_shape": [list(s) for s in steps_for(taxonomy.name, f)] or None,
                          "note": f.note or None} for f in families],
            "representative_request": example["request"] if example else None,
            "canonical_plan": example["plan_text"] if example else None,
            "notes": taxonomy.notes or None,
        })
    return {"generator_version": GENERATOR_VERSION, "status": "candidate taxonomy, not approved",
            "runtime_skills": list(generator.registry.skills), "taxonomy": items}


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def generate(count: int = DEFAULT_COUNT, seed: int = DEFAULT_SEED, source: Path = DEFAULT_SOURCE,
             runtime_root: Path = DEFAULT_RUNTIME_ROOT) -> PilotGenerator:
    generator = PilotGenerator(seed, count, Path(source), Path(runtime_root))
    generator.generate()
    return generator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--taxonomy-out", type=Path, default=DEFAULT_TAXONOMY_OUT)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()

    if arguments.count > DEFAULT_COUNT:
        parser.error(f"pilot stage is capped at {DEFAULT_COUNT} samples")
    if arguments.output.exists() and not arguments.overwrite:
        parser.error(f"{arguments.output} exists; generation is deterministic, rerun with --overwrite")

    generator = generate(arguments.count, arguments.seed, arguments.source, arguments.runtime_root)
    log = generator.log()
    lines = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in generator.rows)
    log_path = arguments.output.with_suffix(".generation_log.json")
    if len(generator.rows) != arguments.count:
        _atomic_write(arguments.output.with_suffix(".partial.jsonl"), lines)
        _atomic_write(log_path, json.dumps(log, ensure_ascii=False, indent=2))
        print(f"FAILED: {len(generator.rows)}/{arguments.count} rows; see {log_path}", file=sys.stderr)
        return 2
    _atomic_write(arguments.output, lines)
    _atomic_write(log_path, json.dumps(log, ensure_ascii=False, indent=2))
    _atomic_write(arguments.taxonomy_out, json.dumps(taxonomy_document(generator), ensure_ascii=False, indent=2))
    print(f"wrote {len(generator.rows)} rows to {arguments.output} "
          f"(attempts={log['attempts']}, rejected={log['rejected_attempts']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
