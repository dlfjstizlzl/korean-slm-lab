#!/usr/bin/env python3
"""Manager v1 / TaskPlan v2 dataset generator (candidates only; no training).

Per sample
----------
1. semantic spec   taxonomy -> family -> intent + slots (entities drawn from
                   the split's own partition)
2. canonical plan  INTENTS[intent](slots) - never looks at the request text
3. realization     the family renders the spec; a verbosity layer independent
                   of the plan adds context
4. validation      Python mirror of the Runtime validator (canonical mode) +
                   payload-copy / user-source checks
5. dedup           exact + normalized requests, few-shot near-duplicates
6. split           decided before realization: families, source documents,
                   curated entities, SQL domains and unsupported items are each
                   partitioned into train/validation/test up front

Modes
-----
pilot : exactly 200 rows, core families, 70/15/15, data/manager/pilot_v2.jsonl
5k    : exactly 5000 rows, core + ext families, 80/10/10, sealed test split,
        data/manager/manager_v1_5k*.jsonl  (refused unless the pilot gate file
        says the pilot may proceed, unless --force-without-pilot-gate)
"""
from __future__ import annotations

import argparse
import hashlib
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

import manager_v2_content as content  # noqa: E402
from manager_dataset_templates import i_ga, wa  # noqa: E402
from manager_v2_registry import load_contract  # noqa: E402
from manager_v2_templates import (  # noqa: E402
    GROUNDED_SLOTS,
    INTENT_SLOTS,
    INTENTS,
    R,
    REGISTERS,
    TAXONOMY,
    TAXONOMY_BY_NAME,
    families_for,
    referenced_skills,
    sent,
    tidy,
)
from manager_v2_validator import canonical_json, validate  # noqa: E402
from stage2_dedup import hamming, simhash  # noqa: E402
from stage2_skill_synth_v2 import SENTENCE_SPLIT, clean_candidate  # noqa: E402

# r2: task constraints + stratified verbosity; 1.1.1 wording fix; 1.2.0 semantic seed = canonical plan + payload
# 1.3.0 realism fixes (sum only for additive metrics, related compare pairs); 1.3.1 unambiguous title categories
GENERATOR_VERSION = "manager-v2-gen-1.3.2"  # 1.3.2: pair builder scans for a categorized first document
SCHEMA_VERSION = 2
SOURCE_FILE = PROJECT_ROOT / "data" / "processed" / "aihub_documents_train.txt"
DATA_DIR = PROJECT_ROOT / "data" / "manager"
OUT_DIR = PROJECT_ROOT / "outputs" / "manager_dataset_v2"
SPLITS = ("train", "validation", "test")
MODES = {
    "pilot": {"count": 200, "seed": 20260925, "tiers": ("core",), "ratios": {"train": 0.70, "validation": 0.15,
                                                                            "test": 0.15},
              "output": DATA_DIR / "pilot_v2.jsonl", "docs": 900, "sibling_share": 0.08},
    "5k": {"count": 5000, "seed": 20260926, "tiers": ("core", "ext"), "ratios": {"train": 0.80, "validation": 0.10,
                                                                               "test": 0.10},
           "output": DATA_DIR / "manager_v1_5k.jsonl", "docs": 20000, "sibling_share": 0.08},
}
MAX_ATTEMPTS = 30
MAX_SEED_REUSE = 6          # clarify/unsupported have small content spaces
FEWSHOT_HAMMING = 3
# none / preface / preface+suffix. r1 used (0.5, 0.35, 0.15) for every plan and
# Pilot v2 r1 failed Q2 (corr 0.53). r2 stratifies by task count so that
# context wording does not track plan length.
VERBOSITY_BY_TASKS = {0: (0.4, 0.4, 0.2), 1: (0.25, 0.45, 0.30), 2: (0.5, 0.35, 0.15), 3: (0.7, 0.25, 0.05),
                      4: (0.8, 0.2, 0.0)}
DETAIL_PROBABILITY = 0.5
DETAIL_INTENTS = ("search", "summary_user", "extract_user", "sql")
PAYLOAD_CLASSES = (("short", 1, 1, 30, 140), ("medium", 2, 3, 80, 380), ("long", 3, 6, 250, 900))


class SlotUnavailable(Exception):
    """A semantic spec could not be built from this draw; counted and retried."""


def normalize_request(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFC", text).lower() if c.isalnum())


def stable_hash(*parts) -> str:
    return hashlib.sha1("\x1f".join(map(str, parts)).encode("utf-8")).hexdigest()[:16]


def largest_remainder(total: int, weights: dict) -> dict:
    norm = sum(weights.values())
    raw = {k: total * v / norm for k, v in weights.items()}
    counts = {k: math.floor(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: (-(raw[k] - counts[k]), list(weights).index(k)))[:total - sum(counts.values())]:
        counts[k] += 1
    return counts


def partition(items: list, ratios: dict, rng: random.Random, minimum_heldout: int = 1) -> dict[str, list]:
    items = list(items)
    rng.shuffle(items)
    n = len(items)
    held = {s: max(minimum_heldout, round(n * ratios[s])) if n >= 1 + 2 * minimum_heldout else 0
            for s in ("validation", "test")}
    result = {"validation": items[:held["validation"]],
              "test": items[held["validation"]:held["validation"] + held["test"]]}
    result["train"] = items[held["validation"] + held["test"]:]
    return result


# --------------------------------------------------------------------------- #
# Source documents
# --------------------------------------------------------------------------- #

def _split_title(line: str):
    words = line.split(" ")
    for size in range(1, 7):
        title, rest = " ".join(words[:size]), " ".join(words[size:])
        if rest.startswith(title) and len(rest) > len(title):
            return title, rest
    return None


def load_documents(limit: int, seed: int, excluded_titles: set[str]) -> tuple[list[dict], dict]:
    lines = SOURCE_FILE.read_text(encoding="utf-8").splitlines()
    order = list(range(len(lines)))
    random.Random(f"{seed}:docs").shuffle(order)
    rejects, docs, seen = Counter(), [], set()
    for line_no in order:
        if len(docs) >= limit:
            break
        parsed = _split_title(unicodedata.normalize("NFC", lines[line_no]))
        if not parsed:
            rejects["no_title"] += 1
            continue
        title, body = clean_candidate(parsed[0]), parsed[1]
        if not (2 <= len(title) <= 20) or not ("가" <= title[-1] <= "힣") or re.search(r"[()\[\]{}\"'“”:/·]", title):
            rejects["title_shape"] += 1
            continue
        if title in seen or title in excluded_titles:
            rejects["duplicate_title"] += 1
            continue
        sentences = [clean_candidate(p) for p in SENTENCE_SPLIT.split(body)]
        sentences = [p + "." for p in sentences if 12 <= len(p) <= 260 and not re.search(r"[\"“”\\�\x00-\x1f]", p)]
        hangul = sum("가" <= c <= "힣" for c in " ".join(sentences))
        if len(sentences) < 1 or hangul < 0.4 * max(1, len(" ".join(sentences))):
            rejects["body"] += 1
            continue
        seen.add(title)
        docs.append({"doc_id": f"aihub_documents_train:L{line_no + 1}", "title": title, "sentences": sentences})
    return docs, {"source": str(SOURCE_FILE.relative_to(PROJECT_ROOT)), "lines": len(lines), "accepted": len(docs),
                  "rejected": dict(rejects)}


def payload_text(doc: dict, rng: random.Random) -> str | None:
    name, lo, hi, cmin, cmax = PAYLOAD_CLASSES[rng.randrange(len(PAYLOAD_CLASSES))]
    count = min(len(doc["sentences"]), rng.randint(lo, hi))
    text = " ".join(doc["sentences"][:count])
    while len(text) > cmax and count > 1:
        count -= 1
        text = " ".join(doc["sentences"][:count])
    return text if cmin <= len(text) <= cmax else None


def detect_fields(text: str) -> list[str]:
    return [name for name, pattern in content.PAYLOAD_FIELDS if re.search(pattern, text)]


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #

class Generator:
    def __init__(self, mode: str, count: int | None = None, seed: int | None = None):
        cfg = MODES[mode]
        self.mode, self.cfg = mode, cfg
        self.count = count if count is not None else cfg["count"]
        self.seed = seed if seed is not None else cfg["seed"]
        self.tiers = cfg["tiers"]
        self.ratios = cfg["ratios"]
        self.contract = load_contract()
        missing = referenced_skills() - set(self.contract.skill_names)
        if missing:
            raise RuntimeError(f"templates use skills absent from the Runtime registry: {sorted(missing)}")
        unused = set(self.contract.skill_names) - referenced_skills()
        if unused:
            raise RuntimeError(f"Runtime skills never generated: {sorted(unused)}")
        self.fewshots = [f["user"] for f in self.contract.few_shots]
        self.fewshot_norm = {normalize_request(t) for t in self.fewshots}
        self.fewshot_hash = [simhash(t) for t in self.fewshots]
        fewshot_text = " ".join(self.fewshots)
        rng = random.Random(f"{self.seed}:partition")

        # curated entities, minus anything the Runtime few-shots mention
        self.excluded_entities = sorted(
            e for d in content.DOMAINS.values() for e in d["entities"]
            if e in fewshot_text or (" " in e and len(e.rsplit(" ", 1)[0]) >= 5
                                     and e.rsplit(" ", 1)[0] in fewshot_text))
        self.entities = {s: defaultdict(list) for s in SPLITS}
        for domain, spec in content.DOMAINS.items():
            pool = [e for e in spec["entities"] if e not in self.excluded_entities]
            parts = partition(pool, self.ratios, random.Random(f"{self.seed}:ent:{domain}"), minimum_heldout=2)
            for split in SPLITS:
                self.entities[split][domain] = parts[split]
        self.sql_domains = partition(sorted(content.SQL_DOMAINS), self.ratios,
                                     random.Random(f"{self.seed}:sql"), minimum_heldout=2)
        self.unsupported = {s: {} for s in SPLITS}
        for category, items in content.UNSUPPORTED_ITEMS.items():
            parts = partition(items, self.ratios, random.Random(f"{self.seed}:unsup:{category}"))
            if "ext" in self.tiers:
                extra = partition(content.UNSUPPORTED_ITEMS_EXT[category], self.ratios,
                                  random.Random(f"{self.seed}:unsup-ext:{category}"))
                parts = {split: parts[split] + extra[split] for split in SPLITS}
            for split in SPLITS:
                self.unsupported[split][category] = parts[split]
        docs, self.doc_stats = load_documents(cfg["docs"], self.seed, set())
        self.docs = partition(docs, self.ratios, rng)
        self.doc_cursor = {s: 0 for s in SPLITS}
        self.family_split = self._assign_families()
        self.quotas = largest_remainder(self.count, {t.name: t.pilot_quota for t in TAXONOMY})

        self.rows: list[dict] = []
        self.exact, self.normalized = set(), set()
        self.seed_uses: Counter = Counter()
        self.attempts = 0
        self.rejects: Counter = Counter()
        self.shortfall: list[dict] = []

    # ---- split assignment ----------------------------------------------------
    def _assign_families(self) -> dict[str, str]:
        assignment = {}
        for taxonomy in TAXONOMY:
            strata = defaultdict(list)
            for family in families_for(taxonomy.name, self.tiers):
                strata[family.stratum].append(family.id)
            for index, (stratum, ids) in enumerate(sorted(strata.items(), key=lambda kv: str(kv[0]))):
                ids = sorted(ids)
                random.Random(f"{self.seed}:fam:{taxonomy.name}:{stratum}").shuffle(ids)
                n = len(ids)
                if n >= 3:
                    n_val = max(1, round(n * self.ratios["validation"]))
                    n_test = max(1, round(n * self.ratios["test"]))
                    plan = ["validation"] * n_val + ["test"] * n_test + ["train"] * (n - n_val - n_test)
                elif n == 2:
                    plan = ["train", ("test", "validation")[index % 2]]
                else:
                    plan = ["train"]
                assignment.update(zip(ids, plan))
        return assignment

    # ---- slot builders -----------------------------------------------------------
    def _doc(self, split: str) -> dict:
        pool = self.docs[split]
        if self.doc_cursor[split] >= len(pool):
            raise RuntimeError(f"document pool exhausted for {split}")
        doc = pool[self.doc_cursor[split]]
        self.doc_cursor[split] += 1
        return doc

    def _curated(self, split, rng, need=1, domain=None):
        domains = [d for d, e in self.entities[split].items() if len(e) >= need]
        domain = domain or domains[rng.randrange(len(domains))]
        return domain, rng.sample(self.entities[split][domain], need)

    def _aspect(self, domain, rng):
        options = content.DOMAINS[domain]["aspects"] if domain else content.GENERAL_ASPECTS
        return options[rng.randrange(len(options))]

    def _search_target(self, split, rng, curated_only=False):
        if curated_only or rng.random() < 0.6:
            domain, (entity,) = self._curated(split, rng)
            aspect, fields = self._aspect(domain, rng)
            return {"T": entity, "A": aspect, "F": fields}, [f"ent:{entity}"]
        doc = self._doc(split)
        aspect, fields = self._aspect(None, rng)
        return {"T": doc["title"], "A": aspect, "F": fields}, [f"doc:{doc['doc_id']}"]

    def _payload(self, split, rng, need_fields=False):
        for _ in range(20):
            doc = self._doc(split)
            text = payload_text(doc, rng)
            if not text:
                continue
            fields = detect_fields(text)
            if need_fields and not fields:
                continue
            chosen = rng.sample(fields, min(len(fields), rng.choice((1, 1, 2)))) if fields else []
            joined = f"{chosen[0]}{wa(chosen[0])} {chosen[1]}" if len(chosen) == 2 else "".join(chosen)
            return doc, text, joined
        raise RuntimeError("could not draw a usable payload")

    def _payload_pair(self, split, rng):
        """Two payloads whose titles share a category suffix (e.g. two stations)."""
        def category(title):
            return next((c for c in sorted(content.TITLE_CATEGORIES, key=len, reverse=True) if title.endswith(c)), None)

        def take(predicate):
            pool, start = self.docs[split], self.doc_cursor[split]
            for offset in range(len(pool) - start):
                doc = pool[start + offset]
                if predicate(doc):
                    text = payload_text(doc, rng)
                    if text:
                        # move the chosen document to the cursor so it is consumed exactly once
                        pool[start], pool[start + offset] = pool[start + offset], pool[start]
                        self.doc_cursor[split] += 1
                        return doc, text, ""
            return None

        first = take(lambda doc: category(doc["title"]) is not None)
        if first is None:
            raise SlotUnavailable("NO_RELATED_PAIR")
        kind = category(first[0]["title"])
        second = take(lambda doc: category(doc["title"]) == kind and doc["title"] != first[0]["title"])
        if second is None:
            raise SlotUnavailable("NO_RELATED_PAIR")
        return first, second

    def _sql(self, split, rng, domain=None):
        domain = domain or self.sql_domains[split][rng.randrange(len(self.sql_domains[split]))]
        spec = content.SQL_DOMAINS[domain]
        metric = spec["metrics"][rng.randrange(len(spec["metrics"]))]
        dim = spec["dims"][rng.randrange(len(spec["dims"]))]
        period = content.PERIODS[rng.randrange(len(content.PERIODS))]
        k = content.TOPK[rng.randrange(len(content.TOPK))]
        additive = not any(marker in metric for marker in content.NON_ADDITIVE_MARKERS)
        form = rng.randrange(4) if additive else rng.choice((0, 1, 3))
        query = [f"{period} {dim}별 {metric}", f"{period} {metric} 상위 {k}개 {dim}", f"{period} {metric} 합계",
                 f"{period} {metric}{i_ga(metric)} 가장 높은 {dim}"][form]
        system = spec["system"][rng.randrange(len(spec["system"]))]
        return {"Q": query, "system": system, "owner": spec["owner"], "metric": metric, "dim": dim}, [f"sqldom:{domain}"]

    def build_slots(self, family, split: str, rng: random.Random, taxonomy: str) -> tuple[dict, list[str]]:
        kind = INTENT_SLOTS[family.intent]
        if kind in ("search", "search_fields"):
            slots, ents = self._search_target(split, rng, curated_only=taxonomy == "hard_implicit_retrieval")
        elif kind == "payload":
            doc, text, _ = self._payload(split, rng)
            slots, ents = {"X": text}, [f"doc:{doc['doc_id']}"]
        elif kind == "payload_fields":
            doc, text, fields = self._payload(split, rng, need_fields=True)
            slots, ents = {"X": text, "F": fields}, [f"doc:{doc['doc_id']}"]
        elif kind == "payload_pair":
            (d1, t1, _), (d2, t2, _) = self._payload_pair(split, rng)
            style = rng.randrange(3)
            x = [f"1) {t1}\n2) {t2}", f"첫 번째 설명: {t1}\n두 번째 설명: {t2}", f"[{d1['title']}] {t1}\n[{d2['title']}] {t2}"][style]
            slots = {"X": x, "X1": t1, "X2": t2, "TT1": d1["title"], "TT2": d2["title"],
                     "C": content.PAYLOAD_COMPARE_CRITERIA[rng.randrange(len(content.PAYLOAD_COMPARE_CRITERIA))]}
            ents = [f"doc:{d1['doc_id']}", f"doc:{d2['doc_id']}"]
        elif kind == "payload_topic":
            doc, text, _ = self._payload(split, rng)
            slots, ents = {"X": text, "T": doc["title"]}, [f"doc:{doc['doc_id']}"]
        elif kind in ("sql", "sql_fields"):
            slots, ents = self._sql(split, rng)
            slots["F"] = content.SQL_RESULT_FIELDS[rng.randrange(len(content.SQL_RESULT_FIELDS))]
        elif kind == "sql_pair":
            domain = self.sql_domains[split][rng.randrange(len(self.sql_domains[split]))]
            spec = content.SQL_DOMAINS[domain]
            metric = spec["metrics"][rng.randrange(len(spec["metrics"]))]
            dim = spec["dims"][rng.randrange(len(spec["dims"]))]
            p1, p2 = content.PERIOD_PAIRS[rng.randrange(len(content.PERIOD_PAIRS))]
            slots = {"Q1": f"{p1} {dim}별 {metric}", "Q2": f"{p2} {dim}별 {metric}",
                     "C": f"{p1}{wa(p1)} {p2}의 {metric}", "system": spec["system"][0], "owner": spec["owner"]}
            ents = [f"sqldom:{domain}"]
        elif kind in ("search_pair", "search_pair2"):
            domain, (e1, e2) = self._curated(split, rng, need=2)
            aspect, fields = self._aspect(domain, rng)
            criteria = content.DOMAINS[domain]["criteria"]
            criterion = criteria[rng.randrange(len(criteria))]
            # the searched aspect *is* the comparison criterion, so one phrase in
            # the request grounds both the searches and the compare instruction
            slots = {"T1": e1, "T2": e2, "A": criterion, "F": fields, "C": criterion}
            if kind == "search_pair2":
                slots["A"] = aspect
                slots["A2"] = self._aspect(domain, rng)[0]
            ents = [f"ent:{e1}", f"ent:{e2}"]
        elif kind == "internal":
            domain, (entity,) = self._curated(split, rng)
            internal = content.DOMAINS[domain]["internal"]
            slots = {"E": entity, "QI": internal[0].format(e=entity), "AE": internal[1], "CI": internal[2],
                     "order": "sql_first" if rng.random() < 0.5 else "search_first"}
            ents = [f"ent:{entity}"]
        elif kind == "clarify":
            reason = family.stratum
            domain, (entity,) = self._curated(split, rng)
            aspect = self._aspect(domain, rng)[0]
            criteria = content.DOMAINS[domain]["criteria"]
            field_names = [name for name, _ in content.PAYLOAD_FIELDS]
            slots = {"reason": reason, "T": entity, "A": aspect, "C": criteria[rng.randrange(len(criteria))],
                     "F": field_names[rng.randrange(len(field_names))],
                     "P": content.PRONOUNS[rng.randrange(len(content.PRONOUNS))],
                     "V": content.VAGUE_DB[rng.randrange(len(content.VAGUE_DB))]}
            ents = [f"ent:{entity}"] if reason in ("missing_compare_target",) else []
        elif kind == "unsupported":
            categories = [c for c, items in self.unsupported[split].items() if items]
            category = categories[rng.randrange(len(categories))]
            items = self.unsupported[split][category]
            item = items[rng.randrange(len(items))]
            slots = {"category": category, "item": item}
            ents = [f"unsup:{category}:{item}"]
            if category == "translate":
                doc = self._doc(split)
                slots["X"] = doc["sentences"][0]
                ents.append(f"doc:{doc['doc_id']}")
        else:
            raise ValueError(kind)
        if family.intent in DETAIL_INTENTS and family.verbosity != "none" and rng.random() < DETAIL_PROBABILITY:
            details = content.TASK_DETAILS[family.intent]
            slots["D"] = details[rng.randrange(len(details))]
        if taxonomy == "hard_long_single":
            pool = content.LONG_CONTEXT_CORE + (content.LONG_CONTEXT_EXT if "ext" in self.tiers else [])
            slots["LC"] = pool[rng.randrange(len(pool))]
        return slots, ents

    # ---- realization -----------------------------------------------------------------
    def _verbosity(self, rng: random.Random, family, task_count: int) -> tuple[int, str, str]:
        if family.verbosity != "random":
            return 0, "", ""
        level = rng.choices((0, 1, 2), weights=VERBOSITY_BY_TASKS[min(task_count, 4)])[0]
        prefaces = content.PREFACES_CORE + (content.PREFACES_EXT if "ext" in self.tiers else [])
        suffixes = content.SUFFIXES_CORE + (content.SUFFIXES_EXT if "ext" in self.tiers else [])
        pre = prefaces[rng.randrange(len(prefaces))] if level >= 1 else ""
        suf = suffixes[rng.randrange(len(suffixes))] if level >= 2 else ""
        return level, pre, suf

    def realize(self, spec: dict, family, split: str, source_type: str, paraphrase_of: str | None) -> bool:
        taxonomy = TAXONOMY_BY_NAME[spec["taxonomy"]]
        plan = INTENTS[family.intent](spec["slots"])
        plan_text = canonical_json(plan)
        payloads = [spec["slots"][k] for k in ("X", "X1", "X2") if spec["slots"].get(k)]
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self.attempts += 1
            rng = random.Random(f"{self.seed}:real:{spec['semantic_seed']}:{family.id}:{len(self.rows)}:{attempt}")
            realizer = R(rng, REGISTERS[rng.randrange(len(REGISTERS))])
            core = family.render(realizer, spec["slots"])
            detail = spec["slots"].get("D")
            if detail:
                clause = realizer.pick((f"단, {detail}.", f"({detail})", f"조건: {detail}.",
                                        f"{detail}, 이 조건으로 부탁해."), "det")
                if spec["slots"].get("X") and core.endswith(spec["slots"]["X"]):
                    # keep the constraint next to the instruction, before the pasted text
                    head = core[: -len(spec["slots"]["X"])].rstrip(": \n")
                    core = f"{sent(head)} {clause}\n\n{spec['slots']['X']}"
                else:
                    core = sent(core) + " " + clause
            level, pre, suf = self._verbosity(rng, family, len(plan["tasks"]))
            request = tidy(unicodedata.normalize("NFC", " ".join(p for p in (pre, core, suf) if p)))
            norm = normalize_request(request)
            reason = None
            if "�" in request:
                reason = "UTF8"
            elif request in self.exact:
                reason = "DUPLICATE_EXACT"
            elif norm in self.normalized:
                reason = "DUPLICATE_NORMALIZED"
            elif not all(re.sub(r"\s", "", spec["slots"][k]) in re.sub(r"\s", "", request)
                         for k in GROUNDED_SLOTS[family.intent] if spec["slots"].get(k)):
                reason = "UNGROUNDED_SLOT"
            elif norm in self.fewshot_norm or any(hamming(simhash(request), h) <= FEWSHOT_HAMMING
                                                   for h in self.fewshot_hash):
                reason = "FEWSHOT_NEAR_DUPLICATE"
            else:
                names = [spec["slots"][k] for k in ("T", "TT1", "TT2") if spec["slots"].get(k)]
                result = validate(plan_text, self.contract, mode="canonical",
                                  payloads=payloads if "user" in plan_text else None, names=names)
                if not result.valid:
                    reason = "INVALID_PLAN:" + result.code
            if reason:
                self.rejects[reason] += 1
                continue
            self.exact.add(request)
            self.normalized.add(norm)
            number = len(self.rows) + 1
            prefix = "mgr-pilot-v2" if self.mode == "pilot" else "mgr-v1-5k"
            tasks = plan["tasks"]
            depth = {}
            for task in tasks:
                refs = [s for s in task["sources"] if s != "user"]
                depth[task["id"]] = 1 + max((depth[s] for s in refs), default=0)
            user_refs = sum(s == "user" for t in tasks for s in t["sources"])
            task_refs = sum(s != "user" for t in tasks for s in t["sources"])
            self.rows.append({
                "id": f"{prefix}-{number:05d}",
                "request": request,
                "plan": plan,
                "plan_text": plan_text,
                "metadata": {
                    "sample_id": f"{prefix}-{number:05d}",
                    "status_label": "candidate",
                    "generator_version": GENERATOR_VERSION,
                    "schema_version": SCHEMA_VERSION,
                    "source_type": source_type,
                    "taxonomy": taxonomy.name,
                    "taxonomy_group": taxonomy.group,
                    "template_family": family.id,
                    "family_tier": family.tier,
                    "family_stratum": family.stratum,
                    "intent": family.intent,
                    "semantic_seed": spec["semantic_seed"],
                    "skills_used": [t["skill"] for t in tasks],
                    "task_count": len(tasks),
                    "dependency_depth": max(depth.values(), default=0),
                    "source_count": user_refs + task_refs,
                    "source_usage": {"user": user_refs, "task": task_refs,
                                     "multi_source_tasks": sum(len(t["sources"]) > 1 for t in tasks)},
                    "status": plan["status"],
                    "output_count": len(plan["outputs"]),
                    "difficulty": taxonomy.difficulty,
                    "hard_case": taxonomy.group == "hard",
                    "hard_case_category": taxonomy.hard_case,
                    "split_group": family.id,
                    "split": split,
                    "sealed_test": self.mode == "5k" and split == "test",
                    "entity_family": spec["entities"],
                    "paraphrased": source_type == "paraphrased",
                    "paraphrase_of": paraphrase_of,
                    "has_user_payload": bool(payloads),
                    "payload_chars": sum(map(len, payloads)),
                    "arguments": {k: v for k, v in spec["slots"].items() if k in ("X", "X1", "X2", "T", "T1", "T2",
                                                                                   "E", "Q", "Q1", "Q2", "QI", "LC",
                                                                                   "D", "TT1", "TT2")},
                    "task_detail": spec["slots"].get("D"),
                    "register": realizer.register,
                    "verbosity": level,
                    "realization_trace": realizer.trace,
                    "seed": self.seed,
                    "attempt": attempt,
                    "fewshot_family_overlap": False,
                },
            })
            return True
        self.shortfall.append({"taxonomy": taxonomy.name, "family": family.id, "semantic_seed": spec["semantic_seed"]})
        return False

    def make_spec(self, family, split: str, taxonomy: str, index: int) -> dict | None:
        for attempt in range(MAX_ATTEMPTS):
            rng = random.Random(f"{self.seed}:spec:{taxonomy}:{split}:{index}:{attempt}")
            try:
                slots, ents = self.build_slots(family, split, rng, taxonomy)
            except SlotUnavailable as reason:
                self.attempts += 1
                self.rejects[str(reason)] += 1
                continue
            # Meaning = canonical plan + the user content it points at. Slots the plan
            # does not use must not create "different" seeds for the same meaning.
            key = {k: slots[k] for k in ("X", "X1", "X2") if slots.get(k)}
            if family.intent in ("clarify", "unsupported"):
                key.update({k: v for k, v in slots.items() if k not in ("LC", "D")})
            seed = stable_hash(canonical_json(INTENTS[family.intent](slots)),
                               json.dumps(key, ensure_ascii=False, sort_keys=True))
            limit = MAX_SEED_REUSE if family.intent in ("clarify", "unsupported") else 1
            if self.seed_uses[seed] >= limit:
                self.attempts += 1
                self.rejects["SEMANTIC_DUPLICATE"] += 1
                continue
            self.seed_uses[seed] += 1
            return {"taxonomy": taxonomy, "slots": slots, "entities": ents, "semantic_seed": seed}
        return None

    def generate(self) -> list[dict]:
        index = 0
        for taxonomy in TAXONOMY:
            quota = self.quotas[taxonomy.name]
            per_split = largest_remainder(quota, self.ratios)
            by_split = {split: [f for f in families_for(taxonomy.name, self.tiers)
                                if self.family_split[f.id] == split] for split in SPLITS}
            for split in ("validation", "test"):
                if not by_split[split]:
                    per_split["train"] += per_split[split]
                    per_split[split] = 0
            for split in SPLITS:
                families = by_split[split]
                cell = per_split[split]
                if not cell:
                    continue
                random.Random(f"{self.seed}:order:{taxonomy.name}:{split}").shuffle(families)
                siblings = math.floor(cell * self.cfg["sibling_share"]) if taxonomy.group == "normal" else 0
                bases = []
                for k in range(cell - siblings):
                    family = families[k % len(families)]
                    spec = self.make_spec(family, split, taxonomy.name, index)
                    index += 1
                    if spec is None:
                        self.shortfall.append({"taxonomy": taxonomy.name, "family": family.id, "reason": "no_spec"})
                        continue
                    if self.realize(spec, family, split, "hard_case" if taxonomy.group == "hard" else "rule_generated",
                                    None):
                        bases.append((spec, family, self.rows[-1]["id"]))
                rng = random.Random(f"{self.seed}:siblings:{taxonomy.name}:{split}")
                for _ in range(siblings):
                    if not bases:
                        break
                    spec, base_family, base_id = bases[rng.randrange(len(bases))]
                    others = [f for f in families if f.id != base_family.id and f.intent == base_family.intent]
                    family = others[rng.randrange(len(others))] if others else base_family
                    self.realize(spec, family, split, "paraphrased", base_id)
        return self.rows

    def log(self) -> dict:
        return {
            "generator_version": GENERATOR_VERSION, "schema_version": SCHEMA_VERSION, "mode": self.mode,
            "seed": self.seed, "requested_count": self.count, "accepted_count": len(self.rows),
            "attempts": self.attempts, "rejected_attempts": sum(self.rejects.values()),
            "reject_reasons": dict(self.rejects), "shortfall": self.shortfall, "quotas": self.quotas,
            "split_ratios": self.ratios, "tiers": list(self.tiers), "family_split": self.family_split,
            "excluded_fewshot_entities": self.excluded_entities,
            "sql_domain_split": self.sql_domains,
            "entity_split": {s: {d: v for d, v in self.entities[s].items()} for s in SPLITS},
            "unsupported_item_split": self.unsupported,
            "documents": self.doc_stats, "documents_used": self.doc_cursor,
            "contract": self.contract.to_dict(),
        }


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def generate(mode: str, count: int | None = None, seed: int | None = None) -> Generator:
    generator = Generator(mode, count, seed)
    generator.generate()
    return generator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=sorted(MODES), required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--force-without-pilot-gate", action="store_true")
    args = parser.parse_args()
    cfg = MODES[args.mode]
    output: Path = cfg["output"]
    if output.exists() and not args.overwrite:
        parser.error(f"{output} exists; generation is deterministic, pass --overwrite to rewrite it")
    if args.mode == "5k" and not args.force_without_pilot_gate:
        gate_file = OUT_DIR / "pilot_v2_stats.json"
        gate = json.loads(gate_file.read_text()).get("proceed_to_5k") if gate_file.exists() else None
        if gate is not True:
            parser.error("pilot v2 gate does not allow 5k generation (pilot_v2_stats.json proceed_to_5k != true)")
    generator = generate(args.mode)
    log = generator.log()
    lines = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in generator.rows)
    log_path = output.with_suffix(".generation_log.json")
    _write(log_path, json.dumps(log, ensure_ascii=False, indent=2))
    if len(generator.rows) != generator.count:
        _write(output.with_suffix(".partial.jsonl"), lines)
        print(f"FAILED: {len(generator.rows)}/{generator.count}; see {log_path}", file=sys.stderr)
        return 2
    _write(output, lines)
    if args.mode == "5k":
        for split, name in (("train", "train"), ("validation", "val"), ("test", "test")):
            part = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in generator.rows
                           if r["metadata"]["split"] == split)
            _write(output.with_name(f"manager_v1_5k_{name}.jsonl"), part)
    print(f"wrote {len(generator.rows)} rows to {output} (attempts={log['attempts']}, "
          f"rejected={log['rejected_attempts']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
