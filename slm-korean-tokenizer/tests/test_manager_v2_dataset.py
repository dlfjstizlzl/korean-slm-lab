"""TaskPlan v2 / Manager v1 dataset tests (Python side).

C++ Runtime behaviour (executor, source resolution, outputs, search/sql
contracts, grammar, orchestration smoke) is tested in
slm-runtime/tests/taskplan_v2_tests.cpp; here the Python mirror is checked
against that same compiled code.
"""
from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))

import manager_v2_generator as gen  # noqa: E402
import manager_v2_validator as v  # noqa: E402
from manager_v2_analysis import REQUIRED_METADATA, keyword_rule, leakage, signature  # noqa: E402
from manager_v2_registry import TOOL, load_contract  # noqa: E402
from manager_v2_templates import FAMILIES, Family, referenced_skills  # noqa: E402

DATA = ROOT / "data" / "manager"
TOOL_AVAILABLE = TOOL.is_file() or (TOOL.parent / "CMakeCache.txt").is_file()
SOURCE_AVAILABLE = gen.SOURCE_FILE.is_file()


def task(i, skill, sources="[]", instruction="지시"):
    return '{"id":%d,"skill":"%s","instruction":"%s","sources":%s}' % (i, skill, instruction, sources)


def ok(tasks, outputs="[0]"):
    return '{"status":"ok","tasks":[' + ",".join(tasks) + '],"outputs":' + outputs + "}"


@unittest.skipUnless(TOOL_AVAILABLE, "slm-runtime build not available")
class TaskPlanV2ValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = load_contract()

    def code(self, text, mode="runtime"):
        result = v.validate(text, self.contract, mode=mode)
        return "OK" if result.valid else result.code

    def test_registry_is_the_runtime_catalog(self):
        self.assertEqual(self.contract.skill_names, ("search", "summary", "extract", "compare", "sql"))
        backends = {s.name: (s.backend, s.min_sources, s.max_sources) for s in self.contract.skills}
        self.assertEqual(backends["search"], ("retrieval", 0, 0))
        self.assertEqual(backends["sql"], ("database", 0, 0))
        for name in ("summary", "extract", "compare"):
            self.assertEqual(backends[name][:2], ("model", 1))
        self.assertEqual(self.contract.max_tasks, 8)
        self.assertEqual(referenced_skills(), set(self.contract.skill_names))

    def test_search_contract_is_retrieval_not_keyword_extraction(self):
        search = self.contract.skill("search")
        self.assertIn("retrieval", search.description.lower())
        self.assertNotIn("keyword", self.contract.system_prompt.lower())
        self.assertNotIn("input_from", self.contract.system_prompt)

    def test_roundtrip_and_compact_serialization(self):
        text = ok([task(0, "search"), task(1, "search"), task(2, "compare", "[0,1]"),
                   task(3, "summary", "[2]")], "[3,2]")
        self.assertEqual(v.canonical_json(v.parse(text)), text)
        spaced = text.replace(",", ", ").replace(":", ": ")
        self.assertEqual(v.canonical_json(v.parse(spaced)), text)
        self.assertEqual(self.code(spaced, "runtime"), "OK")
        self.assertEqual(self.code(spaced, "canonical"), "NON_CANONICAL")

    def test_status_handling(self):
        self.assertEqual(self.code('{"status":"clarify","message":"요약할 내용을 제공해 주세요.","tasks":[],"outputs":[]}'), "OK")
        self.assertEqual(self.code('{"status":"unsupported","message":"불가","tasks":[],"outputs":[]}'), "OK")
        self.assertEqual(self.code('{"status":"clarify","tasks":[],"outputs":[]}'), "STATUS_SEMANTICS")
        self.assertEqual(self.code('{"status":"clarify","message":"x","tasks":[' + task(0, "search") + '],"outputs":[]}'),
                         "STATUS_SEMANTICS")
        self.assertEqual(self.code('{"status":"ok","message":"x","tasks":[' + task(0, "search") + '],"outputs":[0]}'),
                         "STATUS_SEMANTICS")
        self.assertEqual(self.code('{"status":"maybe","tasks":[],"outputs":[]}'), "INVALID_STATUS")
        self.assertEqual(self.code('{"status":"ok","tasks":[],"outputs":[]}'), "EMPTY_PLAN")

    def test_sources_multi_source_and_outputs(self):
        self.assertEqual(self.code(ok([task(0, "sql"), task(1, "search"), task(2, "compare", '["user",0,1]')], "[2]")), "OK")
        self.assertEqual(self.code(ok([task(0, "summary", '["user"]'), task(1, "extract", '["user"]')], "[0,1]")), "OK")
        self.assertEqual(self.code(ok([task(0, "search"), task(1, "summary", "[1]")], "[1]")), "SELF_REFERENCE")
        self.assertEqual(self.code(ok([task(0, "summary", "[1]"), task(1, "search")])), "FUTURE_REFERENCE")
        self.assertEqual(self.code(ok([task(0, "search"), task(1, "summary", "[0,0]")], "[1]")), "DUPLICATE_SOURCE")
        self.assertEqual(self.code(ok([task(0, "summary", '["users"]')])), "INVALID_SOURCE")
        self.assertEqual(self.code(ok([task(0, "summary", '["user"]')], "[1]")), "INVALID_OUTPUT")
        self.assertEqual(self.code(ok([task(0, "summary", '["user"]')], "[0,0]")), "DUPLICATE_OUTPUT")
        self.assertEqual(self.code(ok([task(0, "summary", '["user"]')], "[]")), "EMPTY_OUTPUTS")

    def test_skill_specific_source_constraints(self):
        self.assertEqual(self.code(ok([task(0, "search", '["user"]')])), "SOURCE_COUNT")
        self.assertEqual(self.code(ok([task(0, "sql", '["user"]')])), "SOURCE_COUNT")
        for skill in ("summary", "extract", "compare"):
            self.assertEqual(self.code(ok([task(0, skill)])), "SOURCE_COUNT")
        self.assertEqual(self.code(ok([task(0, "compare", '["user"]')])), "OK")

    def test_rejections(self):
        self.assertEqual(self.code('{"status":"ok","status":"ok","tasks":[],"outputs":[]}'), "JSON_PARSE_ERROR")
        self.assertEqual(self.code(ok(['{"id":0.0,"skill":"search","instruction":"x","sources":[]}'])), "JSON_PARSE_ERROR")
        self.assertEqual(self.code(ok([task(i, "search") for i in range(9)], "[8]")), "TOO_MANY_TASKS")
        self.assertEqual(self.code(ok([task(0, "translate")])), "INVALID_SKILL")
        self.assertEqual(self.code(ok([task(0, "search", "[]", " 　")])), "EMPTY_INSTRUCTION")
        self.assertEqual(self.code(ok([task(0, "search", "[]", "깨짐�")])), "INVALID_TEXT")
        self.assertEqual(self.code(ok([task(0, "search", "[]", "x")]).encode().replace(b'"x"', b'"\xff"')), "INVALID_TEXT")
        self.assertEqual(self.code('{"tasks":[{"id":0,"skill":"search","input":"x"}]}'), "MISSING_FIELD")
        self.assertEqual(self.code(ok(['{"id":0,"skill":"search","instruction":"x","sources":[],"input":"y"}'])),
                         "UNKNOWN_FIELD")

    def test_dataset_only_checks(self):
        long_text = "가나다라마바사아자차카타파하 " * 4
        plan = ok([task(0, "summary", '["user"]', long_text[:25])])
        self.assertEqual(v.validate(plan, self.contract, payloads=[long_text]).code, "PAYLOAD_COPY")
        plan = ok([task(0, "summary", '["user"]', "핵심 내용 요약")])
        self.assertEqual(v.validate(plan, self.contract, payloads=[]).code, "USER_SOURCE_WITHOUT_PAYLOAD")
        warnings = v.validate(ok([task(0, "search"), task(1, "search", "[]", "y")], "[1]"), self.contract).warnings
        self.assertIn("UNUSED_TASK", warnings)

    def test_python_matches_cpp_first_code_on_parity_corpus(self):
        texts = [t for t, _ in v.PARITY_CASES]
        for (text, expected), (accepted, code, detail) in zip(v.PARITY_CASES, v.cpp_validate(texts)):
            python = self.code(text)
            self.assertEqual(python, "OK" if accepted else code, (text, detail))
            self.assertEqual(python, expected, text)

    def test_grammar_accepts_fewshots_and_rejects_invalid(self):
        plans = [f["plan"] for f in self.contract.few_shots]
        self.assertTrue(all(v.cpp_grammar(plans)))
        for plan in plans:
            self.assertEqual(self.code(plan, "canonical"), "OK")
        invalid = [ok([task(0, "summary")]), ok([task(0, "search", '["user"]')]), ok([task(0, "search")], "[1]")]
        self.assertFalse(any(v.cpp_grammar(invalid)))


@unittest.skipUnless(TOOL_AVAILABLE and SOURCE_AVAILABLE, "runtime build or AI Hub source unavailable")
class GeneratorV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.small = gen.generate("pilot", count=60, seed=7)

    def test_exact_requested_count(self):
        self.assertEqual(len(self.small.rows), 60)
        self.assertEqual(len(gen.generate("pilot", count=37, seed=3).rows), 37)

    def test_deterministic_generation(self):
        again = gen.generate("pilot", count=60, seed=7)
        self.assertEqual(json.dumps(self.small.rows, ensure_ascii=False), json.dumps(again.rows, ensure_ascii=False))
        other = gen.generate("pilot", count=60, seed=8)
        same = sum(a["request"] == b["request"] for a, b in zip(self.small.rows, other.rows))
        self.assertLess(same, 5)

    def test_provenance_and_plan_validity(self):
        contract = load_contract()
        for row in self.small.rows:
            for key in REQUIRED_METADATA:
                self.assertIn(key, row["metadata"], key)
            self.assertEqual(row["metadata"]["status_label"], "candidate")
            self.assertEqual(row["metadata"]["schema_version"], 2)
            self.assertTrue(v.validate(row["plan_text"], contract).valid, row["id"])
            self.assertEqual(row["plan_text"], v.canonical_json(row["plan"]))

    def test_plan_comes_from_semantics_not_text(self):
        # canonical plans are a pure function of the intent slots
        for row in self.small.rows:
            family = next(f for f in FAMILIES if f.id == row["metadata"]["template_family"])
            self.assertEqual(row["metadata"]["intent"], family.intent)

    def test_group_aware_split_has_no_leakage(self):
        report = leakage(self.small.rows)
        for key in ("template_family_across_splits", "semantic_seed_across_splits", "entity_family_across_splits",
                    "paraphrase_sibling_across_splits", "near_duplicate_pairs_cross_split"):
            self.assertEqual(report[key], [], key)
        self.assertEqual(report["group_components_spanning_splits"], 0)

    def test_leakage_detector_flags_injected_leak(self):
        rows = copy.deepcopy(self.small.rows)
        first, second = rows[0], rows[1]
        second["metadata"]["template_family"] = first["metadata"]["template_family"]
        second["metadata"]["split"] = "test" if first["metadata"]["split"] != "test" else "train"
        report = leakage(rows)
        self.assertIn(first["metadata"]["template_family"], report["template_family_across_splits"])
        self.assertGreater(report["group_components_spanning_splits"], 0)

    def test_ungrounded_family_is_rejected(self):
        generator = gen.Generator("pilot", count=10, seed=1)
        broken = Family("broken.search", "single_search", "search", lambda r, s: "아무거나 찾아 줘")
        spec = generator.make_spec(broken, "train", "single_search", 0)
        self.assertFalse(generator.realize(spec, broken, "train", "rule_generated", None))
        self.assertEqual(generator.rejects["UNGROUNDED_SLOT"], gen.MAX_ATTEMPTS)

    def test_anti_shortcut_generation(self):
        hard = [r for r in self.small.rows if r["metadata"]["hard_case"]]
        self.assertTrue(hard)
        misses = [r for r in hard if keyword_rule(r) != signature(r["plan"])]
        self.assertGreater(len(misses) / len(hard), 0.5)
        search_word_rows = [r for r in self.small.rows if r["metadata"]["taxonomy"] == "hard_search_word_no_search"]
        for row in search_word_rows:
            self.assertIn("검색", row["request"])
            self.assertNotIn("search", row["metadata"]["skills_used"])

    def test_no_fewshot_replication(self):
        fewshots = {gen.normalize_request(f["user"]) for f in load_contract().few_shots}
        self.assertFalse(any(gen.normalize_request(r["request"]) in fewshots for r in self.small.rows))


class ArtifactTests(unittest.TestCase):

    @unittest.skipUnless((DATA / "pilot_v1.jsonl").is_file(), "pilot v1 missing")
    def test_pilot_v1_is_preserved(self):
        rows = (DATA / "pilot_v1.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(rows), 200)
        self.assertTrue(json.loads(rows[0])["id"].startswith("mgr-pilot-v1-"))

    @unittest.skipUnless((DATA / "pilot_v2.jsonl").is_file() and TOOL_AVAILABLE and SOURCE_AVAILABLE, "pilot v2 missing")
    def test_pilot_v2_is_exactly_200_and_reproducible(self):
        text = (DATA / "pilot_v2.jsonl").read_text(encoding="utf-8")
        self.assertEqual(len(text.splitlines()), 200)
        regenerated = gen.generate("pilot").rows
        self.assertEqual(text, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in regenerated))

    @unittest.skipUnless((DATA / "manager_v1_5k.jsonl").is_file(), "5k missing")
    def test_5k_counts_splits_sealed_test_and_budget(self):
        rows = [json.loads(line) for line in (DATA / "manager_v1_5k.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 5000)
        parts = {name: [json.loads(line) for line in (DATA / f"manager_v1_5k_{name}.jsonl").read_text(
            encoding="utf-8").splitlines()] for name in ("train", "val", "test")}
        self.assertEqual(sum(map(len, parts.values())), 5000)
        self.assertEqual({r["id"] for r in parts["test"]}, {r["id"] for r in rows if r["metadata"]["split"] == "test"})
        self.assertTrue(all(r["metadata"]["sealed_test"] for r in parts["test"]))
        self.assertFalse(any(r["metadata"]["sealed_test"] for r in parts["train"] + parts["val"]))
        stats = json.loads((ROOT / "outputs/manager_dataset_v2/5k_stats.json").read_text())
        self.assertEqual(stats["plans"]["manager_output_over_256"], 0)
        self.assertEqual(stats["plans"]["differential"]["python_cpp_mismatches"], 0)
        self.assertTrue(stats["gates"]["hard_pass"])


if __name__ == "__main__":
    unittest.main()
