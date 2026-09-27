from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))

import manager_dataset_generator as gen  # noqa: E402
import manager_plan_validator as v  # noqa: E402
from manager_dataset_analysis import REQUIRED_METADATA, leakage  # noqa: E402
from manager_dataset_templates import FAMILY_BY_ID, referenced_skills  # noqa: E402
from manager_runtime_registry import (  # noqa: E402
    DEFAULT_RUNTIME_ROOT,
    SOURCE_FILES,
    RegistryMismatch,
    load_registry,
)

RUNTIME_AVAILABLE = (DEFAULT_RUNTIME_ROOT / "src/manager/task_plan.cpp").is_file()
SOURCE_AVAILABLE = gen.DEFAULT_SOURCE.is_file()
PILOT = ROOT / "data" / "manager" / "pilot_v1.jsonl"


def plan_text(*tasks: dict) -> str:
    return json.dumps({"tasks": list(tasks)}, ensure_ascii=False, separators=(",", ":"))


@unittest.skipUnless(RUNTIME_AVAILABLE, "slm-runtime sources not available")
class ValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = load_registry()

    def check(self, text, **kwargs):
        return v.validate_plan_text(text, self.registry, **kwargs)

    def test_registry_is_read_from_runtime(self):
        self.assertEqual(self.registry.skills, ("search", "summary"))
        self.assertEqual(self.registry.max_tasks, 8)
        self.assertEqual(len(self.registry.few_shots), 4)

    def test_registry_mismatch_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            for relative in SOURCE_FILES:
                target = Path(directory) / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(DEFAULT_RUNTIME_ROOT / relative, target)
            client = Path(directory) / "src/manager/manager_client.cpp"
            client.write_text(client.read_text().replace('| "\\"summary\\"" ws', '| "\\"summary\\"" ws | "\\"sql\\"" ws'))
            with self.assertRaises(RegistryMismatch):
                load_registry(directory)

    def test_valid_plan_acceptance(self):
        result = self.check(plan_text({"id": 0, "skill": "search", "input": "가상 메모리"},
                                      {"id": 1, "skill": "summary", "input_from": 0}))
        self.assertTrue(result.valid, result.to_dict())
        self.assertEqual(v.plan_depth(result.plan), 2)

    def test_invalid_skill_rejection(self):
        for skill in ("browser", "sql", "code", "Search"):
            result = self.check(plan_text({"id": 0, "skill": skill, "input": "x"}))
            self.assertIn(v.INVALID_SKILL, result.codes)

    def test_missing_required_argument(self):
        self.assertIn(v.MISSING_ARGUMENT, self.check('{"tasks":[{"id":0,"skill":"search"}]}').codes)
        self.assertIn(v.MISSING_ARGUMENT, self.check('{"tasks":[{"skill":"search","input":"x"}]}').codes)
        self.assertIn(v.MISSING_ARGUMENT, self.check('{"tasks":[{"id":0,"input":"x"}]}').codes)

    def test_invalid_input_from(self):
        out_of_range = self.check(plan_text({"id": 0, "skill": "search", "input": "x"},
                                            {"id": 1, "skill": "summary", "input_from": 7}))
        self.assertIn(v.INVALID_INPUT_FROM, out_of_range.codes)
        negative = self.check('{"tasks":[{"id":0,"skill":"search","input_from":-1}]}')
        self.assertIn(v.INVALID_INPUT_FROM, negative.codes)
        wrong_type = self.check('{"tasks":[{"id":0,"skill":"search","input_from":"0"}]}')
        self.assertIn(v.TYPE_ERROR, wrong_type.codes)

    def test_self_and_future_reference(self):
        self_ref = self.check(plan_text({"id": 0, "skill": "search", "input": "x"},
                                        {"id": 1, "skill": "summary", "input_from": 1}))
        self.assertIn(v.SELF_REFERENCE, self_ref.codes)
        future = self.check(plan_text({"id": 0, "skill": "search", "input": "x"},
                                      {"id": 1, "skill": "summary", "input_from": 2},
                                      {"id": 2, "skill": "search", "input": "y"}))
        self.assertIn(v.FUTURE_REFERENCE, future.codes)

    def test_cycle_detection(self):
        result = self.check(plan_text({"id": 0, "skill": "search", "input_from": 1},
                                      {"id": 1, "skill": "summary", "input_from": 0}))
        self.assertIn(v.CYCLIC_DEPENDENCY, result.codes)
        self.assertIn(v.FUTURE_REFERENCE, result.codes)

    def test_structure_errors(self):
        self.assertIn(v.UNKNOWN_FIELD, self.check('{"tasks":[{"id":0,"skill":"search","input":"x","why":1}]}').codes)
        self.assertIn(v.EMPTY_PLAN, self.check('{"tasks":[]}').codes)
        self.assertIn(v.EMPTY_INPUT, self.check('{"tasks":[{"id":0,"skill":"search","input":"  "}]}').codes)
        self.assertIn(v.INPUT_SOURCE_CONFLICT,
                      self.check('{"tasks":[{"id":0,"skill":"search","input":"x","input_from":0}]}').codes)
        self.assertIn(v.INVALID_TASK_ID, self.check('{"tasks":[{"id":1,"skill":"search","input":"x"}]}').codes)
        many = plan_text(*({"id": i, "skill": "search", "input": "x"} for i in range(9)))
        self.assertIn(v.TOO_MANY_TASKS, self.check(many).codes)
        self.assertIn(v.TYPE_ERROR, self.check('{"tasks":[{"id":true,"skill":"search","input":"x"}]}').codes)
        self.assertIn(v.JSON_PARSE_ERROR, self.check('{"tasks":[{"id":0.0,"skill":"search","input":"x"}]}').codes)
        self.assertIn(v.NO_JSON_OBJECT, self.check("plain text").codes)

    def test_utf8_checks(self):
        self.assertIn(v.UTF8_ERROR, self.check(plan_text({"id": 0, "skill": "search", "input": "깨진�"})).codes)
        self.assertIn(v.UTF8_ERROR, self.check(b'{"tasks":[{"id":0,"skill":"search","input":"\xed\xa0"}]}').codes)

    def test_canonical_mode_is_stricter_than_runtime(self):
        text = "Plan: " + plan_text({"id": 0, "skill": "search", "input": "x"})
        self.assertTrue(self.check(text, mode="runtime").valid)
        self.assertIn(v.NON_CANONICAL, self.check(text, mode="canonical").codes)
        spaced = '{"tasks": [{"id": 0, "skill": "search", "input": "x"}]}'
        self.assertIn(v.NON_CANONICAL, self.check(spaced).codes)

    def test_grounding(self):
        text = plan_text({"id": 0, "skill": "summary", "input": "가상 메모리"})
        self.assertTrue(self.check(text, request="가상 메모리를 요약해 줘").valid)
        self.assertIn(v.INPUT_NOT_GROUNDED, self.check(text, request="페이지 폴트를 요약해 줘").codes)

    def test_warnings(self):
        independent = self.check(plan_text({"id": 0, "skill": "summary", "input": "a"},
                                           {"id": 1, "skill": "search", "input": "b"}))
        self.assertTrue(independent.valid)
        self.assertIn(v.DISCARDED_OUTPUT, independent.warning_codes)
        chained = self.check(plan_text({"id": 0, "skill": "summary", "input": "a"},
                                       {"id": 1, "skill": "summary", "input_from": 0}))
        self.assertIn(v.REDUNDANT_SAME_SKILL_CHAIN, chained.warning_codes)
        duplicate = self.check(plan_text({"id": 0, "skill": "search", "input": "a"},
                                         {"id": 1, "skill": "search", "input": "a"}))
        self.assertIn(v.DUPLICATE_TASK, duplicate.warning_codes)
        ideographic = self.check(plan_text({"id": 0, "skill": "search", "input": "　"}))
        self.assertTrue(ideographic.valid)
        self.assertIn(v.UNICODE_WHITESPACE_INPUT, ideographic.warning_codes)

    def test_parity_cases_primary_code(self):
        for text, expected in v.PARITY_CASES:
            result = self.check(text, mode="runtime")
            if expected is None:
                self.assertTrue(result.valid, (text, result.codes))
            else:
                self.assertFalse(result.valid, text)
                self.assertEqual(result.codes[0], expected, text)


@unittest.skipUnless(RUNTIME_AVAILABLE and bool(shutil.which("clang++") or shutil.which("g++")),
                     "C++ compiler or runtime sources unavailable")
class OracleParityTests(unittest.TestCase):
    def test_python_matches_cpp_task_plan(self):
        registry = load_registry()
        texts = [text for text, _ in v.PARITY_CASES]
        for text, (accepted, detail) in zip(texts, v.runtime_oracle(texts)):
            self.assertEqual(v.validate_plan_text(text, registry, mode="runtime").valid, accepted, (text, detail))

    def test_canonical_json_matches_runtime_to_json(self):
        plans = [
            {"tasks": [{"id": 0, "skill": "search", "input": "한국어 \"인용\" / 역슬래시 \\ 끝"}]},
            {"tasks": [{"id": 0, "skill": "summary", "input": "첫 줄\n둘째 줄"},
                       {"id": 1, "skill": "search", "input_from": 0}]},
        ]
        texts = [v.canonical_json(plan) for plan in plans]
        for text, (accepted, detail) in zip(texts, v.runtime_oracle(texts)):
            self.assertTrue(accepted, detail)
            self.assertEqual(detail, text)


@unittest.skipUnless(RUNTIME_AVAILABLE and SOURCE_AVAILABLE, "runtime or AI Hub source unavailable")
class GeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.small = gen.generate(count=40, seed=11)

    def test_exactly_requested_sample_count(self):
        self.assertEqual(len(self.small.rows), 40)
        self.assertEqual(len(gen.generate(count=23, seed=5).rows), 23)

    def test_same_seed_produces_same_pilot(self):
        again = gen.generate(count=40, seed=11)
        self.assertEqual(json.dumps(self.small.rows, ensure_ascii=False),
                         json.dumps(again.rows, ensure_ascii=False))

    def test_different_seed_changes_realization(self):
        other = gen.generate(count=40, seed=12)
        same = sum(a["request"] == b["request"] for a, b in zip(self.small.rows, other.rows))
        self.assertLess(same, 5)

    def test_metadata_completeness(self):
        for row in self.small.rows:
            for key in REQUIRED_METADATA:
                self.assertIn(key, row["metadata"], key)
            self.assertEqual(row["metadata"]["status"], "candidate")
            self.assertEqual(row["metadata"]["generator_version"], gen.GENERATOR_VERSION)

    def test_plans_valid_and_policy_rows_unlabeled(self):
        registry = load_registry()
        for row in self.small.rows:
            if row["metadata"]["requires_human_policy"]:
                self.assertIsNone(row["plan"])
                self.assertFalse(row["metadata"]["training_candidate"])
                self.assertEqual(row["metadata"]["split"], gen.POLICY_SPLIT)
            else:
                result = v.validate_plan_text(row["plan_text"], registry, request=row["request"])
                self.assertTrue(result.valid, (row["id"], result.to_dict()))

    def test_duplicate_detection(self):
        self.assertEqual(gen.normalize_request("요약해 줘!"), gen.normalize_request("요약해줘"))
        generator = gen.PilotGenerator(3, 10, gen.DEFAULT_SOURCE, DEFAULT_RUNTIME_ROOT)
        family = FAMILY_BY_ID["un_code"]  # one authored request only
        spec = {"semantic_id": "sem-x", "taxonomy": family.taxonomy, "source_kind": "authored",
                "args": {"authored": family.authored[0]}, "entity_family": ["authored"], "split": gen.POLICY_SPLIT}
        self.assertTrue(generator.realize(spec, family, "hard_case", None))
        self.assertFalse(generator.realize(copy.deepcopy(spec), family, "hard_case", None))
        self.assertEqual(generator.rejects["DUPLICATE_REQUEST_EXACT"], gen.MAX_ATTEMPTS)
        self.assertEqual(len(generator.rows), 1)

    def test_skill_list_comes_from_runtime(self):
        self.assertTrue(referenced_skills() <= set(load_registry().skills))
        original = gen.load_registry

        def shrunk(root):
            registry = original(root)
            return registry.__class__(**{**registry.__dict__, "skills": ("search",)})

        gen.load_registry = shrunk
        try:
            with self.assertRaises(RuntimeError):
                gen.PilotGenerator(1, 10, gen.DEFAULT_SOURCE, DEFAULT_RUNTIME_ROOT)
        finally:
            gen.load_registry = original


@unittest.skipUnless(RUNTIME_AVAILABLE and SOURCE_AVAILABLE, "runtime or AI Hub source unavailable")
class SplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = gen.generate(count=200, seed=gen.DEFAULT_SEED).rows

    def test_every_family_has_one_split(self):
        assignment = gen.assign_family_splits(gen.DEFAULT_SEED)
        for row in self.rows:
            self.assertEqual(row["metadata"]["split"], assignment[row["metadata"]["template_family"]])

    def test_group_aware_split_has_no_leakage(self):
        report = leakage(self.rows)
        for key in ("template_family_across_splits", "entity_family_across_splits",
                    "semantic_seed_across_splits", "paraphrase_sibling_across_splits",
                    "policy_rows_inside_training_splits", "near_duplicate_pairs_cross_split"):
            self.assertEqual(report[key], [], key)

    def test_leakage_detector_flags_injected_leak(self):
        rows = copy.deepcopy(self.rows)
        sibling = next(r for r in rows if r["metadata"]["paraphrase_of"])
        sibling["metadata"]["split"] = "test" if sibling["metadata"]["split"] != "test" else "train"
        report = leakage(rows)
        self.assertIn(sibling["id"], report["paraphrase_sibling_across_splits"])
        self.assertIn(sibling["metadata"]["semantic_id"], report["semantic_seed_across_splits"])
        self.assertIn(sibling["metadata"]["template_family"], report["template_family_across_splits"])


@unittest.skipUnless(PILOT.is_file() and RUNTIME_AVAILABLE and SOURCE_AVAILABLE, "pilot not generated")
class PilotArtifactTests(unittest.TestCase):
    def test_pilot_is_exactly_200_candidates_and_reproducible(self):
        text = PILOT.read_text(encoding="utf-8")
        rows = [json.loads(line) for line in text.splitlines()]
        self.assertEqual(len(rows), 200)
        self.assertEqual(len({row["id"] for row in rows}), 200)
        self.assertTrue(all(row["metadata"]["status"] == "candidate" for row in rows))
        seed = rows[0]["metadata"]["seed"]
        regenerated = gen.generate(count=200, seed=seed).rows
        self.assertEqual(text, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in regenerated))


if __name__ == "__main__":
    unittest.main()
