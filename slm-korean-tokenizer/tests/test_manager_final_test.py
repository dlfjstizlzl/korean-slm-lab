"""Final sealed-test evaluation guards and analysis math (never reads the real sealed test)."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))

import manager_final_analysis as fa  # noqa: E402
import manager_sft_data as data  # noqa: E402
from manager_v2_registry import TOOL  # noqa: E402

TOOL_AVAILABLE = TOOL.is_file()
FINAL_OUT = ROOT / "outputs" / "manager_final_test_v1"


def fake_rows(n: int) -> list[dict]:
    plan = {"status": "clarify", "message": "요약할 내용을 제공해 주세요.", "tasks": [], "outputs": []}
    return [{"id": f"t{i}", "request": "요약해줘", "plan": plan,
             "plan_text": json.dumps(plan, ensure_ascii=False, separators=(",", ":")),
             "metadata": {"sealed_test": True, "taxonomy": "clarify", "taxonomy_group": "special", "hard_case": False}}
            for i in range(n)]


class SealedGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.test_file = self.dir / "test.jsonl"
        self.test_file.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in fake_rows(2)) + "\n",
                                  encoding="utf-8")
        digest = hashlib.sha256(self.test_file.read_bytes()).hexdigest()
        self.patches = [mock.patch.dict(data.SPLIT_FILES, {"test": self.test_file}),
                        mock.patch.dict(data.EXPECTED_ROWS, {"test": 2}),
                        mock.patch.object(data, "SEALED_TEST_SHA256", digest),
                        mock.patch.object(data, "CONSUMED_MARKER", self.dir / "SEALED_TEST_CONSUMED.json")]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_default_refuses_and_explicit_unlock_reads(self):
        with self.assertRaises(data.SealedTestError):
            data.load_split("test")
        self.assertEqual(len(data.load_split("test", allow_sealed_test=True)), 2)

    def test_hash_guard(self):
        self.test_file.write_text(self.test_file.read_text() + "\n{}\n")
        with self.assertRaises(data.SealedTestError):
            data.load_split("test", allow_sealed_test=True)

    def test_consumed_marker_refuses_rerun_by_default(self):
        data.CONSUMED_MARKER.write_text('{"status":"consumed"}')
        with self.assertRaises(data.SealedTestConsumedError):
            data.load_split("test", allow_sealed_test=True)
        self.assertEqual(len(data.load_split("test", allow_sealed_test=True, allow_consumed_test=True)), 2)

    def test_run_requires_explicit_flag_and_never_overrides_by_default(self):
        import manager_final_test as ft

        with mock.patch.object(sys, "argv", ["x", "run"]), self.assertRaises(SystemExit):
            ft.main()
        source = (ROOT / "model" / "manager_final_test.py").read_text(encoding="utf-8")
        self.assertIn("allow_consumed_test=args.override_consumed", source)
        self.assertNotIn("allow_consumed_test=True", source)
        self.assertEqual(source.count('data.load_split("test"'), 1)

    def test_training_code_keeps_guard(self):
        for name in ("manager_sft_train.py", "manager_sft_eval.py", "manager_sft_report.py", "manager_final_analysis.py"):
            text = (ROOT / "model" / name).read_text(encoding="utf-8")
            self.assertNotIn("allow_sealed_test=True", text)


class StatsTests(unittest.TestCase):
    def test_wilson_known_values(self):
        lo, hi = fa.wilson(5, 10)
        self.assertAlmostEqual(lo, 0.2366, places=4)
        self.assertAlmostEqual(hi, 0.7634, places=4)
        lo, hi = fa.wilson(0, 10)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.2775, places=4)
        lo, hi = fa.wilson(10, 10)
        self.assertAlmostEqual(lo, 0.7225, places=4)
        self.assertAlmostEqual(hi, 1.0, places=12)

    def test_newcombe_published_example(self):
        # Newcombe (1998) example: 56/70 vs 48/80 -> difference 0.2000, 95% CI 0.0524 to 0.3339
        d, lo, hi = fa.newcombe_diff(48, 80, 56, 70)
        self.assertAlmostEqual(d, 0.2, places=4)
        self.assertAlmostEqual(lo, 0.0524, places=4)
        self.assertAlmostEqual(hi, 0.3339, places=4)

    def test_paired_counts_and_mcnemar(self):
        result = fa.paired([True, True, False, False, True], [True, False, True, False, True])
        self.assertEqual((result["both_correct"], result["base_only_correct"], result["lora_only_correct"],
                          result["both_wrong"]), (2, 1, 1, 1))
        self.assertEqual(result["mcnemar_p"], 1.0)
        self.assertAlmostEqual(fa.paired([False] * 10, [True] * 10)["mcnemar_p"], 2 / 2 ** 10)
        with self.assertRaises(ValueError):
            fa.paired([True], [True, False])

    def test_rate_ci(self):
        c = fa.rate_ci([True, True, False, True])
        self.assertEqual((c["k"], c["n"], c["rate"]), (3, 4, 0.75))


class FidelityTests(unittest.TestCase):
    generic = {"검색", "조회", "요약", "비교", "정보", "최신"}

    def cls(self, gold, pred, request, skill="search"):
        return fa.classify_instruction(gold, pred, request, skill, self.generic)["class"]

    def test_classes(self):
        self.assertIsNone(fa.classify_instruction("a b", "a b", "a b", "search", self.generic))
        self.assertEqual(self.cls("공식 발표 기준 젓갈 역사 검색", "공식 발표 기준젓갈 역사 검색", "젓갈의 역사를 공식 발표 기준"), "A")
        self.assertEqual(self.cls("가진항 위치 정보 검색", "진취가 위치 정보 검색", "가진항 관련 위치 정보를 찾아 줘"), "F")
        self.assertEqual(self.cls("지난해 상위 5개 학년 조회", "지난해 상위 5개 학 year 조회", "지난해 상위 5개 학년을 뽑아", "sql"), "C")
        self.assertEqual(self.cls("수혜 인원과 적용 기준 비교", "수혜 인원 비교", "수혜 인원과 적용 기준을 비교", "compare"), "D")
        self.assertEqual(self.cls("정의 추출", "정의 정리본 추출", "정리본에서 정의만 뽑아", "extract"), "E")
        self.assertEqual(self.cls("화산 최신 정보 검색", "화산 정보 검색", "화산에 대해 알려줘"), "B")

    def test_duplicate_source_and_probes(self):
        self.assertTrue(fa.duplicate_source('{"status":"ok","tasks":[{"id":0,"sources":[0,0]}],"outputs":[0]}'))
        self.assertFalse(fa.duplicate_source('{"status":"ok","tasks":[{"id":1,"sources":["user",0]}],"outputs":[1]}'))
        self.assertFalse(fa.duplicate_source("not json"))
        row = {"request": "A와 B 가격. 비교는 필요 없고 각각 보여 줘.",
               "plan": {"status": "ok", "tasks": [{"id": 0, "skill": "search", "instruction": "x", "sources": []}],
                        "outputs": [0]}, "metadata": {"taxonomy": "multi_output"}}
        tags = fa.probes(row)
        self.assertIn("negated_compare", tags)
        self.assertIn("compare_cue_without_compare", tags)
        self.assertIn("search_without_cue", tags)


@unittest.skipUnless(TOOL_AVAILABLE, "runtime tool unavailable")
class ParityAndSerializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = data.load_split("validation")[:40]
        stored = json.loads((ROOT / "outputs" / "manager_sft_v1" / "best_val_constrained.json").read_text())
        by_id = {r["id"]: r for r in stored["records"]}
        cls.outputs = [by_id[r["id"]]["output"] for r in cls.rows]

    def test_same_pipeline_as_validation(self):
        import manager_final_test as ft
        import manager_sft_eval as ev

        self.assertIs(ft.evaluate, ev.evaluate)
        self.assertIs(ft.generate, ev.generate)
        self.assertEqual(ft.EVAL_CONFIG["max_new_tokens"], ev.MAX_NEW_TOKENS)
        train_source = (ROOT / "model" / "manager_sft_train.py").read_text(encoding="utf-8")
        self.assertIn(f"batch_size={ft.EVAL_CONFIG['batch_size']}", train_source)

    def test_metric_determinism_and_serialization_roundtrip(self):
        import manager_final_test as ft
        from manager_sft_eval import evaluate

        first = evaluate(self.rows, self.outputs)
        second = evaluate(self.rows, self.outputs)
        self.assertEqual(first["overall"], second["overall"])
        lines = [json.dumps(ft.serialize(r, rec, "lora", "constrained"), ensure_ascii=False, sort_keys=True)
                 for r, rec in zip(self.rows, first["records"])]
        restored = [json.loads(line) for line in lines]
        again = evaluate([ft.as_row(p) for p in restored], [p["generated"] for p in restored])
        self.assertEqual(again["overall"], first["overall"])
        self.assertEqual([p["id"] for p in restored], [r["id"] for r in self.rows])


@unittest.skipUnless((FINAL_OUT / "predictions").is_dir(), "final evaluation predictions are local-only")
class FinalArtifactTests(unittest.TestCase):
    def test_every_row_evaluated_exactly_once_per_mode_and_hashes(self):
        hashes = json.loads((FINAL_OUT / "prediction_hashes.json").read_text())["predictions"]
        self.assertEqual(set(hashes), {"base_raw.jsonl", "base_constrained.jsonl", "lora_raw.jsonl",
                                       "lora_constrained.jsonl"})
        orders = []
        for name, digest in hashes.items():
            path = FINAL_OUT / "predictions" / name
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            ids = [json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(ids), 461)
            self.assertEqual(len(set(ids)), 461)
            orders.append(ids)
        self.assertTrue(all(o == orders[0] for o in orders))

    def test_consumed_marker_matches(self):
        marker = json.loads(data.CONSUMED_MARKER.read_text())
        self.assertEqual(marker["status"], "consumed")
        self.assertEqual(marker["test_sha256"], data.SEALED_TEST_SHA256)
        with self.assertRaises(data.SealedTestConsumedError):
            data.load_split("test", allow_sealed_test=True)


if __name__ == "__main__":
    unittest.main()
