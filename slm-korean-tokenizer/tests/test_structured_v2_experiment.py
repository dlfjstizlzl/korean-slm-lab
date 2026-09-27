from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tokenizer"))

from common import CONTROL_TOKENS, assert_control_tokens_atomic, load_tokenizer

VARIANTS = {
    "KO-32K-mixed10-structured-v2": "tokenizer-ko-32000-mixed10-structured-v2",
    "KO-32K-mixed10-structured-v2-json2": "tokenizer-ko-32000-mixed10-structured-v2-json2",
}
CORPORA = {
    "KO-32K-mixed10-structured-v2": "tokenizer_train_mixed10_structured_v2.txt",
    "KO-32K-mixed10-structured-v2-json2": "tokenizer_train_mixed10_structured_v2_json2.txt",
}
BENCHMARK_FILES = (
    "eval_json.txt",
    "eval_json_new.txt",
    "eval_sql.txt",
    "eval_skilllm_structured.txt",
    "eval_skill_dsl.txt",
    "eval_code_url.txt",
    "korean_eval.txt",
)
# Inherited from the mixed10 experiment: one URL extracted from project files
# happens to equal a line of the fixed eight-line code/URL probe. It affects
# every mixed-family tokenizer identically, so it is recorded, not silently fixed.
KNOWN_PREEXISTING_LEAK = {"https://example.com/test?q=memory"}


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle if line.strip()]


class StructuredV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.harvest = json.loads((ROOT / "outputs" / "json_extra_corpus_report.json").read_text(encoding="utf-8"))
        cls.split = json.loads((ROOT / "outputs" / "json_extra_split_report.json").read_text(encoding="utf-8"))
        cls.benchmark = json.loads(
            (ROOT / "outputs" / "tokenizer_structured_v2_benchmark.json").read_text(encoding="utf-8")
        )

    def test_harvest_stopped_on_clean_characters(self):
        self.assertEqual(self.harvest["stop_reason"], "target reached")
        self.assertGreaterEqual(self.harvest["clean_characters"], 3_000_000)
        counts = self.harvest["counts"]
        self.assertEqual(counts["kept"], counts["object"] + counts["array"])
        self.assertEqual(counts["duplicate_new"], 0)
        # The 200k cap never binds: this subset caps documents near 5k characters.
        self.assertEqual(counts["too_large"], 0)
        self.assertLess(self.harvest["size_distribution"]["max"], 10_000)

    def test_split_is_repository_level_with_no_overlap(self):
        self.assertEqual(self.split["split_unit"], "repository")
        self.assertEqual(self.split["seed"], 42)
        self.assertEqual(
            self.split["train_repositories"] + self.split["valid_repositories"],
            self.split["repositories_total"],
        )
        train = set(read_lines(ROOT / "data/processed/structured/json_extra_train.txt"))
        valid = set(read_lines(ROOT / "data/processed/structured/json_extra_valid.txt"))
        self.assertEqual(train & valid, set())

    def test_no_benchmark_leaks_into_any_v2_corpus(self):
        for label, corpus in CORPORA.items():
            trained = {sha256(line) for line in read_lines(ROOT / "data/processed" / corpus)}
            for name in BENCHMARK_FILES:
                leaked = [
                    line for line in read_lines(ROOT / "data" / name)
                    if sha256(line) in trained and line not in KNOWN_PREEXISTING_LEAK
                ]
                self.assertEqual(leaked, [], f"{name} leaked into {label}: {leaked[:3]}")

    def test_new_holdout_unseen_by_every_baseline(self):
        holdout = read_lines(ROOT / "data/eval_json_new.txt")
        self.assertGreater(len(holdout), 1_000)
        for corpus in (
            "tokenizer_train_mixed10_structured.txt",
            "tokenizer_train_mixed10.txt",
            "tokenizer_train_real.txt",
        ):
            seen = {sha256(line) for line in read_lines(ROOT / "data/processed" / corpus)}
            self.assertEqual(sum(1 for line in holdout if sha256(line) in seen), 0, corpus)

    def test_corpora_hold_the_controls_and_repeat_nothing(self):
        for label, corpus in CORPORA.items():
            lines = read_lines(ROOT / "data/processed" / corpus)
            self.assertEqual(len(lines), len(set(lines)), f"{label} repeats lines")
            total = sum(len(line) for line in lines)
            # Same corpus scale as the v1 structured variant (92,514,113).
            self.assertLess(abs(total - 92_514_113), 100_000, f"{label} changed corpus scale")

    def test_tokenizer_invariants(self):
        for label, directory in VARIANTS.items():
            tokenizer = load_tokenizer(str(ROOT / "outputs" / directory))
            metadata = json.loads(
                (ROOT / "outputs" / directory / "training_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(tokenizer), 32_000, label)
            self.assertEqual(metadata["min_frequency"], 2, label)
            self.assertEqual(len(metadata["preserved_base_added_token_ids"]), 26, label)
            self.assertEqual(set(assert_control_tokens_atomic(tokenizer)), set(CONTROL_TOKENS), label)
            self.assertEqual(tokenizer.eos_token, "<|im_end|>")
            self.assertEqual(tokenizer.pad_token, "<|endoftext|>")
            for text in (
                "가상메모리는 프로세스에게 물리 메모리와 독립된 주소 공간을 제공한다.",
                '{"tasks":[{"id":0,"skill":"search","input":"메모리"}]}',
                "SELECT id, name FROM users WHERE created_at >= '2026-09-01';",
            ):
                self.assertEqual(tokenizer.decode(tokenizer.encode(text, add_special_tokens=False)), text)

    def test_json_improves_without_korean_or_sql_collapse(self):
        recovery = self.benchmark["recovery_vs_structured_v1"]["KO-32K-mixed10-structured-v2"]
        self.assertLess(abs(recovery["korean_aggregate_penalty_points"]), 0.5)
        self.assertGreater(recovery["json_old_recovery_points"], 0.0)
        # SQL and code/URL must not degrade; negative penalty means they improved.
        self.assertLessEqual(recovery["sql_penalty_points"], 0.5)
        self.assertLessEqual(recovery["code_url_penalty_points"], 0.5)

    def test_json_dose_response_is_monotonic_but_saturating(self):
        summary = self.benchmark["summary"]
        regressions = [
            -summary[name]["json_old_reduction_percent"]
            for name in (
                "KO-32K-mixed10",
                "KO-32K-mixed10-structured",
                "KO-32K-mixed10-structured-v2-json2",
                "KO-32K-mixed10-structured-v2",
            )
        ]
        self.assertEqual(regressions, sorted(regressions, reverse=True))
        first_step = regressions[0] - regressions[1]
        later_steps = regressions[1] - regressions[3]
        # More than five times the JSON data buys less than the first increment did.
        self.assertGreater(first_step, later_steps)

    def test_vocab_budget_unchanged(self):
        for name in VARIANTS:
            entry = self.benchmark["tokenizers"][name]
            self.assertEqual(sum(entry["vocab_composition"]["counts"].values()), 32_000)
            self.assertEqual(entry["embedding_estimate"]["vocabulary_dependent_parameters"], 32_000 * 1024)


if __name__ == "__main__":
    unittest.main()
