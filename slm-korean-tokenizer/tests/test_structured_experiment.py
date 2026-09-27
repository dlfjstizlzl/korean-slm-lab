from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tokenizer"))

from common import CONTROL_TOKENS, assert_control_tokens_atomic, load_tokenizer


class StructuredExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = json.loads((ROOT / "outputs" / "structured_corpus_report.json").read_text(encoding="utf-8"))
        cls.variant = json.loads((ROOT / "outputs" / "structured_variant_report.json").read_text(encoding="utf-8"))
        cls.benchmark = json.loads(
            (ROOT / "outputs" / "tokenizer_structured_benchmark.json").read_text(encoding="utf-8")
        )

    def test_external_noise_is_removed_and_split_is_document_level(self):
        dropped = self.sources["json_extraction"]["dropped"]
        # The 2.0 MB package-lock dominates the malformed mass and must not survive.
        self.assertEqual(dropped["malformed"], 32)
        self.assertEqual(dropped["minified"], 67)
        self.assertGreater(dropped["malformed_chars"], 2_000_000)
        stack = self.sources["stack_sql_extraction"]["dropped"]
        self.assertEqual(stack["oversized_bulk_dump"], 44)
        self.assertEqual(self.sources["spider_sql_extraction"]["dropped"], {})
        split = self.sources["split"]["__split__"]
        self.assertEqual(split["seed"], 42)
        self.assertEqual(split["valid_ratio"], 0.10)
        self.assertEqual(split["validation_lines_removed_for_train_overlap"], 380)

    def test_variant_holds_the_mixed10_controls(self):
        # Same total size and same 10% mixed budget as KO-32K-mixed10; only the
        # composition inside that budget changed.
        self.assertEqual(self.variant["total_characters"], 92_514_113)
        self.assertEqual(self.variant["seed"], 42)
        self.assertAlmostEqual(self.variant["actual_mixed_percent"], 10.0083302966)
        shares = self.variant["family_percent_of_total"]
        self.assertLess(shares["source_code"], 4.0)
        self.assertGreater(shares["json"], 1.0)
        self.assertGreater(shares["sql"], 1.0)

    def test_structured_corpus_has_no_repeated_lines(self):
        path = ROOT / "data" / "processed" / "tokenizer_train_mixed10_structured.txt"
        seen: set[str] = set()
        total = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                total += 1
                seen.add(line.rstrip("\n"))
        self.assertEqual(total, len(seen))
        self.assertEqual(total, self.variant["total_lines"])

    def test_tokenizer_invariants_hold(self):
        tokenizer = load_tokenizer(str(ROOT / "outputs" / "tokenizer-ko-32000-mixed10-structured"))
        metadata = json.loads(
            (ROOT / "outputs" / "tokenizer-ko-32000-mixed10-structured" / "training_metadata.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(len(tokenizer), 32_000)
        self.assertEqual(metadata["min_frequency"], 2)
        self.assertEqual(len(metadata["preserved_base_added_token_ids"]), 26)
        self.assertEqual(set(assert_control_tokens_atomic(tokenizer)), set(CONTROL_TOKENS))
        for text in (
            "가상메모리는 프로세스에게 물리 메모리와 독립된 주소 공간을 제공한다.",
            '{"skill":"search","input":"Apple Silicon unified memory"}',
            "SELECT id, name FROM users WHERE created_at >= '2026-09-01';",
        ):
            self.assertEqual(tokenizer.decode(tokenizer.encode(text, add_special_tokens=False)), text)

    def test_structured_recovery_without_korean_collapse(self):
        recovery = self.benchmark["recovery_vs_mixed10"]
        # Korean is allowed to move by at most one percentage point.
        self.assertLess(abs(recovery["korean_aggregate_penalty_points"]), 1.0)
        self.assertGreater(recovery["json_recovery_points"], 10.0)
        self.assertGreater(recovery["sql_recovery_points"], 20.0)
        self.assertGreater(recovery["code_url_delta_points"], 10.0)
        self.assertGreaterEqual(recovery["skill_delta_points"], 0.0)

    def test_structured_benchmarks_do_not_overlap_training(self):
        train: set[str] = set()
        with (ROOT / "data" / "processed" / "tokenizer_train_mixed10_structured.txt").open(
            "r", encoding="utf-8"
        ) as handle:
            for line in handle:
                train.add(line.rstrip("\n"))
        for name in ("eval_json.txt", "eval_sql.txt", "eval_skilllm_structured.txt"):
            with (ROOT / "data" / name).open("r", encoding="utf-8") as handle:
                overlap = sum(1 for line in handle if line.rstrip("\n") in train)
            self.assertEqual(overlap, 0, f"{name} overlaps the training corpus")

    def test_vocab_budget_is_unchanged(self):
        tokenizers = self.benchmark["tokenizers"]
        for name in ("KO-32K-current", "KO-32K-mixed10", "KO-32K-mixed10-structured"):
            composition = tokenizers[name]["vocab_composition"]["counts"]
            self.assertEqual(sum(composition.values()), 32_000)
            self.assertEqual(
                tokenizers[name]["embedding_estimate"]["vocabulary_dependent_parameters"], 32_000 * 1024
            )


if __name__ == "__main__":
    unittest.main()
