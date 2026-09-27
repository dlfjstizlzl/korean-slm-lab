from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tokenizer"))

from common import CONTROL_TOKENS, assert_control_tokens_atomic, load_tokenizer


class MixedExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = json.loads((ROOT / "outputs" / "mixed_corpus_report.json").read_text(encoding="utf-8"))
        cls.benchmark = json.loads(
            (ROOT / "outputs" / "tokenizer_mixed_benchmark.json").read_text(encoding="utf-8")
        )

    def test_mixed_corpora_are_real_scale_without_repetition(self):
        self.assertFalse(self.corpus["synthetic_repetition"])
        self.assertEqual(self.corpus["mixed_pool"]["total"]["characters"], 29_863_845)
        variants = {item["label"]: item for item in self.corpus["variants"]}
        self.assertEqual(variants["mixed10"]["total_characters"], 92_514_012)
        self.assertAlmostEqual(variants["mixed10"]["actual_mixed_percent"], 9.9723888312)
        self.assertEqual(variants["mixed15"]["total_characters"], 92_476_931)
        self.assertAlmostEqual(variants["mixed15"]["actual_mixed_percent"], 14.9666169177)

    def test_project_file_holdout_and_validation_are_recorded(self):
        discovered = self.corpus["extraction"]["discovered"]
        self.assertEqual(discovered["project_files"], 2085)
        self.assertEqual(discovered["project_train_files"], 1872)
        self.assertEqual(discovered["project_validation_files"], 213)
        validation = self.corpus["validation"]
        self.assertEqual(validation["retained"], 48159)
        self.assertEqual(validation["train_overlap_removed"], 13774)
        self.assertEqual(validation["statistics"]["duplicate_ratio_percent"], 0.0)

    def test_validation_has_no_exact_overlap_with_either_training_variant(self):
        with (ROOT / "data" / "processed" / "mixed" / "mixed_validation.txt").open(encoding="utf-8") as handle:
            validation_hashes = {
                hashlib.blake2b(line.rstrip("\r\n").encode("utf-8"), digest_size=16).digest()
                for line in handle
                if line.rstrip("\r\n")
            }
        self.assertEqual(len(validation_hashes), 48159)
        for label in ("mixed10", "mixed15"):
            path = ROOT / "data" / "processed" / f"tokenizer_train_{label}.txt"
            overlap = 0
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    text = line.rstrip("\r\n")
                    if text and hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest() in validation_hashes:
                        overlap += 1
            self.assertEqual(overlap, 0, label)

    def test_mixed_tokenizers_preserve_budget_pipeline_and_added_tokens(self):
        current = load_tokenizer(str(ROOT / "outputs" / "real" / "mf2" / "tokenizer-ko-32000"))
        for label in ("mixed10", "mixed15"):
            path = ROOT / "outputs" / f"tokenizer-ko-32000-{label}"
            tokenizer = load_tokenizer(str(path))
            metadata = json.loads((path / "training_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(len(tokenizer), 32000)
            self.assertEqual(metadata["min_frequency"], 2)
            self.assertEqual(metadata["name_suffix"], label)
            self.assertTrue(metadata["keep_comment_lines"])
            self.assertEqual(len(metadata["preserved_base_added_token_ids"]), 26)
            self.assertEqual(set(assert_control_tokens_atomic(tokenizer)), set(CONTROL_TOKENS))
            self.assertEqual(type(tokenizer.backend_tokenizer.normalizer), type(current.backend_tokenizer.normalizer))
            self.assertEqual(type(tokenizer.backend_tokenizer.pre_tokenizer), type(current.backend_tokenizer.pre_tokenizer))
            self.assertEqual(type(tokenizer.backend_tokenizer.decoder), type(current.backend_tokenizer.decoder))
            self.assertEqual(type(tokenizer.backend_tokenizer.post_processor), type(current.backend_tokenizer.post_processor))

    def test_benchmark_captures_korean_penalty_and_code_recovery(self):
        summary = self.benchmark["summary"]
        tradeoff = self.benchmark["tradeoff"]
        # mixed10 expectations were refreshed after the tokenizer was retrained on
        # the current corpus; the earlier constants came from a superseded build.
        self.assertAlmostEqual(summary["KO-32K-mixed10"]["fineweb_reduction_percent"], 34.6566359011)
        self.assertAlmostEqual(summary["KO-32K-mixed15"]["aihub_reduction_percent"], 39.8063494487)
        self.assertAlmostEqual(summary["KO-32K-mixed10"]["code_url_reduction_percent"], -41.7475728155)
        self.assertAlmostEqual(summary["KO-32K-mixed15"]["code_url_reduction_percent"], -38.8349514563)
        self.assertAlmostEqual(tradeoff["KO-32K-mixed10"]["code_recovery_ratio_percent"], 42.6666666667)
        self.assertAlmostEqual(tradeoff["KO-32K-mixed15"]["code_recovery_ratio_percent"], 46.6666666667)

    def test_vocab_composition_and_embedding_budget_are_complete(self):
        tokenizers = self.benchmark["tokenizers"]
        expected_parameters = 32_000 * 1024
        for name in ("KO-32K-current", "KO-32K-mixed10", "KO-32K-mixed15"):
            composition = tokenizers[name]["vocab_composition"]["counts"]
            self.assertEqual(sum(composition.values()), 32000)
            self.assertEqual(tokenizers[name]["embedding_estimate"]["vocabulary_dependent_parameters"], expected_parameters)
        self.assertGreater(
            tokenizers["KO-32K-mixed10"]["vocab_composition"]["counts"]["code-symbol dominant"],
            tokenizers["KO-32K-current"]["vocab_composition"]["counts"]["code-symbol dominant"],
        )


if __name__ == "__main__":
    unittest.main()
