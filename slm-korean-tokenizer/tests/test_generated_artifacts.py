from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tokenizer"))

from common import CONTROL_TOKENS, assert_control_tokens_atomic, load_tokenizer


@unittest.skipUnless((ROOT / "outputs").is_dir(), "local artifact not in repository: outputs/")
class GeneratedArtifactTests(unittest.TestCase):
    def test_hf_tokenizer_sizes_and_control_tokens(self):
        for size in (16000, 24000, 32000):
            path = ROOT / "outputs" / f"tokenizer-ko-{size}"
            self.assertTrue(path.is_dir(), path)
            tokenizer = load_tokenizer(str(path))
            self.assertEqual(len(tokenizer), size)
            self.assertEqual(set(assert_control_tokens_atomic(tokenizer)), set(CONTROL_TOKENS))

    def test_benchmark_contains_all_candidates(self):
        report = json.loads((ROOT / "outputs" / "benchmark.json").read_text(encoding="utf-8"))
        self.assertEqual(report["corpus"]["eval_lines"], 40)
        self.assertEqual(len(report["tokenizers"]), 4)
        self.assertEqual(report["model"]["original_vocab"], 151936)

    def test_sentencepiece_comparison_preserves_control_tokens(self):
        tokenizer = load_tokenizer(str(ROOT / "outputs" / "tokenizer-sp-ko-16000"))
        self.assertEqual(len(tokenizer), 5655)
        self.assertEqual(set(assert_control_tokens_atomic(tokenizer)), set(CONTROL_TOKENS))

    def test_sentencepiece_algorithm_matrix_preserves_control_tokens(self):
        expected_sizes = {
            "tokenizer-sp-unigram-ko-8000": 5667,
            "tokenizer-sp-unigram-ko-16000": 5667,
            "tokenizer-sp-bpe-ko-8000": 8000,
            "tokenizer-sp-bpe-ko-16000": 16000,
        }
        for name, expected_size in expected_sizes.items():
            tokenizer = load_tokenizer(str(ROOT / "outputs" / "algorithms" / name))
            self.assertEqual(len(tokenizer), expected_size)
            self.assertEqual(set(assert_control_tokens_atomic(tokenizer)), set(CONTROL_TOKENS))

    def test_fixed_tokenizers_preserve_base_added_tokens_inside_budget(self):
        for size in (16000, 24000, 32000):
            path = ROOT / "outputs" / "fixed" / f"tokenizer-ko-{size}"
            tokenizer = load_tokenizer(str(path))
            metadata = json.loads((path / "training_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(len(tokenizer), size)
            self.assertEqual(metadata["min_frequency"], 0)
            self.assertEqual(len(metadata["preserved_base_added_token_ids"]), 26)

    def test_diagnostic_split_is_disjoint_and_stratified(self):
        train = (ROOT / "data" / "diagnostic_train.txt").read_text(encoding="utf-8").splitlines()
        evaluation = (ROOT / "data" / "diagnostic_eval.txt").read_text(encoding="utf-8").splitlines()
        metadata = json.loads((ROOT / "data" / "diagnostic_split.json").read_text(encoding="utf-8"))
        self.assertEqual(len(train), 12528)
        self.assertEqual(len(evaluation), 1392)
        self.assertFalse(set(train) & set(evaluation))
        self.assertEqual(metadata["seed"], 42)

    def test_transplant_and_validation_reports_are_consistent(self):
        transplant = json.loads((ROOT / "outputs" / "qwen3-0.6b-ko-init" / "transplant_report.json").read_text(encoding="utf-8"))
        validation = json.loads((ROOT / "outputs" / "transplant_validation.json").read_text(encoding="utf-8"))
        initialized = transplant["exact_match"] + transplant["averaged"] + transplant["random_fallback"]
        self.assertEqual(initialized, transplant["new_vocab"])
        self.assertEqual(validation["config_vocab_size"], transplant["new_vocab"])
        self.assertEqual(validation["forward_pass"], "PASS")
        self.assertTrue(validation["logits_finite"])


if __name__ == "__main__":
    unittest.main()
