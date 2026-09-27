from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tokenizer"))

from common import CONTROL_TOKENS, assert_control_tokens_atomic, load_tokenizer


class RealDataArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = json.loads((ROOT / "outputs" / "corpus_report.json").read_text(encoding="utf-8"))
        cls.benchmark = json.loads(
            (ROOT / "outputs" / "tokenizer_real_benchmark.json").read_text(encoding="utf-8")
        )

    def test_final_corpus_has_expected_real_data_scale_and_provenance(self):
        final = self.corpus["final_corpus"]
        self.assertEqual(final["lines"], 332_637)
        self.assertEqual(final["characters"], 92_513_985)
        self.assertFalse(final["synthetic_repetition"])
        self.assertEqual(
            set(final["sources"]),
            {"fineweb", "aihub_document", "aihub_dialogue", "aihub_query", "aihub_summary"},
        )

        txt = ROOT / final["txt"]
        jsonl = ROOT / final["jsonl"]
        if not txt.is_file() and not jsonl.is_file():
            self.skipTest(f"local artifact not in repository: {final['txt']}")
        self.assertTrue(txt.is_file())
        self.assertTrue(jsonl.is_file())
        with jsonl.open(encoding="utf-8") as handle:
            first = json.loads(next(handle))
        self.assertEqual(set(first), {"text", "source"})
        self.assertIn(first["source"], final["sources"])

    def test_validation_exact_train_overlap_was_removed(self):
        overlap = self.corpus["duplicates"]["validation_exact_train_overlap"]
        self.assertEqual(overlap["dialogue"], {"input": 208528, "retained": 206806, "removed": 1722})
        self.assertEqual(overlap["documents"], {"input": 18462, "retained": 10273, "removed": 8189})
        self.assertEqual(overlap["queries"], {"input": 12761, "retained": 9784, "removed": 2977})
        self.assertEqual(overlap["summaries"], {"input": 47773, "retained": 45504, "removed": 2269})

    def test_primary_candidates_have_exact_vocab_and_preserved_tokens(self):
        for frequency in (2, 5):
            for size in (16000, 32000):
                path = ROOT / "outputs" / "real" / f"mf{frequency}" / f"tokenizer-ko-{size}"
                metadata = json.loads((path / "training_metadata.json").read_text(encoding="utf-8"))
                tokenizer = load_tokenizer(str(path))
                self.assertEqual(len(tokenizer), size)
                self.assertEqual(metadata["actual_vocab_size"], size)
                self.assertEqual(metadata["min_frequency"], frequency)
                self.assertEqual(metadata["train_characters"], 92_513_985)
                self.assertEqual(len(metadata["preserved_base_added_token_ids"]), 26)
                self.assertEqual(set(assert_control_tokens_atomic(tokenizer)), set(CONTROL_TOKENS))

    def test_min_frequency_two_and_five_are_identical(self):
        def digest(path: Path) -> str:
            return hashlib.sha256(path.read_bytes()).hexdigest()

        for size in (16000, 32000):
            for filename in ("tokenizer.json", "vocab.json", "merges.txt"):
                mf2 = ROOT / "outputs" / "real" / "mf2" / f"tokenizer-ko-{size}" / filename
                mf5 = ROOT / "outputs" / "real" / "mf5" / f"tokenizer-ko-{size}" / filename
                self.assertEqual(digest(mf2), digest(mf5), f"{size=} {filename=}")

    def test_real_benchmark_uses_correct_independent_eval_and_expected_totals(self):
        tokenizers = self.benchmark["tokenizers"]
        self.assertEqual(tokenizers["Qwen original"]["datasets"]["independent_eval_40"]["total_characters"], 1491)
        self.assertEqual(tokenizers["Qwen original"]["datasets"]["independent_eval_40"]["total_tokens"], 924)
        self.assertEqual(tokenizers["KO-16000-mf2"]["datasets"]["independent_eval_40"]["total_tokens"], 675)
        self.assertEqual(tokenizers["KO-32000-mf2"]["datasets"]["independent_eval_40"]["total_tokens"], 602)
        self.assertEqual(tokenizers["Qwen + Skill"]["datasets"]["skill_dsl"]["total_tokens"], 61)
        self.assertEqual(tokenizers["KO-32000-mf2"]["datasets"]["code_url"]["total_tokens"], 178)


if __name__ == "__main__":
    unittest.main()
