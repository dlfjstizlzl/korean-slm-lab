from __future__ import annotations

import sys
import tempfile
import unittest
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tokenizer"))

from common import CONTROL_TOKENS, assert_control_tokens_atomic, normalize_text, read_corpus, token_metrics


class FakeTokenizer:
    unk_token_id = None

    def __len__(self):
        return 100

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        if text in CONTROL_TOKENS:
            return [CONTROL_TOKENS.index(text)]
        return list(range(len(text)))

    def convert_ids_to_tokens(self, ids):
        return [f"t{value}" for value in ids]


class CommonTests(unittest.TestCase):
    def test_normalize_text_is_nfc_and_strips(self):
        decomposed = "  가  "
        self.assertEqual(normalize_text(decomposed), "가")
        self.assertTrue(unicodedata.is_normalized("NFC", normalize_text(decomposed)))

    def test_corpus_removes_blank_comments_and_optionally_deduplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.txt"
            path.write_text("\n# note\n가\n가\nskill_dsl\t<검색>테스트\n", encoding="utf-8")
            records = read_corpus(path, deduplicate=True)
        self.assertEqual([record.text for record in records], ["가", "<검색>테스트"])
        self.assertEqual(records[1].category, "skill_dsl")

    def test_control_tokens_are_atomic(self):
        ids = assert_control_tokens_atomic(FakeTokenizer())
        self.assertEqual(set(ids), set(CONTROL_TOKENS))

    def test_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.txt"
            path.write_text("가나다\n라마\n", encoding="utf-8")
            records = read_corpus(path)
        metrics = token_metrics(FakeTokenizer(), records)
        self.assertEqual(metrics["total_sentences"], 2)
        self.assertEqual(metrics["total_characters"], 5)
        self.assertEqual(metrics["total_tokens"], 5)
        self.assertEqual(metrics["p95_tokens_per_sentence"], 3)


if __name__ == "__main__":
    unittest.main()
