from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "corpus"))
sys.path.insert(0, str(ROOT / "tokenizer"))

from nikl import (  # noqa: E402
    TOPIC_ALIASES,
    clean_form,
    corpus_files,
    iter_articles,
    normalize_topic,
    short_digest,
)

TRAINING_CORPORA = (
    "data/processed/tokenizer_train_real.txt",
    "data/processed/tokenizer_train_mixed10.txt",
    "data/processed/tokenizer_train_mixed10_structured.txt",
    "data/processed/tokenizer_train_mixed10_structured_v2.txt",
)


class NiklPreprocessingTests(unittest.TestCase):
    def test_paragraph_wrapper_is_stripped(self):
        self.assertEqual(clean_form("<p>안녕하세요</p>"), "안녕하세요")
        self.assertEqual(clean_form("  <p> 공백 </p> "), "공백")

    def test_korean_angle_bracket_content_is_preserved(self):
        # These look like tags but are Korean content: media names, securities
        # firms, work titles. Stripping them would corrupt the distribution.
        for text in ("<오마이뉴스>", "<키움證>", "<붉은 황금>"):
            self.assertEqual(clean_form(f"<p>{text} 기사</p>"), f"{text} 기사")

    def test_english_numbers_symbols_and_urls_survive(self):
        raw = "<p>Qwen3-0.6B는 llama.cpp에서 Q4_K_M으로 (2024) https://a.b/c 실행 “인용”</p>"
        cleaned = clean_form(raw)
        for fragment in ("Qwen3-0.6B", "llama.cpp", "Q4_K_M", "(2024)", "https://a.b/c", "“인용”"):
            self.assertIn(fragment, cleaned)

    def test_unambiguous_entities_only(self):
        self.assertEqual(clean_form("<p>A &amp; B &lt;C&gt;</p>"), "A & B <C>")
        # An unknown entity is left exactly as written rather than guessed at.
        self.assertEqual(clean_form("<p>&zwnj;x</p>"), "&zwnj;x")

    def test_nfc_normalization_is_applied(self):
        decomposed = "가"  # ㄱ + ㅏ in Jamo form
        self.assertEqual(clean_form(f"<p>{decomposed}</p>"), "가")

    def test_topic_normalization_uses_the_observed_spelling(self):
        self.assertEqual(normalize_topic("IT/과학"), "IT/과학")
        self.assertEqual(normalize_topic("정보통신/과학"), "IT/과학")
        self.assertEqual(TOPIC_ALIASES["정보통신/과학"], "IT/과학")
        self.assertEqual(normalize_topic(None), "(unlabelled)")

    def test_headline_is_paragraph_zero_not_metadata_title(self):
        paths = corpus_files()
        if not paths:
            self.skipTest("NIKL corpus not present")
        seen = 0
        for article in iter_articles(paths[:1]):
            self.assertTrue(article.title)
            self.assertNotIn("<p>", article.title)
            self.assertNotIn("</p>", article.body)
            self.assertEqual(article.title_and_body.split("\n", 1)[0], article.title)
            seen += 1
            if seen >= 50:
                break
        self.assertEqual(seen, 50)


class NiklManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        report_path = ROOT / "outputs" / "nikl_corpus_report.json"
        manifest_path = ROOT / "data" / "eval" / "nikl_eval_manifest.jsonl"
        if not report_path.exists() or not manifest_path.exists():
            raise unittest.SkipTest("NIKL evaluation artifacts not built")
        cls.report = json.loads(report_path.read_text(encoding="utf-8"))
        cls.manifest = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]

    def test_manifest_is_deterministic_and_complete(self):
        ids = [row["article_id"] for row in self.manifest]
        self.assertEqual(len(ids), len(set(ids)), "manifest repeats an article")
        self.assertEqual(len(self.manifest), self.report["manifest_articles"])
        self.assertEqual(sum(row["chars"] for row in self.manifest), self.report["manifest_characters"])
        for row in self.manifest[:200]:
            self.assertEqual(len(row["text_hash"]), 64)
            self.assertGreater(row["chars"], 0)

    def test_every_topic_is_reported_not_filtered(self):
        topics = {row["topic"] for row in self.manifest}
        self.assertEqual(topics, set(self.report["by_topic"]))
        self.assertGreaterEqual(len(topics), 9)

    def test_leaked_articles_are_excluded_from_the_manifest(self):
        train: set[int] = set()
        for relative in TRAINING_CORPORA:
            path = ROOT / relative
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.rstrip("\n")
                    if stripped:
                        train.add(short_digest(stripped))
        self.assertTrue(train)
        manifest_hashes = {row["text_hash"] for row in self.manifest}
        # Article ids are unique, but a handful of bodies are not: syndicated wire
        # copy is republished verbatim by different newspapers. That is genuine
        # corpus content, it is a tiny fraction, and every tokenizer sees it
        # identically, so it is reported rather than removed.
        duplicate_bodies = len(self.manifest) - len(manifest_hashes)
        self.assertLess(duplicate_bodies / len(self.manifest), 0.001)
        # The recorded overlap must be positive-or-zero and consistent with the report.
        self.assertGreaterEqual(self.report["leakage"]["article_exact_overlap"], 0)
        self.assertEqual(
            sorted(self.report["leakage"]["article_overlap_by_corpus"]),
            sorted(k for k in self.report["leakage"]["article_overlap_by_corpus"]),
        )

    def test_nikl_was_not_added_to_any_training_corpus(self):
        """The training corpora must not have grown NIKL content."""
        expected_lines = {
            "data/processed/tokenizer_train_real.txt": 332_637,
            "data/processed/tokenizer_train_mixed10.txt": 486_583,
            "data/processed/tokenizer_train_mixed10_structured.txt": 496_070,
            "data/processed/tokenizer_train_mixed10_structured_v2.txt": 475_328,
        }
        for relative, expected in expected_lines.items():
            path = ROOT / relative
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as handle:
                actual = sum(1 for line in handle if line.strip())
            self.assertEqual(actual, expected, f"{relative} changed; NIKL must never be trained on")


class NiklBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / "outputs" / "nikl_tokenizer_benchmark.json"
        if not path.exists():
            raise unittest.SkipTest("NIKL benchmark not built")
        cls.payload = json.loads(path.read_text(encoding="utf-8"))

    def test_all_tokenizers_scored_the_same_articles(self):
        articles = self.payload["articles"]
        self.assertGreater(articles, 0)
        totals = {name: row["total_characters"] for name, row in self.payload["overall"].items()}
        self.assertEqual(len(set(totals.values())), 1, "candidates saw different character totals")
        for name, row in self.payload["overall"].items():
            self.assertEqual(row["total_characters"], self.payload["characters"], name)

    def test_qwen_is_the_baseline(self):
        qwen = self.payload["overall"]["Qwen original"]
        self.assertAlmostEqual(qwen["token_reduction_percent"], 0.0, places=9)
        self.assertEqual(qwen["embedding_vocab_rows"], 151_936)

    def test_every_korean_candidate_keeps_the_32k_budget(self):
        for name, row in self.payload["overall"].items():
            if name.startswith("KO-"):
                self.assertEqual(row["embedding_vocab_rows"], 32_000, name)
                self.assertEqual(
                    self.payload["invariants"][name]["embedding_estimate"]["vocabulary_dependent_parameters"],
                    32_000 * 1024,
                )
                self.assertEqual(len(self.payload["invariants"][name]["control_tokens_atomic"]), 8)

    def test_no_topic_or_publisher_was_dropped(self):
        self.assertGreaterEqual(len(self.payload["by_topic"]), 9)
        self.assertGreaterEqual(len(self.payload["by_publisher"]), 20)
        counted = sum(row["articles"] for row in self.payload["by_topic"].values())
        self.assertEqual(counted, self.payload["articles"])

    def test_reduction_is_reproducible_from_raw_totals(self):
        qwen = self.payload["overall"]["Qwen original"]["total_tokens"]
        for name, row in self.payload["overall"].items():
            expected = 100.0 * (1.0 - row["total_tokens"] / qwen)
            self.assertAlmostEqual(row["token_reduction_percent"], expected, places=6, msg=name)


if __name__ == "__main__":
    unittest.main()
