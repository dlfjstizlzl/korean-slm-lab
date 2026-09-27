from __future__ import annotations

import json
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "corpus"))
sys.path.insert(0, str(ROOT / "tokenizer"))

from build_recovery_data import (  # noqa: E402
    ForbiddenSourceError,
    assert_record_allowed,
    assert_source_allowed,
)
from build_stage2_corpus import LIMITS, TARGET_SHARES  # noqa: E402
from common import CONTROL_TOKENS  # noqa: E402
from stage2_dedup import (  # noqa: E402
    content_hash,
    ngram_duplicate_ratio,
    normalize,
    quality_verdict,
    simhash,
)
from stage2_pack_simulation import parse_args as pack_args  # noqa: E402,F401
from stage2_skill_synth import generate as generate_skill, skeleton  # noqa: E402
from stage2_substring_screen import normalize as screen_normalize  # noqa: E402

TRAIN = ROOT / "data" / "processed" / "stage2_train.jsonl"
HOLDOUT = ROOT / "data" / "processed" / "stage2_holdout.jsonl"
MANIFEST = ROOT / "data" / "processed" / "stage2_corpus_manifest.jsonl"
# stage2_corpus_report.json is the consolidated cross-stage report; the
# per-build report (supply, binding category, size verdict) is the _v2 file.
CORPUS_REPORT = ROOT / "outputs" / "stage2_corpus_report.json"
BUILD_REPORT = ROOT / "outputs" / "stage2_corpus_report_v2.json"
TOKEN_REPORT = ROOT / "outputs" / "stage2_token_stats.json"
PACK_REPORT = ROOT / "outputs" / "stage2_packing.json"
SCREEN_REPORT = ROOT / "outputs" / "stage2_substring_screen.json"
PACK_MANIFEST = ROOT / "data" / "processed" / "stage2_1024_manifest.jsonl"


# --------------------------------------------------------------------------- #
# NIKL exclusion
# --------------------------------------------------------------------------- #

class NiklExclusionTests(unittest.TestCase):
    def test_source_names_are_rejected(self):
        for source in ("nikl", "NIKL_NEWSPAPER_2025_v1.1", "newspaper_corpus", "NWRW_set"):
            with self.assertRaises(ForbiddenSourceError, msg=source):
                assert_source_allowed(source)

    def test_paths_are_rejected_even_with_clean_source_names(self):
        for path in ("data/NIKL_NEWSPAPER_2025_v1.1/x.json",
                     "data/eval/nikl_eval_manifest.jsonl"):
            with self.assertRaises(ForbiddenSourceError, msg=path):
                assert_source_allowed("korean", path)

    def test_article_ids_are_rejected_in_record_text(self):
        for marker in ("NIRW2500000001.1", "NWRW1900000002"):
            with self.assertRaises(ForbiddenSourceError, msg=marker):
                assert_record_allowed(f"기사 {marker} 내용", "fineweb")

    def test_stage2_readers_never_open_a_nikl_file(self):
        """No string the builder could use as a data path may name a NIKL artefact.

        Prose describing the guard is fine; a usable path is not. So this checks
        every path-shaped string literal in the module, not the file's raw text.
        """
        import ast

        tree = ast.parse((ROOT / "corpus" / "build_stage2_corpus.py").read_text(encoding="utf-8"))
        literals = [
            node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ]
        path_like = [
            value for value in literals
            if " " not in value.strip()
            and ("/" in value or value.endswith((".txt", ".jsonl", ".json", ".csv")))
        ]
        self.assertTrue(path_like, "expected the builder to reference data paths")
        for value in path_like:
            lowered = value.lower()
            for marker in ("nikl", "nirw", "nwrw", "newspaper"):
                self.assertNotIn(marker, lowered, f"{marker!r} in path literal {value!r}")

    def test_the_nikl_directory_is_never_a_readable_source(self):
        nikl = ROOT / "data" / "NIKL_NEWSPAPER_2025_v1.1"
        if nikl.exists():
            with self.assertRaises(ForbiddenSourceError):
                assert_source_allowed("korean_general", nikl)


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #

class NormalizationTests(unittest.TestCase):
    def test_control_characters_and_nul_are_stripped(self):
        self.assertEqual(normalize("가\x00나\x07"), "가나")

    def test_runaway_whitespace_is_collapsed_but_text_is_kept(self):
        self.assertEqual(normalize("a\n\n\n\n\nb"), "a\n\nb")
        self.assertEqual(normalize("a     b"), "a  b")

    def test_language_material_is_not_stripped(self):
        text = "메모리(memory)는 https://example.com/x?y=1 에서 100% 확인된다."
        self.assertEqual(normalize(text), text)

    def test_content_hash_ignores_case_and_whitespace(self):
        self.assertEqual(content_hash("Hello   World"), content_hash("hello world"))
        self.assertNotEqual(content_hash("hello world"), content_hash("hello worlds"))


class QualityFilterTests(unittest.TestCase):
    def test_short_and_long_documents_are_rejected(self):
        self.assertEqual(quality_verdict("짧다", min_chars=10, max_chars=100), "too_short")
        self.assertEqual(quality_verdict("가" * 200, min_chars=10, max_chars=100), "too_long")

    def test_encoding_corruption_is_rejected(self):
        self.assertEqual(
            quality_verdict("정상 텍스트 � 입니다", min_chars=5, max_chars=1000),
            "encoding_corruption",
        )

    def test_minified_blob_is_rejected_by_the_unbroken_run_rule(self):
        self.assertEqual(
            quality_verdict("x" * 900, min_chars=10, max_chars=5000, max_token_length=400),
            "unbroken_run",
        )

    def test_compact_json_survives_a_higher_run_ceiling(self):
        payload = json.dumps({"k": list(range(200))})
        self.assertIsNone(quality_verdict(payload, min_chars=10, max_chars=50_000,
                                          is_code_like=True, max_token_length=8_000))

    def test_repeated_line_documents_are_rejected(self):
        self.assertEqual(
            quality_verdict("같은 줄\n" * 20, min_chars=10, max_chars=5000),
            "repeated_lines",
        )

    def test_code_is_not_penalized_for_being_symbol_heavy(self):
        dense = "a!@#$%^&*()" * 30  # >50% punctuation, but still has word characters
        self.assertEqual(quality_verdict(dense, min_chars=10, max_chars=5000), "symbol_heavy")
        self.assertIsNone(quality_verdict(dense, min_chars=10, max_chars=5000, is_code_like=True))


class RepetitionMetricTests(unittest.TestCase):
    def test_duplicate_ngram_ratio(self):
        self.assertAlmostEqual(ngram_duplicate_ratio("a b c d e", 2), 0.0)
        self.assertGreater(ngram_duplicate_ratio("a b a b a b a b", 2), 0.5)

    def test_simhash_is_deterministic_and_similarity_sensitive(self):
        self.assertEqual(simhash("가나다라마바사"), simhash("가나다라마바사"))
        self.assertNotEqual(simhash("완전히 다른 문장"), simhash("가나다라마바사"))


class SkillSynthesisTests(unittest.TestCase):
    def setUp(self):
        self.records = generate_skill(["가상 메모리 페이지 교체", "통합 메모리 대역폭",
                                       "트랜잭션 격리 수준"], 120_000, seed=42)

    def test_generation_is_deterministic_for_a_fixed_seed(self):
        again = generate_skill(["가상 메모리 페이지 교체", "통합 메모리 대역폭",
                                "트랜잭션 격리 수준"], 120_000, seed=42)
        self.assertEqual([r["text"] for r in self.records], [r["text"] for r in again])

    def test_multiple_plan_shapes_are_produced(self):
        shapes = {record["shape"] for record in self.records}
        self.assertGreaterEqual(len(shapes), 4)
        self.assertIn("failure_retry", shapes)

    def test_no_single_skeleton_dominates_the_output(self):
        counts = Counter(skeleton(record["text"]) for record in self.records)
        self.assertGreater(len(counts), 20)
        self.assertLess(counts.most_common(1)[0][1] / len(self.records), 0.35)

    def test_control_tokens_actually_appear(self):
        blob = "\n".join(record["text"] for record in self.records)
        for token in ("<검색>", "<요약>", "<SQL>", "<코드>", "<종료>"):
            self.assertIn(token, blob)


class ScreenTests(unittest.TestCase):
    def test_screen_normalization_matches_probe_construction(self):
        self.assertEqual(screen_normalize("Hello   World\n"), "hello world")


# --------------------------------------------------------------------------- #
# Built corpus
# --------------------------------------------------------------------------- #

@unittest.skipUnless(TRAIN.exists() and CORPUS_REPORT.exists(), "Stage-2 corpus not built")
class CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = [json.loads(line) for line in TRAIN.open(encoding="utf-8") if line.strip()]
        cls.report = json.loads(CORPUS_REPORT.read_text(encoding="utf-8"))

    def test_every_record_carries_full_provenance(self):
        for record in self.records[:5000]:
            for field in ("source", "source_id", "category", "text", "chars", "sha256"):
                self.assertIn(field, record)
            self.assertEqual(record["chars"], len(record["text"]))
            self.assertEqual(len(record["sha256"]), 64)

    def test_character_accounting_matches_the_records(self):
        total = sum(record["chars"] for record in self.records)
        self.assertEqual(total, json.loads(
            (ROOT / "outputs" / "stage2_substring_screen.json").read_text(encoding="utf-8")
        )["characters_after"])

    def test_category_accounting_covers_exactly_the_planned_categories(self):
        self.assertEqual({record["category"] for record in self.records}, set(TARGET_SHARES))

    def test_category_shares_track_the_target_mixture(self):
        total = sum(record["chars"] for record in self.records)
        by_category = Counter()
        for record in self.records:
            by_category[record["category"]] += record["chars"]
        for category, target in TARGET_SHARES.items():
            self.assertAlmostEqual(by_category[category] / total, target, delta=0.01,
                                   msg=category)

    def test_no_exact_duplicate_documents_survive(self):
        hashes = [content_hash(record["text"]) for record in self.records]
        self.assertEqual(len(hashes), len(set(hashes)))

    def test_no_document_is_artificially_repeated(self):
        texts = Counter(record["sha256"] for record in self.records)
        self.assertEqual(texts.most_common(1)[0][1], 1)

    def test_documents_respect_their_category_length_bounds(self):
        for record in self.records[:20000]:
            low, high, _, _ = LIMITS[record["category"]]
            self.assertGreaterEqual(record["chars"], low, record["category"])
            self.assertLessEqual(record["chars"], high, record["category"])

    def test_no_nikl_material_reached_the_corpus(self):
        for record in self.records[:20000]:
            assert_source_allowed(record["source"])
            assert_record_allowed(record["text"], record["source"])
        self.assertFalse(self.report["nikl"]["used"])
        self.assertEqual(self.report["nikl"]["files_opened"], 0)

    def test_train_and_holdout_are_disjoint_at_document_level(self):
        train_ids = {record["sha256"] for record in self.records}
        holdout = [json.loads(line) for line in HOLDOUT.open(encoding="utf-8") if line.strip()]
        holdout_ids = {record["sha256"] for record in holdout}
        self.assertTrue(holdout_ids)
        self.assertEqual(train_ids & holdout_ids, set())

    def test_manifest_covers_every_document(self):
        manifest = [json.loads(line) for line in MANIFEST.open(encoding="utf-8") if line.strip()]
        splits = Counter(row["split"] for row in manifest)
        self.assertGreater(splits["train"], 0)
        self.assertGreater(splits["holdout"], 0)
        for row in manifest[:2000]:
            self.assertIn(row["category"], TARGET_SHARES)
            self.assertEqual(len(row["sha256"]), 64)


@unittest.skipUnless(TRAIN.exists(), "Stage-2 corpus not built")
class EvaluationProtectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hashes = {content_hash(json.loads(line)["text"])
                      for line in TRAIN.open(encoding="utf-8") if line.strip()}

    def _assert_no_overlap(self, path: Path, kind: str):
        if not path.exists():
            self.skipTest(f"{path} missing")
        overlap = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                text = json.loads(line)["text"] if kind == "jsonl" else line.rstrip("\n")
                if text and content_hash(text) in self.hashes:
                    overlap += 1
        self.assertEqual(overlap, 0, f"{path.name} overlaps the Stage-2 training corpus")

    def test_no_balanced_eval_overlap(self):
        self._assert_no_overlap(ROOT / "data/eval/recovery_balanced_eval.jsonl", "jsonl")

    def test_no_large_korean_eval_overlap(self):
        self._assert_no_overlap(ROOT / "data/eval/recovery_korean_large_eval.jsonl", "jsonl")

    def test_no_fineweb_holdout_overlap(self):
        self._assert_no_overlap(ROOT / "data/processed/fineweb_general_holdout.txt", "txt")

    def test_no_aihub_validation_overlap(self):
        for name in ("aihub_documents_valid_clean.txt", "aihub_dialogue_valid_clean.txt",
                     "aihub_summaries_valid_clean.txt"):
            self._assert_no_overlap(ROOT / "data/processed" / name, "txt")

    def test_no_structured_validation_overlap(self):
        for name in ("structured/structured_valid.jsonl", "structured/json_extra_valid.jsonl",
                     "mixed/mixed_validation.jsonl"):
            self._assert_no_overlap(ROOT / "data/processed" / name, "jsonl")

    @unittest.skipUnless(SCREEN_REPORT.exists(), "substring screen not run")
    def test_substring_contamination_was_screened_and_boilerplate_spared(self):
        report = json.loads(SCREEN_REPORT.read_text(encoding="utf-8"))
        self.assertEqual(report["residual_substring_overlap"], 0)
        self.assertGreater(report["documents_removed"], 0)
        # Shared web boilerplate must not be mistaken for contamination.
        self.assertGreater(sum(report["probes_ignored_as_boilerplate"].values()), 0)


@unittest.skipUnless(TOKEN_REPORT.exists(), "token stats not generated")
class TokenAccountingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads(TOKEN_REPORT.read_text(encoding="utf-8"))

    def test_totals_are_internally_consistent(self):
        totals = self.report["totals"]
        self.assertEqual(totals["tokens"],
                         sum(v["tokens"] for v in self.report["by_category"].values()))
        self.assertEqual(totals["characters"],
                         sum(v["characters"] for v in self.report["by_category"].values()))
        self.assertAlmostEqual(totals["tokens_per_character"],
                               totals["tokens"] / totals["characters"], places=9)

    def test_character_share_and_token_share_are_reported_separately(self):
        for category, values in self.report["by_category"].items():
            self.assertIn("char_share", values)
            self.assertIn("token_share", values)
            self.assertAlmostEqual(
                values["share_drift_pp"],
                (values["token_share"] - values["char_share"]) * 100, places=6, msg=category)

    def test_vocabulary_is_almost_entirely_exercised(self):
        coverage = self.report["vocabulary_coverage"]
        self.assertEqual(coverage["used_tokens"] + coverage["unused_tokens"],
                         self.report["vocab_size"])
        self.assertGreater(coverage["used_share"], 0.95)

    def test_every_skill_control_token_appears(self):
        counts = self.report["vocabulary_coverage"]["control_tokens"]
        self.assertEqual(set(counts), set(CONTROL_TOKENS))
        for token, entry in counts.items():
            self.assertGreater(entry["count"], 0, token)

    def test_expected_epoch_calculation(self):
        budget = self.report["training_budget"]
        self.assertAlmostEqual(budget["epochs_required"],
                               budget["target_tokens"] / budget["corpus_tokens"], places=9)
        self.assertEqual(budget["within_one_epoch"],
                         budget["corpus_tokens"] >= budget["target_tokens"])

    def test_repeated_token_fraction_matches_the_shortfall(self):
        budget = self.report["training_budget"]
        expected = max(0.0, (budget["target_tokens"] - budget["corpus_tokens"])
                       / budget["target_tokens"])
        self.assertAlmostEqual(budget["repeated_token_fraction"], expected, places=9)


@unittest.skipUnless(PACK_REPORT.exists(), "packing simulation not run")
class PackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads(PACK_REPORT.read_text(encoding="utf-8"))

    def test_blocks_are_exactly_the_sequence_length(self):
        report = self.report
        self.assertEqual(report["sequence_length"], 1024)
        self.assertEqual(report["packed_tokens"], report["full_blocks"] * 1024)

    def test_tail_is_dropped_not_padded(self):
        report = self.report
        self.assertLess(report["dropped_tail_tokens"], 1024)
        self.assertEqual(report["stream_tokens"],
                         report["packed_tokens"] + report["dropped_tail_tokens"])

    def test_documents_are_separated_by_an_explicit_boundary_token(self):
        self.assertIn("endoftext", self.report["separator"])
        self.assertEqual(self.report["separator_tokens_per_document"], 1)

    def test_packing_efficiency_is_near_perfect(self):
        self.assertGreater(self.report["packing_efficiency"], 0.99)

    def test_boundary_crossings_are_counted(self):
        self.assertGreater(self.report["document_boundary_crossings"], 0)
        self.assertGreater(self.report["average_documents_per_block"], 1.0)

    @unittest.skipUnless(PACK_MANIFEST.exists(), "packing manifest missing")
    def test_packing_manifest_is_deterministic_and_complete(self):
        rows = [json.loads(line) for line in PACK_MANIFEST.open(encoding="utf-8") if line.strip()]
        self.assertEqual(len(rows), self.report["documents"])
        self.assertEqual(len({row["document_index"] for row in rows}), len(rows))
        blocks = [row["start_block"] for row in rows]
        self.assertEqual(blocks, sorted(blocks))


@unittest.skipUnless(BUILD_REPORT.exists(), "build report missing")
class CorpusSizeVerdictTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads(BUILD_REPORT.read_text(encoding="utf-8"))

    def test_minimum_size_requirement_is_evaluated_and_reported_honestly(self):
        report = self.report
        self.assertEqual(report["minimum_required_chars"], 150_000_000)
        self.assertEqual(
            report["meets_minimum"],
            report["totals"]["train_characters"] >= report["minimum_required_chars"],
        )
        if not report["meets_minimum"]:
            self.assertGreater(report["shortfall_chars"], 0)

    def test_binding_category_actually_binds(self):
        report = self.report
        binding = report["binding_category"]
        caps = {
            category: report["supply"][category] / report["target_shares"][category]
            for category in report["target_shares"]
            if category != "skill_manager" and report["supply"].get(category)
        }
        self.assertEqual(binding, min(caps, key=caps.get))

    def test_no_training_was_performed(self):
        self.assertFalse(self.report["training_performed"])


if __name__ == "__main__":
    unittest.main()
