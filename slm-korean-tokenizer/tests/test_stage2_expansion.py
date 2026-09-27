from __future__ import annotations

import json
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "corpus"))
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "tools"))

from build_recovery_data import ForbiddenSourceError, assert_record_allowed, assert_source_allowed  # noqa: E402
from build_stage2_corpus import FALLBACK_SHARES, TARGET_SHARES  # noqa: E402
from decontaminate_eval import normalize as decon_normalize, shingles  # noqa: E402
from fetch_korean_technical import is_technical_prose, korean_ratio, bullet_ratio  # noqa: E402
from recovery_saturation import ABSOLUTE_THRESHOLD, RELATIVE_THRESHOLD, classify, interval_rates  # noqa: E402
from stage2_dedup import content_hash  # noqa: E402

DECON_BALANCED = ROOT / "data/eval/recovery_balanced_eval_decontaminated.jsonl"
DECON_KOREAN = ROOT / "data/eval/recovery_korean_large_eval_decontaminated.jsonl"
ORIG_BALANCED = ROOT / "data/eval/recovery_balanced_eval.jsonl"
ORIG_KOREAN = ROOT / "data/eval/recovery_korean_large_eval.jsonl"
DECON_REPORT = ROOT / "outputs/eval_decontamination.json"
REEVAL = ROOT / "outputs/decontaminated_reevaluation.json"
DECON_SAT = ROOT / "outputs/decontaminated_saturation.json"
V2_TRAIN = ROOT / "data/processed/stage2_train_v2.jsonl"
V2_REPORT = ROOT / "outputs/stage2_corpus_report_v2.json"
V2_TOKENS = ROOT / "outputs/stage2_token_stats_v2.json"
V2_PACK = ROOT / "outputs/stage2_packing_v2.json"
V2_SCREEN = ROOT / "outputs/stage2_substring_screen_v2.json"
V1_TRAIN = ROOT / "data/processed/stage2_train.jsonl"
ACQUISITIONS = {
    "fineweb": ROOT / "outputs/fineweb_extra_acquisition.json",
    "technical": ROOT / "outputs/korean_technical_acquisition.json",
    "structured": ROOT / "outputs/structured_extra_acquisition.json",
}
RAW = {
    "fineweb": ROOT / "data/raw/fineweb_ko_extra.jsonl",
    "technical": ROOT / "data/raw/korean_technical_wiki.jsonl",
    "structured": ROOT / "data/raw/structured_extra.jsonl",
}


# --------------------------------------------------------------------------- #
# Decontamination
# --------------------------------------------------------------------------- #

class DecontaminationLogicTests(unittest.TestCase):
    def test_shingles_cover_the_text_at_the_given_stride(self):
        pieces = list(shingles("abcdefghij", 4, 3))
        self.assertEqual(pieces[0], "abcd")
        self.assertTrue(all(len(p) == 4 for p in pieces))

    def test_shingles_are_empty_for_text_shorter_than_the_window(self):
        self.assertEqual(list(shingles("abc", 10, 1)), [])

    def test_normalization_matches_the_hashing_convention(self):
        self.assertEqual(decon_normalize("Hello   World\n"), "hello world")


@unittest.skipUnless(DECON_BALANCED.exists(), "decontaminated evals not built")
class DecontaminatedEvalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads(DECON_REPORT.read_text(encoding="utf-8"))
        cls.balanced = [json.loads(l) for l in DECON_BALANCED.open(encoding="utf-8") if l.strip()]
        cls.korean = [json.loads(l) for l in DECON_KOREAN.open(encoding="utf-8") if l.strip()]

    def test_originals_are_preserved(self):
        self.assertTrue(ORIG_BALANCED.exists())
        self.assertTrue(ORIG_KOREAN.exists())
        self.assertTrue(self.report["originals_preserved"])

    def test_decontaminated_sets_are_strict_subsets(self):
        for decon, original in ((self.balanced, ORIG_BALANCED), (self.korean, ORIG_KOREAN)):
            originals = {json.loads(l)["id"] for l in original.open(encoding="utf-8") if l.strip()}
            ids = {record["id"] for record in decon}
            self.assertTrue(ids < originals)

    def test_contaminated_samples_are_actually_gone(self):
        train = [json.loads(l)["text"] for l in
                 (ROOT / "data/processed/recovery_train.jsonl").open(encoding="utf-8") if l.strip()]
        blob = "\n".join(decon_normalize(t) for t in train)
        for records in (self.balanced, self.korean):
            for record in records[:1500]:
                text = decon_normalize(record["text"])
                if len(text) >= 100:
                    self.assertNotIn(text, blob, record["id"])

    def test_boilerplate_was_spared_rather_than_treated_as_contamination(self):
        self.assertGreater(self.report["shingles_ignored_as_boilerplate"], 0)

    def test_removal_is_concentrated_in_korean_technical(self):
        for values in self.report["by_file"].values():
            rates = values["category_removal_rate"]
            self.assertGreater(rates["korean_technical"], rates.get("korean_general", 0.0))

    def test_no_exact_contamination_kind_was_found(self):
        # If exact equality had caught this, the original leakage check would have.
        for values in self.report["by_file"].values():
            self.assertNotIn("A_exact", values["contamination_kinds"])


@unittest.skipUnless(REEVAL.exists(), "re-evaluation not run")
class ReevaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads(REEVAL.read_text(encoding="utf-8"))

    def test_no_training_was_performed(self):
        self.assertFalse(self.report["training_performed"])

    def test_every_checkpoint_was_scored_on_the_decontaminated_set(self):
        self.assertIn("decontaminated", self.report["balanced_eval"])
        self.assertGreaterEqual(len(self.report["results"]), 9)

    def test_initialization_ranking_survives_decontamination(self):
        self.assertEqual(self.report["conclusions"]["initialization_ranking"][0], "transplant_mean")
        self.assertTrue(self.report["conclusions"]["mean_still_best_at_init"])

    def test_phase1_ranking_survives_decontamination(self):
        self.assertTrue(self.report["conclusions"]["mean_still_best_after_phase1"])

    def test_all_rows_still_beats_decomposed_only(self):
        self.assertTrue(self.report["conclusions"]["all_rows_still_beats_decomposed_only"])
        comparison = self.report["bootstrap"]["decomposed_only_minus_all_rows"]
        self.assertGreater(comparison["delta_nll_per_character"], 0.2)
        self.assertTrue(comparison["significant"])

    def test_recovery_curve_is_monotonic(self):
        values = [point["nll_per_character"] for point in self.report["recovery_curve"]]
        self.assertEqual(values, sorted(values, reverse=True))


@unittest.skipUnless(DECON_SAT.exists(), "decontaminated saturation not computed")
class DecontaminatedSaturationTests(unittest.TestCase):
    def test_thresholds_were_not_changed_to_fit_the_new_numbers(self):
        report = json.loads(DECON_SAT.read_text(encoding="utf-8"))
        self.assertEqual(report["thresholds_unchanged"],
                         {"relative": RELATIVE_THRESHOLD, "absolute": ABSOLUTE_THRESHOLD})
        self.assertEqual(RELATIVE_THRESHOLD, 0.25)
        self.assertEqual(ABSOLUTE_THRESHOLD, 0.03)

    def test_saturation_still_holds(self):
        report = json.loads(DECON_SAT.read_text(encoding="utf-8"))
        self.assertTrue(report["saturation"]["saturating"])

    def test_verdict_is_reproducible_from_the_recorded_points(self):
        report = json.loads(DECON_SAT.read_text(encoding="utf-8"))
        points = [(p["tokens"], p["nll_per_character"]) for p in report["points"]]
        self.assertEqual(classify(interval_rates(points))["saturating"],
                         report["saturation"]["saturating"])


# --------------------------------------------------------------------------- #
# Acquisition
# --------------------------------------------------------------------------- #

class TechnicalSelectionTests(unittest.TestCase):
    def test_korean_ratio_and_bullet_ratio(self):
        self.assertGreater(korean_ratio("한국어 문장입니다"), 0.7)
        self.assertLess(korean_ratio("pure ascii text"), 0.05)
        self.assertGreater(bullet_ratio("* a\n* b\n* c"), 0.9)
        self.assertEqual(bullet_ratio("문장 하나.\n문장 둘."), 0.0)

    def test_list_articles_are_rejected(self):
        text = "\n".join(f"* 항목 {i} 기술" for i in range(40))
        self.assertEqual(
            is_technical_prose("목록", text, min_chars=100, min_korean=0.2,
                               max_bullets=0.30, min_distinct=1, min_density=0.1),
            "list_article")

    def test_non_technical_prose_is_rejected(self):
        text = "그는 아침에 일어나 산책을 하였다. " * 40
        self.assertEqual(
            is_technical_prose("일상", text, min_chars=100, min_korean=0.3,
                               max_bullets=0.30, min_distinct=4, min_density=2.0),
            "not_technical_topic")

    def test_technical_article_is_accepted(self):
        text = ("전자 회로의 전압과 전류는 물리 법칙을 따른다. 반도체 소자는 전자 기술의 핵심이다. "
                "이 방정식은 주파수와 전압의 관계를 나타내는 함수이다. ") * 6
        self.assertIsNone(
            is_technical_prose("전자 회로", text, min_chars=100, min_korean=0.3,
                               max_bullets=0.30, min_distinct=4, min_density=2.0))

    def test_non_korean_text_is_rejected(self):
        text = "The transistor is a semiconductor device used to amplify signals. " * 10
        self.assertEqual(
            is_technical_prose("Transistor", text, min_chars=100, min_korean=0.33,
                               max_bullets=0.30, min_distinct=4, min_density=2.0),
            "not_korean_enough")


@unittest.skipUnless(all(p.exists() for p in ACQUISITIONS.values()), "acquisitions not run")
class AcquisitionProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reports = {name: json.loads(path.read_text(encoding="utf-8"))
                       for name, path in ACQUISITIONS.items()}

    def test_every_acquisition_records_dataset_provider_access_and_license(self):
        for name, report in self.reports.items():
            for field in ("dataset", "provider", "access", "license"):
                self.assertIn(field, report, f"{name} missing {field}")
                self.assertTrue(str(report[field]).strip(), f"{name} empty {field}")

    def test_license_metadata_is_specific_rather_than_vague(self):
        for name, report in self.reports.items():
            license_text = report["license"].lower()
            self.assertFalse(
                license_text in ("unknown", "?", "n/a", "collected from the internet"),
                f"{name} has a vague license",
            )
            self.assertGreater(len(license_text), 10, name)

    def test_fineweb_used_a_later_shard_than_the_original_slice(self):
        report = self.reports["fineweb"]
        self.assertGreaterEqual(report["start_shard_index"], 1)
        self.assertNotIn("kor_Hang_removed", " ".join(report["shards_used"]))

    def test_fineweb_documents_are_new_rather_than_the_old_slice_again(self):
        report = self.reports["fineweb"]
        self.assertGreater(report["accepted_chars"], 20_000_000)
        # Near-zero duplicates is the evidence that a fresh shard was used.
        self.assertLess(report["rejected_duplicate"] / max(report["accepted"], 1), 0.01)

    def test_technical_source_is_natural_text_not_llm_generated(self):
        report = self.reports["technical"]
        self.assertIn("wikipedia", report["dataset"].lower())
        self.assertIn("min_technical_density_per_1000_chars", report["selection"])

    def test_technical_acquisition_deduplicated_against_existing_corpus(self):
        self.assertEqual(self.reports["technical"]["rejected_duplicate"], 0)

    def test_every_raw_record_carries_provenance(self):
        for name, path in RAW.items():
            with path.open(encoding="utf-8") as handle:
                for index, line in enumerate(handle):
                    if index >= 200:
                        break
                    record = json.loads(line)
                    for field in ("source", "source_id", "dataset", "license", "chars", "sha256"):
                        self.assertIn(field, record, f"{name} record missing {field}")
                    assert_source_allowed(record["source"])
                    assert_record_allowed(record["text"], record["source"])

    def test_gated_sources_were_not_bypassed(self):
        access = self.reports["structured"]["access"].lower()
        self.assertIn("not bypassed", access)


# --------------------------------------------------------------------------- #
# Rebuilt corpus
# --------------------------------------------------------------------------- #

@unittest.skipUnless(V2_REPORT.exists(), "v2 corpus not built")
class V2CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads(V2_REPORT.read_text(encoding="utf-8"))
        cls.tokens = json.loads(V2_TOKENS.read_text(encoding="utf-8"))
        cls.pack = json.loads(V2_PACK.read_text(encoding="utf-8"))
        cls.screen = json.loads(V2_SCREEN.read_text(encoding="utf-8"))

    def test_v1_corpus_was_preserved(self):
        if not V2_TRAIN.exists():
            self.skipTest("local artifact not in repository: data/processed/stage2_train_v2.jsonl")
        self.assertTrue(V1_TRAIN.exists())
        self.assertNotEqual(V1_TRAIN.resolve(), V2_TRAIN.resolve())

    def test_primary_mixture_was_used(self):
        self.assertEqual(self.report["mixture"], "primary")
        self.assertEqual(self.report["target_shares"], TARGET_SHARES)
        self.assertNotEqual(TARGET_SHARES, FALLBACK_SHARES)

    def test_fallback_mixture_is_defined_but_unused(self):
        self.assertEqual(FALLBACK_SHARES["korean_general"], 0.50)
        self.assertEqual(FALLBACK_SHARES["korean_technical"], 0.25)
        self.assertAlmostEqual(sum(FALLBACK_SHARES.values()), 1.0, places=9)

    def test_exit_condition_a_minimum_characters(self):
        self.assertGreaterEqual(self.screen["characters_after"], 150_000_000)

    def test_exit_condition_b_packed_tokens(self):
        self.assertGreaterEqual(self.pack["packed_tokens"], 60_000_000)

    def test_exit_condition_c_one_epoch(self):
        self.assertLessEqual(self.pack["budget"]["epochs_required"], 1.0)
        self.assertTrue(self.pack["budget"]["within_one_epoch"])
        self.assertEqual(self.tokens["training_budget"]["repeated_token_fraction"], 0.0)

    def test_exit_condition_d_general_plus_technical_token_share(self):
        shares = self.tokens["by_category"]
        combined = shares["korean_general"]["token_share"] + shares["korean_technical"]["token_share"]
        self.assertGreaterEqual(combined, 0.74)

    def test_exit_condition_e_korean_token_share(self):
        shares = self.tokens["by_category"]
        korean = sum(shares[c]["token_share"]
                     for c in ("korean_general", "korean_technical", "dialogue"))
        self.assertGreaterEqual(korean, 0.83)

    def test_exit_condition_f_structured_share_is_about_fifteen_percent(self):
        shares = self.tokens["by_category"]
        structured = sum(shares[c]["token_share"]
                         for c in ("code_url", "json_structured", "sql", "skill_manager"))
        self.assertGreater(structured, 0.13)
        self.assertLess(structured, 0.17)

    def test_exit_condition_g_zero_contamination(self):
        self.assertEqual(self.screen["residual_substring_overlap"], 0)

    def test_exit_condition_h_nikl_never_used(self):
        self.assertFalse(self.report["nikl"]["used"])
        self.assertEqual(self.report["nikl"]["files_opened"], 0)

    def test_exit_condition_i_no_artificial_repetition(self):
        counts = Counter()
        with V2_TRAIN.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    counts[json.loads(line)["sha256"]] += 1
        self.assertEqual(counts.most_common(1)[0][1], 1)

    def test_exit_condition_j_skill_repetition_is_bounded(self):
        """Exit condition J, measured rather than assumed.

        The v2 skill set is regenerated to hold ~3% of a larger corpus, so it is
        not the same document set as v1. Measured full-population duplicate
        4-gram rose 0.2318 -> 0.2356 (+0.0038), a marginal worsening that the
        report states explicitly rather than rounding away. This test pins the
        band so a real regression is caught.
        """
        skill = self.report["repetition"]["skill_manager"]
        self.assertLess(skill["mean_duplicate_4gram"], 0.25)
        self.assertGreater(skill["mean_duplicate_4gram"], 0.20)

    def test_skill_structural_diversity_improved(self):
        synthesis = self.report["skill_synthesis"]
        # v1 had 89 distinct skeletons; regeneration must not reduce variety.
        self.assertGreaterEqual(synthesis["distinct_skeletons"], 89)
        self.assertLess(synthesis["top_skeleton_share"], 0.25)

    def test_skill_share_was_not_inflated_to_pad_the_corpus(self):
        values = self.report["by_category"]["skill_manager"]
        self.assertAlmostEqual(values["char_share"], 0.03, delta=0.005)

    def test_dialogue_was_not_used_to_fill_the_shortfall(self):
        values = self.report["by_category"]["dialogue"]
        self.assertAlmostEqual(values["char_share"], 0.10, delta=0.01)
        self.assertLess(values["supply_utilization"], 0.15)

    def test_new_sources_actually_entered_the_corpus(self):
        # Documents are grouped by category, so a prefix scan would miss the
        # later categories entirely.
        sources = Counter()
        with V2_TRAIN.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    sources[json.loads(line)["source"]] += 1
        for source in ("fineweb2_ko_extra", "korean_wikipedia_technical",
                       "starcoderdata_json", "the_stack_smol_sql"):
            self.assertGreater(sources[source], 0, source)

    def test_token_counts_are_measured_not_estimated(self):
        totals = self.tokens["totals"]
        self.assertEqual(totals["tokens"],
                         sum(v["tokens"] for v in self.tokens["by_category"].values()))
        self.assertGreater(totals["tokens_per_character"], 0.4)

    def test_vocabulary_is_still_almost_fully_exercised(self):
        coverage = self.tokens["vocabulary_coverage"]
        self.assertGreater(coverage["used_share"], 0.99)
        for token, entry in coverage["control_tokens"].items():
            self.assertGreater(entry["count"], 0, token)

    def test_packing_is_deterministic_and_efficient(self):
        self.assertEqual(self.pack["sequence_length"], 1024)
        self.assertGreater(self.pack["packing_efficiency"], 0.99)
        self.assertLess(self.pack["dropped_tail_tokens"], 1024)
        self.assertIn("endoftext", self.pack["separator"])

    def test_epoch_fraction_matches_the_packed_token_count(self):
        budget = self.pack["budget"]
        self.assertAlmostEqual(budget["epochs_required"],
                               60_000_000 / self.pack["packed_tokens"], places=9)


@unittest.skipUnless(V2_TRAIN.exists(), "v2 corpus not built")
class V2LeakageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hashes = {content_hash(json.loads(l)["text"])
                      for l in V2_TRAIN.open(encoding="utf-8") if l.strip()}

    def _check(self, path: Path, kind: str):
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
        self.assertEqual(overlap, 0, path.name)

    def test_no_overlap_with_any_evaluation_set(self):
        for path in (ORIG_BALANCED, ORIG_KOREAN, DECON_BALANCED, DECON_KOREAN):
            self._check(path, "jsonl")

    def test_no_overlap_with_held_out_sources(self):
        self._check(ROOT / "data/processed/fineweb_general_holdout.txt", "txt")
        for name in ("aihub_documents_valid_clean.txt", "aihub_dialogue_valid_clean.txt"):
            self._check(ROOT / "data/processed" / name, "txt")

    def test_no_overlap_with_stage1_recovery_training(self):
        self._check(ROOT / "data/processed/recovery_train.jsonl", "jsonl")

    def test_nikl_directory_is_still_rejected_outright(self):
        nikl = ROOT / "data/NIKL_NEWSPAPER_2025_v1.1"
        if nikl.exists():
            with self.assertRaises(ForbiddenSourceError):
                assert_source_allowed("korean", nikl)


if __name__ == "__main__":
    unittest.main()
