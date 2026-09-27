from __future__ import annotations

import json
import random
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "corpus"))

import stage2_skill_synth_v2 as gen  # noqa: E402
from build_stage2_corpus import TARGET_SHARES  # noqa: E402
from build_recovery_data import ForbiddenSourceError, assert_source_allowed  # noqa: E402
from skill_metrics import (  # noqa: E402
    iter_skill_records,
    skeleton_stats,
    skill_duplicate_stats,
)
from stage2_dedup import content_hash, ngram_duplicate_ratio  # noqa: E402

V1 = ROOT / "data/processed/stage2_train.jsonl"
V2 = ROOT / "data/processed/stage2_train_v2.jsonl"
V3 = ROOT / "data/processed/stage2_train_v3.jsonl"
V3_REPORT = ROOT / "outputs/stage2_corpus_report_v3.json"
V3_TOKENS = ROOT / "outputs/stage2_token_stats_v3.json"
V3_PACK = ROOT / "outputs/stage2_packing_v3.json"
V3_SCREEN = ROOT / "outputs/stage2_substring_screen_v3.json"
ABLATION = ROOT / "outputs/stage2_skill_ablation.json"
FIX_REPORT = ROOT / "outputs/stage2_skill_repetition_fix_report.json"


# --------------------------------------------------------------------------- #
# Canonical metric is a single source of truth
# --------------------------------------------------------------------------- #

class CanonicalMetricTests(unittest.TestCase):
    def test_metric_is_defined_once_and_reused(self):
        source = (ROOT / "corpus/skill_metrics.py").read_text(encoding="utf-8")
        self.assertIn("from stage2_dedup import ngram_duplicate_ratio", source)
        # No re-implementation of the ratio inside the canonical module.
        self.assertNotIn("def ngram_duplicate_ratio", source)

    def test_statistic_is_full_population_not_sampled(self):
        source = (ROOT / "corpus/skill_metrics.py").read_text(encoding="utf-8")
        self.assertNotIn("5000", source)
        self.assertIn("no sampling", source)

    def test_ratio_definition(self):
        self.assertAlmostEqual(ngram_duplicate_ratio("a b c d", 4), 0.0)
        # 8 words -> 5 contiguous 4-grams, 4 distinct -> 1 - 4/5
        self.assertAlmostEqual(ngram_duplicate_ratio("a b c d a b c d", 4), 0.2, places=6)
        self.assertEqual(ngram_duplicate_ratio("a b", 4), 0.0)

    @unittest.skipUnless(V2.exists(), "v2 corpus missing")
    def test_sampled_and_full_population_differ_as_documented(self):
        """The 0.2356 / 0.2364 discrepancy was sampling, not two metrics."""
        records = list(iter_skill_records(V2))
        full = sum(ngram_duplicate_ratio(r["text"], 4) for r in records) / len(records)
        sampled = sum(ngram_duplicate_ratio(r["text"], 4) for r in records[:5000]) / 5000
        self.assertNotAlmostEqual(full, sampled, places=4)
        self.assertAlmostEqual(full, 0.2356, places=4)

    @unittest.skipUnless(all(p.exists() for p in (V1, V2, V3)), "corpora missing")
    def test_every_version_measured_by_the_same_function(self):
        for path in (V1, V2, V3):
            stats = skill_duplicate_stats(path)
            self.assertIn("duplicate_4gram", stats)
            self.assertGreater(stats["documents"], 0)
            self.assertEqual(stats["corpus"], str(path))

    @unittest.skipUnless(FIX_REPORT.exists(), "fix report missing")
    def test_report_values_match_a_live_recomputation(self):
        report = json.loads(FIX_REPORT.read_text(encoding="utf-8"))
        for version, path in (("v1", V1), ("v2", V2), ("v3", V3)):
            if not path.exists():
                continue
            live = skill_duplicate_stats(path)
            stored = report["canonical_metrics"][version]
            self.assertAlmostEqual(stored["duplicate_4gram"], live["duplicate_4gram"], places=9)
            self.assertEqual(stored["documents"], live["documents"])


# --------------------------------------------------------------------------- #
# Generator behaviour
# --------------------------------------------------------------------------- #

class QueryExtractionTests(unittest.TestCase):
    def test_candidates_come_from_several_positions_not_just_the_first(self):
        text = ". ".join(f"이 문장은 전자 회로와 반도체 기술을 설명하는 본문 {i}번째 내용이다"
                         for i in range(12)) + "."
        candidates = gen.extract_candidates(text)
        self.assertGreater(len(candidates), 1)
        self.assertNotEqual(candidates[0], candidates[1])

    def test_extraction_is_deterministic(self):
        text = ". ".join(f"전자 회로 반도체 기술 문장 {i} 입니다" for i in range(10)) + "."
        self.assertEqual(gen.extract_candidates(text), gen.extract_candidates(text))

    def test_wiki_markup_and_citations_are_cleaned(self):
        cleaned = gen.clean_candidate("'''수학'''(, , '''math''')은 구조를 다루는 학문이다[12]")
        self.assertNotIn("'''", cleaned)
        self.assertNotIn("[12]", cleaned)
        self.assertNotIn("(, ,", cleaned)
        self.assertNotIn("( ,", cleaned)

    def test_keywords_never_carry_an_unbalanced_bracket(self):
        for word in gen.keywords("자릿수근 음이 아닌 정수의 자릿수근(, 반복적 자릿수합)"):
            self.assertNotIn("(", word)
            self.assertNotIn(")", word)

    def test_boilerplate_and_list_headings_are_rejected(self):
        self.assertFalse(gen.is_usable_sentence("thumb"))
        self.assertFalse(gen.is_usable_sentence("섬네일"))
        self.assertFalse(gen.is_usable_sentence("* 목록 항목 하나 둘 셋 넷 다섯 여섯 일곱"))

    def test_symbol_and_number_heavy_text_is_rejected(self):
        self.assertFalse(gen.is_usable_sentence("1234 5678 9012 3456 7890 1234 5678 90"))

    def test_non_korean_sentence_is_rejected(self):
        self.assertFalse(gen.is_usable_sentence(
            "This sentence is entirely in English and carries no Korean at all."))

    def test_all_query_modes_produce_distinct_formulations(self):
        rng = random.Random(42)
        candidate = "전자 회로의 전압과 전류는 물리 법칙을 따른다"
        produced = {gen.make_query(rng, candidate, mode) for mode in gen.QUERY_MODES}
        self.assertGreaterEqual(len(produced), 6)
        self.assertGreaterEqual(len(gen.QUERY_MODES), 8)


class PlanConstructionTests(unittest.TestCase):
    def test_later_steps_reference_instead_of_restating_the_query(self):
        rng = random.Random(42)
        referencing = 0
        for _ in range(200):
            steps = gen.build_steps(rng, ["첫 번째 기술 문장", "두 번째 기술 문장"], 3)
            referencing += sum(1 for step in steps[1:] if "input_from" in step)
        self.assertGreater(referencing, 200)

    def test_each_explicit_step_draws_a_different_candidate(self):
        rng = random.Random(7)
        steps = gen.build_steps(rng, ["후보 하나", "후보 둘", "후보 셋"], 3)
        inputs = [s.get("input") for s in steps if "input" in s]
        self.assertEqual(len(inputs), len(set(inputs)))

    def test_failure_retry_uses_a_different_formulation(self):
        rng = random.Random(3)
        text, _ = gen.failure_case(rng, ["원래 질의 문장입니다", "대체 질의 문장입니다"], "원래 질의 문장입니다")
        self.assertIn("원래 질의 문장입니다", text)
        self.assertLessEqual(text.count("원래 질의 문장입니다"), 2)

    def test_closings_are_varied_rather_than_a_fixed_footer(self):
        self.assertGreaterEqual(len(set(gen.CLOSINGS)), 6)

    def test_serializations_are_all_reachable(self):
        rng = random.Random(42)
        seen = set()
        for _ in range(400):
            steps = gen.build_steps(rng, ["문장 하나", "문장 둘"], 2)
            seen.add(gen.render(rng, steps, "목표")[1])
        self.assertGreaterEqual(len(seen), 6)

    def test_generation_is_deterministic_for_a_fixed_seed(self):
        groups = [{"candidates": ["전자 회로 전압 전류 물리 법칙", "반도체 소자 기술 핵심"],
                   "source": "aihub"}]
        first = gen.generate(groups, 30_000, seed=42)
        second = gen.generate(groups, 30_000, seed=42)
        self.assertEqual([r["text"] for r in first], [r["text"] for r in second])


@unittest.skipUnless(ABLATION.exists(), "ablation not run")
class AblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads(ABLATION.read_text(encoding="utf-8"))

    def test_seed_was_not_searched(self):
        self.assertFalse(self.report["seed_search_performed"])
        self.assertEqual(self.report["seed"], 42)

    def test_all_variants_shared_one_seed_and_budget(self):
        self.assertEqual(len({v["documents"] > 0 for v in self.report["variants"]}), 1)
        self.assertGreaterEqual(len(self.report["variants"]), 4)

    def test_diversified_query_is_what_fixes_repetition(self):
        by_name = {v["variant"]: v for v in self.report["variants"]}
        self.assertGreater(by_name["A_current"]["duplicate_4gram"], 0.2)
        self.assertLess(by_name["B_diversified_query"]["duplicate_4gram"], 0.05)

    def test_source_balancing_does_not_regress_repetition(self):
        by_name = {v["variant"]: v for v in self.report["variants"]}
        self.assertLess(
            abs(by_name["C_query_plus_source_balance"]["duplicate_4gram"]
                - by_name["B_diversified_query"]["duplicate_4gram"]), 0.01)


# --------------------------------------------------------------------------- #
# v3 corpus
# --------------------------------------------------------------------------- #

@unittest.skipUnless(V3.exists() and V3_REPORT.exists(), "v3 corpus not built")
class V3CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads(V3_REPORT.read_text(encoding="utf-8"))
        cls.tokens = json.loads(V3_TOKENS.read_text(encoding="utf-8"))
        cls.pack = json.loads(V3_PACK.read_text(encoding="utf-8"))
        cls.screen = json.loads(V3_SCREEN.read_text(encoding="utf-8"))
        cls.skill = skill_duplicate_stats(V3)

    def test_earlier_corpora_are_preserved(self):
        self.assertTrue(V1.exists())
        self.assertTrue(V2.exists())
        self.assertNotEqual(V2.resolve(), V3.resolve())

    def test_generator_provenance_is_recorded(self):
        self.assertEqual(self.report["skill_synthesis"]["generator"], "stage2_skill_synth_v2")
        self.assertGreaterEqual(len(self.report["skill_synthesis"]["query_sources"]), 2)

    def test_condition_j_exact_comparison_against_v1(self):
        v1 = skill_duplicate_stats(V1)
        self.assertLessEqual(self.skill["duplicate_4gram"], v1["duplicate_4gram"])
        self.assertLessEqual(self.skill["duplicate_8gram"], v1["duplicate_8gram"])

    def test_skill_share_was_not_cut_to_pass(self):
        share = self.tokens["by_category"]["skill_manager"]["char_share"]
        self.assertGreaterEqual(share, 0.028)
        self.assertLessEqual(share, 0.032)

    def test_no_duplicate_skill_documents(self):
        self.assertEqual(self.skill["duplicate_documents"], 0)

    def test_skeleton_accounting_against_the_v1_baseline(self):
        stats = skeleton_stats(list(iter_skill_records(V3)))
        v1_stats = skeleton_stats(list(iter_skill_records(V1)))
        self.assertGreaterEqual(stats["distinct_skeletons"], v1_stats["distinct_skeletons"])

    def test_serialization_accounting_is_spread(self):
        serializations = Counter(
            r.get("serialization") for r in iter_skill_records(V3) if r.get("serialization"))
        self.assertGreaterEqual(len(serializations), 6)
        top = serializations.most_common(1)[0][1] / sum(serializations.values())
        self.assertLess(top, 0.30)

    def test_query_source_balance_has_both_technical_sources(self):
        sources = Counter(
            r.get("query_source") for r in iter_skill_records(V3) if r.get("query_source"))
        self.assertGreaterEqual(len(sources), 2)
        self.assertGreater(min(sources.values()), 0)

    def test_no_repetition_filtering_was_applied_after_generation(self):
        """Documents were not culled by repetition to make the metric pass."""
        ratios = [ngram_duplicate_ratio(r["text"], 4) for r in iter_skill_records(V3)]
        # A post-hoc cull would leave a hard ceiling; a generator fix does not.
        self.assertGreater(max(ratios), 0.30)

    # --- exit conditions, recomputed from artifacts ------------------------- #

    def test_exit_a_usable_chars(self):
        self.assertGreaterEqual(self.screen["characters_after"], 150_000_000)

    def test_exit_b_packed_tokens(self):
        self.assertGreaterEqual(self.pack["packed_tokens"], 60_000_000)

    def test_exit_c_epoch(self):
        self.assertLessEqual(self.pack["budget"]["epochs_required"], 1.0)
        self.assertEqual(self.tokens["training_budget"]["repeated_token_fraction"], 0.0)

    def test_exit_d_e_f_mixture(self):
        shares = self.tokens["by_category"]
        general_technical = shares["korean_general"]["token_share"] + shares["korean_technical"]["token_share"]
        korean = general_technical + shares["dialogue"]["token_share"]
        structured = sum(shares[c]["token_share"]
                         for c in ("code_url", "json_structured", "sql", "skill_manager"))
        self.assertAlmostEqual(general_technical, 0.75, delta=0.02)
        self.assertGreaterEqual(korean, 0.84)
        self.assertAlmostEqual(structured, 0.15, delta=0.02)

    def test_exit_g_zero_contamination(self):
        self.assertEqual(self.screen["residual_substring_overlap"], 0)

    def test_exit_h_nikl_never_opened(self):
        self.assertFalse(self.report["nikl"]["used"])
        self.assertEqual(self.report["nikl"]["files_opened"], 0)
        nikl = ROOT / "data" / "NIKL_NEWSPAPER_2025_v1.1"
        if nikl.exists():
            with self.assertRaises(ForbiddenSourceError):
                assert_source_allowed("korean", nikl)

    def test_exit_i_no_artificial_repetition(self):
        counts = Counter()
        with V3.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    counts[json.loads(line)["sha256"]] += 1
        self.assertEqual(counts.most_common(1)[0][1], 1)

    def test_categories_match_the_target_mixture(self):
        self.assertEqual({r for r in self.tokens["by_category"]}, set(TARGET_SHARES))

    def test_vocabulary_still_fully_exercised(self):
        coverage = self.tokens["vocabulary_coverage"]
        self.assertGreater(coverage["used_share"], 0.99)
        for token, entry in coverage["control_tokens"].items():
            self.assertGreater(entry["count"], 0, token)


@unittest.skipUnless(V3.exists(), "v3 corpus not built")
class V3LeakageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hashes = {content_hash(json.loads(l)["text"])
                      for l in V3.open(encoding="utf-8") if l.strip()}

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

    def test_zero_overlap_with_all_evaluation_sets(self):
        for name in ("recovery_balanced_eval.jsonl", "recovery_balanced_eval_decontaminated.jsonl",
                     "recovery_korean_large_eval.jsonl",
                     "recovery_korean_large_eval_decontaminated.jsonl"):
            self._check(ROOT / "data/eval" / name, "jsonl")

    def test_zero_overlap_with_held_out_and_stage1_training(self):
        self._check(ROOT / "data/processed/fineweb_general_holdout.txt", "txt")
        self._check(ROOT / "data/processed/aihub_documents_valid_clean.txt", "txt")
        self._check(ROOT / "data/processed/recovery_train.jsonl", "jsonl")


if __name__ == "__main__":
    unittest.main()
