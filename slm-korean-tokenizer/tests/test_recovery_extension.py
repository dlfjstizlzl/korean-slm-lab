from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "corpus"))

from build_recovery_data import (  # noqa: E402
    ForbiddenSourceError,
    assert_record_allowed,
    assert_source_allowed,
    line_key,
)
from recovery_eval import load_eval  # noqa: E402
from recovery_saturation import (  # noqa: E402
    ABSOLUTE_THRESHOLD,
    RELATIVE_THRESHOLD,
    classify,
    collect_points,
    interval_rates,
)
from recovery_train import ALL_ROWS  # noqa: E402

KOREAN_EVAL = ROOT / "data" / "eval" / "recovery_korean_large_eval.jsonl"
KOREAN_REPORT = ROOT / "outputs" / "korean_large_eval_report.json"
TRAIN_FILE = ROOT / "data" / "processed" / "recovery_train.jsonl"
STAGE1 = ROOT / "outputs" / "models" / "qwen3-0.6b-ko32k-recovery-stage1"
EXTENDED = ROOT / "outputs" / "models" / "qwen3-0.6b-ko32k-recovery-stage1-extended"
EXTENDED_RUN = ROOT / "outputs" / "recovery" / "stage1-extended-mean-all_rows.json"
SATURATION = ROOT / "outputs" / "recovery_saturation.json"
EXTENSION_REPORT = ROOT / "outputs" / "stage1_extension_report.json"


# --------------------------------------------------------------------------- #
# Saturation arithmetic (thresholds fixed in code, not tuned to results)
# --------------------------------------------------------------------------- #

class SaturationTests(unittest.TestCase):
    def test_thresholds_are_the_declared_constants(self):
        self.assertEqual(RELATIVE_THRESHOLD, 0.25)
        self.assertEqual(ABSOLUTE_THRESHOLD, 0.03)

    def test_rate_is_normalized_to_one_million_tokens(self):
        rates = interval_rates([(0, 3.0), (500_000, 2.8)])
        self.assertAlmostEqual(rates[0]["improvement"], 0.2)
        self.assertAlmostEqual(rates[0]["improvement_per_1m"], 0.4)

    def test_absolute_criterion_fires_on_a_small_final_interval(self):
        verdict = classify(interval_rates([(0, 3.0), (1_000_000, 2.9), (2_000_000, 2.88)]))
        self.assertTrue(verdict["absolute_criterion_met"])
        self.assertTrue(verdict["saturating"])

    def test_relative_criterion_fires_on_a_collapsing_rate(self):
        verdict = classify(interval_rates([(0, 5.0), (1_000_000, 3.0), (2_000_000, 2.9)]))
        self.assertTrue(verdict["relative_criterion_met"])
        self.assertFalse(verdict["absolute_criterion_met"])
        self.assertTrue(verdict["saturating"])

    def test_steady_large_improvement_is_not_saturating(self):
        verdict = classify(interval_rates([(0, 5.0), (1_000_000, 4.0), (2_000_000, 3.2)]))
        self.assertFalse(verdict["saturating"])

    def test_points_from_several_runs_merge_and_deduplicate(self):
        runs = [
            {"history": [{"tokens_seen": 0, "nll_per_character": 3.0},
                         {"tokens_seen": 2_000_000, "nll_per_character": 2.1}]},
            {"history": [{"tokens_seen": 2_000_000, "nll_per_character": 2.1},
                         {"tokens_seen": 4_000_000, "nll_per_character": 2.0}]},
        ]
        self.assertEqual(collect_points(runs),
                         [(0, 3.0), (2_000_000, 2.1), (4_000_000, 2.0)])

    def test_zero_length_intervals_are_ignored(self):
        self.assertEqual(interval_rates([(1_000, 2.0), (1_000, 1.9)]), [])


# --------------------------------------------------------------------------- #
# Large Korean evaluation set
# --------------------------------------------------------------------------- #

@unittest.skipUnless(KOREAN_EVAL.exists(), "large Korean eval not built")
class KoreanLargeEvalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = load_eval(KOREAN_EVAL)
        cls.report = json.loads(KOREAN_REPORT.read_text(encoding="utf-8"))

    def test_set_exceeds_the_two_million_character_target(self):
        total = sum(record["characters"] for record in self.records)
        self.assertGreaterEqual(total, 2_000_000)
        self.assertEqual(total, self.report["characters"])

    def test_category_shares_match_the_sixty_thirty_ten_plan(self):
        total = sum(record["characters"] for record in self.records)
        expected = {"korean_general": 0.60, "korean_technical": 0.30, "dialogue_summary": 0.10}
        for category, share in expected.items():
            actual = sum(r["characters"] for r in self.records if r["category"] == category) / total
            self.assertAlmostEqual(actual, share, places=2, msg=category)

    def test_no_overlap_with_the_recovery_training_corpus(self):
        train_keys = {
            line_key(json.loads(line)["text"])
            for line in TRAIN_FILE.open(encoding="utf-8") if line.strip()
        }
        eval_keys = {line_key(record["text"]) for record in self.records}
        self.assertEqual(eval_keys & train_keys, set())
        self.assertEqual(self.report["leakage"]["residual_train_overlap"], 0)

    def test_no_nikl_record_is_present(self):
        for record in self.records:
            assert_source_allowed(record["source"])
            assert_record_allowed(record["text"], record["source"])
        self.assertFalse(self.report["nikl"]["used"])

    def test_nikl_guard_still_rejects_the_sealed_corpus(self):
        with self.assertRaises(ForbiddenSourceError):
            assert_source_allowed("nikl_newspaper")
        with self.assertRaises(ForbiddenSourceError):
            assert_record_allowed("NIRW2500000001.1", "fineweb")

    def test_record_ids_are_unique_and_deterministic(self):
        ids = [record["id"] for record in self.records]
        self.assertEqual(len(ids), len(set(ids)))

    def test_loading_is_deterministic(self):
        self.assertEqual([r["id"] for r in load_eval(KOREAN_EVAL)],
                         [r["id"] for r in self.records])


# --------------------------------------------------------------------------- #
# Resume semantics
# --------------------------------------------------------------------------- #

@unittest.skipUnless(EXTENDED_RUN.exists(), "extended run not completed")
class ExtendedRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = json.loads(EXTENDED_RUN.read_text(encoding="utf-8"))

    def test_resumed_from_the_stage1_checkpoint_not_the_transplant(self):
        self.assertEqual(self.record["resume"]["resumed_from"],
                         "outputs/models/qwen3-0.6b-ko32k-recovery-stage1")

    def test_optimizer_state_could_not_be_restored_and_says_so(self):
        # Stage 1 saved no optimizer state; the report must record that honestly.
        self.assertFalse(self.record["resume"]["exact_optimizer_resume"])

    def test_schedule_continues_one_cosine_instead_of_restarting_at_the_peak(self):
        resume = self.record["resume"]
        self.assertEqual(resume["schedule_total_steps"], 1000)
        self.assertEqual(resume["schedule_start_step"], 500)
        self.assertLess(resume["starting_learning_rate"], self.record["hyperparameters"]["learning_rate"])
        self.assertGreater(resume["starting_learning_rate"], 0.0)

    def test_cumulative_token_accounting(self):
        corpus = self.record["corpus"]
        self.assertEqual(corpus["tokens_before_run"], 2_048_000)
        self.assertEqual(corpus["tokens_this_run"], 2_048_000)
        self.assertEqual(corpus["tokens_trained"], 4_096_000)
        self.assertEqual(corpus["tokens_trained"],
                         corpus["tokens_before_run"] + corpus["tokens_this_run"])

    def test_data_stream_resumes_where_stage1_stopped(self):
        corpus = self.record["corpus"]
        # Stage 1 consumed 500 steps x 8 blocks; this run must not replay them.
        self.assertEqual(corpus["block_cursor_start"], 4000)
        self.assertLess(corpus["block_cursor_start"], corpus["blocks"])

    def test_corpus_wrapped_around_so_part_of_the_run_repeated_data(self):
        # 7,556 blocks total and 4,000 already consumed leaves 3,556 fresh blocks
        # (444 steps); the remaining 56 steps reshuffle and repeat. The report must
        # show this rather than implying 2M wholly fresh tokens.
        corpus = self.record["corpus"]
        fresh_blocks = corpus["blocks"] - corpus["block_cursor_start"]
        self.assertLess(fresh_blocks, 500 * 8)
        self.assertGreater(corpus["epochs"], 1.0)
        self.assertLess(corpus["block_cursor_end"], corpus["block_cursor_start"])

    def test_transformer_stayed_frozen(self):
        parameters = self.record["parameters"]
        self.assertEqual(parameters["frozen_parameters"], 440_467_456)
        self.assertEqual(parameters["unique_trainable_tensors"], 1)
        self.assertEqual(parameters["total_parameters"], 473_235_456)
        self.assertTrue(parameters["tied"])

    def test_all_rows_mode_updated_the_whole_table(self):
        self.assertEqual(self.record["mode"], ALL_ROWS)
        self.assertEqual(self.record["parameters"]["updatable_rows"], 32000)
        self.assertEqual(self.record["parameters"]["updatable_parameters"], 32000 * 1024)

    def test_history_covers_the_requested_cumulative_checkpoints(self):
        tokens = [entry["tokens_seen"] for entry in self.record["history"]]
        self.assertEqual(tokens[0], 2_048_000)
        self.assertEqual(tokens[-1], 4_096_000)
        self.assertGreaterEqual(len(tokens), 5)

    def test_learning_rate_decays_monotonically_across_the_run(self):
        rates = [entry["learning_rate"] for entry in self.record["history"]]
        self.assertEqual(rates, sorted(rates, reverse=True))


@unittest.skipUnless(EXTENDED.is_dir(), "extended checkpoint not saved")
class ExtendedCheckpointTests(unittest.TestCase):
    def test_stage1_checkpoint_was_not_overwritten(self):
        self.assertTrue(STAGE1.is_dir())
        self.assertNotEqual(STAGE1.resolve(), EXTENDED.resolve())

    def test_extended_differs_from_stage1(self):
        from safetensors.torch import load_file

        a = load_file(STAGE1 / "model.safetensors")["model.embed_tokens.weight"]
        b = load_file(EXTENDED / "model.safetensors")["model.embed_tokens.weight"]
        self.assertEqual(a.shape, b.shape)
        self.assertFalse(torch.equal(a, b))

    def test_optimizer_state_is_saved_for_the_next_stage(self):
        self.assertTrue((EXTENDED / "optimizer_state.pt").exists())
        state = torch.load(EXTENDED / "optimizer_state.pt", map_location="cpu", weights_only=False)
        self.assertEqual(state["tokens_seen"], 4_096_000)
        self.assertIn("optimizer", state)
        self.assertIn("scheduler", state)

    def test_structure_is_unchanged(self):
        config = json.loads((EXTENDED / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["vocab_size"], 32000)
        self.assertEqual(config["hidden_size"], 1024)
        self.assertTrue(config["tie_word_embeddings"])

    def test_reloads_with_tied_finite_weights_and_same_parameter_count(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(EXTENDED)
        model = AutoModelForCausalLM.from_pretrained(EXTENDED, dtype="auto")
        self.assertEqual(len(tokenizer), 32000)
        embedding = model.get_input_embeddings().weight
        self.assertEqual(embedding.data_ptr(), model.get_output_embeddings().weight.data_ptr())
        self.assertEqual(tuple(embedding.shape), (32000, 1024))
        self.assertTrue(torch.isfinite(embedding.float()).all())
        self.assertEqual(sum(p.numel() for p in model.parameters()), 473_235_456)

    def test_forward_pass_is_finite(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(EXTENDED)
        model = AutoModelForCausalLM.from_pretrained(EXTENDED, dtype=torch.float32)
        model.eval()
        for text in ("가상 메모리는 운영체제에서", "<검색>Apple Silicon 통합 메모리", "SELECT id FROM t;"):
            encoded = tokenizer(text, return_tensors="pt")
            with torch.inference_mode():
                output = model(**encoded, labels=encoded["input_ids"])
            self.assertTrue(torch.isfinite(output.logits).all(), text)
            self.assertTrue(torch.isfinite(output.loss), text)
            self.assertEqual(output.logits.shape[-1], 32000)


@unittest.skipUnless(SATURATION.exists(), "saturation analysis not generated")
class SaturationReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads(SATURATION.read_text(encoding="utf-8"))

    def test_curve_is_monotonic_in_tokens(self):
        tokens = [point["tokens"] for point in self.report["curve"]]
        self.assertEqual(tokens, sorted(tokens))
        self.assertEqual(len(tokens), len(set(tokens)))

    def test_curve_reaches_the_full_cumulative_budget(self):
        self.assertEqual(self.report["curve"][-1]["tokens"], 4_096_000)

    def test_recovery_ratio_is_consistent_with_the_curve(self):
        final = self.report["curve"][-1]
        gap = self.report["recoverable_gap"]
        expected = (self.report["transplant_nll_per_character"] - final["nll_per_character"]) / gap
        self.assertAlmostEqual(final["recovery_ratio"], expected, places=9)

    def test_verdict_used_the_declared_thresholds(self):
        self.assertEqual(self.report["saturation"]["thresholds"],
                         {"relative": RELATIVE_THRESHOLD, "absolute": ABSOLUTE_THRESHOLD})


@unittest.skipUnless(EXTENSION_REPORT.exists(), "extension report not generated")
class ExtensionReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads(EXTENSION_REPORT.read_text(encoding="utf-8"))

    def test_large_korean_eval_covers_all_three_models(self):
        korean = self.report["korean_large_eval"]
        self.assertEqual(set(korean), {"original_qwen", "stage1", "stage1_extended"})
        for name, entry in korean.items():
            self.assertGreater(entry["overall"]["characters"], 2_000_000, name)

    def test_all_three_models_scored_identical_text(self):
        characters = {entry["overall"]["characters"]
                      for entry in self.report["korean_large_eval"].values()}
        self.assertEqual(len(characters), 1)

    def test_transformer_was_never_unfrozen(self):
        self.assertFalse(self.report["transformer_unfrozen"])

    def test_stage2_plan_gives_the_embedding_a_higher_rate_than_the_blocks(self):
        plan = self.report["stage2_plan"]
        self.assertGreater(plan["embedding_lr"], plan["transformer_lr"])
        self.assertGreater(plan["differential_lr_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
