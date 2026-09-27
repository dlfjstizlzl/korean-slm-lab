from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "tokenizer"))

from common import CONTROL_TOKENS, assert_control_tokens_atomic  # noqa: E402
from transplant_ko32k import (  # noqa: E402
    DECOMPOSED,
    EXACT,
    RANDOM,
    bpe_segments,
    byte_level_alphabet,
    combine_rows,
    has_hangul,
    is_byte_level_string,
    plan_token,
    to_byte_level,
)

CHECKPOINT = ROOT / "outputs" / "models" / "qwen3-0.6b-ko32k-transplanted"
REPORT = ROOT / "outputs" / "embedding_transplant_report.json"
VALIDATION = ROOT / "outputs" / "ko32k_transplant_validation.json"


HANGUL_BYTES = to_byte_level("새말")
CONTROL_BYTES = to_byte_level("<검>")


class FakeBpe:
    """Segments a ByteLevel string with a fixed toy merge table."""

    TABLE = {
        HANGUL_BYTES: [HANGUL_BYTES[:3], HANGUL_BYTES[3:]],
        CONTROL_BYTES: ["<", CONTROL_BYTES[1:-1], ">"],
        "abc": ["ab", "c"],
        "lossy": ["lo"],  # deliberately does not reconstruct
    }

    def tokenize(self, text):
        pieces = self.TABLE.get(text)
        if pieces is None:
            raise ValueError(f"cannot segment {text!r}")
        return [type("T", (), {"value": piece})() for piece in pieces]


VOCAB = {
    "same": 7, HANGUL_BYTES[:3]: 2, HANGUL_BYTES[3:]: 3,
    "<": 27, CONTROL_BYTES[1:-1]: 40, ">": 29,
    "ab": 11, "c": 12, "lo": 13,
}
ALPHABET = byte_level_alphabet()


def plan(token: str, is_added: bool = False, old_rows: int = 200):
    return plan_token(
        token, is_added=is_added, old_vocab=VOCAB, old_rows=old_rows,
        bpe_model=FakeBpe(), alphabet=ALPHABET,
    )


class ByteLevelHelperTests(unittest.TestCase):
    def test_matches_the_tokenizers_byte_level_encoding(self):
        from tokenizers import Tokenizer, models, pre_tokenizers

        reference = Tokenizer(models.BPE())
        reference.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        for text in ("새말", "가상메모리", "<검색>", " memory", "SELECT * FROM t;", "😀"):
            expected = "".join(piece for piece, _ in reference.pre_tokenizer.pre_tokenize_str(text))
            self.assertEqual(to_byte_level(text), expected, text)

    def test_byte_level_alphabet_covers_every_byte(self):
        self.assertEqual(len(ALPHABET), 256)
        self.assertTrue(all(to_byte_level(chr(b)) for b in range(1, 128)))

    def test_raw_hangul_is_not_a_byte_level_string(self):
        self.assertFalse(is_byte_level_string("가상", ALPHABET))
        self.assertTrue(is_byte_level_string("ìĥĪë§IJ", ALPHABET))

    def test_lossy_segmentation_is_rejected(self):
        self.assertEqual(bpe_segments(FakeBpe(), "lossy"), [])
        self.assertEqual(bpe_segments(FakeBpe(), "abc"), ["ab", "c"])

    def test_unsegmentable_input_returns_empty(self):
        self.assertEqual(bpe_segments(FakeBpe(), "unknown"), [])

    def test_hangul_detection(self):
        self.assertTrue(has_hangul("메모리"))
        self.assertFalse(has_hangul("memory"))


class TransplantPolicyTests(unittest.TestCase):
    def test_exact_match_wins_over_decomposition(self):
        strategy, ids, pieces, _ = plan("same")
        self.assertEqual((strategy, ids, pieces), (EXACT, [7], ["same"]))

    def test_exact_match_outside_embedding_rows_falls_through(self):
        strategy, _, _, _ = plan("same", old_rows=5)
        self.assertEqual(strategy, RANDOM)

    def test_byte_level_token_is_decomposed(self):
        strategy, ids, pieces, byte_form = plan(HANGUL_BYTES)
        self.assertEqual((strategy, ids), (DECOMPOSED, [2, 3]))
        self.assertEqual(pieces, [HANGUL_BYTES[:3], HANGUL_BYTES[3:]])
        self.assertEqual(byte_form, HANGUL_BYTES)

    def test_added_token_is_byte_level_encoded_before_decomposition(self):
        strategy, ids, pieces, byte_form = plan("<검>", is_added=True)
        self.assertEqual(byte_form, CONTROL_BYTES)
        self.assertEqual((strategy, ids), (DECOMPOSED, [27, 40, 29]))
        self.assertEqual(pieces, ["<", CONTROL_BYTES[1:-1], ">"])

    def test_unsegmentable_token_uses_random_fallback(self):
        strategy, ids, pieces, _ = plan("unknown")
        self.assertEqual((strategy, ids, pieces), (RANDOM, [], []))

    def test_lossy_segmentation_uses_random_fallback(self):
        self.assertEqual(plan("lossy")[0], RANDOM)


class CombineRowsTests(unittest.TestCase):
    def setUp(self):
        self.source = torch.tensor([[0.0, 0.0], [2.0, 4.0], [6.0, 8.0]])

    def test_mean_averages_source_rows(self):
        result = combine_rows(self.source, [1, 2], ["ab", "c"], "mean")
        torch.testing.assert_close(result, torch.tensor([4.0, 6.0]))

    def test_length_weighted_favours_longer_pieces(self):
        result = combine_rows(self.source, [1, 2], ["abc", "c"], "length_weighted")
        torch.testing.assert_close(result, torch.tensor([3.0, 5.0]))

    def test_single_source_is_identical_under_both_strategies(self):
        for strategy in ("mean", "length_weighted"):
            torch.testing.assert_close(
                combine_rows(self.source, [2], ["c"], strategy), torch.tensor([6.0, 8.0])
            )


@unittest.skipUnless(REPORT.exists(), "transplant report not generated")
class TransplantReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads(REPORT.read_text(encoding="utf-8"))

    def test_every_row_is_accounted_for(self):
        counts = self.report["counts"]
        self.assertEqual(counts["total"], 32000)
        self.assertEqual(
            counts["exact_copy"] + counts["decomposition"] + counts["random_fallback"], 32000
        )

    def test_no_random_fallback(self):
        self.assertEqual(self.report["counts"]["random_fallback"], 0)

    def test_base_added_tokens_are_exact_copies(self):
        self.assertEqual(self.report["counts"]["special_exact"], 26)
        self.assertEqual(self.report["counts"]["special_not_exact"], [])

    def test_control_tokens_are_initialized_from_old_decompositions(self):
        control = self.report["control_tokens"]
        self.assertEqual({entry["token"] for entry in control}, set(CONTROL_TOKENS))
        for entry in control:
            self.assertEqual(entry["strategy"], DECOMPOSED)
            self.assertGreaterEqual(len(entry["source_ids"]), 2)

    def test_exact_copies_are_bitwise_faithful(self):
        cosine = self.report["exact_copy_cosine"]
        self.assertGreater(cosine["sampled"], 100)
        self.assertGreater(cosine["min"], 0.9999)

    def test_embedding_parameters_shrink_by_about_79_percent(self):
        parameters = self.report["parameters"]
        self.assertEqual(parameters["new_embedding"], 32000 * 1024)
        self.assertAlmostEqual(parameters["embedding_reduction_ratio"], 0.789, places=2)

    def test_shapes_and_tying_are_recorded(self):
        structure = self.report["structure"]
        self.assertEqual(structure["new_embedding_shape"], [32000, 1024])
        self.assertEqual(structure["lm_head_shape"], [32000, 1024])
        self.assertTrue(structure["tie_word_embeddings"])
        self.assertTrue(structure["tied_actual"])


@unittest.skipUnless(CHECKPOINT.is_dir(), "transplant checkpoint not generated")
class CheckpointTests(unittest.TestCase):
    def test_config_declares_the_new_vocabulary(self):
        config = json.loads((CHECKPOINT / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["vocab_size"], 32000)
        self.assertEqual(config["hidden_size"], 1024)
        self.assertTrue(config["tie_word_embeddings"])

    def test_tokenizer_files_are_saved_alongside_the_model(self):
        for name in ("tokenizer.json", "tokenizer_config.json",
                     "special_tokens_map.json", "added_tokens.json",
                     "model.safetensors", "transplant_report.json"):
            self.assertTrue((CHECKPOINT / name).exists(), name)

    def test_saved_tokenizer_preserves_special_and_control_tokens(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
        self.assertEqual(len(tokenizer), 32000)
        self.assertEqual(set(assert_control_tokens_atomic(tokenizer)), set(CONTROL_TOKENS))
        self.assertEqual(len(tokenizer.added_tokens_decoder), 34)
        self.assertEqual(tokenizer.eos_token, "<|im_end|>")
        self.assertEqual(tokenizer.pad_token, "<|endoftext|>")

    def test_saved_embedding_matrix_is_finite_and_correctly_shaped(self):
        from safetensors import safe_open

        with safe_open(CHECKPOINT / "model.safetensors", framework="pt") as handle:
            keys = set(handle.keys())
            self.assertIn("model.embed_tokens.weight", keys)
            self.assertNotIn("lm_head.weight", keys)  # tied, so it is not serialized
            weight = handle.get_tensor("model.embed_tokens.weight")
        self.assertEqual(tuple(weight.shape), (32000, 1024))
        self.assertTrue(torch.isfinite(weight.to(torch.float32)).all())


@unittest.skipUnless(VALIDATION.exists(), "validation report not generated")
class ValidationReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads(VALIDATION.read_text(encoding="utf-8"))

    def test_reload_and_structure_pass(self):
        self.assertEqual(self.report["reload"]["model_load"], "PASS")
        self.assertTrue(self.report["structural"]["all_passed"])
        self.assertEqual(self.report["summary"]["failed_checks"], [])

    def test_forward_pass_is_finite_for_every_sample(self):
        forward = self.report["forward"]
        self.assertEqual(set(forward), {"korean", "mixed_technical", "skill_dsl", "json", "sql"})
        for name, entry in forward.items():
            self.assertTrue(entry["passed"], name)
            self.assertTrue(entry["ids_in_range"], name)
            self.assertEqual(entry["logits_shape"][-1], 32000, name)
            self.assertTrue(entry["logits_finite"] and entry["loss_finite"], name)

    def test_generation_smoke_test_does_not_crash(self):
        generation = self.report["generation_smoke"]
        self.assertTrue(generation["passed"])
        self.assertTrue(generation["ids_in_range"])
        self.assertFalse(generation["quality_evaluated"])

    def test_baseline_loss_is_recorded_for_both_models(self):
        baseline = self.report["baseline_eval"]
        for key in ("transplanted", "original"):
            self.assertGreater(baseline[key]["samples"], 0)
            self.assertTrue(0.0 < baseline[key]["nll_per_character"] < 100.0)
        self.assertIn("comparison_caveat", baseline)

    def test_non_embedding_parameters_are_untouched(self):
        parameters = self.report["parameters"]
        self.assertEqual(
            parameters["transplanted"]["non_embedding"], parameters["original"]["non_embedding"]
        )
        self.assertEqual(parameters["transplanted"]["embedding"], 32000 * 1024)


if __name__ == "__main__":
    unittest.main()


class SpecialTokenIdRangeTests(unittest.TestCase):
    """Every special-token id in a checkpoint must index the KO-32K vocabulary.

    Regression guard for a real defect: the transplant synchronized ids with
    `if value is not None`, so the KO-32K tokenizer's absent BOS left Qwen's
    inherited 151643 in config.json — an index outside a 32,000-entry table.
    """

    CHECKPOINTS = sorted((ROOT / "outputs" / "models").glob("*/config.json")) \
        if (ROOT / "outputs" / "models").is_dir() else []

    @unittest.skipUnless(CHECKPOINTS, "no checkpoints present")
    def test_no_special_token_id_is_out_of_range(self):
        offenders = []
        for config_path in self.CHECKPOINTS:
            for name in ("config.json", "generation_config.json"):
                path = config_path.parent / name
                if not path.exists():
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                vocab = payload.get("vocab_size") or 32000
                for key, value in payload.items():
                    if not key.endswith("token_id") or value is None:
                        continue
                    for item in (value if isinstance(value, list) else [value]):
                        if not 0 <= int(item) < vocab:
                            offenders.append(f"{path.parent.name}/{name}:{key}={item}")
        self.assertEqual(offenders, [], f"out-of-range ids: {offenders}")

    @unittest.skipUnless(CHECKPOINTS, "no checkpoints present")
    def test_bos_is_none_because_the_tokenizer_has_none(self):
        for config_path in self.CHECKPOINTS:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertIsNone(payload.get("bos_token_id"), config_path.parent.name)

    def test_transplant_assigns_ids_even_when_the_tokenizer_lacks_them(self):
        source = (ROOT / "model" / "transplant_ko32k.py").read_text(encoding="utf-8")
        marker = 'for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):'
        body = source[source.index(marker):source.index(marker) + 700]
        # The guarded form is what caused the defect; it must not come back.
        self.assertNotIn("if value is not None:\n            setattr(model.config", body)
