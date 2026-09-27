from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "tokenizer"))
sys.path.insert(0, str(ROOT / "corpus"))

from build_recovery_data import (  # noqa: E402
    ForbiddenSourceError,
    assert_record_allowed,
    assert_source_allowed,
    chunk_text,
    line_key,
)
from common import CONTROL_TOKENS  # noqa: E402
from recovery_eval import bootstrap_difference, load_eval  # noqa: E402
from recovery_train import DECOMPOSED_ONLY, build_blocks, freeze_transformer, install_row_mask  # noqa: E402
from transplant_ko32k import combine_rows, surface_weights, to_byte_level  # noqa: E402

EVAL_FILE = ROOT / "data" / "eval" / "recovery_balanced_eval.jsonl"
TRAIN_FILE = ROOT / "data" / "processed" / "recovery_train.jsonl"
DATA_REPORT = ROOT / "outputs" / "recovery_data_report.json"
ROW_GROUPS = ROOT / "outputs" / "ko32k_row_groups.json"
RECOVERY_REPORT = ROOT / "outputs" / "embedding_recovery_report.json"
INIT_DIRS = {
    "mean": ROOT / "outputs" / "models" / "qwen3-0.6b-ko32k-init-mean",
    "length_weighted": ROOT / "outputs" / "models" / "qwen3-0.6b-ko32k-init-length-weighted",
    "norm_matched": ROOT / "outputs" / "models" / "qwen3-0.6b-ko32k-init-norm-matched",
}
STAGE1 = ROOT / "outputs" / "models" / "qwen3-0.6b-ko32k-recovery-stage1"


# --------------------------------------------------------------------------- #
# Initialization strategies
# --------------------------------------------------------------------------- #

class StrategyTests(unittest.TestCase):
    def setUp(self):
        self.source = torch.tensor([[3.0, 4.0], [0.0, 10.0]])
        self.pieces = ["ab", "c"]  # surface weights 2 and 1

    def test_mean_is_the_plain_average(self):
        result = combine_rows(self.source, [0, 1], self.pieces, "mean")
        torch.testing.assert_close(result, torch.tensor([1.5, 7.0]))

    def test_length_weighted_uses_surface_characters(self):
        result = combine_rows(self.source, [0, 1], self.pieces, "length_weighted")
        torch.testing.assert_close(result, torch.tensor([2.0, 6.0]))

    def test_norm_matched_keeps_mean_direction_and_restores_magnitude(self):
        mean = combine_rows(self.source, [0, 1], self.pieces, "mean")
        matched = combine_rows(self.source, [0, 1], self.pieces, "norm_matched")
        target = float(self.source.norm(dim=1).mean())
        self.assertAlmostEqual(float(matched.norm()), target, places=5)
        cosine = torch.nn.functional.cosine_similarity(mean, matched, dim=0)
        self.assertAlmostEqual(float(cosine), 1.0, places=5)

    def test_single_source_row_is_strategy_independent(self):
        for strategy in ("mean", "length_weighted", "norm_matched"):
            torch.testing.assert_close(
                combine_rows(self.source, [1], ["c"], strategy), torch.tensor([0.0, 10.0])
            )

    def test_norm_matched_falls_back_when_the_mean_cancels(self):
        opposed = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
        result = combine_rows(opposed, [0, 1], ["a", "b"], "norm_matched")
        torch.testing.assert_close(result, torch.tensor([0.0, 0.0]))

    def test_surface_weights_count_characters_not_bytes(self):
        # One Hangul syllable is 3 ByteLevel symbols but 1 character.
        pieces = [to_byte_level("한"), to_byte_level("a")]
        self.assertEqual(surface_weights(pieces), [1.0, 1.0])

    def test_continuation_only_piece_weighs_zero(self):
        whole = to_byte_level("가")
        self.assertEqual(surface_weights([whole[:2], whole[2:]]), [1.0, 0.0])

    def test_surface_weights_fall_back_to_byte_lengths(self):
        whole = to_byte_level("가")
        self.assertEqual(surface_weights([whole[1:2], whole[2:]]), [1.0, 1.0])


@unittest.skipUnless(all(d.is_dir() for d in INIT_DIRS.values()), "init checkpoints not generated")
class InitCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from safetensors.torch import load_file

        cls.weights = {
            name: load_file(path / "model.safetensors")["model.embed_tokens.weight"]
            for name, path in INIT_DIRS.items()
        }
        cls.groups = json.loads(ROW_GROUPS.read_text(encoding="utf-8"))

    def test_exact_rows_are_identical_across_all_three_strategies(self):
        rows = torch.tensor(self.groups["exact"], dtype=torch.long)
        reference = self.weights["mean"][rows]
        for name, weight in self.weights.items():
            self.assertTrue(torch.equal(weight[rows], reference), name)

    def test_special_token_rows_are_identical_across_all_three_strategies(self):
        rows = torch.tensor(self.groups["base_added"], dtype=torch.long)
        self.assertEqual(len(self.groups["base_added"]), 26)
        reference = self.weights["mean"][rows]
        for name, weight in self.weights.items():
            self.assertTrue(torch.equal(weight[rows], reference), name)

    def test_strategies_differ_only_on_decomposed_rows(self):
        decomposed = set(self.groups["decomposed"])
        for name, weight in self.weights.items():
            if name == "mean":
                continue
            differing = (self.weights["mean"].float() - weight.float()).abs().sum(dim=1)
            changed = set(torch.nonzero(differing).flatten().tolist())
            self.assertTrue(changed <= decomposed, f"{name} changed non-decomposed rows")

    def test_norm_matched_raises_the_decomposed_norm(self):
        rows = torch.tensor(self.groups["decomposed"], dtype=torch.long)
        norms = {
            name: float(weight[rows].float().norm(dim=1).mean())
            for name, weight in self.weights.items()
        }
        self.assertGreater(norms["norm_matched"], norms["mean"])
        self.assertAlmostEqual(norms["mean"], 0.597, places=2)

    def test_every_checkpoint_shares_one_vocabulary_mapping(self):
        from transformers import AutoTokenizer

        vocabs = [AutoTokenizer.from_pretrained(path).get_vocab() for path in INIT_DIRS.values()]
        for vocab in vocabs[1:]:
            self.assertEqual(vocab, vocabs[0])
        self.assertEqual(len(vocabs[0]), 32000)

    def test_no_checkpoint_contains_nan_or_inf(self):
        for name, weight in self.weights.items():
            self.assertTrue(torch.isfinite(weight.float()).all(), name)


# --------------------------------------------------------------------------- #
# NIKL safeguard and data construction
# --------------------------------------------------------------------------- #

class NiklSafeguardTests(unittest.TestCase):
    def test_nikl_source_name_is_rejected(self):
        for source in ("nikl", "NIKL_NEWSPAPER_2025_v1.1", "nikl_eval"):
            with self.assertRaises(ForbiddenSourceError):
                assert_source_allowed(source)

    def test_nikl_path_is_rejected_even_with_a_clean_source_name(self):
        with self.assertRaises(ForbiddenSourceError):
            assert_source_allowed("korean", "data/eval/nikl_eval_manifest.jsonl")

    def test_nikl_article_id_inside_a_record_is_rejected(self):
        with self.assertRaises(ForbiddenSourceError):
            assert_record_allowed("기사 NIRW2500000001.1 내용", "fineweb")

    def test_ordinary_sources_pass(self):
        assert_source_allowed("fineweb_train", "data/processed/fineweb_general_train.txt")
        assert_record_allowed("가상 메모리는 주소 공간을 제공한다.", "fineweb")

    def test_chunking_respects_the_minimum_length(self):
        chunks = chunk_text("가" * 50, target=20, minimum=40)
        self.assertTrue(all(len(c) >= 40 for c in chunks))

    def test_line_key_ignores_whitespace_and_case(self):
        self.assertEqual(line_key("Hello   World"), line_key("hello world"))


@unittest.skipUnless(EVAL_FILE.exists() and TRAIN_FILE.exists(), "recovery data not built")
class RecoveryDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eval_records = load_eval(EVAL_FILE)
        cls.train_records = [json.loads(l) for l in TRAIN_FILE.open(encoding="utf-8") if l.strip()]

    def test_eval_set_covers_every_planned_category(self):
        categories = {record["category"] for record in self.eval_records}
        self.assertEqual(categories, {
            "korean_general", "korean_technical", "dialogue",
            "json_structured", "code_url", "sql", "skill_dsl",
        })

    def test_eval_set_is_large_enough_to_be_stable(self):
        total = sum(record["characters"] for record in self.eval_records)
        self.assertGreater(total, 500_000)

    def test_eval_category_shares_match_the_plan(self):
        expected = {
            "korean_general": 0.40, "korean_technical": 0.20, "dialogue": 0.10,
            "json_structured": 0.10, "code_url": 0.10, "sql": 0.05, "skill_dsl": 0.05,
        }
        total = sum(record["characters"] for record in self.eval_records)
        for category, share in expected.items():
            actual = sum(r["characters"] for r in self.eval_records if r["category"] == category) / total
            self.assertAlmostEqual(actual, share, places=2, msg=category)

    def test_no_leakage_between_recovery_train_and_eval(self):
        eval_keys = {line_key(record["text"]) for record in self.eval_records}
        train_keys = {line_key(record["text"]) for record in self.train_records}
        self.assertEqual(eval_keys & train_keys, set())

    def test_no_nikl_record_reached_either_split(self):
        for records in (self.eval_records, self.train_records):
            for record in records:
                assert_source_allowed(record["source"])
                assert_record_allowed(record["text"], record["source"])

    def test_skill_dsl_records_use_real_control_tokens(self):
        skill = [r for r in self.eval_records if r["category"] == "skill_dsl"]
        self.assertGreater(len(skill), 100)
        self.assertTrue(any(token in r["text"] for r in skill for token in CONTROL_TOKENS))

    def test_records_are_deterministically_identified(self):
        ids = [record["id"] for record in self.eval_records]
        self.assertEqual(len(ids), len(set(ids)))


# --------------------------------------------------------------------------- #
# Training mechanics
# --------------------------------------------------------------------------- #

class TinyModel(torch.nn.Module):
    """Minimal tied-embedding causal LM standing in for Qwen3."""

    def __init__(self, vocab: int = 12, hidden: int = 4):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, hidden)
        self.block = torch.nn.Linear(hidden, hidden)
        self.head = torch.nn.Linear(hidden, vocab, bias=False)
        self.head.weight = self.embed.weight  # tied

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids, labels=None, **kwargs):
        hidden = self.block(self.embed(input_ids))
        logits = self.head(hidden)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1)
            )
        return type("Out", (), {"logits": logits, "loss": loss})()


class FreezeTests(unittest.TestCase):
    def test_only_the_tied_embedding_is_trainable(self):
        model = TinyModel()
        setup = freeze_transformer(model)
        self.assertTrue(setup["tied"])
        self.assertEqual(setup["unique_trainable_tensors"], 1)
        self.assertEqual(setup["trainable_parameters"], 12 * 4)
        self.assertFalse(model.block.weight.requires_grad)
        self.assertFalse(model.block.bias.requires_grad)

    def test_tied_head_does_not_appear_as_a_second_parameter(self):
        model = TinyModel()
        setup = freeze_transformer(model)
        self.assertEqual(setup["trainable_tensors"], setup["unique_trainable_tensors"])

    def test_frozen_transformer_receives_no_gradient(self):
        model = TinyModel()
        freeze_transformer(model)
        ids = torch.randint(0, 12, (2, 6))
        model(ids, labels=ids).loss.backward()
        self.assertIsNone(model.block.weight.grad)
        self.assertIsNotNone(model.embed.weight.grad)

    def test_row_mask_zeroes_the_gradient_of_protected_rows(self):
        model = TinyModel()
        freeze_transformer(model)
        trainable = [1, 3, 5]
        install_row_mask(model.embed.weight, trainable, "cpu")
        ids = torch.randint(0, 12, (4, 8))
        model(ids, labels=ids).loss.backward()
        grad = model.embed.weight.grad
        for row in range(12):
            if row in trainable:
                continue
            self.assertEqual(float(grad[row].abs().sum()), 0.0, f"row {row} leaked gradient")
        self.assertGreater(float(grad[torch.tensor(trainable)].abs().sum()), 0.0)

    def test_masked_rows_do_not_move_under_adamw(self):
        model = TinyModel()
        setup = freeze_transformer(model)
        install_row_mask(model.embed.weight, [1, 3, 5], "cpu")
        before = model.embed.weight.detach().clone()
        optimizer = torch.optim.AdamW([model.embed.weight], lr=0.1, weight_decay=0.0)
        for _ in range(3):
            ids = torch.randint(0, 12, (4, 8))
            model(ids, labels=ids).loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        after = model.embed.weight.detach()
        protected = [row for row in range(12) if row not in (1, 3, 5)]
        torch.testing.assert_close(after[protected], before[protected])
        self.assertGreater(float((after[[1, 3, 5]] - before[[1, 3, 5]]).abs().sum()), 0.0)

    def test_all_rows_mode_moves_every_row(self):
        model = TinyModel()
        freeze_transformer(model)
        before = model.embed.weight.detach().clone()
        optimizer = torch.optim.AdamW([model.embed.weight], lr=0.1, weight_decay=0.0)
        for _ in range(3):
            ids = torch.arange(12).view(2, 6)
            model(ids, labels=ids).loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        moved = (model.embed.weight.detach() - before).abs().sum(dim=1)
        self.assertEqual(int((moved == 0).sum()), 0)

    def test_training_produces_no_nan_or_inf(self):
        model = TinyModel()
        freeze_transformer(model)
        optimizer = torch.optim.AdamW([model.embed.weight], lr=0.1, weight_decay=0.0)
        for _ in range(5):
            ids = torch.randint(0, 12, (4, 8))
            loss = model(ids, labels=ids).loss
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        self.assertTrue(torch.isfinite(model.embed.weight).all())


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 11 + 1 for c in text]}


class BlockTests(unittest.TestCase):
    def test_blocks_are_full_so_no_padding_is_needed(self):
        records = [{"text": "abcdefghij"} for _ in range(20)]
        blocks = build_blocks(FakeTokenizer(), records, sequence_length=16, seed=42, separator_id=0)
        self.assertEqual(blocks.shape[1], 16)
        self.assertGreater(blocks.shape[0], 0)

    def test_documents_are_separated_by_an_explicit_token(self):
        records = [{"text": "abcd"} for _ in range(40)]
        blocks = build_blocks(FakeTokenizer(), records, sequence_length=10, seed=42, separator_id=0)
        self.assertIn(0, blocks.flatten().tolist())

    def test_block_construction_is_deterministic_for_a_fixed_seed(self):
        records = [{"text": f"doc{i}xyz"} for i in range(50)]
        first = build_blocks(FakeTokenizer(), records, 8, 42, 0)
        second = build_blocks(FakeTokenizer(), records, 8, 42, 0)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, build_blocks(FakeTokenizer(), records, 8, 7, 0)))


class BootstrapTests(unittest.TestCase):
    def test_paired_bootstrap_detects_a_consistent_difference(self):
        records = [{"characters": 100} for _ in range(60)]
        a = {i: 50.0 for i in range(60)}
        b = {i: 40.0 for i in range(60)}
        result = bootstrap_difference(records, a, b, iterations=300, seed=42)
        self.assertAlmostEqual(result["delta_nll_per_character"], 0.1, places=6)
        self.assertTrue(result["significant"])

    def test_paired_bootstrap_reports_no_difference_when_there_is_none(self):
        records = [{"characters": 100} for _ in range(60)]
        a = {i: 50.0 + (i % 5) for i in range(60)}
        result = bootstrap_difference(records, a, dict(a), iterations=300, seed=42)
        self.assertFalse(result["significant"])


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #

@unittest.skipUnless(RECOVERY_REPORT.exists(), "recovery report not generated")
class RecoveryReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads(RECOVERY_REPORT.read_text(encoding="utf-8"))

    def test_all_three_initializations_ran_the_same_budget(self):
        runs = self.report["phase1"]["runs"]
        self.assertEqual(len(runs), 3)
        budgets = {run["corpus"]["tokens_trained"] for run in runs.values()}
        self.assertEqual(len(budgets), 1)

    def test_every_run_kept_the_transformer_frozen(self):
        for name, run in self.report["phase1"]["runs"].items():
            self.assertEqual(run["parameters"]["unique_trainable_tensors"], 1, name)
            self.assertEqual(run["parameters"]["trainable_parameters"], 32000 * 1024, name)
            self.assertEqual(run["parameters"]["total_parameters"], 473_235_456, name)

    def test_decomposed_only_runs_leave_exact_rows_untouched(self):
        for name, run in self.report["phase1"]["runs"].items():
            if run["mode"] != DECOMPOSED_ONLY:
                continue
            drift = run["history"][-1]["exact_row_drift"]
            self.assertEqual(drift["l2_max"], 0.0, name)

    def test_recovery_improved_every_initialization(self):
        for name, run in self.report["phase1"]["runs"].items():
            history = run["history"]
            self.assertLess(history[-1]["nll_per_character"], history[0]["nll_per_character"], name)

    def test_a_winner_was_selected_with_a_recovery_ratio(self):
        self.assertIn(self.report["phase1"]["winner"], INIT_DIRS)
        self.assertIn("recovery_ratio", self.report["final"])


@unittest.skipUnless(STAGE1.is_dir(), "stage-1 checkpoint not saved")
class Stage1CheckpointTests(unittest.TestCase):
    def test_structure_is_unchanged_by_recovery(self):
        config = json.loads((STAGE1 / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["vocab_size"], 32000)
        self.assertEqual(config["hidden_size"], 1024)
        self.assertTrue(config["tie_word_embeddings"])

    def test_reloads_with_a_tied_head_and_finite_weights(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(STAGE1)
        model = AutoModelForCausalLM.from_pretrained(STAGE1, dtype="auto")
        self.assertEqual(len(tokenizer), 32000)
        embedding = model.get_input_embeddings().weight
        head = model.get_output_embeddings().weight
        self.assertEqual(embedding.data_ptr(), head.data_ptr())
        self.assertEqual(tuple(embedding.shape), (32000, 1024))
        self.assertTrue(torch.isfinite(embedding.float()).all())
        self.assertEqual(sum(p.numel() for p in model.parameters()), 473_235_456)

    def test_forward_pass_is_finite_on_every_category(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(STAGE1)
        model = AutoModelForCausalLM.from_pretrained(STAGE1, dtype=torch.float32)
        model.eval()
        for text in ("가상 메모리는 운영체제에서", "<검색>Apple Silicon 통합 메모리",
                     '{"skill":"search"}', "SELECT id FROM users;"):
            encoded = tokenizer(text, return_tensors="pt")
            with torch.inference_mode():
                output = model(**encoded, labels=encoded["input_ids"])
            self.assertTrue(torch.isfinite(output.logits).all(), text)
            self.assertTrue(torch.isfinite(output.loss), text)
            self.assertEqual(output.logits.shape[-1], 32000)


if __name__ == "__main__":
    unittest.main()
