"""Manager LoRA SFT pipeline tests (no sealed test data is ever read)."""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))

import manager_lora as lora  # noqa: E402
import manager_sft_data as data  # noqa: E402
from manager_sft_eval import GrammarSession, RuntimeGrammarProcessor, evaluate, token_bytes_table  # noqa: E402
from manager_v2_registry import TOOL  # noqa: E402

TOKENIZER_AVAILABLE = (data.BASE_MODEL / "tokenizer.json").is_file()
TOOL_AVAILABLE = TOOL.is_file()


def tiny_model():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8, tie_word_embeddings=True)
    return Qwen3ForCausalLM(config)


class SealedTestGuardTests(unittest.TestCase):
    def test_sealed_test_requires_explicit_flag(self):
        with self.assertRaises(data.SealedTestError):
            data.load_split("test")

    @unittest.skipUnless(data.SPLIT_FILES["test"].is_file(), "sealed test file is local-only")
    def test_fingerprint_does_not_parse_rows(self):
        fingerprint = data.sealed_test_fingerprint()
        self.assertTrue(fingerprint["exists"])
        self.assertTrue(fingerprint["matches_recorded_sha256"])
        self.assertEqual(set(fingerprint), {"path", "exists", "sha256", "matches_recorded_sha256"})

    def test_training_scripts_never_request_test(self):
        for name in ("manager_sft_train.py", "manager_sft_eval.py"):
            source = (ROOT / "model" / name).read_text(encoding="utf-8")
            self.assertNotIn('load_split("test"', source)
            self.assertNotIn("allow_sealed_test=True", source)


@unittest.skipUnless(TOKENIZER_AVAILABLE and TOOL_AVAILABLE, "base tokenizer or runtime tool unavailable")
class RenderingAndMaskingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        cls.tokenizer = AutoTokenizer.from_pretrained(data.BASE_MODEL)
        cls.rows = data.load_split("validation")[:40]
        cls.prompt = data.system_prompt()

    def test_prompt_is_runtime_prompt_and_template_parity(self):
        data.assert_template_parity(self.tokenizer, "Qwen3 최신 정보를 찾아 요약해줘")
        rendered = data.render_prompt("요청", self.prompt)
        self.assertTrue(rendered.startswith("<|im_start|>system\n" + self.prompt + "<|im_end|>\n"))
        self.assertTrue(rendered.endswith("<|im_start|>user\n요청<|im_end|>\n<|im_start|>assistant\n"))
        self.assertNotIn("<think>", rendered)

    def test_assistant_only_labels(self):
        for row in self.rows:
            enc = data.encode(row, self.tokenizer, self.prompt)
            prompt_ids = self.tokenizer(data.render_prompt(row["request"], self.prompt), add_special_tokens=False)["input_ids"]
            self.assertEqual(enc.input_ids[:len(prompt_ids)], prompt_ids)
            self.assertTrue(all(l == data.IGNORE_INDEX for l in enc.labels[:len(prompt_ids)]))
            trained = [l for l in enc.labels if l != data.IGNORE_INDEX]
            self.assertEqual(trained, enc.input_ids[len(prompt_ids):])
            self.assertEqual(trained[-1], self.tokenizer.eos_token_id)
            self.assertEqual(self.tokenizer.decode(trained[:-1]), row["plan_text"])

    def test_canonical_target_and_no_fence(self):
        for row in self.rows:
            target = data.target_text(row)
            self.assertEqual(target, row["plan_text"])
            self.assertTrue(target.startswith('{"status":'))
            self.assertNotIn("```", target)
        broken = copy.deepcopy(self.rows[0])
        broken["plan_text"] = broken["plan_text"].replace(",", ", ", 1)
        with self.assertRaises(ValueError):
            data.target_text(broken)

    def test_no_truncation_and_refusal(self):
        encoded = data.encode_all(self.rows, self.tokenizer, 4096)
        self.assertTrue(all(len(e.input_ids) <= 4096 for e in encoded))
        with self.assertRaises(ValueError):
            data.encode_all(self.rows, self.tokenizer, 1000)

    def test_deterministic_rendering(self):
        first = [data.encode(r, self.tokenizer, self.prompt).input_ids for r in self.rows[:5]]
        second = [data.encode(r, self.tokenizer, self.prompt).input_ids for r in self.rows[:5]]
        self.assertEqual(first, second)

    def test_grammar_processor_forces_valid_plan(self):
        table = token_bytes_table(self.tokenizer)
        session = GrammarSession()
        try:
            eos = {self.tokenizer.eos_token_id, self.tokenizer.pad_token_id}
            processor = RuntimeGrammarProcessor(session, table, eos, batch=1, prompt_len=1)
            ids = torch.tensor([[1]])
            generator = torch.Generator().manual_seed(0)
            for _ in range(80):
                scores = torch.randn(1, len(table), generator=generator)
                chosen = int(processor(ids, scores.clone()).argmax())
                if chosen in eos:
                    break
                ids = torch.cat([ids, torch.tensor([[chosen]])], dim=1)
            text = self.tokenizer.decode(ids[0, 1:].tolist())
            self.assertTrue(text.startswith('{"status":"'), text)
        finally:
            session.close()


class LoRATests(unittest.TestCase):
    def test_targets_are_discovered_block_linears(self):
        model = tiny_model()
        names = lora.discover_targets(model)
        self.assertTrue(names)
        self.assertTrue(all(".layers." in f".{n}" for n in names))
        self.assertNotIn("lm_head", " ".join(names))
        self.assertEqual({n.rsplit(".", 1)[-1] for n in names},
                         {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"})

    def test_base_frozen_adapter_trainable_and_identity_at_init(self):
        model = tiny_model()
        ids = torch.tensor([[1, 2, 3, 4]])
        before = model(input_ids=ids).logits
        info = lora.inject(model, rank=4, alpha=8, dropout=0.0, seed=1)
        after = model(input_ids=ids).logits
        self.assertTrue(torch.allclose(before, after), "B=0 makes the adapter an identity at init")
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        self.assertTrue(trainable)
        self.assertTrue(all(".lora_A" in n or ".lora_B" in n for n in trainable))
        self.assertEqual(info["trainable_parameters"], sum(p.numel() for p in model.parameters() if p.requires_grad))
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad, name)
            else:
                self.assertIsNone(parameter.grad, name)
        self.assertIsNone(model.get_input_embeddings().weight.grad)

    def test_adapter_save_reload_roundtrip(self):
        model = tiny_model()
        lora.inject(model, rank=4, alpha=8, dropout=0.0, seed=1)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    parameter.add_(torch.randn_like(parameter) * 0.1)
        ids = torch.tensor([[5, 6, 7]])
        model.eval()
        expected = model(input_ids=ids).logits
        with tempfile.TemporaryDirectory() as directory:
            lora.save_adapter(model, Path(directory), {"lora": {"rank": 4, "alpha": 8}})
            fresh = tiny_model()
            lora.load_adapter(fresh, Path(directory))
            self.assertTrue(torch.allclose(fresh(input_ids=ids).logits, expected, atol=1e-6))


@unittest.skipUnless(TOOL_AVAILABLE, "runtime tool unavailable")
class MetricTests(unittest.TestCase):
    def rows(self):
        gold_ok = {"status": "ok", "tasks": [{"id": 0, "skill": "search", "instruction": "A 검색", "sources": []},
                                             {"id": 1, "skill": "summary", "instruction": "A 요약", "sources": [0]}],
                   "outputs": [1]}
        gold_clarify = {"status": "clarify", "message": "요약할 내용을 제공해 주세요.", "tasks": [], "outputs": []}
        from manager_v2_validator import canonical_json

        meta = {"taxonomy": "t", "taxonomy_group": "normal", "hard_case": False}
        return [{"id": "a", "plan": gold_ok, "plan_text": canonical_json(gold_ok), "metadata": meta},
                {"id": "b", "plan": gold_clarify, "plan_text": canonical_json(gold_clarify), "metadata": meta}]

    def test_exact_structural_and_failures(self):
        rows = self.rows()
        exact = [rows[0]["plan_text"], rows[1]["plan_text"]]
        result = evaluate(rows, exact)
        self.assertEqual(result["overall"]["canonical_exact"], 1.0)
        self.assertEqual(result["overall"]["structural_exact"], 1.0)
        self.assertEqual(result["overall"]["dependency_edge_f1"], 1.0)

        wording = rows[0]["plan_text"].replace("A 요약", "A를 요약")
        wrong_source = rows[0]["plan_text"].replace('"sources":[0]', '"sources":["user"]')
        result = evaluate(rows, [wording, '{"status":"ok","tasks":[{"id":0,"skill":"summary","instruction":"x","sources":["user"]}],"outputs":[0]}'])
        records = result["records"]
        self.assertTrue(records[0]["structural_exact"])
        self.assertFalse(records[0]["canonical_exact"])
        self.assertEqual(records[0]["failures"], ["instruction_mismatch"])
        self.assertFalse(records[1]["status_correct"])
        self.assertIn("plan_instead_of_clarify", records[1]["failures"])
        self.assertEqual(result["status_recall"]["clarify"], 0.0)

        result = evaluate(rows, [wrong_source, "not json"])
        records = result["records"]
        self.assertFalse(records[0]["structural_exact"])
        self.assertTrue(records[0]["skill_sequence_correct"])
        self.assertFalse(records[0]["sources_exact"])
        self.assertIn("wrong_source", records[0]["failures"])
        self.assertEqual(records[1]["failures"], ["malformed_json"])
        overall = result["overall"]
        self.assertAlmostEqual(overall["dependency_edge_precision"], 1 / 2)
        self.assertAlmostEqual(overall["dependency_edge_recall"], 1 / 2)

    def test_status_metric(self):
        rows = self.rows()
        result = evaluate(rows, ['{"status":"clarify","message":"x","tasks":[],"outputs":[]}', rows[1]["plan_text"]])
        self.assertEqual(result["overall"]["status_accuracy"], 0.5)
        self.assertEqual(result["status_recall"]["ok"], 0.0)
        self.assertIn("clarify_instead_of_plan", result["records"][0]["failures"])


class ReportMathTests(unittest.TestCase):
    def test_exact_mcnemar(self):
        from manager_sft_report import mcnemar_p

        self.assertEqual(mcnemar_p(0, 0), 1.0)
        self.assertAlmostEqual(mcnemar_p(0, 10), 2 / 2 ** 10)
        self.assertAlmostEqual(mcnemar_p(5, 5), 1.0)
        self.assertLess(mcnemar_p(10, 60), 1e-8)


class DatasetHashTests(unittest.TestCase):
    def test_hashes_match_training_config(self):
        config_path = ROOT / "outputs" / "manager_sft_v1" / "training_config.json"
        if not config_path.is_file() or not data.SPLIT_FILES["train"].is_file():
            self.skipTest("training has not been configured or the dataset is local-only")
        provenance = json.loads(config_path.read_text())["provenance"]
        self.assertEqual(provenance["train_sha256"], data.sha256_file(data.SPLIT_FILES["train"]))
        self.assertEqual(provenance["validation_sha256"], data.sha256_file(data.SPLIT_FILES["validation"]))
        self.assertEqual(provenance["sealed_test"]["sha256"], data.SEALED_TEST_SHA256)


if __name__ == "__main__":
    unittest.main()
