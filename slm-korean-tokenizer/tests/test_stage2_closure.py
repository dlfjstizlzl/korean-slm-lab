"""Stage-2 closure: metric definitions, validators, NIKL plumbing, report and artifacts.

Pure-function tests run on CPU in well under a second each. Artifact tests skip
when the corresponding closure output has not been produced yet.
"""
from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "corpus"))

import generation_eval as ge  # noqa: E402
import nikl_model_eval as nme  # noqa: E402
import stage2_final_report as sfr  # noqa: E402
from recovery_eval import evaluate  # noqa: E402

CANONICAL = ROOT / "outputs" / "models" / "qwen3-0.6b-ko32k-stage2-signal2678"
FV = ROOT / "outputs" / "final_validation"


# --------------------------------------------------------------------------- #
# A uniform "model" makes every token cost exactly log(V) nats.
# --------------------------------------------------------------------------- #

class UniformModel(torch.nn.Module):
    def __init__(self, vocab: int):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=vocab)
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask=None):
        batch, length = input_ids.shape
        return SimpleNamespace(logits=torch.zeros(batch, length, self.config.vocab_size))


class CharTokenizer:
    """One token per character, like a tokenizer that never merges."""

    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=True, **kwargs):
        return {"input_ids": [1 + (ord(c) % 50) for c in text]}


class PairTokenizer(CharTokenizer):
    """One token per two characters: same text, different segmentation."""

    def __call__(self, text, add_special_tokens=True, **kwargs):
        return {"input_ids": [1 + (ord(text[i]) % 50) for i in range(0, len(text), 2)]}


def records(*texts, category="c"):
    return [{"text": t, "characters": len(t), "category": category} for t in texts]


class NllPerCharacterDenominatorTests(unittest.TestCase):
    V = 64

    def run_eval(self, tokenizer, recs, **kwargs):
        return evaluate(UniformModel(self.V), tokenizer, recs, device="cpu", **kwargs)

    def test_numerator_excludes_first_token_and_denominator_counts_all_characters(self):
        result = self.run_eval(CharTokenizer(), records("가나다라마", "abc"))
        expected = (4 + 2) * math.log(self.V) / 8
        self.assertAlmostEqual(result["overall"]["nll_per_character"], expected, places=6)
        self.assertEqual(result["overall"]["characters"], 8)
        self.assertEqual(result["overall"]["predicted_tokens"], 6)

    def test_denominator_is_independent_of_the_tokenizer(self):
        a = self.run_eval(CharTokenizer(), records("가나다라마바"))
        b = self.run_eval(PairTokenizer(), records("가나다라마바"))
        self.assertEqual(a["overall"]["characters"], b["overall"]["characters"])
        self.assertNotEqual(a["overall"]["tokens"], b["overall"]["tokens"])

    def test_context_prefix_scores_every_content_token_and_stays_out_of_counts(self):
        result = self.run_eval(CharTokenizer(), records("가나다"), context_prefix_ids=[0])
        self.assertAlmostEqual(result["overall"]["nll"], 3 * math.log(self.V), places=5)
        self.assertEqual(result["overall"]["tokens"], 3)
        self.assertEqual(result["overall"]["characters"], 3)

    def test_default_scores_long_records_in_full(self):
        text = "가" * 40
        full = self.run_eval(CharTokenizer(), records(text), max_length=10)
        self.assertEqual(full["records_over_max_length"], 1)
        self.assertEqual(full["truncated_records"], 0)
        self.assertAlmostEqual(full["overall"]["nll"], 39 * math.log(self.V), places=4)

    def test_legacy_truncation_loses_nll_but_not_characters(self):
        legacy = self.run_eval(CharTokenizer(), records("가" * 40), max_length=10, truncate=True)
        self.assertEqual(legacy["truncated_records"], 1)
        self.assertAlmostEqual(legacy["overall"]["nll"], 9 * math.log(self.V), places=4)
        self.assertEqual(legacy["overall"]["characters"], 40)

    def test_padding_does_not_contribute(self):
        together = self.run_eval(CharTokenizer(), records("가나", "가나다라마바사"), batch_size=8)
        apart = [self.run_eval(CharTokenizer(), records(t), batch_size=1) for t in ("가나", "가나다라마바사")]
        self.assertAlmostEqual(together["overall"]["nll"], sum(r["overall"]["nll"] for r in apart), places=4)

    def test_width_bucketing_does_not_change_nll(self):
        recs = records("가나다", "가나다라마바사아자", "가")
        plain = self.run_eval(CharTokenizer(), recs)
        bucketed = self.run_eval(CharTokenizer(), recs, pad_to_multiple=16)
        self.assertAlmostEqual(plain["overall"]["nll"], bucketed["overall"]["nll"], places=4)
        self.assertEqual(plain["overall"]["predicted_tokens"], bucketed["overall"]["predicted_tokens"])

    def test_category_key_is_configurable(self):
        recs = [{"text": "가나다", "characters": 3, "topic": "경제"}]
        result = self.run_eval(CharTokenizer(), recs, category_key="topic")
        self.assertEqual(list(result["by_category"]), ["경제"])


# --------------------------------------------------------------------------- #
# Repetition metrics
# --------------------------------------------------------------------------- #

class RepetitionMetricTests(unittest.TestCase):
    def test_historical_keys_are_unchanged(self):
        metrics = ge.repetition_metrics([1, 2, 3, 1, 2, 3, 1, 2], "abc")
        for key in ("new_tokens", "distinct_tokens", "distinct_ratio", "longest_single_token_run",
                    "duplicate_4gram_fraction", "duplicate_8gram_fraction", "replacement_chars"):
            self.assertIn(key, metrics)
        self.assertAlmostEqual(metrics["duplicate_4gram_fraction"], 1 - 3 / 5)
        self.assertEqual(metrics["distinct_ratio"], 3 / 8)

    def test_stage1_extension_report_uses_the_single_implementation(self):
        import stage1_extension_report
        self.assertIs(stage1_extension_report.repetition_metrics, ge.repetition_metrics)
        source = (ROOT / "model" / "stage1_extension_report.py").read_text(encoding="utf-8")
        self.assertNotIn("def repetition_metrics", source)

    def test_longest_repeated_span(self):
        self.assertEqual(ge.longest_repeated_span([1, 2, 3, 9, 1, 2, 3]), 3)
        self.assertEqual(ge.longest_repeated_span([1, 2, 3, 4]), 0)
        self.assertEqual(ge.longest_repeated_span([5, 5, 5, 5]), 3)  # overlapping occurrences

    def test_distinct_2(self):
        metrics = ge.repetition_metrics([1, 2, 1, 2, 1], "x")
        self.assertAlmostEqual(metrics["distinct_2"], 2 / 4)

    def test_loop_detection_fires_on_a_phrase_loop(self):
        text = "시작 문장입니다. " + "그리고 어떤 것이 가장 많이 사용되는 것인가? " * 5
        metrics = ge.repetition_metrics(list(range(40)), text)
        self.assertTrue(metrics["loop_detected"])
        self.assertGreaterEqual(metrics["loop_copies"], 3)

    def test_loop_detection_tolerates_a_cut_final_copy(self):
        text = "도입 " + "abc def " * 6 + "abc d"
        self.assertTrue(ge.repetition_metrics([1], text)["loop_detected"])

    def test_no_loop_on_diverse_text(self):
        text = "가상 메모리는 운영체제가 제공하는 추상화로, 프로세스마다 독립된 주소 공간을 보장한다."
        self.assertFalse(ge.repetition_metrics(list(range(20)), text)["loop_detected"])

    def test_whitespace_runs_are_not_loops(self):
        self.assertFalse(ge.repetition_metrics([1], "문장" + " " * 50)["loop_detected"])

    def test_trailing_replacement_char_is_not_an_integrity_failure(self):
        self.assertTrue(ge.repetition_metrics([1], "정상 텍스트�")["unicode_ok"])
        self.assertFalse(ge.repetition_metrics([1], "깨진�텍스트")["unicode_ok"])
        self.assertFalse(ge.repetition_metrics([1], "제어\x07문자")["unicode_ok"])

    def test_empty_generation(self):
        self.assertTrue(ge.repetition_metrics([], "  \n")["empty"])


class FailureSignatureTests(unittest.TestCase):
    STAGE1 = json.loads((ROOT / "outputs" / "stage1_extension_report.json").read_text(encoding="utf-8")) \
        if (ROOT / "outputs" / "stage1_extension_report.json").exists() else None

    def test_recorded_stage1_failures_are_detected(self):
        if not self.STAGE1:
            self.skipTest("stage1 report missing")
        skill = self.STAGE1["generation"]["stage1"]["skill"]["text"]
        json_text = self.STAGE1["generation"]["stage1"]["json"]["text"]
        self.assertTrue(ge.FAILURE_SIGNATURES["repeated_10000"].search(skill))
        self.assertTrue(ge.FAILURE_SIGNATURES["repeated_input_sql"].search(json_text))

    def test_clean_text_has_no_signature(self):
        sig = ge.failure_signatures("p", '{"skill":"sql","input":"월 매출"} 10000원', {"loop_detected": False})
        self.assertFalse(any(sig.values()))


# --------------------------------------------------------------------------- #
# Structured validity
# --------------------------------------------------------------------------- #

class StructuredValidityTests(unittest.TestCase):
    def test_json_valid_with_trailing_text(self):
        result = ge.json_validity('{"a": ', '1, "b": [2]}\n다음 문장', {"type": "object", "required": ["a", "b"]})
        self.assertTrue(result["json_parse_valid"])
        self.assertTrue(result["json_schema_valid"])

    def test_json_invalid_loop(self):
        result = ge.json_validity('{"skill":"search","input":"', 'sql","input":"sql","input":"sql",', None)
        self.assertFalse(result["json_parse_valid"])

    def test_json_schema_enum_and_required(self):
        schema = {"type": "object", "required": ["skill", "input"], "enum": {"skill": ["search", "sql"]}}
        self.assertFalse(ge.json_validity('{"skill": "', 'fly", "input": "x"}', schema)["json_schema_valid"])
        self.assertFalse(ge.json_validity('{"skill": "', 'sql"}', schema)["json_schema_valid"])
        self.assertTrue(ge.json_validity('{"skill": "', 'sql", "input": "x"}', schema)["json_schema_valid"])

    def test_json_array_schema(self):
        self.assertTrue(ge.json_validity("[1, ", "2]", {"type": "array"})["json_schema_valid"])
        self.assertFalse(ge.json_validity("{", '"a": 1}', {"type": "array"})["json_schema_valid"])

    def test_sql_name_errors_count_as_syntactically_valid(self):
        self.assertTrue(ge.sql_validity("-- c\nSELECT", " COUNT(*) FROM users WHERE y = 2024;")["sql_syntax_valid"])

    def test_sql_parse_errors_are_invalid(self):
        self.assertFalse(ge.sql_validity("SELECT name FROM t WHERE", " price > AND")["sql_syntax_valid"])
        self.assertFalse(ge.sql_validity("SELECT", " ")["sql_syntax_valid"])

    def test_sql_statement_stops_at_semicolon_or_blank_line(self):
        self.assertEqual(ge.first_sql_statement("SELECT", " a FROM t;\n잡담"), "SELECT a FROM t")
        self.assertEqual(ge.first_sql_statement("SELECT", " a FROM t\n\n설명"), "SELECT a FROM t")

    def test_dsl_valid_lines(self):
        for line in ("<검색>서울 날씨", "<요약><이전0>", "<요약><이전1><종료>", "<종료>",
                     '{"skill":"sql","input":"x"}', "오류: timeout"):
            self.assertTrue(ge.dsl_line_valid(line), line)
        for line in ("AppliSil 10000 10000", "<검색", '{"skill":"fly"}', "그냥 문장"):
            self.assertFalse(ge.dsl_line_valid(line), line)

    def test_grammar_accepts_every_v3_generator_form(self):
        """Recall check against the source of truth: the v3 Skill generator output."""
        path = ROOT / "data" / "processed" / "stage2_train_v3.jsonl"
        if not path.exists():
            self.skipTest("stage2 corpus not present")
        rejected = checked = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("category") != "skill_manager":
                    continue
                checked += 1
                rejected += not ge.skill_dsl_validity("", record["text"], complete=True)["dsl_all_lines_valid"]
        self.assertGreater(checked, 20_000)
        self.assertEqual(rejected, 0)

    def test_generator_line_forms(self):
        for line in ("<SQL>SELECT a FROM t WHERE x >= 3;<요약>step1<검색>step2<종료>",
                     "1. <SQL> SELECT * FROM t WHERE a < 3;", "2. <검색> step1 결과 사용",
                     "확인된 내용만 요약해 회신한다. <종료>", "목표: 서울 날씨", "계획: <검색> 서울",
                     "결과: step1 완료", "<이전2>", "재시도: <코드>파일 읽기"):
            self.assertTrue(ge.dsl_line_valid(line), line)
        for line in ("텍스트<검색>중간", "<종료> 뒤 텍스트", "1995년 1월 1일, 1995년 1월 1일,"):
            self.assertFalse(ge.dsl_line_valid(line), line)

    def test_multiline_json_plan_is_one_unit(self):
        text = '{\n  "goal": "x",\n  "plan": [{"step": 1, "skill": "search", "input": "q"}]\n}\n'
        result = ge.skill_dsl_validity("<이전0>\n", text)
        self.assertEqual(result["dsl_lines"], 1)
        self.assertTrue(result["dsl_all_lines_valid"])

    def test_prompt_lines_are_never_judged(self):
        result = ge.skill_dsl_validity("<검색>제주도 여행\n", "AppliSil 10000\n")
        self.assertEqual(result["dsl_lines"], 1)
        self.assertFalse(result["dsl_first_line_valid"])

    def test_prompt_partial_line_counts_only_when_extended(self):
        result = ge.skill_dsl_validity("<검색>Apple Silicon", "\nAppliSil 10000 10000\n")
        self.assertFalse(result["dsl_first_line_valid"])
        result = ge.skill_dsl_validity("<코드>", "파이썬 파일 읽기\n<요약><이전0>\n")
        self.assertTrue(result["dsl_all_lines_valid"])

    def test_unfinished_last_line_is_dropped_unless_stopped(self):
        self.assertEqual(ge.skill_dsl_validity("<검색>a\n", "<요약><이전0>\n<요")["dsl_lines"], 1)
        self.assertEqual(ge.skill_dsl_validity("<검색>a\n", "<요약><이전0>", complete=True)["dsl_lines"], 1)

    def test_instruction_check_uses_first_line(self):
        self.assertTrue(ge.instruction_check(" 서울입니다.\n부산", "서울")["instruction_match_first_line"])
        self.assertFalse(ge.instruction_check(" 모르겠다\n서울", "서울")["instruction_match_first_line"])


class PromptSetTests(unittest.TestCase):
    def setUp(self):
        self.prompts = ge.load_prompts(ROOT / ge.PROMPTS_FILE)

    def test_prompt_set_is_versioned_and_complete(self):
        ids = [p["id"] for p in self.prompts]
        self.assertEqual(len(ids), len(set(ids)))
        categories = {p["category"] for p in self.prompts}
        self.assertEqual(categories, {"legacy_stage1", "korean_general", "korean_technical",
                                      "dialogue_summary", "json_structured", "sql", "skill_dsl",
                                      "instruction"})

    def test_legacy_prompts_are_verbatim(self):
        import stage1_extension_report
        legacy = {p["prompt"] for p in self.prompts if p["category"] == "legacy_stage1"}
        self.assertEqual(legacy, set(stage1_extension_report.PROMPTS.values()))

    def test_prompts_are_not_copied_from_the_stage2_corpus(self):
        """Content prompts (any Hangul) must not occur in training text.

        Pure-ASCII format scaffolds are exempt: json_02 (`{"skill": "summary", "input": "`)
        is the Skill JSON schema itself and occurs in 617 skill_manager documents.
        """
        path = ROOT / "data" / "processed" / "stage2_train_v3.jsonl"
        if not path.exists():
            self.skipTest("stage2 corpus not present")
        probes = [p["prompt"].strip() for p in self.prompts
                  if p["category"] != "legacy_stage1" and len(p["prompt"].strip()) >= 12
                  and any("가" <= c <= "힣" for c in p["prompt"])]
        self.assertGreater(len(probes), 25)
        hits = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = json.loads(line)["text"]
                hits += [p for p in probes if p in text]
        self.assertEqual(hits, [])


class DeterministicGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not CANONICAL.exists():
            raise unittest.SkipTest("canonical checkpoint missing")
        from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
        cls.tokenizer = AutoTokenizer.from_pretrained(CANONICAL)
        torch.manual_seed(0)
        config = Qwen3Config(vocab_size=32000, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                             num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                             tie_word_embeddings=True)
        cls.model = Qwen3ForCausalLM(config).eval()

    def test_greedy_is_deterministic(self):
        record = {"id": "t", "category": "korean_general", "prompt": "가상 메모리는"}
        a = ge.generate_one(self.model, self.tokenizer, record, "cpu", ge.GREEDY, None, 12)
        b = ge.generate_one(self.model, self.tokenizer, record, "cpu", ge.GREEDY, None, 12)
        self.assertEqual(a["generated"], b["generated"])

    def test_seeded_sampling_is_reproducible(self):
        record = {"id": "t", "category": "sql", "prompt": "SELECT"}
        a = ge.generate_one(self.model, self.tokenizer, record, "cpu", ge.SAMPLING, 1, 12)
        b = ge.generate_one(self.model, self.tokenizer, record, "cpu", ge.SAMPLING, 1, 12)
        self.assertEqual(a["generated"], b["generated"])
        self.assertIn("sql_syntax_valid", a["structured"])

    def test_stop_tokens_resolve_to_the_same_strings(self):
        self.assertEqual(ge.stop_token_ids(self.tokenizer), [2, 0])


# --------------------------------------------------------------------------- #
# NIKL plumbing
# --------------------------------------------------------------------------- #

class NiklPlumbingTests(unittest.TestCase):
    def test_selection_is_deterministic_and_nested(self):
        ids = [f"NIRW2500000001.{i}" for i in range(5000)]
        one = {i for i in ids if nme.selection_bucket(i) < 100}
        quarter = {i for i in ids if nme.selection_bucket(i) < 25}
        self.assertTrue(quarter <= one)
        self.assertEqual(one, {i for i in ids if nme.selection_bucket(i) < 100})
        self.assertTrue(20 <= len(one) <= 90)

    def test_length_buckets_match_the_tokenizer_benchmark(self):
        from build_nikl_eval import length_bucket as benchmark_bucket
        for n in (0, 499, 500, 999, 1000, 1999, 2000, 3999, 4000, 7541):
            self.assertEqual(nme.length_bucket(n), benchmark_bucket(n))

    def test_summary_arithmetic(self):
        rows = [{"id": "a", "characters": 100, "nll": 150.0, "tokens": 40, "paragraphs": 2, "topic": "x"},
                {"id": "b", "characters": 300, "nll": 450.0, "tokens": 120, "paragraphs": 5, "topic": "y"}]
        s = nme.summarize_rows(rows)
        self.assertAlmostEqual(s["nll_per_character"], 1.5)
        self.assertAlmostEqual(s["bits_per_character"], 1.5 / math.log(2))
        self.assertAlmostEqual(s["tokens_per_character"], 0.4)
        self.assertEqual(s["paragraphs"], 7)
        self.assertEqual(set(nme.grouped(rows, "topic")), {"x", "y"})

    def test_paired_bootstrap_sign_and_interval(self):
        a = [{"id": str(i), "nll": 1.0 * (i % 7 + 10), "characters": 10} for i in range(200)]
        b = [{"id": str(i), "nll": r["nll"] + 2.0, "characters": 10} for i, r in enumerate(a)]
        result = nme.paired_bootstrap(a, b, iterations=200)
        self.assertAlmostEqual(result["delta_nll_per_character"], -0.2)
        self.assertLessEqual(result["ci95_low"], -0.2 + 1e-9)
        self.assertGreaterEqual(result["ci95_high"], -0.2 - 1e-9)

    def test_manifest_verification_rejects_mismatched_text(self):
        from nikl import Article
        article = Article("X.1", "사회", "", "p", "20240101", "제목", "본문 내용")
        manifest = {"X.1": {"chars": 99, "text_hash": "0", "topic": "사회", "publisher": "p"}}
        original = nme.iter_articles
        nme.iter_articles = lambda paths: iter([article])
        try:
            with self.assertRaises(RuntimeError):
                list(nme.iter_selected(manifest, 1.0, "unused"))
        finally:
            nme.iter_articles = original

    def test_nikl_is_never_a_training_corpus(self):
        for path in nme.MODEL_TRAINING_CORPORA.values():
            self.assertNotIn("nikl", path.lower())
        from build_recovery_data import ForbiddenSourceError, assert_source_allowed
        with self.assertRaises(ForbiddenSourceError):
            assert_source_allowed("data/NIKL_NEWSPAPER_2025_v1.1/x.json")


# --------------------------------------------------------------------------- #
# Checkpoint consistency (JSON only, no weights)
# --------------------------------------------------------------------------- #

class CheckpointConsistencyTests(unittest.TestCase):
    def setUp(self):
        if not CANONICAL.exists():
            self.skipTest("canonical checkpoint missing")

    def test_tokenizer_and_model_vocab_agree(self):
        from transformers import AutoTokenizer
        config = json.loads((CANONICAL / "config.json").read_text(encoding="utf-8"))
        tokenizer = AutoTokenizer.from_pretrained(CANONICAL)
        self.assertEqual(config["vocab_size"], 32000)
        self.assertEqual(len(tokenizer), 32000)
        self.assertTrue(config["tie_word_embeddings"])

    def test_no_special_token_id_outside_the_vocabulary_in_any_checkpoint(self):
        audit = sfr.special_token_audit()
        self.assertGreater(len(audit["checkpoints"]), 0)
        self.assertEqual(audit["checkpoints_with_invalid_ids"], [])

    def test_control_tokens_are_atomic(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(CANONICAL)
        for token in ge.CONTROL_TOKENS:
            self.assertEqual(len(tokenizer(token, add_special_tokens=False)["input_ids"]), 1, token)

    def test_canonical_metadata_fixes_are_in_place(self):
        tokenizer = json.loads((CANONICAL / "tokenizer.json").read_text(encoding="utf-8"))
        generation = json.loads((CANONICAL / "generation_config.json").read_text(encoding="utf-8"))
        config = json.loads((CANONICAL / "config.json").read_text(encoding="utf-8"))
        self.assertIsNone(tokenizer["truncation"])
        self.assertEqual(generation["eos_token_id"], [2, 0])
        self.assertTrue(config["use_cache"])
        self.assertNotIn(151643, [config.get("bos_token_id"), generation.get("bos_token_id")])

    def test_training_state_matches_the_declared_stop_point(self):
        state = json.loads((CANONICAL / "stage2_state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["step"], 2678)
        self.assertEqual(state["tokens_seen"], 43_876_352)
        self.assertEqual(state["tokens_seen"], 2678 * 16_384)


# --------------------------------------------------------------------------- #
# Report generation
# --------------------------------------------------------------------------- #

def fake_gen_summary(stage2_loop=0.1, qwen_loop=0.1, json_rate=1.0, sql_rate=1.0, dsl=1.0, sig=0):
    def block(loop, j, s, d, signature):
        overall = {"samples": 39, "loop_detected_rate": loop, "empty_rate": 0.0, "unicode_ok_rate": 1.0,
                   "failure_signature_counts": {"repeated_10000": signature, "repeated_input_sql": 0}}
        cats = {"json_structured": {"samples": 5, "json_parse_valid_rate": j},
                "sql": {"samples": 5, "sql_syntax_valid_rate": s},
                "skill_dsl": {"samples": 5, "dsl_first_line_valid_rate": d}}
        return {"greedy": {"overall": overall, "by_category": cats},
                "sampled": {"overall": dict(overall), "by_category": cats}}
    return {"summary": {"original_qwen": block(qwen_loop, 1.0, 1.0, 0.0, 0),
                        "stage1_final": block(0.5, 0.2, 0.2, 0.0, 1),
                        "stage2_signal2678": block(stage2_loop, json_rate, sql_rate, dsl, sig)}}


def gate_context(**overrides):
    checks = {f"{s}/{m}": {"within_tolerance": True} for s in ("balanced", "korean")
              for m in ("original_qwen", "stage1_final", "stage2_signal2678")}
    ctx = {
        "smoke": {"all_pass": True, "checks": {"reload_identical": True, "no_nan_inf": True,
                                               "no_invalid_special_ids": True, "control_tokens_ok": True}},
        "reproduction": {"reproduction_checks": checks},
        "nikl_full": {"complete": True, "same_articles_for_all_models": True,
                      "overall": {"original_qwen": {"articles": 1_000}}},
        "nikl_manifest_articles": 1_000,
        "generation": fake_gen_summary(),
        "special_audit": {"checkpoints_with_invalid_ids": []},
        "artifacts_present": {"x": True},
        "pytest": {"tests": 10, "failures": 0, "errors": 0, "skipped": 0},
        "nikl_provenance": {"ok": 1}, "canonical": {"ok": 1},
    }
    ctx.update(overrides)
    return ctx


class GateTests(unittest.TestCase):
    def test_all_criteria_pass(self):
        decision = sfr.gate(gate_context())
        self.assertTrue(decision["pass"], decision["blockers"])
        self.assertEqual(decision["verdict"], "Stage-2 closed — ready for Manager-SLM dataset/SFT design")
        self.assertEqual(len(decision["criteria"]), 10)

    def test_incomplete_nikl_blocks(self):
        decision = sfr.gate(gate_context(nikl_full=None))
        self.assertFalse(decision["pass"])
        self.assertEqual(decision["blockers"], ["D_nikl_full_model_evaluation_completed"])

    def test_reproduction_miss_blocks(self):
        ctx = gate_context()
        ctx["reproduction"]["reproduction_checks"]["korean/stage2_signal2678"]["within_tolerance"] = False
        self.assertIn("C_korean_large_eval_reproducible", sfr.gate(ctx)["blockers"])

    def test_loop_margin_and_stage1_signature_block(self):
        self.assertIn("E_no_severe_systematic_generation_collapse",
                      sfr.gate(gate_context(generation=fake_gen_summary(stage2_loop=0.5)))["blockers"])
        self.assertIn("E_no_severe_systematic_generation_collapse",
                      sfr.gate(gate_context(generation=fake_gen_summary(sig=1)))["blockers"])

    def test_structured_collapse_blocks(self):
        self.assertIn("G_no_catastrophic_structured_collapse",
                      sfr.gate(gate_context(generation=fake_gen_summary(json_rate=0.4)))["blockers"])
        self.assertIn("G_no_catastrophic_structured_collapse",
                      sfr.gate(gate_context(generation=fake_gen_summary(dsl=0.4)))["blockers"])

    def test_failing_tests_block(self):
        ctx = gate_context(pytest={"tests": 10, "failures": 1, "errors": 0, "skipped": 0})
        self.assertIn("I_pytest_all_pass", sfr.gate(ctx)["blockers"])

    def test_waiver_closes_without_turning_fail_into_pass(self):
        waiver = {"waived_criteria": {"D_nikl_full_model_evaluation_completed": "x"}}
        decision = sfr.gate(gate_context(nikl_full=None, closure_decision=waiver))
        self.assertFalse(decision["pass"])
        self.assertFalse(decision["criteria"]["D_nikl_full_model_evaluation_completed"]["pass"])
        self.assertTrue(decision["closed_by_decision"])
        self.assertIn("waived, not passed", decision["verdict"])

    def test_waiver_does_not_cover_other_failures(self):
        waiver = {"waived_criteria": {"D_nikl_full_model_evaluation_completed": "x"}}
        ctx = gate_context(nikl_full=None, closure_decision=waiver,
                           pytest={"tests": 10, "failures": 1, "errors": 0, "skipped": 0})
        decision = sfr.gate(ctx)
        self.assertFalse(decision["closed"])
        self.assertEqual(decision["verdict"], "Stage-2 not closed — blockers remain")

    def test_thresholds_are_the_declared_constants(self):
        self.assertEqual(sfr.GATE_E_GREEDY_LOOP_MARGIN, 0.20)
        self.assertEqual(sfr.GATE_E_SAMPLED_LOOP_MARGIN, 0.10)
        self.assertEqual(sfr.GATE_G_MIN_RATIO_TO_QWEN, 0.5)
        self.assertEqual(sfr.GATE_G_MIN_DSL_FIRST_LINE, 0.5)


class LearningCurveTests(unittest.TestCase):
    def test_curve_uses_the_recorded_history_and_final_point(self):
        training = json.loads((ROOT / "outputs" / "stage2_training_report.json").read_text(encoding="utf-8"))
        curve = sfr.learning_curve(training)
        self.assertEqual(curve["points"][0]["tokens"], 0)
        self.assertEqual(curve["points"][-1]["tokens"], 43_876_352)
        last = curve["last_interval"]
        self.assertAlmostEqual(-last["improvement"], -0.0038, places=4)
        self.assertLess(curve["halves"]["second_half_20.5M_to_41.0M"]["improvement_per_1m"],
                        curve["halves"]["first_half_0_to_20.5M"]["improvement_per_1m"])
        self.assertIn("never measured", curve["not_measured"])


class ReportArtifactTests(unittest.TestCase):
    REPORT = ROOT / "outputs" / "stage2_final_report.md"

    def setUp(self):
        if not self.REPORT.exists():
            self.skipTest("report not generated yet")
        self.text = self.REPORT.read_text(encoding="utf-8")
        self.payload = json.loads((ROOT / "outputs" / "stage2_final_report.json").read_text(encoding="utf-8"))

    def test_all_eighteen_sections_exist(self):
        for number in range(1, 19):
            self.assertRegex(self.text, rf"\n## {number}\. ")

    def test_forbidden_claims_are_absent(self):
        lowered = self.text.lower()
        for phrase in ("all capabilities improved", "english capability is fully preserved",
                       "catastrophic forgetting is impossible", "never used before",
                       "would definitely"):
            self.assertNotIn(phrase, lowered)

    def test_stop_point_is_disclosed(self):
        self.assertIn("43,876,352", self.text)
        self.assertIn("60M", self.text)

    def test_nothing_after_stage2_was_started(self):
        self.assertIn("SFT", self.payload["not_started"])
        self.assertIn("Stage-2 resume to 60M", self.payload["not_started"])


if __name__ == "__main__":
    unittest.main()
