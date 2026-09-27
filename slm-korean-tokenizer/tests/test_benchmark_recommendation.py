from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tokenizer"))

from benchmark_tokenizers import choose_balanced


class RecommendationTests(unittest.TestCase):
    def test_no_jointly_eligible_candidate_does_not_crash(self):
        results = [
            {
                "name": "low-token-high-fallback",
                "kind": "candidate",
                "control_tokens_atomic": True,
                "metrics": {"total_tokens": 100, "byte_fallback_percent": 40.0, "vocab_size": 8000},
            },
            {
                "name": "high-token-low-fallback",
                "kind": "candidate",
                "control_tokens_atomic": True,
                "metrics": {"total_tokens": 120, "byte_fallback_percent": 0.0, "vocab_size": 16000},
            },
        ]
        chosen, reason = choose_balanced(results)
        self.assertEqual(chosen, "low-token-high-fallback")
        self.assertIn("No candidate simultaneously", reason)


if __name__ == "__main__":
    unittest.main()
