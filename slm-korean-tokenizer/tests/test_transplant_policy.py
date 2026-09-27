from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))

from transplant_embeddings import source_ids_for_token


class FakeBase:
    unk_token_id = 99

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"새말": [2, 3], "": [], "모름": [99]}.get(text, [4])


class FakeNew:
    def decode(self, ids, **kwargs):
        del kwargs
        return {10: "같음", 11: "새말", 12: "모름", 13: "�"}[ids[0]]


class TransplantPolicyTests(unittest.TestCase):
    def test_exact_match_precedes_surface_retokenization(self):
        policy, ids, _ = source_ids_for_token(FakeBase(), FakeNew(), "same-token", 10, {"same-token": 7}, 20)
        self.assertEqual((policy, ids), ("exact", [7]))

    def test_surface_uses_average_policy(self):
        policy, ids, _ = source_ids_for_token(FakeBase(), FakeNew(), "new-token", 11, {}, 20)
        self.assertEqual((policy, ids), ("averaged", [2, 3]))

    def test_unknown_uses_random_fallback(self):
        policy, ids, _ = source_ids_for_token(FakeBase(), FakeNew(), "unknown-token", 12, {}, 20)
        self.assertEqual((policy, ids), ("random", []))

    def test_invalid_utf8_surface_uses_random_fallback(self):
        policy, ids, _ = source_ids_for_token(FakeBase(), FakeNew(), "byte-fragment", 13, {}, 20)
        self.assertEqual((policy, ids), ("random", []))


if __name__ == "__main__":
    unittest.main()
