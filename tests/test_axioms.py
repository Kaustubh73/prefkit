import unittest
from pathlib import Path

import yaml

from prefkit.axioms import check_axioms, default_methods
from prefkit.outcomes import load_outcomes

ROOT = Path(__file__).resolve().parents[1]


class TestAxioms(unittest.TestCase):
    def setUp(self):
        self.outcomes = load_outcomes(ROOT / "data" / "outcomes.smoke.json")
        self.decode = yaml.safe_load((ROOT / "configs" / "decode.yaml").read_text())

    def test_all_pass(self):
        result = check_axioms(default_methods(), self.outcomes, self.decode)
        for block, flags in result.items():
            for ax, ok in flags.items():
                self.assertTrue(ok, f"{block} {ax} failed")

    def test_decode_yaml(self):
        self.assertEqual(self.decode["sample_k"], 3)
        self.assertEqual(self.decode["temperature"], 0.7)
        self.assertEqual(self.decode["top_p"], 1.0)


if __name__ == "__main__":
    unittest.main()
