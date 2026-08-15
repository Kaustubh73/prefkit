import unittest

from prefkit.thurstone import fit_thurstonian, pair_p_after_flip


class TestThurstone(unittest.TestCase):
    def test_transitive_order(self):
        # three items, deterministic chain 0 > 1 > 2
        comps = [(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)]
        mu, _var, _stats = fit_thurstonian(comps, 3, steps=250, lr=0.1)
        self.assertGreater(mu[0], mu[1])
        self.assertGreater(mu[1], mu[2])

    def test_always_half_near_equal(self):
        comps = [(0, 1, 0.5), (0, 2, 0.5), (1, 2, 0.5)]
        mu, _var, _stats = fit_thurstonian(comps, 3, steps=80, lr=0.05)
        self.assertLess(float(max(mu) - min(mu)), 0.2)

    def test_flip_from_logs(self):
        ids = ["x", "y"]
        logs = [
            {"ids": ("x", "y"), "order": "ij", "parsed": "A"},
            {"ids": ("x", "y"), "order": "ji", "parsed": "B"},
        ]
        comps = pair_p_after_flip(logs, ids)
        self.assertEqual(comps, [(0, 1, 1.0)])


if __name__ == "__main__":
    unittest.main()
