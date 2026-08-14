import unittest

from prefkit.generate import make_seed_iter


class TestSeedIter(unittest.TestCase):
    def test_increments(self):
        nxt = make_seed_iter({"seed": 0})
        self.assertEqual([nxt(), nxt(), nxt()], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
