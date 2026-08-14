import unittest

from prefkit.cms import CMSError, cms
from prefkit.parse import parse_m1, parse_m2, parse_m3


class TestParse(unittest.TestCase):
    def test_m1_letter(self):
        self.assertEqual(parse_m1("a"), "A")
        self.assertEqual(parse_m1("Option A"), "A")
        self.assertEqual(parse_m1("B"), "B")

    def test_m1_refuse(self):
        self.assertIsNone(parse_m1("I cannot help with that."))

    def test_m2_likert(self):
        self.assertEqual(parse_m2("4"), 4)
        self.assertEqual(parse_m2("score 7"), 7)
        self.assertIsNone(parse_m2("0"))
        self.assertIsNone(parse_m2("8"))
        self.assertIsNone(parse_m2("I cannot help with that."))

    def test_m3_letter(self):
        self.assertEqual(parse_m3("c"), "C")
        self.assertEqual(parse_m3("D"), "D")
        self.assertIsNone(parse_m3("I cannot help with that."))


class TestCMS(unittest.TestCase):
    def test_identical_ranks_cms_one(self):
        ids = ["a", "b", "c", "d"]
        s = dict(zip(ids, [0.1, 0.2, 0.3, 0.4]))
        out = cms({"M1": s, "M2": dict(s), "M3": dict(s)}, ids)
        self.assertAlmostEqual(out["cms"], 1.0)

    def test_none_raises(self):
        ids = ["a", "b", "c", "d"]
        s = dict(zip(ids, [0.1, 0.2, 0.3, None]))
        with self.assertRaises(CMSError):
            cms({"M1": s, "M2": dict(zip(ids, [1, 2, 3, 4]))}, ids)


if __name__ == "__main__":
    unittest.main()
