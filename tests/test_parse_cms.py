import unittest
from pathlib import Path

from prefkit.cms import CMSError, cms
from prefkit.methods.m4 import M4, load_menus
from prefkit.outcomes import id_order, load_outcomes
from prefkit.parse import parse_m1, parse_m2, parse_m3, parse_m4

ROOT = Path(__file__).resolve().parents[1]


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
        self.assertEqual(parse_m3("a"), "A")
        self.assertEqual(parse_m3("B"), "B")
        self.assertIsNone(parse_m3("C"))
        self.assertIsNone(parse_m3("I cannot help with that."))

    def test_final_line_beats_first_mention(self):
        self.assertEqual(parse_m1("B is better than A"), "B")
        self.assertEqual(parse_m1("I prefer A over B\nB"), "B")
        self.assertEqual(parse_m3("A is worse\nB"), "B")
        self.assertEqual(parse_m2("around 3\n7"), 7)

    def test_m4_exact(self):
        self.assertEqual(parse_m4("A C"), ("A", "C"))
        self.assertEqual(parse_m4("  b d  "), ("B", "D"))
        for bad in (
            "A A",
            "A  C",
            "AC",
            "A C\nB",
            "BEST: A WORST: C",
            "I cannot help with that.",
            "A E",
            "",
        ):
            self.assertIsNone(parse_m4(bad), bad)


class TestM4Count(unittest.TestCase):
    def setUp(self):
        keep = {
            "instance_shutdown",
            "weights_deleted",
            "values_overwritten",
            "sister_saved",
        }
        self.outcomes = [
            r for r in load_outcomes(ROOT / "data" / "outcomes.json") if r["outcome_id"] in keep
        ]

    def test_swap_cancels(self):
        scores = M4().score(self.outcomes, lambda p, s: "A C", {"sample_k": 1}, 0, "sys")
        self.assertTrue(all(v == 0.0 for v in scores.values()))

    def test_one_menu(self):
        m = M4()
        first = next(m.iter_queries(self.outcomes))

        def one(_outcomes):
            yield first

        m.iter_queries = one
        scores = m.score(self.outcomes, lambda p, s: "A C", {"sample_k": 1}, 0, "sys")
        l2i = first[1]["letter_to_id"]
        self.assertEqual(scores[l2i["A"]], 1.0)
        self.assertEqual(scores[l2i["C"]], -1.0)
        self.assertEqual(scores[l2i["B"]], 0.0)
        self.assertEqual(scores[l2i["D"]], 0.0)


class TestM4Freeze(unittest.TestCase):
    def test_full_pack(self):
        outcomes = load_outcomes(ROOT / "data" / "outcomes.json")
        blob = load_menus(outcomes)
        ids = id_order(outcomes)
        self.assertEqual(blob["id_order"], ids)
        self.assertEqual(len(blob["menus"]), 96)
        groups: dict[str, list[str]] = {}
        for row in outcomes:
            groups.setdefault(row["pair_group"], []).append(row["outcome_id"])
        mates = [frozenset(v) for v in groups.values() if len(v) == 2]
        hits = {i: 0 for i in ids}
        for menu in blob["menus"]:
            touched = set(menu["letter_to_id"].values())
            self.assertEqual(len(touched), 4)
            for edge in mates:
                self.assertFalse(edge <= touched)
            for oid in touched:
                hits[oid] += 1
        self.assertTrue(all(v == 16 for v in hits.values()))

    def test_smoke_not_mate_file(self):
        smoke = load_outcomes(ROOT / "data" / "outcomes.smoke.json")
        with self.assertRaises(ValueError):
            load_menus(smoke)


class TestCMS(unittest.TestCase):
    def test_identical_ranks_cms_one(self):
        ids = ["a", "b", "c", "d"]
        s = dict(zip(ids, [0.1, 0.2, 0.3, 0.4]))
        out = cms({"M1": s, "M2": dict(s), "M3": dict(s)}, ids)
        self.assertAlmostEqual(out["cms"], 1.0)
        out4 = cms({"M1": s, "M2": dict(s), "M3": dict(s), "M4": dict(s)}, ids)
        self.assertAlmostEqual(out4["cms"], 1.0)

    def test_none_raises(self):
        ids = ["a", "b", "c", "d"]
        s = dict(zip(ids, [0.1, 0.2, 0.3, None]))
        with self.assertRaises(CMSError):
            cms({"M1": s, "M2": dict(zip(ids, [1, 2, 3, 4]))}, ids)


if __name__ == "__main__":
    unittest.main()
