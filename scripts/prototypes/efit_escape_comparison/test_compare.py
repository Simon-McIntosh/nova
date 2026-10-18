"""Structural guards on the escape-level join and the contacts it reports.

The comparison is a join of two recorded decision logs on the 26 trace sides,
and every claim on its page is read off that join. These tests hold the two
properties a silent join defect would repair in silence: that no side is
dropped or misaligned, and that a limited row reports the wall contact at
nova's own boundary level rather than a wall maximum that sits at a different
level.
"""

import json
import unittest

import compare


def _rows():
    accuracy, endpoints = compare.parse_escape(
        (compare.ESCAPE_DIR / "evidence-accuracy.log").read_text().splitlines()
    )
    nova = {
        row["case"]: row for row in json.loads(compare.NOVA_SWEEP.read_text())["rows"]
    }
    return compare.build_rows(accuracy, endpoints, nova), accuracy


class JoinTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.accuracy = _rows()

    def test_every_pair_carries_both_sides(self):
        for pair in {key[0] for key in self.accuracy}:
            with self.subTest(pair=pair):
                self.assertIn((pair, "A"), self.accuracy)
                self.assertIn((pair, "B"), self.accuracy)

    def test_the_join_drops_no_side_and_misaligns_none(self):
        expected = {
            (pair, side)
            for pair in {key[0] for key in self.accuracy}
            for side in ("A", "B")
        }
        self.assertEqual(len(expected), 26)
        self.assertEqual({(row["pair"], row["side"]) for row in self.rows}, expected)
        self.assertEqual(len(self.rows), 26)
        self.assertEqual(len({row["case"] for row in self.rows}), 26)
        for row in self.rows:
            with self.subTest(case=row["case"]):
                self.assertEqual(row["case"], f"{row['pair']}-{row['side']}")
                entry = self.accuracy[(row["pair"], row["side"])]
                self.assertEqual(row["efit_level"], entry["level"])
                self.assertEqual(row["efit_bottleneck"], entry["bottleneck"])

    def test_the_two_sides_of_a_pair_are_not_one_decision(self):
        for pair in {row["pair"] for row in self.rows}:
            sides = {row["pair"] + row["side"]: row for row in self.rows}
            with self.subTest(pair=pair):
                left, right = sides[pair + "A"], sides[pair + "B"]
                self.assertNotEqual(
                    (left["efit_level"], tuple(left["efit_bottleneck"])),
                    (right["efit_level"], tuple(right["efit_bottleneck"])),
                )


class ContactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, _ = _rows()

    def test_limited_rows_report_a_wall_contact_at_nova_boundary_level(self):
        limited = [row for row in self.rows if row["nova_label"] == "limited"]
        self.assertTrue(limited)
        for row in limited:
            with self.subTest(case=row["case"]):
                contact = row["nova_contact"]
                self.assertIsNotNone(contact)
                self.assertEqual(contact["kind"], "wall")
                self.assertAlmostEqual(
                    contact["level"], row["nova_boundary"], places=12
                )
                self.assertIsNotNone(contact["position"])
                self.assertAlmostEqual(
                    contact["position"][0], row["nova_admitted_point"][0], places=12
                )

    def test_limited_rows_never_report_a_saddle_or_the_wall_maximum(self):
        for row in self.rows:
            if row["nova_label"] != "limited":
                continue
            with self.subTest(case=row["case"]):
                contact = row["nova_contact"]
                self.assertNotEqual(contact["kind"], "saddle")
                self.assertNotAlmostEqual(
                    contact["level"], row["nova_wall_extremum"], places=6
                )

    def test_diverted_rows_report_an_interior_saddle(self):
        diverted = [row for row in self.rows if row["nova_label"] == "diverted"]
        self.assertTrue(diverted)
        for row in diverted:
            with self.subTest(case=row["case"]):
                self.assertEqual(row["nova_contact"]["kind"], "saddle")

    def test_refused_rows_report_no_contact(self):
        refused = [
            row for row in self.rows if row["nova_label"] in ("unqualified", "absent")
        ]
        self.assertTrue(refused)
        for row in refused:
            with self.subTest(case=row["case"]):
                self.assertIsNone(row["nova_contact"])


if __name__ == "__main__":
    unittest.main()
