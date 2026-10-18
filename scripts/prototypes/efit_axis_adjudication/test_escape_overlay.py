"""Guards on the escape overlay and on the wall-loop roles of a panel.

Two claims the panels make must not regress: every panel carries imas-efit's
escape bottleneck, including a panel whose nova rebuild failed, and the wall is
split into vessel and obstacles by the shared domain owner rather than by a
private copy, with that owner's refusal of an ambiguous axis propagated.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import plot as adjudication_plot
from analyse import REPORT
from efit_field_swap.crosscheck import domain_rings
from efit_field_swap.trace_record import read_trace_record
from nova.media.sources.frame import inside_wall_units
from records import parse_record


def _markers(ax):
    return [line for line in ax.lines if line.get_marker() == "D"]


def _blocking_rebuild_row(escape):
    return {
        "case": "synthetic-crossing-A",
        "path": REPORT / "crossing" / "synthetic_crossing_a.txt",
        "contour_levels": [0.7, 0.5, 0.3],
        "census": [],
        "efit": {},
        "adapter_a": None,
        "adapter_b": None,
        "escape": escape,
        "rebuild": {"status": "rebuild failed", "grad_norm_at_axis": 1.0},
    }


class EscapeOverlayTest(unittest.TestCase):
    def test_the_bottleneck_marker_takes_its_own_colour(self):
        fig, ax = plt.subplots()
        adjudication_plot._draw_escape_bottleneck(ax, {"bottleneck": [0.4, 0.6]})
        self.assertEqual(len(_markers(ax)), 1)
        self.assertEqual(
            _markers(ax)[0].get_markeredgecolor(), adjudication_plot.COLOUR["escape"]
        )
        plt.close(fig)

    def test_census_flag_is_checked_even_when_position_is_inside(self):
        row = _blocking_rebuild_row(None)
        record = parse_record(row["path"])
        position = [record.axis_r, record.axis_z]
        wall = adjudication_plot._typed_wall_units(
            adjudication_plot._wall_loops_by_role(row["path"])
        )
        self.assertTrue(inside_wall_units([position], wall)[0])
        row["census"] = [
            {"r": position[0], "z": position[1], "kind": "saddle", "inside_wall": False}
        ]
        fig, ax = plt.subplots()
        with self.assertRaisesRegex(ValueError, "outside the vessel"):
            adjudication_plot.draw_panel(ax, row)
        plt.close(fig)

    def test_census_geometry_is_checked_even_when_flag_is_inside(self):
        row = _blocking_rebuild_row(None)
        record = parse_record(row["path"])
        position = [record.rgrid.min() - 1.0, record.axis_z]
        wall = adjudication_plot._typed_wall_units(
            adjudication_plot._wall_loops_by_role(row["path"])
        )
        self.assertFalse(inside_wall_units([position], wall)[0])
        row["census"] = [
            {"r": position[0], "z": position[1], "kind": "saddle", "inside_wall": True}
        ]
        fig, ax = plt.subplots()
        with self.assertRaisesRegex(ValueError, "outside the vessel"):
            adjudication_plot.draw_panel(ax, row)
        plt.close(fig)

    def test_grid_node_saddle_proxy_is_refused_before_painting(self):
        row = _blocking_rebuild_row(None)
        row["efit"] = {
            "saddle_source_position": [0.4, 0.6],
            "saddle_plot_position": [0.4, 0.6],
        }
        fig, ax = plt.subplots()
        with self.assertRaisesRegex(ValueError, "grid-node saddle proxy"):
            adjudication_plot.draw_panel(ax, row)
        plt.close(fig)

    def test_a_failed_rebuild_panel_still_carries_the_bottleneck(self):
        row = _blocking_rebuild_row({"bottleneck": [0.4, 0.6]})
        fig, ax = plt.subplots()
        adjudication_plot.draw_panel(ax, row)
        self.assertEqual(len(_markers(ax)), 1)
        plt.close(fig)


class WallLoopRoleTest(unittest.TestCase):
    def test_roles_come_from_the_shared_domain_split(self):
        path = REPORT / "crossing" / "synthetic_crossing_a.txt"
        vessel, obstacles = domain_rings(read_trace_record(path))
        roles = adjudication_plot._wall_loops_by_role(path)
        self.assertEqual(roles[0][1], "vessel")
        self.assertEqual(len(roles), 1 + len(obstacles))
        np.testing.assert_allclose(roles[0][0], vessel)

    def test_an_ambiguous_axis_containment_is_refused(self):
        outer = np.asarray(
            [
                (-1.0, -1.0),
                (1.0, -1.0),
                (1.0, 1.0),
                (0.0, 1.0),
                (-1.0, 1.0),
                (-1.0, -1.0),
            ]
        )
        inner = np.asarray(
            [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5)]
        )
        ambiguous = SimpleNamespace(axis_r=0.0, axis_z=0.0, wall_rings=(outer, inner))
        with mock.patch.object(
            adjudication_plot, "read_trace_record", return_value=ambiguous
        ):
            with self.assertRaisesRegex(ValueError, "axis-containing wall loop"):
                adjudication_plot._wall_loops_by_role("unused")


if __name__ == "__main__":
    unittest.main()
