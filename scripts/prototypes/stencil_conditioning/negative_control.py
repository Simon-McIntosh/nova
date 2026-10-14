"""Run a ring-selection test with the base implementation restored in memory."""

from __future__ import annotations

import ast
import argparse
import subprocess
import textwrap

import pytest

from nova.biot import plasmagrid


TEST = "tests/test_plasma_cell_partition.py::test_wall_distorted_ring_is_excluded"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    args = parser.parse_args()
    source = subprocess.check_output(
        ["git", "show", f"{args.base}:nova/biot/plasmagrid.py"], text=True
    )
    tree = ast.parse(source)
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PlasmaGrid"
    )
    method = next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == "loop_neighbour_vertices"
    )
    lines = source.splitlines()
    original = "\n".join(lines[method.lineno - 1 : method.end_lineno])
    namespace = {
        "np": plasmagrid.np,
        "hex_ring_slots": plasmagrid.hex_ring_slots,
    }
    exec(textwrap.dedent(original), namespace)
    plasmagrid.PlasmaGrid.loop_neighbour_vertices = staticmethod(
        namespace["loop_neighbour_vertices"]
    )
    print(f"mutation=restored {args.base} ring selection; test={TEST}", flush=True)
    return pytest.main(["-p", "no:cacheprovider", TEST, "-q"])


if __name__ == "__main__":
    raise SystemExit(main())
