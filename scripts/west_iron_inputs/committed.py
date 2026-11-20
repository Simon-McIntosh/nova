"""Read the WEST 62751 values the committed records hold.

A verifier that compares a regenerated pin against constants transcribed into
its own source would pass on a number mis-transcribed in both places. These
readers take each value out of the record that owns it, so the comparison
below is against the committed record rather than against a copy of it.

Each reader raises when its record no longer states the value, so an edited or
moved record fails loudly instead of silently leaving the comparison short.
"""

from __future__ import annotations

import os
from pathlib import Path
import re

SOLVE_FRAGMENT = (
    "docs/evidence/fragments/iron-core-magnetisation/icm-west-62751-solve.html"
)
PAIR_FRAGMENT = (
    "docs/evidence/fragments/iron-core-magnetisation/icm-west-62751-galerkin-pair.html"
)
PLAN = "docs/plans/iron-core-magnetisation.html"
MATERIAL_TEST = "tests/test_magnetisation.py"


def record_root() -> Path:
    """Return the tree the committed records are read from."""
    return Path(
        os.environ.get("NOVA_WEST_RECORDS_ROOT", Path(__file__).resolve().parents[2])
    ).resolve()


def read(relative: str) -> str:
    return (record_root() / relative).read_text()


def _numbers(text: str, pattern: str, count: int, what: str) -> list[float]:
    match = re.search(pattern, text)
    if match is None:
        raise ValueError(f"the committed record no longer states {what}")
    values = [float(item) for item in match.groups()]
    if len(values) != count:
        raise ValueError(f"{what} parsed {len(values)} values, expected {count}")
    return values


def probe_radii() -> list[float]:
    """The three probe radii, from the fragment that quotes them."""
    return _numbers(
        read(PAIR_FRAGMENT),
        r"Probes are R = ([\d.]+), ([\d.]+) and ([\d.]+) m",
        3,
        "the probe radii",
    )


def air_microtesla_pair() -> list[float]:
    """The air-core field magnitudes in microtesla, from the mesh-pair record."""
    return _numbers(
        read(PAIR_FRAGMENT),
        r"air-core field magnitudes there are ([\d.]+), ([\d.]+) and ([\d.]+)",
        3,
        "the air-core field in microtesla",
    )


def air_plan_microtesla() -> float:
    """The air-core field at +1 m as the plan quotes it, rounded to microtesla."""
    return _numbers(
        read(PLAN),
        r"([\d.]+)\s*[µμ]T air-core",
        1,
        "the plan's air-core field",
    )[0]


def air_millitesla_solve() -> list[float]:
    """The air-core field magnitudes in millitesla, from the solve record's table."""
    rows = re.findall(
        r"<tr><td>\d+</td><td>[\d.]+</td><td>[\d.]+</td><td>([\d.]+)</td></tr>",
        read(SOLVE_FRAGMENT),
    )
    if len(rows) != 3:
        raise ValueError(
            f"the solve record states {len(rows)} air-core rows, expected 3"
        )
    return [float(value) for value in rows]


def table_b_knots() -> list[float]:
    """The first two B knots, from the solve record's material table."""
    return _numbers(
        read(SOLVE_FRAGMENT),
        r"First two B knots</td><td>([\d.]+) and ([\d.]+) T",
        2,
        "the first two B knots",
    )


def table_mu_origin() -> list[float]:
    """The first two relative permeabilities, from the solve record's table."""
    return _numbers(
        read(SOLVE_FRAGMENT),
        r"Corresponding relative permeabilities</td><td>([\d.]+) and ([\d.]+)",
        2,
        "the first two relative permeabilities",
    )


def table_mu_maximum() -> float:
    """The table's upper bound, from the mesh-pair record's colour-scale note."""
    return _numbers(
        read(PAIR_FRAGMENT),
        r"table's own range, ([\d.]+) to ([\d.]+)",
        2,
        "the table range",
    )[1]


def source_time() -> float:
    """The source time the pin records, from the solve record's prose."""
    return _numbers(
        read(SOLVE_FRAGMENT),
        r"WEST 62751 at ([\d.]+) s",
        1,
        "the source time",
    )[0]


def material_test_mu_origin() -> float:
    """The origin permeability the material test asserts."""
    return _numbers(
        read(MATERIAL_TEST),
        r"assert mu\[0\] == pytest\.approx\(([\d.]+), rel=",
        1,
        "the material test's origin permeability",
    )[0]
