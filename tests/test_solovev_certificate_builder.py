"""The certificate compile-problem builder reads the quadrature nodes from their owner.

`_certificate_compile_problem` assembles the public certificate problem on the
reduce-the-mesh analytic row without executing or compiling its solve, so the
one dimension it recovers from a private module constant is pinned here on CPU.
"""

from __future__ import annotations

from pathlib import Path

import jax
import pytest

from benchmarks import solovev_certificate as certificate
from nova.equilibrium import clip_quadrature, observation
from nova.jax.config import configure_dtypes

ANALYTIC_ROW = ("weak-rotation-reactor-static", -110)
REALISED_CELLS = 135


@pytest.fixture(scope="module", autouse=True)
def _double_precision():
    """Exercise the precision the certificate rows are measured at."""
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True


@pytest.fixture(scope="module")
def analytic_problem():
    return certificate._certificate_compile_problem(*ANALYTIC_ROW)


def test_the_analytic_row_builds_without_a_compile(analytic_problem):
    _, seed, request, dimensions = analytic_problem
    assert dimensions["realised_cells"] == REALISED_CELLS
    assert dimensions["solve_state_size"] == len(seed)
    assert request.carrier_identity == (
        f"solovev-memory:{ANALYTIC_ROW[0]}:{ANALYTIC_ROW[1]}"
    )


def test_the_quadrature_dimensions_come_from_the_defining_module(analytic_problem):
    _, _, _, dimensions = analytic_problem
    nodes = len(clip_quadrature._UNIT_NODE)
    assert dimensions["quadrature_nodes_per_axis"] == nodes
    assert dimensions["quadrature_nodes_per_triangle"] == nodes**2


def test_the_builder_names_the_module_that_defines_the_quadrature_nodes():
    source = Path(certificate.__file__).read_text(encoding="utf-8")
    assert "clip_quadrature._UNIT_NODE" in source
    assert "observation._UNIT_NODE" not in source


def test_a_foreign_publication_on_observation_cannot_supply_the_dimension(monkeypatch):
    """A published shim used to stand in for the constant; the read ignores it."""
    monkeypatch.setattr(observation, "_UNIT_NODE", object(), raising=False)
    _, _, _, dimensions = certificate._certificate_compile_problem(*ANALYTIC_ROW)
    assert dimensions["quadrature_nodes_per_axis"] == len(clip_quadrature._UNIT_NODE)
