"""Check source ownership against a graph whose expansion can be counted by hand."""

# ruff: noqa: E402

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp

from scripts.prototypes.order2.attribution import attribute_graph
from tests.equilibrium.test_topology_read import _jaxpr_equation_counts


@jax.jit
def _pair(value):
    doubled = value + value
    return jnp.sin(doubled)


def _twice(value):
    return _pair(value) + _pair(value)


def test_source_ranking_counts_each_expanded_equation():
    graph = jax.make_jaxpr(_twice)(jnp.asarray(1.0))
    expected = _jaxpr_equation_counts(graph)
    assert expected == {"unique_equations": 5, "expanded_equations": 7}
    attribution = attribute_graph(graph)
    assert attribution["expanded_equations"] == 7
    ranking = attribution["source_function_ranking"]
    assert sum(row["expanded_equations"] for row in ranking) == 7
    assert any(
        row["file"].endswith("test_attribution.py")
        and row["function"] == "_pair"
        and row["line"] > 0
        for row in ranking
    )
