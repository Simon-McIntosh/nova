"""Moment-vector contraction at fixed flux targets."""

import jax
import jax.numpy as jnp
import numpy as np
from shapely.geometry import Polygon
import xarray

from nova.biot.biotframe import Source, Target
from nova.biot.limiter import Limiter
from nova.biot.null import Null1D
from nova.biot.solve import Solve
from nova.biot.target import FluxTarget
import nova.biot.target as target_module


def _hexagon(radius, height, section_radius=0.065):
    angle = np.pi / 6 + np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    return np.column_stack(
        [
            radius + section_radius * np.cos(angle),
            height + section_radius * np.sin(angle),
        ]
    )


def _assembled_flux_blocks():
    centres = np.array([[2.94, -0.04], [3.06, 0.04]])
    source = Source(
        {
            "x": centres[:, 0],
            "y": np.zeros(len(centres)),
            "z": centres[:, 1],
            "segment": ["polysection"] * len(centres),
            "poly": [Polygon(_hexagon(*centre)) for centre in centres],
            "frame": [f"Coil{index}" for index in range(len(centres))],
            "nturn": np.ones(len(centres)),
            "plasma": np.ones(len(centres), dtype=bool),
            "link": [""] * len(centres),
        }
    )
    coordinate = np.array([[2.72, -0.16], [3.0, 0.01], [3.31, 0.22]])
    solve = Solve(
        source,
        Target({"x": coordinate[:, 0], "z": coordinate[:, 1]}),
        attrs=["Psi", "PsiR", "PsiZ"],
        turns=[False, False],
        reduce=[False, False],
    )
    names = ("Psi", "PsiR", "PsiZ")
    return coordinate, tuple(jnp.asarray(solve.data[name]) for name in names)


def _flux_target(source_target, blocks, coordinate):
    uniform, radial, vertical = blocks
    return FluxTarget(
        source_target,
        uniform,
        Null1D(jnp.asarray(coordinate)),
        plasma_target_r=radial,
        plasma_target_z=vertical,
    )


def test_flux_target_contracts_three_current_moment_vectors_on_assembled_mesh():
    """The traced target matches direct evaluation of three assembled blocks."""
    coordinate, blocks = _assembled_flux_blocks()
    target = _flux_target(jnp.zeros((len(coordinate), 1)), blocks, coordinate)
    moments = (
        jnp.asarray([1.4, -0.2]),
        jnp.asarray([0.08, -0.03]),
        jnp.asarray([-0.04, 0.06]),
    )

    expected = sum(
        matrix @ vector for matrix, vector in zip(blocks, moments, strict=True)
    )
    np.testing.assert_array_equal(target.internal(moments), expected)


def test_zero_current_moments_preserve_uniform_contraction_bitwise():
    """Zero companions reduce the contraction to the uniform matrix."""
    coordinate, blocks = _assembled_flux_blocks()
    target = _flux_target(jnp.zeros((len(coordinate), 1)), blocks, coordinate)
    current = jnp.asarray([1.4, -0.2])
    zero = jnp.zeros_like(current)
    previous = blocks[0] @ current

    np.testing.assert_array_equal(target.internal((current, zero, zero)), previous)
    np.testing.assert_array_equal(target.internal(current), previous)


def test_flux_target_pytree_round_trips_moment_blocks():
    """The widened child tuple reconstructs every matrix and its null locator."""
    coordinate, blocks = _assembled_flux_blocks()
    source_target = jnp.arange(3.0).reshape(3, 1)
    target = _flux_target(source_target, blocks, coordinate)

    children, aux_data = target.tree_flatten()
    assert len(children) == 5
    assert aux_data == {}
    leaves, structure = jax.tree_util.tree_flatten(target)
    restored = jax.tree_util.tree_unflatten(structure, leaves)

    np.testing.assert_array_equal(restored.source_target, source_target)
    np.testing.assert_array_equal(restored.plasma_target, blocks[0])
    np.testing.assert_array_equal(restored.plasma_target_r, blocks[1])
    np.testing.assert_array_equal(restored.plasma_target_z, blocks[2])
    np.testing.assert_array_equal(restored.coordinate, jnp.asarray(coordinate))


def test_limiter_target_uses_keyword_fields_without_changing_values(monkeypatch):
    """Limiter construction names every target field it forwards."""
    coordinate = np.array([[2.7, -0.2], [3.0, 0.0], [2.7, 0.2]])
    source_target = np.arange(6.0).reshape(3, 2)
    plasma_target = np.arange(9.0).reshape(3, 3)
    limiter = Limiter(
        data=xarray.Dataset(
            {
                "x": ("target", coordinate[:, 0]),
                "z": ("target", coordinate[:, 1]),
                "Psi": (("target", "source"), source_target),
                "Psi_": (("target", "plasma"), plasma_target),
            }
        )
    )
    limiter.load_arrays()

    class KeywordTarget:
        def __init__(self, *, source_target, plasma_target, null):
            self.source_target = source_target
            self.plasma_target = plasma_target
            self.null = null

    monkeypatch.setattr(target_module, "FluxTarget", KeywordTarget)
    built = limiter.target

    np.testing.assert_array_equal(built.source_target, source_target)
    np.testing.assert_array_equal(built.plasma_target, plasma_target)
    np.testing.assert_array_equal(built.null.coordinate, jnp.asarray(coordinate))
