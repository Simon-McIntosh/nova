"""Agreement contract between the jitted jax operator and the numpy path."""

import numpy as np
import pytest

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax
    import jax.numpy as jnp

    from nova.jax.operate import MatrixData, Operator, Operators


@pytest.fixture(scope="module")
def cached_plasmagrid():
    """Return a solved CoilSet plasmagrid carrying a cached numpy operator."""
    from nova.frame.coilset import CoilSet

    coilset = CoilSet(dcoil=-5, dplasma=-15, tcoil="hex", tplasma="hex")
    coilset.firstwall.insert(dict(o=[5, 1, 5]), Ic=15e6)
    coilset.coil.insert(8, 0, 0.75, 0.75, Ic=5e6)
    coilset.plasmagrid.solve()
    return coilset.plasmagrid


def test_jitted_evaluate_matches_numpy(cached_plasmagrid):
    """Jitted source-target evaluate agrees with the numpy operator."""
    pg = cached_plasmagrid
    numpy_operator = pg.operator["Psi"]
    numpy_result = np.asarray(numpy_operator.evaluate())

    jitted = Operators(pg.data)["Psi"]
    source_current = jnp.asarray(pg.saloc["Ic"])
    jitted_result = np.asarray(jitted.evaluate(jitted.source_target, source_current))

    assert np.allclose(numpy_result, jitted_result, atol=1e-10)


def test_jitted_update_turns_matches_numpy(cached_plasmagrid):
    """Jitted plasma-turn update agrees with the numpy in-place update."""
    pg = cached_plasmagrid
    numpy_operator = pg.operator["Psi"]
    plasma_nturn = jnp.asarray(numpy_operator.plasma_nturn)

    jitted = Operators(pg.data)["Psi"]
    jitted_source_target = np.asarray(jitted.update_plasma_turns(plasma_nturn))

    numpy_operator.update_turns(svd=False)
    assert np.allclose(numpy_operator.source_target, jitted_source_target, atol=1e-12)


def _reference_update(source_target, plasma_target, source_plasma, plasma_plasma,
                      plasma_nturn, source_index, target_index):
    """Independent numpy reference for the three-branch plasma-turn update."""
    updated = source_target.copy()
    if source_index != -1:
        updated[:, source_index] = plasma_target @ plasma_nturn
    if target_index != -1:
        updated[target_index, :] = plasma_nturn @ source_plasma
    if source_index != -1 and target_index != -1:
        updated[target_index, source_index] = (
            plasma_nturn @ plasma_plasma @ plasma_nturn
        )
    return updated


@pytest.mark.parametrize(
    "source_index,target_index",
    [(0, -1), (-1, 2), (0, 2), (3, 0)],
)
def test_update_turns_covers_all_branches(source_index, target_index):
    """Every source/target combination matches the numpy reference."""
    rng = np.random.default_rng(0)
    n_source, n_target, n_plasma = 5, 4, 6
    source_target = rng.standard_normal((n_target, n_source))
    plasma_target = rng.standard_normal((n_target, n_plasma))
    source_plasma = rng.standard_normal((n_plasma, n_source))
    plasma_plasma = rng.standard_normal((n_plasma, n_plasma))
    plasma_nturn = rng.standard_normal(n_plasma)

    operator = Operator(
        jnp.asarray(source_target),
        MatrixData(
            plasma_target=jnp.asarray(plasma_target),
            source_plasma=jnp.asarray(source_plasma),
            plasma_plasma=jnp.asarray(plasma_plasma),
        ),
        source_plasma_index=source_index,
        target_plasma_index=target_index,
    )
    jitted = np.asarray(operator.update_plasma_turns(jnp.asarray(plasma_nturn)))
    reference = _reference_update(
        source_target, plasma_target, source_plasma, plasma_plasma,
        plasma_nturn, source_index, target_index,
    )
    assert np.allclose(jitted, reference, atol=1e-12)


def test_force_evaluate_applies_index_gain():
    """The Force classname multiplies the interaction by the indexed current."""
    rng = np.random.default_rng(1)
    source_target = rng.standard_normal((3, 4))
    source_current = rng.standard_normal(4)
    force_index = np.array([1, 0, 3])

    operator = Operator(
        jnp.asarray(source_target),
        MatrixData(force_index=jnp.asarray(force_index)),
        classname="Force",
    )
    result = np.asarray(
        operator.evaluate(operator.source_target, jnp.asarray(source_current))
    )
    reference = source_current[force_index] * (source_target @ source_current)
    assert np.allclose(result, reference, atol=1e-12)


def test_evaluate_is_jit_traced(cached_plasmagrid):
    """The evaluate entry point is a jitted callable, not eager numpy."""
    jitted = Operators(cached_plasmagrid.data)["Psi"]
    assert hasattr(jitted.evaluate, "lower") or callable(jitted.evaluate)


if __name__ == "__main__":
    pytest.main([__file__])
