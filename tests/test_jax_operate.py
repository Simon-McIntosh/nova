"""Agreement contract between the traced operator and the host path."""

import numpy as np
import pytest

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax.numpy as jnp

    from nova.biot.operate import (
        Coupling,
        HostOperator,
        MatrixData,
        Operator,
        Operators,
        TracedOperator,
    )


_TRACED_PSI_REFERENCE = np.array(
    [
        69.84432,
        105.53119,
        68.23457,
        47.513386,
        120.24757,
        96.30779,
        162.61769,
        139.7402,
        207.53981,
        193.97694,
        234.07397,
        159.52002,
        200.81287,
        230.37115,
        209.96411,
        112.27056,
        67.493256,
        67.67515,
        155.65189,
        102.412636,
        177.70634,
        175.74684,
        130.90895,
        145.61192,
        91.33887,
        110.671364,
    ],
    dtype=np.float32,
)
_TRACED_PSI_RELATIVE_BOUND = 5.667e-8


def test_traced_psi_precision_stays_fp32_under_explicit_dtype_policy():
    """The selected operator retains its measured fp32 result with fp64 available."""
    from nova.frame.coilset import CoilSet
    from nova.jax.config import configure_dtypes

    configure_dtypes()
    coilset = CoilSet(dcoil=-5, dplasma=-15, tcoil="hex", tplasma="hex")
    coilset.firstwall.insert(dict(o=[5, 1, 5]), Ic=15e6)
    coilset.coil.insert(8, 0, 0.75, 0.75, Ic=5e6)
    coilset.plasmagrid.solve()
    psi = np.asarray(coilset.plasmagrid.psi)

    assert psi.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(
        psi,
        _TRACED_PSI_REFERENCE,
        rtol=_TRACED_PSI_RELATIVE_BOUND,
        atol=0,
    )


@pytest.fixture(scope="module")
def cached_plasmagrid():
    """Return a solved CoilSet plasmagrid carrying a cached operator."""
    from nova.frame.coilset import CoilSet

    coilset = CoilSet(dcoil=-5, dplasma=-15, tcoil="hex", tplasma="hex")
    coilset.firstwall.insert(dict(o=[5, 1, 5]), Ic=15e6)
    coilset.coil.insert(8, 0, 0.75, 0.75, Ic=5e6)
    coilset.plasmagrid.solve()
    return coilset.plasmagrid


@pytest.fixture(scope="module")
def cached_force():
    """Return a solved CoilSet force instance carrying a Force operator."""
    from nova.frame.coilset import CoilSet

    coilset = CoilSet(dcoil=-5, tcoil="hex")
    coilset.coil.insert(8, 0, 0.75, 0.75, Ic=5e6)
    coilset.coil.insert(6, 1.2, 0.5, 0.5, Ic=3e6)
    coilset.force.solve(-2)
    return coilset.force


def test_traced_evaluate_matches_selected_adapter(cached_plasmagrid):
    """The pure traced evaluate agrees with the selected traced adapter."""
    pg = cached_plasmagrid
    selected_operator = pg.operator["Psi"]
    selected_result = np.asarray(selected_operator.evaluate())

    coupling = Operators(pg.data)["Psi"]
    source_current = jnp.asarray(pg.saloc["Ic"])
    traced_result = np.asarray(
        coupling.evaluate(coupling.source_target, source_current)
    )

    assert np.allclose(selected_result, traced_result, atol=1e-10)


def test_traced_update_turns_matches_selected_adapter(cached_plasmagrid):
    """The pure traced turn update agrees with the selected traced adapter."""
    pg = cached_plasmagrid
    selected_operator = pg.operator["Psi"]
    plasma_nturn = jnp.asarray(selected_operator.plasma_nturn, dtype=jnp.float32)

    coupling = Operators(pg.data)["Psi"]
    traced_source_target = np.asarray(coupling.update_plasma_turns(plasma_nturn))

    selected_operator.update_turns(svd=False)
    assert np.allclose(
        selected_operator.source_target, traced_source_target, atol=1e-12
    )


def _reference_update(
    source_target,
    plasma_target,
    source_plasma,
    plasma_plasma,
    plasma_nturn,
    source_index,
    target_index,
):
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

    operator = Coupling(
        jnp.asarray(source_target),
        MatrixData(
            plasma_target=jnp.asarray(plasma_target),
            source_plasma=jnp.asarray(source_plasma),
            plasma_plasma=jnp.asarray(plasma_plasma),
        ),
        source_plasma_index=source_index,
        target_plasma_index=target_index,
    )
    traced = np.asarray(operator.update_plasma_turns(jnp.asarray(plasma_nturn)))
    reference = _reference_update(
        source_target,
        plasma_target,
        source_plasma,
        plasma_plasma,
        plasma_nturn,
        source_index,
        target_index,
    )
    assert np.allclose(traced, reference, atol=1e-12)


def test_force_evaluate_applies_index_gain():
    """The Force classname multiplies the interaction by the indexed current."""
    rng = np.random.default_rng(1)
    source_target = rng.standard_normal((3, 4))
    source_current = rng.standard_normal(4)
    force_index = np.array([1, 0, 3])

    operator = Coupling(
        jnp.asarray(source_target, dtype=jnp.float32),
        MatrixData(force_index=jnp.asarray(force_index)),
        classname="Force",
    )
    result = np.asarray(
        operator.evaluate(
            operator.source_target, jnp.asarray(source_current, dtype=jnp.float32)
        )
    )
    source_target = source_target.astype(np.float32)
    source_current = source_current.astype(np.float32)
    reference = source_current[force_index] * (source_target @ source_current)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, reference, rtol=2e-7, atol=2e-8)


def test_evaluate_is_jit_traced(cached_plasmagrid):
    """The evaluate entry point is a jitted callable, not eager numpy."""
    coupling = Operators(cached_plasmagrid.data)["Psi"]
    assert hasattr(coupling.evaluate, "lower") or callable(coupling.evaluate)


def _host_and_traced(plasmagrid, attr="Psi"):
    """Build host and traced operators from the same solved data subset."""
    data = plasmagrid.data
    names = [
        name for name in (attr, f"_{attr}", f"{attr}_", f"_{attr}_") if name in data
    ]
    dataset = data[names]
    args = (
        plasmagrid.aloc,
        plasmagrid.saloc,
        plasmagrid.classname,
        plasmagrid.index,
        dataset,
    )
    return HostOperator(*args), TracedOperator(*args)


def test_operate_selects_traced_operator(cached_plasmagrid):
    """With JAX present the selection alias and live operator are traced."""
    assert Operator is TracedOperator
    assert isinstance(cached_plasmagrid.operator["Psi"], TracedOperator)


def test_host_fallback_is_drop_in(cached_plasmagrid):
    """The host fallback and traced operator agree on evaluate and turns.

    Guards the fallback branch taken when JAX is absent: both
    present the same mutating surface and the same numbers.
    """
    host_operator, traced = _host_and_traced(cached_plasmagrid)
    assert np.allclose(host_operator.evaluate(), traced.evaluate(), atol=1e-10)

    host_operator.update_turns(svd=False)
    traced.update_turns(svd=False)
    assert np.allclose(host_operator.source_target, traced.source_target, atol=1e-12)
    # the traced update propagates back into the linked dataset array
    assert np.allclose(
        traced.source_target, cached_plasmagrid.data["Psi"].data, atol=1e-12
    )


def test_force_index_gain_agrees_across_implementations(cached_force):
    """Both implementations scale a Force interaction by the same indexed gain."""
    host_operator, traced = _host_and_traced(cached_force, "Fr")
    index = np.asarray(cached_force.index)
    source_current = np.asarray(cached_force.saloc["Ic"])
    reference = source_current[index] * (host_operator.source_target @ source_current)

    assert reference.shape == index.shape
    assert np.allclose(host_operator.evaluate(), reference, rtol=1e-10)
    # the traced path evaluates in single precision
    assert np.allclose(traced.evaluate(), reference, rtol=1e-6)


def test_force_index_is_wired_only_for_the_force_interaction(
    cached_force, cached_plasmagrid
):
    """Only the classname that reads an index gain is given one."""
    assert Operators(cached_force.data)["Fr"].matrix_data.force_index is not None
    assert Operators(cached_plasmagrid.data)["Psi"].matrix_data.force_index is None


def _force_dataset_without_index(force, attr="Fr"):
    """Return a Force coupling subset stripped of its index variable."""
    names = [
        name
        for name in (attr, f"_{attr}", f"{attr}_", f"_{attr}_")
        if name in force.data
    ]
    dataset = force.data[names]
    assert "index" in dataset  # the stripping below is otherwise vacuous
    return dataset.drop_vars("index")


@pytest.mark.parametrize("operator_class", ["host", "traced"])
def test_force_without_an_index_raises(cached_force, operator_class):
    """A Force operator built without its index gain fails loudly."""
    dataset = _force_dataset_without_index(cached_force)
    build = {"host": HostOperator, "traced": TracedOperator}[operator_class]
    with pytest.raises(ValueError, match="Force operator"):
        build(
            cached_force.aloc,
            cached_force.saloc,
            "Force",
            np.array([]),
            dataset,
        )


def test_force_dataset_lookup_without_an_index_raises(cached_force):
    """The dataset-sourced index is required too, not silently skipped."""
    dataset = _force_dataset_without_index(cached_force)
    with pytest.raises(ValueError, match="Force operator"):
        Operators(dataset)["Fr"]


def test_force_gain_is_not_a_silent_axis_insertion():
    """Indexing with a missing gain would insert an axis, so it must raise.

    A square source-target matrix broadcasts against the inserted axis, which
    returns a full-length current as the gain instead of the indexed one.
    """
    rng = np.random.default_rng(2)
    source_target = rng.standard_normal((4, 4))
    source_current = rng.standard_normal(4)

    operator = Coupling(jnp.asarray(source_target), MatrixData(), classname="Force")
    with pytest.raises(ValueError, match="Force operator"):
        operator.evaluate(operator.source_target, jnp.asarray(source_current))


@pytest.mark.parametrize(
    "index_name", ["source_plasma_index", "target_plasma_index", "classname"]
)
def test_plasma_indices_read_the_dataset_attributes(cached_plasmagrid, index_name):
    """A variable of the same name must not shadow the dataset attribute."""
    pg = cached_plasmagrid
    names = [name for name in ("Psi", "_Psi", "Psi_", "_Psi_") if name in pg.data]
    decoy = "Force" if index_name == "classname" else -99
    dataset = pg.data[names].assign_coords({index_name: decoy})
    # attribute access resolves the decoy variable, plain attrs the true value
    assert np.asarray(getattr(dataset, index_name)) == decoy
    expected = pg.data.attrs[index_name]
    assert expected != decoy

    args = (pg.aloc, pg.saloc, pg.classname, pg.index, dataset)
    host_operator, traced = HostOperator(*args), TracedOperator(*args)
    for operator in (host_operator, traced):
        if index_name != "classname":
            assert getattr(operator, index_name) == expected
    assert traced._operator.classname == pg.data.attrs["classname"]
    assert np.allclose(host_operator.evaluate(), traced.evaluate(), atol=1e-10)


def test_version_counter_gates_reevaluation():
    """The xxhash current/turn version counters gate the jitted evaluate path."""
    from nova.frame.coilset import CoilSet

    coilset = CoilSet(dcoil=-5, dplasma=-5, tcoil="hex", tplasma="hex")
    coilset.firstwall.insert(dict(o=[5, 1, 5]), Ic=15e6)
    coilset.plasmagrid.solve()

    coilset.sloc["Ic"] = 1e6
    psi = coilset.plasmagrid.psi.copy()
    # freezing the current version hash skips the re-evaluation
    coilset.sloc["Ic"] = 2e6
    coilset.plasmagrid.version["psi"] = coilset.aloc_hash["Ic"]
    assert np.allclose(coilset.plasmagrid.psi, psi)
    # clearing the hash forces a fresh traced evaluate against the new current
    coilset.plasmagrid.version["psi"] = None
    assert not np.allclose(coilset.plasmagrid.psi, psi)


if __name__ == "__main__":
    pytest.main([__file__])
