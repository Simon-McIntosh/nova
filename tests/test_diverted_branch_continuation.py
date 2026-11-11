"""Saved-state diagnostics use the measurement prototype's lower-root reader."""

import json

import jax
import numpy as np

from nova.jax.config import configure_dtypes

configure_dtypes()
assert jax.config.jax_enable_x64


def test_saved_state_refresh_reuses_measurement_lower_root_reader(
    tmp_path, monkeypatch
):
    from scripts.prototypes.diiid_diverted_branch import measure, refresh

    radius, height = np.linspace(1.0, 2.0, 5), np.linspace(-1.0, 1.0, 5)
    flux = (radius[:, None] - 1.5) ** 2 - (height[None, :] + 0.5) ** 2
    state = flux.ravel()
    axis, saddle = [1.5, 0.5], [1.5, -0.5]
    data = dict(
        radius=radius,
        height=height,
        flux=flux,
        state=state,
        axis=axis,
        reference_axis=axis,
        limiter=np.array(
            [[1.0, -1.0], [2.0, -1.0], [2.0, 1.0], [1.0, 1.0], [1.0, -1.0]]
        ),
    )
    np.savez_compressed(tmp_path / "terminal.npz", **data)
    np.savez_compressed(tmp_path / "inner-0.npz", state=state, initial=state)
    topology = dict(selected_saddle_rz_m=saddle, axis_rz_m=axis)
    record = dict(
        efit_x_point_rz_m=saddle,
        inner=[dict(index=0)],
        recovery={},
        trip_topology=[dict(trip=0, initial=dict(topology), returned=dict(topology))],
        terminal_topology=dict(topology),
    )
    (tmp_path / "receipt.json").write_text(json.dumps(record))
    calls = []
    owner = measure.lower_saddle_reader

    def observed(data, reference):
        read = owner(data, reference)

        def sample(state, selected, axis):
            calls.append(np.asarray(axis))
            return read(state, selected, axis)

        return sample

    monkeypatch.setattr(measure, "lower_saddle_reader", observed)
    monkeypatch.setattr("sys.argv", ["refresh", "--row", str(tmp_path)])
    refresh.main()
    assert len(calls) == 3, "refresh bypassed the measurement's lower-root owner"
    refreshed = json.loads((tmp_path / "receipt.json").read_text())
    terminal = refreshed["terminal_topology"]
    assert terminal["lower_saddle_qualified"]
    assert terminal["lower_saddle_distance_m"] < 1e-12
    np.testing.assert_allclose(terminal["lower_saddle_rz_m"], saddle, atol=1e-12)
