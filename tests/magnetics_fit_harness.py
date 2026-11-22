"""Shared receipt and diagnostic-source helpers for the magnetics fit tests."""

from __future__ import annotations

import numpy as np


def _persist_forward(receipt, path):
    """Persist the complete forward result before deriving any score."""
    import copyreg
    import pickle
    from types import MappingProxyType
    from nova.equilibrium.forward_operator import ForwardTopologyState

    copyreg.pickle(type(MappingProxyType({})), lambda value: (dict, (dict(value),)))
    copyreg.pickle(
        ForwardTopologyState,
        lambda value: (
            ForwardTopologyState.tree_unflatten,
            (None, value.tree_flatten()[0]),
        ),
    )
    pending = path.with_suffix(".pending")
    with pending.open("wb") as stream:
        pickle.dump(receipt, stream)
    pending.replace(path)


def _persist_forward_failure(error, output, name):
    """Retain the public call's terminal history when receipt qualification raises."""
    import json
    import pickle
    import traceback
    from nova.equilibrium.fixed_point import FixedPointResult

    frames = list(traceback.walk_tb(error.__traceback__))
    histories = [
        frame.f_locals["history"]
        for frame, _ in frames
        if isinstance(frame.f_locals.get("history"), FixedPointResult)
    ]
    summary = dict(case=name, exception=type(error).__name__, message=str(error))
    if histories:
        history = histories[-1]
        destination = output / f"{name}-rejected-history.pkl"
        with destination.open("wb") as stream:
            pickle.dump(history, stream)
        summary.update(
            residual=float(history.residual),
            converged=bool(history.converged),
            termination_reason=int(history.termination_reason),
            trips=int(history.active_set_iterations),
            finite_residual_entries=int(
                np.count_nonzero(np.isfinite(np.asarray(history.active_set_residuals)))
            ),
            history_path=str(destination),
        )
    admitted = [
        frame.f_locals["admitted"]
        for frame, _ in frames
        if frame.f_code.co_name == "require_qualified_axis"
    ]
    if admitted:
        summary["axis_admitted"] = bool(admitted[-1])
    (output / f"{name}-generating-failure.json").write_text(
        json.dumps(summary, indent=2)
    )
    print("GENERATING_FAILURE " + json.dumps(summary), flush=True)




class _PlasmaDiagnosticSource:
    """Bind terminal cell currents to fixed authored diagnostic sections."""

    def __init__(self, sections):
        from nova.frame.coilset import CoilSet

        self.biot = CoilSet(dcoil=-1, field_attrs=["Psi", "Br", "Bz"])
        for index, section in enumerate(sections):
            self.biot.coil.insert(
                np.asarray(section),
                nturn=1.0,
                Ic=0.0,
                name=f"cell_{index}",
            )
        for name in ("point", "probe", "poloidal_flux_loop"):
            factory = getattr(self.biot, name)
            self._cache_factory(factory)

    @staticmethod
    def _cache_factory(factory):
        """Retain factory-owned Green tables while source currents vary."""
        original = factory.solve
        tables = {}

        def solve(*args):
            if args:
                coordinates = np.asarray(args[0], dtype=np.float64)
            else:
                coordinates = np.column_stack(
                    (factory.target["x"], factory.target["z"])
                )
            key = coordinates.shape, coordinates.tobytes()
            if key not in tables:
                original(*args)
                tables[key] = factory.data.copy(deep=True)
            else:
                factory.data = tables[key]
                factory.load_operators()

        factory.solve = solve

    def __call__(self, receipt):
        self.biot.sloc["Ic"] = np.asarray(receipt.terminal_state.cell_current)
        return self.biot


def _physical_sensor_rows(biot, sensors):
    """Synthesize row-ordered observations through the diagnostic kernel owners."""
    from nova.equilibrium.observation_kernels import (
        virtual_flux_loops,
        virtual_poloidal_probes,
    )

    points = np.column_stack((sensors.r, sensors.z))
    loops = np.asarray(sensors.flux_loop, dtype=bool)
    rows = np.empty(sensors.number)
    loop_points, loop_rows, fractions = sensors.loop_quadrature
    rows[loops] = 0.0
    if len(loop_points):
        np.add.at(
            rows,
            loop_rows,
            virtual_flux_loops(biot, loop_points).values * fractions,
        )
    angle = np.deg2rad(sensors.angle[~loops])
    direction = np.column_stack((np.cos(angle), np.sin(angle)))
    rows[~loops] = virtual_poloidal_probes(biot, points[~loops], direction).values
    return rows


