"""Probe the step rule of the held-map pass on the Solov'ev known state.

Runs the known state of the known-answer recovery record through
:class:`nova.equilibrium.flux_iteration.FluxIterationFit` for a list of
``step_rule``/``relaxation`` configurations in sequence inside one allocation,
printing one JSON line per pass and one result line per configuration. It
introduces no second construction: the case, rows and starting field come from
the same harness ``run.py`` uses.

    python scripts/prototypes/fit_inside_iteration/probe_step_rule.py \
        --output <dir> [--configs contraction:1.0,picard:0.5]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from scripts.prototypes.fit_inside_iteration.run import (  # noqa: E402
    _Factory,
    _known_case,
)


def _progress(config):
    def report(index, misfit, update):
        print(
            "PASS "
            + json.dumps(
                dict(config=config, index=index, misfit=misfit, update=update)
            ),
            flush=True,
        )

    return report


def _run(rule, relaxation, case, args):
    from nova.equilibrium.flux_iteration import FluxIterationFit

    config = f"{rule}:{relaxation}"
    fitter = FluxIterationFit(
        case["profile"],
        case["magnetics"],
        case["rows"],
        coefficient_basis=case["basis"],
        prescribed_current=case["prescribed_current"],
    )
    fitted = fitter.fit(
        case["measurement"],
        case["initial_flux"],
        plasma_current_scale=case["plasma_current"],
        on_pass=_progress(config),
        max_passes=args.max_passes,
        flux_tolerance=args.flux_tolerance,
        relaxation=relaxation,
        step_rule=rule,
    )
    flux = np.asarray(fitted.terminal_state.flux)
    span = abs(float(case["generated_topology"].flux_span))
    error = float(
        np.max(
            np.abs(flux[: case["cells"]] - case["generated_flux"][: case["cells"]])[
                case["core"]
            ]
        )
        / span
    )
    residual = float(np.max(np.abs(fitted.whitened_residuals)))
    stopped_pass = (
        fitted.stalled_pass
        if fitted.stalled_pass is not None
        else fitted.rejected_pass
        if fitted.rejected_pass is not None
        else fitted.axis_lost_pass
    )
    stopped_rule = (
        "converged"
        if fitted.converged
        else "stalled"
        if fitted.stalled_pass is not None
        else "rejected"
        if fitted.rejected_pass is not None
        else "axis-lost"
        if fitted.axis_lost_pass is not None
        else "budget"
    )
    return dict(
        config=config,
        step_rule=rule,
        relaxation=relaxation,
        converged=bool(fitted.converged),
        passes=len(fitted.passes),
        pass_fraction=[item.fraction for item in fitted.passes],
        pass_update=[item.flux_update for item in fitted.passes],
        pass_misfit=[item.misfit for item in fitted.passes],
        stopped_rule=stopped_rule,
        stopped_pass=stopped_pass,
        flux_map_error_over_span=error,
        whitened_residual_sup=residual,
        bands_met=bool(error <= 1.0e-6 and residual <= 1.0e-8),
        last_flux_update=float(fitted.last_flux_update),
        coefficients=fitted.coefficients.tolist(),
        singular_values=fitted.singular_values.tolist(),
        flux_evaluations=fitted.flux_evaluations,
        wall_seconds=fitted.wall_seconds,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case", default="solovev")
    parser.add_argument(
        "--configs",
        default="cycle:1.0,contraction:1.0,contraction:0.5,picard:0.5,picard:0.25",
    )
    parser.add_argument("--max-passes", type=int, default=64)
    parser.add_argument("--flux-tolerance", type=float, default=1.0e-10)
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    import jax

    from nova.jax.config import configure_dtypes

    configure_dtypes()
    print(
        "PROBE_HEADER "
        + json.dumps(
            dict(device=str(jax.devices()[0]), module=__file__, case=args.case)
        ),
        flush=True,
    )
    factory = _Factory(output / "evidence")
    case = _known_case(args.case, factory)
    results = []
    for spec in [item for item in args.configs.split(",") if item]:
        rule, _, value = spec.partition(":")
        row = _run(rule, float(value or 1.0), case, args)
        results.append(row)
        (output / f"{row['config'].replace(':', '-')}.json").write_text(
            json.dumps(row, indent=2)
        )
        print("PROBE_RESULT " + json.dumps(row), flush=True)
    (output / "probe-table.json").write_text(json.dumps(results, indent=2))
    print("PROBE_TABLE " + json.dumps(results), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())