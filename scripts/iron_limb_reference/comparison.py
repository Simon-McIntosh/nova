"""Tabulate stored annulus values with mesh changes and comparator choices."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _norm(value):
    result = float(np.linalg.norm(value))
    if not np.isfinite(result) or result == 0:
        raise ValueError("comparison requires finite nonzero field scales")
    return result


def _row(
    mu,
    name,
    surface,
    previous_surface,
    axisymmetric,
    previous_axisymmetric,
    fit,
    scale_count=1,
    flux=False,
):
    surface = np.asarray(surface)
    previous_surface = np.asarray(previous_surface)
    axisymmetric = np.asarray(axisymmetric)
    previous_axisymmetric = np.asarray(previous_axisymmetric)
    extrapolated = np.asarray(fit["extrapolated"]) if fit["valid"] else None
    comparator = axisymmetric if extrapolated is None else extrapolated
    denominator = _norm(comparator) if flux else _norm(surface)
    fine_denominator = _norm(axisymmetric) if flux else _norm(surface)
    return dict(
        mu=mu,
        measure=name,
        surface_finest=float(surface)
        if flux
        else _norm(surface) / np.sqrt(scale_count),
        surface_last_step_relative=float(
            np.linalg.norm(surface - previous_surface) / _norm(surface)
        ),
        axisymmetric_finest=float(axisymmetric)
        if flux
        else _norm(axisymmetric) / np.sqrt(scale_count),
        axisymmetric_last_step_relative=float(
            np.linalg.norm(axisymmetric - previous_axisymmetric) / _norm(axisymmetric)
        ),
        axisymmetric_extrapolate=None
        if extrapolated is None
        else float(extrapolated)
        if flux
        else _norm(extrapolated) / np.sqrt(scale_count),
        axisymmetric_extrapolation_correction_relative=None
        if extrapolated is None
        else float(np.linalg.norm(extrapolated - axisymmetric) / _norm(extrapolated)),
        comparator="finest" if extrapolated is None else "extrapolated",
        difference_relative=float(np.linalg.norm(surface - comparator) / denominator),
        finest_difference_relative=float(
            np.linalg.norm(surface - axisymmetric) / fine_denominator
        ),
    )


def comparison_rows(receipt, vessel_count):
    """Return flux, vessel vector rms and every outside total-vector reading.

    Flux differences use the selected axisymmetric comparator as denominator.
    Field differences use the accepted finest surface field norm, preserving
    the reference error convention. Vessel values are vector rms; outside
    values are total-field magnitudes. Differences retain vector direction.
    Both the extrapolate/fallback and finest comparisons are emitted so their
    distinct comparator choices cannot be confused.
    """
    if not receipt.get("completed"):
        raise ValueError("annulus receipt is not complete")
    rows = []
    for index, fine in enumerate(receipt["surface"][-1]["rows"]):
        mu = fine["mu"]
        surface = [level["rows"][index] for level in receipt["surface"][-2:]]
        axisymmetric = [level["rows"][index] for level in receipt["axisymmetric"][-2:]]
        fit = receipt["axisymmetric_extrapolation"][index]
        if not all(value["mu"] == mu for value in surface + axisymmetric + [fit]):
            raise ValueError("permeability rows are not aligned")
        rows.append(
            _row(
                mu,
                "flux",
                surface[1]["column_flux_Wb"],
                surface[0]["column_flux_Wb"],
                axisymmetric[1]["column_flux_Wb"],
                axisymmetric[0]["column_flux_Wb"],
                fit["flux"],
                flux=True,
            )
        )
        sf, sp = (np.asarray(value["iron_average_T"]) for value in surface[::-1])
        af, ap = (np.asarray(value["iron_average_T"]) for value in axisymmetric[::-1])
        rows.append(
            _row(
                mu,
                "vessel",
                sf[:vessel_count],
                sp[:vessel_count],
                af[:vessel_count],
                ap[:vessel_count],
                fit["vessel"],
                scale_count=vessel_count,
            )
        )
        for k, outside_fit in enumerate(fit["outside"]):
            point = vessel_count + k
            air = np.asarray(surface[1]["air_T"])[point]
            for value in surface + axisymmetric:
                if not np.allclose(value["air_T"][point], air, rtol=1e-12, atol=1e-15):
                    raise ValueError("external probe field differs between mesh rows")
            rows.append(
                _row(
                    mu,
                    f"outside_{k}",
                    sf[point] + air,
                    sp[point] + air,
                    af[point] + air,
                    ap[point] + air,
                    outside_fit,
                )
            )
    return rows


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--vessel-count", type=int, default=16)
    args = parser.parse_args()
    data = args.receipt.read_bytes()
    rows = comparison_rows(json.loads(data), args.vessel_count)
    print(
        json.dumps(
            dict(
                receipt=str(args.receipt),
                receipt_sha256=hashlib.sha256(data).hexdigest(),
                rows=rows,
            ),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
