"""Inventory EFM post-processed quantities against ``FluxSurfaceGeometry``.

The output is a machine-readable evidence record.  It inventories only EFM
post-processing outputs: equilibrium inputs, the flux map itself, detector
predictions, fit diagnostics, configuration flags and coordinate arrays are
deliberately outside the derived set.  Every admitted item is required in all
selected shots and its stored metadata is reported without unit rewriting.
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FROZEN_SHOTS = (21978, 21983, 21985, 21986, 21989, 22086)
DEFAULT_STORE = Path("/work/projects/imas_gpu/mast/level1/shots")


@dataclass(frozen=True)
class Pairing:
    """Interpretation of one stored EFM post-processing output."""

    name: str
    classification: str
    nova: str | None
    coordinate: str
    domain: str
    normalisation: str
    resampling: str
    reason: str


def _pair(
    name: str,
    nova: str,
    *,
    coordinate: str = "time scalar",
    domain: str = "one converged equilibrium",
    normalisation: str = "stored physical units",
    resampling: str = "none",
) -> Pairing:
    return Pairing(
        name,
        "paired",
        nova,
        coordinate,
        domain,
        normalisation,
        resampling,
        "",
    )


def _absent(
    name: str,
    reason: str,
    *,
    coordinate: str = "time scalar",
    domain: str = "one converged equilibrium",
    normalisation: str = "stored definition",
) -> Pairing:
    return Pairing(
        name,
        "absent-in-nova",
        None,
        coordinate,
        domain,
        normalisation,
        "not applicable",
        reason,
    )


def _unpairable(
    name: str,
    reason: str,
    *,
    coordinate: str = "time scalar",
    domain: str = "one converged equilibrium",
    normalisation: str = "not sufficiently specified by store metadata",
) -> Pairing:
    return Pairing(
        name,
        "unpairable",
        None,
        coordinate,
        domain,
        normalisation,
        "none; comparison is prohibited",
        reason,
    )


PSI_PROFILE = {
    "coordinate": "efm/psi_norm, axis=0 to LCFS=1",
    "domain": "nested closed surfaces through the stored LCFS",
    "normalisation": "source COCOS 3 normalised poloidal flux",
    "resampling": (
        "monotone linear interpolation of the Nova value against "
        "FluxSurfaceGeometry.psi_norm onto efm/psi_norm; no extrapolation"
    ),
}


def _profile_pair(name: str, nova: str, **overrides: str) -> Pairing:
    """Return a pairing on the common stored poloidal-flux coordinate."""
    return _pair(name, nova, **{**PSI_PROFILE, **overrides})


def _profile_absent(name: str, reason: str) -> Pairing:
    """Return an absent Nova field on the stored poloidal-flux coordinate."""
    return _absent(
        name,
        reason,
        coordinate=PSI_PROFILE["coordinate"],
        domain=PSI_PROFILE["domain"],
        normalisation=PSI_PROFILE["normalisation"],
    )


PAIRINGS = (
    _profile_pair("psi_norm", "psi_norm"),
    _profile_pair("areap_c", "area"),
    _profile_pair(
        "fpsi_c",
        "field_function",
        normalisation="F=R*Bphi in T m; unchanged from source COCOS 3",
    ),
    _profile_pair(
        "qpsi_c",
        "safety_factor",
        normalisation="multiply source COCOS 3 q by -1 for COCOS 17",
    ),
    _profile_pair("volp_c", "volume"),
    _pair("plasma_area", "area[-1]", domain="stored LCFS"),
    _pair("plasma_volume", "volume[-1]", domain="stored LCFS"),
    _pair(
        "q_axis",
        "safety_factor at psi_norm=0",
        normalisation="multiply source COCOS 3 q by -1 for COCOS 17",
    ),
    _pair(
        "q_90",
        "safety_factor at psi_norm=0.90",
        normalisation="multiply source COCOS 3 q by -1 for COCOS 17",
        resampling="monotone linear interpolation against Nova psi_norm",
    ),
    _pair(
        "q_95",
        "safety_factor at psi_norm=0.95",
        normalisation="multiply source COCOS 3 q by -1 for COCOS 17",
        resampling="monotone linear interpolation against Nova psi_norm",
    ),
    _pair(
        "q_100",
        "safety_factor at the outermost resolvable surface",
        domain="stored LCFS versus Nova's last pre-separatrix surface",
        normalisation="multiply source COCOS 3 q by -1 for COCOS 17",
        resampling=(
            "compare only at Nova's maximum psi_norm and label the EFIT "
            "separatrix value non-coincident; never extrapolate to psi_norm=1"
        ),
    ),
    _pair(
        "bvac_val",
        "vacuum_field",
        normalisation="vacuum toroidal field in T at efm/bvac_r",
    ),
    _pair("bvac_r", "reference_radius", normalisation="metres"),
    _pair(
        "psi_axis",
        "axis_flux",
        normalisation="multiply Wb/rad by 2*pi to Nova total poloidal flux Wb",
    ),
    _pair(
        "psi_boundary",
        "boundary_flux",
        domain="stored LCFS",
        normalisation="multiply Wb/rad by 2*pi to Nova total poloidal flux Wb",
    ),
    _pair("magnetic_axis_r", "magnetic_axis[0]", normalisation="metres"),
    _pair("magnetic_axis_z", "magnetic_axis[1]", normalisation="metres"),
    _profile_absent("elongpsi_c", "FluxSurfaceGeometry carries no surface elongation"),
    _profile_absent("pol_length", "FluxSurfaceGeometry carries no poloidal perimeter"),
    _profile_absent(
        "ppsi_c", "pressure is a source profile, not a FluxSurfaceGeometry field"
    ),
    _profile_absent("pwpsi_c", "rotational pressure is outside FluxSurfaceGeometry"),
    _profile_absent(
        "triang_lpsi_c", "FluxSurfaceGeometry carries no lower triangularity"
    ),
    _profile_absent(
        "triang_upsi_c", "FluxSurfaceGeometry carries no upper triangularity"
    ),
    _absent("betan", "FluxSurfaceGeometry carries no normalised beta"),
    _absent("betap", "FluxSurfaceGeometry carries no poloidal beta"),
    _unpairable(
        "betapd",
        "a diamagnetic-flux calibration and its pressure convention would have to be invented",
    ),
    _absent("betat", "FluxSurfaceGeometry carries no toroidal beta"),
    _unpairable(
        "betatd",
        "a diamagnetic-flux calibration and its toroidal-field convention would have to be invented",
    ),
    _absent(
        "bphi_rgeom", "FluxSurfaceGeometry carries no total field at the geometric axis"
    ),
    _absent(
        "bphi_rmag", "FluxSurfaceGeometry carries no total field at the magnetic axis"
    ),
    _absent(
        "bphi_squared", "FluxSurfaceGeometry carries no toroidal-field volume integral"
    ),
    _absent(
        "bpol_squared", "FluxSurfaceGeometry carries no poloidal-field volume integral"
    ),
    _absent(
        "bvac_rgeom",
        "FluxSurfaceGeometry carries no field sampled at the geometric axis",
    ),
    _absent(
        "bvac_rmag", "FluxSurfaceGeometry carries no field sampled at the magnetic axis"
    ),
    _unpairable(
        "cm_bdry",
        "the detected-boundary algorithm and its acceptance threshold are not stored",
        coordinate="normalised poloidal flux scalar",
        domain="detected boundary surface",
    ),
    _absent("current_centrd_r", "FluxSurfaceGeometry carries no current centroid"),
    _absent("current_centrd_z", "FluxSurfaceGeometry carries no current centroid"),
    _unpairable(
        "diamag_fluxc",
        "the diamagnetic-loop calibration and subtraction convention are not in the store",
    ),
    _absent(
        "elongation",
        "FluxSurfaceGeometry carries no LCFS elongation",
        domain="stored LCFS",
    ),
    _unpairable(
        "elongation_axis",
        "the axis differential limit needs a derivative stencil convention not stored",
        domain="magnetic-axis limit",
    ),
    _absent("geom_axis_rc", "FluxSurfaceGeometry carries no geometric-axis radius"),
    _absent("geom_axis_zc", "FluxSurfaceGeometry carries no geometric-axis height"),
    _absent(
        "lcfs_length",
        "FluxSurfaceGeometry carries no LCFS perimeter",
        domain="stored LCFS",
    ),
    _absent(
        "lcfs_r",
        "FluxSurfaceGeometry carries no boundary contour",
        coordinate="boundary vertex",
    ),
    _absent(
        "lcfs_z",
        "FluxSurfaceGeometry carries no boundary contour",
        coordinate="boundary vertex",
    ),
    _absent("lcfsn_c", "FluxSurfaceGeometry carries no boundary representation count"),
    _absent("li", "FluxSurfaceGeometry carries no internal inductance"),
    _absent(
        "minor_radius",
        "FluxSurfaceGeometry carries no geometric minor radius",
        domain="stored LCFS",
    ),
    _absent("plasma_current_c", "FluxSurfaceGeometry carries no total plasma current"),
    _absent(
        "plasma_current_rz",
        "FluxSurfaceGeometry carries no current-density field",
        coordinate="efm/gridz by efm/gridr",
        domain="stored reconstruction grid",
    ),
    _absent("plasma_energy", "FluxSurfaceGeometry carries no stored thermal energy"),
    _absent(
        "q=1_radius", "FluxSurfaceGeometry carries no resonant-surface midplane radius"
    ),
    _absent(
        "q=2_radius", "FluxSurfaceGeometry carries no resonant-surface midplane radius"
    ),
    _absent(
        "q=3_radius", "FluxSurfaceGeometry carries no resonant-surface midplane radius"
    ),
    _absent(
        "qr",
        "FluxSurfaceGeometry carries no midplane-radius profile",
        coordinate="efm/rvals midplane radius",
        domain="Z=0 radial chord",
    ),
    _absent("qstar", "FluxSurfaceGeometry carries no engineering q-star estimate"),
    _absent("rpsi100_in", "FluxSurfaceGeometry carries no inboard surface intercept"),
    _absent("rpsi100_out", "FluxSurfaceGeometry carries no outboard surface intercept"),
    _absent("rpsi90_in", "FluxSurfaceGeometry carries no inboard surface intercept"),
    _absent("rpsi90_out", "FluxSurfaceGeometry carries no outboard surface intercept"),
    _absent("rpsi95_in", "FluxSurfaceGeometry carries no inboard surface intercept"),
    _absent("rpsi95_out", "FluxSurfaceGeometry carries no outboard surface intercept"),
    _unpairable("rt", "the Shafranov RT definition and normalisation are not stored"),
    _unpairable(
        "shaf_integral_1", "the integral's integrand and normalisation are not stored"
    ),
    _unpairable(
        "shaf_integral_2", "the integral's integrand and normalisation are not stored"
    ),
    _unpairable(
        "shaf_integral_3", "the integral's integrand and normalisation are not stored"
    ),
    _absent("triang_lower", "FluxSurfaceGeometry carries no lower LCFS triangularity"),
    _absent("triang_upper", "FluxSurfaceGeometry carries no upper LCFS triangularity"),
    _unpairable(
        "wplasmd",
        "a diamagnetic-flux calibration and energy convention would have to be invented",
    ),
    _absent("wpol", "FluxSurfaceGeometry carries no poloidal magnetic energy"),
    _absent("xpoint1_rc", "FluxSurfaceGeometry carries no X-point geometry"),
    _absent("xpoint1_zc", "FluxSurfaceGeometry carries no X-point geometry"),
    _absent("xpoint2_rc", "FluxSurfaceGeometry carries no X-point geometry"),
    _absent("xpoint2_zc", "FluxSurfaceGeometry carries no X-point geometry"),
)


NON_DERIVED_ARRAYS = {
    "all_times",
    "chisq_magnetic",
    "cnvrgd_times",
    "cutip",
    "diamag_fluxx",
    "fcoil_ang1",
    "fcoil_ang2",
    "fcoil_c",
    "fcoil_chisq",
    "fcoil_circ",
    "fcoil_height",
    "fcoil_n",
    "fcoil_r",
    "fcoil_turns",
    "fcoil_width",
    "fcoil_x",
    "fcoil_xmult",
    "fcoil_z",
    "fcurbd",
    "ffprime",
    "ffprime_coefs",
    "ffprime_coefs_n",
    "final_chisq",
    "fwtbdry",
    "fwtbp",
    "fwtfc",
    "fwtmp",
    "fwtsi",
    "gridr",
    "gridz",
    "ip_times",
    "irod",
    "iteration_error",
    "jr",
    "kffcur",
    "kfffnc",
    "kppcur",
    "kppfnc",
    "kwwcur",
    "kwwfnc",
    "lcfs_coords",
    "limiterr",
    "limiterz",
    "mag_probe_n",
    "magpr_ang",
    "magpr_c",
    "magpr_len",
    "magpr_r",
    "magpr_x",
    "magpr_z",
    "magpri_chisq",
    "n_iterations",
    "nh",
    "npress",
    "num_iterations",
    "nw",
    "p2ar_c",
    "p2br_c",
    "p2cr_c",
    "passnumber",
    "pcurbd",
    "plasma_current_x",
    "pprime",
    "pprime_coefs",
    "pprime_coefs_n",
    "pprimew",
    "pr_c",
    "profile_r",
    "profile_z",
    "psi_loop_n",
    "psir",
    "psirz",
    "r",
    "rbdry",
    "rvals",
    "rvtor",
    "scalepr",
    "sigbdry",
    "silop_c",
    "silop_chisq",
    "silop_dphi",
    "silop_r",
    "silop_x",
    "silop_z",
    "status",
    "time",
    "time_",
    "wcurbd",
    "z",
    "zbdry",
}


NOVA_FIELDS = {
    "rho_tor_norm": ("unmatched", "EFM stores no toroidal-flux coordinate"),
    "rho_tor": ("unmatched", "EFM stores no toroidal-flux radius"),
    "psi_norm": ("matched", "efm/psi_norm"),
    "poloidal_flux": ("matched", "efm/psi_norm with efm/psi_axis and efm/psi_boundary"),
    "toroidal_flux": ("unmatched", "EFM stores no enclosed toroidal flux"),
    "volume": ("matched", "efm/volp_c and efm/plasma_volume"),
    "area": ("matched", "efm/areap_c and efm/plasma_area"),
    "volume_derivative": (
        "unmatched",
        "EFM stores no derivative with respect to rho_tor",
    ),
    "volume_flux_derivative": (
        "unmatched",
        "EFM stores no derivative with respect to poloidal flux",
    ),
    "field_function": ("matched", "efm/fpsi_c"),
    "safety_factor": ("matched", "efm/qpsi_c and scalar q samples"),
    "inverse_square_radius": ("unmatched", "EFM stores no <1/R^2> surface average"),
    "gradient_rho": ("unmatched", "EFM stores no <|grad rho|> surface average"),
    "gradient_rho_squared": (
        "unmatched",
        "EFM stores no <|grad rho|^2> surface average",
    ),
    "gradient_rho_squared_over_radius_squared": (
        "unmatched",
        "EFM stores no <|grad rho|^2/R^2> surface average",
    ),
    "boundary_rho_tor": ("unmatched", "EFM stores no toroidal-flux boundary radius"),
    "vacuum_field": ("matched", "efm/bvac_val"),
    "reference_radius": ("matched", "efm/bvac_r"),
    "axis_flux": ("matched", "efm/psi_axis after multiplication by 2*pi"),
    "boundary_flux": ("matched", "efm/psi_boundary after multiplication by 2*pi"),
    "magnetic_axis": ("matched", "efm/magnetic_axis_r and efm/magnetic_axis_z"),
}


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _metadata(array: Any) -> dict[str, Any]:
    attributes = dict(array.attrs)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "dimensions": _json_value(attributes.get("_ARRAY_DIMENSIONS", [])),
        "units": attributes.get("units", ""),
        "description": attributes.get("description"),
    }


def build_report(store: Path, shots: tuple[int, ...] = FROZEN_SHOTS) -> dict[str, Any]:
    """Read the selected stores and return the complete pairing census."""
    import zarr

    names = tuple(item.name for item in PAIRINGS)
    if len(names) != len(set(names)):
        raise ValueError("the derived-quantity catalogue contains duplicate paths")
    if set(NOVA_FIELDS) != {
        "rho_tor_norm",
        "rho_tor",
        "psi_norm",
        "poloidal_flux",
        "toroidal_flux",
        "volume",
        "area",
        "volume_derivative",
        "volume_flux_derivative",
        "field_function",
        "safety_factor",
        "inverse_square_radius",
        "gradient_rho",
        "gradient_rho_squared",
        "gradient_rho_squared_over_radius_squared",
        "boundary_rho_tor",
        "vacuum_field",
        "reference_radius",
        "axis_flux",
        "boundary_flux",
        "magnetic_axis",
    }:
        raise ValueError("Nova field catalogue must exactly match FluxSurfaceGeometry")

    shapes: dict[str, dict[str, list[int]]] = {name: {} for name in names}
    reference: dict[str, dict[str, Any]] = {}
    arrays_per_shot: dict[str, int] = {}
    for shot in shots:
        group = zarr.open_group(str(store / f"{shot}.zarr"), mode="r")["efm"]
        stored_names = set(group.array_keys())
        arrays_per_shot[str(shot)] = len(stored_names)
        unreviewed = sorted(stored_names.difference(names, NON_DERIVED_ARRAYS))
        missing_reviewed = sorted(
            set(names).union(NON_DERIVED_ARRAYS).difference(stored_names)
        )
        if unreviewed or missing_reviewed:
            raise ValueError(
                f"shot {shot} changed EFM schema: unreviewed={unreviewed}, "
                f"missing_reviewed={missing_reviewed}"
            )
        missing = sorted(set(names).difference(group.array_keys()))
        if missing:
            raise ValueError(f"shot {shot} is missing derived arrays {missing}")
        for name in names:
            item = _metadata(group[name])
            shapes[name][str(shot)] = item["shape"]
            if name not in reference:
                reference[name] = item
            elif {
                key: item[key]
                for key in ("dtype", "dimensions", "units", "description")
            } != {
                key: reference[name][key]
                for key in ("dtype", "dimensions", "units", "description")
            }:
                raise ValueError(f"metadata for efm/{name} differs in shot {shot}")

    quantities = []
    for pairing in PAIRINGS:
        quantities.append(
            {
                "store_path": f"efm/{pairing.name}",
                **reference[pairing.name],
                "shapes_by_shot": shapes[pairing.name],
                "flux_coordinate": pairing.coordinate,
                "domain": pairing.domain,
                "normalisation": pairing.normalisation,
                "resampling_rule": pairing.resampling,
                "classification": pairing.classification,
                "nova_counterpart": pairing.nova,
                "reason": pairing.reason,
            }
        )

    class_counts = {
        classification: sum(item.classification == classification for item in PAIRINGS)
        for classification in ("paired", "absent-in-nova", "unpairable")
    }
    enumerated_total = len(PAIRINGS)
    if sum(class_counts.values()) != enumerated_total:
        raise ValueError("classification counts do not cover the derived set")
    nova_counts = {
        status: sum(value[0] == status for value in NOVA_FIELDS.values())
        for status in ("matched", "unmatched")
    }
    if sum(nova_counts.values()) != 21:
        raise ValueError("Nova-side accounting must contain exactly 21 fields")

    return {
        "store": str(store),
        "shots": list(shots),
        "scope": {
            "included": "EFM post-processing outputs and geometry landmarks",
            "excluded": (
                "equilibrium inputs, psirz, detector predictions, fit diagnostics, "
                "configuration flags and bare coordinate arrays except psi_norm"
            ),
            "arrays_per_shot": arrays_per_shot,
            "reviewed_array_total": len(PAIRINGS) + len(NON_DERIVED_ARRAYS),
            "excluded_array_count": len(NON_DERIVED_ARRAYS),
            "excluded_arrays": sorted(NON_DERIVED_ARRAYS),
        },
        "source_convention": {
            "cocos": 3,
            "target_cocos": 17,
            "poloidal_flux": "multiply Wb/rad by 2*pi",
            "safety_factor": "multiply by -1",
            "plasma_current": "apply the separate per-shot sign census",
        },
        "flux_surface_average_inventory": {
            "explicit_average_paths": [],
            "statement": (
                "All 164 arrays were reviewed; EFM stores surface geometry and "
                "flux-function profiles but no explicit flux-surface average."
            ),
        },
        "quantities": quantities,
        "class_counts": class_counts,
        "enumerated_total": enumerated_total,
        "nova_fields": [
            {"name": name, "status": status, "evidence": evidence}
            for name, (status, evidence) in NOVA_FIELDS.items()
        ],
        "nova_class_counts": nova_counts,
        "nova_total": 21,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args.store)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
