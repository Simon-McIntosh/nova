"""Classify EFM quantities that are absent from ``FluxSurfaceGeometry``.

This benchmark is an evidence generator, not a production implementation.  It
closes over the census, verifies every cited Nova symbol against its source,
and records a repository-wide identifier search for quantities classified as
absent.  Its JSON output separates independent cross-check opportunities from
fields that would only enlarge Nova's public record.
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from efit_derived_quantity_census import PAIRINGS


ROOT = Path(__file__).resolve().parents[1]
NOVA_ROOT = ROOT / "nova"
CONTRACT_REFERENCE = "docs/plans/flux-function-forward-equilibrium.html#s5"
CONTRACT_FIELDS = (
    "V-prime",
    "inverse-square-radius surface average",
    "diffusion-coefficient surface averages",
    "F on the toroidal-flux coordinate",
    "safety factor on the toroidal-flux coordinate",
)


@dataclass(frozen=True)
class Triage:
    """One census quantity and the evidence for its disposition."""

    name: str
    classification: str
    evidence: str
    convention_compatibility: str
    reference_crosscheck: bool | None
    crosscheck_reason: str
    module_path: str | None = None
    symbol: str | None = None
    search_terms: tuple[str, ...] = ()


def _computed(
    name: str,
    module_path: str,
    symbol: str,
    compatibility: str,
    crosscheck: bool,
    reason: str,
) -> Triage:
    return Triage(
        name=name,
        classification="COMPUTED-ELSEWHERE",
        evidence=f"{module_path}:{symbol}",
        convention_compatibility=compatibility,
        reference_crosscheck=crosscheck,
        crosscheck_reason=reason,
        module_path=module_path,
        symbol=symbol,
    )


def _absent(name: str, *search_terms: str, reason: str) -> Triage:
    return Triage(
        name=name,
        classification="GENUINELY-ABSENT",
        evidence="repository-wide Python identifier search under nova/",
        convention_compatibility="not applicable; Nova has no definition",
        reference_crosscheck=None,
        crosscheck_reason=reason,
        search_terms=search_terms,
    )


INTERIOR_SURFACE = (
    "Compatible on coincident closed contours: both definitions use the extrema "
    "or arc length of the contour at the requested normalised poloidal flux."
)
INTERIOR_CHECK = (
    "Valid after interpolation on the common interior psi_norm domain; no LCFS "
    "endpoint or extrapolation is involved."
)
BOUNDARY_MISMATCH = (
    "Not a valid scalar cross-check: EFM encloses its scalar to the stored LCFS, "
    "whereas Nova's traced record stops at psi_N=0.995.  The existing comparison "
    "already showed about one-percent boundary scalar offsets beside 3e-4 interior "
    "profile agreement."
)


TRIAGE = (
    _computed(
        "elongpsi_c",
        "nova/geometry/curve.py",
        "Elongation.elongation",
        INTERIOR_SURFACE,
        True,
        INTERIOR_CHECK,
    ),
    _computed(
        "pol_length",
        "nova/geometry/curve.py",
        "Curve.length",
        INTERIOR_SURFACE,
        True,
        INTERIOR_CHECK,
    ),
    _computed(
        "ppsi_c",
        "nova/equilibrium/source.py",
        "DomainProfile.pressure",
        "Compatible for the static closure: both are pressure in Pa as a function of normalised poloidal flux.  A rotating solve must instead use its declared radius-dependent primitive.",
        True,
        "Valid for a static source on the common interior psi_norm samples, with the same boundary-pressure primitive and flux convention.",
    ),
    _computed(
        "pwpsi_c",
        "nova/equilibrium/rotation.py",
        "RotatingDomainProfile.reference_pressure",
        "Potentially related but not convention-pinned: Nova defines pressure at its declared reference major radius, while the store only calls this p_omega and does not identify its reference radius.",
        False,
        "Not valid without the missing reference-radius and rotational-closure convention.",
    ),
    _computed(
        "triang_lpsi_c",
        "nova/geometry/curve.py",
        "Triangularity.triangularity_lower",
        INTERIOR_SURFACE,
        True,
        INTERIOR_CHECK,
    ),
    _computed(
        "triang_upsi_c",
        "nova/geometry/curve.py",
        "Triangularity.triangularity_upper",
        INTERIOR_SURFACE,
        True,
        INTERIOR_CHECK,
    ),
    _absent(
        "betan",
        "betan",
        "normalized_beta",
        "normalised_beta",
        reason="Nova has no normalised-beta definition; adding it would introduce an engineering normalisation rather than satisfy the transport geometry contract.",
    ),
    _computed(
        "betap",
        "nova/equilibrium/observation.py",
        "IntegralObservation.poloidal_beta",
        "Incompatible denominator conventions: Nova uses 4 integral(p dV)/(mu0 R_volume Ip^2); EFM specifies 2 mu0 <p>/<Bp>_LCFS^2.",
        False,
        "Not a valid numerical cross-check until a common boundary-average and reference-radius convention is declared.",
    ),
    _absent(
        "betat",
        "betat",
        "toroidal_beta",
        reason="Nova has no toroidal-beta observation using the EFM vacuum-field-at-axis convention.",
    ),
    _absent(
        "bphi_rgeom",
        "bphi_rgeom",
        "toroidal_field_at_geometric_axis",
        reason="Nova computes the field function but no total toroidal-field sample at the geometric axis.",
    ),
    _absent(
        "bphi_rmag",
        "bphi_rmag",
        "toroidal_field_at_magnetic_axis",
        reason="Nova computes the field function but no total toroidal-field sample at the magnetic axis.",
    ),
    _absent(
        "bphi_squared",
        "bphi_squared",
        "toroidal_field_integral",
        reason="Nova does not form the EFM plasma-volume integral of total Bphi squared.",
    ),
    _computed(
        "bpol_squared",
        "nova/equilibrium/observation.py",
        "IntegralObservation.poloidal_field_integral",
        "Compatible: both are the plasma-volume integral of Bp squared in m^3 T^2.",
        True,
        "Valid on the same labelled core and cell quadrature; compare before applying any energy factor.",
    ),
    _absent(
        "bvac_rgeom",
        "bvac_rgeom",
        "vacuum_field_at_geometric_axis",
        reason="Nova stores a vacuum-field reference pair but does not publish this resampled scalar.",
    ),
    _absent(
        "bvac_rmag",
        "bvac_rmag",
        "vacuum_field_at_magnetic_axis",
        reason="Nova stores a vacuum-field reference pair but does not publish this resampled scalar.",
    ),
    _computed(
        "current_centrd_r",
        "nova/equilibrium/moment.py",
        "MomentInversion.centroid_r",
        "Compatible: both are the toroidal-current-weighted R coordinate.",
        True,
        "Valid when both use the same reconstructed current-density grid and current sign.",
    ),
    _computed(
        "current_centrd_z",
        "nova/equilibrium/moment.py",
        "MomentInversion.centroid_z",
        "Compatible: both are the toroidal-current-weighted Z coordinate.",
        True,
        "Valid when both use the same reconstructed current-density grid and current sign.",
    ),
    _computed(
        "elongation",
        "nova/geometry/curve.py",
        "Elongation.elongation",
        "Compatible algebraic definition, (Zmax-Zmin)/(Rmax-Rmin), but evaluated on different outer contours in the available records.",
        False,
        BOUNDARY_MISMATCH,
    ),
    _computed(
        "geom_axis_rc",
        "nova/geometry/curve.py",
        "PointGeometry.geometric_radius",
        "Compatible bounding-box midpoint definition, but evaluated on different outer contours.",
        False,
        BOUNDARY_MISMATCH,
    ),
    _computed(
        "geom_axis_zc",
        "nova/geometry/curve.py",
        "PointGeometry.geometric_height",
        "Compatible bounding-box midpoint definition, but evaluated on different outer contours.",
        False,
        BOUNDARY_MISMATCH,
    ),
    _computed(
        "lcfs_length",
        "nova/geometry/curve.py",
        "Curve.length",
        "Compatible polygonal arc-length definition, but evaluated on different outer contours.",
        False,
        BOUNDARY_MISMATCH,
    ),
    _computed(
        "lcfs_r",
        "nova/equilibrium/moment.py",
        "MomentReconstruction.ring",
        "Compatible as boundary R vertices after an explicit common resampling, but Nova's ring and EFM's stored LCFS are not coincident here.",
        False,
        BOUNDARY_MISMATCH,
    ),
    _computed(
        "lcfs_z",
        "nova/equilibrium/moment.py",
        "MomentReconstruction.ring",
        "Compatible as boundary Z vertices after an explicit common resampling, but Nova's ring and EFM's stored LCFS are not coincident here.",
        False,
        BOUNDARY_MISMATCH,
    ),
    _absent(
        "lcfsn_c",
        "lcfsn_c",
        "boundary_point_count",
        reason="A representation vertex count is not a physical metric and Nova intentionally has no corresponding quantity.",
    ),
    _computed(
        "li",
        "nova/equilibrium/observation.py",
        "IntegralObservation.internal_inductance",
        "Not convention-compatible as stored: Nova normalises 2 integral(Bp^2 dV)/(mu0^2 R_volume Ip^2), while EFM specifies volume-average Bp squared divided by an LCFS surface-average squared.",
        False,
        "Not valid until the reference radius and LCFS surface-average conventions are made identical.",
    ),
    _computed(
        "minor_radius",
        "nova/geometry/curve.py",
        "PointGeometry.minor_radius",
        "Compatible half-width definition, but evaluated on different outer contours.",
        False,
        BOUNDARY_MISMATCH,
    ),
    _computed(
        "plasma_current_c",
        "nova/equilibrium/observation.py",
        "IntegralObservation.plasma_current",
        "Compatible: both integrate toroidal current density over the confined plasma cross-section.",
        True,
        "Valid on the same grid, core label and current-sign convention.",
    ),
    _computed(
        "plasma_current_rz",
        "nova/equilibrium/source.py",
        "ForwardSource.current_density",
        "Compatible physical quantity, toroidal current density in A/m^2, after matching grid layout and current sign.",
        True,
        "Valid cellwise after resampling onto a common R-Z grid and applying the already pinned per-shot sign convention.",
    ),
    _computed(
        "plasma_energy",
        "nova/equilibrium/observation.py",
        "IntegralObservation.pressure_integral",
        "Not convention-compatible without an equation-of-state factor: Nova stores integral(p dV), while EFM publishes thermal energy without stating the pressure-to-energy multiplier or species convention.",
        False,
        "The stored scalar cannot cross-check Nova until its thermodynamic energy convention is supplied.",
    ),
    _absent(
        "q=1_radius",
        "q_1_radius",
        "q1_radius",
        reason="Nova does not locate or publish the magnetic-axis chord half-width of the q=1 surface.",
    ),
    _absent(
        "q=2_radius",
        "q_2_radius",
        "q2_radius",
        reason="Nova does not locate or publish the magnetic-axis chord half-width of the q=2 surface.",
    ),
    _absent(
        "q=3_radius",
        "q_3_radius",
        "q3_radius",
        reason="Nova does not locate or publish the magnetic-axis chord half-width of the q=3 surface.",
    ),
    _absent(
        "qr",
        "safety_factor_midplane",
        reason="Nova has no safety-factor profile parameterised by the Z=0 radial chord.",
    ),
    _absent(
        "qstar",
        "qstar",
        "q_star",
        reason="Nova has no engineering q-star estimate; it is not the contract safety factor on toroidal-flux coordinate.",
    ),
    _absent(
        "rpsi100_in",
        "rpsi100_in",
        "psi100_inboard_radius",
        reason="Nova has no named inboard intercept at psi_N=1.",
    ),
    _absent(
        "rpsi100_out",
        "rpsi100_out",
        "psi100_outboard_radius",
        reason="Nova has no named outboard intercept at psi_N=1.",
    ),
    _absent(
        "rpsi90_in",
        "rpsi90_in",
        "psi90_inboard_radius",
        reason="Nova has no named inboard intercept at psi_N=0.90.",
    ),
    _absent(
        "rpsi90_out",
        "rpsi90_out",
        "psi90_outboard_radius",
        reason="Nova has no named outboard intercept at psi_N=0.90.",
    ),
    _absent(
        "rpsi95_in",
        "rpsi95_in",
        "psi95_inboard_radius",
        reason="Nova has no named inboard intercept at psi_N=0.95.",
    ),
    _absent(
        "rpsi95_out",
        "rpsi95_out",
        "psi95_outboard_radius",
        reason="Nova has no named outboard intercept at psi_N=0.95.",
    ),
    _computed(
        "triang_lower",
        "nova/geometry/curve.py",
        "Triangularity.triangularity_lower",
        "Compatible extrema-based definition, but evaluated on different outer contours.",
        False,
        BOUNDARY_MISMATCH,
    ),
    _computed(
        "triang_upper",
        "nova/geometry/curve.py",
        "Triangularity.triangularity_upper",
        "Compatible extrema-based definition, but evaluated on different outer contours.",
        False,
        BOUNDARY_MISMATCH,
    ),
    _computed(
        "wpol",
        "nova/equilibrium/observation.py",
        "IntegralObservation.poloidal_field_integral",
        "Compatible after the exact EFM factor 1/(2 mu0): Nova retains the unscaled integral of Bp squared over plasma volume.",
        True,
        "Valid on the same labelled core and quadrature after applying 1/(2 mu0).",
    ),
    _computed(
        "xpoint1_rc",
        "nova/equilibrium/topology.py",
        "TopologyState.x_point",
        "Nova selects the primary X-point by flux extremum; EFM only labels first and does not store the ordering rule.",
        False,
        "Not valid until EFM's first-X-point ordering is shown to mean Nova's primary X-point.",
    ),
    _computed(
        "xpoint1_zc",
        "nova/equilibrium/topology.py",
        "TopologyState.x_point",
        "Nova selects the primary X-point by flux extremum; EFM only labels first and does not store the ordering rule.",
        False,
        "Not valid until EFM's first-X-point ordering is shown to mean Nova's primary X-point.",
    ),
    _absent(
        "xpoint2_rc",
        "xpoint2_rc",
        "secondary_x_point",
        reason="Nova retains only the primary X-point in TopologyState and has no secondary-X-point output.",
    ),
    _absent(
        "xpoint2_zc",
        "xpoint2_zc",
        "secondary_x_point",
        reason="Nova retains only the primary X-point in TopologyState and has no secondary-X-point output.",
    ),
)


def _source_symbols(path: Path) -> set[str]:
    """Open one module and return its module and class-qualified symbols."""
    tree = ast.parse(path.read_text())
    symbols: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            symbols.add(node.name)
        if not isinstance(node, ast.ClassDef):
            continue
        symbols.add(node.name)
        for child in node.body:
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                symbols.add(f"{node.name}.{child.name}")
            elif isinstance(child, ast.AnnAssign) and isinstance(
                child.target, ast.Name
            ):
                symbols.add(f"{node.name}.{child.target.id}")
    return symbols


def _identifier_hits(terms: tuple[str, ...]) -> list[dict[str, object]]:
    """Search every Nova Python file for definition-like identifier tokens."""
    patterns = tuple(re.compile(rf"\b{re.escape(term)}\b") for term in terms)
    hits = []
    for path in sorted(NOVA_ROOT.rglob("*.py")):
        for line_number, line in enumerate(
            path.read_text(errors="replace").splitlines(), 1
        ):
            stripped = line.lstrip()
            if not (
                stripped.startswith("def ")
                or stripped.startswith("class ")
                or re.match(r"[A-Za-z_][A-Za-z0-9_]*\s*[:=]", stripped)
            ):
                continue
            if any(pattern.search(stripped) for pattern in patterns):
                hits.append(
                    {
                        "path": str(path.relative_to(ROOT)),
                        "line": line_number,
                        "text": stripped,
                    }
                )
    return hits


def build_report() -> dict[str, object]:
    """Validate the classification and return its evidence record."""
    census = {item.name for item in PAIRINGS if item.classification == "absent-in-nova"}
    classified = {item.name for item in TRIAGE}
    if len(TRIAGE) != len(classified):
        raise ValueError("triage contains duplicate quantity names")
    if census != classified:
        raise ValueError(
            f"triage does not close over census: missing={sorted(census - classified)}, "
            f"extra={sorted(classified - census)}"
        )

    opened_modules: dict[str, list[str]] = {}
    searches: dict[str, dict[str, object]] = {}
    for item in TRIAGE:
        if item.classification == "COMPUTED-ELSEWHERE":
            assert item.module_path is not None and item.symbol is not None
            path = ROOT / item.module_path
            symbols = _source_symbols(path)
            if item.symbol not in symbols:
                raise ValueError(
                    f"cited symbol {item.symbol} not found in {item.module_path}"
                )
            opened_modules[item.module_path] = sorted(symbols)
        elif item.classification == "GENUINELY-ABSENT":
            hits = _identifier_hits(item.search_terms)
            if hits:
                raise ValueError(
                    f"definition search for {item.name} was not empty: {hits}"
                )
            searches[item.name] = {
                "root": "nova/",
                "files": len(tuple(NOVA_ROOT.rglob("*.py"))),
                "terms": item.search_terms,
                "definition_like_hits": hits,
            }

    counts = {
        name: sum(item.classification == name for item in TRIAGE)
        for name in (
            "NEEDED-BY-CONTRACT",
            "COMPUTED-ELSEWHERE",
            "GENUINELY-ABSENT",
        )
    }
    crosscheckable = sum(item.reference_crosscheck is True for item in TRIAGE)
    report = {
        "contract": {
            "declaring_section": CONTRACT_REFERENCE,
            "public_output": "ForwardProfile",
            "declared_fields": CONTRACT_FIELDS,
            "classification_result": (
                "None of the 47 EFM-only census entries is one of these fields. "
                "F and safety factor were already paired by the census; V-prime, "
                "inverse-square-radius and diffusion metrics exist on Nova's side "
                "but have no EFM array."
            ),
        },
        "counts": {**counts, "total": len(TRIAGE)},
        "outside_code_crosscheckable": crosscheckable,
        "nova_public_record_gain_required": counts["NEEDED-BY-CONTRACT"],
        "interpretation": (
            f"EFM can independently cross-check {crosscheckable} quantities Nova "
            "already computes elsewhere; zero of the 47 would be added merely to "
            "satisfy the transport public-output contract."
        ),
        "quantities": [asdict(item) for item in TRIAGE],
        "symbol_verification": {
            "method": "opened each cited module and parsed its Python syntax tree",
            "opened_modules": sorted(opened_modules),
        },
        "absence_search": {
            "method": (
                "searched definition-like lines in every nova/**/*.py file for "
                "each recorded exact identifier or explicit semantic alias"
            ),
            "results": searches,
        },
    }
    if sum(counts.values()) != 47:
        raise ValueError(f"class counts do not sum to 47: {counts}")
    return report


def main() -> None:
    """Write the validated triage as JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rendered = json.dumps(build_report(), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
