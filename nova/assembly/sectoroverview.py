"""Collate tilt and deviation data for installed TFC sectors.

Provides a consolidated view of coil positions, ILIS plane tilts, and
inter/intra-sector gaps across all installed sectors. Designed to be
updated incrementally as new sector data becomes available.
"""

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import pandas

from nova.assembly.fiducialilis import FiducialIlis
from nova.assembly.fiducialpit import FiducialPit
from nova.assembly.ilisnominal import NominalIlis
from nova.assembly.transform import Rotate


@dataclass
class SectorOverview:
    """Collate alignment data for installed TFC sectors.

    Combines per-coil position parameters, ILIS plane tilts, and gap
    measurements into a single overview for assessment of installed
    coil alignment relative to targets.

    Parameters
    ----------
    sectors : dict[int, list[int]]
        Sector to coil mapping for installed sectors
    phase : str
        Phase selection strategy ('latest', literal phase name)
    pcr : bool
        Apply PCR deviation corrections
    augment : bool
        Augment partial ILIS data using hybrid CCL+ILIS recovery
    private : bool
        Use private (in-work) data files
    """

    sectors: dict[int, list[int]] = field(
        default_factory=lambda: {5: [16, 5], 6: [12, 13], 7: [8, 9], 8: [4, 11]}
    )
    phase: str = "latest"
    pcr: bool = True
    augment: bool = True
    private: bool = False

    def __post_init__(self):
        """Build overview from pit data."""
        self.pit = FiducialPit(
            sectors=self.sectors,
            phase=self.phase,
            pcr=self.pcr,
            private=self.private,
            augment=self.augment,
        )
        self.nominal = NominalIlis()
        self.rotate = Rotate()

    @cached_property
    def coil_positions(self) -> pandas.DataFrame:
        """Per-coil position parameters from ILIS-projected CCL fiducials.

        Returns DataFrame with columns:
        - sector, phase, position: sector metadata
        - radial, tangential, vertical: translations (mm)
        - roll_length, yaw_length, pitch_length: rotation-induced offsets (mm)
        """
        rows = []
        for sector, sd in self.pit.sector_data.items():
            phase = self.pit._resolved_phases.get(sector, self.phase)
            try:
                positions = sd.extract_coil_positions(pcr=self.pcr)
                for coil, row in positions.iterrows():
                    pos_idx = self.pit.coil_position.get(coil, -1)
                    rows.append(
                        {
                            "coil": coil,
                            "sector": sector,
                            "phase": phase,
                            "position": pos_idx,
                            "angle_deg": pos_idx * 20,
                            **row.to_dict(),
                        }
                    )
            except (KeyError, ValueError):
                pass
        df = pandas.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("position").reset_index(drop=True)
        return df

    @cached_property
    def ilis_tilts(self) -> pandas.DataFrame:
        """ILIS plane tilts relative to nominal geometry.

        For each coil, measures the angular deviation of fitted ILIS
        planes from the nominal ILIS directions. Reports roll (rotation
        about radial axis) and yaw (rotation about vertical axis)
        components extracted from the normal vector deviation.

        Returns DataFrame with columns:
        - coil, sector: identification
        - feature: ILIS surface ('+1', '-1', '0')
        - roll_urad: roll tilt in microradians
        - yaw_urad: yaw tilt in microradians
        - tilt_urad: total angular tilt magnitude
        - augmented: True if surface was recovered from reference
        """
        # Get nominal normals
        nom_planes = self.nominal.planes
        nom_plus = nom_planes.loc[(0, "ILIS +1"), ["nx", "ny", "nz"]].values
        nom_minus = nom_planes.loc[(0, "ILIS -1"), ["nx", "ny", "nz"]].values
        nom_mid = nom_planes.loc[(0, "ILIS 0"), ["nx", "ny", "nz"]].values

        nom_normals = {
            "ILIS +1": nom_plus / np.linalg.norm(nom_plus),
            "ILIS -1": nom_minus / np.linalg.norm(nom_minus),
            "ILIS 0": nom_mid / np.linalg.norm(nom_mid),
        }

        rows = []
        for sector, sd in self.pit.sector_data.items():
            ilis = FiducialIlis(sd.ilis, pcr=False)

            # Detect which surfaces were augmented
            augmented_surfaces = set()
            if hasattr(sd, "_augmented_surfaces"):
                augmented_surfaces = sd._augmented_surfaces

            for coil in sd.delta:
                for feature in ["ILIS +1", "ILIS -1", "ILIS 0"]:
                    try:
                        plane = ilis.planes.loc[(coil, feature)]
                    except KeyError:
                        continue

                    n_meas = plane[["nx", "ny", "nz"]].values.astype(float)
                    n_meas = n_meas / np.linalg.norm(n_meas)
                    n_nom = nom_normals[feature]

                    # Angular deviation decomposed into roll and yaw
                    # Roll = rotation about x (radial): changes nz component
                    # Yaw = rotation about z (vertical): changes nx component
                    # relative to the nominal normal direction
                    dn = n_meas - n_nom

                    # For ILIS surfaces, nominal normal is ~(±sin10°, cos10°, 0)
                    # Roll (rot about x) manifests as change in nz
                    # Yaw (rot about z) manifests as change in nx (relative to ny)
                    roll_rad = np.arcsin(np.clip(dn[2], -1, 1))
                    yaw_rad = np.arcsin(np.clip(-dn[0] / n_nom[1], -1, 1))
                    tilt_rad = np.arccos(np.clip(np.dot(n_meas, n_nom), -1, 1))

                    is_augmented = (coil, feature) in augmented_surfaces

                    rows.append(
                        {
                            "coil": coil,
                            "sector": sector,
                            "feature": feature,
                            "roll_urad": roll_rad * 1e6,
                            "yaw_urad": yaw_rad * 1e6,
                            "tilt_urad": tilt_rad * 1e6,
                            "augmented": is_augmented,
                        }
                    )

        return pandas.DataFrame(rows)

    @cached_property
    def gaps(self) -> pandas.DataFrame:
        """Inter and intra-sector gap measurements."""
        return self.pit.gaps

    @cached_property
    def gap_targets(self) -> dict[str, float]:
        """Target gap values."""
        return {
            "within_sector": self.pit.within_sector_target,
            "between_sector": self.pit.between_sector_target,
            "limit": self.pit.gap_limit,
        }

    @cached_property
    def gap_deviations(self) -> pandas.DataFrame:
        """Gap deviations from target values.

        Returns DataFrame with gap measurements and deviation from target.
        """
        gaps = self.gaps.copy()
        if gaps.empty:
            return gaps

        targets = []
        for _, row in gaps.iterrows():
            if row.gap_type == "intra-sector":
                targets.append(self.pit.within_sector_target)
            else:
                targets.append(self.pit.between_sector_target)

        gaps["target"] = targets
        gaps["deviation"] = gaps["gap_mean"] - gaps["target"]
        gaps["label"] = [
            "%d-%d" % (int(r.coil_first), int(r.coil_second))
            for _, r in gaps.iterrows()
        ]
        return gaps

    def summary(self) -> str:
        """Return formatted text summary of all installed sectors."""
        lines = []
        lines.append("=" * 72)
        lines.append("Sector Overview - Installed TFC Alignment Summary")
        lines.append("=" * 72)

        # Sector phases
        lines.append("\nSector Phases:")
        for sector in sorted(self.pit.sector_data):
            phase = self.pit._resolved_phases.get(sector, "?")
            coils = list(self.pit.sector_data[sector].delta.keys())
            lines.append("  S%d: %s (coils %s)" % (sector, phase, coils))

        # Position parameters table
        lines.append("\nCoil Position Parameters (mm):")
        lines.append("-" * 72)
        pos = self.coil_positions
        if not pos.empty:
            fmt_cols = [
                "coil",
                "sector",
                "radial",
                "tangential",
                "vertical",
                "roll_length",
                "yaw_length",
                "pitch_length",
            ]
            lines.append(
                pos[fmt_cols].to_string(
                    index=False,
                    float_format="%.3f",
                )
            )

        # Position statistics
        lines.append("\nPosition Statistics:")
        lines.append("-" * 72)
        try:
            lines.append(self.pit.position_summary().to_string(index=False))
        except (ValueError, KeyError):
            lines.append("  (insufficient data)")

        # ILIS tilts
        lines.append("\nILIS Plane Tilts (midplane only, urad):")
        lines.append("-" * 72)
        tilts = self.ilis_tilts
        mid_tilts = tilts[tilts.feature == "ILIS 0"]
        if not mid_tilts.empty:
            fmt = mid_tilts[
                ["coil", "sector", "roll_urad", "yaw_urad", "tilt_urad", "augmented"]
            ]
            lines.append(fmt.to_string(index=False, float_format="%.1f"))

        # Gap measurements
        lines.append("\nGap Measurements (mm):")
        lines.append("-" * 72)
        gd = self.gap_deviations
        if not gd.empty:
            lines.append(
                "  %-8s  %-14s  %7s  %7s  %7s  %7s"
                % ("gap", "type", "mean", "std", "target", "dev")
            )
            for _, row in gd.iterrows():
                lines.append(
                    "  %-8s  %-14s  %7.3f  %7.3f  %7.3f  %+7.3f"
                    % (
                        row.label,
                        row.gap_type,
                        row.gap_mean,
                        row.gap_std,
                        row.target,
                        row.deviation,
                    )
                )

            # Cumulative gap assessment
            cum_gap = gd.gap_mean.sum()
            n_gaps = len(gd)
            projected_cum = cum_gap * 18 / n_gaps if n_gaps > 0 else 0
            lines.append(
                "\n  Measured cumulative gap (%d of 18): %.2f mm" % (n_gaps, cum_gap)
            )
            lines.append(
                "  Projected 18-gap cumulative: %.2f mm (target: 33 mm, limit: 36 mm)"
                % projected_cum
            )

        lines.append("\n" + "=" * 72)
        return "\n".join(lines)

    def print_summary(self):
        """Print formatted summary."""
        print(self.summary())


if __name__ == "__main__":
    overview = SectorOverview(
        sectors={5: [16, 5], 6: [12, 13], 7: [8, 9], 8: [4, 11]},
        phase="latest",
        pcr=True,
        augment=True,
    )
    overview.print_summary()
