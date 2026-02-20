"""Collate tilt and deviation data for installed TFC sectors.

Provides a consolidated view of coil positions, ILIS plane tilts, and
inter/intra-sector gaps across all installed sectors. Designed to be
updated incrementally as new sector data becomes available.
"""

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import pandas

from scipy.spatial.transform import Rotation

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
    private : bool
        Use private (in-work) data files
    """

    sectors: dict[int, list[int]] = field(
        default_factory=lambda: {5: [16, 5], 6: [12, 13], 7: [8, 9], 8: [4, 11]}
    )
    phase: str = "latest"
    pcr: bool = True
    private: bool = False

    def __post_init__(self):
        """Build overview from pit data."""
        self.pit = FiducialPit(
            sectors=self.sectors,
            phase=self.phase,
            pcr=self.pcr,
            private=self.private,
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
    def sector_transforms(self) -> pandas.DataFrame:
        """Rigid body transforms from target to measurement phase per sector.

        Decomposes each sector's corrected transform into CCL Kabsch
        and ILIS correction components, reported as Euler angles
        (roll, pitch, yaw) and translations. The corrected transform
        represents the installation error for each sector.
        """
        rows = []
        for sector, sd in self.pit.sector_data.items():
            transform = getattr(sd, "_transform", None)
            if transform is None:
                continue
            phase = self.pit._resolved_phases.get(sector, self.phase)
            ref_phase = getattr(sd, "_reference_phase", "In-pit target")
            coils = list(sd.delta.keys())
            for label, R_key, t_key in [
                ("Kabsch", "R_ccl", "t_ccl"),
                ("ILIS correction", "delta_R", "delta_t"),
                ("Corrected", "R_aug", "t_aug"),
            ]:
                euler = Rotation.from_matrix(transform[R_key]).as_euler("XYZ") * 1e6
                t = transform[t_key]
                rows.append(
                    {
                        "sector": sector,
                        "coils": str(coils),
                        "reference": ref_phase,
                        "phase": phase,
                        "component": label,
                        "roll_urad": euler[0],
                        "pitch_urad": euler[1],
                        "yaw_urad": euler[2],
                        "tx_mm": t[0],
                        "ty_mm": t[1],
                        "tz_mm": t[2],
                    }
                )
        return pandas.DataFrame(rows)

    @cached_property
    def gap_contributions(self) -> pandas.DataFrame:
        """Decompose gap deviations into tangential (mean gap) and roll (wedge).

        Mean gap deviation is driven by tangential displacements from
        datum.  The wedge (gap variation with height) is driven by coil
        roll.  Positive wedge means the gap is wider at the top.
        """
        gd = self.gap_deviations
        pos = self.coil_positions
        if gd.empty or pos.empty:
            return pandas.DataFrame()

        from nova.assembly.fiducialdata import FiducialData

        nom = FiducialData.fiducials()
        z_span = abs(nom.loc["B", "z"] - nom.loc["A", "z"])

        rows = []
        for _, gap in gd.iterrows():
            c1, c2 = int(gap.coil_first), int(gap.coil_second)

            p1 = pos[pos.coil == c1]
            p2 = pos[pos.coil == c2]
            if p1.empty or p2.empty:
                continue
            p1 = p1.iloc[0]
            p2 = p2.iloc[0]

            # Raw tangential displacements from datum
            tan_first = float(p1.tangential)
            tan_second = float(p2.tangential)

            # Linear fit of gap vs z to extract tilt
            z_vals, g_vals = [], []
            for z_key, g_key in [
                ("z_bottom", "gap_bottom"),
                ("z_middle", "gap_middle"),
                ("z_top", "gap_top"),
            ]:
                z = gap.get(z_key, np.nan)
                g = gap.get(g_key, np.nan)
                if not np.isnan(g) and not np.isnan(z):
                    z_vals.append(z)
                    g_vals.append(g)

            if len(z_vals) >= 2:
                z_arr = np.array(z_vals)
                g_arr = np.array(g_vals)
                slope, _ = np.polyfit(z_arr, g_arr, 1)
                z_meas = z_arr[-1] - z_arr[0]
                tilt = slope * z_meas
            else:
                tilt = np.nan
                z_meas = 0

            # Raw roll displacements scaled to measurement span
            scale = z_meas / z_span if z_span > 0 else 0
            roll_first = float(p1.roll_length) * scale
            roll_second = float(p2.roll_length) * scale

            rows.append(
                {
                    "pair": gap.label,
                    "gap_type": gap.gap_type,
                    "coil_first": c1,
                    "coil_second": c2,
                    "gap_mean": gap.gap_mean,
                    "target": gap.target,
                    "mean_gap": gap.gap_mean - gap.target,
                    "tan_first": tan_first,
                    "tan_second": tan_second,
                    "wedge": tilt,
                    "roll_first": roll_first,
                    "roll_second": roll_second,
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

        # Sector installation transforms
        transforms = self.sector_transforms
        if not transforms.empty:
            lines.append("\nSector Installation Transforms:")
            lines.append(
                "  Corrected = Kabsch + ILIS correction (installation error per sector)"
            )
            lines.append("-" * 72)
            for sector in sorted(transforms.sector.unique()):
                st = transforms[transforms.sector == sector]
                ref = st.iloc[0].reference
                phase = st.iloc[0].phase
                coils = st.iloc[0].coils
                lines.append("  S%d: %s -> %s (coils %s)" % (sector, ref, phase, coils))
                hdr = "    %-17s  %8s  %8s  %8s  %8s  %8s  %8s" % (
                    "component",
                    "roll",
                    "pitch",
                    "yaw",
                    "tx",
                    "ty",
                    "tz",
                )
                units = "    %-17s  %8s  %8s  %8s  %8s  %8s  %8s" % (
                    "",
                    "(urad)",
                    "(urad)",
                    "(urad)",
                    "(mm)",
                    "(mm)",
                    "(mm)",
                )
                lines.append(hdr)
                lines.append(units)
                lines.append("    " + "-" * len(hdr.strip()))
                for _, row in st.iterrows():
                    lines.append(
                        "    %-17s  %+8.1f  %+8.1f  %+8.1f  %+8.3f  %+8.3f  %+8.3f"
                        % (
                            row.component,
                            row.roll_urad,
                            row.pitch_urad,
                            row.yaw_urad,
                            row.tx_mm,
                            row.ty_mm,
                            row.tz_mm,
                        )
                    )
                lines.append("")

        # Gap analysis: mean deviation (tangential) and tilt (roll)
        contrib = self.gap_contributions
        if not contrib.empty:
            lines.append("\nGap Analysis (mm, inner edge):")
            lines.append("-" * 72)
            hdr = "  %-5s  %-14s  %6s  %6s  %8s  %+7s" % (
                "pair",
                "type",
                "mean",
                "target",
                "mean gap",
                "wedge",
            )
            lines.append(hdr)
            lines.append("  " + "-" * len(hdr.strip()))
            for _, row in contrib.iterrows():
                c1, c2 = int(row.coil_first), int(row.coil_second)
                wedge_str = (
                    "%+7.3f" % row.wedge if not np.isnan(row.wedge) else "    n/a"
                )
                lines.append(
                    "  %-5s  %-14s  %6.3f  %6.3f  %+8.3f  %s"
                    % (
                        row.pair,
                        row.gap_type,
                        row.gap_mean,
                        row.target,
                        row.mean_gap,
                        wedge_str,
                    )
                )
                # Attribution: tangential -> dev, roll -> tilt
                lines.append(
                    "  %5s  %14s  %14s  %+8.3f  %+7.3f"
                    % ("", "", "coil %d" % c1, row.tan_first, row.roll_first)
                )
                lines.append(
                    "  %5s  %14s  %14s  %+8.3f  %+7.3f"
                    % ("", "", "coil %d" % c2, row.tan_second, row.roll_second)
                )

            # Cumulative gap assessment
            cum_gap = contrib.gap_mean.sum()
            n_gaps = len(contrib)
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
    )
    overview.print_summary()
