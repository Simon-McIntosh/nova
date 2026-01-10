"""Integrate fiducial measurements from pit-installed sectors.

Combines measurements from installed sectors to calculate inter-sector and
intra-sector gaps using ILIS planes and CCL points.
"""

from dataclasses import dataclass, field
from functools import cached_property
from typing import ClassVar

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas
import scipy.spatial.transform

from nova.assembly.fiducialdata import FiducialData
from nova.assembly.fiducialilis import FiducialIlis
from nova.assembly.fiducialsector import FiducialSector
from nova.assembly.ilisnominal import NominalIlis
from nova.assembly.transform import Rotate
from nova.graphics.plot import Plot1D

alt.renderers.enable("html")


def rotate_to_angle(vector: np.ndarray, angle: float) -> np.ndarray:
    """Rotate vector about z-axis by angle (radians)."""
    rotation = scipy.spatial.transform.Rotation.from_euler("z", angle)
    return rotation.apply(vector)


@dataclass
class FiducialPit(Plot1D):
    """Integrate multi-sector pit fiducial data for gap analysis.

    Loads fiducial measurements from installed sectors, transforms coils
    to pit coordinates, and calculates inter/intra-sector gaps.

    Parameters
    ----------
    sectors : dict[int, list[int]]
        Sector to coil mapping, e.g. {5: [16, 5], 6: [12, 13], ...}
    phase : str
        Measurement phase to load (e.g. "In-pit target")
    pcr : bool
        Apply PCR deviation corrections to ILIS planes
    private : bool
        Use private (in-work) data files

    Attributes
    ----------
    location : list[int]
        Toroidal position of each coil index around the machine (0-17)
    gaps : pandas.DataFrame
        Calculated gap data with inter/intra-sector classification
    statistics : pandas.DataFrame
        Summary statistics for gap components
    """

    sectors: dict[int, list[int]] = field(
        default_factory=lambda: {5: [16, 5], 6: [12, 13], 7: [8, 9], 8: [4, 11]}
    )
    phase: str = "In-pit target"
    pcr: bool = True
    private: bool = False

    # Target gap specifications (mm)
    intra_sector_target: float = 2.0  # Target gap within a sector
    inter_sector_target: float = 1.5  # Target gap between sectors (smaller)

    # Toroidal position of each coil index - imported from FiducialData
    # to maintain single source of truth
    location: ClassVar[list[int]] = FiducialData.location

    # Installed sector positions: Sectors 5-6-7-8 form a contiguous cluster
    # at positions 8-15 (160°-300°)

    def __post_init__(self):
        """Load sector data and build gap analysis."""
        self.rotate = Rotate()
        self.nominal = NominalIlis()
        self._load_sectors()
        self._build_coil_positions()
        self._build_gap_pairs()
        self._calculate_nominal_gaps()

    def _load_sectors(self):
        """Load fiducial data for each sector."""
        self.sector_data: dict[int, FiducialSector] = {}
        for sector, coils in self.sectors.items():
            try:
                self.sector_data[sector] = FiducialSector(
                    phase=self.phase,
                    sectors={sector: coils},
                    private=self.private,
                )
            except (FileNotFoundError, KeyError) as e:
                print(f"Warning: Could not load sector {sector}: {e}")

    @cached_property
    def coils(self) -> list[int]:
        """Return list of all coils across loaded sectors."""
        return [
            coil
            for sector_data in self.sector_data.values()
            for coil in sector_data.delta.keys()
        ]

    @cached_property
    def coil_sectors(self) -> dict[int, int]:
        """Return mapping from coil to sector number."""
        return {
            coil: sector
            for sector, sector_data in self.sector_data.items()
            for coil in sector_data.delta.keys()
        }

    def _build_coil_positions(self):
        """Build position index for each coil."""
        self.coil_position: dict[int, int] = {}
        for coil in self.coils:
            if coil <= len(self.location):
                self.coil_position[coil] = self.location.index(coil)

    def _build_gap_pairs(self):
        """Build list of coil pairs that form gaps.

        Identifies adjacent coils and classifies as inter/intra-sector gaps.
        """
        # Sort coils by their pit position
        sorted_coils = sorted(
            [(coil, pos) for coil, pos in self.coil_position.items()],
            key=lambda x: x[1],
        )

        self.gap_pairs: list[dict] = []
        for i in range(len(sorted_coils) - 1):
            coil_a, pos_a = sorted_coils[i]
            coil_b, pos_b = sorted_coils[i + 1]

            # Check if coils are adjacent (consecutive positions)
            if pos_b - pos_a == 1:
                sector_a = self.coil_sectors.get(coil_a)
                sector_b = self.coil_sectors.get(coil_b)

                gap_type = "intra-sector" if sector_a == sector_b else "inter-sector"

                self.gap_pairs.append(
                    {
                        "coil_plus": coil_a,  # ILIS +1 side
                        "coil_minus": coil_b,  # ILIS -1 side
                        "position": pos_a,
                        "gap_type": gap_type,
                        "sector_a": sector_a,
                        "sector_b": sector_b,
                    }
                )

    def _calculate_nominal_gaps(self):
        """Calculate nominal gap values for comparison with measured gaps.

        The nominal gap is calculated by rotating the nominal ILIS planes
        to the gap midplane (same transformation as measured planes).
        """
        # Get nominal planes for ILIS +1 and ILIS -1
        nom_plus = self.nominal.planes.loc[(0, "ILIS +1"), ["x", "y", "z"]].values
        nom_minus = self.nominal.planes.loc[(0, "ILIS -1"), ["x", "y", "z"]].values

        # Rotate both to sector midplane (same as intra-sector gap transform)
        half_angle = np.pi / 18  # 10°

        # ILIS -1 is on the "first coil" side (needs +10° rotation)
        nom_minus_rot = rotate_to_angle(nom_minus, half_angle)
        # ILIS +1 is on the "second coil" side (needs -10° rotation)
        nom_plus_rot = rotate_to_angle(nom_plus, -half_angle)

        # Nominal gap components (plus - minus, same order as measured)
        gap_vector = nom_plus_rot - nom_minus_rot

        # Get midpoint for unit vector calculation
        midpoint = (nom_plus_rot + nom_minus_rot) / 2
        phi_mid = np.arctan2(midpoint[1], midpoint[0])

        radial_unit = np.array([np.cos(phi_mid), np.sin(phi_mid), 0])
        tangential_unit = np.array([-np.sin(phi_mid), np.cos(phi_mid), 0])
        vertical_unit = np.array([0, 0, 1])

        self.nominal_gap = {
            "radial": float(np.dot(gap_vector, radial_unit)),
            "tangential": float(np.dot(gap_vector, tangential_unit)),
            "vertical": float(np.dot(gap_vector, vertical_unit)),
        }

        # Also calculate gap_normal using nominal normals
        nom_n_plus = self.nominal.planes.loc[(0, "ILIS +1"), ["nx", "ny", "nz"]].values
        nom_n_minus = self.nominal.planes.loc[(0, "ILIS -1"), ["nx", "ny", "nz"]].values
        nom_n_plus_rot = rotate_to_angle(nom_n_plus, -half_angle)
        nom_n_minus_rot = rotate_to_angle(nom_n_minus, half_angle)
        avg_normal = (nom_n_plus_rot - nom_n_minus_rot) / 2
        avg_normal = avg_normal / np.linalg.norm(avg_normal)
        self.nominal_gap["gap_normal"] = float(np.dot(gap_vector, avg_normal))

    @cached_property
    def ilis_data(self) -> pandas.DataFrame:
        """Return combined ILIS data from all sectors."""
        frames = []
        for sector, sector_data in self.sector_data.items():
            if hasattr(sector_data, "ilis") and not sector_data.ilis.empty:
                df = sector_data.ilis.copy()
                df["sector"] = sector
                frames.append(df)
        if not frames:
            return pandas.DataFrame()
        return pandas.concat(frames, ignore_index=True)

    @cached_property
    def ilis(self) -> FiducialIlis:
        """Return FiducialIlis instance for all pit data."""
        return FiducialIlis(self.ilis_data, pcr=self.pcr)

    def _get_coil_transform(self, coil: int, position: int):
        """Return transform function to unclock coil to gap position.

        For gap calculation, we need to rotate coils to align with
        the gap midplane between two adjacent coils.
        """
        # The first coil in each sector pair is clocked, second is anticlocked
        # to bring them to the sector midplane
        sorted_coils = sorted(self.coil_position.items(), key=lambda x: x[1])
        coil_list = [c for c, _ in sorted_coils]
        if coil not in coil_list:
            return lambda x: x

        idx = coil_list.index(coil)
        # Determine if this coil is the "plus" or "minus" side
        # Plus side needs anticlocking, minus side needs clocking
        # to meet at the midplane
        return self.rotate.anticlock if idx % 2 == 0 else self.rotate.clock

    def _rotate_plane_to_position(
        self,
        plane: pandas.Series,
        coil: int,
        target_position: float,
    ) -> pandas.Series:
        """Rotate plane coordinates to align with a target toroidal position.

        Parameters
        ----------
        plane : pandas.Series
            ILIS plane with x, y, z, nx, ny, nz
        coil : int
            Coil number
        target_position : float
            Target position index (can be fractional for gap midplanes)

        Returns
        -------
        pandas.Series
            Rotated plane
        """
        plane = plane.copy()
        position = self.coil_position.get(coil)
        if position is None:
            return plane

        # Determine current coil angle and target angle
        # Each position step is 20° (360° / 18 coils)
        sector = self.coil_sectors.get(coil)
        sector_coils = list(self.sector_data[sector].delta.keys())
        is_first = sector_coils.index(coil) == 0

        # Within a sector, first coil needs clocking, second needs anticlocking
        # to meet at the sector midplane (which is at position + 0.5)
        half_angle = np.pi / 18  # 10°

        if is_first:
            # First coil in sector: rotate by -half_angle (anticlock)
            rotation_angle = -half_angle
        else:
            # Second coil in sector: rotate by +half_angle (clock)
            rotation_angle = half_angle

        # Transform point
        point = np.array([plane["x"], plane["y"], plane["z"]])
        plane[["x", "y", "z"]] = rotate_to_angle(point, rotation_angle)

        # Transform normal
        normal = np.array([plane["nx"], plane["ny"], plane["nz"]])
        plane[["nx", "ny", "nz"]] = rotate_to_angle(normal, rotation_angle)

        return plane

    def _unclock_plane(self, plane: pandas.Series, coil: int) -> pandas.Series:
        """Transform plane coordinates to pit reference frame."""
        plane = plane.copy()
        position = self.coil_position.get(coil)
        if position is None:
            return plane

        # Determine clocking based on position in sector
        sector = self.coil_sectors.get(coil)
        sector_coils = list(self.sector_data[sector].delta.keys())
        is_first = sector_coils.index(coil) == 0

        transform = self.rotate.anticlock if is_first else self.rotate.clock

        # Transform point
        point = np.array([plane["x"], plane["y"], plane["z"]])
        plane[["x", "y", "z"]] = transform(point)

        # Transform normal
        normal = np.array([plane["nx"], plane["ny"], plane["nz"]])
        plane[["nx", "ny", "nz"]] = transform(normal)

        return plane

    def _get_machine_angle(self, coil: int) -> float:
        """Return the machine angle (radians) for a coil.

        Position 0 is at phi = 0, and each position is 20° apart.
        """
        position = self.coil_position.get(coil, 0)
        return position * 2 * np.pi / 18  # 20° per position

    def _get_sector_offset_angle(self, coil: int) -> float:
        """Return the offset angle (radians) of coil from its sector midplane.

        First coil in sector is at -10°, second is at +10°.
        """
        sector = self.coil_sectors.get(coil)
        if sector is None:
            return 0
        sector_coils = list(self.sector_data[sector].delta.keys())
        is_first = sector_coils.index(coil) == 0
        half_angle = np.pi / 18  # 10°
        return -half_angle if is_first else half_angle

    def _rotate_to_machine_frame(
        self, plane: pandas.Series, coil: int
    ) -> pandas.Series:
        """Rotate plane from sector coordinates to machine coordinates.

        Sector coordinates have y-axis pointing radially outward at sector center.
        Machine coordinates have y-axis pointing radially outward at phi = 0.
        """
        plane = plane.copy()
        sector = self.coil_sectors.get(coil)
        if sector is None:
            return plane

        # Get the sector midplane angle in machine coordinates
        # This is the angle to rotate from sector coords to machine coords
        sector_coils = list(self.sector_data[sector].delta.keys())
        first_coil = sector_coils[0]
        first_coil_position = self.coil_position.get(first_coil, 0)
        # Sector midplane is 0.5 positions after the first coil
        sector_midplane_position = first_coil_position + 0.5
        sector_midplane_angle = sector_midplane_position * 2 * np.pi / 18

        # Rotate from sector frame to machine frame
        point = np.array([plane["x"], plane["y"], plane["z"]])
        plane[["x", "y", "z"]] = rotate_to_angle(point, sector_midplane_angle)

        normal = np.array([plane["nx"], plane["ny"], plane["nz"]])
        plane[["nx", "ny", "nz"]] = rotate_to_angle(normal, sector_midplane_angle)

        return plane

    def _rotate_gap_planes(
        self,
        plane_plus: pandas.Series,
        plane_minus: pandas.Series,
        coil_plus: int,
        coil_minus: int,
    ) -> tuple[pandas.Series, pandas.Series]:
        """Rotate both planes to meet at the gap midplane.

        ILIS data is in sector module coordinates where:
        - y-axis points radially outward at sector center
        - x-axis points tangentially (positive toward positive phi)
        - Coils are at ±10° from sector midplane

        For intra-sector gaps: both coils rotate to sector midplane
        For inter-sector gaps: both coils rotate to machine frame, then
            both rotate to the gap midplane location
        """
        sector_plus = self.coil_sectors.get(coil_plus)
        sector_minus = self.coil_sectors.get(coil_minus)

        plane_plus = plane_plus.copy()
        plane_minus = plane_minus.copy()

        half_angle = np.pi / 18  # 10° per half-sector

        if sector_plus == sector_minus:
            # Intra-sector gap: rotate both coils to sector midplane
            sector_coils = list(self.sector_data[sector_plus].delta.keys())

            # First coil in sector is at -10° (negative x in sector coords)
            # → needs +10° rotation (clock) to reach midplane
            # Second coil in sector is at +10° (positive x in sector coords)
            # → needs -10° rotation (anticlock) to reach midplane
            for coil, plane in [
                (coil_plus, plane_plus),
                (coil_minus, plane_minus),
            ]:
                is_first = sector_coils.index(coil) == 0
                angle = half_angle if is_first else -half_angle

                point = np.array([plane["x"], plane["y"], plane["z"]])
                plane[["x", "y", "z"]] = rotate_to_angle(point, angle)

                normal = np.array([plane["nx"], plane["ny"], plane["nz"]])
                plane[["nx", "ny", "nz"]] = rotate_to_angle(normal, angle)
        else:
            # Inter-sector gap: coils from different sectors
            # Transform both to gap frame (y-axis pointing outward at gap)

            # Get sector midplane angles in machine coords
            sector_plus_coils = list(self.sector_data[sector_plus].delta.keys())
            first_coil_plus = sector_plus_coils[0]
            pos_plus = self.coil_position.get(first_coil_plus, 0)
            sector_plus_angle = (pos_plus + 0.5) * 2 * np.pi / 18

            sector_minus_coils = list(self.sector_data[sector_minus].delta.keys())
            first_coil_minus = sector_minus_coils[0]
            pos_minus = self.coil_position.get(first_coil_minus, 0)
            sector_minus_angle = (pos_minus + 0.5) * 2 * np.pi / 18

            # The gap is at the boundary between sectors
            # Gap position = average of adjacent coil positions
            pos_coil_plus = self.coil_position.get(coil_plus, 0)
            pos_coil_minus = self.coil_position.get(coil_minus, 0)
            gap_position = (pos_coil_plus + pos_coil_minus) / 2
            gap_angle = gap_position * 2 * np.pi / 18

            # Coil offset from its sector midplane
            is_first_plus = sector_plus_coils.index(coil_plus) == 0
            offset_plus = -half_angle if is_first_plus else half_angle

            is_first_minus = sector_minus_coils.index(coil_minus) == 0
            offset_minus = -half_angle if is_first_minus else half_angle

            # Rotation from sector coords to gap coords:
            # gap_angle - sector_angle - coil_offset
            angle_plus = gap_angle - sector_plus_angle - offset_plus

            point = np.array([plane_plus["x"], plane_plus["y"], plane_plus["z"]])
            plane_plus[["x", "y", "z"]] = rotate_to_angle(point, angle_plus)
            normal = np.array([plane_plus["nx"], plane_plus["ny"], plane_plus["nz"]])
            plane_plus[["nx", "ny", "nz"]] = rotate_to_angle(normal, angle_plus)

            angle_minus = gap_angle - sector_minus_angle - offset_minus

            point = np.array([plane_minus["x"], plane_minus["y"], plane_minus["z"]])
            plane_minus[["x", "y", "z"]] = rotate_to_angle(point, angle_minus)
            normal = np.array([plane_minus["nx"], plane_minus["ny"], plane_minus["nz"]])
            plane_minus[["nx", "ny", "nz"]] = rotate_to_angle(normal, angle_minus)

        return plane_plus, plane_minus

    def _clock_transform(
        self, data: pandas.DataFrame, coil: int, is_first: bool
    ) -> pandas.DataFrame:
        """Apply clocking transform to rotate coil data to sector midplane.

        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame with x, y, z columns
        coil : int
            Coil number (for group identification)
        is_first : bool
            True if this is the first coil in the sector pair

        Returns
        -------
        pandas.DataFrame
            Transformed data with x, y, z rotated to midplane
        """
        data = data.copy()
        # First coil rotates anticlock (+10° in our convention), second clocks (-10°)
        transform = self.rotate.anticlock if is_first else self.rotate.clock
        data.loc[:, ["x", "y", "z"]] = transform(data.loc[:, ["x", "y", "z"]].values)
        return data

    def _clock_planes(
        self, planes: pandas.DataFrame, coil: int, is_first: bool
    ) -> pandas.DataFrame:
        """Apply clocking transform to planes (both position and normal).

        Parameters
        ----------
        planes : pandas.DataFrame
            Planes DataFrame with x, y, z, nx, ny, nz columns
        coil : int
            Coil number
        is_first : bool
            True if first coil in sector pair

        Returns
        -------
        pandas.DataFrame
            Transformed planes
        """
        planes = planes.copy()
        transform = self.rotate.anticlock if is_first else self.rotate.clock
        planes.loc[:, ["x", "y", "z"]] = transform(
            planes.loc[:, ["x", "y", "z"]].values
        )
        planes.loc[:, ["nx", "ny", "nz"]] = transform(
            planes.loc[:, ["nx", "ny", "nz"]].values
        )
        return planes

    def calculate_gaps(self) -> pandas.DataFrame:
        """Calculate gaps between adjacent coils using point cloud projection.

        Uses the FiducialSector method:
        1. Clock both ILIS planes to meet at the gap midplane
        2. Compute the intersection midplane
        3. Project point cloud data onto the midplane
        4. Gap = difference in offsets from midplane

        For intra-sector gaps:
        - First coil (lower position) contributes ILIS +1 (positive-phi side)
        - Second coil (higher position) contributes ILIS -1 (negative-phi side)
        - Both coils clock to the sector midplane

        For inter-sector gaps:
        - First coil contributes ILIS -1 (its negative side faces the gap)
        - Second coil contributes ILIS +1 (its positive side faces the gap)
        - Note: Inter-sector gaps not yet implemented

        Returns
        -------
        pandas.DataFrame
            Gap measurements with columns:
            - coil_first, coil_second: Adjacent coil pair
            - gap_type: 'inter-sector' or 'intra-sector'
            - gap_mean: Mean gap from point cloud offsets (mm)
            - gap_std: Standard deviation of gap (mm)
            - offset_first_mean/std: First plane offset statistics
            - offset_second_mean/std: Second plane offset statistics
        """
        gaps = []

        for pair in self.gap_pairs:
            coil_first = pair["coil_plus"]  # First coil (lower position)
            coil_second = pair["coil_minus"]  # Second coil (higher position)
            sector_a = pair["sector_a"]
            sector_b = pair["sector_b"]

            try:
                if sector_a == sector_b:
                    # Intra-sector gap
                    result = self._calculate_intra_sector_gap(
                        coil_first, coil_second, sector_a, pair
                    )
                else:
                    # Inter-sector gap
                    result = self._calculate_inter_sector_gap(
                        coil_first, coil_second, sector_a, sector_b, pair
                    )

                gaps.append(result)

            except (KeyError, IndexError) as e:
                print(
                    f"Warning: Could not calculate gap for "
                    f"{coil_first}-{coil_second}: {e}"
                )

        return pandas.DataFrame(gaps)

    def _calculate_intra_sector_gap(
        self, coil_first: int, coil_second: int, sector: int, pair: dict
    ) -> dict:
        """Calculate gap between two coils within the same sector.

        Parameters
        ----------
        coil_first : int
            First coil number (lower position)
        coil_second : int
            Second coil number (higher position)
        sector : int
            Sector number containing both coils
        pair : dict
            Gap pair metadata

        Returns
        -------
        dict
            Gap measurement results
        """
        # For intra-sector gaps:
        # - First coil provides ILIS +1 (positive-phi side faces the gap)
        # - Second coil provides ILIS -1 (negative-phi side faces the gap)
        feature_first = "ILIS +1"
        feature_second = "ILIS -1"

        sector_coils = list(self.sector_data[sector].delta.keys())

        # Create FiducialIlis for this sector
        sector_data = self.sector_data[sector]
        ilis = FiducialIlis(sector_data.ilis, pcr=self.pcr)

        # Clock planes to midplane
        def clock_group(planes, coil):
            is_first = sector_coils.index(coil) == 0
            return self._clock_planes(planes, coil, is_first)

        sector_planes = ilis.planes.groupby(["coil"], group_keys=False).apply(
            lambda x: clock_group(x, x.name)
        )

        # Clock point cloud data
        def clock_data_group(data, coil):
            is_first = sector_coils.index(coil) == 0
            return self._clock_transform(data, coil, is_first)

        sector_data_clocked = ilis.data.groupby(["coil"], group_keys=False).apply(
            lambda x: clock_data_group(x, x.name)
        )

        # Get the sector index (which planes form the gap)
        sector_index = [
            (coil_first, feature_first),
            (coil_second, feature_second),
        ]

        # Calculate midplane by intersecting the two clocked planes
        midplane = ilis.intersect(sector_planes.loc[sector_index])

        # Calculate offsets from midplane for each ILIS surface
        offsets = []
        for coil, feature in sector_index:
            plane_mask = (sector_data_clocked.coil == coil) & (
                sector_data_clocked.feature == feature
            )
            plane_offset = ilis.offset(
                sector_data_clocked.loc[plane_mask, ["x", "y", "z"]], midplane
            )
            offsets.append(
                {
                    "coil": coil,
                    "feature": feature,
                    "mean": plane_offset.mean(),
                    "std": plane_offset.std(),
                    "count": len(plane_offset),
                }
            )

        # Gap = second offset - first offset
        # Midplane is between them, so offsets have opposite signs
        gap_mean = offsets[1]["mean"] - offsets[0]["mean"]
        gap_std = np.sqrt(offsets[0]["std"] ** 2 + offsets[1]["std"] ** 2)

        # Get PCR deviation offsets if available
        plane_first = sector_planes.loc[(coil_first, feature_first)]
        plane_second = sector_planes.loc[(coil_second, feature_second)]
        pcr_first = plane_first.get("offset", 0)
        pcr_second = plane_second.get("offset", 0)

        return {
            "coil_first": coil_first,
            "coil_second": coil_second,
            "gap_type": pair["gap_type"],
            "sector_a": pair["sector_a"],
            "sector_b": pair["sector_b"],
            "position": pair["position"],
            "gap_mean": gap_mean,
            "gap_std": gap_std,
            "offset_first_mean": offsets[0]["mean"],
            "offset_first_std": offsets[0]["std"],
            "offset_second_mean": offsets[1]["mean"],
            "offset_second_std": offsets[1]["std"],
            "points_first": offsets[0]["count"],
            "points_second": offsets[1]["count"],
            "pcr_first": pcr_first,
            "pcr_second": pcr_second,
        }

    def _calculate_inter_sector_gap(
        self,
        coil_first: int,
        coil_second: int,
        sector_a: int,
        sector_b: int,
        pair: dict,
    ) -> dict:
        """Calculate gap between two coils from adjacent sectors.

        For inter-sector gaps, the gap is formed by:
        - First sector's second coil ILIS +1 (faces toward next sector)
        - Next sector's first coil ILIS -1 (faces toward previous sector)

        The key difference from intra-sector is the rotation direction:
        - For intra-sector: first coil anticlocks (+10°), second clocks (-10°)
          to meet at sector midplane (y≈0 in sector coords)
        - For inter-sector: first coil clocks (-10°), second anticlocks (+10°)
          to meet at the inter-sector boundary

        Sector coordinate system has y-axis pointing radially outward at sector
        midplane, so both cases result in the gap-forming ILIS surfaces meeting
        at y≈0 after transformation.

        Parameters
        ----------
        coil_first : int
            First coil number (second coil of sector_a, lower position)
        coil_second : int
            Second coil number (first coil of sector_b, higher position)
        sector_a : int
            First sector number
        sector_b : int
            Second sector number
        pair : dict
            Gap pair metadata

        Returns
        -------
        dict
            Gap measurement results
        """
        # For inter-sector gaps:
        # - coil_first is the second coil of sector_a, contributes ILIS +1
        # - coil_second is the first coil of sector_b, contributes ILIS -1
        feature_first = "ILIS +1"
        feature_second = "ILIS -1"

        # Process each sector's data separately with FiducialIlis
        ilis_a = FiducialIlis(self.sector_data[sector_a].ilis, pcr=self.pcr)
        ilis_b = FiducialIlis(self.sector_data[sector_b].ilis, pcr=self.pcr)

        # For inter-sector gaps, the clocking is OPPOSITE to intra-sector:
        # - coil_first (second in sector_a, at +10° from midplane) needs to reach
        #   gap at midplane + 20°, so rotates by +10° more = anticlock
        # - coil_second (first in sector_b, at -10° from midplane) needs to reach
        #   gap at midplane - 20°, so rotates by -10° more = clock

        def clock_planes(planes):
            """Apply clock transform (-10°) to planes."""
            planes = planes.copy()
            planes.loc[:, ["x", "y", "z"]] = self.rotate.clock(
                planes.loc[:, ["x", "y", "z"]].values
            )
            planes.loc[:, ["nx", "ny", "nz"]] = self.rotate.clock(
                planes.loc[:, ["nx", "ny", "nz"]].values
            )
            return planes

        def anticlock_planes(planes):
            """Apply anticlock transform (+10°) to planes."""
            planes = planes.copy()
            planes.loc[:, ["x", "y", "z"]] = self.rotate.anticlock(
                planes.loc[:, ["x", "y", "z"]].values
            )
            planes.loc[:, ["nx", "ny", "nz"]] = self.rotate.anticlock(
                planes.loc[:, ["nx", "ny", "nz"]].values
            )
            return planes

        def clock_data(data):
            """Apply clock transform (-10°) to point cloud data."""
            data = data.copy()
            data.loc[:, ["x", "y", "z"]] = self.rotate.clock(
                data.loc[:, ["x", "y", "z"]].values
            )
            return data

        def anticlock_data(data):
            """Apply anticlock transform (+10°) to point cloud data."""
            data = data.copy()
            data.loc[:, ["x", "y", "z"]] = self.rotate.anticlock(
                data.loc[:, ["x", "y", "z"]].values
            )
            return data

        # Apply transforms: coil_first anticlocks (+10°), coil_second clocks (-10°)
        planes_first = anticlock_planes(ilis_a.planes.loc[[coil_first]])
        data_first = anticlock_data(ilis_a.data[ilis_a.data.coil == coil_first].copy())

        planes_second = clock_planes(ilis_b.planes.loc[[coil_second]])
        data_second = clock_data(ilis_b.data[ilis_b.data.coil == coil_second].copy())

        # Extract the specific ILIS planes for the gap
        plane_first = planes_first.loc[(coil_first, feature_first)]
        plane_second = planes_second.loc[(coil_second, feature_second)]

        # Create combined planes DataFrame for intersection
        sector_index = [
            (coil_first, feature_first),
            (coil_second, feature_second),
        ]
        rotated_planes = pandas.DataFrame(
            [plane_first, plane_second],
            index=pandas.MultiIndex.from_tuples(
                sector_index, names=["coil", "feature"]
            ),
        )

        # Calculate midplane
        midplane = FiducialIlis.intersect(rotated_planes)

        # Filter point cloud data for the specific ILIS features
        data_first_filtered = data_first[data_first.feature == feature_first]
        data_second_filtered = data_second[data_second.feature == feature_second]

        # Calculate offsets from midplane
        offsets = []
        for data_rotated, coil, feature in [
            (data_first_filtered, coil_first, feature_first),
            (data_second_filtered, coil_second, feature_second),
        ]:
            if data_rotated.empty:
                offsets.append(
                    {"coil": coil, "feature": feature, "mean": 0, "std": 0, "count": 0}
                )
                continue

            # Calculate offset using midplane
            normal = midplane.loc[["nx", "ny", "nz"]].values
            normal = normal / np.linalg.norm(normal)
            point = midplane.loc[["x", "y", "z"]].values
            v = data_rotated[["x", "y", "z"]].values - point
            plane_offset = np.dot(v, normal)

            offsets.append(
                {
                    "coil": coil,
                    "feature": feature,
                    "mean": plane_offset.mean(),
                    "std": plane_offset.std(),
                    "count": len(plane_offset),
                }
            )

        # Gap = second offset - first offset
        gap_mean = offsets[1]["mean"] - offsets[0]["mean"]
        gap_std = np.sqrt(offsets[0]["std"] ** 2 + offsets[1]["std"] ** 2)

        # Get PCR deviation offsets from original (unclocked) planes
        pcr_first = ilis_a.planes.loc[(coil_first, feature_first)].get("offset", 0)
        pcr_second = ilis_b.planes.loc[(coil_second, feature_second)].get("offset", 0)

        return {
            "coil_first": coil_first,
            "coil_second": coil_second,
            "gap_type": pair["gap_type"],
            "sector_a": pair["sector_a"],
            "sector_b": pair["sector_b"],
            "position": pair["position"],
            "gap_mean": gap_mean,
            "gap_std": gap_std,
            "offset_first_mean": offsets[0]["mean"],
            "offset_first_std": offsets[0]["std"],
            "offset_second_mean": offsets[1]["mean"],
            "offset_second_std": offsets[1]["std"],
            "points_first": offsets[0]["count"],
            "points_second": offsets[1]["count"],
            "pcr_first": pcr_first,
            "pcr_second": pcr_second,
        }

    @cached_property
    def gaps(self) -> pandas.DataFrame:
        """Return cached gap calculations."""
        return self.calculate_gaps()

    def statistics(self, gap_type: str | None = None) -> pandas.DataFrame:
        """Calculate summary statistics for gap measurements.

        Parameters
        ----------
        gap_type : str, optional
            Filter by gap type: 'inter-sector', 'intra-sector', or None for all

        Returns
        -------
        pandas.DataFrame
            Statistics with mean, std, min, max for gap_mean and gap_std
        """
        gaps = self.gaps
        if gap_type is not None:
            gaps = gaps[gaps["gap_type"] == gap_type]

        if gaps.empty:
            return pandas.DataFrame()

        components = ["gap_mean", "gap_std"]

        stats = gaps[components].agg(["mean", "std", "min", "max"]).T
        stats["count"] = len(gaps)
        return stats

    def summary(self) -> pandas.DataFrame:
        """Return combined statistics for inter, intra, and all gaps."""
        results = []

        for gap_type in ["intra-sector", "inter-sector", None]:
            stats = self.statistics(gap_type)
            if not stats.empty:
                label = gap_type if gap_type else "combined"
                stats["type"] = label
                results.append(stats)

        if not results:
            return pandas.DataFrame()

        return pandas.concat(results)

    def plot_gaps(self, figsize=(10, 6)):
        """Plot gap measurements as bar chart with error bars.

        Shows gap_mean ± gap_std for each gap pair,
        colored by gap type (intra/inter-sector).
        """
        if self.gaps.empty:
            print("No gaps to plot")
            return

        fig, ax = plt.subplots(figsize=figsize)
        colors = {"intra-sector": "C0", "inter-sector": "C1"}

        labels = [
            f"{row.coil_first}-{row.coil_second}" for _, row in self.gaps.iterrows()
        ]
        x = np.arange(len(labels))
        bar_colors = [colors[gt] for gt in self.gaps["gap_type"]]

        ax.bar(
            x,
            self.gaps["gap_mean"].values,
            yerr=self.gaps["gap_std"].values,
            color=bar_colors,
            alpha=0.8,
            edgecolor="k",
            capsize=3,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("Gap (coil-coil)")
        ax.set_ylabel("Gap (mm)")
        ax.set_title(f"Pit Gap Analysis - {self.phase}")

        ax.axhline(
            self.intra_sector_target,
            color="C0",
            linestyle="--",
            alpha=0.5,
            label=f"intra target: {self.intra_sector_target}",
        )
        ax.axhline(
            self.inter_sector_target,
            color="C1",
            linestyle="--",
            alpha=0.5,
            label=f"inter target: {self.inter_sector_target}",
        )
        ax.legend(fontsize="small")
        ax.axhline(0, color="gray", linewidth=0.5)

        # Add value labels
        for i, (_, row) in enumerate(self.gaps.iterrows()):
            ax.annotate(
                f"{row.gap_mean:.2f}",
                xy=(i, row.gap_mean),
                xytext=(0, 3 if row.gap_mean >= 0 else -12),
                textcoords="offset points",
                ha="center",
                va="bottom" if row.gap_mean >= 0 else "top",
                fontsize=8,
            )

        fig.tight_layout()
        return fig, ax

    def plot_statistics(self, figsize=(8, 6)):
        """Plot summary statistics as bar chart with error bars.

        Shows mean gap ± between-gap std for each gap type.
        """
        if self.gaps.empty:
            print("No statistics to plot")
            return

        fig, ax = plt.subplots(figsize=figsize)

        gap_types = ["intra-sector", "inter-sector", "combined"]
        x = np.arange(len(gap_types))

        means = []
        stds = []
        for gap_type in gap_types:
            if gap_type == "combined":
                subset = self.gaps
            else:
                subset = self.gaps[self.gaps["gap_type"] == gap_type]
            if not subset.empty:
                means.append(subset["gap_mean"].mean())
                stds.append(subset["gap_mean"].std())
            else:
                means.append(0)
                stds.append(0)

        ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8, edgecolor="k")
        ax.set_xticks(x)
        ax.set_xticklabels(gap_types)
        ax.set_ylabel("Gap (mm)")
        ax.set_title(f"Gap Statistics Summary - {self.phase}")
        ax.axhline(0, color="gray", linewidth=0.5)

        # Add target lines
        ax.axhline(
            self.intra_sector_target,
            color="C0",
            linestyle="--",
            alpha=0.5,
            label=f"intra target: {self.intra_sector_target}",
        )
        ax.axhline(
            self.inter_sector_target,
            color="C1",
            linestyle="--",
            alpha=0.5,
            label=f"inter target: {self.inter_sector_target}",
        )
        ax.legend()

        fig.tight_layout()
        return fig, ax

    def print_summary(self):
        """Print formatted summary of gap statistics."""
        print(f"\n{'=' * 60}")
        print(f"Pit Gap Analysis Summary - {self.phase}")
        print(f"{'=' * 60}")

        print(f"\nLoaded sectors: {list(self.sector_data.keys())}")
        print(f"Coils: {self.coils}")
        print(f"Number of gaps: {len(self.gaps)}")

        print(f"\n{'Sector Layout in Pit':^60}")
        print("-" * 60)
        for sector, coils in self.sectors.items():
            positions = [self.coil_position.get(c, "?") for c in coils]
            print(f"  Sector {sector}: coils {coils} at positions {positions}")

        print(f"\n{'Gap Pairs':^60}")
        print("-" * 60)
        for pair in self.gap_pairs:
            print(
                f"  {pair['coil_plus']:2d} ↔ {pair['coil_minus']:2d}  "
                f"({pair['gap_type']:12s}) "
                f"sectors {pair['sector_a']}-{pair['sector_b']}"
            )

        print(f"\n{'Gap Measurements':^60}")
        print("-" * 60)
        if not self.gaps.empty:
            for _, row in self.gaps.iterrows():
                print(
                    f"  {row.coil_first:2.0f} ↔ {row.coil_second:2.0f}: "
                    f"{row.gap_mean:6.3f} ± {row.gap_std:.3f} mm "
                    f"({row.gap_type})"
                )

        print(f"\n{'Statistics by Gap Type':^60}")
        print("-" * 60)

        for gap_type in ["intra-sector", "inter-sector", None]:
            label = gap_type if gap_type else "Combined"
            stats = self.statistics(gap_type)
            if not stats.empty:
                print(f"\n{label}:")
                print(stats.to_string())

        print(f"\n{'=' * 60}\n")

    def _get_phases_newest_first(self, sector: int) -> list[str]:
        """Return measurement phases from a sector workbook, newest first.

        Reads all sheets from the workbook and returns them in reverse order,
        so the most recent phase is first. This allows fallback to earlier
        phases if the latest doesn't contain required data.

        Parameters
        ----------
        sector : int
            Sector number

        Returns
        -------
        list[str]
            Phase names ordered from newest to oldest
        """
        import openpyxl
        from nova.assembly.sectorfile import SectorFile

        sf = SectorFile(sector=sector, private=self.private)
        filepath = sf.datadir + "/" + sf.filename + ".xlsx"
        book = openpyxl.load_workbook(filepath, read_only=True)
        sheets = [sh for sh in book.sheetnames if sh not in ["Metadata", "Nominal"]]
        book.close()
        return list(reversed(sheets)) if sheets else [self.phase]

    def position_statistics(
        self,
        assembly_phase: int | None = None,
        phase_selection: str = "tfgs_landing",
    ) -> tuple[pandas.DataFrame, alt.Chart]:
        """Calculate position statistics from installed sectors.

        Extracts coil position parameters using the specified phase selection
        strategy. Calculates sample statistics and variance estimates with
        confidence intervals.

        Parameters
        ----------
        assembly_phase : int | None
            Number of sectors to include (5=sector 5 only, 6=sectors 5-6, etc.).
            If None, uses all installed sectors.
        phase_selection : str
            Phase selection strategy:
            - "latest": Use the last sheet from each workbook (most recent data)
            - "tfgs_landing": Prefer TFGS Landing phase, fall back to In-pit target
            - "in_pit": Use In-pit target phase
            - Any other string: Use that exact phase name

        Returns
        -------
        tuple[pandas.DataFrame, alt.Chart]
            DataFrame with position statistics and Altair chart visualization.
        """
        from nova.assembly.sectordata import SectorData

        # Map assembly phase to sectors
        sector_order = [5, 6, 7, 8]  # Installation order
        if assembly_phase is not None:
            if assembly_phase < 5 or assembly_phase > 8:
                raise ValueError(f"assembly_phase must be 5-8, got {assembly_phase}")
            n_sectors = assembly_phase - 4
            active_sectors = sector_order[:n_sectors]
        else:
            active_sectors = list(self.sectors.keys())

        # Trial.py reference windows (half-widths for uniform distributions)
        # From Vault class: theta: [1.5, 1.5, 3, 3, 2, 2, 5]
        # Maps to: radial, tangential, roll_length, yaw_length,
        #          radial_ccl, tangential_ccl, radial_wall
        trial_windows = {
            "radial": 1.5,
            "tangential": 1.5,
            "vertical": 1.5,  # Not in trial but similar to translations
            "roll_length": 3.0,
            "yaw_length": 3.0,
            "pitch_length": 3.0,  # Similar to other rotation lengths
        }

        # Collect position data from each sector
        all_positions = []

        for sector in active_sectors:
            coils = self.sectors.get(sector, [])
            if not coils:
                continue

            # Determine best phase to use based on selection strategy
            try:
                sd = SectorData(sector, private=self.private)
                available_phases = sd.phase
            except (FileNotFoundError, KeyError):
                continue

            # Build list of phases to try (in priority order)
            phases_to_try: list[str] = []
            match phase_selection:
                case "latest":
                    # Try phases from newest to oldest until one works
                    phases_to_try = self._get_phases_newest_first(sector)
                case "tfgs_landing":
                    for tfgs_phase in ["TFGS Landing", "TFGS landing"]:
                        if tfgs_phase in available_phases:
                            phases_to_try.append(tfgs_phase)
                    phases_to_try.append(self.phase)  # fallback
                case "in_pit":
                    phases_to_try = ["In-pit target"]
                case _:
                    # Use exact phase name provided
                    phases_to_try = [phase_selection]

            # Try each phase until extraction succeeds
            extraction_success = False
            for phase_to_use in phases_to_try:
                try:
                    sector_data = FiducialSector(
                        phase=phase_to_use,
                        sectors={sector: coils},
                        private=self.private,
                    )
                    positions = sector_data.extract_coil_positions(pcr=self.pcr)
                    # Reset index to make 'coil' a column
                    positions = positions.reset_index()
                    positions["sector"] = sector
                    positions["phase"] = phase_to_use
                    all_positions.append(positions)
                    extraction_success = True
                    break  # Success, move to next sector
                except (FileNotFoundError, KeyError, ValueError):
                    continue  # Try next phase

            if not extraction_success:
                print(f"Warning: No valid phase found for sector {sector}")

        if not all_positions:
            raise ValueError("No position data available")

        # Combine all positions
        data = pandas.concat(all_positions, ignore_index=True)

        # Calculate statistics for each parameter
        params = [
            "radial",
            "tangential",
            "vertical",
            "roll_length",
            "yaw_length",
            "pitch_length",
        ]
        stats = []

        for param in params:
            values = data[param].dropna()
            if len(values) == 0:
                continue

            sample_mean = values.mean()
            sample_std = values.std(ddof=1)
            sample_var = values.var(ddof=1)

            # Chi-squared confidence interval for variance
            # (n-1)*s^2 / chi2_upper < sigma^2 < (n-1)*s^2 / chi2_lower
            from scipy import stats as sp_stats

            alpha = 0.05
            df = len(values) - 1
            if df > 0:
                chi2_lower = sp_stats.chi2.ppf(alpha / 2, df)
                chi2_upper = sp_stats.chi2.ppf(1 - alpha / 2, df)
                var_lower = df * sample_var / chi2_upper
                var_upper = df * sample_var / chi2_lower
                std_lower = np.sqrt(var_lower)
                std_upper = np.sqrt(var_upper)
            else:
                std_lower = std_upper = np.nan

            # Trial window half-width
            trial_hw = trial_windows.get(param, np.nan)

            stats.append(
                {
                    "parameter": param,
                    "n": len(values),
                    "mean": sample_mean,
                    "std": sample_std,
                    "std_lower_95": std_lower,
                    "std_upper_95": std_upper,
                    "trial_halfwidth": trial_hw,
                }
            )

        stats_df = pandas.DataFrame(stats)

        # Build data for Altair chart
        chart_data = []
        for _, row in data.iterrows():
            for param in params:
                if pandas.notna(row.get(param)):
                    chart_data.append(
                        {
                            "coil": int(row["coil"]),
                            "sector": int(row["sector"]),
                            "phase": row["phase"],
                            "parameter": param,
                            "value": row[param],
                        }
                    )

        chart_df = pandas.DataFrame(chart_data)

        # Add statistics to chart data
        for _, stat_row in stats_df.iterrows():
            param = stat_row["parameter"]
            chart_df.loc[chart_df["parameter"] == param, "mean"] = stat_row["mean"]
            chart_df.loc[chart_df["parameter"] == param, "std"] = stat_row["std"]
            chart_df.loc[chart_df["parameter"] == param, "std_lower_95"] = stat_row[
                "std_lower_95"
            ]
            chart_df.loc[chart_df["parameter"] == param, "std_upper_95"] = stat_row[
                "std_upper_95"
            ]
            chart_df.loc[chart_df["parameter"] == param, "trial_halfwidth"] = stat_row[
                "trial_halfwidth"
            ]

        # Create Altair chart
        chart = self._build_position_chart(chart_df, stats_df, len(active_sectors))

        return stats_df, chart

    def _build_position_chart(
        self,
        chart_df: pandas.DataFrame,
        stats_df: pandas.DataFrame,
        n_sectors: int,
    ) -> alt.Chart:
        """Build Altair visualization for position statistics.

        Creates a 3x2 grid with translations on top row and rotations on bottom.

        Parameters
        ----------
        chart_df : pandas.DataFrame
            Per-coil position data
        stats_df : pandas.DataFrame
            Summary statistics
        n_sectors : int
            Number of sectors included

        Returns
        -------
        alt.Chart
            Altair chart object
        """
        # Parameter layout: 3x2 grid
        # Top row: radial, tangential, vertical
        # Bottom row: roll_length, yaw_length, pitch_length
        param_order = [
            "radial",
            "tangential",
            "vertical",
            "roll_length",
            "yaw_length",
            "pitch_length",
        ]
        row_map = {
            "radial": 0,
            "tangential": 0,
            "vertical": 0,
            "roll_length": 1,
            "yaw_length": 1,
            "pitch_length": 1,
        }
        col_map = {
            "radial": 0,
            "tangential": 1,
            "vertical": 2,
            "roll_length": 0,
            "yaw_length": 1,
            "pitch_length": 2,
        }

        chart_df = chart_df.copy()
        chart_df["row"] = chart_df["parameter"].map(row_map)
        chart_df["col"] = chart_df["parameter"].map(col_map)

        # Base chart
        base = alt.Chart(chart_df).properties(width=180, height=150)

        # Bars for each coil value
        bars = base.mark_bar(opacity=0.7).encode(
            x=alt.X("coil:O", title="Coil"),
            y=alt.Y("value:Q", title="mm"),
            color=alt.Color(
                "sector:N",
                scale=alt.Scale(scheme="category10"),
                legend=alt.Legend(title="Sector"),
            ),
            tooltip=["coil", "sector", "phase", "value"],
        )

        # Mean line (dashed black)
        mean_rule = base.mark_rule(color="black", strokeDash=[4, 4]).encode(
            y="mean(mean):Q",
        )

        # Trial window lines (±half-width as dashed red lines)
        # Use transform to create both positive and negative lines
        trial_upper = base.mark_rule(
            color="red", strokeDash=[2, 2], opacity=0.7
        ).encode(
            y="mean(trial_halfwidth):Q",
        )
        trial_lower = (
            base.transform_calculate(neg_trial="-datum.trial_halfwidth")
            .mark_rule(color="red", strokeDash=[2, 2], opacity=0.7)
            .encode(y="mean(neg_trial):Q")
        )

        # Std bands using error bars
        std_band = base.mark_errorbar(extent="stdev", color="gray").encode(
            y=alt.Y("value:Q"),
        )

        # Combined chart - layer bars, mean, std, trial lines, then facet
        combined = bars + mean_rule + std_band + trial_upper + trial_lower

        chart = combined.facet(
            row=alt.Row("row:O", title=None, header=alt.Header(labels=False)),
            column=alt.Column(
                "parameter:N",
                title=None,
                sort=param_order,
            ),
        )

        # Add title
        chart = chart.properties(
            title=alt.TitleParams(
                text=f"Coil Position Statistics ({n_sectors} sectors)",
                subtitle=[
                    "Top: translations (mm) | Bottom: rotation lengths (mm)",
                    "Black dashed: mean | Red dashed: trial.py windows",
                ],
            )
        )

        return chart

    def plot_position_evolution(
        self,
        phase_selection: str = "tfgs_landing",
    ) -> alt.Chart:
        """Plot how position statistics evolve as sectors are installed.

        Creates charts showing variance estimates with confidence intervals
        for each assembly phase (5, 6, 7, 8 sectors).

        Parameters
        ----------
        phase_selection : str
            Phase selection strategy (see position_statistics for options).

        Returns
        -------
        alt.Chart
            Altair chart showing variance evolution
        """
        evolution_data = []

        for phase in range(5, 9):
            try:
                # Only include sectors up to this phase
                available_sectors = {
                    k: v for k, v in self.sectors.items() if k <= phase
                }
                if not available_sectors:
                    continue

                stats_df, _ = self.position_statistics(
                    assembly_phase=phase,
                    phase_selection=phase_selection,
                )

                for _, row in stats_df.iterrows():
                    evolution_data.append(
                        {
                            "assembly_phase": phase,
                            "n_coils": int(row["n"]),
                            "parameter": row["parameter"],
                            "std": row["std"],
                            "std_lower_95": row["std_lower_95"],
                            "std_upper_95": row["std_upper_95"],
                            "trial_halfwidth": row["trial_halfwidth"],
                        }
                    )
            except (ValueError, KeyError) as e:
                print(f"Warning: Phase {phase}: {e}")

        if not evolution_data:
            raise ValueError("No evolution data available")

        evo_df = pandas.DataFrame(evolution_data)

        # Parameter layout
        param_order = [
            "radial",
            "tangential",
            "vertical",
            "roll_length",
            "yaw_length",
            "pitch_length",
        ]

        base = alt.Chart(evo_df).properties(width=150, height=120)

        # Std estimate with confidence interval
        line = base.mark_line(point=True).encode(
            x=alt.X("assembly_phase:O", title="Sectors installed"),
            y=alt.Y("std:Q", title="σ (mm)"),
            color=alt.value("steelblue"),
        )

        band = base.mark_area(opacity=0.3).encode(
            x="assembly_phase:O",
            y="std_lower_95:Q",
            y2="std_upper_95:Q",
        )

        # Trial reference line
        trial_line = base.mark_rule(color="red", strokeDash=[2, 2]).encode(
            y="mean(trial_halfwidth):Q",
        )

        chart = (band + line + trial_line).facet(
            column=alt.Column("parameter:N", sort=param_order, title=None),
        )

        chart = chart.properties(
            title=alt.TitleParams(
                text="Variance Evolution with Assembly Progress",
                subtitle="Blue: estimated σ with 95% CI | Red: trial.py window",
            )
        )

        return chart


if __name__ == "__main__":
    # Sectors 5, 6, 7 are adjacent in the pit
    # Sector 8 contains coils 4 and 11 (coil 11 is adjacent to sector 7)
    sectors = {
        5: [16, 5],
        6: [12, 13],
        7: [8, 9],
        8: [4, 11],
    }

    # Load pit data
    pit = FiducialPit(
        sectors=sectors,
        phase="In-pit target",
        pcr=True,
        private=False,
    )

    # Print summary
    pit.print_summary()

    # Position statistics with latest phase from each workbook
    print("\n" + "=" * 60)
    print("Position Statistics (latest phase from each workbook)")
    print("=" * 60)
    stats_df, position_chart = pit.position_statistics(phase_selection="latest")
    print(stats_df.to_string())

    # Show evolution chart
    evolution_chart = pit.plot_position_evolution(phase_selection="latest")
    evolution_chart.show()

    # Show position chart
    position_chart.show()

    # Plot gaps
    pit.plot_gaps()
    plt.show()

    # Plot statistics
    pit.plot_statistics()
    plt.show()
