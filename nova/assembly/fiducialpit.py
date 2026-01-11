"""Integrate fiducial measurements from pit-installed sectors.

Combines measurements from installed sectors to calculate inter-sector and
intra-sector gaps using ILIS planes and CCL points.
"""

from dataclasses import dataclass, field
from functools import cached_property
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
import pandas
import scipy.spatial.transform
from scipy.interpolate import griddata

import seaborn as sns

from nova.assembly.fiducialdata import FiducialData
from nova.assembly.fiducialilis import FiducialIlis
from nova.assembly.fiducialsector import FiducialSector
from nova.assembly.ilisnominal import NominalIlis
from nova.assembly.transform import Rotate
from nova.graphics.plot import Plot1D


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
    # Cumulative gap limit: 36mm, cumulative gap target: 33mm
    # Within-sector gaps are 0.4mm smaller, between-sector gaps 0.4mm larger
    # so the average equals 33/18 ≈ 1.833mm
    within_sector_target: float = 33 / 18 - 0.4  # Target gap within a sector
    between_sector_target: float = 33 / 18 + 0.4  # Target gap between sectors
    gap_limit: float = 2.0  # Single gap limit (cumulative 36mm / 18 gaps)

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

    def _resolve_phase(self, sector: int) -> str:
        """Resolve phase selection strategy to actual phase name.

        Supports both literal phase names and selection strategies:
        - "latest": Most recent sheet with valid data
        - "tfgs_landing": TFGS Landing phase, fallback to In-pit target
        - "in_pit": In-pit target phase
        - Any other string: Use as literal phase name
        """
        from nova.assembly.sectordata import SectorData

        match self.phase:
            case "latest":
                phases_to_try = self._get_phases_newest_first(sector)
            case "tfgs_landing":
                try:
                    sd = SectorData(sector, private=self.private)
                    available = sd.phase
                except (FileNotFoundError, KeyError):
                    available = []
                phases_to_try = []
                # Include various TFGS sheet naming conventions
                for tfgs in ["AFTER TFGS landing", "TFGS Landing", "TFGS landing"]:
                    if tfgs in available:
                        phases_to_try.append(tfgs)
                phases_to_try.append("In-pit target")
            case "in_pit":
                phases_to_try = ["In-pit target"]
            case _:
                return self.phase  # Literal phase name

        # Try each phase until one works (has ILIS data)
        for phase in phases_to_try:
            try:
                fs = FiducialSector(
                    phase=phase,
                    sectors={sector: self.sectors[sector]},
                    private=self.private,
                )
                # Check that ILIS data exists for position extraction
                if len(fs.ilis) > 0:
                    return phase
            except (FileNotFoundError, KeyError, ValueError):
                continue
        return phases_to_try[0] if phases_to_try else "In-pit target"

    def _load_sectors(self):
        """Load fiducial data for each sector."""
        self.sector_data: dict[int, FiducialSector] = {}
        self._resolved_phases: dict[int, str] = {}
        for sector, coils in self.sectors.items():
            try:
                resolved = self._resolve_phase(sector)
                self._resolved_phases[sector] = resolved
                self.sector_data[sector] = FiducialSector(
                    phase=resolved,
                    sectors={sector: coils},
                    private=self.private,
                )
                print(f"Sector {sector}: loaded phase '{resolved}'")
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

        # Get plane objects for PCR offsets
        plane_first = sector_planes.loc[(coil_first, feature_first)]
        plane_second = sector_planes.loc[(coil_second, feature_second)]

        # Create regular grid for sampling (matching fiducialsector approach)
        grid_r, grid_z = np.mgrid[
            slice(ilis.data.r.min(), ilis.data.r.max(), 20j),
            slice(ilis.data.z.min(), ilis.data.z.max(), 40j),
        ]

        # Calculate offsets and interpolate to grid for each surface
        grid_offsets = []
        for coil, feature in sector_index:
            plane_mask = (sector_data_clocked.coil == coil) & (
                sector_data_clocked.feature == feature
            )
            plane_data = sector_data_clocked.loc[plane_mask]
            plane_offset = ilis.offset(plane_data[["x", "y", "z"]], midplane)

            # Interpolate offsets onto regular grid
            grid_offset = griddata(
                plane_data[["r", "z"]].values,
                plane_offset,
                (grid_r, grid_z),
                method="linear",
            )
            grid_offsets.append(grid_offset)

        # Gap at each grid point = offset_second - offset_first
        gap_grid = grid_offsets[1] - grid_offsets[0]

        # Compute statistics from grid (excludes NaN from interpolation boundaries)
        valid_mask = ~np.isnan(gap_grid)
        valid_gaps = gap_grid[valid_mask]
        gap_mean = float(np.mean(valid_gaps))
        gap_std = float(np.std(valid_gaps))
        gap_min = float(np.min(valid_gaps))
        gap_max = float(np.max(valid_gaps))

        # Gap at equatorial midplane (z=0): interpolate directly at z=0
        r_range = np.linspace(ilis.data.r.min(), ilis.data.r.max(), 20)
        z0_points = np.column_stack([r_range, np.zeros_like(r_range)])

        # Interpolate each surface offset at z=0
        midplane_offsets = []
        for coil, feature in sector_index:
            plane_mask = (sector_data_clocked.coil == coil) & (
                sector_data_clocked.feature == feature
            )
            plane_data = sector_data_clocked.loc[plane_mask]
            plane_offset = ilis.offset(plane_data[["x", "y", "z"]], midplane)

            z0_offset = griddata(
                plane_data[["r", "z"]].values,
                plane_offset,
                z0_points,
                method="linear",
            )
            midplane_offsets.append(z0_offset)

        # Gap at z=0 = offset_second - offset_first
        gap_z0 = midplane_offsets[1] - midplane_offsets[0]
        valid_z0 = gap_z0[~np.isnan(gap_z0)]
        if len(valid_z0) > 0:
            gap_midplane = float(np.mean(valid_z0))
        else:
            gap_midplane = gap_mean

        # Get PCR deviation offsets if available
        pcr_first = plane_first.get("offset", 0)
        pcr_second = plane_second.get("offset", 0)

        return {
            "coil_first": coil_first,
            "coil_second": coil_second,
            "gap_type": pair["gap_type"],
            "sector_a": pair["sector_a"],
            "sector_b": pair["sector_b"],
            "position": pair["position"],
            "gap_midplane": gap_midplane,
            "gap_mean": gap_mean,
            "gap_std": gap_std,
            "gap_min": gap_min,
            "gap_max": gap_max,
            "grid_points": len(valid_gaps),
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
        data_first_filtered = data_first[data_first.feature == feature_first].copy()
        data_second_filtered = data_second[data_second.feature == feature_second].copy()

        # Combine data to get common grid bounds
        all_data = pandas.concat([data_first_filtered, data_second_filtered])

        # Create regular grid for sampling (matching fiducialsector approach)
        grid_r, grid_z = np.mgrid[
            slice(all_data.r.min(), all_data.r.max(), 20j),
            slice(all_data.z.min(), all_data.z.max(), 40j),
        ]

        # Calculate offsets and interpolate to grid for each surface
        def calculate_offset(data):
            """Calculate offset from midplane for point cloud data."""
            normal = midplane.loc[["nx", "ny", "nz"]].values
            normal = normal / np.linalg.norm(normal)
            point = midplane.loc[["x", "y", "z"]].values
            v = data[["x", "y", "z"]].values - point
            return np.dot(v, normal)

        grid_offsets = []
        for data_filtered in [data_first_filtered, data_second_filtered]:
            if data_filtered.empty:
                grid_offsets.append(np.full_like(grid_r, np.nan))
                continue

            plane_offset = calculate_offset(data_filtered)

            # Interpolate offsets onto regular grid
            grid_offset = griddata(
                data_filtered[["r", "z"]].values,
                plane_offset,
                (grid_r, grid_z),
                method="linear",
            )
            grid_offsets.append(grid_offset)

        # Gap at each grid point = offset_second - offset_first
        gap_grid = grid_offsets[1] - grid_offsets[0]

        # Compute statistics from grid (excludes NaN from interpolation boundaries)
        valid_mask = ~np.isnan(gap_grid)
        valid_gaps = gap_grid[valid_mask]
        if len(valid_gaps) > 0:
            gap_mean = float(np.mean(valid_gaps))
            gap_std = float(np.std(valid_gaps))
            gap_min = float(np.min(valid_gaps))
            gap_max = float(np.max(valid_gaps))

            # Gap at equatorial midplane (z=0): interpolate directly at z=0
            r_range = np.linspace(all_data.r.min(), all_data.r.max(), 20)
            z0_points = np.column_stack([r_range, np.zeros_like(r_range)])

            # Interpolate each surface offset at z=0
            midplane_offsets = []
            for data_filtered in [data_first_filtered, data_second_filtered]:
                if data_filtered.empty:
                    midplane_offsets.append(np.full(len(r_range), np.nan))
                    continue

                plane_offset = calculate_offset(data_filtered)
                z0_offset = griddata(
                    data_filtered[["r", "z"]].values,
                    plane_offset,
                    z0_points,
                    method="linear",
                )
                midplane_offsets.append(z0_offset)

            # Gap at z=0 = offset_second - offset_first
            gap_z0 = midplane_offsets[1] - midplane_offsets[0]
            valid_z0 = gap_z0[~np.isnan(gap_z0)]
            if len(valid_z0) > 0:
                gap_midplane = float(np.mean(valid_z0))
            else:
                gap_midplane = gap_mean
        else:
            gap_mean = gap_std = gap_min = gap_max = gap_midplane = 0.0

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
            "gap_midplane": gap_midplane,
            "gap_mean": gap_mean,
            "gap_std": gap_std,
            "gap_min": gap_min,
            "gap_max": gap_max,
            "grid_points": len(valid_gaps),
            "pcr_first": pcr_first,
            "pcr_second": pcr_second,
        }

    @cached_property
    def gaps(self) -> pandas.DataFrame:
        """Return cached gap calculations."""
        return self.calculate_gaps()

    def gap_profile(
        self,
        coil_first: int,
        coil_second: int,
        radius: float | None = None,
        n_points: int = 100,
    ) -> pandas.DataFrame:
        """Extract gap profile along z at a specific radius.

        Parameters
        ----------
        coil_first : int
            First coil number
        coil_second : int
            Second coil number
        radius : float, optional
            Radius at which to extract profile. If None, uses inner ILIS radius.
        n_points : int
            Number of points along z

        Returns
        -------
        pandas.DataFrame
            Gap profile with columns: z, gap, r
        """
        # Find the gap pair
        pair = None
        for p in self.gap_pairs:
            if p["coil_plus"] == coil_first and p["coil_minus"] == coil_second:
                pair = p
                break
            if p["coil_plus"] == coil_second and p["coil_minus"] == coil_first:
                pair = p
                coil_first, coil_second = coil_second, coil_first
                break

        if pair is None:
            raise ValueError(f"No gap pair found for coils {coil_first}-{coil_second}")

        sector_a = pair["sector_a"]
        sector_b = pair["sector_b"]

        if sector_a == sector_b:
            return self._gap_profile_intra(
                coil_first, coil_second, sector_a, radius, n_points
            )
        else:
            return self._gap_profile_inter(
                coil_first, coil_second, sector_a, sector_b, radius, n_points
            )

    def _gap_profile_intra(
        self,
        coil_first: int,
        coil_second: int,
        sector: int,
        radius: float | None,
        n_points: int,
    ) -> pandas.DataFrame:
        """Extract gap profile for intra-sector gap."""
        feature_first = "ILIS +1"
        feature_second = "ILIS -1"

        sector_coils = list(self.sector_data[sector].delta.keys())
        sector_data = self.sector_data[sector]
        ilis = FiducialIlis(sector_data.ilis, pcr=self.pcr)

        # Clock planes and data to midplane
        def clock_group(planes, coil):
            is_first = sector_coils.index(coil) == 0
            return self._clock_planes(planes, coil, is_first)

        sector_planes = ilis.planes.groupby(["coil"], group_keys=False).apply(
            lambda x: clock_group(x, x.name)
        )

        def clock_data_group(data, coil):
            is_first = sector_coils.index(coil) == 0
            return self._clock_transform(data, coil, is_first)

        sector_data_clocked = ilis.data.groupby(["coil"], group_keys=False).apply(
            lambda x: clock_data_group(x, x.name)
        )

        sector_index = [
            (coil_first, feature_first),
            (coil_second, feature_second),
        ]

        midplane = ilis.intersect(sector_planes.loc[sector_index])

        # Use inner radius if not specified, nudged inward to stay within convex hull
        if radius is None:
            r_min = ilis.data.r.min()
            r_max = ilis.data.r.max()
            radius = r_min + 0.01 * (r_max - r_min)

        # Create z range spanning the ILIS data, nudged inward
        z_min = ilis.data.z.min()
        z_max = ilis.data.z.max()
        z_margin = 0.01 * (z_max - z_min)
        z_range = np.linspace(z_min + z_margin, z_max - z_margin, n_points)
        query_points = np.column_stack([np.full(n_points, radius), z_range])

        # Interpolate each surface offset at the query points
        profile_offsets = []
        for coil, feature in sector_index:
            plane_mask = (sector_data_clocked.coil == coil) & (
                sector_data_clocked.feature == feature
            )
            plane_data = sector_data_clocked.loc[plane_mask]
            plane_offset = ilis.offset(plane_data[["x", "y", "z"]], midplane)

            offset_interp = griddata(
                plane_data[["r", "z"]].values,
                plane_offset,
                query_points,
                method="linear",
            )
            profile_offsets.append(offset_interp)

        # Gap = offset_second - offset_first
        gap_profile = profile_offsets[1] - profile_offsets[0]

        return pandas.DataFrame({"z": z_range, "gap": gap_profile, "r": radius})

    def _gap_profile_inter(
        self,
        coil_first: int,
        coil_second: int,
        sector_a: int,
        sector_b: int,
        radius: float | None,
        n_points: int,
    ) -> pandas.DataFrame:
        """Extract gap profile for inter-sector gap."""
        feature_first = "ILIS +1"
        feature_second = "ILIS -1"

        ilis_a = FiducialIlis(self.sector_data[sector_a].ilis, pcr=self.pcr)
        ilis_b = FiducialIlis(self.sector_data[sector_b].ilis, pcr=self.pcr)

        def clock_planes(planes):
            planes = planes.copy()
            planes.loc[:, ["x", "y", "z"]] = self.rotate.clock(
                planes.loc[:, ["x", "y", "z"]].values
            )
            planes.loc[:, ["nx", "ny", "nz"]] = self.rotate.clock(
                planes.loc[:, ["nx", "ny", "nz"]].values
            )
            return planes

        def anticlock_planes(planes):
            planes = planes.copy()
            planes.loc[:, ["x", "y", "z"]] = self.rotate.anticlock(
                planes.loc[:, ["x", "y", "z"]].values
            )
            planes.loc[:, ["nx", "ny", "nz"]] = self.rotate.anticlock(
                planes.loc[:, ["nx", "ny", "nz"]].values
            )
            return planes

        def clock_data(data):
            data = data.copy()
            data.loc[:, ["x", "y", "z"]] = self.rotate.clock(
                data.loc[:, ["x", "y", "z"]].values
            )
            return data

        def anticlock_data(data):
            data = data.copy()
            data.loc[:, ["x", "y", "z"]] = self.rotate.anticlock(
                data.loc[:, ["x", "y", "z"]].values
            )
            return data

        planes_first = anticlock_planes(ilis_a.planes.loc[[coil_first]])
        data_first = anticlock_data(ilis_a.data[ilis_a.data.coil == coil_first].copy())

        planes_second = clock_planes(ilis_b.planes.loc[[coil_second]])
        data_second = clock_data(ilis_b.data[ilis_b.data.coil == coil_second].copy())

        plane_first = planes_first.loc[(coil_first, feature_first)]
        plane_second = planes_second.loc[(coil_second, feature_second)]

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

        midplane = FiducialIlis.intersect(rotated_planes)

        # Combine data to get z range
        data_first_filtered = data_first[data_first.feature == feature_first].copy()
        data_second_filtered = data_second[data_second.feature == feature_second].copy()
        all_data = pandas.concat([data_first_filtered, data_second_filtered])

        # Use inner radius if not specified, nudged inward to stay within convex hull
        if radius is None:
            r_min = all_data.r.min()
            r_max = all_data.r.max()
            radius = r_min + 0.01 * (r_max - r_min)

        # Create z range nudged inward to stay within convex hull
        z_min = all_data.z.min()
        z_max = all_data.z.max()
        z_margin = 0.01 * (z_max - z_min)
        z_range = np.linspace(z_min + z_margin, z_max - z_margin, n_points)
        query_points = np.column_stack([np.full(n_points, radius), z_range])

        def calculate_offset(data):
            normal = midplane.loc[["nx", "ny", "nz"]].values
            normal = normal / np.linalg.norm(normal)
            point = midplane.loc[["x", "y", "z"]].values
            v = data[["x", "y", "z"]].values - point
            return np.dot(v, normal)

        profile_offsets = []
        for data_filtered in [data_first_filtered, data_second_filtered]:
            if data_filtered.empty:
                profile_offsets.append(np.full(n_points, np.nan))
                continue

            plane_offset = calculate_offset(data_filtered)
            offset_interp = griddata(
                data_filtered[["r", "z"]].values,
                plane_offset,
                query_points,
                method="linear",
            )
            profile_offsets.append(offset_interp)

        gap_profile = profile_offsets[1] - profile_offsets[0]

        return pandas.DataFrame({"z": z_range, "gap": gap_profile, "r": radius})

    def plot_gap_profile(
        self,
        coil_first: int,
        coil_second: int,
        radius: float | None = None,
        measurements: pandas.DataFrame | None = None,
        figsize: tuple = (10, 8),
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot gap profile along z at a specific radius.

        Parameters
        ----------
        coil_first : int
            First coil number
        coil_second : int
            Second coil number
        radius : float, optional
            Radius at which to extract profile. If None, uses inner ILIS radius.
        measurements : pandas.DataFrame, optional
            Measured gap data with columns 'z' and 'gap' to overlay
        figsize : tuple
            Figure size

        Returns
        -------
        tuple[plt.Figure, plt.Axes]
            Figure and axes
        """
        profile = self.gap_profile(coil_first, coil_second, radius=radius)

        with sns.plotting_context("poster"):
            fig, ax = plt.subplots(figsize=figsize)

            # Plot interpolated gap profile
            valid = ~profile["gap"].isna()
            ax.plot(
                profile.loc[valid, "z"],
                profile.loc[valid, "gap"],
                "-",
                color="C0",
                linewidth=2,
                label=f"Predicted (r={profile['r'].iloc[0]:.0f} mm)",
            )

            # Overlay measurements if provided
            if measurements is not None:
                ax.scatter(
                    measurements["z"],
                    measurements["gap"],
                    s=100,
                    color="C1",
                    marker="o",
                    edgecolor="k",
                    label="Measured",
                    zorder=5,
                )

            ax.set_xlabel("z (mm)")
            ax.set_ylabel("Gap (mm)")
            ax.set_title(f"Gap Profile: Coils {coil_first}-{coil_second}")
            ax.legend()
            ax.axhline(0, color="gray", linewidth=0.5)

            sns.despine(ax=ax)
            fig.tight_layout()

        return fig, ax

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

    def plot_gaps(self, figsize=(12, 8)):
        """Plot gap measurements as bar chart with min/max range markers.

        Shows gap_midplane (mean gap at z=0) as bar height with horizontal
        lines indicating [gap_min, gap_max] range from grid samples.
        Colored by gap type (within/between-sector).
        """
        if self.gaps.empty:
            print("No gaps to plot")
            return

        with sns.plotting_context("poster"):
            fig, ax = plt.subplots(figsize=figsize)
            colors = {"intra-sector": "C0", "inter-sector": "C1"}
            labels_map = {
                "intra-sector": "within-sector",
                "inter-sector": "between-sector",
            }

            bar_labels = [
                f"{row.coil_first}-{row.coil_second}" for _, row in self.gaps.iterrows()
            ]
            x = np.arange(len(bar_labels))
            bar_width = 0.8

            # Create bars with legend labels for first occurrence of each type
            plotted_types = set()
            for i, (_, row) in enumerate(self.gaps.iterrows()):
                gap_type = row["gap_type"]
                label = labels_map[gap_type] if gap_type not in plotted_types else None

                # Use gap_midplane (mean gap at z=0) as bar height
                gap_value = row["gap_midplane"]

                ax.bar(
                    x[i],
                    gap_value,
                    width=bar_width,
                    color=colors[gap_type],
                    alpha=0.8,
                    edgecolor="k",
                    label=label,
                )

                # Add min/max range markers as dark gray horizontal lines
                ax.hlines(
                    row["gap_min"],
                    x[i] - bar_width / 3,
                    x[i] + bar_width / 3,
                    colors="darkgray",
                    linewidth=2,
                )
                ax.hlines(
                    row["gap_max"],
                    x[i] - bar_width / 3,
                    x[i] + bar_width / 3,
                    colors="darkgray",
                    linewidth=2,
                )
                # Connect min/max with vertical line
                ax.vlines(
                    x[i],
                    row["gap_min"],
                    row["gap_max"],
                    colors="darkgray",
                    linewidth=1,
                )

                plotted_types.add(gap_type)

                # Add sector label at base of within-sector (intra-sector) bars
                if gap_type == "intra-sector":
                    sector = int(row["sector_a"])
                    ax.text(
                        x[i],
                        0.05,
                        f"S{sector}",
                        ha="center",
                        va="bottom",
                        fontsize=24,
                        color="lightgray",
                        fontweight="bold",
                    )

            ax.set_xticks(x)
            ax.set_xticklabels(bar_labels, rotation=45, ha="right")
            ax.set_xlabel("Gap (coil-coil)")
            ax.set_ylabel("Gap (mm)")
            ax.set_title(f"Pit Gap Analysis - {self.phase}")

            # Add target lines with text labels on the right side (not in legend)
            xlim = ax.get_xlim()
            ax.axhline(
                self.within_sector_target,
                color="C0",
                linestyle="--",
                alpha=0.7,
            )
            ax.text(
                xlim[1],
                self.within_sector_target,
                f" {self.within_sector_target:.1f}",
                va="center",
                ha="left",
                color="C0",
            )
            ax.axhline(
                self.between_sector_target,
                color="C1",
                linestyle="--",
                alpha=0.7,
            )
            ax.text(
                xlim[1],
                self.between_sector_target,
                f" {self.between_sector_target:.1f}",
                va="center",
                ha="left",
                color="C1",
            )
            ax.legend(loc="upper left")
            ax.axhline(0, color="gray", linewidth=0.5)

            # Add value labels at top of bar
            for i, (_, row) in enumerate(self.gaps.iterrows()):
                gap_value = row.gap_midplane
                ax.annotate(
                    f"{gap_value:.2f}",
                    xy=(i, gap_value),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                )

            sns.despine(ax=ax)
            fig.tight_layout()
        return fig, ax

    def plot_statistics(self, figsize=(10, 7)):
        """Plot summary statistics as bar chart with error bars.

        Shows mean gap ± between-gap std for each gap type.
        """
        if self.gaps.empty:
            print("No statistics to plot")
            return

        with sns.plotting_context("poster"):
            fig, ax = plt.subplots(figsize=figsize)

            gap_types = ["intra-sector", "inter-sector", "combined"]
            colors = {"intra-sector": "C0", "inter-sector": "C1", "combined": "C2"}
            labels_map = {
                "intra-sector": "within-sector",
                "inter-sector": "between-sector",
                "combined": "combined",
            }
            x = np.arange(len(gap_types))

            means = []
            stds = []
            for gap_type in gap_types:
                if gap_type == "combined":
                    subset = self.gaps
                else:
                    subset = self.gaps[self.gaps["gap_type"] == gap_type]
                if not subset.empty:
                    means.append(subset["gap_midplane"].mean())
                    stds.append(subset["gap_midplane"].std())
                else:
                    means.append(0)
                    stds.append(0)

            bar_colors = [colors[gt] for gt in gap_types]
            display_labels = [labels_map[gt] for gt in gap_types]
            ax.bar(
                x,
                means,
                yerr=stds,
                capsize=5,
                alpha=0.8,
                edgecolor="k",
                color=bar_colors,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(display_labels)
            ax.set_ylabel("Gap (mm)")
            ax.set_title(f"Gap Statistics Summary - {self.phase}")
            ax.axhline(0, color="gray", linewidth=0.5)

            # Add limit line with text label (no legend)
            xlim = ax.get_xlim()
            ax.axhline(
                self.gap_limit,
                color="C3",
                linestyle="--",
                alpha=0.7,
            )
            ax.text(
                xlim[1],
                self.gap_limit,
                f" limit: {self.gap_limit:.1f}",
                va="center",
                ha="left",
                color="C3",
            )

            sns.despine(ax=ax)
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
        # Return sheets in reverse order (newest first), fallback to In-pit target
        return list(reversed(sheets)) if sheets else ["In-pit target"]

    def position_statistics(
        self,
        assembly_phase: int | None = None,
    ) -> tuple[pandas.DataFrame, tuple[plt.Figure, np.ndarray]]:
        """Calculate position statistics from installed sectors.

        Extracts coil position parameters using the phase configured at
        class initialization. Calculates sample statistics and variance
        estimates with confidence intervals.

        Parameters
        ----------
        assembly_phase : int | None
            Number of sectors to include in installation order (1-4).
            Phase 1=SM6, 2=SM6+SM7, 3=SM6+SM7+SM5, 4=all four sectors.
            If None, uses all installed sectors.

        Returns
        -------
        tuple[pandas.DataFrame, tuple[plt.Figure, np.ndarray]]
            DataFrame with position statistics and matplotlib figure/axes.
        """
        # Map assembly phase to sectors (in installation order: 6, 7, 5, 8)
        # Based on git history: SM6 Feb 2025, SM7 Mar 2025, SM5 Aug 2025, SM8 Oct 2025
        sector_order = [6, 7, 5, 8]
        if assembly_phase is not None:
            if assembly_phase < 1 or assembly_phase > 4:
                raise ValueError(f"assembly_phase must be 1-4, got {assembly_phase}")
            active_sectors = sector_order[:assembly_phase]
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

        # Collect position data from each sector using already-loaded sector_data
        all_positions = []

        for sector in active_sectors:
            if sector not in self.sector_data:
                continue

            sector_data = self.sector_data[sector]
            phase_used = self._resolved_phases.get(sector, self.phase)

            try:
                positions = sector_data.extract_coil_positions(pcr=self.pcr)
                # Reset index to make 'coil' a column
                positions = positions.reset_index()
                positions["sector"] = sector
                positions["phase"] = phase_used
                all_positions.append(positions)
            except (KeyError, ValueError) as e:
                print(f"Warning: Could not extract positions for sector {sector}: {e}")

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
    ) -> tuple[plt.Figure, np.ndarray]:
        """Build matplotlib visualization for position statistics.

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
        tuple[plt.Figure, np.ndarray]
            Matplotlib figure and axes array
        """
        # Parameter layout: 2 rows x 3 cols
        # Top row: radial, tangential, vertical (translations)
        # Bottom row: roll_length, yaw_length, pitch_length (rotations)
        param_grid = [
            ["radial", "tangential", "vertical"],
            ["roll_length", "yaw_length", "pitch_length"],
        ]

        # Display labels with underscores replaced by spaces
        param_labels = {p: p.replace("_", " ") for p in sum(param_grid, [])}

        # Installation order for sectors
        sector_order = [6, 7, 5, 8]

        # Order coils by sector installation order, then by position within sector
        coil_order = []
        for sector in sector_order:
            sector_coils = chart_df[chart_df["sector"] == sector]["coil"].unique()
            # Sort by toroidal position within sector
            sector_coils_sorted = sorted(
                sector_coils, key=lambda c: self.location.index(c)
            )
            coil_order.extend(sector_coils_sorted)

        # Assign colors by installation order
        sector_colors = {s: f"C{i}" for i, s in enumerate(sector_order)}

        # Create figure with 2x3 grid
        axes = self.mpl_axes.generate(style="1d", nrows=2, ncols=3, figsize=(10, 5))
        fig = self.mpl_axes.fig

        # Trial windows for reference lines
        trial_windows = {
            "radial": 1.5,
            "tangential": 1.5,
            "vertical": 1.5,
            "roll_length": 3.0,
            "yaw_length": 3.0,
            "pitch_length": 3.0,
        }

        for row_idx, row_params in enumerate(param_grid):
            for col_idx, param in enumerate(row_params):
                ax = axes[row_idx, col_idx]
                param_data = chart_df[chart_df["parameter"] == param]

                if param_data.empty:
                    ax.set_visible(False)
                    continue

                # Plot bars for each coil
                x_positions = np.arange(len(coil_order))
                bar_width = 0.8

                for i, coil in enumerate(coil_order):
                    coil_data = param_data[param_data["coil"] == coil]
                    if coil_data.empty:
                        continue
                    value = coil_data["value"].iloc[0]
                    sector = coil_data["sector"].iloc[0]
                    ax.bar(
                        i,
                        value,
                        width=bar_width,
                        color=sector_colors[sector],
                        edgecolor="none",
                        alpha=0.8,
                    )

                # Add trial window lines with limit labels
                trial_hw = trial_windows.get(param, 1.5)
                ax.axhline(trial_hw, color="C3", linestyle=":", linewidth=1, alpha=0.7)
                ax.axhline(-trial_hw, color="C3", linestyle=":", linewidth=1, alpha=0.7)

                # Axis formatting - set xlim first so we can position text
                ax.set_xlim(-0.5, len(coil_order) - 0.5)

                # Add limit label above upper line, right-aligned within plot
                ax.text(
                    len(coil_order) - 0.6,
                    trial_hw,
                    f"limit {trial_hw:.1f}mm",
                    ha="right",
                    va="bottom",
                    fontsize=7,
                    color="C3",
                )

                # Add title inside plot, upper right
                ax.text(
                    0.97,
                    0.95,
                    param_labels[param],
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=10,
                    fontweight="bold",
                )

                # Only show x-axis labels on bottom row
                if row_idx == 1:
                    ax.set_xticks(x_positions)
                    ax.set_xticklabels(coil_order, fontsize=8)
                    ax.set_xlabel("TF coil", fontsize=9)
                else:
                    ax.set_xticks([])

                # Only show y-axis label on leftmost column
                if col_idx == 0:
                    ax.set_ylabel("alignment, mm", fontsize=9)

        # Despine all axes
        self.mpl_axes.despine(axes.flatten())

        # Add legend for sectors in installation order
        handles = []
        labels = []
        for s in sector_order:
            if s in chart_df["sector"].values:
                # Check if this sector uses target phase
                sector_phases = chart_df[chart_df["sector"] == s]["phase"].unique()
                is_target = any("target" in p.lower() for p in sector_phases)
                suffix = " (target)" if is_target else ""
                handles.append(
                    plt.Line2D([0], [0], color=sector_colors[s], linewidth=6, alpha=0.8)
                )
                labels.append(f"Sector {s}{suffix}")
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            frameon=False,
            fontsize=8,
            ncol=len(handles),
        )

        fig.suptitle(
            "Coil Position Statistics",
            fontsize=11,
            fontweight="bold",
            y=1.02,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        return fig, axes

    def plot_position_evolution(self) -> tuple[plt.Figure, np.ndarray]:
        """Plot how position statistics evolve as sectors are installed.

        Creates charts showing variance estimates with confidence intervals
        for each assembly phase (1-4 sectors).

        Returns
        -------
        tuple[plt.Figure, np.ndarray]
            Matplotlib figure and axes array
        """
        evolution_data = []

        # Sector installation order: 6, 7, 5, 8
        sector_order = [6, 7, 5, 8]

        # Assembly phases 1-4 correspond to installation order: 6, 7, 5, 8
        for phase in range(1, 5):
            try:
                stats_df, _ = self.position_statistics(
                    assembly_phase=phase,
                )

                for _, row in stats_df.iterrows():
                    evolution_data.append(
                        {
                            "assembly_phase": phase,
                            "sector": sector_order[phase - 1],
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

        # Parameter layout: 2 rows x 3 cols
        param_grid = [
            ["radial", "tangential", "vertical"],
            ["roll_length", "yaw_length", "pitch_length"],
        ]

        # Create figure with 2x3 grid
        axes = self.mpl_axes.generate(style="1d", nrows=2, ncols=3, figsize=(10, 5))
        fig = self.mpl_axes.fig

        # X-axis labels as sector numbers
        x_labels = [str(s) for s in sector_order]

        for row_idx, row_params in enumerate(param_grid):
            for col_idx, param in enumerate(row_params):
                ax = axes[row_idx, col_idx]
                param_data = evo_df[evo_df["parameter"] == param]

                if param_data.empty:
                    ax.set_visible(False)
                    continue

                phases = param_data["assembly_phase"].values
                std_vals = param_data["std"].values
                std_lower = param_data["std_lower_95"].values
                std_upper = param_data["std_upper_95"].values
                trial_hw = param_data["trial_halfwidth"].iloc[0]

                # Plot confidence interval band
                ax.fill_between(
                    phases, std_lower, std_upper, color="C0", alpha=0.25, linewidth=0
                )

                # Plot std line with markers
                ax.plot(phases, std_vals, "o-", color="C0", markersize=5, linewidth=1.5)

                # Trial reference line with limit label
                ax.axhline(trial_hw, color="C3", linestyle=":", linewidth=1, alpha=0.7)
                ax.text(
                    4.4,
                    trial_hw,
                    f"limit {trial_hw:.1f}mm",
                    ha="right",
                    va="bottom",
                    fontsize=7,
                    color="C3",
                )

                # Add title inside plot, upper right
                param_label = param.replace("_", " ")
                ax.text(
                    0.97,
                    0.95,
                    param_label,
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=10,
                )

                # Axis formatting
                ax.set_xlim(0.5, 4.5)
                ax.set_ylim(0, 2 * trial_hw)
                ax.set_xticks([1, 2, 3, 4])
                ax.set_xticklabels(x_labels)

                # Only show x-axis labels on bottom row
                if row_idx == 1:
                    ax.set_xlabel("Sector", fontsize=9)

                # Only show y-axis label on leftmost column
                if col_idx == 0:
                    ax.set_ylabel("σ, mm", fontsize=9)

        # Despine all axes
        self.mpl_axes.despine(axes.flatten())

        fig.suptitle(
            "Variance Evolution with Assembly Progress",
            fontsize=11,
            fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        return fig, axes


if __name__ == "__main__":
    # Sectors 5, 6, 7 are adjacent in the pit
    # Sector 8 contains coils 4 and 11 (coil 11 is adjacent to sector 7)
    sectors = {
        6: [12, 13],
        7: [8, 9],
        5: [16, 5],
        8: [4, 11],
    }

    # Load pit data
    pit = FiducialPit(
        sectors=sectors,
        phase="latest",
        pcr=True,
        private=False,
    )

    # Print summary
    pit.print_summary()

    # Position statistics using the phase configured at init
    print("\n" + "=" * 60)
    print("Position Statistics")
    print("=" * 60)
    stats_df, (position_fig, position_axes) = pit.position_statistics()
    print(stats_df.to_string())

    # Show evolution chart
    evolution_fig, evolution_axes = pit.plot_position_evolution()

    # Plot gaps
    pit.plot_gaps()

    # Plot statistics
    pit.plot_statistics()

    # compare gaps
    # Extract gap profile for S7 (coils 8-9) at inner radius
    profile = pit.gap_profile(8, 9)

    # Create measurement DataFrame
    measurements = pandas.DataFrame(
        {
            "z": [4469, 3575, 2681, 1788, 0, -1788, -2681, -3575, -4465],
            "gap": [0.72, 0.78, 0.75, 0.64, 0.45, 0.48, 0.55, 0.86, 0.9],
        }
    )

    # Plot with measurements overlay
    fig, ax = pit.plot_gap_profile(8, 9, measurements=measurements)

    plt.show()
