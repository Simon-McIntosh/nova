"""Manage TFC fiducial data for coil and sector allignment."""

from dataclasses import dataclass, field
from typing import ClassVar
from warnings import warn

import altair as alt
import itertools
import numpy as np
import pandas
from scipy.spatial.transform import Rotation

from nova.assembly.fiducialccl import Fiducial, FiducialRE, FiducialIDM
from nova.assembly.fiducialilis import FiducialIlis
from nova.assembly.ilisnominal import NominalIlis
from nova.assembly.sectordata import SectorData
from nova.assembly.transform import Rotate

alt.renderers.enable("html")


@dataclass
class FiducialSector(Fiducial):
    """Manage Reverse Engineering fiducial data."""

    phase: str = "FAT supplier"
    sector: dict[int, int] = field(init=False, repr=False, default_factory=dict)
    sectors: dict[int, list] | list[int] = field(
        init=True, repr=False, default_factory=lambda: dict.fromkeys(range(1, 10), [])
    )
    version: str = "latest"
    private: bool = False
    augment: bool = True
    fiducial_target: dict[str, pandas.DataFrame] | dict = field(
        init=False, repr=False, default_factory=dict
    )
    variance: dict[str, pandas.DataFrame] | dict = field(
        init=False, repr=False, default_factory=dict
    )
    ilis: pandas.DataFrame = field(init=False, repr=False, default_factory=dict)

    sheets: ClassVar[dict[str, str]] = {
        "FATsup": "FAT supplier",
        "SSAT": "SSAT BR",
        "FAT": "FAT supplier",
    }

    def __post_init__(self):
        """Propogate origin."""
        self._set_phase()
        super().__post_init__()
        self.source = "Reverse Engineering IDM datasets (xls workbooks)"
        self.origin = [self.origin[coil - 1] for coil in self.delta]
        self._load_fiducial_targets()
        self._load_variance()
        self._load_case()
        self._load_ilis()
        if self.augment:
            self.augment_ilis()

    def _set_phase(self):
        """Expand short string phase label or resolve 'latest' to actual phase.

        For 'latest', finds the newest phase that contains valid ILIS data.
        Priority: TFGS phases > In-pit target > last sheet in workbook.
        """
        self.phase = self.sheets.get(self.phase, self.phase)
        if self.phase == "latest":
            # Resolve 'latest' by getting phases from the first sector
            first_sector = next(iter(self.sectors.keys()))
            data = SectorData(
                first_sector,
                self.sectors[first_sector],
                private=self.private,
                version=self.version,
            )
            # Try in-pit phases first (preferred for installed coils)
            # Include various TFGS sheet naming conventions
            preferred_phases = [
                "AFTER TFGS landing",
                "TFGS Landing",
                "TFGS landing",
                "In-pit target",
            ]
            for phase in preferred_phases:
                if phase in data.phase:
                    self.phase = phase
                    return
            # Fallback to last sheet in workbook order
            if data.phase:
                self.phase = data.phase[-1]
            else:
                self.phase = "In-pit target"  # Ultimate fallback

    def _load_deltas(self):
        """Implement load deltas abstractmethod."""
        columns = ["dx", "dy", "dz"]
        for sector, coil in self.sectors.items():
            data = SectorData(sector, coil, private=self.private, version=self.version)
            print(data.filename)
            self.sectors[sector] = data.coil
            for coil, ccl in data.ccl[self.phase].items():
                self.sector[coil] = sector
                self.delta[coil] = ccl.loc[self.target, columns]

    def _load_fiducial_targets(self):
        """Load unique fiducial targets."""
        columns = ["x", "y", "z"]
        for sector, coil in self.sectors.items():
            data = SectorData(sector, coil, private=self.private, version=self.version)
            for coil, fiducial_target in data.data.items():
                nominal = fiducial_target["Nominal"]
                nominal.index = nominal.index.droplevel([0, 1])
                nominal.rename(index={"F'": "F"}, inplace=True)
                self.fiducial_target[coil] = nominal.loc[self.target, columns]

    def _load_variance(self):
        columns = ["ux", "uy", "uz"]
        for sector, coil in self.sectors.items():
            data = SectorData(sector, coil, private=self.private, version=self.version)
            for coil, ccl in data.ccl[self.phase].items():
                two_sigma = ccl.loc[self.target, columns]
                self.variance[coil] = (two_sigma / 2) ** 2
                self.variance[coil] = self.variance[coil].rename(
                    columns={col: f"s2{col[-1]}" for col in columns}
                )

    def _load_case(self):
        """Load case fiducials."""
        columns = ["x", "y", "z"]
        self.case = {}
        for sector, coil in self.sectors.items():
            data = SectorData(sector, coil, private=self.private, version=self.version)
            for coil, fiducial in data.data.items():
                self.case[coil] = (
                    fiducial[self.phase].xs("Fiducial", level=1).loc[:, columns]
                )

    @staticmethod
    def _extract_ilis(points, ilis, ro, count):
        if points.empty:
            return pandas.DataFrame()

        points.loc[:, "coil"] = points.index.get_level_values(0)
        points = points.droplevel(0)
        points.loc[:, "r"] = np.linalg.norm(points.loc[:, ["x", "y"]], axis=1)
        points.loc[:, "phi"] = np.arctan2(points.y, points.x)
        points.loc[:, "phi"] -= points.loc[:, "phi"].mean()
        points.loc[:, "ro_phi"] = ro * points.loc[:, "phi"]

        # identifiy dataset
        points.loc[:, "id"] = next(count)

        """
        # identify outliers  #  # MinCovDet(random_state=2025)
        from sklearn.neighbors import LocalOutlierFactor

        clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
        points.loc[:, "outlier_factor"] = clf.fit_predict(
            points.loc[:, ["x", "y", "z"]]
        )

        cov = sklearn.covariance.MinCovDet().fit(
            points.loc[:, ["x", "y", "z"]]
        )
        points.loc[:, "mahalanobis"] = cov.mahalanobis(points.loc[:, ["x", "y", "z"]])
        """
        points.loc[:, "feature"] = f"ILIS {ilis}"
        points.reset_index(inplace=True)

        return points

    def _load_ilis(self):
        columns = ["x", "y", "z"]
        ilis = {}
        for sector, coil in self.sectors.items():
            data = SectorData(sector, coil, private=self.private, version=self.version)
            for coil, fiducial in data.data.items():
                ilis[coil] = 2 * [[]]
                for i, key in enumerate(["ILIS +1 side", "ILIS -1 side"]):
                    try:
                        ilis[coil][i] = (
                            fiducial[self.phase].xs(key, level=1).loc[:, columns]
                        )
                    except KeyError:
                        warn(f"coil {coil} {key} not found in sector {sector}")
                        ilis[coil][i] = pandas.DataFrame()
                        pass

        ro = 2600
        count = itertools.count(0)
        self.ilis = pandas.concat(
            [
                pandas.concat(
                    [
                        self._extract_ilis(p, i, ro, count)
                        for p, i in zip(points, ["+1", "-1"])
                    ]
                )
                for points in ilis.values()
            ]
        )

    def _clock_coil(self, data: np.ndarray, coil: int) -> np.ndarray:
        """Clock coil data to sector midplane frame.

        First coil in sector: anticlock (+10°)
        Second coil in sector: clock (-10°)
        """
        rotate = Rotate()
        all_coils = list(self.delta.keys())
        if all_coils.index(coil) == 0:
            return rotate.anticlock(data)
        return rotate.clock(data)

    def _unclock_coil(self, data: np.ndarray, coil: int) -> np.ndarray:
        """Unclock coil data from sector midplane frame to coil-local."""
        rotate = Rotate()
        all_coils = list(self.delta.keys())
        if all_coils.index(coil) == 0:
            return rotate.clock(data)  # reverse of anticlock
        return rotate.anticlock(data)  # reverse of clock

    def augment_ilis(
        self,
        reference_phase: str = "In-pit target",
        reference_version: str | None = None,
    ):
        """Augment partial ILIS data using hybrid CCL + ILIS rigid body recovery.

        All computations are performed in the sector midplane frame
        (coils clocked to common frame) to ensure intra-sector gap
        invariance. The pattern follows fiducialfit.write_rigid_body:
        clock → transform → unclock.

        Parameters
        ----------
        reference_phase : str
            Phase with complete ILIS data (default: "In-pit target")
        reference_version : str | None
            Workbook version for reference data. If None, uses the version
            that has the reference phase with full ILIS data.
        """
        # Identify which ILIS surfaces are present
        expected_features = {"ILIS +1", "ILIS -1"}
        all_coils = list(self.delta.keys())
        present = {}
        for coil in all_coils:
            coil_data = self.ilis[self.ilis.coil == coil] if len(self.ilis) else None
            if coil_data is not None and len(coil_data) > 0:
                present[coil] = set(coil_data.feature.unique())
            else:
                present[coil] = set()

        missing = {coil: expected_features - feats for coil, feats in present.items()}
        has_missing = any(missing.values())

        # Find a version with complete reference ILIS data
        ref_version = reference_version or self._find_reference_version(reference_phase)
        if ref_version is None and reference_phase != self.phase:
            # Fall back to current phase as reference
            warn(
                f"No workbook version found with phase '{reference_phase}' "
                f"for sector {next(iter(self.sectors.keys()))}. "
                f"Falling back to current phase '{self.phase}'."
            )
            reference_phase = self.phase
            ref_version = self._find_reference_version(reference_phase)
        if ref_version is None:
            warn(
                f"No workbook version found with phase '{reference_phase}' "
                f"for sector {next(iter(self.sectors.keys()))}. "
                "Skipping ILIS augmentation."
            )
            return
        sector = next(iter(self.sectors.keys()))
        ref = FiducialSector(
            phase=reference_phase,
            sectors={sector: all_coils},
            version=ref_version,
            private=self.private,
            augment=False,
        )

        # Verify reference has complete ILIS (required for augmentation)
        if has_missing:
            for coil in all_coils:
                ref_coil = ref.ilis[ref.ilis.coil == coil]
                ref_feats = set(ref_coil.feature.unique()) if len(ref_coil) else set()
                if ref_feats != expected_features:
                    warn(
                        f"Reference phase '{reference_phase}' v{ref_version} "
                        f"missing ILIS for coil {coil}: "
                        f"{expected_features - ref_feats}"
                    )

        # Step 1: Sector-level CCL Kabsch in clocked sector frame
        R_ccl, t_ccl = self._compute_ccl_kabsch(ref)

        # Step 2: ILIS correction from measured surfaces (in clocked frame)
        measured_surfaces = []
        for coil, feats in present.items():
            for feat in feats:
                measured_surfaces.append((coil, feat))

        delta_R, delta_t = self._compute_ilis_correction(
            ref, R_ccl, t_ccl, measured_surfaces
        )

        # Step 3: Augmented transform (sector frame)
        R_aug = delta_R @ R_ccl
        t_aug = t_ccl + delta_t

        # Store transform components for downstream access
        self._transform = {
            "R_ccl": R_ccl,
            "t_ccl": t_ccl,
            "delta_R": delta_R,
            "delta_t": delta_t,
            "R_aug": R_aug,
            "t_aug": t_aug,
        }
        self._reference_phase = reference_phase

        # Report Euler angle comparison (roll=Rx, pitch=Ry, yaw=Rz)
        euler_ccl = Rotation.from_matrix(R_ccl).as_euler("XYZ") * 1e6
        euler_aug = Rotation.from_matrix(R_aug).as_euler("XYZ") * 1e6
        euler_delta = Rotation.from_matrix(delta_R).as_euler("XYZ") * 1e6
        print(
            f"Euler angles (roll, pitch, yaw) [urad]:\n"
            f"  CCL Kabsch:   ({euler_ccl[0]:.1f}, "
            f"{euler_ccl[1]:.1f}, {euler_ccl[2]:.1f})\n"
            f"  ILIS delta:   ({euler_delta[0]:.1f}, "
            f"{euler_delta[1]:.1f}, {euler_delta[2]:.1f})\n"
            f"  Corrected:    ({euler_aug[0]:.1f}, "
            f"{euler_aug[1]:.1f}, {euler_aug[2]:.1f})"
        )

        if not has_missing:
            return

        # Step 4: Correct CCL deltas with ILIS-constrained rigid body
        # Apply incremental correction (delta_R, delta_t) to measured
        # CCL positions about their centroid, preserving per-fiducial
        # residuals while updating the rigid body component.
        cur_pts_clocked = []
        for coil in all_coils:
            for target in self.delta[coil].index:
                cur_nominal = self.fiducial_target[coil].loc[target, ["x", "y", "z"]]
                cur_delta = self.delta[coil].loc[target, ["dx", "dy", "dz"]]
                cur_pos = cur_nominal.values + cur_delta.values
                cur_pts_clocked.append(
                    self._clock_coil(cur_pos.reshape(1, 3), coil).ravel()
                )
        cur_centroid = np.mean(cur_pts_clocked, axis=0)

        for coil in all_coils:
            for target in self.delta[coil].index:
                cur_nominal = self.fiducial_target[coil].loc[target, ["x", "y", "z"]]
                cur_delta = self.delta[coil].loc[target, ["dx", "dy", "dz"]]
                cur_pos = cur_nominal.values + cur_delta.values

                clocked = self._clock_coil(cur_pos.reshape(1, 3), coil).ravel()
                corrected = delta_R @ (clocked - cur_centroid) + cur_centroid + delta_t
                unclocked = self._unclock_coil(corrected.reshape(1, 3), coil).ravel()

                self.delta[coil].loc[target, ["dx", "dy", "dz"]] = (
                    unclocked - cur_nominal.values
                )

        # Step 5: clock → transform → unclock reference ILIS point clouds
        augmented_frames = []
        self._augmented_surfaces: set[tuple] = set()
        if isinstance(self.ilis, pandas.DataFrame) and len(self.ilis) > 0:
            count_start = int(self.ilis.id.max()) + 1
        else:
            count_start = 0
        count = itertools.count(count_start)

        for coil in all_coils:
            for feat in sorted(missing[coil]):
                ref_mask = (ref.ilis.coil == coil) & (ref.ilis.feature == feat)
                ref_points = ref.ilis.loc[ref_mask].copy()
                if ref_points.empty:
                    continue

                # clock → transform → unclock
                xyz_ref = ref_points[["x", "y", "z"]].values
                clocked = self._clock_coil(xyz_ref, coil)
                transformed = (R_aug @ clocked.T).T + t_aug
                xyz_aug = self._unclock_coil(transformed, coil)

                ref_points.loc[:, ["x", "y", "z"]] = xyz_aug
                ref_points.loc[:, "r"] = np.linalg.norm(
                    ref_points[["x", "y"]].values, axis=1
                )
                ref_points.loc[:, "phi"] = np.arctan2(
                    ref_points.y.values, ref_points.x.values
                )
                ref_points.loc[:, "phi"] -= ref_points.phi.mean()
                ref_points.loc[:, "ro_phi"] = 2600 * ref_points.phi.values
                ref_points.loc[:, "id"] = next(count)

                augmented_frames.append(ref_points)
                self._augmented_surfaces.add((coil, feat))

                print(
                    f"  Augmented coil {coil} {feat}: "
                    f"{len(ref_points)} points from reference"
                )

        if augmented_frames:
            if isinstance(self.ilis, pandas.DataFrame) and len(self.ilis) > 0:
                self.ilis = pandas.concat(
                    [self.ilis] + augmented_frames, ignore_index=True
                )
            else:
                self.ilis = pandas.concat(augmented_frames, ignore_index=True)

        n_measured = sum(len(f) for f in present.values())
        n_missing = sum(len(f) for f in missing.values())
        print(
            f"ILIS augmentation: {n_measured} measured, "
            f"{n_missing} recovered from reference"
        )

    def _find_reference_version(self, reference_phase: str) -> str | None:
        """Find the latest workbook version containing the reference phase."""
        from nova.assembly.sectorfile import SectorFile
        from packaging.version import Version

        sector = next(iter(self.sectors.keys()))
        sf = SectorFile(sector=sector, private=self.private)

        # Try versions from newest to oldest, skipping current
        current = Version(self.version) if self.version != "latest" else None
        parsed_versions = [Version(v) for v in sf.versions]

        for ver in sorted(parsed_versions, reverse=True):
            if current and ver == current:
                continue
            ver_str = str(ver)
            try:
                test = SectorData(
                    sector,
                    list(self.sectors[sector]),
                    version=ver_str,
                    private=self.private,
                )
                if reference_phase in test.phase:
                    # Check ILIS completeness
                    test_fs = FiducialSector(
                        phase=reference_phase,
                        sectors={sector: list(self.sectors[sector])},
                        version=ver_str,
                        private=self.private,
                        augment=False,
                    )
                    all_have_ilis = all(
                        len(test_fs.ilis[test_fs.ilis.coil == c]) > 0
                        for c in self.sectors[sector]
                    )
                    if all_have_ilis:
                        print(f"Using reference version {ver_str}")
                        return ver_str
            except (FileNotFoundError, KeyError):
                continue

        # Also check current version
        try:
            sector = next(iter(self.sectors.keys()))
            test = SectorData(
                sector,
                list(self.sectors[sector]),
                version=self.version,
                private=self.private,
            )
            if reference_phase in test.phase:
                return self.version
        except (FileNotFoundError, KeyError):
            pass

        return None

    def _compute_ccl_kabsch(
        self, ref: "FiducialSector"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute sector-level SVD/Kabsch rigid body from CCL fiducials.

        All CCL positions are clocked to sector midplane frame before
        computing the Kabsch alignment, following the pattern from
        fiducialfit.write_rigid_body.

        Returns
        -------
        R : ndarray (3, 3)
            Rotation matrix (sector frame) from reference to current
        t : ndarray (3,)
            Translation vector (sector frame) from reference to current
        """
        ref_pts = []
        cur_pts = []
        for coil in self.delta:
            if coil not in ref.delta:
                continue
            targets = list(self.delta[coil].index)
            for target in targets:
                if target not in ref.delta[coil].index:
                    continue

                ref_nominal = ref.fiducial_target[coil].loc[target, ["x", "y", "z"]]
                ref_delta = ref.delta[coil].loc[target, ["dx", "dy", "dz"]]
                ref_abs = ref_nominal.values + ref_delta.values

                cur_nominal = self.fiducial_target[coil].loc[target, ["x", "y", "z"]]
                cur_delta = self.delta[coil].loc[target, ["dx", "dy", "dz"]]
                cur_abs = cur_nominal.values + cur_delta.values

                # Clock to sector midplane frame
                ref_pts.append(self._clock_coil(ref_abs, coil))
                cur_pts.append(self._clock_coil(cur_abs, coil))

        ref_pts = np.array(ref_pts, dtype=float)
        cur_pts = np.array(cur_pts, dtype=float)

        # Kabsch algorithm
        ref_c = ref_pts.mean(axis=0)
        cur_c = cur_pts.mean(axis=0)

        H = (ref_pts - ref_c).T @ (cur_pts - cur_c)
        U, _, Vt = np.linalg.svd(H)

        d = np.linalg.det(Vt.T @ U.T)
        S = np.diag([1, 1, np.sign(d)])
        R = Vt.T @ S @ U.T

        t = cur_c - R @ ref_c

        # Report quality
        transformed = (R @ ref_pts.T).T + t
        residuals = cur_pts - transformed
        rms = np.sqrt(np.mean(residuals**2))
        euler = Rotation.from_matrix(R).as_euler("XYZ", degrees=False)
        print(
            f"CCL Kabsch: RMS={rms:.3f} mm, "
            f"t=({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}) mm, "
            f"euler=({euler[0] * 1e6:.1f}, {euler[1] * 1e6:.1f}, "
            f"{euler[2] * 1e6:.1f}) urad"
        )

        return R, t

    def _compute_ilis_correction(
        self,
        ref: "FiducialSector",
        R_ccl: np.ndarray,
        t_ccl: np.ndarray,
        measured_surfaces: list[tuple],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute angular and offset corrections from measured ILIS surfaces.

        All normals and positions are clocked to sector midplane frame
        before comparison, consistent with the sector-frame Kabsch.

        Parameters
        ----------
        ref : FiducialSector
            Reference phase data with complete ILIS
        R_ccl : ndarray (3, 3)
            CCL Kabsch rotation (sector frame)
        t_ccl : ndarray (3,)
            CCL Kabsch translation (sector frame)
        measured_surfaces : list of (coil, feature) tuples
            Surfaces with measured ILIS data

        Returns
        -------
        delta_R : ndarray (3, 3)
            Residual rotation correction (sector frame)
        delta_t : ndarray (3,)
            Residual translation correction (sector frame)
        """
        if not measured_surfaces:
            return np.eye(3), np.zeros(3)

        # Fit planes to measured and reference ILIS data (coil-local frames)
        measured_ilis = FiducialIlis(self.ilis, pcr=False)
        ref_ilis = FiducialIlis(ref.ilis, pcr=False)

        delta_angles = []
        delta_offsets = []

        for coil, feature in measured_surfaces:
            try:
                meas_plane = measured_ilis.planes.loc[(coil, feature)]
                ref_plane = ref_ilis.planes.loc[(coil, feature)]
            except KeyError:
                continue

            # Clock reference normal and point to sector frame
            n_ref = self._clock_coil(
                ref_plane[["nx", "ny", "nz"]].values.astype(float).reshape(1, 3),
                coil,
            ).ravel()
            n_predicted = R_ccl @ n_ref
            n_predicted /= np.linalg.norm(n_predicted)

            # Clock measured normal to sector frame
            n_measured = self._clock_coil(
                meas_plane[["nx", "ny", "nz"]].values.astype(float).reshape(1, 3),
                coil,
            ).ravel()
            n_measured /= np.linalg.norm(n_measured)

            # Rotation from predicted to measured normal
            cross = np.cross(n_predicted, n_measured)
            sin_angle = np.linalg.norm(cross)
            cos_angle = np.dot(n_predicted, n_measured)

            if sin_angle > 1e-10:
                axis = cross / sin_angle
                angle = np.arctan2(sin_angle, cos_angle)
                delta_angles.append((axis, angle))

            # Clock reference and measured positions to sector frame
            p_ref = self._clock_coil(
                ref_plane[["x", "y", "z"]].values.astype(float).reshape(1, 3),
                coil,
            ).ravel()
            p_predicted = R_ccl @ p_ref + t_ccl

            p_measured = self._clock_coil(
                meas_plane[["x", "y", "z"]].values.astype(float).reshape(1, 3),
                coil,
            ).ravel()

            # Normal-direction offset
            offset = np.dot(p_measured - p_predicted, n_measured)
            delta_offsets.append(offset * n_measured)

        if delta_angles:
            avg_rotvec = np.zeros(3)
            for axis, angle in delta_angles:
                avg_rotvec += axis * angle
            avg_rotvec /= len(delta_angles)
            delta_R = Rotation.from_rotvec(avg_rotvec).as_matrix()

            angle_mag = np.linalg.norm(avg_rotvec)
            print(
                f"ILIS correction: {angle_mag * 1e6:.1f} urad angular, "
                f"{len(delta_angles)} surface(s)"
            )
        else:
            delta_R = np.eye(3)

        if delta_offsets:
            delta_t = np.mean(delta_offsets, axis=0)
            print(f"ILIS offset correction: {np.linalg.norm(delta_t):.3f} mm")
        else:
            delta_t = np.zeros(3)

        return delta_R, delta_t

    def compare(self, source="RE"):
        """Compare fiducial sector data with previous RE dataset."""
        match source:
            case "IDM":
                previous = FiducialIDM()
            case "RE":
                previous = FiducialRE()
            case _:
                raise ValueError(f"source {source} not in [RE, IDM]")

        for coil, ccl in self.delta.items():
            if coil not in previous.delta:
                continue
            _ccl = previous.delta[coil]
            if source == "RE":
                _ccl = _ccl = previous.delta[coil].xs("FAT", 1)
            change = ccl.loc[:, ["dx", "dy", "dz"]] - _ccl
            if not np.allclose(np.array(change, float), 0):
                print(f"\ncoil #{coil}")
                print(change)

    def ccl_deviation(self) -> pandas.DataFrame:
        """Return measurement-datum deviation masked to constrained directions.

        Uses FiducialFit.fiducial_index to blank unconstrained cells:
            - radial (dx): F, D, E
            - toroidal (dy): A, F, D, E
            - vertical (dz): C, D, F, E
        """
        from nova.assembly.fiducialfit import FiducialFit

        targets = list(self.target)
        frames = []
        for coil, delta in self.delta.items():
            frame = delta[["dx", "dy", "dz"]].copy()
            for axis, key in zip("xyz", FiducialFit.fiducial_index):
                constrained = [targets[i] for i in FiducialFit.fiducial_index[key]]
                mask = frame.index.difference(constrained)
                frame.loc[mask, f"d{axis}"] = np.nan
            frame.columns = pandas.MultiIndex.from_product(
                [[f"Coil {coil}"], frame.columns]
            )
            frames.append(frame)
        return pandas.concat(frames, axis=1)

    def extract_coil_positions(self, pcr: bool = True) -> pandas.DataFrame:
        """Extract coil position parameters for trial.py Monte Carlo simulations.

        Projects inboard CCL fiducials (A, B, H) onto the ILIS midplane, then
        extracts position and orientation parameters from the projected points.

        Sector module coordinates:
        - x: radial (positive outward)
        - y: tangential (positive toward increasing phi)
        - z: vertical (positive upward)

        Parameters extracted (all in mm):
        - radial: Radial displacement from projected CCL fiducial H (midplane)
        - tangential: Tangential displacement from projected CCL fiducial H (midplane)
        - vertical: RMS vertical displacement from outer CCL fiducials (C, D, E, F)
          Matches FiducialFit.fiducial_index["vertical"] constraint points
        - roll_length: Roll-induced tangential offset at A-B span (~7.4m)
          (from dy gradient along z, i.e., A→B tangential difference)
        - yaw_length: Yaw-induced tangential offset at H-G span (~8.0m)
          (from dy gradient along x, i.e., H→G tangential difference)
        - pitch_length: Pitch-induced radial offset at A-B span (~7.4m)
          (from dx gradient along z, i.e., A→B radial difference)

        These values can be compared directly with trial.py alignment tolerances:
        - radial/tangential: Uniform(±1.5mm)
        - roll_length/yaw_length: Uniform(±3mm)

        Parameters
        ----------
        pcr : bool
            Apply PCR deviation corrections to ILIS planes

        Returns
        -------
        pandas.DataFrame
            Position parameters indexed by coil number
        """
        from nova.assembly.fiducialdata import FiducialData

        if self.ilis.empty:
            raise ValueError("No ILIS data available for position extraction")

        # Get nominal fiducial positions
        fiducial_nominal = FiducialData.fiducials()

        # Reference lengths based on actual CCL fiducial spans
        # Used to convert rotation angles to equivalent displacements (mm)
        # for comparison with trial.py alignment tolerances
        z_a = fiducial_nominal.loc["A", "z"]
        z_b = fiducial_nominal.loc["B", "z"]
        x_h = fiducial_nominal.loc["H", "x"]
        x_g = fiducial_nominal.loc["G", "x"]

        length = {
            "pitch": abs(z_b - z_a) / 1e3,  # A-B vertical span (m) ~7.4m
            "roll": abs(z_b - z_a) / 1e3,  # A-B vertical span (m) ~7.4m
            "yaw": abs(x_g - x_h) / 1e3,  # H-G radial span (m) ~8.0m
        }

        # Create FiducialIlis instance for plane fitting and projection
        ilis = FiducialIlis(self.ilis, pcr=pcr)

        # Build absolute CCL positions (nominal + delta) for projection
        inboard_targets = ["A", "B", "H"]

        results = []
        sector_coils = list(self.delta.keys())

        for coil in sector_coils:
            ccl_delta = self.delta[coil]
            ccl_nominal = self.fiducial_target.get(coil, fiducial_nominal)

            # Build absolute positions for inboard CCL targets
            ccl_abs = pandas.DataFrame(
                index=inboard_targets,
                columns=["x", "y", "z"],
                dtype=float,
            )
            for target in inboard_targets:
                if target in ccl_delta.index:
                    ccl_abs.loc[target] = (
                        ccl_nominal.loc[target, ["x", "y", "z"]]
                        + ccl_delta.loc[target, ["dx", "dy", "dz"]].values
                    )

            # Add coil column for projection groupby
            ccl_abs["coil"] = coil

            # Project inboard CCL points onto ILIS midplane
            projected = ilis.project(ccl_abs, plane="ILIS 0")

            # Calculate deltas from nominal (projected - nominal)
            delta_projected = pandas.DataFrame(
                index=inboard_targets,
                columns=["dx", "dy", "dz"],
                dtype=float,
            )
            for target in inboard_targets:
                delta_projected.loc[target] = (
                    projected.loc[target].values - ccl_nominal.loc[target].values
                )

            # Get outboard CCL delta (not projected, used for yaw)
            delta_g = ccl_delta.loc["G"] if "G" in ccl_delta.index else None

            # --- Extract position parameters ---
            # Radial: dx at H (projected)
            radial = delta_projected.loc["H", "dx"]

            # Tangential: dy at H (projected onto ILIS midplane)
            tangential = delta_projected.loc["H", "dy"]

            # Vertical: RMS of dz from outer fiducials (C, D, E, F)
            # Matches FiducialFit.fiducial_index["vertical"] = [2, 1, -1, -2]
            vertical_targets = ["C", "D", "E", "F"]
            vertical_deltas = []
            for target in vertical_targets:
                if target in ccl_delta.index:
                    vertical_deltas.append(ccl_delta.loc[target, "dz"])
            if vertical_deltas:
                vertical = np.sqrt(np.mean(np.array(vertical_deltas) ** 2))
                # Preserve sign: use mean if all same sign, else RMS magnitude
                if all(d >= 0 for d in vertical_deltas) or all(
                    d <= 0 for d in vertical_deltas
                ):
                    vertical = np.mean(vertical_deltas)
            else:
                vertical = np.nan

            # Roll: rotation about radial (x) axis
            # Measured from dy gradient along z (A→B tangential difference)
            # roll_angle = (dy_B - dy_A) / (z_B - z_A)
            z_a = fiducial_nominal.loc["A", "z"]
            z_b = fiducial_nominal.loc["B", "z"]
            dy_a = delta_projected.loc["A", "dy"]
            dy_b = delta_projected.loc["B", "dy"]
            roll_rad = (dy_b - dy_a) / (z_b - z_a)
            roll_length_val = roll_rad * length["roll"] * 1e3  # mm

            # Yaw: rotation about vertical (z) axis
            # Measured from dy gradient along x (H→G tangential difference)
            # yaw_angle = (dy_G - dy_H) / (x_G - x_H)
            if delta_g is not None:
                x_h = fiducial_nominal.loc["H", "x"]
                x_g = fiducial_nominal.loc["G", "x"]
                dy_h = delta_projected.loc["H", "dy"]
                dy_g = delta_g["dy"]
                yaw_rad = (dy_g - dy_h) / (x_g - x_h)
                yaw_length_val = yaw_rad * length["yaw"] * 1e3  # mm
            else:
                yaw_length_val = np.nan

            # Pitch: rotation about tangential (y) axis
            # Measured from dx gradient along z (A→B radial difference)
            # pitch_angle = (dx_B - dx_A) / (z_B - z_A)
            dx_a = delta_projected.loc["A", "dx"]
            dx_b = delta_projected.loc["B", "dx"]
            pitch_rad = (dx_b - dx_a) / (z_b - z_a)
            pitch_length_val = pitch_rad * length["pitch"] * 1e3  # mm

            is_first = sector_coils.index(coil) == 0

            results.append(
                {
                    "coil": coil,
                    "sector": self.sector.get(coil),
                    "is_first": is_first,
                    "radial": radial,
                    "tangential": tangential,
                    "vertical": vertical,
                    "roll_length": roll_length_val,
                    "yaw_length": yaw_length_val,
                    "pitch_length": pitch_length_val,
                }
            )

        return pandas.DataFrame(results).set_index("coil").sort_index()

    def gap_profile(
        self,
        radius: float | None = None,
        n_points: int = 100,
    ) -> pandas.DataFrame:
        """Extract intra-sector gap profile along z at a specific radius.

        Computes the gap between facing ILIS surfaces of adjacent coils
        in this sector, evaluated along a vertical line at fixed radius.

        Parameters
        ----------
        radius : float, optional
            Radius at which to extract profile. If None, uses inner ILIS radius.
        n_points : int
            Number of points along z.

        Returns
        -------
        pandas.DataFrame
            Gap profile with columns: z, gap, r
        """
        from scipy.interpolate import griddata

        coil_list = list(self.delta.keys())
        if len(coil_list) < 2:
            raise ValueError("Gap profile requires two coils in sector")

        coil_first, coil_second = coil_list[0], coil_list[1]
        feature_first, feature_second = "ILIS +1", "ILIS -1"
        rotate = Rotate()

        ilis = FiducialIlis(self.ilis, pcr=False)

        # Clock planes and data to sector midplane
        def clock_planes(planes, coil):
            planes = planes.copy()
            is_first = coil_list.index(coil) == 0
            transform = rotate.anticlock if is_first else rotate.clock
            planes.loc[:, ["x", "y", "z"]] = transform(
                planes.loc[:, ["x", "y", "z"]].values
            )
            planes.loc[:, ["nx", "ny", "nz"]] = transform(
                planes.loc[:, ["nx", "ny", "nz"]].values
            )
            return planes

        def clock_data(data, coil):
            data = data.copy()
            is_first = coil_list.index(coil) == 0
            transform = rotate.anticlock if is_first else rotate.clock
            data.loc[:, ["x", "y", "z"]] = transform(
                data.loc[:, ["x", "y", "z"]].values
            )
            return data

        sector_planes = ilis.planes.groupby(["coil"], group_keys=False).apply(
            lambda x: clock_planes(x, x.name)
        )

        sector_data = ilis.data.groupby(["coil"], group_keys=False).apply(
            lambda x: clock_data(x, x.name)
        )

        sector_index = [
            (coil_first, feature_first),
            (coil_second, feature_second),
        ]

        midplane = ilis.intersect(sector_planes.loc[sector_index])

        # Get data extents for z range
        data_first = ilis.data[
            (ilis.data.coil == coil_first) & (ilis.data.feature == feature_first)
        ]
        data_second = ilis.data[
            (ilis.data.coil == coil_second) & (ilis.data.feature == feature_second)
        ]

        r_min = max(data_first.r.min(), data_second.r.min())
        r_max = min(data_first.r.max(), data_second.r.max())
        z_min = max(data_first.z.min(), data_second.z.min())
        z_max = min(data_first.z.max(), data_second.z.max())

        r_margin = 0.05 * (r_max - r_min)
        z_margin = 0.05 * (z_max - z_min)

        if radius is None:
            radius = r_min + r_margin

        z_range = np.linspace(z_min + z_margin, z_max - z_margin, n_points)

        midplane_point = midplane.loc[["x", "y", "z"]].values
        midplane_normal = midplane.loc[["nx", "ny", "nz"]].values
        midplane_normal = midplane_normal / np.linalg.norm(midplane_normal)

        offset_profiles = []
        for coil, feature in sector_index:
            plane_mask = (sector_data.coil == coil) & (sector_data.feature == feature)
            plane_data = sector_data.loc[plane_mask]

            v = plane_data.loc[:, ["x", "y", "z"]].values - midplane_point
            point_offsets = np.dot(v, midplane_normal)

            profile_points = np.column_stack([np.full_like(z_range, radius), z_range])
            offset_profile = griddata(
                plane_data.loc[:, ["r", "z"]].values,
                point_offsets,
                profile_points,
                method="linear",
            )
            offset_profiles.append(offset_profile)

        gap_profile = offset_profiles[1] - offset_profiles[0]

        return pandas.DataFrame({"z": z_range, "gap": gap_profile, "r": radius})

    def plot_gap_profile(
        self,
        radius: float | None = None,
        n_points: int = 100,
        measurements: pandas.DataFrame | None = None,
    ):
        """Plot intra-sector gap profile along z at inner radius.

        Parameters
        ----------
        radius : float, optional
            Radius at which to extract profile. If None, uses inner ILIS radius.
        n_points : int
            Number of points along z.
        measurements : pandas.DataFrame, optional
            Measured gap data with columns 'z' and 'gap' to overlay.

        Returns
        -------
        tuple[Figure, Axes]
        """
        import matplotlib.pyplot as plt

        profile = self.gap_profile(radius=radius, n_points=n_points)
        coil_list = list(self.delta.keys())

        fig, ax = plt.subplots(figsize=(10, 8))
        valid = ~profile["gap"].isna()
        ax.plot(
            profile.loc[valid, "z"],
            profile.loc[valid, "gap"],
            "-",
            linewidth=2,
            label=f"Predicted (r={profile['r'].iloc[0]:.0f} mm)",
        )

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

        mean_gap = profile.loc[valid, "gap"].mean()
        ax.axhline(mean_gap, ls="--", color="gray", label=f"Mean: {mean_gap:.2f} mm")

        ax.set_xlabel("z (mm)")
        ax.set_ylabel("Gap (mm)")
        ax.set_title(f"Gap Profile: Coils {coil_list[0]}-{coil_list[1]}")
        ax.legend()
        fig.tight_layout()

        return fig, ax


if __name__ == "__main__":
    phase = "SSAT BR"
    phase = "SSAT target"
    # phase = "SSAT AR"
    # phase = "SSAT AL"
    # phase = "SSAT AR2"
    # phase = "SSAT AR target"
    # phase = "In-pit target"
    # phase = "TFGS Landing"

    sectors = {7: [8, 9]}
    # sectors = {6: [12, 13]}
    # sectors = {5: [16, 5]}
    sectors = {8: [4, 11]}
    # sectors = {4: [2, 3]}
    sectors = {1: [14, 15]}

    fiducial = FiducialSector(phase=phase, sectors=sectors, private=True)
    fiducial.compare("IDM")

    deviation = fiducial.ccl_deviation()
    print("\nCCL deviation (measurement - datum), constrained dirs only:")
    print(deviation.to_string())

    # Cylindrical deviation using FiducialFit's delta() method
    from nova.assembly.fiducialassess import FiducialAssess

    assess = FiducialAssess(
        phase=phase,
        sectors=sectors,
        fill=False,
        infer=True,
        ilis=True,
        ilis_pcr=True,
        method="rms",
        coupled=False,
        private=True,
    )
    assess.build()

    print("\nCylindrical deviation (dr, r*dphi, dz), constrained dirs only:")
    print(assess.deviation("fiducial").to_string())

    print("\nError summary (constrained fiducials):")
    print(assess.summary("fiducial").to_string())

    # Gap profile at inner radius
    if len(list(fiducial.delta.keys())) >= 2:
        profile = fiducial.gap_profile()
        print(f"\nMean gap: {profile.gap.mean():.3f} mm")
        fiducial.plot_gap_profile()

    ccl = pandas.concat(fiducial.delta).rename(
        {"dx": "x", "dy": "y", "dz": "z"}, axis=1
    ) + pandas.concat(fiducial.fiducial_target)
    ccl.loc[:, "r"] = np.linalg.norm(ccl.loc[:, ["x", "y"]], axis=1)
    ccl.loc[:, "phi"] = np.arctan2(ccl.y, ccl.x)
    ccl.loc[:, "ro_phi"] = 2600 * ccl.phi
    ccl.loc[:, "coil"] = ccl.index.get_level_values(0)
    ccl = ccl.droplevel(0)
    ccl.reset_index(inplace=True, names="Name")
    ccl = ccl.loc[ccl.Name.map(lambda i: i in ["A", "B", "H"])]
    ccl.loc[:, "feature"] = "CCL"

    # drop coils with no ilis
    ccl = ccl[ccl.coil.map(lambda x, coils=fiducial.ilis.coil.unique(): x in coils)]

    """
    ccl_a = ccl.copy()
    ccl_a.loc[:, "ilis"] = fiducial.ilis.type.iloc[-1]

    ccl_b = ccl.copy()
    ccl_b.loc[:, "ilis"] = fiducial.ilis.type.iloc[0]j

    ccl = pandas.concat([ccl_a, ccl_b], axis=0)
    """

    ilis = FiducialIlis(fiducial.ilis, pcr=False)
    nominal = NominalIlis()

    from nova.assembly.transform import Rotate
    from scipy.interpolate import griddata

    rotate = Rotate()

    def clock(plane, coil, cords=[("x", "y", "z")]):
        if list(sectors.values())[0].index(coil) == 0:
            transform = rotate.anticlock
        else:
            transform = rotate.clock

        for c in cords:
            plane.loc[:, c] = transform(plane.loc[:, c])
        return plane

    coil_list = list(sectors.values())[0]
    has_two_coils = len(coil_list) >= 2

    sector_planes = ilis.planes.groupby(["coil"], group_keys=False).apply(
        lambda x: clock(x, x.name, cords=[("x", "y", "z"), ("nx", "ny", "nz")])
    )

    sector_data = ilis.data.groupby(["coil"], group_keys=False).apply(
        lambda x: clock(x, x.name)
    )

    grid_r, grid_z = np.mgrid[
        slice(ilis.data.r.min(), ilis.data.r.max(), 20j),
        slice(ilis.data.z.min(), ilis.data.z.max(), 40j),
    ]

    if has_two_coils:
        # Two-coil gap analysis: intersect facing ILIS planes from adjacent coils
        sector_index = [
            (coil, plane) for coil, plane in zip(coil_list, ("ILIS +1", "ILIS -1"))
        ]

        midplane = ilis.intersect(sector_planes.loc[sector_index])

        offset = []
        offset_data_list = []
        grid_y_list = []

        for coil, feature in sector_index:
            plane_index = (sector_data.coil == coil) & (sector_data.feature == feature)
            plane_offset = ilis.offset(
                sector_data.loc[plane_index, ("x", "y", "z")], midplane
            )

            offset.append(
                griddata(
                    sector_data.loc[plane_index, ("r", "z")],
                    plane_offset,
                    (grid_r, grid_z),
                    method="linear",
                )
            )

            # Store offset data for plotting
            plane_data = sector_data.loc[plane_index, ["Name", "r", "z"]].copy()
            plane_data["offset"] = plane_offset
            plane_data["coil"] = coil
            plane_data["feature"] = feature
            offset_data_list.append(plane_data)

            plane = sector_planes.loc[(coil, feature)]
            print(plane.nx)
            grid_y_list.append(
                plane.y
                - (plane.nx * (grid_r - plane.x) + plane.nz * (grid_z - plane.z))
                / plane.ny
            )

        grid_y = grid_y_list[1] - grid_y_list[0]

        # Combine all offset data
        offset_df = pandas.concat(offset_data_list, ignore_index=True)

        # Create first and last plane dataframes
        first_plane = offset_df[offset_df["feature"] == sector_index[0][1]].copy()
        last_plane = offset_df[offset_df["feature"] == sector_index[-1][1]].copy()

        gap_plane = pandas.DataFrame(
            {
                "r": grid_r.flatten(),
                "z": grid_z.flatten(),
                "offset": offset[1].flatten() - offset[0].flatten(),
            }
        )
        gap_plane["coil"] = 0
        gap_plane["feature"] = "Gap"

        # Combine all data for plotting
        plot_data = pandas.concat(
            [first_plane, gap_plane, last_plane], ignore_index=True
        )
        facet_sort = ["ILIS +1", "Gap", "ILIS -1"]

    else:
        # Single-coil analysis: show both ILIS planes relative to coil's own midplane
        print("Single coil analysis: gap calculation skipped (requires 2 coils)")

        coil = coil_list[0]
        sector_index = [(coil, "ILIS +1"), (coil, "ILIS -1")]

        # Use coil's own midplane for offset calculations
        coil_midplane = ilis.planes.loc[(coil, "ILIS 0")]

        offset_data_list = []
        for _, feature in sector_index:
            plane_index = (sector_data.coil == coil) & (sector_data.feature == feature)
            plane_offset = ilis.offset(
                sector_data.loc[plane_index, ("x", "y", "z")], coil_midplane
            )
            plane_data = sector_data.loc[plane_index, ["Name", "r", "z"]].copy()
            plane_data["offset"] = plane_offset
            plane_data["coil"] = coil
            plane_data["feature"] = feature
            offset_data_list.append(plane_data)

        plot_data = pandas.concat(offset_data_list, ignore_index=True)
        facet_sort = ["ILIS +1", "ILIS -1"]

    # Create Altair chart
    sector_number = next(iter(sectors.keys()))
    offset_chart_title = f"Sector {sector_number} - {phase}"
    base = alt.Chart(plot_data).mark_circle(size=60)

    chart = (
        base.encode(
            x=alt.X("r:Q", title="Radius (r)", scale=alt.Scale(domain=[2000, 3200])),
            y=alt.Y("z:Q", title="Height (z)"),
            color=alt.Color(
                "offset:Q",
                title="Offset",
                scale=alt.Scale(scheme="redblue"),
            ),
            tooltip=[
                alt.Tooltip("coil:N", title="Coil"),
                alt.Tooltip("Name"),
                alt.Tooltip("r:Q", title="r", format=".2f"),
                alt.Tooltip("z:Q", title="z", format=".2f"),
                alt.Tooltip("offset:Q", title="Offset", format=".4f"),
            ],
            facet=alt.Facet(
                "feature:N",
                title=offset_chart_title,
                header=alt.Header(labelFontSize=12),
                sort=facet_sort,
            ),
        )
        .properties(width=250, height=300)
        .resolve_scale(color="independent")
        .configure_axis(grid=True)
        .interactive()
    )

    chart.show()

    # print(nominal.angle_to_xz(nominal.planes))

    print("***")

    print(nominal.analize_offsets(fiducial.ilis))

    print(nominal.analize_offsets(ilis.planes))

    print("***")

    data = pandas.merge(ilis.data, ccl, how="outer")

    # data.loc[:, 'ro_phi'] = data.x

    data.loc[data.feature == "CCL", "type"] = "original"
    ccl_points = data.loc[data.feature == "CCL", :].copy()

    ccl_index = data.feature == "CCL"

    data.loc[ccl_index, ["x", "y", "z"]] = ilis.project(data.loc[ccl_index, :])
    data.loc[ccl_index, "r"] = np.linalg.norm(data.loc[ccl_index, ["x", "y"]], axis=1)
    data.loc[ccl_index, "phi"] = np.arctan2(
        data.loc[ccl_index, "y"], data.loc[ccl_index, "x"]
    )
    data.loc[ccl_index, "ro_phi"] = 2600 * data.loc[ccl_index, "phi"]

    # data.loc[:, "phi"] -= data.loc[:, "phi"].mean()

    data.loc[data.feature == "CCL", "Name"] = data.loc[
        data.feature == "CCL", "Name"
    ].map(lambda name: f"{name}'")

    data.loc[data.feature == "CCL", "type"] = "projected"

    data = pandas.concat([data, ccl_points])

    # data.loc[:, "ro_phi"] = data.y

    # Calculate x-axis scale from clean (non-outlier) ILIS data
    ilis_mask = data.feature.isin(["ILIS +1", "ILIS -1"])
    clean_mask = ilis_mask & (~data.outlier.fillna(False))
    if clean_mask.sum() > 0:
        clean_ro_phi = data.loc[clean_mask, "ro_phi"]
        ro_phi_margin = (clean_ro_phi.max() - clean_ro_phi.min()) * 0.1
        ro_phi_domain = [
            clean_ro_phi.min() - ro_phi_margin,
            clean_ro_phi.max() + ro_phi_margin,
        ]
    else:
        ro_phi_domain = None  # Use automatic scaling

    base = alt.Chart(data, width=125, height=175)

    select = {
        "ILIS": alt.FieldOneOfPredicate(
            field="feature", oneOf=[f"ILIS {sign}1" for sign in ["+", "-"]]
        ),
        "ILIS_outlier": alt.datum.outlier,
        # "ILIS_outlier": alt.datum.outlier_factor < 0,
        "CCL": alt.datum.feature == "CCL",
    }

    # X-axis with scale based on clean data (outliers may appear clipped)
    x_scale = (
        alt.X("ro_phi", scale=alt.Scale(domain=ro_phi_domain, clamp=True))
        if ro_phi_domain
        else alt.X("ro_phi")
    )

    scatter = (
        base.mark_circle(size=60)
        .transform_filter(select["ILIS"])
        .encode(
            x=x_scale,
            y="z",
            color=alt.Color("r").title("radius").scale(scheme="blueorange"),
            tooltip=["Name", "offset"],
        )
    )

    fit = (
        base.mark_line()
        .transform_filter(select["ILIS"])
        .transform_filter(select["ILIS_outlier"])
        .transform_regression("z", "ro_phi", groupby=["coil", "ilis"])
        .mark_line(color="gray")
        .encode(x=x_scale, y="z")
    )

    outlier = (
        base.mark_circle(size=80, color="red", filled=False)
        .transform_filter(select["ILIS_outlier"])
        .encode(x=x_scale, y="z", tooltip=["Name", "offset"])
    )

    ccl_points = (
        base.mark_point(size=60, color="black")
        .transform_filter(select["CCL"])
        .encode(
            x=x_scale,
            y="z",
            tooltip=["Name", "ro_phi", "y"],
            shape=alt.Shape("type"),
            # color=alt.Color("transform").scale(scheme="set2"),
        )
    )

    ccl_text = (
        base.mark_text(align="center", baseline="middle", dy=12)
        .transform_filter(select["CCL"])
        .encode(text="Name", x=x_scale, y="z")
    )

    # row=alt.Row("coil:N"),
    # column=alt.Column("ilis:N").sort(["-1", "1"]),
    # color="r:Q",
    # tooltip=["Name", "r", "phi"],
    # ).show()

    # fit = base

    chart = scatter + outlier + ccl_points + ccl_text

    ilis_chart_title = f"Sector {sector_number} - {phase}"
    chart = (
        chart.facet(
            row="coil",
            column=alt.Column("feature", title=ilis_chart_title).sort(
                ["ILIS -1", "CCL", "ILIS +1"]
            ),
        )
        .configure_axis(grid=False)
        .interactive()
    )
    chart.resolve_scale(x="shared", y="shared", color="shared").show()
    """
    chart = chart.mark_circle(size=60).encode(
        x="ro_phi",
        y="z",
        # row=alt.Row("coil"),
        # column=alt.Column("ilis").sort(["-1", "1"]),
        color="r",
        tooltip=["Name", "r", "phi"],
    )
    # chart += chart.mark_circle(size=80, color="red").encode()

    chart = chart.transform_regression("ro_phi", "z").mark_line()

    chart = (
        chart.resolve_scale(x="shared", y="shared")
        .configure_axis(grid=False)
        .configure_view(stroke=None)
    )
    """

    """
    import vedo

    ilis = FiducialIlis(fiducial.ilis)

    points = data.loc[:, ['x', 'y', 'z', 'r', 'ro_phi']].copy()
    #points.loc[:, 'r'] /= 6000
    #points.loc[:, 'z'] /= 20
    points['offset'] = 0

    coil = 13
    plane = 'ILIS -1'

    index = (data.feature == plane) & (data.coil == coil)

    points.loc[index, 'offset'] = 500* ilis.offset(points.loc[index], (coil, plane))

    vedo.Points(points.loc[index, ['r', 'z', 'offset']]).generate_delaunay2d(
        tol=0.000001).c('green').show(axes=1).close()
    """
