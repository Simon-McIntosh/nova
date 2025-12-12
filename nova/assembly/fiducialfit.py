"""Manage fitting algorithums for TF coils and SSAT sectors."""

from dataclasses import dataclass, field
from functools import cached_property
from typing import ClassVar
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import xarray

from nova.assembly.fiducialdata import FiducialData
from nova.assembly.fiducialplotter import FiducialPlotter
from nova.assembly.sectordata import SectorData
from nova.assembly.transform import Rotate


@dataclass
class FiducialFit(FiducialData):
    """Extend FiducialData class to include fitting algorithms."""

    filename: str = "fiducial_fit"
    private: bool = False
    variance: float | str = "file"
    infer: bool = True
    method: str = "rms"
    samples: int = 10
    radial_offset: float = (33.04 - 36) / (2 * np.pi)
    data: xarray.Dataset = field(init=False, repr=False, default_factory=xarray.Dataset)

    weights: ClassVar[list[float]] = [1, 1, 0.25]
    fiducial_index: ClassVar[dict[str, list[int]]] = {
        "radial": [5, 3, 4],  # ILIS fiducials
        #"toroidal": slice(None),  # toroidal (all)
        "toroidal": [0, 5, 3, 4],  # reduced toroidal constraint set
        "vertical": [2, 1, -1, -2],  # vertical C, D and E, F
    }

    # --- Clocking infrastructure for sector fits ---

    @property
    def is_sector(self) -> bool:
        """True if fitting multiple coils as a sector."""
        return self.data.sizes["coil"] > 1

    @cached_property
    def _rotate(self) -> Rotate:
        """Return Rotate instance for sector clocking."""
        return Rotate()

    def _coil_index(self, coil) -> int:
        """Return index of coil in sector (0 = first, 1 = second)."""
        return list(self.data.coil.values).index(int(coil))

    def clock_coil(self, data: np.ndarray, coil) -> np.ndarray:
        """Clock data to sector frame. Identity for single-coil fits.
        
        First coil in sector: anticlock (-half_angle)
        Second coil in sector: clock (+half_angle)
        """
        if not self.is_sector:
            return data
        if self._coil_index(coil) == 0:
            return self._rotate.anticlock(data)
        return self._rotate.clock(data)

    def unclock_coil(self, data: np.ndarray, coil) -> np.ndarray:
        """Unclock data from sector frame. Identity for single-coil fits."""
        if not self.is_sector:
            return data
        if self._coil_index(coil) == 0:
            return self._rotate.clock(data)  # reverse of anticlock
        return self._rotate.anticlock(data)  # reverse of clock

    # --- End clocking infrastructure ---

    @property
    def fiducial_attrs(self):
        """Extend fiducial_attrs to include fit parameters."""
        return super().fiducial_attrs | {
            attr: getattr(self, attr)
            for attr in ["infer", "method", "samples", "radial_offset", "weights"]
        }

    def build(self):
        """Extend build to include fiducial fitting."""
        super().build()
        self.data = self.data.rename(dict(space="cartesian"))
        self.load_target()
        self.load_measurement()
        self.evaluate_gpr("fiducial", "gpr")
        self.fit()
        self.evaluate_gpr("fiducial_fit", "fit_gpr")

    def _sector_transform(self, opt_x: np.ndarray, points: np.ndarray, coil) -> np.ndarray:
        """Apply transform in sector frame: clock → transform → unclock.
        
        For single-coil fits, clocking is identity.
        For sector fits, transform was computed in clocked frame.
        """
        clocked = self.clock_coil(points, coil)
        transformed = self.transform(opt_x, clocked)
        return self.unclock_coil(transformed, coil)

    def write(self, sheet: str, opt_x=None):
        """Write fits to source xls files.
        
        Args:
            sheet: Target sheet name to write to.
            opt_x: Optional transform to apply. Uses fitted transform if None.
        
        Note: For derived phases (in-silico transforms), use write_rigid_body() instead.
        """
        ilis_source = self.dataset.ilis

        for sector in tqdm(self.data.sector.data, "updating xls workbooks"):
            sectordata = SectorData(sector)
            coils = self.data.coils.sel(sector=sector)

            if opt_x is None:
                opt_x = self.data.opt_x

            with sectordata.openbook(), sectordata.savebook():
                if sheet not in sectordata.book.sheetnames:
                    sectordata.book.create_sheet(sheet)
                last_sheet = sectordata.book.sheetnames[-2]
                worksheet = sectordata.book[sheet]

                workcell = {
                    "coil": sectordata._coil_index(last_sheet),
                    "fiducial": sectordata.locate("Fiducial", last_sheet),
                    "ilis_p": sectordata.locate("ILIS +1 side", last_sheet),
                    "ilis_m": sectordata.locate("ILIS -1 side", last_sheet),
                }

                for coil in coils.data:
                    cell_index = sectordata.coil.index(coil)
                    opt_x_coil = opt_x.sel(coil=coil)

                    std = (
                        self.data.fiducial_fit_gpr_std.sel(coil=coil)
                        .sortby("target")
                        .data
                    )
                    # write coil header
                    sectordata.write(
                        worksheet,
                        workcell["coil"][cell_index],
                        np.array(
                            [["Coil", "Point", "Name", "X", "Y", "Z", "uX", "uY", "uZ"]]
                        ),
                    )
                    sectordata.write(
                        worksheet,
                        workcell["coil"][cell_index],
                        np.array([[coil]]),
                        offset=(1, 0),
                    )
                    sectordata.write(
                        worksheet,
                        workcell["coil"][cell_index],
                        np.array([["CCL"]]),
                        offset=(1, 1),
                    )
                    sectordata.write(
                        worksheet,
                        workcell["coil"][cell_index],
                        sectordata.data[coil]["Nominal"]
                        .index.get_level_values("Name")
                        .values[:, np.newaxis],
                        # self.data.target.sortby("target").data[:, np.newaxis],
                        offset=(1, 2),
                    )
                    # write transformed ccl data (sector: clock → transform → unclock)
                    sectordata.write(
                        worksheet,
                        workcell["coil"][cell_index],
                        np.append(
                            self._sector_transform(
                                opt_x_coil.data,
                                self.data.fiducial.sel(coil=coil).sortby("target").data,
                                coil,
                            ),
                            2 * std,
                            axis=1,
                        ),
                        offset=(1, 3),
                    )

                    self._write_transform(
                        worksheet, workcell["coil"][cell_index], opt_x_coil
                    )
                    # write fiducial header
                    sectordata.write(
                        worksheet,
                        workcell["fiducial"][cell_index],
                        np.array([["Fiducial"]]),
                    )
                    sectordata.write(
                        worksheet,
                        workcell["fiducial"][cell_index],
                        self.dataset.case[coil]
                        .index.get_level_values("Name")
                        .values[:, np.newaxis],
                        offset=(0, 1),
                    )
                    # write transformed case fiducial data
                    sectordata.write(
                        worksheet,
                        workcell["fiducial"][cell_index],
                        self._sector_transform(
                            opt_x_coil.data,
                            self.dataset.case[coil].loc[:, ["x", "y", "z"]].values,
                            coil,
                        ),
                        offset=(0, 2),
                    )

                    # write ilis
                    for side, xls_index in zip(
                        ["+1", "-1"],
                        [
                            workcell["ilis_p"][cell_index],
                            workcell["ilis_m"][cell_index],
                        ],
                    ):
                        ilis_data = ilis_source.loc[
                            (ilis_source.coil == coil)
                            & (ilis_source.feature == f"ILIS {side}")
                        ]
                        sectordata.write(
                            worksheet,
                            xls_index,
                            np.array([[f"ILIS {side} side"]]),
                        )
                        sectordata.write(
                            worksheet,
                            xls_index,
                            ilis_data.Name.values[:, np.newaxis],
                            offset=(0, 1),
                        )
                        # write transformed ilis data (sector: clock → transform → unclock)
                        sectordata.write(
                            worksheet,
                            xls_index,
                            self._sector_transform(
                                opt_x_coil.data,
                                ilis_data.loc[:, ["x", "y", "z"]].values,
                                coil,
                            ),
                            offset=(0, 2),
                        )

    def write_rigid_body(self, target_sheet: str):
        """Write target phase as rigid body transform of fitted phase.
        
        For in-silico transforms where target = rigid_body(fitted_phase).
        Uses fitted phase ILIS/CCL data, applies rigid body transform.
        
        Args:
            target_sheet: Sheet name to write (e.g., 'In-pit target')
        """
        from nova.assembly.fiducialsector import FiducialSector
        
        # Load fitted phase data as reference
        ref_dataset = FiducialSector(
            phase=self.phase, sectors=self.sectors, private=self.private
        )
        
        # Compute rigid body transform from reference CCL to target CCL
        for sector in tqdm(self.data.sector.data, "writing rigid body transform"):
            sectordata = SectorData(sector, private=self.private)
            coils = self.data.coils.sel(sector=sector).data
            
            # Get CCL from both phases (clocked to sector frame)
            def get_clocked_ccl(phase):
                pts = []
                for i, coil in enumerate(coils):
                    df = sectordata.data[coil][phase]
                    ccl = df.xs('CCL', level=1)[['x', 'y', 'z']].values
                    pts.append(self.clock_coil(ccl, coil))
                return np.vstack(pts)
            
            ccl_ref = get_clocked_ccl(self.phase)
            ccl_tgt = get_clocked_ccl(target_sheet)
            
            # Fit rigid body transform
            src_c, tgt_c = ccl_ref.mean(0), ccl_tgt.mean(0)
            H = (ccl_ref - src_c).T @ (ccl_tgt - tgt_c)
            U, _, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[-1] *= -1
                R = Vt.T @ U.T
            t = tgt_c - R @ src_c
            
            # Check residuals
            transformed = (R @ ccl_ref.T).T + t
            rms = np.sqrt(np.mean((ccl_tgt - transformed)**2))
            print(f"  Rigid body RMS: {rms:.6f} mm, t={t}")
            
            # Now write: apply reference fit + rigid body to reference data
            with sectordata.openbook(), sectordata.savebook():
                if target_sheet not in sectordata.book.sheetnames:
                    sectordata.book.create_sheet(target_sheet)
                last_sheet = sectordata.book.sheetnames[-2]
                worksheet = sectordata.book[target_sheet]
                
                workcell = {
                    "coil": sectordata._coil_index(last_sheet),
                    "fiducial": sectordata.locate("Fiducial", last_sheet),
                    "ilis_p": sectordata.locate("ILIS +1 side", last_sheet),
                    "ilis_m": sectordata.locate("ILIS -1 side", last_sheet),
                }
                
                for coil in coils:
                    cell_index = sectordata.coil.index(coil)
                    opt_x_coil = self.data.opt_x.sel(coil=coil)
                    
                    # Rigid body only (reference data is already fitted)
                    def rigid_transform(pts, coil):
                        clocked = self.clock_coil(pts, coil)
                        rigid = (R @ np.asarray(clocked).T).T + t
                        return self.unclock_coil(rigid, coil)
                    
                    # Write CCL (reference data is already fitted, just apply rigid body)
                    ref_ccl = ref_dataset.delta[coil].values + ref_dataset.fiducial_target[coil].values
                    ccl_transformed = rigid_transform(ref_ccl, coil)
                    
                    std = self.data.fiducial_fit_gpr_std.sel(coil=coil).sortby("target").data
                    sectordata.write(
                        worksheet, workcell["coil"][cell_index],
                        np.array([["Coil", "Point", "Name", "X", "Y", "Z", "uX", "uY", "uZ"]]),
                    )
                    sectordata.write(worksheet, workcell["coil"][cell_index], np.array([[coil]]), offset=(1, 0))
                    sectordata.write(worksheet, workcell["coil"][cell_index], np.array([["CCL"]]), offset=(1, 1))
                    sectordata.write(
                        worksheet, workcell["coil"][cell_index],
                        sectordata.data[coil]["Nominal"].index.get_level_values("Name").values[:, np.newaxis],
                        offset=(1, 2),
                    )
                    sectordata.write(
                        worksheet, workcell["coil"][cell_index],
                        np.append(ccl_transformed, 2 * std, axis=1),
                        offset=(1, 3),
                    )
                    
                    self._write_transform(worksheet, workcell["coil"][cell_index], opt_x_coil)
                    
                    # Write case fiducials
                    sectordata.write(worksheet, workcell["fiducial"][cell_index], np.array([["Fiducial"]]))
                    sectordata.write(
                        worksheet, workcell["fiducial"][cell_index],
                        ref_dataset.case[coil].index.get_level_values("Name").values[:, np.newaxis],
                        offset=(0, 1),
                    )
                    case_transformed = rigid_transform(ref_dataset.case[coil].loc[:, ["x", "y", "z"]].values, coil)
                    sectordata.write(worksheet, workcell["fiducial"][cell_index], case_transformed, offset=(0, 2))
                    
                    # Write ILIS
                    for side, xls_index in zip(["+1", "-1"], [workcell["ilis_p"][cell_index], workcell["ilis_m"][cell_index]]):
                        ilis_data = ref_dataset.ilis.loc[
                            (ref_dataset.ilis.coil == coil) & (ref_dataset.ilis.feature == f"ILIS {side}")
                        ]
                        sectordata.write(worksheet, xls_index, np.array([[f"ILIS {side} side"]]))
                        sectordata.write(worksheet, xls_index, ilis_data.Name.values[:, np.newaxis], offset=(0, 1))
                        ilis_transformed = rigid_transform(ilis_data.loc[:, ["x", "y", "z"]].values, coil)
                        sectordata.write(worksheet, xls_index, ilis_transformed, offset=(0, 2))

    def _write_transform(self, worksheet, xls_index, opt_x):
        """Write transform to worksheet."""
        worksheet.cell(xls_index[0] - 4, xls_index[1] + 2, "transform")
        worksheet.cell(xls_index[0] - 5, xls_index[1] + 6, "Intrinsic Euler angles")
        for j, (label, value) in enumerate(zip(opt_x.transform.data, opt_x.data)):
            match len(label):
                case 1:
                    label = f"d{label} [mm]"
                case 2:
                    label = f"{label[0].upper()} [deg]"
            worksheet.cell(xls_index[0] - 4, xls_index[1] + 3 + j, label)
            worksheet.cell(xls_index[0] - 3, xls_index[1] + 3 + j, value)

    def load_target(self):
        """Load target geometories in cylindrical coordinate system."""
        # self.data["centerline_target"] = (
        #    xarray.DataArray(1, [("coil", self.data.coil.data)])
        #    * self.data.centerline_target
        # )
        for attr in ["fiducial_target", "centerline_target"]:
            self.data[f"{attr}_cyl"] = Rotate.to_cylindrical(self.data[attr])

    def load_measurement(self):
        """Load reference measurements."""
        self.data["fiducial"] = self.data.fiducial_target + self.data.fiducial_delta
        self.data["centerline"] = (
            self.data.centerline_target + self.data.centerline_delta
        )

    def evaluate_gpr(self, target="fiducial", postfix="gpr"):
        """Evaluate gpr in cylindrical coordinate system."""
        delta = Rotate.to_cylindrical(self.data[target]) - self.data.fiducial_target_cyl
        fiducial = f"fiducial_{postfix}"
        fiducial_std = f"fiducial_{postfix}_std"
        centerline = f"centerline_{postfix}"
        sample = f"sample_{postfix}"
        self.data[fiducial] = xarray.zeros_like(self.data.fiducial_target_cyl)
        self.data[fiducial_std] = xarray.zeros_like(self.data.fiducial_target_cyl)
        self.data[centerline] = xarray.zeros_like(self.data.centerline_target_cyl)
        self.data[sample] = (
            xarray.zeros_like(self.data[centerline])
            .expand_dims(dict(samples=self.samples), axis=-1)
            .copy()
        )
        for coil_index in range(self.data.sizes["coil"]):
            for space_index in range(self.data.sizes["cylindrical"]):
                self.load_gpr(coil_index, space_index)
                self.gpr.fit(delta[coil_index, :, space_index])
                (
                    self.data[fiducial][coil_index, :, space_index],
                    self.data[fiducial_std][coil_index, :, space_index],
                ) = self.gpr.predict(
                    self.data.target_length[coil_index], return_std=True
                )
                self.data[centerline][coil_index, :, space_index] = self.gpr.predict(
                    self.data.arc_length
                )
                self.data[sample][coil_index, :, space_index, :] = self.gpr.sample(
                    self.data.arc_length, self.samples
                )
        self.data[fiducial] += self.data.fiducial_target_cyl
        self.data[centerline] += self.data.centerline_target_cyl
        self.data[sample] += self.data.centerline_target_cyl
        for attr in [fiducial, centerline, sample]:
            self.data[attr] = Rotate.to_cartesian(self.data[attr])

    def plot_samples(self, label: str, coil_index: int, samples=10):
        """Load gaussian process regressor."""
        delta = Rotate.to_cylindrical(self.data[label]) - self.data.target_cyl
        axes = plt.subplots(3, 1, sharex=True)[1]
        for space_index in range(self.data.sizes["cylindrical"]):
            self.gpr.fit(delta[coil_index, :, space_index])
            self.gpr.predict(self.data.arc_length)
            self.gpr.plot(
                axes[space_index],
                text=False,
                marker="X",
                marker_color="C1",
                line_color="C0",
            )
            if samples > 0:
                self.gpr.sample(self.data.arc_length, samples)
                self.gpr.plot_samples(axes[space_index])

            self.gpr.predict(self.data.target_length)
            axes[space_index].plot(self.data.target_length, self.gpr.data.y_mean, "dC0")
            coord = str(self.data.cylindrical[space_index].values)
            coord = coord.replace("phi", r"\phi")
            axes[space_index].set_ylabel(rf"${coord}$")
        axes[-1].set_xlabel("arc length")
        axes[0].set_title(f"TF{self.data.coil[coil_index].values:1d}")
        plt.tight_layout()
        plt.savefig("gpr.png")

    def transform(self, x, points):
        """Return points transformed by vector x."""
        points = points.copy()
        points += x[:3]
        if len(x) == 6:
            rotate = Rotation.from_euler("XYZ", x[-3:], degrees=True)
            try:
                # xarray with coil dimension
                for coil in points.coil:
                    points.loc[{"coil": coil}] = rotate.apply(
                        points.sel(coil=coil).data
                    )
            except (AttributeError, TypeError):
                # numpy array or xarray without coil dimension
                if hasattr(points, 'values'):
                    points[:] = rotate.apply(points.values)
                else:
                    points[:] = rotate.apply(points)
        return points

    def delta(self, points):
        """Return coil-frame deltas in clocked sector frame."""
        offset = np.zeros_like(points)
        offset[..., 0] -= self.radial_offset

        # Clock both points and targets for sector fits
        points_clocked = points.copy()
        targets = self.data.fiducial_target.loc[points.coil].copy()
        targets_clocked = targets.copy()

        if self.is_sector:
            # Check if points has multiple coils or a single coil
            coil_dim = points.coil
            if coil_dim.ndim == 0:
                # Single coil scalar (0-d array)
                coil = int(coil_dim)
                points_clocked[:] = self.clock_coil(points.values, coil)
                targets_clocked[:] = self.clock_coil(targets.values, coil)
            else:
                # Multi-coil array
                for coil in coil_dim.values:
                    points_clocked.loc[{"coil": coil}] = self.clock_coil(
                        points.sel(coil=coil).values, coil
                    )
                    targets_clocked.loc[{"coil": coil}] = self.clock_coil(
                        targets.sel(coil=coil).values, coil
                    )

        return (
            Rotate.to_cylindrical(points_clocked)
            + offset
            - Rotate.to_cylindrical(targets_clocked)
        )

    @staticmethod
    def error_vector(delta, method):
        """Return error vector."""
        error = np.zeros(3)

        match method:
            case "rms":
                error[0] = np.mean(
                    delta[..., FiducialFit.fiducial_index["radial"], 0] ** 2
                )

                # toroidal_weight = np.ones_like(delta[..., 1])
                # toroidal_weight[3:6] = 10  # factor x10 for toroidal ILIS fiducials
                # error[1] = np.mean((toroidal_weight * delta[..., 1]) ** 2)
                # error[1] = np.mean(delta[..., [0, 1, 5, 3, 4, -1], 1] ** 2)

                error[1] = np.mean(
                    delta[..., FiducialFit.fiducial_index["toroidal"], 1] ** 2
                )

                # reduced toroidal constraint set
                # error[1] = np.mean(delta[..., [0, 5, 3, 4], 1] ** 2)

                error[2] = np.mean(
                    delta[..., FiducialFit.fiducial_index["vertical"], 2] ** 2
                )

                # drop F and C z constraint
                # error[2] = np.mean(delta[..., [1, 0, -1], 2] ** 2)

            case "max":
                error[0] = np.max(
                    abs(delta[..., FiducialFit.fiducial_index["radial"], 0])
                )
                error[1] = np.max(
                    abs(delta[..., FiducialFit.fiducial_index["toroidal"], 1])
                )
                error[2] = np.max(
                    abs(delta[..., FiducialFit.fiducial_index["vertical"], 2])
                )
            case _:
                raise NotImplementedError(f"Method {method} not implemented.")
        return error

    def transform_error(self, x, points, method):
        """Return transform error vector."""
        points = self.transform(x, points)
        return self.point_error(points, method)

    def weighted_transform_error(self, x, points, method):
        """Return weighted transform error vector."""
        return self.transform_error(x, points, method=method) * self.weights

    def point_error(self, points, method=None):
        """Return error vector."""
        if method is None:
            method = self.method
        delta = self.delta(points)
        return self.error_vector(delta, method)

    def max_transform_error(self, x, points):
        """Return maximum error."""
        return np.max(self.weighted_transform_error(x, points, method="max"))

    def rms_transform_error(self, x, points):
        """Return mean error."""
        return np.sqrt(np.mean(self.weighted_transform_error(x, points, method="rms")))

    def scalar_error(self, x, points):
        """Return scalar measure for fit error."""
        return getattr(self, f"{self.method}_transform_error")(x, points)

    @property
    def point_name(self):
        """Return reference point name."""
        if self.infer:
            return "fiducial_gpr"
        return "fiducial"

    def points(self, coil):
        """Return reference points (lab frame - clocking handled by delta())."""
        points = self.data[self.point_name].sel(coil=coil).copy()
        if self.infer:
            if len(self.dataset.ilis) == 0:
                Warning("ILIS data not found")
                return points
            target = ["B", "H", "A"]
            # points.loc[target] = self.data.fiducial.sel(coil=coil).loc[target]
            # project gpr fiducials to ILIS mid-plane
            frame = points.to_pandas()
            frame["coil"] = np.array(coil)

            _points = self.fiducial_ilis.project(frame).loc[target]
            _phi = np.arctan2(_points.y, _points.x)
            radius = np.sqrt(
                points.loc[target, "x"] ** 2 + points.loc[target, "y"] ** 2
            )
            points.loc[target, "x"] = radius * np.cos(_phi)
            points.loc[target, "y"] = radius * np.sin(_phi)

        return points

    @staticmethod
    def join(name: str, post_fix: str):
        """Return variable name with post_fix if set."""
        if post_fix:
            return "_".join([name, post_fix])
        return name

    def fit(self):
        """Perform sector fit."""
        transform_attrs = [
            "fiducial",
            "centerline",
            "fiducial_gpr",
            "centerline_gpr",
        ]
        for attr in transform_attrs:
            self.data[f"{attr}_fit"] = xarray.zeros_like(self.data[attr])

        self.data.coords["transform"] = ["x", "y", "z", "xx", "yy", "zz"]
        self.data["opt_x"] = xarray.DataArray(
            0.0,
            coords=[self.data.coil, self.data.transform],
            dims=["coil", "transform"],
        )
        for post_fix in ["", "gpr"]:
            error_attr = self.join("error", post_fix)
            self.data[error_attr] = xarray.DataArray(
                0.0,
                coords=[self.data.coil],
                dims=["coil"],
            )
            self.data[f"{error_attr}_fit"] = xarray.zeros_like(self.data[error_attr])
        for coil in tqdm(self.data.coil, "fitting coils"):
            points = xarray.concat(
                [self.points(coil=coil).copy() for coil in self.data.coil], "coil"
            )
            # points = self.points(coil=coil)
            # TODO fix for single vs sector
            xo = np.zeros(self.data.sizes["transform"])
            opt = minimize(
                self.scalar_error,
                xo,
                method="SLSQP",
                args=(points,),
                options={'ftol': 1e-6},
            )
            print(opt)
            if not opt.success:
                warnings.warn(f"optimization failed {opt}")
            self.data["opt_x"].loc[{"coil": coil}] = opt.x
            for attr in transform_attrs:
                # For sector fits: clock → transform → unclock
                self.data[f"{attr}_fit"].loc[{"coil": coil}] = self._sector_transform(
                    opt.x, self.data[attr].loc[{"coil": coil}].copy().data, coil
                )
            for post_fix in ["", "gpr"]:
                error_attr = self.join("error", post_fix)
                fiducial_attr = self.join("fiducial", post_fix)
                fiducial_points = self.data[fiducial_attr].sel(coil=coil)
                self.data[error_attr].loc[{"coil": coil}] = self.scalar_error(
                    xo, fiducial_points
                )
                self.data[f"{error_attr}_fit"].loc[{"coil": coil}] = self.scalar_error(
                    opt.x, fiducial_points
                )

    @cached_property
    def plotter(self):
        """Return FiducialPlotter instance."""
        return FiducialPlotter(self.data, factor=500)

    def plot_fit(self, coil_index, postfix=""):
        """Plot fits."""
        self.plotter.target(coil_index)
        stage = 1 + int(self.infer)
        self.plotter(postfix, stage, coil_index)
        title = f"Coil{self.data.coil[coil_index].data}"
        title += f"\norigin:{self.data.origin[coil_index].data}"
        title += f" phase:{self.phase}"
        title += f"\ninfer:{self.infer} method: {self.method}"
        title += f"\nilis:{self.ilis} pcr: {self.ilis_pcr}"
        self.plotter.axes[0].set_title(title, fontsize="large")
        if postfix[-3:] == "fit":
            self.text_fit(self.plotter.axes[0], coil_index)
            self.text_transform(self.plotter.axes[0], coil_index)

    def plot_transform(self, coil_index=0):
        """Plot transform text."""
        self.plotter("fit")
        self.text_transform(self.plotter.axes[0], coil_index)
        self.plotter.axes[0].set_title("transform: reference -> fit")

    def text_transform(self, axes, coil_index):
        """Display text transform."""
        opt_x = self.data.opt_x[coil_index].values
        deg_to_mm = 10570 * np.pi / 180
        angle_unit = "mm"  # r"$^o$"
        axes.text(
            0.3,
            0.5,
            f"dx: {opt_x[0]:1.3f}mm\n"
            + f"dy: {opt_x[1]:1.3f}mm\n"
            + f"dz: {opt_x[2]:1.3f}mm\n"
            + f"rx: {opt_x[3] * deg_to_mm:1.3}"
            + angle_unit
            + "\n"
            + f"ry: {opt_x[4] * deg_to_mm:1.3}"
            + angle_unit
            + "\n"
            + f"rz: {opt_x[5] * deg_to_mm:1.3}"
            + angle_unit,
            va="center",
            ha="left",
            transform=axes.transAxes,
            fontsize="small",
        )

    def reference_error(self, method: str):
        """Return reference error vector."""
        return self.point_error(self.data.reference.copy(), method)

    def text_fit(self, axes, coil_index):
        """Display text transform."""
        opt_x = self.data.opt_x[coil_index].data
        coil = self.data.coil[coil_index].data
        # points = self.data[self.point_name][coil_index]
        points = self.points(coil)
        error = {
            "rms": np.sqrt(self.transform_error(opt_x, points, "rms")),
            "max": self.transform_error(opt_x, points, "max"),
        }
        text = ""
        for i, coordinate in enumerate(
            ["radial: A,B,H", "toroidal: all", "vertical: C,D,E,F"]
        ):
            text += "\n" + coordinate + "\n"
            for method in ["rms", "max"]:
                text += f"    {method}: {error[method][i]:1.2f}\n"
        axes.text(
            0.9,
            0.5,
            text,
            va="center",
            ha="left",
            transform=axes.transAxes,
            fontsize="small",
        )

    def _get_delta(self, attr, fit=True):
        """Return ensemble deltas."""
        source_attr = attr
        if fit:
            source_attr += "_fit"
        return self.data[source_attr] - self.data[f"{attr}_target"]

    def plot_ensemble(self, fit=True, factor=250, axes=None, color=None):
        """Plot fit ensemble."""
        self.axes = self.set_axes("2d", nrows=1, ncols=2, sharey=True, axes=axes)
        for j in range(2):
            self.axes[j].plot(
                self.data.centerline_target[0, :, 0],
                self.data.centerline_target[0, :, 2],
                "gray",
                ls="--",
            )
        limits = self.axes_limit
        if color is None:
            color = [0, 0]

        centerline_delta = self._get_delta("centerline", fit)
        fiducial_delta = self._get_delta("fiducial", fit)

        for i in range(self.data.sizes["coil"]):
            j = 0 if self.data.origin[i] == "EU" else 1
            self.axes[j].plot(
                self.data.centerline_target[i, :, 0]
                + factor * centerline_delta[i, :, 0],
                self.data.centerline_target[i, :, 2]
                + factor * centerline_delta[i, :, 2],
                color=f"C{color[j]}",
                label=f"{self.data.coil[i].values:02d}",
            )
            self.axes[j].plot(
                self.data.fiducial_target[i, :, 0] + factor * fiducial_delta[i, :, 0],
                self.data.fiducial_target[i, :, 2] + factor * fiducial_delta[i, :, 2],
                ".",
                color=f"C{color[j]}",
            )
            color[j] += 1
        """
        for j, origin in enumerate(["EU", "JA"]):
            self.axes[j].legend(
                fontsize="large", loc="center", bbox_to_anchor=[0.4, 0.5]
            )
            # self.axes[j].set_title(f"{origin} {self.phase}")
        """
        limits[0]["x"] = [1500, 11000]
        limits[1]["x"] = [1500, 11000]
        limits[0]["y"] = [-8000, 8000]
        limits[1]["y"] = [-8000, 8000]
        self.axes_limit = limits

    @staticmethod
    def mask_frame(frame) -> pd.DataFrame:
        """Mask non-controlled fiducial deltas."""
        for axis, index_axis in zip(("xyz"), FiducialFit.fiducial_index.keys()):
            mask = [
                target
                for target in frame.index
                if target not in frame.index[FiducialFit.fiducial_index[index_axis]]
            ]
            frame.loc[mask, axis] = ""
        return frame

    @staticmethod
    def to_pandas(data, target="fiducial", datum="fiducial_target"):
        """Evaluate fit of target to datum."""
        target_cyl = Rotate.to_cylindrical(data[target])
        target_cyl.loc[..., "r"] -= data.radial_offset
        delta = target_cyl - Rotate.to_cylindrical(data[datum])
        delta = Rotate.to_cartesian(delta)
        frames = []
        for coil in data.coil.values:
            frame = delta.sel(coil=coil).to_pandas()  # .sortby("target")
            frame = fiducial.mask_frame(frame)
            frame.columns = pd.MultiIndex.from_product(
                [[f"Coil {coil}"], frame.columns]
            )
            frames.append(frame)
        return pd.concat(frames, axis=1)


if __name__ == "__main__":
    phase = "FAT supplier"
    phase = "FAT IO"
    phase = "SSAT BR"
    phase = "SSAT target"
    phase = "SSAT AR"
    phase = "SSAT AL"
    # phase = "SSAT AR2"

    # phase = "TFGS landing"
    # phase = "M0607"

    # phase = "SSAT target"
    # phase = "SSAT AL"
    # sectors = {7: [8, 9]}
    # sectors = {6: [12, 13]}
    # sectors = {5: [16]} # 16, 5
    sectors = {8: [4, 11]}

    fiducial = FiducialFit(
        phase=phase,
        sectors=sectors,
        fill=False,
        infer=True,
        ilis=True,
        ilis_pcr=True,
        method="rms",
    )
    #fiducial.build()

    for i in range(fiducial.data.sizes["coil"]):
        fiducial.plot_gpr_array(i, 2)

    for coil_index in range(fiducial.data.sizes["coil"]):
        fiducial.plot_fit(coil_index)
        fiducial.plot_fit(coil_index, "fit")
        del fiducial.plotter

    # fiducial.plot_ensemble(True, 250)

    #fiducial.write("_SSAT AL")
    # fiducial.write("SSAT target")
    fiducial.write("In-pit target")

    # print deltas
    coil_index = 0
    opt_x = fiducial.data.opt_x[coil_index].values
    # delta = fiducial.delta()

    """
    for coil in range(18):
        try:
            coil_index = fiducial.data.coil.sel(coil=coil).location.data
        except (KeyError, IndexError):
            continue
        fiducial.plotter.reset_axes()
        fiducial.plot_fit(coil_index)
        fiducial.plot_fit(coil_index, "fit")
        plt.tight_layout()
        plt.savefig(f"IDM_TF{coil}_fit.png")
    """

    # fiducial.plot_transform()

    # fiducial.plot_fit("target")
    # print(fiducial.data.target_cyl)

    # fiducial.plot()

    # coil = 4

    pd.options.display.precision = 3
    # print(to_pandas(fiducial.data, target="fiducial_fit"))

    print(FiducialFit.to_pandas(fiducial.data, target="fiducial_fit_gpr"))

    print(FiducialFit.to_pandas(fiducial.data, target="fiducial_fit"))

    print(FiducialFit.to_pandas(fiducial.data, target="fiducial_fit_gpr"))
