"""Assess fiducial measurement deviations without fitting.

Uses FiducialFit's battle-tested delta() and error_vector() methods
to evaluate measurement deviations in cylindrical coordinates
(dr, r*dphi, dz), without performing the SLSQP optimization fit.

The delta() method properly decomposes Cartesian positions into
cylindrical deviations from target, including radial offset correction.
This avoids the pitfalls of naively differencing Cartesian coordinates
for coils at non-zero toroidal angles.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nova.assembly.fiducialfit import FiducialFit


@dataclass
class FiducialAssess(FiducialFit):
    """Assess fiducial measurement deviations without fitting."""

    filename: str = "fiducial_assess"

    def build(self):
        """Build measurement data without fitting."""
        super(FiducialFit, self).build()
        self.data = self.data.rename(dict(space="cartesian"))
        self.load_target()
        self.load_measurement()
        self.evaluate_gpr("fiducial", "gpr")

    def deviation(self, source: str = "fiducial") -> pd.DataFrame:
        """Return cylindrical deviations masked to constrained directions.

        Uses FiducialFit.delta() to compute (dr, r*dphi, dz) deviations
        with radial offset, then masks unconstrained directions per
        fiducial_index.

        Parameters
        ----------
        source : str
            Data variable: 'fiducial' (raw) or 'fiducial_gpr' (smoothed)
        """
        targets = list(self.data.target.values)
        frames = []
        for coil in self.data.coil.values:
            points = self.data[source].sel(coil=coil)
            delta = self.delta(points)
            frame = delta.to_pandas()
            frame.columns = ["dr", "r_dphi", "dz"]
            for col, key in zip(frame.columns, FiducialFit.fiducial_index):
                constrained = [targets[i] for i in FiducialFit.fiducial_index[key]]
                mask = frame.index.difference(constrained)
                frame.loc[mask, col] = np.nan
            frame.columns = pd.MultiIndex.from_product(
                [[f"Coil {coil}"], frame.columns]
            )
            frames.append(frame)
        return pd.concat(frames, axis=1)

    def summary(self, source: str = "fiducial") -> pd.DataFrame:
        """Return RMS and max error per coil.

        Uses FiducialFit.point_error() which evaluates only the
        constrained fiducial indices defined in fiducial_index.

        Parameters
        ----------
        source : str
            Data variable: 'fiducial' (raw) or 'fiducial_gpr' (smoothed)
        """
        rows = []
        for coil in self.data.coil.values:
            points = self.data[source].sel(coil=coil)
            rms = np.sqrt(self.point_error(points, "rms"))
            max_err = self.point_error(points, "max")
            rows.append(
                {
                    "coil": int(coil),
                    "rms_r": rms[0],
                    "rms_rphi": rms[1],
                    "rms_z": rms[2],
                    "max_r": max_err[0],
                    "max_rphi": max_err[1],
                    "max_z": max_err[2],
                }
            )
        return pd.DataFrame(rows).set_index("coil")


if __name__ == "__main__":
    phase = "TFGS Landing"

    sectors = {7: [8, 9]}
    sectors = {6: [12, 13]}
    sectors = {5: [16, 5]}
    # sectors = {8: [4, 11]}
    # sectors = {4: [2, 3]}

    assess = FiducialAssess(
        phase=phase,
        sectors=sectors,
        fill=False,
        infer=True,
        ilis=True,
        ilis_pcr=True,
        method="rms",
        coupled=False,
    )
    assess.build()

    pd.options.display.precision = 3

    print("\nCylindrical deviation (dr, r*dphi, dz), constrained dirs only:")
    print("Source: raw measurement")
    print(assess.deviation("fiducial").to_string())

    print("\nSource: GPR-smoothed")
    print(assess.deviation("fiducial_gpr").to_string())

    print("\nError summary (constrained fiducials):")
    print("Source: raw measurement")
    print(assess.summary("fiducial").to_string())

    print("\nSource: GPR-smoothed")
    print(assess.summary("fiducial_gpr").to_string())
