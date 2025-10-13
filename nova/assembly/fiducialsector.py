"""Manage TFC fiducial data for coil and sector allignment."""

from dataclasses import dataclass, field
from typing import ClassVar
from warnings import warn

import altair as alt
import itertools
import numpy as np
import pandas

from nova.assembly.fiducialccl import Fiducial, FiducialRE, FiducialIDM
from nova.assembly.fiducialilis import FiducialIlis
from nova.assembly.ilisnominal import NominalIlis
from nova.assembly.sectordata import SectorData

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

    def _set_phase(self):
        """Expand short string phase label."""
        self.phase = self.sheets.get(self.phase, self.phase)

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


if __name__ == "__main__":
    phase = "SSAT BR"
    phase = "SSAT target"
    # phase = "SSAT AL"
    # phase = "SSAT AR2"
    # phase = "SSAT AR target"
    # phase="In-pit target"

    sectors = {7: [8, 9]}
    sectors = {6: [12, 13]}
    sectors = {5: [16, 5]}
    sectors = {8: [4, 11]}

    fiducial = FiducialSector(phase=phase, sectors=sectors, private=True)
    fiducial.compare("IDM")

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

    sector_index = [
        (coil, plane)
        for coil, plane in zip(list(sectors.values())[0], ("ILIS +1", "ILIS -1"))
    ]

    sector_planes = ilis.planes.groupby(["coil"], group_keys=False).apply(
        lambda x: clock(x, x.name, cords=[("x", "y", "z"), ("nx", "ny", "nz")])
    )

    sector_data = ilis.data.groupby(["coil"], group_keys=False).apply(
        lambda x: clock(x, x.name)
    )

    # sector_data.groupby(["coil"]).apply(lambda x: griddata(x.loc[:, ('x', 'y', 'z')])

    # coils = ilis.data.coil.unique()

    # print(nominal.angle_to_xz(ilis.planes))
    # print(nominal.angle_to_xy(ilis.planes))

    midplane = ilis.intersect(sector_planes.loc[sector_index])

    grid_r, grid_z = np.mgrid[
        slice(ilis.data.r.min(), ilis.data.r.max(), 20j),
        slice(ilis.data.z.min(), ilis.data.z.max(), 40j),
    ]

    offset = []
    offset_data_list = []
    
    for coil, feature in sector_index:
        plane_index = (sector_data.coil == coil) & (sector_data.feature == feature)
        plane_offset = ilis.offset(
            sector_data.loc[plane_index, ("x", "y", "z")], midplane
        )

        offset.append(griddata(
            sector_data.loc[plane_index, ("r", "z")],
            plane_offset,
            (grid_r, grid_z),
            method="linear",
        ))
        
        # Store offset data for plotting
        plane_data = sector_data.loc[plane_index, ["r", "z"]].copy()
        plane_data["offset"] = plane_offset
        plane_data["coil"] = coil
        plane_data["feature"] = feature
        offset_data_list.append(plane_data)
    
    # Combine all offset data
    offset_df = pandas.concat(offset_data_list, ignore_index=True)
    
    # Create first and last plane dataframes
    first_plane = offset_df[offset_df["feature"] == sector_index[0][1]].copy()
    
    last_plane = offset_df[offset_df["feature"] == sector_index[-1][1]].copy()

    gap_plane = pandas.DataFrame({'r': grid_r.flatten(), 'z': grid_z.flatten(), 
                                  "offset": offset[1].flatten() - offset[0].flatten()})
    gap_plane["coil"] = 0
    gap_plane["feature"] = 'Gap'
    
    # Combine all data for plotting
    plot_data = pandas.concat([first_plane, gap_plane, last_plane], ignore_index=True)
    
    # Create three-column Altair chart with synced tooltips
    base = alt.Chart(plot_data).mark_circle(size=60)
    
    chart = base.encode(
        x=alt.X("r:Q", title="Radius (r)", scale=alt.Scale(domain=[2000, 3200])),
        y=alt.Y("z:Q", title="Height (z)"),
        color=alt.Color(
            "offset:Q",
            title="Offset",
            scale=alt.Scale(scheme="redblue"),
        ),
        tooltip=[
            alt.Tooltip("coil:N", title="Coil"),
            alt.Tooltip("r:Q", title="r", format=".2f"),
            alt.Tooltip("z:Q", title="z", format=".2f"),
            alt.Tooltip("offset:Q", title="Offset", format=".4f"),
        ],
        facet=alt.Facet(
            "feature:N",
            title=None,
            header=alt.Header(labelFontSize=12),
            sort=["ILIS +1", "Gap", "ILIS -1"]
        )
    ).properties(
        width=250,
        height=300
    ).resolve_scale(
        color="independent"
    ).configure_axis(
        grid=True
    ).interactive()
    
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

    base = alt.Chart(data, width=125, height=175)

    select = {
        "ILIS": alt.FieldOneOfPredicate(
            field="feature", oneOf=[f"ILIS {sign}1" for sign in ["+", "-"]]
        ),
        "ILIS_outlier": alt.datum.outlier,
        # "ILIS_outlier": alt.datum.outlier_factor < 0,
        "CCL": alt.datum.feature == "CCL",
    }

    scatter = (
        base.mark_circle(size=60)
        .transform_filter(select["ILIS"])
        .encode(
            x="ro_phi",
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
        .encode(x="ro_phi", y="z")
    )

    outlier = (
        base.mark_circle(size=80, color="red", filled=False)
        .transform_filter(select["ILIS_outlier"])
        .encode(x="ro_phi", y="z", tooltip=["Name", "offset"])
    )

    ccl_points = (
        base.mark_point(size=60, color="black")
        .transform_filter(select["CCL"])
        .encode(
            x="ro_phi",
            y="z",
            tooltip=["Name", "ro_phi", "y"],
            shape=alt.Shape("type"),
            # color=alt.Color("transform").scale(scheme="set2"),
        )
    )

    ccl_text = (
        base.mark_text(align="center", baseline="middle", dy=12)
        .transform_filter(select["CCL"])
        .encode(text="Name", x="ro_phi", y="z")
    )

    # row=alt.Row("coil:N"),
    # column=alt.Column("ilis:N").sort(["-1", "1"]),
    # color="r:Q",
    # tooltip=["Name", "r", "phi"],
    # ).show()

    # fit = base

    chart = scatter + outlier + ccl_points + ccl_text

    chart = (
        chart.facet(
            row="coil", column=alt.Column("feature").sort(["ILIS -1", "CCL", "ILIS +1"])
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

    vedo.Points(points.loc[index, ['r', 'z', 'offset']]).generate_delaunay2d(tol=0.000001).c('green').show(axes=1).close()
    """
