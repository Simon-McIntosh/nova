"""Apply structural deformation to TF coilset."""

from dataclasses import dataclass, field
import os

import numpy as np
import pandas
import pathlib
import pyvista as pv

from nova.assembly.ansysvtk import AnsysVTK
from nova.assembly.clusterturns import ClusterTurns
from nova.assembly.datadir import DataDir
from nova.assembly.plotter import Plotter
from nova.assembly.windingpack import WindingPack
from nova.assembly.uniformwindingpack import UniformWindingPack
from nova.utilities.time import clock


@dataclass
class CCL(DataDir, Plotter):
    """Post-process Ansys output from F4E's 18TF coil model."""

    cluster: int = 1
    scenario: dict[str, int] = field(init=False, repr=False)
    ansys: pv.PolyData = field(init=False, repr=False)
    mesh: pv.PolyData = field(init=False, repr=False)

    def __post_init__(self):
        """Load database."""
        super().__post_init__()
        self.subset = "WP"
        self.load()

    def __str__(self):
        """Return Ansys model descriptor (takes time - remote mount)."""
        return AnsysVTK(*self.args).__str__()

    def reload(self, file):
        """Reload source file."""
        self.file = file
        self.__post_init__()

    def load_ensemble(self, prefix=None):
        """Load ensemble ccl dataset and store reduced data in vtk format."""
        _file = self.file
        paths = list(pathlib.Path(self.rst_folder).rglob("*.rst"))
        files = [
            file
            for path in paths
            if not os.path.isfile(
                self.ccl_file.replace(self.file, file := path.name[:-4])
            )
        ]
        if prefix is not None:
            files = [file for file in files if file[: len(prefix)] == prefix]
            # files = [file for file in files if file[-3:] == 'f4e']
        nfiles = len(files)
        tick = clock(nfiles, header=f"loading {nfiles} *.rst files {files}")
        for file in files:
            self.reload(file)
            tick.tock()
        self.reload(_file)

    def load(self):
        """Load vtm data file."""
        try:
            self.mesh = pv.read(self.ccl_file)
        except FileNotFoundError:
            self.load_ansys()
            self.load_ccl()
        if self.cluster:
            self.mesh = ClusterTurns(self.mesh, self.cluster).mesh

    def recalculate(self):
        """Reload vtm datafile."""
        self.load_ansys()
        self.load_ccl()
        if self.cluster:
            self.mesh = ClusterTurns(self.mesh, self.cluster).mesh

    def load_ansys(self):
        """Load ansys vtk mesh."""
        self.ansys = AnsysVTK(*self.args).mesh

    def load_windingpack(self):
        """Load conductor windingpack."""
        if self.cluster is not None:
            return UniformWindingPack().mesh
        return WindingPack("TFC1_CL").mesh

    def load_ccl(self):
        """Load referance windingpack ccl."""
        self.mesh = self.load_windingpack()
        self.mesh = self.interpolate_coils(self.mesh, self.ansys)
        mesh = self.mesh.copy()
        self.mesh.clear_data()
        self.mesh["arc_length"] = mesh["arc_length"]
        self.mesh.field_data.update(self.ansys.field_data)
        for scn in self.ansys.field_data["scenario"]:
            try:
                self.mesh[scn] = mesh[scn]
            except KeyError:
                pass
        try:
            self.mesh["turns"] = mesh["turns"]
        except KeyError:
            pass
        self.mesh.save(self.ccl_file)

    def interpolate_coils(self, source, target, sharpness=3, radius=1.5, n_cells=7):
        """Retun interpolated mesh."""
        return source.interpolate(
            target, sharpness=sharpness, radius=radius, strategy="closest_point"
        )

    def csv_file(self, scenario: str):
        """Return csv filename."""
        if self.cluster == 1:
            return os.path.join(self.directory, f"{self.file}_{scenario}.csv")
        return os.path.join(
            self.directory, f"{self.file}_{scenario}_{self.cluster}.csv"
        )

    def to_dataframe(self, scenario):
        """Return mesh as dataframe."""
        mesh = self.mesh.copy()
        mesh.points += mesh[scenario]
        frames = list()
        for cell in range(mesh.n_cells):
            points = mesh.cell_points(cell)
            n_seg = len(points) - 1
            coil = int(mesh["coil"][cell])
            cluster = int(mesh["cluster"][cell])
            nturn = int(mesh["nturn"][cell])
            cpoint = (points[1:] + points[:-1]) / 2  # centerpoint
            vector = points[1:] - points[:-1]
            data = dict(
                coil=np.full(n_seg, f"TF{coil+1}"),
                cluster=np.full(n_seg, cluster),
                nturn=np.full(n_seg, nturn),
                x=cpoint[:, 0],
                y=cpoint[:, 1],
                z=cpoint[:, 2],
                dx=vector[:, 0],
                dy=vector[:, 1],
                dz=vector[:, 2],
            )
            frames.append(pandas.DataFrame(data))
        frame = pandas.concat(frames)
        frame.to_csv(self.csv_file(scenario), index=False)

    def export(self):
        """Export dataset."""
        files = [f"kp{i}" for i in range(5, 10)]
        for file in files:
            for cluster in [1]:  # [1, 5, 10]:
                self.cluster = cluster
                self.reload(file)
                for scenario in ["EOB"]:  # ['TFonly', 'SOD', 'EOB']:
                    self.to_dataframe(scenario)

    def animate(self, displace: str, view="xy", max_factor=100):
        """Animate displacement."""
        filename = os.path.join(self.directory, self.file)
        super().animate(filename, displace, view=view, max_factor=max_factor)


if __name__ == "__main__":
    ccl = CCL("TFCgapsG10", "k1", cluster=None)

    ccl.mesh["TFonly-cooldown"] = ccl.mesh["TFonly"] - ccl.mesh["cooldown"]

    # mesh = ccl.mesh.slice(normal=[0, 0, 1])
    # clip = pv.Cylinder(direction=(0, 0, 1), radius=3)
    # ccl.mesh = mesh.clip_surface(clip, invert=True)

    # ccl.animate('TFonly-cooldown', 'iso')
    ccl.warp(100)

    # ccl.plot()

    """
    #ccl.recalculate()
    #ccl.export()
    #ccl.to_dataframe('EOB')

    #ccl.load_ensemble()

    #ccl.mesh['EOB-cooldown'] = ccl.mesh['EOB'] - ccl.mesh['cooldown']

    #ccl.to_dataframe('EOB')

    #ccl.export()
    #ccl.plot('TFonly', 'cooldown', factor=180)
    #

    #ccl.warp(50, displace='TFonly')

    #ccl.animate('TFonly-cooldown', view='xy')
    """
