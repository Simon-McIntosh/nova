"""Post-process data from ansys result files."""
from dataclasses import dataclass, field

import ansys.dpf.core as dpf
from ansys.dpf import post
import numpy as np
import pyvista as pv
import pyvista.examples

from nova.assembly.datadir import DataDir
from nova.assembly.plotter import Plotter


@dataclass
class AnsysPost(DataDir, Plotter):
    """Manage access to Ansys results file."""

    model: dpf.Model = field(init=False, repr=False)
    time_support: list[float] = field(init=False)
    time_scoping: list[int] = field(init=False)
    meshed_region: dpf.MeshedRegion = field(init=False, repr=False)
    mesh: pv.UnstructuredGrid = field(init=False, repr=False)

    def __post_init__(self):
        """Load results model."""
        super().__post_init__()
        self.load()

    def __str__(self):
        """Return Ansys model descriptor."""
        try:
            return self.model.__str__()
        except AttributeError:
            return dpf.Model(self.rst_file).__str__()

    def load(self):
        """Load vtk mesh file."""
        try:
            print(self.ansys_file)
            self.mesh = pv.read(self.ansys_file)
            self.time_support = self.mesh.field_data["time_support"]
            self.time_scoping = self.mesh.field_data["time_scoping"]
        except FileNotFoundError:
            self.load_ansys()

    def load_ansys(self):
        """Load Ansys rst file."""
        self.model = dpf.Model(self.rst_file)
        self.time_support = self.model.metadata.time_freq_support.frequencies.data
        self.time_scoping = range(1, self.model.metadata.time_freq_support.n_sets + 1)
        self.load_meshed_region()
        self.mesh = self.meshed_region.grid
        self.mesh["ids"] = self.meshed_region.nodes.scoping.ids
        self.mesh.field_data["time_support"] = self.time_support
        self.mesh.field_data["time_scoping"] = self.time_scoping
        self.load_displacement()
        self.load_vonmises()
        self.mesh.save(self.ansys_file)

    def load_meshed_region(self):
        """Return scoped dpf meshed region."""
        if self.subset == "all":
            self.meshed_region = self.model.metadata.meshed_region
            return
        mesh = dpf.Operator("mesh::by_scoping")  # operator instantiation
        mesh.inputs.mesh.connect(self.model.metadata.meshed_region)
        mesh.inputs.scoping.connect(self.mesh_scoping)
        self.meshed_region = mesh.outputs.mesh()

    @property
    def mesh_scoping(self):
        """Return named selection mesh scoping."""
        if self.subset == "all":
            return self.model.metadata.meshed_region.nodes.scoping
        scope = dpf.Operator("scoping_provider_by_ns")
        scope.inputs.requested_location.connect(post.locations.nodal)
        scope.inputs.named_selection_name.connect(self.subset)
        scope.inputs.data_sources.connect(self.model.metadata.data_sources)
        return scope.outputs.mesh_scoping

    def load_field(self, label, operator, component_number):
        """Load field data from Ansys rst file."""
        displace = dpf.Operator(operator)
        displace.inputs.mesh.connect(self.meshed_region)
        displace.inputs.time_scoping.connect(self.time_scoping)
        if self.mesh_scoping is not None:
            displace.inputs.mesh_scoping.connect(self.mesh_scoping)
        displace.inputs.data_sources.connect(self.model.metadata.data_sources)
        displace.inputs.requested_location.connect(post.locations.nodal)
        fields = displace.outputs.fields_container()
        self.store_fields(label, fields, component_number=component_number)

    def load_displacement(self):
        """Load displacment field to vtk dataset."""
        self.load_field("disp", "U", 3)

    def load_vonmises(self):
        """Load VonMises stress."""
        self.load_field("vm", "S_eqv", 1)

    def load_gap(self):
        """Load contact gap to vtk dataset."""
        displace = dpf.Operator("ECT_GAP")
        displace.inputs.mesh.connect(self.meshed_region)
        displace.inputs.time_scoping.connect(self.time_scoping)
        if self.mesh_scoping is not None:
            displace.inputs.mesh_scoping.connect(self.mesh_scoping)
        displace.inputs.data_sources.connect(self.model.metadata.data_sources)
        displace.inputs.requested_location.connect(post.locations.nodal)
        fields = displace.outputs.fields_container()
        self.store_fields("gap", fields, component_number=1)

    @property
    def n_nodes(self):
        """Return node number for dpf scoped mesh."""
        return self.meshed_region.nodes.n_nodes

    @property
    def map_scoping(self):
        """Return meshed region map scoping."""
        return self.meshed_region.nodes.map_scoping

    def store_fields(self, label: str, fields: dpf.FieldsContainer, component_number=3):
        """Store fields to pyvista mesh."""
        for i in range(len(fields)):
            name = f"{label}-{i}"
            self.mesh[name] = np.full((self.n_nodes, component_number), np.nan)
            index, mask = self.map_scoping(fields[i].scoping)
            self.mesh[name][index] = fields[i].data[mask]

    def select(self, index, inplace=True):
        """Return body, update mesh if inplace==True."""
        body = self.mesh.split_bodies()[index]
        if inplace:
            self.mesh = body
        return body

    def subplot(self):
        """Generate linked loadcase subplot."""
        plotter = pv.Plotter(shape=(3, 4), border=False)

        index = 0
        for col in range(4):
            for row in range(3):
                plotter.subplot(row, col)
                self.plot(loadcase=index + 6, plotter=plotter, opacity=0, factor=2000)
                index += 1
        plotter.link_views()
        plotter.show()


if __name__ == "__main__":
    ansys = AnsysPost("TFCgapsG10", "k1", "wp")

    # ansys.warp(500)

    """
    ansys.select(0)
    ansys.mesh += AnsysPost('TFCgapsG10', 'w1', 'case_ol').select(1)

    airplane = pv.examples.load_airplane()
    airplane.scale(5e-3, inplace=True)
    airplane.translate([-4.5, -10, 0], inplace=True)
    airplane.rotate_z(10, inplace=True)
    #ansys.mesh += airplane
    """
    ansys.plot()
