"""Post-process data from ansys result files."""
from dataclasses import dataclass, field
import os

import ansys.dpf.core as dpf
from ansys.dpf import post
import numpy as np
import pyvista as pv

from nova.structural.datadir import AnsysDataDir
from nova.structural.plotter import Plotter


@dataclass
class AnsysPost(AnsysDataDir, Plotter):
    """Manage access to Ansys results file."""

    model: dpf.Model = field(init=False, repr=False)
    time_support: list[float] = field(init=False)
    time_scoping: list[int] = field(init=False)
    meshed_region: dpf.MeshedRegion = field(init=False, repr=False)
    mesh: pv.PolyData = field(init=False, repr=False)

    def __post_init__(self):
        """Load results model."""
        super().__post_init__()
        self.load()

    def __str__(self):
        """Return Ansys model descriptor."""
        return self.model.__str__()

    @property
    def rst_file(self):
        """Return rst file path."""
        return os.path.join(self.directory, f'{self.file}.rst')

    @property
    def vtk_file(self):
        """Return vtk file path."""
        if self.subset == 'all':
            return os.path.join(self.directory, f'{self.file}.vtk')
        return os.path.join(self.directory, f'{self.file}_{self.subset}.vtk')

    def load(self):
        """Load vtk mesh file."""
        try:
            self.mesh = pv.read(self.vtk_file)
            self.time_support = self.mesh.field_arrays['time_support']
            self.time_scoping = self.mesh.field_arrays['time_scoping']
        except FileNotFoundError:
            self.load_ansys()

    def load_ansys(self):
        """Load Ansys rst file."""
        self.model = dpf.Model(self.rst_file)
        self.time_support = \
            self.model.metadata.time_freq_support.frequencies.data
        self.time_scoping = \
            range(1, self.model.metadata.time_freq_support.n_sets+1)
        self.load_meshed_region()
        self.mesh = self.meshed_region.grid
        self.mesh.field_arrays['time_support'] = self.time_support
        self.mesh.field_arrays['time_scoping'] = self.time_scoping
        self.load_displacement()
        self.mesh.save(self.vtk_file)

    def load_meshed_region(self):
        """Return scoped dpf meshed region."""
        if self.subset == 'all':
            self.meshed_region = self.model.metadata.meshed_region
            return
        mesh = dpf.Operator('mesh::by_scoping')  # operator instantiation
        mesh.inputs.mesh.connect(self.model.metadata.meshed_region)
        mesh.inputs.scoping.connect(self.mesh_scoping)
        self.meshed_region = mesh.outputs.mesh()

    @property
    def mesh_scoping(self):
        """Return named selection mesh scoping."""
        if self.subset == 'all':
            return self.model.metadata.meshed_region.nodes.scoping
        scope = dpf.Operator('scoping_provider_by_ns')
        scope.inputs.requested_location.connect(post.locations.nodal)
        scope.inputs.named_selection_name.connect(self.subset)
        scope.inputs.data_sources.connect(self.model.metadata.data_sources)
        return scope.outputs.mesh_scoping

    def load_displacement(self):
        """Load displacment field to vtk dataset."""
        displace = dpf.Operator('U')
        displace.inputs.mesh.connect(self.meshed_region)
        displace.inputs.time_scoping.connect(self.time_scoping)
        if self.mesh_scoping is not None:
            displace.inputs.mesh_scoping.connect(self.mesh_scoping)
        displace.inputs.data_sources.connect(self.model.metadata.data_sources)
        displace.inputs.requested_location.connect(post.locations.nodal)
        fields = displace.outputs.fields_container()
        self.store_fields('displacement', fields)

    @property
    def n_nodes(self):
        """Return node number for dpf scoped mesh."""
        return self.meshed_region.nodes.n_nodes

    @property
    def map_scoping(self):
        """Return meshed region map scoping."""
        return self.meshed_region.nodes.map_scoping

    def store_fields(self, label: str, fields: dpf.FieldsContainer,
                     component_number=3):
        """Store fields to pyvista mesh."""
        for i in range(len(fields)):
            name = f'{label}-{i}'
            self.mesh[name] = \
                np.full((self.n_nodes, component_number), np.nan)
            index, mask = self.map_scoping(fields[i].scoping)
            self.mesh[name][index] = fields[i].data[mask]

    def plot(self, loadcase=-1, factor=75):
        """Return pyvista plotter with mesh displacement."""
        if loadcase < 0:
            loadcase = self.time_scoping[loadcase]-1
        self.warp(f'displacement-{loadcase}', factor=factor, opacity=0.5)


if __name__ == '__main__':

    ansys = AnsysPost('TFC2_CC_design', 'p3', 'TF1_OIS_DOWN')
    ansys.plot(-1, factor=75)
