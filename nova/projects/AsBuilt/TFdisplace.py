from dataclasses import dataclass, field
import os

import ansys.dpf.core as dpf
from ansys.dpf import post
import numpy as np
import pyvista as pv

from nova.definitions import root_dir
from nova.utilities.time import clock


@dataclass
class DataDir:
    """Manage file paths."""

    file: str
    name: str = 'all'
    directory: str = field(repr=False, default=None)

    def __post_init__(self):
        """Set data directory."""
        if self.directory is None:
            self.directory = os.path.join(root_dir, 'data/Ansys/TFC18')

    @property
    def metadata(self):
        """Return file metadata."""
        return self.file, self.name, self.directory


@dataclass
class AnsysDPF(DataDir):
    """Access Ansys results."""

    model: dpf.Model = field(init=False, repr=False)
    mesh: dpf.MeshedRegion = field(init=False, repr=False)
    dataset: pv.DataSet = field(init=False, repr=False)

    def __post_init__(self):
        """Load results model."""
        super().__post_init__()
        self.model = dpf.Model(self.rst_file)

    def __str__(self):
        """Return Ansys model descriptor."""
        return self.model.__str__()

    @property
    def rst_file(self):
        """Return rst file path."""
        return os.path.join(self.directory, f'{self.file}.rst')

    def load_dataset(self):
        """Load dpf model instance."""
        self.mesh = self.load_mesh()
        self.dataset = self.mesh.grid
        self.load_displacement()
        return self.dataset

    def load_mesh(self):
        """Return scoped mesh."""
        mesh = dpf.Operator('mesh::by_scoping')  # operator instantiation
        mesh.inputs.mesh.connect(self.model.metadata.meshed_region)
        if self.name != 'all':
            mesh.inputs.scoping.connect(self.mesh_scoping)
        return mesh.outputs.mesh()

    @property
    def mesh_scoping(self):
        """Return named selection mesh scoping."""
        scope = dpf.Operator('scoping_provider_by_ns')
        scope.inputs.requested_location.connect(post.locations.nodal)
        scope.inputs.named_selection_name.connect(self.name)
        scope.inputs.data_sources.connect(self.model.metadata.data_sources)
        return scope.outputs.mesh_scoping

    def load_displacement(self):
        """Load displacment field to vtk dataset."""
        displace = dpf.Operator('U')
        displace.inputs.mesh.connect(self.mesh)
        displace.inputs.time_scoping.connect(
            range(1, self.model.metadata.time_freq_support.n_sets+1))
        if self.mesh_scoping is not None:
            displace.inputs.mesh_scoping.connect(self.mesh_scoping)
        displace.inputs.data_sources.connect(self.model.metadata.data_sources)
        displace.inputs.requested_location.connect(post.locations.nodal)
        self.store_field('displacement', displace.outputs.fields_container())

    def store_field(self, label: str, fields: dpf.FieldsContainer):
        """Store field to DataSet."""
        for i in range(len(fields)):
            name = f'{label}_{i}'
            self.dataset[name] = np.full((self.mesh.nodes.n_nodes, 3), np.nan)
            index, mask = self.mesh.nodes.map_scoping(fields[i].scoping)
            self.dataset[name][index] = fields[i].data[mask]


@dataclass
class TFC18(DataDir):
    """Post-process Ansys output from F4E's 18TF coil model."""

    multiblock: pv.MultiBlock = field(init=False, repr=False)
    dataset: pv.DataSet = field(init=False, repr=False)

    def __post_init__(self):
        """Load database."""
        super().__post_init__()
        self.load()

    def __str__(self):
        """Return Ansys model descriptor."""
        return AnsysDPF(*self.metadata).__str__()

    @property
    def vtm_file(self):
        """Return vtk file path."""
        return os.path.join(self.directory, f'{self.file}.vtm')

    def load(self):
        """Load vtm data file."""
        try:
            self.multiblock = pv.read(self.vtm_file)
        except FileNotFoundError:
            self.multiblock = pv.MultiBlock()
        self.load_dataset(self.name)

    def store(self):
        """Store vtm file."""
        self.multiblock.save(self.vtm_file)

    def load_dataset(self, name):
        """Return vtk DataSet."""
        self.name = name
        if self.name not in self.multiblock.keys():
            ansys = AnsysDPF(*self.metadata)
            self.multiblock[self.name] = ansys.load_dataset()
            self.store()
        self.dataset = self.multiblock[self.name]

    def extract_winding_pack(self):
        """Extract winding pack displacments."""
        self.load_dataset('WP')
        tick = clock(18, header='Extracting winding pack displacements')
        for i in range(18):
            self.load_dataset(f'E_WP_{i+1}')
            tick.tock()

    def plot(self, factor=0):
        """Plot dataset."""
        if factor == 0:
            return self.dataset.plot()
        warp = self.dataset.warp_by_vector('displacement', factor=factor)
        plotter = pv.Plotter()
        plotter.add_mesh(self.dataset, color='w')
        plotter.add_mesh(warp)
        plotter.show()


if __name__ == '__main__':

    #tf = TFC18('v4', 'WP')
    #tf.extract_winding_pack()
    #tf.plot(65)

    ansys = AnsysDPF('v4', 'E_WP_18')
    dataset = ansys.load_dataset()

    #tf.grid('WP')
    '''
    #tf.select('N_WP_CENTERLINE')
    tf.select('WP')

    #tf.store()

    disp = tf.displacment()




    plotter = pv.Plotter()
    plotter.add_mesh(grid, color='w')
    plotter.add_mesh(warp)

    plotter.show()

    #tf.grid.plot(disp)
    '''



    '''
    grid = tf.grid
    disp = tf.disp

    grid['disp'] = tf.disp.data




    #nodes = tf.mesh.nodes.coordinates_field.data
    #plt.plot(field.daa)
    #tf.grid.plot()
    #tf.name_select('WP')
    '''
