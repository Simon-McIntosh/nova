from dataclasses import dataclass, field
import os

import ansys.dpf.core as dpf
from ansys.dpf import post
import numpy as np
import pyvista

from nova.definitions import root_dir


@dataclass
class TFC18:

    file: str
    directory: str = None
    mesh_scoping: dpf.Scoping = field(init=False, repr=False, default=None)
    mesh: dpf.MeshedRegion = field(init=False, repr=False, default=None)

    def __post_init__(self):
        """Set directory and load model."""
        if self.directory is None:
            self.directory = os.path.join(root_dir, 'data/Ansys/TFC18')
        self.load(self.file)

    def __str__(self):
        """Return model description."""
        return self.model.__str__()

    def load(self, file):
        """Update filename and load model."""
        self.file = file
        self.model = dpf.Model(os.path.join(self.directory, f'{file}.rst'))

    def select(self, name, loc='nodal'):
        """Define named selection mesh scoping."""
        scope = dpf.Operator('scoping_provider_by_ns')
        scope.inputs.requested_location.connect(getattr(post.locations, loc))
        scope.inputs.named_selection_name.connect(name)
        scope.inputs.data_sources.connect(self.model.metadata.data_sources)
        mesh_scoping = scope.outputs.mesh_scoping()
        self.mesh_scoping = mesh_scoping
        self.mesh = self._mesh()

    def intersect(self, names: list[str]):
        """Return mesh scoping as intersection of two named_selections."""
        scope = dpf.Operator('scoping::intersect')
        scope.inputs.scopingA.connect(self.select(names[0]))
        scope.inputs.scopingB.connect(self.select(names[1]))
        self.mesh_scoping = scope.outputs.intersection()
        self.mesh = self._mesh()

    def _mesh(self):
        """Return scoped mesh."""
        mesh = dpf.Operator('mesh::by_scoping')  # operator instantiation
        mesh.inputs.mesh.connect(self.model.metadata.meshed_region)
        if self.mesh_scoping is not None:
            mesh.inputs.scoping.connect(self.mesh_scoping)
        return mesh.outputs.mesh()

    @property
    def disp(self):
        """Return displacment field."""
        displace = dpf.Operator('U')
        displace.inputs.mesh.connect(self.mesh)
        if self.mesh_scoping is not None:
            displace.inputs.mesh_scoping.connect(self.mesh_scoping)
        #    #displace.inputs.mesh_scoping.connect(self.mesh_scoping)
        displace.inputs.data_sources.connect(self.model.metadata.data_sources)
        displace.inputs.requested_location.connect(post.locations.nodal)
        return displace.outputs.fields_container()[0]


'''
op = dpf.Operator('scoping_provider_by_ns')  # operator instantiation
op.inputs.requested_location.connect(my_requested_location)
op.inputs.named_selection_name.connect(my_named_selection_name)
op.inputs.int_inclusive.connect(my_int_inclusive)  # optional
op.inputs.streams_container.connect(my_streams_container)  # optional
op.inputs.data_sources.connect(my_data_sources)
my_mesh_scoping = op.outputs.mesh_scoping()
'''

#mesh_scoping.inputs.data_sources.connect(model.metadata.data_sources)
#wp_scoping = mesh_scoping.outputs.mesh_scoping()

#wp_mesh = dpf.Operator('mesh::by_scoping')  # operator instantiation
#wp_mesh.inputs.scoping.connect(mesh_scoping.outputs.mesh_scoping)


if __name__ == '__main__':

    tf = TFC18('V4')

    #tf.select('N_WP_CENTERLINE')
    tf.select('WP')

    disp = tf.disp
    grid = tf.mesh.grid
    grid['disp'] = np.full((tf.mesh.nodes.n_nodes, 3), np.nan)
    ind, mask = tf.mesh.nodes.map_scoping(disp.scoping)
    grid['disp'][ind] = disp.data[mask]

    warp = grid.warp_by_vector('disp', factor=75)


    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, color='w')
    plotter.add_mesh(warp)

    plotter.show()


    #tf.grid.plot(disp)
    '''
    grid = tf.grid
    disp = tf.disp

    grid['disp'] = tf.disp.data




    #nodes = tf.mesh.nodes.coordinates_field.data
    #plt.plot(field.daa)
    #tf.grid.plot()
    #tf.name_select('WP')
    '''
