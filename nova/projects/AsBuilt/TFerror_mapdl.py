import os

import ansys.dpf.core as dpf
from ansys.dpf import post
import pyvista

from nova.definitions import root_dir

rst_file = os.path.join(root_dir, "data/Ansys/TFC18/V4.rst")

model = dpf.Model(rst_file)
mesh_scoping = model.operator("scoping_provider_by_ns")
mesh_scoping.inputs.requested_location.connect("Nodal")
mesh_scoping.inputs.named_selection_name.connect("WP")
# mesh_scoping.inputs.data_sources.connect(model.metadata.data_sources)
wp_scoping = mesh_scoping.outputs.mesh_scoping()

wp_mesh = dpf.Operator("mesh::by_scoping")  # operator instantiation
wp_mesh.inputs.scoping.connect(mesh_scoping.outputs.mesh_scoping)
wp_mesh.inputs.mesh.connect(model.metadata.meshed_region)
# wp_mesh = op.outputs.mesh()

wp_skin = dpf.Operator("meshed_skin_sector")
wp_skin.inputs.connect(wp_mesh.outputs.mesh)

# wp_grid = wp_skin.outputs.mesh().grid
# node_scoping = wp_skin.outputs.nodes_mesh_scoping()

# part for grouping by_material/by_el_shape
# mesh_provider = Operator("MeshProvider")
# mesh_provider.inputs.data_sources.connect(data_sources)


mesh_provider = dpf.Operator("MeshProvider")
mesh_provider.inputs.data_sources.connect(model.metadata.data_sources)

# mesh = dpf.Operator('mesh::by_scoping')  # operator instantiation
# mesh.inputs.scoping.connect(wp_skin.outputs.nodes_mesh_scoping)
# mesh.inputs.mesh.connect(mesh_provider.outputs.mesh)

wp_body = dpf.Operator("scoping::by_property")
wp_body.inputs.mesh.connect(mesh_provider.outputs.mesh)
wp_body.inputs.requested_location.connect(post.locations.nodal)
wp_body.inputs.label1.connect("mat")

# wp_body = dpf.Operator("scoping_provider_by_prop")
# wp_body.inputs.requested_location.connect(post.locations.nodal)
# wp_body.inputs.property_name.connect("material")
# wp_body.inputs.data_sources.connect(model.metadata.data_sources)
# wp_body.inputs.property_id.connect(1)


disp = model.operator("U")
disp.inputs.mesh_scoping.connect(wp_body.outputs.mesh_scoping)
fields = disp.outputs.fields_container()


'''
#self._result_operator.inputs.mesh_scoping.connect(scop_op.outputs.mesh_scoping)
#self._chained_operators[scop_op.name] = """This operator will compute a scoping from a grouping option. Its output (mesh_scoping) will be connected with the mesh_scoping input of the result operator."""


#disp = dpf.Operator('Rescope')  # operator instantiation
#disp.inputs.connect(model.operator('U').outputs.fields_container)
#disp.inputs.mesh_scoping.connect(node_scoping)
#my_fields_container = op.outputs.fields_container()

#disp.inputs.mesh_scoping.connect(op.outputs.nodes_mesh_scoping())

solution = post.load_solution(rst_file)

cooldown = solution.displacement(set=2, node_scoping=node_scoping)
TFonly = solution.displacement(set=3, node_scoping=node_scoping)
'''

"""
#norm = disp.norm.get_data_at_field()
wp_grid['disp'] = TFonly.vector.get_data_at_field() - \
    cooldown.vector.get_data_at_field()


warp_grid = wp_grid.warp_by_vector('disp', factor=50)

plotter = pyvista.Plotter()
plotter.add_mesh(wp_grid, color='w')
plotter.add_mesh(warp_grid)
plotter.show()
#disp.norm.plot_contour(show_edges=False, opacity=1, nan_opacity=0,
#                       use_transparency=False, style='surface')
"""
